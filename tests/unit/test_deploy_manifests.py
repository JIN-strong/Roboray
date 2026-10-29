"""Structural contract of deploy manifest rendering (no real cluster needed -- the
locally verifiable part of spec §9.1).

Full schema validation of the KubeRay CRs can only be done on a cluster with the operator;
what is locked here is the **structural keys of the rendered output** -- every numeric value
in spec §6.1/§6.2 is pinned verbatim to prevent "docs say A, manifest says B" drift.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
K8S = REPO / "deploy" / "k8s"


def _render(template: str, extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env.setdefault("IMAGE", "registry.example.com/roboray/base:v0.26.0")
    env.setdefault("NODE", "node-1")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(K8S / "render.py"), template],
        capture_output=True, text=True, env=env, check=True,
    )
    assert "__" not in proc.stdout, f"leftover placeholder in rendered output: {proc.stdout[:500]}"
    return yaml.safe_load(proc.stdout)


def test_raycluster_defaults_and_autoscaling():
    doc = _render("raycluster.yaml")
    spec = doc["spec"]
    assert doc["kind"] == "RayCluster"
    assert spec["rayVersion"] == "2.59.0"
    # Review Focus #3: num-cpus: 0 is a Ray logical resource (the head carries no pipeline
    # work), separate from requests.cpu=6 at the Pod layer -- each is pinned independently
    # to prevent accidental mixing.
    assert spec["headGroupSpec"]["rayStartParams"]["num-cpus"] == "0"
    head_container = spec["headGroupSpec"]["template"]["spec"]["containers"][0]
    assert head_container["resources"]["requests"]["cpu"] == "6"
    assert spec["enableInTreeAutoscaling"] is True
    assert spec["autoscalerOptions"]["idleTimeoutSeconds"] == 300


def test_raycluster_worker_groups_scale_from_zero():
    spec = _render("raycluster.yaml")["spec"]
    groups = {g["groupName"]: g for g in spec["workerGroupSpecs"]}
    assert set(groups) == {"gpu-workers", "cpu-workers"}
    for name, max_key in (("gpu-workers", "MAX_GPU_WORKERS"), ("cpu-workers", "MAX_CPU_WORKERS")):
        g = groups[name]
        assert g["replicas"] == 0 and g["minReplicas"] == 0  # scale from zero: give the autoscaler work to do
        assert g["maxReplicas"] == int(os.environ.get(max_key, {"MAX_GPU_WORKERS": "2",
                                                                "MAX_CPU_WORKERS": "4"}[max_key]))
    assert groups["gpu-workers"]["rayStartParams"]["num-gpus"] == "1"
    assert groups["cpu-workers"]["rayStartParams"].get("num-gpus", "0") in ("0",)


def test_raycluster_storage_layers_and_platform_conventions():
    """Three-layer storage + platform hard conventions, preserved verbatim."""
    doc = _render("raycluster.yaml")
    gpu = doc["spec"]["workerGroupSpecs"][0]["template"]["spec"]
    volumes = {v["name"]: v for v in gpu["volumes"]}
    # workspace = node-local hostPath (prerequisite for E1 node affinity)
    assert volumes["ego"]["hostPath"]["path"] == "/home/ubuntu/ego"
    # shared input/output volume slots exist (default fallback hostPath; multi-node swaps in PVC/NFS)
    assert "shared-input" in volumes and "shared-output" in volumes
    for pod_spec in (doc["spec"]["headGroupSpec"]["template"]["spec"], gpu):
        assert pod_spec["securityContext"]["seccompProfile"]["type"] == "Unconfined"


def test_render_rejects_unknown_placeholders(tmp_path):
    """Review Focus #1: a __XXX__ the renderer does not know must error out, not output silently."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: __WHOAMI__\n")
    env = dict(os.environ)
    env["IMAGE"] = "x"
    proc = subprocess.run(
        [sys.executable, str(K8S / "render.py"), str(bad)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 2
    assert "WHOAMI" in proc.stderr


def test_rayjob_key_fields():
    doc = _render("rayjob.yaml")
    assert doc["kind"] == "RayJob"
    assert doc["metadata"]["name"] == "roboray-job"
    assert doc["metadata"]["namespace"] == "roboray"
    spec = doc["spec"]
    assert spec["submissionMode"] == "K8sJobMode"  # operator builds the submission-client K8s Job pod
    assert spec["shutdownAfterJobFinishes"] is True  # job finished -> cluster torn down (serverless feel)
    assert spec["ttlSecondsAfterFinished"] == 3600
    assert "roboray" in spec["entrypoint"] and "roboray.cli" in spec["entrypoint"]
    # The RayJob environment injects the cluster connection; the entrypoint passes no --ray-address.
    assert "--ray-address" not in spec["entrypoint"]


def test_rayjob_submitter_pod_template():
    """The K8sJobMode submission-client pod only inherits the image; seccomp / ego volume /
    venv PATH must be added explicitly, otherwise the submission client cannot start
    (seccomp blocks threads) or the ray version mismatches (PATH resolves to the image's
    bundled 2.58)."""
    spec = _render("rayjob.yaml")["spec"]
    tpl = spec["submitterPodTemplate"]
    pod = tpl["spec"]
    # real-cluster lesson (2026-10-25): without restartPolicy the submission Job built by
    # the operator is rejected by K8s (restartPolicy: Required value: valid values:
    # "OnFailure", "Never")
    assert pod.get("restartPolicy") == "Never"
    assert pod["securityContext"]["seccompProfile"]["type"] == "Unconfined"
    # ego volume on the pod, mounted at /ego in the container
    assert pod["volumes"][0]["name"] == "ego"
    assert pod["volumes"][0]["hostPath"]["path"] == "/home/ubuntu/ego"
    c = pod["containers"][0]
    assert c["name"] == "ray-job-submitter"
    assert {m["name"]: m["mountPath"] for m in c["volumeMounts"]}["ego"] == "/ego"
    # PATH prefixed with the venv bin (otherwise ray resolves to the image's bundled 2.58)
    path_env = {e["name"]: e["value"] for e in c["env"]}["PATH"]
    assert path_env.startswith("/ego/roboray-venv/bin:")
    # No command/args: KubeRay configureSubmitterContainer only injects the default
    # `ray job submit -- <entrypoint>` when container.Command is empty; given a non-empty
    # command the operator no longer injects it, and the submitter degrades into running the
    # entrypoint locally (no GPU, no cluster connection).
    assert not c.get("command"), f"submitter must not set command; it would clobber the operator's submission injection: {c.get('command')}"
    assert "args" not in c


def test_rayjob_embedded_cluster_matches_standalone():
    """Review Focus #2: rayClusterSpec must be a verbatim copy of raycluster.yaml's spec --
    any change on either side of image/env/resources/volume placement/elasticity thresholds
    must be synced, otherwise the job cluster and the long-running cluster diverge."""
    job = _render("rayjob.yaml")["spec"]["rayClusterSpec"]
    standalone = _render("raycluster.yaml")["spec"]
    # No allowed differences at present; if a field is ever confirmed to legitimately differ
    # in the RayJob scenario, add it to this set explicitly with the reason (confirm it is
    # "intentional" and not "forgot to sync" before adding).
    _ALLOWED_DIFFS: set[str] = set()
    assert ({k: v for k, v in job.items() if k not in _ALLOWED_DIFFS}
            == {k: v for k, v in standalone.items() if k not in _ALLOWED_DIFFS})
