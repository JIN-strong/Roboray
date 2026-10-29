# E3 Elastic Resource Model and Deployment Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the spec §6/§7 elastic resource model in deployment artifacts: autoscaling resident cluster, RayJob (K8sJobMode) job submission, CPU/GPU worker groups, head num-cpus 0, three-layer storage layout, and the supporting scripts/docs.

**Architecture:** add `deploy/k8s/render.py` as the **single** manifest-rendering entry point (shared by apply.sh and the tests -- eliminates the duplicated sed logic); rework `raycluster.yaml` (head 0 CPU, dual worker groups scaling from 0, autoscaling on, three-layer storage volume slots); add `rayjob.yaml` (K8sJobMode + shutdownAfterJobFinishes) and `submit_job.sh`; the structural contract is asserted with pytest against the rendered output (no real cluster needed).

**Tech Stack:** KubeRay ray.io/v1 (operator v1.6.2 semantics), kubectl client 1.35 (structure-only validation), python yaml, bash

**Spec:** `docs/superpowers/specs/2026-10-21-ray-runtime-completion-design.md` sections 6, 7 (§6.1 RayCluster / §6.2 RayJob / §6.3 CPU-GPU division / §7 deployment artifacts / §4.6 three-layer storage)

## Global Constraints

- Test command: `RAY_ADDRESS=local python -m pytest -q --tb=short`.
- **Regression baseline: 277 passed / 0 failed**; only grows.
- **Numbers verbatim from the spec**: head `num-cpus: "0"`; `enableInTreeAutoscaling: true` + `idleTimeoutSeconds: 300`; gpu-workers/cpu-workers `replicas: 0, minReplicas: 0`; the three idle tiers 30/240/300 ascend and 240 < 300.
- **Multi-node semantics unchanged**: worker groups get **no node pinning beyond** `num-gpus` (affinity is the job of E1/E2's actor-placement mechanism, spec §4.4); `rayVersion: "2.59.0"` unchanged.
- **Three-layer storage (spec §4.6)**: input/output shared volumes, workspace node-local hostPath. This cluster has no shared storage provision yet -- the shared volume slot uses an explicit placeholder (`__SHARED_CLAIM__`) and defaults back to hostPath to stay single-node usable; the docs state how to swap in multi-node alternatives.
- Platform hard conventions kept verbatim: `seccompProfile: {type: Unconfined}`, PATH prefix `/ego/roboray-venv/bin`, `rayVersion: "2.59.0"`.
- This plan does **no real-cluster verification** (spec §9.2/D13); the delivery criteria are rendered-structure assertions + full regression.

## Review Focus

(each has a test in the task in parentheses)

1. **Leftover placeholders**: a `__XXX__` left in the manifest after rendering → deployment fails immediately. The renderer must detect residual placeholders and error out; the test asserts no `__` residue in the rendered output. (Task 1)
2. **RayJob drift from the embedded rayClusterSpec**: if the cluster spec embedded in the RayJob diverges from the standalone raycluster.yaml (image/env/volume slots), the submitted job cluster and the resident cluster behave differently. The test asserts the key keys are equal field by field. (Task 2)
3. **Autoscaler's own resources after head num-cpus 0**: head 0 CPU does not affect the operator/autoscaler (they are separate Deployments/Pods), but KubeRay requires the head container to be schedulable -- `requests.cpu: "6"` is Pod-level resources, a different thing from `rayStartParams.num-cpus: "0"` (a Ray logical resource); they must not be conflated. The test asserts the two equal "0" and 6 respectively. (Task 1)
4. **teardown leaves autoscaler-managed Pods**: `delete raycluster` cascades to workers, but a leftover k8s Job (the RayJob K8sJobMode driver) does not cascade -- teardown must delete the k8s Job explicitly. (Task 3) (corrected 2026-10-24: that k8s Job is the **parent Job of the submission-client pod**; the Ray driver process runs inside the head container, not in that Job's pod.)
5. **status.sh's GATE judgment is distorted under autoscaling**: currently `TOTAL_FREE_GPUS = capacity - used`; on a cluster that scales from 0, 0 GPU usage is the norm and GATE=OPEN is always true. It needs RayJob/cluster status output, and the GATE semantics become "has GPUs immediately available **or** a cluster that can auto-scale". (Task 3)

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `deploy/k8s/render.py` | **New**: the only render entry point -- read a manifest template, replace `__PLACEHOLDER__`, detect residue, print to stdout | Create |
| `deploy/k8s/raycluster.yaml` | head 0 CPU, autoscaling, dual worker groups (from 0), three-layer storage volume slots | Modify |
| `deploy/k8s/rayjob.yaml` | **New**: RayJob (K8sJobMode) template | Create |
| `deploy/k8s/submit_job.sh` | **New**: render and submit the RayJob | Create |
| `deploy/k8s/apply.sh` | Use render.py; accept MAX_GPU_WORKERS/MAX_CPU_WORKERS | Modify |
| `deploy/k8s/teardown.sh` | Explicitly delete the k8s Job (K8sJobMode driver residue) | Modify |
| `deploy/k8s/status.sh` | autoscaling/RayJob status output + extended GATE semantics | Modify |
| `tests/unit/test_deploy_manifests.py` | **New**: rendered-structure contract tests | Create |
| `README.md` | deployment command updates | Modify |

> **Correction 2026-10-24**: the "K8sJobMode driver residue" wording in the teardown row above was imprecise -- that k8s Job is the
> **parent Job of the submission-client pod**; the Ray driver process runs inside the head container (same correction as item 4 above).

---

### Task 1: renderer + RayCluster manifest rework

**Files:**
- Create: `deploy/k8s/render.py`
- Modify: `deploy/k8s/raycluster.yaml`
- Create: `tests/unit/test_deploy_manifests.py`

**Interfaces:**
- Produces:
  - `render.py`: `python deploy/k8s/render.py <template.yaml>`, reads env vars `IMAGE` (default `registry.example.com/roboray/base:v0.26.0`), `NODE` (default `<gpu-node-ip>`), `MAX_GPU_WORKERS` (default 2), `MAX_CPU_WORKERS` (default 4), `SHARED_CLAIM` (default empty → hostPath fallback); prints the rendered result to stdout; **exits 2 with any residual `__XXX__` placeholder**, listing them to stderr.
  - `raycluster.yaml` new structural keys (consumed by Task 2's drift assertion): `spec.enableInTreeAutoscaling is True`, `spec.autoscalerOptions.idleTimeoutSeconds == 300`, head `rayStartParams["num-cpus"] == "0"` and container `resources.requests.cpu == "6"`, two workerGroupSpecs (`gpu-workers`/`cpu-workers`, both `replicas: 0, minReplicas: 0`, maxReplicas `__MAX_GPU_WORKERS__`/`__MAX_CPU_WORKERS__`).

- [ ] **Step 1: write failing tests** (create `tests/unit/test_deploy_manifests.py`)

```python
"""Structural contract of the deployment-manifest rendering (no real cluster needed --
the locally verifiable part of spec §9.1).

Full schema validation of the KubeRay CR can only happen on a cluster with the
operator; what we pin here are the **structural keys of the rendered output** --
every number in spec §6.1/§6.2 is pinned verbatim, to stop "the doc says A, the
manifest says B" drift.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
K8S = REPO / "deploy" / "k8s"


def _render(template: str, extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env.setdefault("IMAGE", "registry.example.com/roboray/base:v0.26.0")
    env.setdefault("NODE", "<gpu-node-ip>")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(K8S / "render.py"), template],
        capture_output=True, text=True, env=env, check=True,
    )
    assert "__" not in proc.stdout, f"residual placeholder in rendering: {proc.stdout[:500]}"
    return yaml.safe_load(proc.stdout)


def test_raycluster_defaults_and_autoscaling():
    doc = _render("raycluster.yaml")
    spec = doc["spec"]
    assert doc["kind"] == "RayCluster"
    assert spec["rayVersion"] == "2.59.0"
    # Review Focus #3: num-cpus: 0 is a Ray logical resource (the head carries no
    # pipeline work), a different thing from the Pod-level requests.cpu=6 -- pin both
    # separately to stop them being conflated.
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
        assert g["replicas"] == 0 and g["minReplicas"] == 0  # scale from 0: gives the autoscaler work to do
        assert g["maxReplicas"] == int(os.environ.get(max_key, {"MAX_GPU_WORKERS": "2",
                                                                "MAX_CPU_WORKERS": "4"}[max_key]))
    assert groups["gpu-workers"]["rayStartParams"]["num-gpus"] == "1"
    assert groups["cpu-workers"]["rayStartParams"].get("num-gpus", "0") in ("0",)


def test_raycluster_storage_layers_and_platform_conventions():
    """Three-layer storage + platform hard conventions kept verbatim."""
    doc = _render("raycluster.yaml")
    gpu = doc["spec"]["workerGroupSpecs"][0]["template"]["spec"]
    volumes = {v["name"]: v for v in gpu["volumes"]}
    # workspace = node-local hostPath (the premise of E1 node affinity)
    assert volumes["ego"]["hostPath"]["path"] == "/home/ubuntu/ego"
    # input/output shared volume slots exist (hostPath fallback by default; swap for PVC/NFS on multi-node)
    assert "shared-input" in volumes and "shared-output" in volumes
    for pod_spec in (doc["spec"]["headGroupSpec"]["template"]["spec"], gpu):
        assert pod_spec["securityContext"]["seccompProfile"]["type"] == "Unconfined"


def test_render_rejects_unknown_placeholders(tmp_path, monkeypatch):
    """Review Focus #1: a __XXX__ the renderer does not know must error rather than be
    silently emitted."""
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
```

- [ ] **Step 2: run and confirm failure** (`render.py` does not exist → all 4 FAIL)

- [ ] **Step 3: write the renderer** (create `deploy/k8s/render.py`)

```python
#!/usr/bin/env python3
"""K8s manifest renderer -- the **single** entry point shared by apply.sh and the tests.

Replaces the __PLACEHOLDER__ tokens in a template with values from environment
variables; after rendering, scans for residual placeholders and exits 2 if any are
found (deploying a manifest that still has __XXX__ is guaranteed to fail, so an
early failure is better than a silent one).
Eliminates the duplicated sed logic in apply.sh: a template key rename is a one-place
change.
"""

import os
import re
import sys
from pathlib import Path

_DEFAULTS = {
    "IMAGE": "registry.example.com/roboray/base:v0.26.0",
    "NODE": "<gpu-node-ip>",
    "MAX_GPU_WORKERS": "2",
    "MAX_CPU_WORKERS": "4",
    "SHARED_CLAIM": "",  # empty = shared volumes fall back to hostPath (single node); on multi-node put the PVC/NFS name
}


def render(template_text: str, values: dict) -> str:
    out = template_text
    for key, val in values.items():
        out = out.replace(f"__{key}__", val)
    left = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", out)))
    if left:
        sys.stderr.write(f"residual placeholders after rendering (unknown to the renderer): {', '.join(left)}\n")
        sys.stderr.write(f"known placeholders: {', '.join(sorted(_DEFAULTS))}\n")
        raise SystemExit(2)
    return out


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: render.py <template.yaml>\n")
        return 2
    values = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    # shared-volume placeholder: SHARED_CLAIM empty -> hostPath fallback (same host as ego);
    # non-empty -> the PVC name
    values["SHARED_INPUT"] = values["SHARED_CLAIM"] or "hostPath-shared-input"
    values["SHARED_OUTPUT"] = values["SHARED_CLAIM"] or "hostPath-shared-output"
    template = Path(sys.argv[1]).read_text(encoding="utf-8")
    sys.stdout.write(render(template, values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(Note: `SHARED_INPUT/SHARED_OUTPUT` are derived keys, and the template's shared-volume hostPath path uses `__SHARED_INPUT__` etc.; when `SHARED_CLAIM` is non-empty, the manifest's volume definition would need a branch -- **YAML cannot branch conditionally**, so the manifest declares both volumes `shared-input`/`shared-output` as hostPath by default, and on multi-node an operator swaps the volume blocks for PVCs (documented); the renderer only does placeholder replacement. Simplification: `SHARED_INPUT`/`SHARED_OUTPUT` are inserted as path values only.)

- [ ] **Step 4: rework `raycluster.yaml`** (key points; all keys are in spec §6.1):
  - `spec.enableInTreeAutoscaling: true`; `spec.autoscalerOptions: {idleTimeoutSeconds: 300}`
  - head: `rayStartParams.num-cpus: "0"` (`dashboard-host` kept); container resources keep 6 CPU
  - `workerGroupSpecs` becomes two groups:
    - `gpu-workers`: `replicas: 0` / `minReplicas: 0` / `maxReplicas: __MAX_GPU_WORKERS__`; `rayStartParams: {num-gpus: "1"}`; keep seccompUnconfined/ego hostPath/all env/resources
    - `cpu-workers`: same-shaped new group, `maxReplicas: __MAX_CPU_WORKERS__`; `rayStartParams: {num-cpus: "8", num-gpus: "0"}`; resources `{cpu: "8", memory: "16Gi"}` (no nvidia.com/gpu); env only needs PATH/PYTHONPATH/OPENBLAS (no model envs or GPU-related items)
  - three-layer storage: head/gpu worker volumes add `shared-input`/`shared-output` (hostPath `path: /home/ubuntu/ego-shared/__SHARED_INPUT__`, `__SHARED_OUTPUT__`, `type: DirectoryOrCreate` by default) mounted at `/data/input`, `/data/output`; cpu-workers do not mount ego (no workspace needed) and mount shared-output (for the driver on the head to write the manifest; cpu-workers mounting it is an extension slot)
    > **Correction 2026-10-24**: the "driver starts a separate pod via K8sJobMode / driver writes the manifest" wording here was imprecise -- the K8s Job pod of K8sJobMode is the **submission client**; the Ray driver process runs inside the head container.
  - file-header comment updated: autoscaling, scaling from 0, three-layer storage and the multi-node swap instructions (replace the shared-* volume blocks with PVC/NFS)

- [ ] **Step 5: run the tests and confirm they pass** (all 4 green) + full regression ≥281 passed, 0 failed

- [ ] **Step 6: commit**

```bash
git add deploy/k8s/render.py deploy/k8s/raycluster.yaml tests/unit/test_deploy_manifests.py
git commit -m "feat(deploy): manifest renderer + RayCluster elastic rework (head 0 CPU, dual worker groups from 0, autoscaling 300s)"
```

---

### Task 2: RayJob manifest + submit_job.sh

**Files:**
- Create: `deploy/k8s/rayjob.yaml`
- Modify: `deploy/k8s/render.py` (skip if no code change -- same placeholder set in the template), `tests/unit/test_deploy_manifests.py`

**Interfaces:**
- Produces: `rayjob.yaml` key keys (consumed by Task 3's scripts and docs): `spec.submissionMode == "K8sJobMode"`, `spec.shutdownAfterJobFinishes is True`, `spec.ttlSecondsAfterFinished == 3600`, `spec.rayClusterSpec` equals the key keys of raycluster.yaml **field by field** (Review Focus #2)

- [ ] **Step 1: write failing tests** (append to `test_deploy_manifests.py`)

```python
def test_rayjob_key_fields():
    doc = _render("rayjob.yaml")
    assert doc["kind"] == "RayJob"
    spec = doc["spec"]
    assert spec["submissionMode"] == "K8sJobMode"  # the driver runs in a separate K8s Job pod (does not crowd the head)
    assert spec["shutdownAfterJobFinishes"] is True  # job done -> cluster torn down (serverless feel)
    assert spec["ttlSecondsAfterFinished"] == 3600
    assert "roboray" in spec["entrypoint"] and "roboray.cli" in spec["entrypoint"]


def test_rayjob_embedded_cluster_matches_standalone():
    """Review Focus #2: the embedded rayClusterSpec's key keys must match the standalone
    manifest, otherwise the submitted job cluster and the resident cluster diverge
    (image/env/volume slots)."""
    job = _render("rayjob.yaml")["spec"]["rayClusterSpec"]
    standalone = _render("raycluster.yaml")["spec"]
    assert job["rayVersion"] == standalone["rayVersion"]
    assert job["headGroupSpec"]["rayStartParams"] == standalone["headGroupSpec"]["rayStartParams"]
    j_groups = {g["groupName"]: g for g in job["workerGroupSpecs"]}
    s_groups = {g["groupName"]: g for g in standalone["workerGroupSpecs"]}
    assert set(j_groups) == set(s_groups)
    for name in s_groups:
        assert j_groups[name]["rayStartParams"] == s_groups[name]["rayStartParams"]
        assert j_groups[name]["maxReplicas"] == s_groups[name]["maxReplicas"]
```

> **Correction 2026-10-24**: in `test_rayjob_key_fields`, the `# the driver runs in a separate K8s Job pod (does not crowd the head)`
> comment is imprecise -- the K8s Job pod is the **submission client** (runs `ray job submit`); per Ray's design the Ray driver process
> **runs inside the head container**.

- [ ] **Step 2: run and confirm failure** (`rayjob.yaml` does not exist)

- [ ] **Step 3: write `rayjob.yaml`** -- structure: `apiVersion: ray.io/v1`, `kind: RayJob`, `metadata: {name: roboray-job, namespace: roboray}`; `spec` includes `entrypoint: cd /ego/app && /ego/roboray-venv/bin/python -m roboray.cli run --config /ego/app/configs/default.yaml` (the `__ENTRYPOINT__` placeholder may override), `submissionMode: K8sJobMode`, `shutdownAfterJobFinishes: true`, `ttlSecondsAfterFinished: 3600`, `rayClusterSpec` (copy the full spec of raycluster.yaml, `metadata.name` changed to `roboray-job-cluster`). File-header comment: K8sJobMode = the operator starts a separate K8s Job pod that runs the driver (does not consume head/worker resources); when the CLI runs inside the Job, `ray.is_initialized()` is False → it still does `ray.init(address=auto?)` -- **note**: inside a RayJob `RAY_ADDRESS` is not needed; the Ray Job worker environment injects the cluster connection automatically; the entrypoint does not pass `--ray-address`.
  > **Correction 2026-10-24**: ① "the operator starts a separate K8s Job pod that runs the driver (does not consume head/worker resources)" was imprecise -- the K8s Job pod is the **submission client** (runs `ray job submit`); per Ray's design the Ray driver process **runs inside the head container**. ② The embedded `rayClusterSpec` has no `metadata.name`; the cluster name is **auto-derived by the operator with a random suffix** (`<name>-<rand5>` / release-1.0 `<name>-raycluster-<rand5>`), so `roboray-job-cluster` **must not be hardcoded**; the actual delivery also added `spec.submitterPodTemplate` (the operator does not inherit seccomp/ego volumes/venv PATH).
- [ ] **Step 4: run the tests and confirm they pass** (incl. the drift assertion)
- [ ] **Step 5: commit**

```bash
git add deploy/k8s/rayjob.yaml tests/unit/test_deploy_manifests.py
git commit -m "feat(deploy): RayJob(K8sJobMode) manifest -- tear the cluster down when the job finishes, driver does not crowd the head"
```

> **Correction 2026-10-24**: the "driver does not crowd the head" in the example commit message above was imprecise -- the K8s Job pod of
> K8sJobMode is the **submission client** (runs `ray job submit`); per Ray's design the Ray driver process **runs inside the head
> container**. (Historical commit messages are not rewritten; this is only registered here.)

---

### Task 3: the three scripts updated + GATE semantics extended

**Files:**
- Modify: `deploy/k8s/apply.sh`, `deploy/k8s/teardown.sh`, `deploy/k8s/status.sh`
- Create: `deploy/k8s/submit_job.sh`

**Interfaces:**
- Produces:
  - `apply.sh`: switch to `render.py`; accept `MAX_GPU_WORKERS`/`MAX_CPU_WORKERS`/`SHARED_CLAIM`
  - `submit_job.sh`: render rayjob.yaml → `kubectl apply -f -`; `ENTRYPOINT` env var overridable
  - `teardown.sh`: add deleting the k8s Job (`delete job -l ray.io/cluster=...` is not portable -- directly `delete job --all`); RayJob deletion order: delete the rayjob first (it manages both the job and the cluster), then fallback-delete residual jobs/pods
  - `status.sh`: extend the GATE output -- besides `TOTAL_FREE_GPUS`, output `RAYCLUSTER_AUTOSCALING` (true/false) and the `RAYJOB` list; the GATE judgment becomes "`TOTAL_FREE_GPUS>0` or an autoscaling cluster exists" → OPEN (spec §7 semantics: GPUs immediately available **or** auto-scalable)

- [ ] **Steps** (scripts have no pytest; acceptance = existing full regression does not regress + `bash -n` syntax check + Task 1/2's structure tests stay green):
  1. `apply.sh`: `python3 "$ROOT/deploy/k8s/render.py" "$ROOT/deploy/k8s/raycluster.yaml" | kubectl apply -f -` (env vars passed through to the renderer); namespace as before
  2. `submit_job.sh` new (set -euo pipefail; `ENTRYPOINT="${ENTRYPOINT:-}"` non-empty exports to the renderer... -- **the renderer does not currently know ENTRYPOINT**: add `"ENTRYPOINT": "cd /ego/app && /ego/roboray-venv/bin/python -m roboray.cli run --config /ego/app/configs/default.yaml"` to render.py's `_DEFAULTS`, and use `__ENTRYPOINT__` in rayjob.yaml's entrypoint; Task 2's completed test is unaffected -- it asserts on substrings)
  3. `teardown.sh`: order rayjob → k8s job → raycluster → pod (delete the rayjob first to stop reconcile from resurrecting things)
  4. `status.sh`: add RayCluster autoscaling/replicas output and the RayJob list; extend the GATE logic
  5. `bash -n` the three scripts + full regression ≥281 passed, 0 failed + Task 1/2 tests stay green
- [ ] **Commit**:

```bash
git add deploy/k8s/apply.sh deploy/k8s/teardown.sh deploy/k8s/status.sh deploy/k8s/submit_job.sh deploy/k8s/render.py deploy/k8s/rayjob.yaml tests/unit/test_deploy_manifests.py
git commit -m "feat(deploy): submit_job/apply/teardown/status four scripts -- RayJob submission, cascade cleanup, GATE extension"
```

---

### Task 4: docs + full wrap-up

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-10-21-ray-runtime-completion-design.md` (check off the §9.2 unverified column)

**Steps**:
1. README "Deploy to K8s / KubeRay" section update: submit jobs with `bash deploy/k8s/submit_job.sh` (RayJob, auto-tears down the cluster when the job finishes); the resident cluster with `apply.sh`; explain "scale from 0 + autoscaling" and the three tiers 30/240/300
2. Mark the four §9.2 unverified items with their current status item by item (still unverified -- this plan only delivers rendered-structure assertions; real-cluster autoscaling/RayJob verification remains out of scope, and must not be reworded)
3. Full regression ≥281 passed, 0 failed
4. Commit `docs(e3): deployment command updates and current status of unverified items`

---

## Self-Review

**1. Spec coverage (§6/§7 item by item)**

| spec item | covering task |
|---|---|
| 6.1 enableInTreeAutoscaling + idleTimeoutSeconds 300 + head num-cpus 0 + dual worker groups from 0 | Task 1 (pinned verbatim by tests) |
| 6.2 RayJob K8sJobMode + shutdownAfterJobFinishes + ttl 3600 + embedded rayClusterSpec | Task 2 (incl. drift assertion) |
| 6.3 CPU/GPU division (driver on cpu-workers) | Task 2 (K8sJobMode is the driver pod, does not crowd the head) + Task 1 (cpu-workers group) |
| 7 five deployment artifacts (raycluster rework/rayjob/submit_job/status/teardown+apply) | Task 1/2/3 |
| 4.6 three-layer storage | Task 1 (shared-* volume slots + default hostPath fallback + multi-node swap docs) |
| 9.1 rendered-structure assertions | all Task 1/2 tests; "kubectl --dry-run=client" became the more portable python structure assertions (there is no CRD schema locally, so client dry-run adds no validation power for CRs -- this deviation is explained in the Step 1 test docstring) |

> **Correction 2026-10-24**: the "driver on cpu-workers" / "K8sJobMode is the driver pod, does not crowd the head"
> in the 6.3 row above were imprecise -- the K8s Job pod of K8sJobMode is the **submission client** (runs `ray job submit`); the Ray driver process
> per Ray's design **runs inside the head container**. The cpu-workers extension-slot semantics are unchanged.

**2. Placeholder scan**: no TBD/TODO; render.py's SHARED placeholder design is explained in the body with its trade-off.

**3. Type consistency**: `render.py <template>` CLI, the placeholder set (IMAGE/NODE/MAX_GPU_WORKERS/MAX_CPU_WORKERS/SHARED_CLAIM→SHARED_INPUT/SHARED_OUTPUT/ENTRYPOINT) is consistent across Task 1/2/3; the drift-assertion keys of `rayClusterSpec` (rayVersion/rayStartParams/workerGroupSpecs.maxReplicas) align with Task 1's structural keys.

**4. Review Focus**: the five map to Task 1 (residual-placeholder detection, num-cpus 0 vs requests.cpu 6 pinned separately), Task 2 (drift assertion), Task 3 (teardown cascade, GATE extension).

**Known uncovered (registered)**: autoscaling scale-out/reclaim and RayJob end-to-end on a real cluster (spec §9.2/D13 explicitly out of scope); the interplay of `idleTimeoutSeconds: 300` with E2's `actor_idle_exit_s=240` is only documented (no code interaction).
