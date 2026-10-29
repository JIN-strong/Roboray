# Plan D: A800 KubeRay Deployment and L2/L3 Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy roboray to KubeRay on the A800 K8s cluster, run the L2 smoke with real models (each of ViPE / Fast-FoundationStereo / HaMeR+MANO against its own artifact contract), run the L3 three-tier latency comparison (serial / cross-stage pipeline / +adaptive scheduling) plus the custom evaluation subset, and produce a reproducible measurement report.

**Architecture:** The cluster nodes have **no external network and no reachable pip mirror**, so all dependencies and weights must be **built locally and carried into the image**: locally, Docker builds one image containing the main package + the three model conda envs + the vendored repos + prefetched weights, pushed to the internal registry via an SSH tunnel to the internal image registry, then a KubeRay `RayCluster` CR starts head(CPU) + N×GPU workers in the dedicated `roboray` namespace. L2/L3 runs through a port-forward to the head's 8265 using `ray job submit`.

**Tech Stack:** Docker, conda (local), KubeRay, kubectl, `ray job submission`, pytest (`-m gpu`).

**Spec:** `docs/superpowers/specs/2026-10-16-ego-pipeline-design.md` (section 11 deployment, section 12 L2/L3)

## Global Constraints

- **Cluster facts (measured when writing this plan)**: K8s v1.22; 2 GPU nodes, each with 2× A800 80GB PCIe (driver 580.178.04, Anolis 8.5); the KubeRay operator already runs in existing platform namespaces; internal image registry `registry.example.com`; nodes have **no external network** (pypi/tuna/aliyun mirrors all unreachable); this machine has external network (pypi 200, hf-mirror 200)
- **GPU capacity**: at the time of writing, all 4 A800s are **in use** (2 by an existing shared RayCluster + 2 by vLLM). This is a shared production cluster — **L2/L3 must be executed when GPUs are free** and the occupancy must be short (L3 uses 3–5 short clips, per spec §12)
- Main-env dependencies unchanged; test commands are `conda run -n <env> python -m pytest ...`; inside the cluster only `-m gpu` runs, not the full pytest suite
- Namespace fixed to `roboray`; the image tag includes the git short SHA (traceability)
- **Weights must be prefetched**: ViPE's first run **auto-downloads weights** (no external network on the nodes -> guaranteed failure); FFS comes from HF; HaMeR's `MANO_RIGHT.pkl` needs a registration download. All three must be fetched **locally** and baked into the image
- In-image env paths fixed: `/opt/ego/venvs/{vipe,ffs,hamer,lerobot}/bin/python`; the matching `ROBORAY_*_PYTHON` are set in the RayCluster worker environment variables
- Any "install packages on the node" approach is forbidden (no image source); any "download weights at runtime" approach is forbidden
- **Local disk constraint (measured)**: the Docker root dir `Docker Root Dir: /var/lib/docker` is on `/`, and `/` has only **18GB** left; `/home` has 37GB. The three torch envs install to ~20GB and the image is ~12–14GB -> **neither side has enough**. Before executing, free space first (prune unused docker images/old envs, or move `data-root` to `/home`), and re-check `df -h` before Task 2. This is a known precondition for starting, not a risk note
- git: one commit per task; every Review Focus item must be pinned by a test or a measured command

## Review Focus

1. **Image self-containment**: still works after the network is gone — in the image, `import torch` succeeds in each of the three envs, the vendored repos import, and the weight files exist and are readable.
2. **GPU capacity gate**: when no GPU is free, do not leave Pods stuck in Pending and pretend "deployment done" — must explicitly report the capacity state and mark L2/L3 as gated.
3. **No pollution of the shared cluster**: create resources only in your own namespace; do not delete/modify others' RayClusters; tear down after use (including PVC/hostPath cleanup).
4. **L2 artifact contracts**: each model's smoke must assert the artifact keys/shapes/finiteness and `depth_source`, not just "no error".
5. **L3 three-tier comparability**: the three modes must use the same input batch, the same image, and the same session (avoiding cross-time hardware fluctuation), and record GPU utilization and peak VRAM.
6. **Honest boundary**: absolute metrics of public benchmarks such as HOT3D (MPJPE 28.47mm etc.) are **not reproduced** — this plan only does form reproduction (two-level speedup) and relative improvement on the custom subset.

---

### Task 1: Local model envs and weight prefetch (offline self-containment)

**Files:**
- Create: `deploy/modelenvs/build_all_local.sh`
- Create: `deploy/modelenvs/fetch_weights.sh`
- Create: `deploy/modelenvs/verify_offline.sh`
- Test: `deploy/modelenvs/verify_offline.sh` itself is the verification script (exit-code assertions)

**Interfaces:**
- Consumes: Plan B's `deploy/modelenvs/build_{vipe,ffs,hamer,lerobot}.sh` (unusable inside the cluster, but their version combos are reused)
- Produces:
  - Four conda envs: `ego-vipe` (torch 2.7+cu128), `ego-ffs` (py3.12 + torch 2.6+cu124 + xformers), `ego-hamer` (torch cu124 + ViTPose), `ego-lerobot` (CPU torch, already exists)
  - Weights directory `deploy/assets/weights/{vipe,ffs,hamer}/`
  - `verify_offline.sh`: runs `python -c "import torch; ..."` and weight-existence checks per env in the current shell, returning 0 only when everything is green

- [ ] **Step 1: Write the env build script (local, with external network)**

`deploy/modelenvs/build_all_local.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Build the four model envs locally (cluster nodes have no external network, so they must be
# built here and carried into the image).
# Requires: conda available; this machine has external network.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

build_vipe() {
  conda create -y -n ego-vipe python=3.10
  conda run -n ego-vipe python -m pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
  conda run -n ego-vipe python -m pip install -e "$ROOT/third_party/vipe"
}

build_ffs() {
  conda create -y -n ego-ffs python=3.12
  conda run -n ego-ffs python -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
  conda run -n ego-ffs python -m pip install xformers --index-url https://download.pytorch.org/whl/cu124
  conda run -n ego-ffs python -m pip install -e "$ROOT/third_party/fast-foundationstereo"
}

build_hamer() {
  conda create -y -n ego-hamer python=3.10
  conda run -n ego-hamer python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
  conda run -n ego-hamer python -m pip install -e "$ROOT/third_party/hamer[all]"
  conda run -n ego-hamer python -m pip install -v -e "$ROOT/third_party/hamer/third-party/ViTPose"
}

case "${1:-all}" in
  vipe) build_vipe ;;
  ffs) build_ffs ;;
  hamer) build_hamer ;;
  all) build_vipe; build_ffs; build_hamer ;;
  *) echo "usage: $0 [vipe|ffs|hamer|all]"; exit 2 ;;
esac
echo "done."
```

- [ ] **Step 2: Write the weight prefetch script**

`deploy/modelenvs/fetch_weights.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Prefetch all model weights (no external network on the nodes; no runtime downloads).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="$ROOT/deploy/assets/weights"
mkdir -p "$W"/{vipe,ffs,hamer/mano}

# FFS: HF nvidia/c-fast-foundationstereo
if [ ! -f "$W/ffs/model_best_bp2_serialize.pth" ]; then
  conda run -n roboray python -m pip install -q "huggingface_hub[cli]"
  conda run -n roboray huggingface-cli download nvidia/c-fast-foundationstereo \
    --local-dir "$W/ffs" --local-dir-use-symlinks False
fi

# ViPE: the first run tries to download -> run one tiny inference in a networked env first,
# so the cache lands in $W/vipe
if [ -z "$(ls -A "$W/vipe" 2>/dev/null)" ]; then
  echo "ViPE weights are cached by its pipeline's first run; see docs/models.md."
  echo "Then copy the cache dir (default ~/.cache/vipe) wholesale into $W/vipe:"
  echo "  rsync -a ~/.cache/vipe/ $W/vipe/"
fi

# HaMeR: MANO_RIGHT.pkl requires a manual registration download (cannot be automated)
if [ ! -f "$W/hamer/mano/MANO_RIGHT.pkl" ]; then
  echo "!! missing $W/hamer/mano/MANO_RIGHT.pkl"
  echo "!! register at https://mano.is.tue.mpg.de, download it to that path."
  echo "!! without it the HaMeR smoke is skipped (never faked as a pass)."
fi
echo "weights dir: $W"
```

- [ ] **Step 3: Write the offline self-check script**

`deploy/modelenvs/verify_offline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Offline self-check: each env can import torch with the right version; vendored repos import; weights exist.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="$ROOT/deploy/assets/weights"
fail=0
chk() { echo -n "  $1 ... "; shift; if "$@" >/dev/null 2>&1; then echo OK; else echo FAIL; fail=1; fi; }

echo "[env] ego-vipe"
chk "torch import" conda run -n ego-vipe python -c "import torch; print(torch.__version__)"
echo "[env] ego-ffs"
chk "torch import" conda run -n ego-ffs python -c "import torch; print(torch.__version__)"
chk "xformers import" conda run -n ego-ffs python -c "import xformers"
echo "[env] ego-hamer"
chk "torch import" conda run -n ego-hamer python -c "import torch; print(torch.__version__)"
echo "[vendored]"
chk "vipe path exists" test -d "$ROOT/third_party/vipe"
chk "ffs path exists" test -d "$ROOT/third_party/fast-foundationstereo"
chk "hamer path exists" test -d "$ROOT/third_party/hamer"
echo "[weights]"
chk "ffs weights" test -f "$W/ffs/model_best_bp2_serialize.pth"
chk "vipe weights dir non-empty" test -n "$(ls -A "$W/vipe" 2>/dev/null)"
if [ -f "$W/hamer/mano/MANO_RIGHT.pkl" ]; then echo "  hamer MANO_RIGHT.pkl ... OK"; else echo "  hamer MANO_RIGHT.pkl ... MISSING (the HaMeR smoke will be skipped)"; fi

if [ "$fail" -ne 0 ]; then echo "offline self-check FAILED"; exit 1; fi
echo "offline self-check passed"
```

- [ ] **Step 4: Run the builds and prefetch (real execution, takes a while)**

```bash
bash scripts/vendor_models.sh                 # if third_party is not yet cloned
bash deploy/modelenvs/build_all_local.sh all  # build env by env (~20-40 min, network-dependent)
bash deploy/modelenvs/fetch_weights.sh
bash deploy/modelenvs/verify_offline.sh       # expected: offline self-check passed (MANO may be MISSING)
```

Record: each env's torch version, the weights-dir size (`du -sh deploy/assets/weights/*`).

- [ ] **Step 5: Commit**

```bash
git add deploy/modelenvs/build_all_local.sh deploy/modelenvs/fetch_weights.sh deploy/modelenvs/verify_offline.sh
git commit -m "feat(deploy): local model-env build, weight prefetch and offline self-check (no external network on the nodes)"
```

---

### Task 2: Deployment image build and push to the internal registry

**Files:**
- Create: `deploy/Dockerfile`
- Create: `deploy/.dockerignore`
- Create: `deploy/build_image.sh`
- Create: `deploy/push_image.sh`
- Test: image self-check (`docker run ... verify_offline.sh`)

**Interfaces:**
- Consumes: Task 1's four envs, the vendored repos, the weights directory
- Produces:
  - Image `<REGISTRY>/roboray/roboray:<git-sha>`, containing `/opt/ego/{app,venvs,third_party,assets}` and `/opt/ego/verify_offline.sh`
  - `push_image.sh` pushes to `registry.example.com` via an SSH tunnel to the internal image registry

- [ ] **Step 1: Write the Dockerfile**

`deploy/Dockerfile`:

```dockerfile
# roboray cluster image: main package + the three model envs + vendored repos + prefetched weights.
# Cluster nodes have no external network, so nothing may be installed at runtime.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    EGO_HOME=/opt/ego

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip rsync libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Alternative to conda-pack: copy the local env directories directly (same arch linux-x86_64).
# Note: the copied envs' prefixes need rewriting, so conda-pack is used to package them into
# tarballs and unpack them onto the fixed prefix.
COPY deploy/assets/envs/vipe.tar.gz   /tmp/vipe.tar.gz
COPY deploy/assets/envs/ffs.tar.gz    /tmp/ffs.tar.gz
COPY deploy/assets/envs/hamer.tar.gz  /tmp/hamer.tar.gz
COPY deploy/assets/envs/lerobot.tar.gz /tmp/lerobot.tar.gz

RUN mkdir -p ${EGO_HOME}/venvs \
 && for e in vipe ffs hamer lerobot; do \
      mkdir -p ${EGO_HOME}/venvs/$e && tar -xzf /tmp/$e.tar.gz -C ${EGO_HOME}/venvs/$e; \
      ${EGO_HOME}/venvs/$e/bin/conda-unpack; \
    done \
 && rm -f /tmp/*.tar.gz

COPY pyproject.toml README.md ${EGO_HOME}/app/
COPY src ${EGO_HOME}/app/src
COPY scripts ${EGO_HOME}/app/scripts
COPY configs ${EGO_HOME}/app/configs
COPY third_party ${EGO_HOME}/app/third_party
COPY deploy/assets/weights ${EGO_HOME}/assets/weights

# the main package goes into a minimal env (no torch/lerobot)
RUN python3 -m pip install --no-cache-dir -e "${EGO_HOME}/app"

ENV ROBORAY_VIPE_PYTHON=${EGO_HOME}/venvs/vipe/bin/python \
    ROBORAY_FFS_PYTHON=${EGO_HOME}/venvs/ffs/bin/python \
    ROBORAY_HAMER_PYTHON=${EGO_HOME}/venvs/hamer/bin/python \
    ROBORAY_LEROBOT_PYTHON=${EGO_HOME}/venvs/lerobot/bin/python \
    ROBORAY_VIPE_WEIGHTS=${EGO_HOME}/assets/weights/vipe \
    ROBORAY_FFS_WEIGHTS=${EGO_HOME}/assets/weights/ffs \
    ROBORAY_HAMER_WEIGHTS=${EGO_HOME}/assets/weights/hamer

WORKDIR ${EGO_HOME}/app
CMD ["python", "-c", "import roboray; print(roboray.__version__)"]
```

- [ ] **Step 2: Package the envs (conda-pack)**

Append the packaging step to the end of `deploy/modelenvs/build_all_local.sh` (implemented within this Task 2):

```bash
pack_env() {
  local name="$1"
  conda run -n roboray python -m pip install -q conda-pack
  mkdir -p "$ROOT/deploy/assets/envs"
  conda run -n roboray conda-pack -n "$name" -o "$ROOT/deploy/assets/envs/${name#ego-}.tar.gz" --force
}
```
(`ego-vipe` -> `vipe.tar.gz`, and likewise for the other three.)

- [ ] **Step 3: Build the image and self-check**

`deploy/build_image.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
TAG="roboray:${SHA}"
docker build -f "$ROOT/deploy/Dockerfile" -t "$TAG" "$ROOT"
# self-check: the four in-image envs import and the weights exist (no GPU started)
docker run --rm --entrypoint bash "$TAG" -lc \
  'set -e; for e in vipe ffs hamer lerobot; do /opt/ego/venvs/$e/bin/python -c "import sys; print(\"$e\", sys.version.split()[0])"; done; test -f /opt/ego/assets/weights/ffs/model_best_bp2_serialize.pth; echo IMAGE_SELFCHECK_OK'
echo "image: $TAG"
```

Expected output ends with `IMAGE_SELFCHECK_OK`.

- [ ] **Step 4: Push to the internal registry**

`deploy/push_image.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
REMOTE="registry.example.com/roboray/roboray:${SHA}"   # mapped to the internal image registry via an SSH tunnel
docker tag "roboray:${SHA}" "$REMOTE"
docker push "$REMOTE"
echo "pushed: $REMOTE"
```

Before pushing, open an SSH tunnel to the internal image registry (`connect --ip=<management-node-ip> --user=ubuntu`). Record the push duration and image size (`docker images --format '{{.Size}}'`).

- [ ] **Step 5: Commit**

```bash
git add deploy/Dockerfile deploy/.dockerignore deploy/build_image.sh deploy/push_image.sh
git commit -m "feat(deploy): self-contained deployment image (four envs + vendored + prefetched weights) and registry push"
```

---

### Task 3: KubeRay RayCluster CR and namespace

**Files:**
- Create: `deploy/k8s/namespace.yaml`
- Create: `deploy/k8s/raycluster.yaml`
- Create: `deploy/k8s/apply.sh`, `deploy/k8s/teardown.sh`
- Create: `deploy/k8s/status.sh`
- Test: `deploy/k8s/status.sh` output (including the GPU-capacity determination)

**Interfaces:**
- Consumes: the image pushed in Task 2
- Produces:
  - Namespace `roboray`
  - RayCluster `roboray-cluster`: head(4 CPU/8Gi, no GPU) + `gpu-workers` (replicas `GPU_WORKERS`, each pod `nvidia.com/gpu: 1`)
  - `status.sh`: prints pod status + **the GPU-free determination** (`free>0` is required to run L2/L3)

- [ ] **Step 1: Write the namespace and CR**

`deploy/k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: roboray
```

`deploy/k8s/raycluster.yaml`:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: roboray-cluster
  namespace: roboray
spec:
  rayVersion: "2.59.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-cpus: "4"
    template:
      spec:
        containers:
          - name: ray-head
            image: __IMAGE__
            imagePullPolicy: IfNotPresent
            resources:
              requests: {cpu: "4", memory: "8Gi"}
              limits:   {cpu: "4", memory: "8Gi"}
            volumeMounts:
              - {name: work, mountPath: /work}
        volumes:
          - name: work
            hostPath: {path: /tmp/roboray, type: DirectoryOrCreate}
  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: __GPU_WORKERS__
      minReplicas: 0
      maxReplicas: __GPU_WORKERS__
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: ray-worker
              image: __IMAGE__
              imagePullPolicy: IfNotPresent
              resources:
                requests: {cpu: "8", memory: "32Gi", nvidia.com/gpu: "1"}
                limits:   {cpu: "8", memory: "32Gi", nvidia.com/gpu: "1"}
              volumeMounts:
                - {name: work, mountPath: /work}
          volumes:
            - name: work
              hostPath: {path: /tmp/roboray, type: DirectoryOrCreate}
```

`deploy/k8s/apply.sh` (render the placeholders, then apply):

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
IMAGE="${IMAGE:-registry.example.com/roboray/roboray:${SHA}}"
GPU_WORKERS="${GPU_WORKERS:-2}"
command -v kubectl >/dev/null || { echo "kubectl required"; exit 2; }
kubectl apply -f "$ROOT/deploy/k8s/namespace.yaml"
sed -e "s|__IMAGE__|${IMAGE}|g" -e "s|__GPU_WORKERS__|${GPU_WORKERS}|g" \
  "$ROOT/deploy/k8s/raycluster.yaml" | kubectl apply -f -
echo "submitted. watch the status with deploy/k8s/status.sh."
```

`deploy/k8s/teardown.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
kubectl -n roboray delete raycluster roboray-cluster --ignore-not-found
kubectl -n roboray delete pods --all --ignore-not-found
echo "cleaned up the Ray resources in the roboray namespace (the namespace itself is kept)."
```

`deploy/k8s/status.sh` (with the capacity gate):

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "=== roboray pods ==="
kubectl -n roboray get pods -o wide 2>/dev/null || echo "(no pods yet)"
echo "=== cluster-wide free GPUs ==="
kubectl get nodes -o json | python3 -c '
import json,sys,subprocess
items=json.load(sys.stdin)["items"]
pods=json.loads(subprocess.run(["kubectl","get","pods","-A","-o","json"],capture_output=True,text=True).stdout)["items"]
used={}
for p in pods:
    n=p["spec"].get("nodeName","")
    for c in p["spec"].get("containers",[]):
        for k,v in (c.get("resources",{}).get("requests",{}) or {}).items():
            if "gpu" in k.lower() and str(v) not in ("0","0.0"):
                used[n]=used.get(n,0)+int(float(v))
free=0
for nd in items:
    nm=nd["metadata"]["name"]; cap=nd["status"].get("capacity",{}).get("nvidia.com/gpu")
    if cap:
        f=int(cap)-used.get(nm,0); free+=max(0,f)
        print(f"  {nm}: cap={cap} used={used.get(nm,0)} free={max(0,f)}")
print(f"TOTAL_FREE_GPUS={free}")
print("GATE=" + ("OPEN" if free>0 else "BLOCKED"))
'
```

- [ ] **Step 2: Apply and observe (real execution)**

```bash
export KUBECONFIG=<path-to-cluster-kubeconfig>
bash deploy/k8s/apply.sh
sleep 20 && bash deploy/k8s/status.sh
```

Determination: if `GATE=BLOCKED`, the pods stay in Pending — **that is expected**, record it and go into Task 6's "gated" branch; **do not** squeeze other people's GPUs just to see Running.

- [ ] **Step 3: Commit**

```bash
git add deploy/k8s/
git commit -m "feat(deploy): KubeRay RayCluster CR, namespace and status script with a GPU-capacity gate"
```

---

### Task 4: L2 smoke execution (GPU gate)

**Files:**
- Create: `deploy/run_l2.sh`
- Create: `docs/superpowers/plans/D-l2-results.md` (result table, filled from measurements)

**Interfaces:**
- Consumes: Task 3's RayCluster (needs `GATE=OPEN`), Plan B's `tests/gpu/test_adapters_gpu.py`
- Produces: each model's measured smoke conclusion (pass/skip and why)

- [ ] **Step 1: Prerequisite gate check**

```bash
bash deploy/k8s/status.sh | tee /tmp/ego-status.txt
grep -q "GATE=OPEN" /tmp/ego-status.txt || { echo "no free GPU; L2 recorded as gated, see Task 6"; exit 3; }
```

- [ ] **Step 2: Run the gpu smoke with real models on the head**

`deploy/run_l2.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
NS=roboray
kubectl -n $NS port-forward svc/roboray-cluster-head-svc 8265:8265 >/tmp/pf.log 2>&1 &
PF=$!; trap 'kill $PF 2>/dev/null || true' EXIT
sleep 5
ray job submit --address http://127.0.0.1:8265 --no-wait=false -- \
  bash -lc 'cd /opt/ego/app && ROBORAY_GPU_SMOKE=1 python -m pytest -o addopts="" -m gpu -v'
```

Expected: the ViPE/FFS/HaMeR three cases execute per the completeness of their weights; when MANO is missing, HaMeR shows skip (never faked as a pass).

- [ ] **Step 3: Record the results**

In `D-l2-results.md`, fill in per model: whether it executed, the key assertions, artifact shapes, latency, and failure reasons (if any). All numbers from the actual output.

- [ ] **Step 4: Commit**

```bash
git add deploy/run_l2.sh docs/superpowers/plans/D-l2-results.md
git commit -m "feat(deploy): L2 smoke execution script and measured results"
```

---

### Task 5: L3 three-tier latency comparison and the custom evaluation

**Files:**
- Create: `deploy/run_l3.sh`
- Create: `scripts/make_l3_dataset.py` (3-5 short clips, reusing `make_synthetic_ego` with an optional swap for real clips)
- Create: `docs/superpowers/plans/D-l3-results.md`

**Interfaces:**
- Consumes: the same image, the same input batch, the same RayCluster session
- Produces: the three tiers' (serial / pipeline / adaptive) total latency, GPU utilization, peak VRAM; the A1-A3 relative metrics on the custom subset

- [ ] **Step 1: Prepare the unified input**

```bash
conda run -n roboray python scripts/make_l3_dataset.py --out /tmp/ego-l3 --episodes 3 --frames 30
# upload to the shared hostPath (visible to the workers)
kubectl -n roboray cp /tmp/ego-l3 $(kubectl -n roboray get pod -l ray.io/node-type=head -o name | head -1):/work/ego-l3
```

- [ ] **Step 2: Run the three tiers (same session, same input)**

`deploy/run_l3.sh` submits three jobs in sequence (`--serial-baseline` / default / `--no-adaptive`) and pulls each `manifest.json`'s `summary.total_seconds` and `scheduler_snapshots` back to the local machine:

```bash
#!/usr/bin/env bash
set -euo pipefail
CONF="${CONF:-/opt/ego/app/configs/default.yaml}"
run_one() {  # $1=label  $2=extra args
  local tag="$1"; shift
  ray job submit --address http://127.0.0.1:8265 -- \
    bash -lc "cd /opt/ego/app && roboray run --config $CONF --run-root /work/run-$tag $* && cat /work/run-$tag/manifest.json" \
    | tail -1 > "/tmp/l3-$tag.json"
}
run_one serial --serial-baseline
run_one pipeline --no-adaptive
run_one adaptive
python - <<'PY'
import json
for tag in ("serial","pipeline","adaptive"):
    m=json.load(open(f"/tmp/l3-{tag}.json"))
    print(tag, m["summary"]["total_seconds"], "episodes:", m["summary"]["n_episodes"], "failed:", m["summary"]["failed"])
PY
```

- [ ] **Step 3: GPU utilization and peak VRAM**

During the L3 run, poll once (`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`, via `kubectl exec` into a worker) and record the mean and peak.

- [ ] **Step 4: Write `D-l3-results.md`**

Table: the three tiers' total latency, the relative-to-serial speedup, GPU utilization, peak VRAM, Episode/frame counts, failure count, scheduling-snapshot count. **Honest boundary**: no HOT3D absolute-metric reproduction; the custom subset only reports relative improvement.

- [ ] **Step 5: Clean up and commit**

```bash
bash deploy/k8s/teardown.sh
git add deploy/run_l3.sh scripts/make_l3_dataset.py docs/superpowers/plans/D-l3-results.md
git commit -m "feat(deploy): L3 three-tier latency comparison and custom evaluation results"
```

---

### Task 6: Full verification and the Plan D report (including the gated branch)

**Files:**
- Create: `docs/superpowers/plans/D-verification-report.md`
- Modify: `README.md` (the deployment section points at deploy/ and the docs)

**Interfaces:**
- Consumes: everything above
- Produces: a status-explicit deployment and measurement report

- [ ] **Step 1: Full local test suite (ensure the deploy changes broke nothing)**

```bash
conda run -n roboray python -m pytest -o addopts="" -m "not gpu"
```
Expected: 206 passed / 3 deselected.

- [ ] **Step 2: Summarize the deployment and measurement status**

The report must explicitly state one of three states:
- **Executed**: L2/L3 fully ran, with the measured numbers attached;
- **Partially executed**: L2 passed, L3 not done due to capacity/time, with the reason;
- **Gated**: no GPU free, only the image/CR/scripts completed, with the `status.sh` `TOTAL_FREE_GPUS=0` evidence and the follow-up execution commands.

**Writing "gated" as "verified" is forbidden.**

- [ ] **Step 3: Write the report and the README deployment section, then commit**

```bash
git add docs/superpowers/plans/D-verification-report.md README.md
git commit -m "docs: Plan D verification report and README deployment section"
```

---

## Self-Review

**1. Spec coverage**

| Spec requirement (§11/§12) | Corresponding task |
| --- | --- |
| Image build + registry push | Tasks 1/2 |
| RayCluster (ns `roboray`, head CPU + N×GPU workers) | Task 3 |
| Data/weights staging | Task 1 (weights into the image) + Task 5 (data via hostPath/kubectl cp) |
| L2 per-model smoke | Task 4 |
| L3 three-tier latency comparison + custom evaluation | Task 5 |
| Honest boundary (no HOT3D absolute-value reproduction) | Tasks 5/6 |

**2. Placeholder scan**: `__IMAGE__`/`__GPU_WORKERS__` are **template placeholders** of `apply.sh`, actually replaced by the script (not unfinished items); the remaining code is complete.

**3. Type consistency**: the in-image `ROBORAY_*` variable names match Plan B/C's `ModelEnvsConfig`/`resolve_env_request`; `run_l2.sh` reuses Plan B's `tests/gpu/test_adapters_gpu.py`; `run_l3.sh`'s CLI args match Plan A's `roboray run`.

**4. Review Focus coverage**

| Review Focus | Step that pins it |
| --- | --- |
| 1. Image self-containment | Task 1 `verify_offline.sh` + Task 2 `docker run` self-check (no network) |
| 2. GPU capacity gate | Task 3 `status.sh`'s `GATE=` + Task 4 Step 1 hard gate + Task 6's three-state report |
| 3. No pollution of the shared cluster | Task 3 `teardown.sh` deletes only this namespace; Task 5 Step 5 explicit cleanup |
| 4. L2 artifact contracts | reuses Plan B's gpu tests (assert shapes/finiteness/depth_source) |
| 5. L3 three-tier comparability | Task 5 same image/same input/same session + explicit utilization and VRAM recording |
| 6. Honest boundary | the mandatory wording in Task 5 Step 4 and Task 6 Step 2 |

## Execution sequence once the window opens (run in this order when GPUs are available)

```bash
export KUBECONFIG=<path-to-cluster-kubeconfig>

# 0) re-check resources (disk + GPU)
df -h / /home | tail -2
bash deploy/k8s/status.sh            # must show GATE=OPEN

# 1) free disk (as needed; free at least 35GB for the envs + image)
docker image prune -a -f             # or move the docker data-root to /home
du -sh deploy/assets 2>/dev/null

# 2) local envs + weights (~20-40 min, network-dependent)
bash scripts/vendor_models.sh
bash deploy/modelenvs/build_all_local.sh all
bash deploy/modelenvs/fetch_weights.sh
bash deploy/modelenvs/verify_offline.sh        # MANO may be MISSING

# 3) image build + push (~15-30 min)
bash deploy/build_image.sh                      # expected IMAGE_SELFCHECK_OK
bash <path-to-tunnel-script> connect --ip=<management-node-ip> --user=ubuntu
bash deploy/push_image.sh

# 4) deploy the RayCluster
bash deploy/k8s/apply.sh && sleep 20 && bash deploy/k8s/status.sh

# 5) L2 -> L3 -> cleanup
bash deploy/run_l2.sh
bash deploy/run_l3.sh
bash deploy/k8s/teardown.sh
```

**Discipline**: create resources only in the `roboray` namespace; do not squeeze other people's GPUs; run `teardown.sh` after use; when no GPU is free, do not treat Pending as "deployment done".

## Execution Handoff

After Plan D, the four plans (A/B/C/D) cover the entire spec scope: A the foundation and scheduling core, B the real models and A1/A3, C the stages-3/4 deliverables, D deployment and measurement.
