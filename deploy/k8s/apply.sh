#!/usr/bin/env bash
set -euo pipefail
# Bring up a **long-running** RayCluster (head stays online; jobs connect to it directly).
# For one-shot jobs use submit_job.sh (RayJob: spawns a fresh cluster + a separate Job,
# torn down when done).
#   Correction 2026-10-24: that K8s Job pod is the **submission client** (runs `ray job submit`);
#   the Ray driver process by Ray design **runs inside the head container**, not that Job pod.
#
# Environment variables:
#   IMAGE (required)      -- image, of the form <registry>/roboray:<tag>
#   NODE                  -- node holding the ego-models hostPath (nodeSelector)
#   MAX_GPU_WORKERS       -- gpu-workers group maxReplicas (default 2)
#   MAX_CPU_WORKERS       -- cpu-workers group maxReplicas (default 4)
#   SHARED_CLAIM          -- path component for the shared volume (empty = fall back to hostPath, single node).
#                           Note: this is **not a PVC switch** -- it is only spliced into the hostPath path;
#                           for multi-node setups replace the volume blocks in the manifest manually
#                           (see the header of raycluster.yaml).
#
# Rendering always goes through render.py (single entry point): changing a template key name
# changes one place, and any leftover placeholder causes exit 2.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KUBECTL="${KUBECTL:-kubectl}"
PYTHON="${PYTHON:-python3}"
IMAGE="${IMAGE:?set IMAGE=<registry>/roboray:<tag>}"
NODE="${NODE:-node-1}"   # node holding the ego-models hostPath
"$KUBECTL" apply -f "$ROOT/deploy/k8s/namespace.yaml"
IMAGE="$IMAGE" NODE="$NODE" MAX_GPU_WORKERS="${MAX_GPU_WORKERS:-2}" MAX_CPU_WORKERS="${MAX_CPU_WORKERS:-4}" SHARED_CLAIM="${SHARED_CLAIM:-}" \
  "$PYTHON" "$ROOT/deploy/k8s/render.py" "$ROOT/deploy/k8s/raycluster.yaml" | "$KUBECTL" apply -f -
echo "RayCluster submitted (node=${NODE}). Watch: $KUBECTL -n roboray get pods -o wide"
