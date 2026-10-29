#!/usr/bin/env bash
set -euo pipefail
# Submit a **one-shot** RayJob (serverless feel).
#
# Division of labor with apply.sh:
#   apply.sh   brings up a long-running RayCluster (head stays online).
#   This script submits a RayJob: the operator spawns a separate Ray cluster from
#              spec.rayClusterSpec, then a separate **submission-client** K8s Job pod
#              (K8sJobMode, runs `ray job submit`).
#              Correction 2026-10-24: the Ray driver process by Ray design **runs in
#              the head container**, not that K8s Job pod. When the job finishes, the
#              whole cluster is torn down per shutdownAfterJobFinishes.
#
# Environment variables:
#   ENTRYPOINT           -- override the job entry command (default cd /ego/app && ... roboray.cli run).
#                           If empty, render.py's default value applies.
#   IMAGE (required)     -- same as apply.sh, of the form <registry>/roboray:<tag>
#   NODE                 -- same as apply.sh (hostPath node affinity)
#   MAX_GPU_WORKERS      -- gpu-workers group maxReplicas (default 2)
#   MAX_CPU_WORKERS      -- cpu-workers group maxReplicas (default 4)
#   SHARED_CLAIM         -- only a path component of the shared-volume hostPath (empty = use default name).
#                           **Not a PVC switch** -- for multi-node setups replace the
#                           shared-input/shared-output volume blocks in rayjob.yaml manually.
#
# Rendering always goes through render.py (single entry point); leftover placeholders cause exit 2.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KUBECTL="${KUBECTL:-kubectl}"
PYTHON="${PYTHON:-python3}"
ENTRYPOINT="${ENTRYPOINT:-}"
IMAGE="${IMAGE:?set IMAGE=<registry>/roboray:<tag>}"
NODE="${NODE:-node-1}"

# Fresh cluster: create the namespace first (running only submit_job.sh must not fail for a missing ns).
"$KUBECTL" apply -f "$ROOT/deploy/k8s/namespace.yaml"

ENTRYPOINT="$ENTRYPOINT" IMAGE="$IMAGE" NODE="$NODE" \
  MAX_GPU_WORKERS="${MAX_GPU_WORKERS:-2}" MAX_CPU_WORKERS="${MAX_CPU_WORKERS:-4}" SHARED_CLAIM="${SHARED_CLAIM:-}" \
  "$PYTHON" "$ROOT/deploy/k8s/render.py" "$ROOT/deploy/k8s/rayjob.yaml" | "$KUBECTL" apply -f -

echo "RayJob submitted (name=roboray-job). Watch: $KUBECTL -n roboray get rayjob"
echo "The cluster name is derived by the operator (random suffix, non-deterministic) -- see the actual name with:"
echo "  $KUBECTL -n roboray get raycluster"
echo "teardown cleans up by type with --all and does not depend on the cluster name."
