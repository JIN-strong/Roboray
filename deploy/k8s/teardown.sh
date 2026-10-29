#!/usr/bin/env bash
set -uo pipefail
# Tear down roboray K8s resources. By default the namespace is kept (faster redeploys);
# --all deletes the namespace as well.
#
# Cascade order (rayjob must be deleted **first**):
#   1. RayJob -- it manages the K8s Job and the cluster it spawns; if deleted later it gets revived by reconciliation.
#   2. Job    -- parent Job of the submission-client pod (the Ray driver process runs in the head container);
#                deleting the raycluster does not cascade to it.
#   3. RayCluster
#   4. Pod    -- catch-all cleanup of leftovers.
#
# Usage:
#   bash deploy/k8s/teardown.sh          # delete RayJob / Job / RayCluster / Pod, keep ns
#   bash deploy/k8s/teardown.sh --all    # delete the namespace too
NS="${NS:-roboray}"
KUBECTL="${KUBECTL:-kubectl}"

echo "== Deleting RayJob (first: it manages the Job and its spawned cluster; deleted later it is revived by reconciliation) =="
"$KUBECTL" -n "$NS" delete rayjob --all --ignore-not-found 2>/dev/null || true

echo "== Deleting K8s Job (parent Job of the submission-client pod; the Ray driver process runs in the head container; deleting the raycluster does not cascade) =="
"$KUBECTL" -n "$NS" delete job --all --ignore-not-found 2>/dev/null || true

echo "== Deleting RayCluster =="
"$KUBECTL" -n "$NS" delete raycluster --all --ignore-not-found 2>/dev/null || true

echo "== Deleting Pod =="
"$KUBECTL" -n "$NS" delete pod --all --ignore-not-found 2>/dev/null || true

if [ "${1:-}" = "--all" ]; then
  echo "== Deleting namespace =="
  "$KUBECTL" delete namespace "$NS" --ignore-not-found 2>/dev/null || true
fi

echo "== Remaining =="
"$KUBECTL" -n "$NS" get all 2>/dev/null || echo "  (namespace deleted)"
