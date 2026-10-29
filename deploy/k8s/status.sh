#!/usr/bin/env bash
set -uo pipefail
# Print roboray RayCluster status, elasticity config, and GPU idle determination.
#
# Convention (see docs/superpowers/plans/2026-10-19-plan-d-deploy-l2-l3.md item 28):
#   **When no GPU is free, a Pod must not be stuck in Pending pretending deployment
#   is complete** -- this script explicitly prints TOTAL_FREE_GPUS; when it is 0 and
#   no auto-scaling cluster exists, L2/L3 must be marked gated.
#
# GATE semantics (spec §7): OPEN iff
#   TOTAL_FREE_GPUS > 0  or  a RayCluster with enableInTreeAutoscaling: true exists.
#   A cluster that scales from zero normally sits at zero usage, so "an autoscaling
#   cluster exists" counts as deployable too.
#
# Not implemented (registered, not a downgrade): the **number of idle-unload events**
# is not output by this script. It is a driver-side runtime artifact and can only be
# queried from the run's timeline/manifest (manifest.json scheduler_snapshots / timeline),
# while status.sh only reads k8s (this script does not read the workspace) -- so it is
# honestly not reported; do not assume coverage from its absence.
#
# Usage: bash deploy/k8s/status.sh [namespace]   (default roboray)
NS="${1:-roboray}"
KUBECTL="${KUBECTL:-kubectl}"
PYTHON="${PYTHON:-python3}"

echo "== Node GPU capacity =="
NODES_PY=$(cat <<'PY'
import json, sys
total = 0
for n in json.load(sys.stdin)["items"]:
    name = n["metadata"]["name"]
    cap = int(n["status"].get("capacity", {}).get("nvidia.com/gpu", 0) or 0)
    alloc = int(n["status"].get("allocatable", {}).get("nvidia.com/gpu", 0) or 0)
    total += cap
    if cap:
        print(f"  {name}: capacity={cap} allocatable={alloc}")
print(f"TOTAL_GPUS={total}")
PY
)
{ "$KUBECTL" get nodes -o json 2>/dev/null || true; } | "$PYTHON" -c "$NODES_PY" 2>/dev/null || echo "(could not parse nodes; check kubectl permissions)"

echo "== Pods occupying GPUs =="
PODS_PY=$(cat <<'PY'
import json, sys
used = 0
for p in json.load(sys.stdin)["items"]:
    if p["status"].get("phase") != "Running":
        continue
    for c in p["spec"]["containers"]:
        g = c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu")
        if g:
            used += int(g)
            ns = p["metadata"]["namespace"]
            nm = p["metadata"]["name"]
            nd = p["spec"].get("nodeName")
            print(f"  {ns}/{nm} node={nd} gpu={g}")
print(f"USED_GPUS={used}")
PY
)
{ "$KUBECTL" get pods -A -o json 2>/dev/null || true; } | "$PYTHON" -c "$PODS_PY" 2>/dev/null || echo "(could not parse Pods)"

echo "== RayCluster / Pod status =="
"$KUBECTL" -n "$NS" get raycluster 2>/dev/null || echo "  (no RayCluster)"
"$KUBECTL" -n "$NS" get pods -o wide 2>/dev/null || echo "  (no Pods)"

echo "== RayCluster elasticity =="
RC_PY=$(cat <<'PY'
import json, sys
try:
    clusters = json.load(sys.stdin)["items"]
except Exception:
    clusters = None
if not clusters:
    print("  (no RayCluster or no permission)")
    print("RAYCLUSTER_AUTOSCALING=none")
    print("RAYCLUSTER_MAX_GPUS=0")
    raise SystemExit(0)
auto = False
max_gpus = 0
for c in clusters:
    spec = c.get("spec", {})
    a = spec.get("enableInTreeAutoscaling") is True
    auto = auto or a
    tot = 0
    for g in spec.get("workerGroupSpecs", []):
        if g.get("rayStartParams", {}).get("num-gpus") == "1":
            tot += int(g.get("maxReplicas", 0) or 0)
    max_gpus += tot
    print(f'  {c["metadata"]["name"]}: autoscaling={str(a).lower()} max_gpu_workers={tot}')
print(f"RAYCLUSTER_AUTOSCALING={str(auto).lower()}")
print(f"RAYCLUSTER_MAX_GPUS={max_gpus}")
PY
)
{ "$KUBECTL" -n "$NS" get raycluster -o json 2>/dev/null || true; } | "$PYTHON" -c "$RC_PY" 2>/dev/null || echo "  RAYCLUSTER_AUTOSCALING=none"

echo "== RayJob =="
JOB_PY=$(cat <<'PY'
import json, sys
try:
    jobs = json.load(sys.stdin)["items"]
except Exception:
    jobs = None
if not jobs:
    print("  (none)")
    print("RAYJOBS=none")
    raise SystemExit(0)
names = []
for j in jobs:
    m = j.get("metadata", {})
    st = j.get("status", {}).get("jobStatus", "?")
    print(f'  {m.get("name", "?")}: status={st}')
    names.append(f'{m.get("name", "?")}({st})')
print("RAYJOBS=" + " ".join(names))
PY
)
{ "$KUBECTL" -n "$NS" get rayjob -o json 2>/dev/null || true; } | "$PYTHON" -c "$JOB_PY" 2>/dev/null || echo "  RAYJOBS=none"

echo "== Determination =="
echo "(note: the number of idle-unload events is not printed here -- it must be queried from the"
echo " run's timeline/manifest: manifest.json scheduler_snapshots/timeline; status.sh only reads"
echo " k8s and does not read the workspace.)"
GATE_PY=$(cat <<'PY'
import json, os, subprocess
kc = os.environ.get("KUBECTL", "kubectl")
ns = os.environ.get("NS", "roboray")

def kget(args):
    return subprocess.check_output([kc, *args], stderr=subprocess.DEVNULL)

free = "unknown"
try:
    nodes = json.loads(kget(["get", "nodes", "-o", "json"]))["items"]
    cap = sum(int(n["status"].get("capacity", {}).get("nvidia.com/gpu", 0) or 0) for n in nodes)
    used = 0
    for p in json.loads(kget(["get", "pods", "-A", "-o", "json"]))["items"]:
        if p["status"].get("phase") != "Running":
            continue
        for c in p["spec"]["containers"]:
            g = c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu")
            if g:
                used += int(g)
    free = str(cap - used)
except Exception:
    free = "unknown"

auto = False
try:
    for c in json.loads(kget(["-n", ns, "get", "raycluster", "-o", "json"]))["items"]:
        if c.get("spec", {}).get("enableInTreeAutoscaling") is True:
            auto = True
            break
except Exception:
    pass

print(f"{free} {str(auto).lower()}")
PY
)
GATE_LINE=$(KUBECTL="$KUBECTL" NS="$NS" "$PYTHON" -c "$GATE_PY")
FREE="${GATE_LINE%% *}"
AUTOSCALING="${GATE_LINE##* }"
echo "TOTAL_FREE_GPUS=$FREE"
if [ "$AUTOSCALING" = "true" ]; then
  echo "GATE=OPEN (an enableInTreeAutoscaling cluster exists and can scale up; TOTAL_FREE_GPUS=$FREE only reflects idle right now)"
elif [ "$FREE" = "unknown" ]; then
  echo "GATE=UNKNOWN (cannot determine; do not claim deployment on this basis)"
elif [ "$FREE" -gt 0 ] 2>/dev/null; then
  echo "GATE=OPEN (free GPUs available; L2/L3 can run)"
else
  echo "GATE=CLOSED (no free GPUs and no auto-scaling cluster: L2/L3 must be marked gated, not silently done)"
fi
