#!/usr/bin/env bash
set -uo pipefail
# L3 three-tier timing comparison: runs serial / pipeline / adaptive sequentially **within
# the same RayCluster session**, all three tiers on the **same input** (only the scheduling
# switches differ), avoiding hardware drift across time windows.
#
# Exact semantics of the three tiers (see the header of src/roboray/runtime/driver.py):
#   serial_baseline=True                   -> per-Episode, per-stage serial (baseline)
#   serial_baseline=False, adaptive=False  -> cross-stage pipelining + static pooling
#   serial_baseline=False, adaptive=True   -> cross-stage pipelining + adaptive scheduling (target form)
# Hence: serial = --serial-baseline; pipeline = --no-adaptive; adaptive = default (no switch).
#
# Each tier uses a **separate output.uri** (the output.uri in the config is an absolute path
# and the CLI's --run-root only decides the workspace, it does not change it; without
# separating them the three tiers would overwrite each other's manifests).
#
# Run inside the KubeRay head pod:
#   kubectl -n roboray exec <head> -- bash -lc "bash /ego/app/deploy/run_l3.sh"
#
# Environment variables: CONF (base config) / TAG (output prefix) / P (interpreter)
CONF="${CONF:-/ego/l3-test.yaml}"
TAG="${TAG:-l3-run-test}"
P="${P:-/ego/roboray-venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.." || exit 1

make_conf() {  # $1=tier name -> /tmp/l3-$1.yaml (output.uri points at that tier's own directory)
  "$P" - "$CONF" "/tmp/l3-$1.yaml" "/ego/${TAG}-$1" <<'PY'
import sys, yaml
src, dst, root = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg.setdefault("output", {})["uri"] = root + "/out"
with open(dst, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
PY
}

run_one() {
  local name="$1"; shift
  make_conf "$name" || return 1
  rm -rf "/ego/${TAG}-${name}"
  echo "########## ${name} ##########"
  "$P" -m roboray.cli run --config "/tmp/l3-$name.yaml" --ray-address auto \
        --run-root "/ego/${TAG}-${name}" "$@" 2>&1 | tail -10
}

run_one serial   --serial-baseline
run_one pipeline --no-adaptive
run_one adaptive

echo "########## Summary ##########"
for n in serial pipeline adaptive; do
  m="/ego/${TAG}-${n}/out/manifest.json"
  if [ -f "$m" ]; then
    "$P" - "$n" "$m" <<'PY'
import json, sys
tag, path = sys.argv[1], sys.argv[2]
m = json.load(open(path, encoding="utf-8"))
s = m["summary"]
print(f'{tag}: total_seconds={s["total_seconds"]:.3f} n_episodes={s["n_episodes"]} failed={s["failed"]}')
PY
  else
    echo "${n}: no manifest ($m)"
  fi
done
