#!/usr/bin/env python3
"""K8s manifest renderer -- the **single** entry point shared by apply.sh and the tests.

Substitutes __PLACEHOLDER__ tokens in the template with environment variables; after
rendering it scans for leftover placeholders and exits 2 if any are found (deploying a
manifest with a __XXX__ in it would fail anyway; failing early beats failing silently).
Removes the duplicate sed logic in apply.sh: changing a template key name changes one place.
"""

import os
import re
import sys
from pathlib import Path

_DEFAULTS = {
    "IMAGE": "registry.example.com/roboray/base:v0.26.0",
    "NODE": "node-1",
    "MAX_GPU_WORKERS": "2",
    "MAX_CPU_WORKERS": "4",
    # SHARED_CLAIM is only a path component of the shared-volume hostPath (**not** a PVC
    # switch); for multi-node setups replace the shared-input / shared-output volume blocks
    # in the manifest manually.
    "SHARED_CLAIM": "",
    # RayJob entrypoint placeholder (the rayjob.yaml template uses __ENTRYPOINT__).
    "ENTRYPOINT": (
        "cd /ego/app && /ego/roboray-venv/bin/python -m roboray.cli run "
        "--config /ego/app/configs/default.yaml"
    ),
}


def render(template_text: str, values: dict) -> str:
    out = template_text
    for key, val in values.items():
        out = out.replace(f"__{key}__", val)
    left = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", out)))
    if left:
        sys.stderr.write(f"leftover placeholders after rendering (unknown to renderer): {', '.join(left)}\n")
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
    # Shared-volume placeholders: SHARED_CLAIM is only a path component of the shared-volume
    # hostPath (empty = use the default name). **Not** a PVC switch -- for multi-node setups
    # replace the shared-input / shared-output volume blocks manually (see the raycluster.yaml
    # header note).
    values["SHARED_INPUT"] = values["SHARED_CLAIM"] or "hostPath-shared-input"
    values["SHARED_OUTPUT"] = values["SHARED_CLAIM"] or "hostPath-shared-output"
    # The template name may be relative to cwd or a bare filename -- the latter is resolved
    # against the render.py directory (deploy/k8s).
    arg = Path(sys.argv[1])
    template_path = arg
    if not arg.is_absolute() and not arg.exists():
        template_path = Path(__file__).resolve().parent / arg
    template = template_path.read_text(encoding="utf-8")
    sys.stdout.write(render(template, values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
