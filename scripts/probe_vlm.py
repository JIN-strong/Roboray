#!/usr/bin/env python
"""VLM reachability probe: can the platform vLLM be used through the openai_vlm path (spec §5.1).

Three-stage probe: list models (best-effort) -> minimal text chat -> 8x8 PNG multimodal chat.
stdout's first line is a machine-readable PROBE_RESULT=<tag>, tag in
{reachable_multimodal, reachable_text_only, auth_failed, unreachable}.
Exit code: reachable_multimodal -> 0, everything else -> 1.

Must run inside a cluster pod (login-node reachability does not imply pod-network reachability):
  python scripts/probe_vlm.py --base-url http://<host>:<port>/v1 --model <name>
"""
import argparse
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

__all__ = ["probe", "classify_exc", "resolve_key"]


def resolve_key(api_key_env: str) -> tuple[str, list[str]]:
    """Get the key from an env var; fall back to a dummy 'EMPTY' if missing (vLLM convention)."""
    lines = []
    key = os.environ.get(api_key_env, "")
    if key:
        lines.append(f"auth: using env var {api_key_env}")
    else:
        key = "EMPTY"
        lines.append(f"warning: {api_key_env} is empty, using dummy key 'EMPTY' (vLLM convention)")
    return key, lines


def classify_exc(exc: Exception) -> str:
    """Exception -> tag. 401/403 or an auth-named type -> auth_failed; otherwise -> unreachable."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "auth_failed"
    if "auth" in type(exc).__name__.lower():
        return "auth_failed"
    return "unreachable"


def _tiny_png_data_uri() -> str:
    """8x8 grayscale PNG -> data URI (cv2-encoded, within core deps)."""
    import cv2
    import numpy as np

    ok, enc = cv2.imencode(".png", np.full((8, 8), 128, np.uint8))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode()


def _content(prompt: str, image_uri: str | None) -> object:
    """Same multimodal format as semantic_models._build_content."""
    if image_uri is None:
        return prompt
    return [{"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_uri}}]


def probe(base_url: str, model: str, api_key: str, timeout: float = 30.0,
          client_factory=None) -> tuple[str, list[str]]:
    """Run the three-stage probe; return (tag, human-readable detail lines). client_factory can be injected for tests."""
    details: list[str] = []
    if client_factory is None:
        from openai import OpenAI  # lazy: same as semantic_models (Global Constraints)
        client_factory = OpenAI
    try:
        client = client_factory(base_url=base_url, api_key=api_key, timeout=timeout)
    except Exception as exc:
        return classify_exc(exc), [f"client construction failed: {type(exc).__name__}: {exc}"]
    # 1) list models (best-effort)
    try:
        names = [m.id for m in client.models.list().data]
        details.append(f"models: {names}")
    except Exception as exc:
        details.append(f"models list failed: {type(exc).__name__}: {exc}")
        return classify_exc(exc), details
    # 2) minimal text chat
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=8, timeout=timeout,
            messages=[{"role": "user", "content": "ping; answer only pong"}])
        details.append(f"text: {repr((resp.choices[0].message.content or '')[:40])}")
    except Exception as exc:
        details.append(f"text chat failed: {type(exc).__name__}: {exc}")
        return classify_exc(exc), details
    # 3) multimodal
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=8, timeout=timeout,
            messages=[{"role": "user",
                       "content": _content("What is in this image? Answer in one word.", _tiny_png_data_uri())}])
        details.append(f"multimodal: {repr((resp.choices[0].message.content or '')[:40])}")
    except Exception as exc:
        details.append(f"multimodal failed: {type(exc).__name__}: {exc}")
        return "reachable_text_only", details
    return "reachable_multimodal", details


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True, help="chat model name")
    ap.add_argument("--api-key-env", default="VLM_API_KEY")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    key, lines = resolve_key(args.api_key_env)
    for line in lines:
        print(line)
    tag, details = probe(args.base_url, args.model, key, timeout=args.timeout)
    print(f"PROBE_RESULT={tag}")
    for line in details:
        print(f"  {line}")
    return 0 if tag == "reachable_multimodal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
