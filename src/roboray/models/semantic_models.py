"""Action-semantics model: OpenAI-compatible VLM (pluggable).

Decision: an unreachable VLM raises an error, no geometric degradation.
Config sources: env vars ROBORAY_VLM_BASE_URL / ROBORAY_VLM_MODEL / <vlm_api_key_env>.
"""

import json
import os
from pathlib import Path

import numpy as np

__all__ = ["OpenAIVlm", "VlmError"]


class VlmError(RuntimeError):
    """VLM call failed or is not configured."""


def _build_content(prompt: str, images: list) -> object:
    """Build the user message content: multimodal (text + keyframes) when there are
    images, pure text otherwise."""
    if not images:
        return prompt
    return [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": uri}} for uri in images
    ]


def _extract_labels(raw, n_segments: int) -> list:
    """Best-effort extraction of the label list from the model reply; on failure fall
    back to generic names (must not crash)."""
    labels = None
    text = (raw or "").strip()
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                labels = parsed
        except json.JSONDecodeError:
            start, end = text.find("["), text.rfind("]")
            if 0 <= start < end:
                try:
                    parsed = json.loads(text[start:end + 1])
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        labels = parsed
                except json.JSONDecodeError:
                    labels = None
    if not labels:
        labels = [f"segment_{i}" for i in range(n_segments)]
    if len(labels) < n_segments:
        labels = labels + [f"segment_{i}" for i in range(len(labels), n_segments)]
    return labels[:n_segments]


class OpenAIVlm:
    name = "openai_vlm"

    def __init__(self, client_factory=None, base_url=None, model=None, api_key_env="VLM_API_KEY"):
        self._client_factory = client_factory
        self._base_url = base_url
        self._model = model
        self._api_key_env = api_key_env

    def _resolve(self):
        base_url = self._base_url or os.environ.get("ROBORAY_VLM_BASE_URL", "")
        model = self._model or os.environ.get("ROBORAY_VLM_MODEL", "qwen-vl")
        api_key = os.environ.get(self._api_key_env, "")
        if not base_url:
            raise VlmError(
                "VLM is not configured: set action_understanding.vlm_base_url (or the "
                "ROBORAY_VLM_BASE_URL environment variable)."
            )
        if not api_key:
            raise VlmError(
                f"VLM credentials missing: environment variable {self._api_key_env} is "
                "empty (its name is set by action_understanding.vlm_api_key_env)."
            )
        return base_url, model, api_key

    def label(self, prompt: str, images: list, out_path: Path, n_segments: int = 0) -> None:
        base_url, model, api_key = self._resolve()
        factory = self._client_factory
        if factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - the main env has openai
                raise VlmError("the openai package is required (main dependency)") from exc
            factory = OpenAI
        try:
            client = factory(base_url=base_url, api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _build_content(prompt, images)}],
            )
            raw = resp.choices[0].message.content or ""
        except VlmError:
            raise
        except Exception as exc:
            raise VlmError(f"VLM call failed (base_url={base_url}, model={model}): {exc}") from exc
        labels = _extract_labels(raw, n_segments)
        np.savez(out_path, labels=np.array(labels, dtype=object),
                 segment_start=np.zeros(len(labels), np.int32),
                 segment_end=np.zeros(len(labels), np.int32))
