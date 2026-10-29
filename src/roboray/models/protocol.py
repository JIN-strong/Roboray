"""Model interface protocols. Both real models and test stand-ins implement these."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = ["SceneModel", "HandModel", "SemanticModel", "resolution_key"]

_RESOLUTION_BUCKET_PX = 64


def resolution_key(width: int, height: int) -> str:
    """Quantize the resolution to 64px buckets, so the cost model keeps one EMA per bucket.

    Resolutions already aligned to 32px (e.g. 640x480) keep their value, so common
    sizes like 480p are not mis-bucketed into the 448 tier; other sizes round down to
    the nearest 64px bucket (e.g. 641 -> 640) so jitter does not fragment the EMA.
    """

    def _bucket(x: int) -> int:
        x = int(x)
        if x % (_RESOLUTION_BUCKET_PX // 2) == 0:
            return x
        return max(_RESOLUTION_BUCKET_PX, (x // _RESOLUTION_BUCKET_PX) * _RESOLUTION_BUCKET_PX)

    return f"{_bucket(width)}x{_bucket(height)}"


@runtime_checkable
class SceneModel(Protocol):
    name: str

    def load(self, device: str) -> None:
        """Load weights to the given device inside the actor."""

    def infer(self, frames: dict[str, Any], calib: dict[str, Any], out_path: Path) -> None:
        """Run Scene reconstruction and write the artifact to out_path."""

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        """Cold-start cost prior (seconds)."""


@runtime_checkable
class HandModel(Protocol):
    name: str

    def load(self, device: str) -> None: ...
    def infer(self, frames: dict[str, Any], calib: dict[str, Any], out_path: Path) -> None: ...
    def cost_hint(self, n_frames: int, resolution_key: str) -> float: ...


@runtime_checkable
class SemanticModel(Protocol):
    name: str

    def label(self, prompt: str, images: list, out_path: Path, n_segments: int = 0) -> None:
        """Write action semantics to out_path (NPZ/JSON is up to the implementation;
        the contract is defined by the action stage).

        n_segments is a segment-count hint (optional length constraint); the
        implementation should return exactly that many results.
        """
