"""Structured artifact read/write helpers: uniformly stamp a schema version.

All products (scene/hand/fused/action etc. NPZ) go through ``save_artifact`` for writing
and ``load_artifact`` for reading, so version validation applies end to end: writers stamp,
readers validate, and mismatched/corrupt versions fail immediately with a readable
ValueError instead of being silently consumed.
"""

import numpy as np

from roboray.runtime.episode import ARTIFACT_SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "save_artifact", "load_artifact", "check_artifact_file"]

# Re-exported so consumers (tests) can reference the same constant from here.
SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION


def save_artifact(path, *, compressed: bool = False, **arrays):
    """Write a named-dict as an NPZ stamped with the schema version.

    ``compressed=True`` uses ``np.savez_compressed`` (used for large outputs such as the
    LeRobot data.npz); version stamp and decompression read logic stay consistent.
    """
    if compressed:
        np.savez_compressed(path, __schema_version__=ARTIFACT_SCHEMA_VERSION, **arrays)
    else:
        np.savez(path, __schema_version__=ARTIFACT_SCHEMA_VERSION, **arrays)
    return path


def check_artifact_file(path) -> bool:
    """File-level structural validation: readable, schema version stamp present, version matches.

    (Same semantics as ``EpisodeWorkspace.check_artifact``, but takes a bare path, so
    modules inside stages that do not depend on the workspace object can reuse it.)
    """
    p = path
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            if "__schema_version__" not in z.files:
                return False
            return int(z["__schema_version__"]) == ARTIFACT_SCHEMA_VERSION
    except (OSError, ValueError, EOFError):
        return False


def load_artifact(path) -> dict:
    """Read a structured artifact and validate its version; corrupt/missing/version-mismatch
    always raise a readable ValueError."""
    p = path
    if not p.exists():
        raise ValueError(
            f"artifact missing: {p.name} (expected schema version {ARTIFACT_SCHEMA_VERSION})"
        )
    try:
        z = np.load(p, allow_pickle=True)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"artifact corrupt: {p.name} ({exc})") from exc
    with z:
        try:
            version = int(np.asarray(z["__schema_version__"]).ravel()[0])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"artifact {p.name} missing or invalid schema version stamp ({exc})"
            ) from exc
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact {p.name} schema version mismatch: file has {version}, "
                f"current supports {ARTIFACT_SCHEMA_VERSION}, rerun that stage"
            )
        return {k: z[k] for k in z.files if k != "__schema_version__"}
