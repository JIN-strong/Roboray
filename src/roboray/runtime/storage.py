"""Storage abstraction. Plan A implements POSIX only; OSS/EMR is an optional plugin."""

from pathlib import Path
from typing import Iterator, Protocol

__all__ = ["Storage", "PosixStorage", "make_storage"]


class Storage(Protocol):
    def exists(self, rel: str) -> bool: ...
    def is_dir(self, rel: str) -> bool: ...
    def listdir(self, rel: str) -> list[str]: ...
    def read_bytes(self, rel: str) -> bytes: ...
    def write_bytes(self, rel: str, data: bytes) -> None: ...
    def local_path(self, rel: str) -> Path: ...
    def walk(self, rel: str) -> list[str]: ...


class PosixStorage:
    """Local filesystem storage rooted at ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def _p(self, rel: str) -> Path:
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path escapes root: {rel!r} resolves to {candidate}, outside root {self.root}")
        return candidate

    def exists(self, rel: str) -> bool:
        return self._p(rel).exists()

    def is_dir(self, rel: str) -> bool:
        return self._p(rel).is_dir()

    def listdir(self, rel: str) -> list[str]:
        return sorted(p.name for p in self._p(rel).iterdir())

    def read_bytes(self, rel: str) -> bytes:
        return self._p(rel).read_bytes()

    def write_bytes(self, rel: str, data: bytes) -> None:
        p = self._p(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def local_path(self, rel: str) -> Path:
        return self._p(rel)

    def walk(self, rel: str) -> list[str]:
        base = self._p(rel)
        if not base.exists():
            return []
        out = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
        return sorted(out)


def make_storage(uri: str) -> Storage:
    if uri.startswith("posix://"):
        return PosixStorage(Path(uri[len("posix://") :]))
    if "://" in uri:
        scheme = uri.split("://", 1)[0]
        raise ValueError(
            f"unsupported storage scheme {scheme!r}: only posix:// or bare filesystem "
            "paths are implemented; OSS/EMR is an optional future plugin."
        )
    return PosixStorage(Path(uri))
