from pathlib import Path

import pytest

from roboray.runtime.storage import PosixStorage, make_storage


def test_write_read_roundtrip(tmp_path):
    s = PosixStorage(tmp_path)
    s.write_bytes("a/b.bin", b"hi")
    assert s.exists("a/b.bin")
    assert s.read_bytes("a/b.bin") == b"hi"
    assert s.listdir("a") == ["b.bin"]


def test_local_path_points_into_root(tmp_path):
    s = PosixStorage(tmp_path)
    assert s.local_path("x/y").parent == tmp_path / "x"


def test_walk_lists_relative_files(tmp_path):
    s = PosixStorage(tmp_path)
    s.write_bytes("ep1/meta.json", b"{}")
    s.write_bytes("ep2/meta.json", b"{}")
    assert sorted(s.walk(".")) == ["ep1/meta.json", "ep2/meta.json"]


def test_is_dir_distinguishes_file_and_dir(tmp_path):
    s = PosixStorage(tmp_path)
    s.write_bytes("README.md", b"note")
    s.write_bytes("ep1/meta.json", b"{}")
    assert s.is_dir(".")
    assert s.is_dir("ep1") is True
    assert s.is_dir("ep1/meta.json") is False
    assert s.is_dir("ep1/meta.json/..") is True
    assert s.is_dir("does-not-exist") is False


def test_make_storage_accepts_bare_path(tmp_path):
    assert isinstance(make_storage(str(tmp_path)), PosixStorage)


def test_make_storage_accepts_posix_scheme(tmp_path):
    assert isinstance(make_storage(f"posix://{tmp_path}"), PosixStorage)


def test_unsupported_scheme_is_rejected():
    with pytest.raises(ValueError) as exc:
        make_storage("oss://bucket/raw")
    assert "oss" in str(exc.value)


def test_path_traversal_is_rejected(tmp_path):
    s = PosixStorage(tmp_path)
    with pytest.raises(ValueError):
        s.local_path("../outside")
    with pytest.raises(ValueError):
        s.write_bytes("a/../../outside", b"x")
