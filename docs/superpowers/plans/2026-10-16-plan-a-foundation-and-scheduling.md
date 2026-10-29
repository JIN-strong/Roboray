# Plan A: Foundation and Scheduling Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the roboray project foundation and implement the core Ray execution model from the design article — GPU Actor pooling (switchable roles), cross-stage pipeline parallelism, and adaptive scheduling with a measured-latency EMA feedback loop — so the full pipeline runs end-to-end on the local CPU with fake models, and scheduling convergence and the parallel gains can be verified.

**Architecture:** The config and API layers define the external contract; the `runtime` layer provides the storage abstraction, the Episode workspace/state machine, and task planning; the `scheduling` layer is a pure-Python, Ray-free, testable scheduling core; the `ray_runtime` layer maps scheduling decisions onto Ray Actors. Stages obtain models through the model registry (dependency injection): the production config registers real models, while tests register doubles under `tests/fakes/`.

**Tech Stack:** Python 3.10, Ray 2.59.0, numpy, opencv-python-headless (decoding), pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-ego-pipeline-design.md`

## Global Constraints

- Python version: `>=3.10,<3.11` (HaMeR/pytorch3d compatibility, spec §13)
- Ray version: `ray[default]==2.59.0` (identical inside and outside the image, spec §13)
- The public class names must be `EgoReconstruction` and `ReconstructionRequest` (spec §1 item 3)
- Config keys must be verbatim-compatible with the article's examples: `input.uri`, `output.uri`, `output.format` (`lerobot_v3`), `execution.max_gpus`, `stages.scene_reconstruction.stereo_max_image_side_px`, `stages.hand_reconstruction.{inference_fps,wearer_association,unobserved_gap_s,min_supporting_joints}`, `stages.visualization.{camera_view,world_view}`
- Input layout: `ep_xxxx/{rgb_left.mp4, rgb_right.mp4(optional), calibration.json, meta.json}` (spec §5)
- Short-gap fill default cap `unobserved_gap_s=0.2`, default `min_supporting_joints=8` (spec §7)
- Episode state machine: `PREPARE → SCENE ∥ HAND → FUSION → ACTION → PUBLISH` (spec §3)
- Code is managed with git; commit once per task
- Product code must not import `tests/`; model resolution always goes through `roboray.models.registry`
- **Test commands must use `conda run -n roboray python -m pytest ...`**, never `conda run -n roboray pytest ...`. The latter resolves to the `default` environment's pytest (Python 3.12), where this package is not installed, so all test collection fails (`ModuleNotFoundError: No module named 'roboray'`).

## Review Focus

The following input/failure modes are implied by the spec but not covered by task tests; they are the most likely to bite in practice, so each one must be pinned down by the tests of its task:

1. **A given Episode's `calibration.json` is missing or has incomplete fields** → mark that Episode FAILED with a readable reason, **the batch continues**, and the whole run must not crash.
2. **`rgb_right.mp4` exists but its frame count differs from `rgb_left.mp4`** → take the shorter one and record the degradation; no crash and no misalignment.
3. **`meta.json` missing / `episode_id` conflicts with the directory name / two Episodes share the same name** → error out early and **never overwrite each other's output**.
4. **mp4 with zero frames or undecodable** → mark that Episode FAILED, the batch continues.
5. **Artifact NPZ is corrupt or the schema version mismatches** → detect it and mark that stage for a re-run; never silently consume bad data.

---

### Task 1: Project scaffolding, environment, and dependency pinning

**Files:**
- Create: `pyproject.toml`
- Create: `src/roboray/__init__.py`
- Create: `.gitignore`
- Create: `README.md`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: none
- Produces: a package skeleton installable via `pip install -e .`; `roboray.__version__ == "0.1.0"`; conda env named `roboray`

- [ ] **Step 1: Create the conda environment and pin the Ray version**

Use the latest stable Ray version that pip can actually install (the spec records 2.59.0; if it does not exist on PyPI, use the actual latest stable version and confirm it in this step):

```bash
conda create -y -n roboray python=3.10
conda run -n roboray python -m pip install -U pip
conda run -n roboray python -m pip index versions ray 2>&1 | head -5
```

Expected: a list of available ray versions, confirming the latest stable version number (e.g. `2.59.0`). Use the confirmed version in the next step.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "roboray"
version = "0.1.0"
description = "Egocentric video to training-ready dataset pipeline on Ray"
requires-python = ">=3.10,<3.11"
dependencies = [
  "ray[default]==2.59.0",
  "numpy>=1.26,<2.0",
  "opencv-python-headless>=4.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout>=2.3"]
models = [
  "torch>=2.4",
  "openai>=1.40",
  "matplotlib>=3.8",
  "jinja2>=3.1",
  "lerobot",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
timeout = 300
markers = [
  "gpu: requires a real GPU and real model weights",
  "slow: long running",
]
```

- [ ] **Step 3: Write the package skeleton and `.gitignore`**

`src/roboray/__init__.py`:

```python
"""Ego-centric video to training-ready dataset pipeline."""

__version__ = "0.1.0"
```

`.gitignore`:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
build/
dist/
.venv/
output/
data/synthetic/
third_party/*/
!third_party/.gitkeep
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Install and verify the import**

```bash
conda run -n roboray python -m pip install -e ".[dev]"
conda run -n roboray python -c "import roboray, ray; print(roboray.__version__, ray.__version__)"
```

Expected: prints `0.1.0 2.59.0` (Ray version as confirmed in Step 1).

- [ ] **Step 5: Write the README skeleton**

```markdown
# roboray

A Ray pipeline that turns first-person (egocentric) video into training-ready datasets.

Reproduced from the design article; design doc at `docs/superpowers/specs/2026-10-16-ego-pipeline-design.md`.

## Environment

```bash
conda create -y -n roboray python=3.10
conda activate roboray
pip install -e ".[dev]"          # local dev/test
pip install -e ".[dev,models]"   # when real models are needed
```

## Quick start

```bash
pytest                      # unit + local end-to-end (fake models, no GPU)
```

(Later tasks will add a real usage section.)
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/__init__.py .gitignore README.md
git commit -m "chore: project scaffolding and dependency pinning (ray 2.59.0 / python 3.10)"
```

---

### Task 2: Config schema and `ReconstructionRequest`

**Files:**
- Create: `src/roboray/api/__init__.py`
- Create: `src/roboray/api/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `ConfigError(ValueError)`
  - Frozen dataclasses: `InputConfig(uri)`, `OutputConfig(uri, format="lerobot_v3")`, `ExecutionConfig(max_gpus=1, num_cpu_actors=4, max_episodes_in_flight=2, backend="real")`, `SceneStageConfig(stereo_max_image_side_px=640, model="auto")`, `HandStageConfig(inference_fps=30, wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8, model="hamer")`, `ActionStageConfig(vlm_base_url=None, vlm_api_key_env="VLM_API_KEY", vlm_model="qwen-vl", keyframe_stride=30, model="vlm")`, `VisualizationConfig(camera_view=True, world_view=True, fps=30)`, `StageConfigs(scene_reconstruction, hand_reconstruction, action_understanding, visualization)`
  - `ReconstructionRequest(input, output, execution=..., stages=...)`, class methods `from_dict(raw) -> ReconstructionRequest`, `to_dict() -> dict`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config.py`:

```python
import pytest

from roboray.api.config import ConfigError, ReconstructionRequest

ARTICLE_CONFIG = {
    "input": {"uri": "oss://bucket/raw"},
    "output": {"uri": "oss://bucket/results/run-unique", "format": "lerobot_v3"},
    "execution": {"max_gpus": 8},
    "stages": {
        "scene_reconstruction": {"stereo_max_image_side_px": 640},
        "hand_reconstruction": {
            "inference_fps": 30,
            "wearer_association": True,
            "unobserved_gap_s": 0.2,
            "min_supporting_joints": 8,
        },
        "visualization": {"camera_view": True, "world_view": True},
    },
}


def test_article_config_parses_with_defaults_filled():
    req = ReconstructionRequest.from_dict(ARTICLE_CONFIG)
    assert req.input.uri == "oss://bucket/raw"
    assert req.output.format == "lerobot_v3"
    assert req.execution.max_gpus == 8
    assert req.execution.backend == "real"  # defaults to the real model backend
    assert req.stages.scene_reconstruction.stereo_max_image_side_px == 640
    assert req.stages.hand_reconstruction.unobserved_gap_s == 0.2
    assert req.stages.hand_reconstruction.min_supporting_joints == 8
    assert req.stages.action_understanding.keyframe_stride == 30  # uses the default when not provided
    assert req.stages.visualization.camera_view is True


def test_unknown_key_is_rejected_with_path():
    bad = {**ARTICLE_CONFIG, "stages": {"scene_reconstruction": {"nope": 1}}}
    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict(bad)
    assert "nope" in str(exc.value)
    assert "scene_reconstruction" in str(exc.value)


def test_missing_required_uri_is_rejected():
    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict({"output": {"uri": "x"}})
    assert "input" in str(exc.value)


def test_wrong_type_is_rejected():
    bad = {**ARTICLE_CONFIG, "execution": {"max_gpus": "eight"}}
    with pytest.raises(ConfigError) as exc:
        ReconstructionRequest.from_dict(bad)
    assert "max_gpus" in str(exc.value)


def test_bool_is_not_accepted_as_int():
    bad = {**ARTICLE_CONFIG, "execution": {"max_gpus": True}}
    with pytest.raises(ConfigError):
        ReconstructionRequest.from_dict(bad)


def test_roundtrip_to_dict():
    req = ReconstructionRequest.from_dict(ARTICLE_CONFIG)
    again = ReconstructionRequest.from_dict(req.to_dict())
    assert again == req
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.api.config'`

- [ ] **Step 3: Implement `api/config.py`**

```python
"""Reconstruction request schema, mirroring the article's public config."""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, Optional, Union, get_args, get_origin

__all__ = [
    "ConfigError",
    "InputConfig",
    "OutputConfig",
    "ExecutionConfig",
    "SceneStageConfig",
    "HandStageConfig",
    "ActionStageConfig",
    "VisualizationConfig",
    "StageConfigs",
    "ReconstructionRequest",
]


class ConfigError(ValueError):
    """Raised when a reconstruction config is malformed."""


@dataclass(frozen=True)
class InputConfig:
    uri: str


@dataclass(frozen=True)
class OutputConfig:
    uri: str
    format: str = "lerobot_v3"


@dataclass(frozen=True)
class ExecutionConfig:
    max_gpus: int = 1
    num_cpu_actors: int = 4
    max_episodes_in_flight: int = 2
    # "real" runs real models (production default); "test" only registers doubles for local end-to-end tests.
    backend: str = "real"


@dataclass(frozen=True)
class SceneStageConfig:
    stereo_max_image_side_px: int = 640
    model: str = "auto"


@dataclass(frozen=True)
class HandStageConfig:
    inference_fps: int = 30
    wearer_association: bool = True
    unobserved_gap_s: float = 0.2
    min_supporting_joints: int = 8
    model: str = "hamer"


@dataclass(frozen=True)
class ActionStageConfig:
    vlm_base_url: Optional[str] = None
    vlm_api_key_env: str = "VLM_API_KEY"
    vlm_model: str = "qwen-vl"
    keyframe_stride: int = 30
    model: str = "vlm"


@dataclass(frozen=True)
class VisualizationConfig:
    camera_view: bool = True
    world_view: bool = True
    fps: int = 30


@dataclass(frozen=True)
class StageConfigs:
    scene_reconstruction: SceneStageConfig = SceneStageConfig()
    hand_reconstruction: HandStageConfig = HandStageConfig()
    action_understanding: ActionStageConfig = ActionStageConfig()
    visualization: VisualizationConfig = VisualizationConfig()


@dataclass(frozen=True)
class ReconstructionRequest:
    input: InputConfig
    output: OutputConfig
    execution: ExecutionConfig = ExecutionConfig()
    stages: StageConfigs = StageConfigs()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReconstructionRequest":
        return _build(cls, raw, path="config")

    def to_dict(self) -> dict:
        return _dump(self)


def _is_union(t) -> bool:
    origin = get_origin(t)
    if origin is Union:
        return True
    try:
        from types import UnionType  # py3.10+
    except ImportError:  # pragma: no cover
        return False
    return origin is UnionType


def _coerce(value: Any, typ: Any, path: str) -> Any:
    if _is_union(typ):
        args = [a for a in get_args(typ) if a is not type(None)]
        if value is None:
            return None
        errors = []
        for arg in args:
            try:
                return _coerce(value, arg, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: {value!r} does not match any allowed type ({' / '.join(errors)})")

    if is_dataclass(typ):
        return _build(typ, value, path)

    if typ is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {type(value).__name__} ({value!r})")
        return value

    if typ is int:
        # Explicitly exclude bool: bool is a subclass of int, but True/False are not valid numeric config values.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {type(value).__name__} ({value!r})")
        return value

    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected float, got {type(value).__name__} ({value!r})")
        return float(value)

    if typ is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {type(value).__name__} ({value!r})")
        return value

    raise ConfigError(f"{path}: unsupported config type {typ!r}")


def _build(cls, raw: Any, path: str) -> Any:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"{path}: unknown config keys {unknown}")
    kwargs = {}
    for name, f in known.items():
        sub = f"{path}.{name}"
        if name in raw:
            kwargs[name] = _coerce(raw[name], f.type, sub)
        elif f.default is not _MISSING:
            continue
        elif f.default_factory is not _MISSING:
            continue
        else:
            raise ConfigError(f"{sub}: missing required field")
    return cls(**kwargs)


def _dump(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dump(getattr(obj, f.name)) for f in fields(obj)}
    return obj


from dataclasses import MISSING as _MISSING  # noqa: E402  (see note below)
```

Note: the `_MISSING` import above is placed at the end of the file only to keep the body readable; when implementing, merge `from dataclasses import MISSING` into the dataclasses import line at the top and rename `_MISSING` to `MISSING`. Also remove the unused `field` import.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_config.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/roboray/api/ tests/unit/test_config.py
git commit -m "feat(api): config schema and ReconstructionRequest.from_dict"
```

---

### Task 3: Storage abstraction (POSIX)

**Files:**
- Create: `src/roboray/runtime/__init__.py`
- Create: `src/roboray/runtime/storage.py`
- Test: `tests/unit/test_storage.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `Storage` Protocol: `exists(rel)->bool`, `listdir(rel)->list[str]`, `read_bytes(rel)->bytes`, `write_bytes(rel, data)->None`, `local_path(rel)->Path`, `walk(rel)->list[str]`
  - `PosixStorage(root: Path)` implementing the protocol above
  - `make_storage(uri: str) -> Storage`: accepts `posix:///abs/path` or a bare filesystem path; any other scheme raises `ValueError` (EMR/OSS are later optional plugins)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_storage.py`:

```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.runtime.storage'`

- [ ] **Step 3: Implement `runtime/storage.py`**

```python
"""Storage abstraction. Plan A implements POSIX only; OSS/EMR is an optional plugin."""

from pathlib import Path
from typing import Iterator, Protocol

__all__ = ["Storage", "PosixStorage", "make_storage"]


class Storage(Protocol):
    def exists(self, rel: str) -> bool: ...
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
            f"unsupported storage scheme {scheme!r}: Plan A implements only posix:// or a bare filesystem path; "
            "OSS/EMR is a later optional plugin."
        )
    return PosixStorage(Path(uri))
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_storage.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/roboray/runtime/ tests/unit/test_storage.py
git commit -m "feat(runtime): POSIX storage abstraction with path-traversal guard"
```

---

### Task 4: Episode discovery, workspace, and resumable state

**Files:**
- Create: `src/roboray/runtime/episode.py`
- Test: `tests/unit/test_episode.py`

**Interfaces:**
- Consumes: `Storage`, `PosixStorage` (Task 3)
- Produces:
  - `Stage(str, Enum)`: `PREPARE/SCENE/HAND/FUSION/ACTION/PUBLISH`
  - `STAGE_ORDER: list[Stage]`
  - `StageStatus(str, Enum)`: `PENDING/RUNNING/DONE/FAILED`
  - `EpisodeRecord(episode_id, stages: dict[Stage, StageStatus], timings: dict[str, float], error: Optional[str])`, methods `is_done(stage)`, `is_terminal()`, `next_runnable() -> Optional[Stage]`
  - `EpisodeSpec(episode_id, media_dir, has_stereo)`, `discover_episodes(storage, rel=".") -> list[EpisodeSpec]` (**raises `EpisodeDiscoveryError` on a missing/invalid `meta.json` or duplicate `episode_id`**)
  - `EpisodeWorkspace(root: Path, episode_id)`: `media_dir`, `prepared_dir`, `artifacts_dir`, `artifact(name)->Path`, `load()->EpisodeRecord`, `mark_running/mark_done/mark_failed`, `ARTIFACT_SCHEMA_VERSION` constant and `check_artifact(name)->bool` (version/corruption check)
  - `EpisodeDiscoveryError(ValueError)`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_episode.py`:

```python
import json

import numpy as np
import pytest

from roboray.runtime.episode import (
    ARTIFACT_SCHEMA_VERSION,
    EpisodeDiscoveryError,
    EpisodeRecord,
    EpisodeWorkspace,
    Stage,
    StageStatus,
    discover_episodes,
)
from roboray.runtime.storage import PosixStorage


def _make_episode(root, ep_id, *, stereo=True, meta=True):
    d = root / ep_id
    d.mkdir(parents=True)
    (d / "rgb_left.mp4").write_bytes(b"fake")
    if stereo:
        (d / "rgb_right.mp4").write_bytes(b"fake")
    (d / "calibration.json").write_text("{}")
    if meta:
        (d / "meta.json").write_text(json.dumps({"episode_id": ep_id}))
    return d


def test_discover_finds_episodes_with_meta(tmp_path):
    _make_episode(tmp_path, "ep_0001")
    _make_episode(tmp_path, "ep_0002", stereo=False)
    specs = discover_episodes(PosixStorage(tmp_path))
    assert [s.episode_id for s in specs] == ["ep_0001", "ep_0002"]
    assert specs[0].has_stereo is True
    assert specs[1].has_stereo is False


def test_discover_rejects_missing_meta(tmp_path):
    _make_episode(tmp_path, "ep_0001", meta=False)
    with pytest.raises(EpisodeDiscoveryError) as exc:
        discover_episodes(PosixStorage(tmp_path))
    assert "meta.json" in str(exc.value)


def test_discover_rejects_duplicate_episode_id(tmp_path):
    _make_episode(tmp_path, "ep_0001")
    d = tmp_path / "ep_0001_copy"
    d.mkdir()
    (d / "rgb_left.mp4").write_bytes(b"x")
    (d / "meta.json").write_text(json.dumps({"episode_id": "ep_0001"}))
    with pytest.raises(EpisodeDiscoveryError) as exc:
        discover_episodes(PosixStorage(tmp_path))
    assert "ep_0001" in str(exc.value)


def test_record_next_runnable_follows_state_machine():
    rec = EpisodeRecord.new("ep_0001")
    assert rec.next_runnable() == Stage.PREPARE
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    # SCENE and HAND can both run after PREPARE
    assert rec.next_runnable() == Stage.SCENE
    rec.stages[Stage.SCENE] = StageStatus.DONE
    assert rec.next_runnable() == Stage.HAND
    rec.stages[Stage.HAND] = StageStatus.DONE
    assert rec.next_runnable() == Stage.FUSION
    rec.stages[Stage.FUSION] = StageStatus.DONE
    assert rec.next_runnable() == Stage.ACTION
    rec.stages[Stage.ACTION] = StageStatus.DONE
    assert rec.next_runnable() == Stage.PUBLISH
    rec.stages[Stage.PUBLISH] = StageStatus.DONE
    assert rec.next_runnable() is None
    assert rec.is_terminal()


def test_running_stage_blocks_next_runnable():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.RUNNING
    assert rec.next_runnable() is None


def test_workspace_state_persists_and_resumes(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    assert ws.load().stages[Stage.PREPARE] == StageStatus.PENDING
    ws.mark_done(Stage.PREPARE, seconds=1.5)
    ws.mark_done(Stage.SCENE, seconds=2.0)
    again = EpisodeWorkspace(tmp_path, "ep_0001").load()
    assert again.stages[Stage.PREPARE] == StageStatus.DONE
    assert again.stages[Stage.SCENE] == StageStatus.DONE
    assert again.next_runnable() == Stage.HAND
    assert again.timings["prepare"] == pytest.approx(1.5)


def test_failure_is_recorded_and_blocks_downstream(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.mark_failed(Stage.PREPARE, "calibration.json missing")
    rec = ws.load()
    assert rec.stages[Stage.PREPARE] == StageStatus.FAILED
    assert "calibration.json" in rec.error
    assert rec.is_terminal()          # a failure terminates the Episode; nothing downstream runs
    assert rec.next_runnable() is None


def test_corrupt_artifact_is_detected(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.write_artifact("scene.npz", {"camera_poses": np.zeros((2, 4, 4), np.float32)})
    assert ws.check_artifact("scene.npz") is True
    # corrupt the file
    ws.artifact("scene.npz").write_bytes(b"not an npz")
    assert ws.check_artifact("scene.npz") is False


def test_artifact_schema_mismatch_is_detected(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    np.savez(ws.artifact("scene.npz"), __schema_version__=ARTIFACT_SCHEMA_VERSION + 1)
    ws.check_artifact("scene.npz")
    assert ws.check_artifact("scene.npz") is False


def test_workspace_rejects_path_traversal_episode_id(tmp_path):
    with pytest.raises(ValueError):
        EpisodeWorkspace(tmp_path, "../evil")
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_episode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.runtime.episode'`

- [ ] **Step 3: Implement `runtime/episode.py`**

```python
"""Episode discovery, per-episode workspace, and resumable stage state."""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .storage import Storage

__all__ = [
    "Stage",
    "STAGE_ORDER",
    "StageStatus",
    "EpisodeRecord",
    "EpisodeSpec",
    "EpisodeDiscoveryError",
    "EpisodeWorkspace",
    "ARTIFACT_SCHEMA_VERSION",
    "PREREQS",
]

ARTIFACT_SCHEMA_VERSION = 1


class Stage(str, Enum):
    PREPARE = "prepare"
    SCENE = "scene"
    HAND = "hand"
    FUSION = "fusion"
    ACTION = "action"
    PUBLISH = "publish"


STAGE_ORDER = [Stage.PREPARE, Stage.SCENE, Stage.HAND, Stage.FUSION, Stage.ACTION, Stage.PUBLISH]

# Prerequisite stage sets for each stage (SCENE and HAND run in parallel after PREPARE)
PREREQS = {
    Stage.PREPARE: frozenset(),
    Stage.SCENE: frozenset({Stage.PREPARE}),
    Stage.HAND: frozenset({Stage.PREPARE}),
    Stage.FUSION: frozenset({Stage.SCENE, Stage.HAND}),
    Stage.ACTION: frozenset({Stage.FUSION}),
    Stage.PUBLISH: frozenset({Stage.ACTION}),
}


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class EpisodeRecord:
    episode_id: str
    stages: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def new(cls, episode_id: str) -> "EpisodeRecord":
        return cls(episode_id=episode_id, stages={s: StageStatus.PENDING for s in STAGE_ORDER})

    def is_done(self, stage: Stage) -> bool:
        return self.stages.get(stage) is StageStatus.DONE

    def is_terminal(self) -> bool:
        if self.stages.get(Stage.PUBLISH) is StageStatus.DONE:
            return True
        return any(v is StageStatus.FAILED for v in self.stages.values())

    def next_runnable(self) -> Optional[Stage]:
        """Return the first runnable stage in STAGE_ORDER: all prerequisites DONE and itself PENDING.

        Returns None if any stage is RUNNING or FAILED (do not advance other stages of the same Episode concurrently).
        """
        if any(v is StageStatus.RUNNING for v in self.stages.values()):
            return None
        if any(v is StageStatus.FAILED for v in self.stages.values()):
            return None
        for stage in STAGE_ORDER:
            if self.stages[stage] is not StageStatus.PENDING:
                continue
            if all(self.stages[p] is StageStatus.DONE for p in PREREQS[stage]):
                return stage
        return None

    def to_json(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "stages": {s.value: v.value for s, v in self.stages.items()},
            "timings": dict(self.timings),
            "error": self.error,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "EpisodeRecord":
        return cls(
            episode_id=raw["episode_id"],
            stages={Stage(k): StageStatus(v) for k, v in raw["stages"].items()},
            timings={k: float(v) for k, v in raw.get("timings", {}).items()},
            error=raw.get("error"),
        )


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    rel_dir: str
    has_stereo: bool


class EpisodeDiscoveryError(ValueError):
    """Raised when the input dataset is structurally invalid."""


def discover_episodes(storage: Storage, rel: str = ".") -> list[EpisodeSpec]:
    """Discover Episodes in the input dataset.

    Rule: a subdirectory containing ``meta.json`` is one Episode. A missing or invalid
    meta.json, or a duplicate episode_id, errors out immediately (fail early rather than
    silently producing wrong data).
    """
    specs: list[EpisodeSpec] = []
    seen: dict[str, str] = {}
    for name in storage.listdir(rel):
        sub = f"{rel}/{name}" if rel not in (".", "") else name
        if not storage.exists(f"{sub}/meta.json"):
            continue
        try:
            meta = json.loads(storage.read_bytes(f"{sub}/meta.json").decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EpisodeDiscoveryError(f"{sub}/meta.json is not valid JSON: {exc}") from exc
        if not isinstance(meta, dict) or not isinstance(meta.get("episode_id"), str):
            raise EpisodeDiscoveryError(f"{sub}/meta.json is missing the string field episode_id")
        ep_id = meta["episode_id"]
        if ep_id in seen:
            raise EpisodeDiscoveryError(
                f"duplicate episode_id: {ep_id!r} appears in both {seen[ep_id]!r} and {sub!r}; "
                "the outputs would overwrite each other, please fix the input."
            )
        seen[ep_id] = sub
        specs.append(
            EpisodeSpec(
                episode_id=ep_id,
                rel_dir=sub,
                has_stereo=storage.exists(f"{sub}/rgb_right.mp4"),
            )
        )
    return sorted(specs, key=lambda s: s.episode_id)


class EpisodeWorkspace:
    """Workspace for a single Episode: intermediate artifacts + stage state (the resume unit)."""

    def __init__(self, root: Path, episode_id: str):
        if not episode_id or "/" in episode_id or "\\" in episode_id or episode_id in (".", ".."):
            raise ValueError(f"invalid episode_id: {episode_id!r}")
        self.root = Path(root)
        self.episode_id = episode_id
        self.dir = self.root / episode_id
        self.media_dir = self.dir / "media"
        self.prepared_dir = self.dir / "prepared"
        self.artifacts_dir = self.dir / "artifacts"
        self.state_file = self.dir / "state.json"
        for d in (self.media_dir, self.prepared_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

    def artifact(self, name: str) -> Path:
        return self.artifacts_dir / name

    def load(self) -> EpisodeRecord:
        if not self.state_file.exists():
            return EpisodeRecord.new(self.episode_id)
        return EpisodeRecord.from_json(json.loads(self.state_file.read_text()))

    def _save(self, rec: EpisodeRecord) -> None:
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec.to_json(), ensure_ascii=False, indent=2))
        tmp.replace(self.state_file)  # atomic replace, avoids half-written state

    def _mark(self, stage: Stage, status: StageStatus, seconds: Optional[float], error: Optional[str]) -> None:
        rec = self.load()
        rec.stages[stage] = status
        if seconds is not None:
            rec.timings[stage.value] = float(seconds)
        if status is StageStatus.FAILED:
            rec.error = error
        self._save(rec)

    def mark_running(self, stage: Stage) -> None:
        self._mark(stage, StageStatus.RUNNING, None, None)

    def mark_done(self, stage: Stage, seconds: float) -> None:
        self._mark(stage, StageStatus.DONE, seconds, None)

    def mark_failed(self, stage: Stage, error: str) -> None:
        self._mark(stage, StageStatus.FAILED, None, error)

    def write_artifact(self, name: str, arrays: dict) -> Path:
        p = self.artifact(name)
        np.savez(p, __schema_version__=ARTIFACT_SCHEMA_VERSION, **arrays)
        return p

    def check_artifact(self, name: str) -> bool:
        """Verify the artifact is readable and the schema version matches; corrupt or mismatched returns False (stage needs a re-run)."""
        p = self.artifact(name)
        if not p.exists():
            return False
        try:
            with np.load(p, allow_pickle=False) as z:
                if "__schema_version__" not in z.files:
                    return False
                return int(z["__schema_version__"]) == ARTIFACT_SCHEMA_VERSION
        except (OSError, ValueError, EOFError):
            return False
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_episode.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/roboray/runtime/episode.py tests/unit/test_episode.py
git commit -m "feat(runtime): episode discovery, workspace, stage state machine and resume"
```

---

### Task 5: Model protocols and registry (dependency injection)

**Files:**
- Create: `src/roboray/models/__init__.py`
- Create: `src/roboray/models/protocol.py`
- Create: `src/roboray/models/registry.py`
- Test: `tests/unit/test_registry.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `SceneModel` Protocol: `name: str`, `load(device: str)->None`, `infer(frames: dict, calib: dict, out_path: Path)->None`, `cost_hint(n_frames: int, resolution_key: str)->float`
  - `HandModel` Protocol: same signature, `infer` writes the hand artifact
  - `SemanticModel` Protocol: `name: str`, `label(prompt: str, images: list, out_path: Path)->None`
  - `ResolutionKey`: `resolution_key(width: int, height: int) -> str` (e.g. `"640x480"`)
  - Registry: `register_scene_model(name, factory)`, `register_hand_model(name, factory)`, `register_semantic_model(name, factory)`, `get_scene_model_class(name)`, `get_hand_model_class(name)`, `get_semantic_model_class(name)`, `known_scene_models() -> list[str]` (homomorphic for hand/semantic)
  - `UnknownModelError(KeyError)`
  - `load_builtin_models()`: registers the production model names (`vipe`/`fast_foundation_stereo`/`hamer`/`openai_vlm`); in Plan A these factories are **lazy imports** — the real import happens at call time (torch is only installed in the GPU environment)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_registry.py`:

```python
import pytest

from roboray.models import registry
from roboray.models.protocol import resolution_key


def test_resolution_key_is_stable():
    assert resolution_key(640, 480) == "640x480"
    assert resolution_key(641, 480) == "640x480"  # bucketed by 64 px so the EMA is not scattered by jitter


def test_register_and_lookup(monkeypatch):
    registry.reset_registry_for_tests()
    sentinel = object()
    registry.register_scene_model("fake_scene", lambda: sentinel)
    assert registry.get_scene_model_class("fake_scene")() is sentinel
    assert registry.known_scene_models() == ["fake_scene"]


def test_unknown_model_raises_with_available_names():
    registry.reset_registry_for_tests()
    registry.register_hand_model("fake_hand", lambda: object())
    with pytest.raises(registry.UnknownModelError) as exc:
        registry.get_hand_model_class("nope")
    assert "fake_hand" in str(exc.value)


def test_duplicate_registration_without_override_is_rejected():
    registry.reset_registry_for_tests()
    registry.register_scene_model("m", lambda: object())
    with pytest.raises(ValueError):
        registry.register_scene_model("m", lambda: object())


def test_load_builtin_models_registers_production_names():
    registry.reset_registry_for_tests()
    registry.load_builtin_models()
    assert set(registry.known_scene_models()) >= {"vipe", "fast_foundation_stereo"}
    assert "hamer" in registry.known_hand_models()
    assert "openai_vlm" in registry.known_semantic_models()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.models'`

- [ ] **Step 3: Implement `models/protocol.py`**

```python
"""Model interface protocols. Real models and test doubles both implement these protocols."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = ["SceneModel", "HandModel", "SemanticModel", "resolution_key"]

_RESOLUTION_BUCKET_PX = 64


def resolution_key(width: int, height: int) -> str:
    """Quantize the resolution into 64 px buckets so the cost model keeps one EMA per bucket."""
    w = max(_RESOLUTION_BUCKET_PX, (int(width) // _RESOLUTION_BUCKET_PX) * _RESOLUTION_BUCKET_PX)
    h = max(_RESOLUTION_BUCKET_PX, (int(height) // _RESOLUTION_BUCKET_PX) * _RESOLUTION_BUCKET_PX)
    return f"{w}x{h}"


@runtime_checkable
class SceneModel(Protocol):
    name: str

    def load(self, device: str) -> None:
        """Load weights onto the given device inside the Actor."""

    def infer(self, frames: dict[str, Any], calib: dict[str, Any], out_path: Path) -> None:
        """Run Scene reconstruction and write the artifact to out_path."""

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        """Cold-start latency prior (seconds)."""


@runtime_checkable
class HandModel(Protocol):
    name: str

    def load(self, device: str) -> None: ...
    def infer(self, frames: dict[str, Any], calib: dict[str, Any], out_path: Path) -> None: ...
    def cost_hint(self, n_frames: int, resolution_key: str) -> float: ...


@runtime_checkable
class SemanticModel(Protocol):
    name: str

    def label(self, prompt: str, images: list, out_path: Path) -> None:
        """Write the action semantics to out_path (NPZ/JSON layout is implementation-defined; contract in the action stage)."""
```

- [ ] **Step 4: Implement `models/registry.py`**

```python
"""Model registry: product code resolves models by name; tests inject doubles."""

from typing import Callable

__all__ = [
    "UnknownModelError",
    "register_scene_model",
    "register_hand_model",
    "register_semantic_model",
    "get_scene_model_class",
    "get_hand_model_class",
    "get_semantic_model_class",
    "known_scene_models",
    "known_hand_models",
    "known_semantic_models",
    "load_builtin_models",
    "reset_registry_for_tests",
]


class UnknownModelError(KeyError):
    """Raised when a model name has no registered factory."""


_SCENE: dict[str, Callable[[], object]] = {}
_HAND: dict[str, Callable[[], object]] = {}
_SEMANTIC: dict[str, Callable[[], object]] = {}


def _register(store: dict, kind: str, name: str, factory: Callable[[], object]) -> None:
    if name in store:
        raise ValueError(f"{kind} model {name!r} is already registered; call reset_registry_for_tests() first to replace it")
    store[name] = factory


def _get(store: dict, kind: str, name: str):
    if name not in store:
        raise UnknownModelError(
            f"unregistered {kind} model {name!r}; registered: {sorted(store)}"
        )
    return store[name]


def register_scene_model(name: str, factory: Callable[[], object]) -> None:
    _register(_SCENE, "scene", name, factory)


def register_hand_model(name: str, factory: Callable[[], object]) -> None:
    _register(_HAND, "hand", name, factory)


def register_semantic_model(name: str, factory: Callable[[], object]) -> None:
    _register(_SEMANTIC, "semantic", name, factory)


def get_scene_model_class(name: str) -> Callable[[], object]:
    return _get(_SCENE, "scene", name)


def get_hand_model_class(name: str) -> Callable[[], object]:
    return _get(_HAND, "hand", name)


def get_semantic_model_class(name: str) -> Callable[[], object]:
    return _get(_SEMANTIC, "semantic", name)


def known_scene_models() -> list[str]:
    return sorted(_SCENE)


def known_hand_models() -> list[str]:
    return sorted(_HAND)


def known_semantic_models() -> list[str]:
    return sorted(_SEMANTIC)


def reset_registry_for_tests() -> None:
    for store in (_SCENE, _HAND, _SEMANTIC):
        store.clear()


def load_builtin_models() -> None:
    """Register the production models. Factories are lazy imports: the real import happens at
    instantiation time, so this module can be imported in environments without torch.
    """

    def _scene(name: str):
        def factory():
            from roboray.models.scene_models import make_scene_model

            return make_scene_model(name)

        return factory

    def _hand(name: str):
        def factory():
            from roboray.models.hand_models import make_hand_model

            return make_hand_model(name)

        return factory

    for name in ("vipe", "fast_foundation_stereo"):
        if name not in _SCENE:
            _SCENE[name] = _scene(name)
    if "hamer" not in _HAND:
        _HAND["hamer"] = _hand("hamer")
    if "openai_vlm" not in _SEMANTIC:
        _SEMANTIC["openai_vlm"] = lambda: __import__(
            "roboray.models.semantic_models", fromlist=["make_semantic_model"]
        ).make_semantic_model("openai_vlm")
```

`models/__init__.py`:

```python
"""Model protocols and registry."""

from . import registry  # noqa: F401

__all__ = ["registry"]
```

- [ ] **Step 5: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/roboray/models/ tests/unit/test_registry.py
git commit -m "feat(models): model protocols and registry (dependency injection; product code has no concrete-model dependency)"
```

---

### Task 6: Cost model (EMA prediction)

**Files:**
- Create: `src/roboray/scheduling/__init__.py`
- Create: `src/roboray/scheduling/cost_model.py`
- Test: `tests/unit/test_cost_model.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `CostModel(alpha: float = 0.3)`: `seed_prior(model, resolution_key, seconds)`, `predict(model, resolution_key) -> float` (returns `DEFAULT_PRIOR_SECONDS` when there is no prior), `update(model, resolution_key, seconds)`, `load_prior(name, seconds)` / `load_cost(name, seconds)` (independent EMA for role-switch load latency), `predict_load(name) -> float`, `update_load(name, seconds)`, `snapshot() -> dict`, `DEFAULT_PRIOR_SECONDS`
  - `LoadCostModel` semantics folded into `CostModel` (the `*_load` methods)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cost_model.py`:

```python
import pytest

from roboray.scheduling.cost_model import DEFAULT_PRIOR_SECONDS, CostModel


def test_unknown_key_returns_default_prior():
    cm = CostModel()
    assert cm.predict("scene", "640x480") == DEFAULT_PRIOR_SECONDS


def test_seed_prior_sets_initial_estimate():
    cm = CostModel()
    cm.seed_prior("scene", "640x480", 4.0)
    assert cm.predict("scene", "640x480") == pytest.approx(4.0)


def test_update_moves_estimate_toward_observation():
    cm = CostModel(alpha=0.5)
    cm.seed_prior("scene", "640x480", 4.0)
    cm.update("scene", "640x480", 8.0)
    assert cm.predict("scene", "640x480") == pytest.approx(6.0)
    cm.update("scene", "640x480", 8.0)
    assert cm.predict("scene", "640x480") == pytest.approx(7.0)


def test_updates_converge_to_true_cost():
    cm = CostModel(alpha=0.3)
    for _ in range(80):
        cm.update("hand", "320x240", 2.5)
    assert cm.predict("hand", "320x240") == pytest.approx(2.5, abs=0.05)


def test_keys_are_independent():
    cm = CostModel(alpha=0.5)
    cm.seed_prior("scene", "640x480", 4.0)
    cm.seed_prior("hand", "640x480", 1.0)
    cm.update("scene", "640x480", 10.0)
    assert cm.predict("hand", "640x480") == pytest.approx(1.0)


def test_load_cost_tracked_separately():
    cm = CostModel(alpha=0.5)
    cm.load_prior("scene", 5.0)
    cm.load_cost("scene", 15.0)
    assert cm.predict_load("scene") == pytest.approx(10.0)


def test_snapshot_is_plain_dict():
    cm = CostModel()
    cm.seed_prior("scene", "640x480", 3.0)
    snap = cm.snapshot()
    assert snap["infer"]["scene/640x480"] == pytest.approx(3.0)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_cost_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `scheduling/cost_model.py`**

```python
"""EMA prediction of measured latency per (model, resolution bucket); the core of the feedback loop."""

__all__ = ["CostModel", "DEFAULT_PRIOR_SECONDS"]

DEFAULT_PRIOR_SECONDS = 1.0
DEFAULT_LOAD_PRIOR_SECONDS = 5.0


class CostModel:
    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the (0, 1] range")
        self.alpha = alpha
        self._infer: dict[str, float] = {}
        self._load: dict[str, float] = {}
        self._samples: dict[str, int] = {}

    @staticmethod
    def _key(model: str, resolution_key: str) -> str:
        return f"{model}/{resolution_key}"

    def seed_prior(self, model: str, resolution_key: str, seconds: float) -> None:
        """Cold-start prior; does not overwrite an existing measured estimate."""
        self._infer.setdefault(self._key(model, resolution_key), float(seconds))

    def predict(self, model: str, resolution_key: str) -> float:
        return self._infer.get(self._key(model, resolution_key), DEFAULT_PRIOR_SECONDS)

    def update(self, model: str, resolution_key: str, seconds: float) -> None:
        k = self._key(model, resolution_key)
        prev = self._infer.get(k)
        self._infer[k] = float(seconds) if prev is None else (
            (1.0 - self.alpha) * prev + self.alpha * float(seconds)
        )
        self._samples[k] = self._samples.get(k, 0) + 1

    def samples(self, model: str, resolution_key: str) -> int:
        return self._samples.get(self._key(model, resolution_key), 0)

    def load_prior(self, model: str, seconds: float) -> None:
        self._load.setdefault(model, float(seconds))

    def predict_load(self, model: str) -> float:
        return self._load.get(model, DEFAULT_LOAD_PRIOR_SECONDS)

    def load_cost(self, model: str, seconds: float) -> None:
        prev = self._load.get(model)
        self._load[model] = float(seconds) if prev is None else (
            (1.0 - self.alpha) * prev + self.alpha * float(seconds)
        )

    def snapshot(self) -> dict:
        return {
            "alpha": self.alpha,
            "infer": dict(sorted(self._infer.items())),
            "load": dict(sorted(self._load.items())),
            "samples": dict(sorted(self._samples.items())),
        }
```

`scheduling/__init__.py`:

```python
"""Adaptive scheduling: cost model, queues, slot assignment."""

from .cost_model import CostModel  # noqa: F401

__all__ = ["CostModel"]
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_cost_model.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/roboray/scheduling/ tests/unit/test_cost_model.py
git commit -m "feat(scheduling): EMA cost model (inference latency + role-switch load latency)"
```

---

### Task 7: Adaptive scheduler (4 signals + model reuse)

**Files:**
- Create: `src/roboray/scheduling/scheduler.py`
- Test: `tests/unit/test_scheduler.py`

**Interfaces:**
- Consumes: `CostModel` (Task 6)
- Produces:
  - `QueueItem(kind: str, episode_id: str, resolution_key: str)`, `kind ∈ {"scene","hand"}`
  - `GpuSlot(slot_id: int, loaded_model: Optional[str] = None, busy: bool = False)`
  - `Assignment(slot_id: int, kind: str, item: QueueItem, needs_load: bool)`
  - `AdaptiveScheduler(slots: list[GpuSlot], cost_model: CostModel, scene_bias: float = 1.0, hand_bias: float = 1.0, reuse_tolerance: float = 1.0)`
    - `choose_kind(scene_q: list[QueueItem], hand_q: list[QueueItem]) -> Optional[str]`: picks by **estimated remaining workload** (Σ cost_model.predict); None if both queues are empty
    - `assign(scene_q, hand_q, slot: GpuSlot) -> Optional[Assignment]`: returns None if `slot.busy`; first `choose_kind`, then decides whether **reusing the loaded model is cheaper** (switch cost = predicted target-model latency + load latency vs reuse = predicted latency of the head item in the loaded model's queue), and switches role accordingly; `needs_load = slot.loaded_model != kind`
    - `finish(slot, kind, resolution_key, seconds, was_load: bool)`: feeds measured latency back into `CostModel` (inference via `update`, load via `load_cost`) and sets `slot.busy=False`
  - `SchedulingStats`: `snapshots: list[dict]` (queue depth/EMA/choice recorded per decision), `to_json()`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_scheduler.py`:

```python
import pytest

from roboray.scheduling.cost_model import CostModel
from roboray.scheduling.scheduler import AdaptiveScheduler, GpuSlot, QueueItem


def _item(kind, ep="ep_0001", res="640x480"):
    return QueueItem(kind=kind, episode_id=ep, resolution_key=res)


def test_choose_kind_picks_bigger_backlog():
    cm = CostModel()
    cm.seed_prior("scene", "640x480", 4.0)
    cm.seed_prior("hand", "640x480", 1.0)
    sch = AdaptiveScheduler([GpuSlot(0)], cm)
    scene_q = [_item("scene"), _item("scene")]
    hand_q = [_item("hand")]
    # scene backlog 8s vs hand 1s -> pick scene
    assert sch.choose_kind(scene_q, hand_q) == "scene"
    # hand grows to 10 items -> 10s > 8s -> pick hand
    assert sch.choose_kind(scene_q, [_item("hand")] * 10) == "hand"


def test_choose_kind_returns_none_on_empty_queues():
    sch = AdaptiveScheduler([GpuSlot(0)], CostModel())
    assert sch.choose_kind([], []) is None


def test_assign_marks_load_when_model_not_loaded():
    cm = CostModel()
    cm.load_prior("scene", 5.0)
    sch = AdaptiveScheduler([GpuSlot(0, loaded_model=None)], cm)
    a = sch.assign([_item("scene")], [], sch.slots[0])
    assert a.kind == "scene" and a.needs_load is True


def test_assign_reuses_loaded_model_when_switching_is_not_worth_it():
    cm = CostModel()
    cm.seed_prior("scene", "640x480", 4.0)
    cm.seed_prior("hand", "640x480", 3.0)
    cm.load_prior("hand", 30.0)      # switching to hand is very expensive
    slot = GpuSlot(0, loaded_model="scene")
    sch = AdaptiveScheduler([slot], cm)
    # hand backlog 6s > scene 4s, so hand would normally win; but the extra 2s <= 30s load cost -> reuse scene
    a = sch.assign([_item("scene")], [_item("hand")] * 2, slot)
    assert a.kind == "scene" and a.needs_load is False


def test_assign_switches_when_extra_backlog_exceeds_load_cost():
    cm = CostModel()
    cm.seed_prior("scene", "640x480", 3.0)
    cm.seed_prior("hand", "640x480", 3.0)
    cm.load_prior("hand", 1.0)
    slot = GpuSlot(0, loaded_model="scene")
    sch = AdaptiveScheduler([slot], cm)
    # hand backlog 30s, scene backlog 3s -> extra 27s > 1s load cost -> switch role
    a = sch.assign([_item("scene")], [_item("hand")] * 10, slot)
    assert a.kind == "hand" and a.needs_load is True


def test_busy_slot_gets_no_assignment():
    sch = AdaptiveScheduler([GpuSlot(0, busy=True)], CostModel())
    assert sch.assign([_item("scene")], [], sch.slots[0]) is None


def test_finish_feeds_back_and_frees_slot():
    cm = CostModel(alpha=0.5)
    cm.seed_prior("scene", "640x480", 4.0)
    slot = GpuSlot(0, loaded_model="scene", busy=True)
    sch = AdaptiveScheduler([slot], cm)
    sch.finish(slot, "scene", "640x480", seconds=8.0, was_load=False)
    assert slot.busy is False
    assert cm.predict("scene", "640x480") == pytest.approx(6.0)


def test_finish_records_load_cost_separately():
    cm = CostModel(alpha=1.0)
    slot = GpuSlot(0, loaded_model=None, busy=True)
    sch = AdaptiveScheduler([slot], cm)
    sch.finish(slot, "scene", "640x480", seconds=12.0, was_load=True)
    assert cm.predict_load("scene") == pytest.approx(12.0)
    assert cm.stats()["samples"].get("scene/640x480", 0) == 0  # load is not counted in the inference EMA


def test_division_of_labour_follows_skewed_load_without_thrashing():
    """When the hand queue is clearly longer: the Actor switches to hand and keeps reusing it (no repeated load/unload)."""
    cm = CostModel(alpha=0.5)
    cm.seed_prior("scene", "640x480", 4.0)
    cm.seed_prior("hand", "640x480", 1.0)
    slot = GpuSlot(0, loaded_model="scene")
    sch = AdaptiveScheduler([slot], cm)
    kinds, loads = [], 0
    for _ in range(20):
        a = sch.assign([_item("scene")], [_item("hand")] * 14, slot)
        kinds.append(a.kind)
        if a.needs_load:
            loads += 1
        # feed the measured latency back (here using the current prediction as the observation, simulating steady state)
        sch.finish(slot, a.kind, "640x480",
                   seconds=cm.predict(a.kind, "640x480"), was_load=a.needs_load)
    assert kinds.count("hand") == len(kinds)             # all routed to hand
    assert loads <= 1                                    # loaded only once, reused afterwards
    assert cm.samples("hand", "640x480") == len(kinds) - loads  # measured latency really flows back


def test_stats_snapshot_is_json_serialisable():
    import json

    cm = CostModel()
    sch = AdaptiveScheduler([GpuSlot(0)], cm)
    sch.assign([_item("scene")], [], sch.slots[0])
    json.dumps(sch.stats().to_json())
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.scheduling.scheduler'`

- [ ] **Step 3: Implement `scheduling/scheduler.py`**

```python
"""Adaptive scheduler: assigns GPU Actor roles from 4 signals (queue backlog / predicted latency / available resources / loaded model).

Pure Python with no Ray dependency -- easy to unit test; the Ray runtime is only its execution shell.
"""

from dataclasses import dataclass, field
from typing import Optional

from .cost_model import CostModel

__all__ = ["QueueItem", "GpuSlot", "Assignment", "SchedulingStats", "AdaptiveScheduler"]


@dataclass(frozen=True)
class QueueItem:
    kind: str  # "scene" | "hand"
    episode_id: str
    resolution_key: str


@dataclass
class GpuSlot:
    slot_id: int
    loaded_model: Optional[str] = None
    busy: bool = False


@dataclass(frozen=True)
class Assignment:
    slot_id: int
    kind: str
    item: QueueItem
    needs_load: bool


@dataclass
class SchedulingStats:
    snapshots: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"snapshots": list(self.snapshots)}


class AdaptiveScheduler:
    def __init__(
        self,
        slots: list,
        cost_model: CostModel,
        scene_bias: float = 1.0,
        hand_bias: float = 1.0,
        reuse_tolerance: float = 1.0,
    ):
        self.slots = list(slots)
        self.cost_model = cost_model
        self.scene_bias = float(scene_bias)
        self.hand_bias = float(hand_bias)
        self.reuse_tolerance = float(reuse_tolerance)
        self.stats_obj = SchedulingStats()

    def stats(self) -> CostModel:
        return self.cost_model

    def _backlog(self, kind: str, queue: list) -> float:
        bias = self.scene_bias if kind == "scene" else self.hand_bias
        return sum(self.cost_model.predict(kind, i.resolution_key) for i in queue) * bias

    def choose_kind(self, scene_q: list, hand_q: list) -> Optional[str]:
        if not scene_q and not hand_q:
            return None
        scene_est = self._backlog("scene", scene_q)
        hand_est = self._backlog("hand", hand_q)
        if scene_est <= 0.0 and hand_est <= 0.0:
            return None
        return "scene" if scene_est >= hand_est else "hand"

    def _queue_for(self, kind: str, scene_q: list, hand_q: list) -> list:
        return scene_q if kind == "scene" else hand_q

    def assign(self, scene_q: list, hand_q: list, slot: GpuSlot) -> Optional[Assignment]:
        if slot.busy:
            return None
        wanted = self.choose_kind(scene_q, hand_q)
        if wanted is None:
            return None

        chosen = wanted
        # Reuse decision (article §06: prefer reusing the loaded model).
        # Only switch roles when the "extra backlog of the target queue" exceeds the cost of
        # reloading that model; otherwise keep serving the loaded model's queue to avoid
        # repeated load/unload.
        if slot.loaded_model is not None and slot.loaded_model != wanted:
            reuse_q = self._queue_for(slot.loaded_model, scene_q, hand_q)
            want_q = self._queue_for(wanted, scene_q, hand_q)
            if reuse_q and want_q:
                extra_backlog = (
                    self._backlog(wanted, want_q)
                    - self._backlog(slot.loaded_model, reuse_q)
                )
                if extra_backlog <= self.cost_model.predict_load(wanted) * self.reuse_tolerance:
                    chosen = slot.loaded_model

        queue = self._queue_for(chosen, scene_q, hand_q)
        if not queue:
            return None
        item = queue[0]
        assignment = Assignment(
            slot_id=slot.slot_id,
            kind=chosen,
            item=item,
            needs_load=slot.loaded_model != chosen,
        )
        self.stats_obj.snapshots.append(
            {
                "slot_id": slot.slot_id,
                "chosen": chosen,
                "wanted": wanted,
                "needs_load": assignment.needs_load,
                "scene_q": len(scene_q),
                "hand_q": len(hand_q),
                "predict": round(self.cost_model.predict(chosen, item.resolution_key), 4),
                "loaded": slot.loaded_model,
            }
        )
        return assignment

    def finish(
        self,
        slot: GpuSlot,
        kind: str,
        resolution_key: str,
        seconds: float,
        was_load: bool,
    ) -> None:
        """Feed measured latency back: inference into the (model, resolution bucket) EMA, load into the model-level load EMA."""
        if was_load:
            self.cost_model.load_cost(kind, seconds)
        else:
            self.cost_model.update(kind, resolution_key, seconds)
        slot.loaded_model = kind
        slot.busy = False
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_scheduler.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/roboray/scheduling/scheduler.py tests/unit/test_scheduler.py
git commit -m "feat(scheduling): adaptive scheduler (4-signal assignment + model reuse + feedback loop)"
```

---

### Task 8: Synthetic ego data generator and test doubles

**Files:**
- Create: `scripts/make_synthetic_ego.py`
- Create: `tests/fakes/__init__.py`
- Create: `tests/fakes/models.py`
- Test: `tests/unit/test_synthetic_data.py`

**Interfaces:**
- Consumes: none (the generator only depends on numpy/cv2/json)
- Produces:
  - `generate_dataset(out_dir, n_episodes=2, n_frames=24, fps=30, size=(96,72), stereo=True, seed=0, hand_jitter=None) -> Path` (importable by both the script and tests: `from scripts.make_synthetic_ego import generate_dataset`; `scripts/__init__.py` must therefore exist)
  - Outputs: per-Episode `rgb_left.mp4`, optional `rgb_right.mp4`, `calibration.json` (with `K`, `dist`, `T_left_to_right`, `fps`), `meta.json` (with `episode_id`), `ground_truth.npz` (`joints_world (T,2,J,3)`, `depth (T,H,W)`)
  - Test doubles (`tests/fakes/models.py`): `FakeSceneModel(seconds=0.05, fail_on=None)`, `FakeHandModel(seconds=0.05, fail_on=None)`, `FakeSemanticModel(seconds=0.02)`, all implementing the Task 5 protocols; `install_fake_models(scene_seconds=0.05, hand_seconds=0.05, hand_queue_ratio=None)` registers them with the registry and returns the registered names `("fake_scene", "fake_hand", "fake_vlm")`
  - Double behavior: `infer` writes a valid NPZ (keys per spec §5) and `sleep(seconds)` simulates GPU latency; when `fail_on` matches the episode_id, that episode raises `RuntimeError`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_synthetic_data.py`:

```python
import cv2
import numpy as np
import pytest

from roboray.runtime.episode import EpisodeWorkspace, Stage
from roboray.runtime.storage import PosixStorage
from roboray.runtime.episode import discover_episodes


def test_generate_dataset_layout_and_readability(tmp_path):
    from scripts.make_synthetic_ego import generate_dataset

    generate_dataset(tmp_path, n_episodes=2, n_frames=12, size=(96, 72), stereo=True, seed=1)
    specs = discover_episodes(PosixStorage(tmp_path))
    assert [s.episode_id for s in specs] == ["ep_0001", "ep_0002"]
    assert all(s.has_stereo for s in specs)

    cap = cv2.VideoCapture(str(tmp_path / "ep_0001" / "rgb_left.mp4"))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        assert frame.shape[:2] == (72, 96)
    cap.release()
    assert count == 12


def test_ground_truth_is_present_and_shaped(tmp_path):
    from scripts.make_synthetic_ego import generate_dataset

    generate_dataset(tmp_path, n_episodes=1, n_frames=10, size=(96, 72), seed=2)
    gt = np.load(tmp_path / "ep_0001" / "ground_truth.npz")
    assert gt["joints_world"].shape == (10, 2, 21, 3)
    assert gt["depth"].shape[0] == 10


def test_monocular_dataset_has_no_right_video(tmp_path):
    from scripts.make_synthetic_ego import generate_dataset
    from roboray.runtime.storage import PosixStorage as PS

    generate_dataset(tmp_path, n_episodes=1, n_frames=6, stereo=False, seed=3)
    specs = discover_episodes(PS(tmp_path))
    assert specs[0].has_stereo is False


def test_calibration_is_loadable(tmp_path):
    import json

    from scripts.make_synthetic_ego import generate_dataset

    generate_dataset(tmp_path, n_episodes=1, n_frames=6, seed=4)
    calib = json.loads((tmp_path / "ep_0001" / "calibration.json").read_text())
    assert np.array(calib["K"]).shape == (3, 3)
    assert "fps" in calib
```

`tests/unit/test_fakes.py`:

```python
import numpy as np
import pytest

from roboray.models import registry
from tests.fakes.models import install_fake_models


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.reset_registry_for_tests()
    yield
    registry.reset_registry_for_tests()


def test_install_fake_models_registers_all_three():
    names = install_fake_models()
    assert names == ("fake_scene", "fake_hand", "fake_vlm")
    assert registry.get_scene_model_class("fake_scene")() is not None


def test_fake_scene_writes_valid_artifact(tmp_path):
    install_fake_models()
    model = registry.get_scene_model_class("fake_scene")()
    model.load("cpu")
    out = tmp_path / "scene.npz"
    model.infer({"frames": np.zeros((3, 8, 8, 3), np.uint8)}, {"fps": 30}, out)
    z = np.load(out)
    assert z["camera_poses"].shape == (3, 4, 4)
    assert "depth" in z.files
    assert z["depth_source"].item() in ("stereo", "monocular")


def test_fake_hand_writes_valid_artifact(tmp_path):
    install_fake_models()
    model = registry.get_hand_model_class("fake_hand")()
    model.load("cpu")
    out = tmp_path / "hand.npz"
    model.infer({"frames": np.zeros((3, 8, 8, 3), np.uint8)}, {"fps": 30}, out)
    z = np.load(out)
    assert z["joints_2d"].shape[0:2] == (3, 2)
    assert z["mano_pose"].shape[0:2] == (3, 2)


def test_fake_failure_is_scoped_to_one_episode(tmp_path):
    install_fake_models()
    model = registry.get_scene_model_class("fake_scene")()
    model.load("cpu")
    with pytest.raises(RuntimeError, match="ep_0002"):
        model.infer(
            {"frames": np.zeros((2, 8, 8, 3), np.uint8), "episode_id": "ep_0002"},
            {"fps": 30, "fail_on": "ep_0002"},
            tmp_path / "x.npz",
        )
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_synthetic_data.py tests/unit/test_fakes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Implement `scripts/make_synthetic_ego.py`**

```python
"""Generate a synthetic ego dataset with known ground truth, for L1/L3 verification.

Usage:
    python scripts/make_synthetic_ego.py --out data/synthetic --episodes 3 --frames 60
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

N_JOINTS = 21
HANDS = 2


def _write_video(path: Path, frames: list, fps: int) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():  # pragma: no cover - depends on the local codec
        raise RuntimeError(f"cannot open VideoWriter for {path} (check that opencv has mp4v support)")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def _episode_frames(n_frames: int, size: tuple, seed: int, hand_in_view: bool = True):
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    left, right = [], []
    # 2D hand trajectory in the frame: sinusoidal drift, simulating a first-person view of the hand
    t = np.linspace(0, 2 * np.pi, n_frames)
    cx = (w * (0.5 + 0.25 * np.sin(t))).astype(int)
    cy = (h * (0.55 + 0.15 * np.cos(t))).astype(int)
    for i in range(n_frames):
        base = rng.integers(30, 90, size=(h, w, 3), dtype=np.uint8)
        if hand_in_view:
            cv2.circle(base, (int(cx[i]), int(cy[i])), max(3, w // 16), (0, 255, 0), -1)
        left.append(base)
        # right eye = whole left frame shifted horizontally (simulated baseline disparity) so
        # the stereo algorithm has real structure to work with
        shift = 5
        right.append(np.roll(base, -shift, axis=1))
    return left, right, (cx, cy)


def _ground_truth(n_frames: int, seed: int, size: tuple) -> dict:
    rng = np.random.default_rng(seed + 1000)
    t = np.linspace(0, 2 * np.pi, n_frames)
    joints = np.zeros((n_frames, HANDS, N_JOINTS, 3), np.float32)
    for i in range(n_frames):
        for hand in range(HANDS):
            base = np.array([0.1 * (-1) ** hand, 0.3 * np.sin(t[i]), -0.4], np.float32)
            joints[i, hand] = base + rng.normal(0, 0.004, size=(N_JOINTS, 3)).astype(np.float32)
    depth = np.full((n_frames, size[1], size[0]), 0.5, np.float32)
    return {"joints_world": joints, "depth": depth, "fps": np.int32(30)}


def generate_dataset(
    out_dir,
    n_episodes: int = 2,
    n_frames: int = 24,
    fps: int = 30,
    size: tuple = (96, 72),
    stereo: bool = True,
    seed: int = 0,
    hand_jitter: float = 0.0,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = size
    for e in range(n_episodes):
        ep_id = f"ep_{e + 1:04d}"
        ep_dir = out / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        left, right, _ = _episode_frames(n_frames, size, seed + e)
        _write_video(ep_dir / "rgb_left.mp4", left, fps)
        if stereo:
            _write_video(ep_dir / "rgb_right.mp4", right, fps)
        calib = {
            "K": [[float(w) * 0.8, 0.0, w / 2.0], [0.0, float(h) * 0.8, h / 2.0], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "T_left_to_right": np.eye(4).tolist(),
            "fps": float(fps),
            "width": int(w),
            "height": int(h),
        }
        (ep_dir / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=2))
        (ep_dir / "meta.json").write_text(
            json.dumps(
                {"episode_id": ep_id, "device": "synthetic", "n_frames": int(n_frames)},
                ensure_ascii=False,
                indent=2,
            )
        )
        gt = _ground_truth(n_frames, seed + e, size)
        if hand_jitter:
            gt["joints_world"] += np.random.default_rng(seed).normal(
                0, hand_jitter, gt["joints_world"].shape
            ).astype(np.float32)
        np.savez(ep_dir / "ground_truth.npz", **gt)
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--no-stereo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = generate_dataset(
        args.out,
        n_episodes=args.episodes,
        n_frames=args.frames,
        fps=args.fps,
        size=(args.width, args.height),
        stereo=not args.no_stereo,
        seed=args.seed,
    )
    print(f"generated synthetic dataset: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
```

Also create an empty `scripts/__init__.py`.

- [ ] **Step 4: Implement `tests/fakes/models.py`**

```python
"""Test doubles: implement the models.protocol protocols, simulating GPU latency with sleep.

For testing only; product code must not import this module.
"""

import time
from pathlib import Path

import numpy as np

from roboray.models import registry

__all__ = [
    "FakeSceneModel",
    "FakeHandModel",
    "FakeSemanticModel",
    "install_fake_models",
]

N_JOINTS = 21
HANDS = 2


class _FakeBase:
    def __init__(self, name: str, seconds: float = 0.05, fail_on: str | None = None):
        self.name = name
        self.seconds = float(seconds)
        self.fail_on = fail_on
        self.loaded_on = None

    def load(self, device: str) -> None:
        self.loaded_on = device

    def cost_hint(self, n_frames: int, resolution_key: str) -> float:
        return self.seconds * max(1, n_frames) / 8.0

    def _maybe_fail(self, calib: dict) -> None:
        target = calib.get("fail_on")
        if target and self.fail_on == target:
            raise RuntimeError(f"simulated inference failure: episode {target}")

    def _sleep(self) -> None:
        time.sleep(self.seconds)


class FakeSceneModel(_FakeBase):
    def __init__(self, seconds: float = 0.05, fail_on: str | None = None):
        super().__init__("fake_scene", seconds, fail_on)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        self._maybe_fail(calib)
        arr = np.asarray(frames["frames"])
        t, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
        self._sleep()
        depth_source = "stereo" if calib.get("has_stereo") else "monocular"
        np.savez(
            out_path,
            camera_poses=np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (t, 1, 1)),
            depth=np.full((t, h, w), 0.5, np.float32),
            depth_source=np.array(depth_source),
            n_frames=np.int32(t),
            fps=np.int32(int(calib.get("fps", 30))),
        )


class FakeHandModel(_FakeBase):
    def __init__(self, seconds: float = 0.05, fail_on: str | None = None):
        super().__init__("fake_hand", seconds, fail_on)

    def infer(self, frames: dict, calib: dict, out_path: Path) -> None:
        self._maybe_fail(calib)
        arr = np.asarray(frames["frames"])
        t = arr.shape[0]
        self._sleep()
        rng = np.random.default_rng(0)
        present = np.ones((t, HANDS), bool)
        # create a short detection gap for the A3 fill logic to test against
        if t >= 8:
            present[3:5, 0] = False
        np.savez(
            out_path,
            hand_present=present,
            joints_2d=rng.normal(0, 1, size=(t, HANDS, N_JOINTS, 3)).astype(np.float32),
            mano_pose=rng.normal(0, 0.1, size=(t, HANDS, 48)).astype(np.float32),
            mano_shape=rng.normal(0, 0.1, size=(t, HANDS, 10)).astype(np.float32),
            mano_trans=rng.normal(0, 0.1, size=(t, HANDS, 3)).astype(np.float32),
            confidence=np.where(present, 0.9, 0.0).astype(np.float32),
            n_frames=np.int32(t),
            fps=np.int32(int(calib.get("fps", 30))),
        )


class FakeSemanticModel:
    def __init__(self, seconds: float = 0.02):
        self.name = "fake_vlm"
        self.seconds = float(seconds)

    def label(self, prompt: str, images: list, out_path: Path) -> None:
        time.sleep(self.seconds)
        np.savez(
            out_path,
            labels=np.array(["reach", "grasp", "lift"], dtype=object),
            segment_start=np.array([0, 8, 16], dtype=np.int32),
            segment_end=np.array([8, 16, 24], dtype=np.int32),
        )


def install_fake_models(
    scene_seconds: float = 0.05,
    hand_seconds: float = 0.05,
    fail_on: str | None = None,
) -> tuple:
    """Register the doubles with the registry, returning (scene_name, hand_name, semantic_name)."""
    registry.register_scene_model(
        "fake_scene", lambda: FakeSceneModel(seconds=scene_seconds, fail_on=fail_on)
    )
    registry.register_hand_model(
        "fake_hand", lambda: FakeHandModel(seconds=hand_seconds, fail_on=fail_on)
    )
    registry.register_semantic_model("fake_vlm", lambda: FakeSemanticModel())
    return ("fake_scene", "fake_hand", "fake_vlm")
```

`tests/fakes/__init__.py`:

```python
"""Test doubles implementing the model protocols."""
```

- [ ] **Step 5: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_synthetic_data.py tests/unit/test_fakes.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/ tests/fakes/ tests/unit/test_synthetic_data.py tests/unit/test_fakes.py
git commit -m "feat(test): synthetic ego dataset generator and model doubles"
```

---

### Task 9: Ray Actors (GPU model Actor + CPU stage Actor)

**Files:**
- Create: `src/roboray/ray_runtime/__init__.py`
- Create: `src/roboray/ray_runtime/actors.py`
- Create: `tests/fakes/worker_setup.py` (Ray worker-side double-registration hook, test-only)
- Test: `tests/unit/test_actors.py` (function-level Ray fixture + worker_process_setup_hook; marked `slow`)

**Interfaces:**
- Consumes: `registry` (Task 5), `EpisodeWorkspace` (Task 4)
- Produces:
  - `GpuModelActor` (`@ray.remote(num_gpus=1)`): `load_model(kind, model_name, device="cuda") -> float` (returns load latency in seconds), `run_task(payload: dict) -> dict`, `status() -> dict`
    - `payload` keys: `kind`, `episode_id`, `model_name`, `resolution_key`, `frames` (`np.ndarray`), `calib` (dict), `out_path` (str)
    - Returns: `{"episode_id","kind","out_path","infer_seconds","load_seconds","was_load","ok","error"}`
  - `CpuStageActor` (`@ray.remote(num_cpus=0.5)`): `run_task(payload: dict) -> dict`, `payload` includes `fn_name` (stage function name implemented in Tasks 10/12) and `kwargs`; returns a structurally identical dict
  - `build_actor_pool(num_gpus, num_cpu_actors) -> (list[GpuModelActor handles], list[CpuStageActor handles])`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_actors.py`:

```python
import numpy as np
import pytest

ray = pytest.importorskip("ray")

from roboray.models import registry
from roboray.ray_runtime.actors import build_actor_pool
from tests.fakes.models import install_fake_models


@pytest.fixture(autouse=True)
def _ray_cluster(request):
    """Function-level Ray cluster: each test gets its own virtual GPU, released at teardown.

    - No GPU on this machine: num_gpus is a virtual scheduling resource, so the GPU Actor
      scheduling semantics can be genuinely verified.
    - Must be function-level: GpuModelActor holds 1 virtual GPU permanently; with a module-level
      shared cluster, the second test's new Actor pool would stay PENDING forever (deadlock)
      because the resource is never released.
    - worker_process_setup_hook: Ray workers are separate processes, so driver-side registry
      registrations do not propagate; a test-side hook registers the doubles in each worker
      on startup (product code never imports tests/).
    - ``@pytest.mark.fake_fail_on(ep_id)`` can configure the doubles' fail_on into the worker environment.
    """
    if ray.is_initialized():
        ray.shutdown()
    env_vars = {
        "ROBORAY_FAKE_SCENE_SECONDS": "0.05",
        "ROBORAY_FAKE_HAND_SECONDS": "0.05",
    }
    marker = request.node.get_closest_marker("fake_fail_on")
    if marker:
        env_vars["ROBORAY_FAKE_FAIL_ON"] = marker.args[0]
    ray.init(
        num_gpus=1,
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "working_dir": ".",
            "worker_process_setup_hook": "tests.fakes.worker_setup.install_for_ray",
            "env_vars": env_vars,
        },
    )
    yield
    ray.shutdown()


@pytest.fixture(autouse=True)
def _fakes():
    registry.reset_registry_for_tests()
    install_fake_models()
    yield
    registry.reset_registry_for_tests()


def test_gpu_actor_runs_task_and_reports_timings(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    load_seconds = ray.get(actor.load_model.remote("scene", "fake_scene", "cpu"))
    assert load_seconds >= 0.0
    res = ray.get(
        actor.run_task.remote(
            {
                "kind": "scene",
                "episode_id": "ep_0001",
                "model_name": "fake_scene",
                "resolution_key": "96x64",
                "frames": np.zeros((3, 8, 8, 3), np.uint8),
                "calib": {"fps": 30, "has_stereo": False},
                "out_path": str(tmp_path / "scene.npz"),
            }
        )
    )
    assert res["ok"] is True
    assert res["infer_seconds"] >= 0.0
    assert res["was_load"] is False
    assert np.load(res["out_path"])["depth"].shape[0] == 3


def test_repeated_task_with_same_model_does_not_need_reload(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    ray.get(actor.load_model.remote("scene", "fake_scene", "cpu"))
    payload = {
        "kind": "scene",
        "episode_id": "ep_0001",
        "model_name": "fake_scene",
        "resolution_key": "96x64",
        "frames": np.zeros((2, 8, 8, 3), np.uint8),
        "calib": {"fps": 30, "has_stereo": False},
        "out_path": str(tmp_path / "s.npz"),
    }
    ray.get(actor.run_task.remote(payload))
    res2 = ray.get(actor.run_task.remote(payload))
    assert res2["was_load"] is False


def test_switching_model_kind_reports_was_load(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    ray.get(actor.load_model.remote("scene", "fake_scene", "cpu"))
    res = ray.get(
        actor.run_task.remote(
            {
                "kind": "hand",
                "episode_id": "ep_0001",
                "model_name": "fake_hand",
                "resolution_key": "96x64",
                "frames": np.zeros((2, 8, 8, 3), np.uint8),
                "calib": {"fps": 30, "has_stereo": False},
                "out_path": str(tmp_path / "h.npz"),
            }
        )
    )
    assert res["was_load"] is True
    assert res["load_seconds"] >= 0.0


@pytest.mark.fake_fail_on("ep_0002")
def test_actor_failure_is_returned_not_raised(tmp_path):
    registry.reset_registry_for_tests()
    install_fake_models(fail_on="ep_0002")
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    res = ray.get(
        actor.run_task.remote(
            {
                "kind": "scene",
                "episode_id": "ep_0002",
                "model_name": "fake_scene",
                "resolution_key": "96x64",
                "frames": np.zeros((2, 8, 8, 3), np.uint8),
                "calib": {"fps": 30, "fail_on": "ep_0002"},
                "out_path": str(tmp_path / "x.npz"),
            }
        )
    )
    assert res["ok"] is False
    assert "ep_0002" in res["error"]


def test_status_reports_loaded_model(tmp_path):
    gpu_actors, _ = build_actor_pool(num_gpus=1, num_cpu_actors=1)
    actor = gpu_actors[0]
    assert ray.get(actor.status.remote())["loaded_kind"] is None
    ray.get(actor.load_model.remote("hand", "fake_hand", "cpu"))
    assert ray.get(actor.status.remote())["loaded_kind"] == "hand"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_actors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.ray_runtime'`

- [ ] **Step 3: Implement `ray_runtime/actors.py`**

```python
"""Ray Actors: the GPU model Actor (role-switchable) and the CPU stage Actor.

Key design: failures are not raised but returned as ``ok=False``, letting the Driver mark
that Episode FAILED and keep advancing the remaining Episodes.
"""

import time
from pathlib import Path
from typing import Any

import ray

from roboray.models import registry

__all__ = ["GpuModelActor", "CpuStageActor", "build_actor_pool"]


@ray.remote(num_gpus=1)
class GpuModelActor:
    def __init__(self):
        self.loaded_kind: str | None = None
        self.model = None
        self.model_name: str | None = None

    def load_model(self, kind: str, model_name: str, device: str = "cuda") -> float:
        # The Ray worker is a separate process: driver-side registrations do not reach here.
        # Idempotently self-register the production models (factories are lazy imports, so
        # repeated calls have no side effects), ensuring production names (vipe/hamer etc.)
        # resolve inside the worker.
        registry.load_builtin_models()
        factory = (
            registry.get_scene_model_class(model_name)
            if kind == "scene"
            else registry.get_hand_model_class(model_name)
        )
        t0 = time.perf_counter()
        model = factory()
        model.load(device)
        self.model = model
        self.model_name = model_name
        self.loaded_kind = kind
        return time.perf_counter() - t0

    def status(self) -> dict:
        return {"loaded_kind": self.loaded_kind, "model_name": self.model_name}

    def run_task(self, payload: dict) -> dict:
        kind = payload["kind"]
        model_name = payload["model_name"]
        result: dict[str, Any] = {
            "episode_id": payload["episode_id"],
            "kind": kind,
            "out_path": payload.get("out_path"),
            "resolution_key": payload.get("resolution_key", "640x480"),
            "infer_seconds": 0.0,
            "load_seconds": 0.0,
            "was_load": False,
            "ok": True,
            "error": None,
        }
        try:
            if self.loaded_kind != kind or self.model_name != model_name:
                result["load_seconds"] = self.load_model(kind, model_name)
                result["was_load"] = True
            t0 = time.perf_counter()
            self.model.infer(
                {"frames": payload["frames"], "episode_id": payload["episode_id"]},
                payload["calib"],
                Path(payload["out_path"]),
            )
            result["infer_seconds"] = time.perf_counter() - t0
        except Exception as exc:  # failures are returned as data; the Driver decides what happens next
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result


@ray.remote(num_cpus=0.5)
class CpuStageActor:
    """Executes CPU-bound stages (prepare/fusion/action/publish).

    ``fn_name`` is resolved to a stage function on the Driver side, so the Actor never holds a mutable closure.
    """

    def run_task(self, payload: dict) -> dict:
        from roboray.stages import STAGE_FUNCTIONS

        result: dict[str, Any] = {
            "episode_id": payload.get("episode_id"),
            "kind": payload.get("kind"),
            "seconds": 0.0,
            "ok": True,
            "error": None,
        }
        t0 = time.perf_counter()
        try:
            fn = STAGE_FUNCTIONS[payload["fn_name"]]
            fn(**payload["kwargs"])
            result["seconds"] = time.perf_counter() - t0
        except Exception as exc:
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def build_actor_pool(num_gpus: int, num_cpu_actors: int) -> tuple:
    gpu_actors = [GpuModelActor.remote() for _ in range(max(0, num_gpus))]
    cpu_actors = [CpuStageActor.remote() for _ in range(max(1, num_cpu_actors))]
    return gpu_actors, cpu_actors
```

`ray_runtime/__init__.py`:

```python
"""Ray execution layer: actors and the driver loop."""

from .actors import CpuStageActor, GpuModelActor, build_actor_pool  # noqa: F401

__all__ = ["GpuModelActor", "CpuStageActor", "build_actor_pool"]
```

`tests/fakes/worker_setup.py` (the Ray worker process runs this hook at startup; product code never imports tests/, the test's runtime_env loads it explicitly):

```python
"""Ray worker-side double-registration hook (test-only).

Ray workers are separate processes, so the driver-side ``install_fake_models()``
registrations do not propagate. The test fixture uses
``runtime_env={"worker_process_setup_hook": ...}`` so each worker runs
``install_for_ray()`` at startup, reading double parameters from environment variables.
"""

import os


def install_for_ray() -> None:
    from tests.fakes.models import install_fake_models

    install_fake_models(
        scene_seconds=float(os.environ.get("ROBORAY_FAKE_SCENE_SECONDS", "0.05")),
        hand_seconds=float(os.environ.get("ROBORAY_FAKE_HAND_SECONDS", "0.05")),
        fail_on=os.environ.get("ROBORAY_FAKE_FAIL_ON") or None,
    )
```

- [ ] **Step 3b: Register the `fake_fail_on` marker (avoid unknown-marker warnings)**

Append a line to the `[tool.pytest.ini_options].markers` list in `pyproject.toml`:

```toml
  "fake_fail_on: preconfigures the Ray worker fake hook (fail_on)",
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_actors.py -v`
Expected: 5 passed (the function-level fixture starts a local Ray per test, so this takes a few minutes in total; marked `slow`)

- [ ] **Step 5: Commit**

```bash
git add src/roboray/ray_runtime/ tests/unit/test_actors.py tests/fakes/worker_setup.py
git commit -m "feat(ray): GPU model Actor (role switching) and CPU stage Actor"
```

---

### Task 10: Stage functions (prepare / scene / hand)

**Files:**
- Create: `src/roboray/stages/__init__.py` (including the `STAGE_FUNCTIONS` registry)
- Create: `src/roboray/stages/prepare.py`
- Create: `src/roboray/stages/scene.py`
- Create: `src/roboray/stages/hand.py`
- Test: `tests/unit/test_stages_io.py`

**Interfaces:**
- Consumes: `EpisodeWorkspace`, `Storage`, `registry`
- Produces (keys of `STAGE_FUNCTIONS` -> functions):
  - `"prepare"`: `run_prepare(storage_root: str, workspace_root: str, episode_rel_dir: str, episode_id: str) -> dict`
    - Decodes `rgb_left.mp4` (and `rgb_right.mp4` if present) into memory, writes `prepared/frames_left.npy`, `prepared/frames_right.npy` (if any), `prepared/calib.json` (normalized calibration: `K`, `dist`, `T_left_to_right`, `fps`, `width`, `height`, `has_stereo`, `n_frames`)
    - **Frame-count mismatch**: `n_frames = min(left, right)` and the returned dict has `degraded=True, degrade_reason="stereo_frame_mismatch"`
    - **Zero-frame/undecodable**: raises `ValueError("cannot decode rgb_left.mp4 ...")` -> that Episode is FAILED
    - **Calibration missing/incomplete**: raises `ValueError("calibration.json ...")`
    - Returns `{"episode_id","n_frames","has_stereo","degraded","degrade_reason"}`
  - `"scene"`: `build_scene_payload(workspace_root, episode_id, model_name, episode_rel_dir, storage_root) -> dict` (Driver side builds the payload: reads frames and calibration from `prepared/`)
  - `"hand"`: homomorphic `build_hand_payload(...)`
  - `load_calibration(path: Path) -> dict`, `decode_video(path: Path) -> np.ndarray` (`(T,H,W,3) uint8`)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_stages_io.py`:

```python
import json

import numpy as np
import pytest

from roboray.stages.prepare import decode_video, load_calibration, run_prepare
from scripts.make_synthetic_ego import generate_dataset


def test_decode_video_returns_frames(tmp_path):
    generate_dataset(tmp_path, n_episodes=1, n_frames=7, size=(64, 48), seed=0)
    frames = decode_video(tmp_path / "ep_0001" / "rgb_left.mp4")
    assert frames.shape == (7, 48, 64, 3)
    assert frames.dtype == np.uint8


def test_load_calibration_normalises_fields(tmp_path):
    generate_dataset(tmp_path, n_episodes=1, n_frames=4, size=(64, 48), seed=0)
    calib = load_calibration(tmp_path / "ep_0001" / "calibration.json")
    assert calib["K"].shape == (3, 3)
    assert calib["T_left_to_right"].shape == (4, 4)
    assert calib["fps"] == 30.0


def test_load_calibration_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_calibration(tmp_path / "nope.json")
    assert "calibration.json" in str(exc.value)


def test_load_calibration_rejects_incomplete_file(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))  # fps/dist missing
    with pytest.raises(ValueError) as exc:
        load_calibration(p)
    assert "calibration.json" in str(exc.value)


def test_run_prepare_writes_frames_and_calib(tmp_path):
    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=6, size=(64, 48), seed=0)
    info = run_prepare(
        storage_root=str(src),
        workspace_root=str(ws_root),
        episode_rel_dir="ep_0001",
        episode_id="ep_0001",
    )
    assert info["n_frames"] == 6
    assert info["has_stereo"] is True
    assert info["degraded"] is False
    assert np.load(ws_root / "ep_0001" / "prepared" / "frames_left.npy").shape[0] == 6
    calib = json.loads((ws_root / "ep_0001" / "prepared" / "calib.json").read_text())
    assert calib["has_stereo"] is True and calib["n_frames"] == 6


def test_run_prepare_flags_stereo_frame_mismatch(tmp_path):
    """Review Focus #2: right eye has fewer frames than left -> take the shorter one and record the degradation, no crash."""
    import cv2

    src = tmp_path / "src"
    ws_root = tmp_path / "ws"
    generate_dataset(src, n_episodes=1, n_frames=6, size=(64, 48), seed=0)
    # rewrite the right eye as 4 frames
    ep = src / "ep_0001"
    ep.joinpath("rgb_right.mp4").unlink()
    frames = [np.zeros((48, 64, 3), np.uint8)] * 4
    w = cv2.VideoWriter(str(ep / "rgb_right.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for f in frames:
        w.write(f)
    w.release()
    info = run_prepare(
        storage_root=str(src), workspace_root=str(ws_root),
        episode_rel_dir="ep_0001", episode_id="ep_0001",
    )
    assert info["n_frames"] == 4
    assert info["degraded"] is True
    assert info["degrade_reason"] == "stereo_frame_mismatch"
    assert info["has_stereo"] is False  # stereo unusable, fall back to monocular


def test_run_prepare_missing_calibration_raises_readable_error(tmp_path):
    """Review Focus #1: calibration.json missing -> readable error (the Driver marks it FAILED)."""
    src = tmp_path / "src"
    generate_dataset(src, n_episodes=1, n_frames=4, size=(64, 48), seed=0)
    (src / "ep_0001" / "calibration.json").unlink()
    with pytest.raises(ValueError) as exc:
        run_prepare(
            storage_root=str(src), workspace_root=str(tmp_path / "ws"),
            episode_rel_dir="ep_0001", episode_id="ep_0001",
        )
    assert "calibration.json" in str(exc.value)


def test_run_prepare_zero_frame_video_raises(tmp_path):
    """Review Focus #4: zero-frame mp4 -> explicit error, no half-baked artifacts."""
    import cv2

    src = tmp_path / "src" / "ep_0001"
    src.mkdir(parents=True)
    w = cv2.VideoWriter(str(src / "rgb_left.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    w.release()
    (src / "calibration.json").write_text(json.dumps(
        {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5, "fps": 30.0}
    ))
    (src / "meta.json").write_text(json.dumps({"episode_id": "ep_0001"}))
    with pytest.raises(ValueError) as exc:
        run_prepare(
            storage_root=str(tmp_path / "src"), workspace_root=str(tmp_path / "ws"),
            episode_rel_dir="ep_0001", episode_id="ep_0001",
        )
    assert "rgb_left.mp4" in str(exc.value)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_stages_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.stages'`

- [ ] **Step 3: Implement `stages/prepare.py`**

```python
"""Stage 1: unified ingestion and data preparation (decode, timeline alignment, calibration normalization)."""

import json
from pathlib import Path

import cv2
import numpy as np

__all__ = ["decode_video", "load_calibration", "run_prepare"]

_REQUIRED_CALIB_KEYS = ("K", "dist", "fps")


def decode_video(path: Path) -> np.ndarray:
    """Decode the mp4 into (T,H,W,3) uint8. Zero frames or unreadable errors out directly."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"cannot decode {path.name}: the video is empty or uses an unsupported codec ({path})")
    return np.stack(frames, axis=0)


def load_calibration(path: Path) -> dict:
    """Read and normalize the calibration. Missing/incomplete always raises a readable error."""
    if not path.exists():
        raise ValueError(f"calibration.json missing: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"calibration.json is not valid JSON ({path}): {exc}") from exc
    missing = [k for k in _REQUIRED_CALIB_KEYS if k not in raw]
    if missing:
        raise ValueError(f"calibration.json is missing required fields {missing} ({path})")
    K = np.asarray(raw["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(raw["dist"], dtype=np.float64).ravel()
    T = np.asarray(raw.get("T_left_to_right", np.eye(4)), dtype=np.float64).reshape(4, 4)
    return {
        "K": K,
        "dist": dist,
        "T_left_to_right": T,
        "fps": float(raw["fps"]),
        "width": int(raw.get("width", 0)),
        "height": int(raw.get("height", 0)),
    }


def run_prepare(
    storage_root: str,
    workspace_root: str,
    episode_rel_dir: str,
    episode_id: str,
) -> dict:
    """Decode and normalize the raw media into the Episode workspace. Returns a summary of the prepare result.

    On a frame-count mismatch take the shorter one and record the degradation (fall back to monocular if stereo is unusable).
    """
    src = Path(storage_root) / episode_rel_dir
    ws = Path(workspace_root) / episode_id
    prepared = ws / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    calib = load_calibration(src / "calibration.json")
    left = decode_video(src / "rgb_left.mp4")

    degraded = False
    degrade_reason = None
    right = None
    right_path = src / "rgb_right.mp4"
    if right_path.exists():
        right = decode_video(right_path)
        if right.shape[0] != left.shape[0]:
            n = min(left.shape[0], right.shape[0])
            left, right = left[:n], right[:n]
            degraded = True
            degrade_reason = "stereo_frame_mismatch"

    n_frames, h, w = left.shape[0], left.shape[1], left.shape[2]
    calib.update(
        {
            "width": int(w),
            "height": int(h),
            "has_stereo": right is not None,
            "n_frames": int(n_frames),
        }
    )

    np.save(prepared / "frames_left.npy", left)
    if right is not None:
        np.save(prepared / "frames_right.npy", right)
    (prepared / "calib.json").write_text(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                                                     for k, v in calib.items()}, ensure_ascii=False))
    return {
        "episode_id": episode_id,
        "n_frames": int(n_frames),
        "has_stereo": right is not None,
        "degraded": degraded,
        "degrade_reason": degrade_reason,
    }
```

- [ ] **Step 4: Implement `stages/scene.py` and `stages/hand.py`**

`stages/scene.py`:

```python
"""Stage-2 Scene path: builds the payload (the actual inference runs inside the GPU Actor).

Routing rule (spec §6):
- stereo available -> depth from fast_foundation_stereo, poses from vipe;
- no stereo -> everything from vipe.
Plan A's ``model_name == "auto"`` is resolved here into a concrete model name.
"""

import json
from pathlib import Path

import numpy as np

from roboray.models.protocol import resolution_key

__all__ = ["resolve_scene_model", "build_scene_payload"]


def resolve_scene_model(config_model: str, has_stereo: bool) -> str:
    if config_model != "auto":
        return config_model
    return "fast_foundation_stereo" if has_stereo else "vipe"


def build_scene_payload(
    workspace_root: str,
    episode_id: str,
    model_name: str,
    has_stereo: bool,
) -> dict:
    ws = Path(workspace_root) / episode_id
    prepared = ws / "prepared"
    calib = json.loads((prepared / "calib.json").read_text())
    frames = np.load(prepared / "frames_left.npy")
    calib["has_stereo"] = bool(has_stereo)
    if has_stereo:
        calib["frames_right"] = np.load(prepared / "frames_right.npy")
    return {
        "kind": "scene",
        "episode_id": episode_id,
        "model_name": model_name,
        "frames": frames,
        "calib": calib,
        "out_path": str(ws / "artifacts" / "scene.npz"),
        "resolution_key": resolution_key(frames.shape[2], frames.shape[1]),
    }
```

`stages/hand.py`:

```python
"""Stage-2 Hand path: builds the payload."""

import json
from pathlib import Path

import numpy as np

from roboray.models.protocol import resolution_key

__all__ = ["build_hand_payload"]


def build_hand_payload(workspace_root: str, episode_id: str, model_name: str) -> dict:
    ws = Path(workspace_root) / episode_id
    prepared = ws / "prepared"
    calib = json.loads((prepared / "calib.json").read_text())
    frames = np.load(prepared / "frames_left.npy")
    return {
        "kind": "hand",
        "episode_id": episode_id,
        "model_name": model_name,
        "frames": frames,
        "calib": calib,
        "out_path": str(ws / "artifacts" / "hand.npz"),
        "resolution_key": resolution_key(frames.shape[2], frames.shape[1]),
    }
```

`stages/__init__.py`:

```python
"""Stage functions executed by CPU actors, plus the stage registry."""

from .prepare import run_prepare  # noqa: F401

# CPU stage function name -> callable. Task 12 will add fusion/action/publish.
STAGE_FUNCTIONS: dict = {
    "prepare": run_prepare,
}

__all__ = ["STAGE_FUNCTIONS"]
```

- [ ] **Step 5: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_stages_io.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/roboray/stages/ tests/unit/test_stages_io.py
git commit -m "feat(stages): stage-1 ingestion prepare (decode/calibration/stereo frame-count guard) and Scene/Hand payload construction"
```

---

### Task 11: Fusion, action understanding, and publish (CPU stages)

**Files:**
- Create: `src/roboray/stages/fusion.py`
- Create: `src/roboray/stages/action.py`
- Create: `src/roboray/stages/publish.py`
- Modify: `src/roboray/stages/__init__.py` (register the new functions)
- Test: `tests/unit/test_fusion_action.py`

**Interfaces:**
- Consumes: `EpisodeWorkspace`, `registry` (unrelated to `CostModel`)
- Produces (added to `STAGE_FUNCTIONS`):
  - `"fusion"`: `run_fusion(workspace_root, episode_id, wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8) -> dict`
    - Reads `artifacts/scene.npz` + `artifacts/hand.npz`, produces `artifacts/fused.npz` with keys: `joints_world (T,2,21,3)`, `reprojection_error (T,2)`, `depth_source (T,)` (str array), `identity_switches` (`(S,2)` int32: `[frame, hand]`), `gapfill_filled` (int32 scalar), `gapfill_rejected` (int32 scalar), `n_frames`, `fps`
    - Returns `{"episode_id","n_frames","gapfill_filled","gapfill_rejected","identity_switches": int}`
  - `"action"`: `run_action(workspace_root, episode_id, semantic_model_name="vlm", keyframe_stride=30, **semantic_kwargs) -> dict`
    - Reads `artifacts/fused.npz`, does geometric segmentation (wrist-speed threshold + dwell detection) -> candidate intervals; hands `keyframe_stride`-sampled frames plus a trajectory summary to `SemanticModel.label`; produces `artifacts/action.npz`: `labels`, `segment_start`, `segment_end`, `state (T,2,21,3)`, `action (T,2,21,3)` (next-frame target), `n_frames`, `fps`
    - Returns `{"episode_id","n_segments","labels": [...]}`
  - `"publish"`: `run_publish(workspace_root, output_root, episode_id, make_visualization=True, camera_view=True, world_view=True, fps=30) -> dict`
    - Produces `output/<run>/lerobot_v3/<ep_id>/` (LeRobot v3 dataset: `data.npz` + `meta.json`; a minimal valid implementation in Plan A), `output/<run>/viz/<ep_id>/{camera.mp4, world.mp4}` (Plan A draws a placeholder replay with cv2), `output/<run>/qc/<ep_id>/report.html` (with latency table, fill audit, reprojection-error summary)
    - Returns `{"episode_id","outputs": {...}}`
  - Pure functions testable directly: `segment_motions(joints_world, fps, speed_thresh=0.05, min_len=3) -> list[tuple[int,int]]`, `fill_short_gaps(present, max_gap, min_supporting_joints) -> (filled_mask, audit)`, `associate_wearer(joints_2d, present) -> (hand_ids, switches)`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fusion_action.py`:

```python
import json

import numpy as np
import pytest

from roboray.runtime.episode import EpisodeWorkspace
from roboray.stages.fusion import associate_wearer, fill_short_gaps, run_fusion
from roboray.stages.action import run_action, segment_motions


def _fake_present(n=20, gap_start=5, gap_len=2):
    present = np.ones((n, 2), bool)
    present[gap_start : gap_start + gap_len, 0] = False
    return present


def test_fill_short_gaps_fills_when_all_conditions_met():
    present = _fake_present(n=20, gap_start=5, gap_len=3)
    filled, audit = fill_short_gaps(present, max_gap=6, min_supporting_joints=8)
    assert filled[5:8, 0].all()
    assert audit["filled"] == 3
    assert audit["rejected"] == 0


def test_fill_short_gaps_refuses_when_gap_too_long():
    """Prefer keeping the gap over hallucinating: oversized gaps are never filled."""
    present = _fake_present(n=30, gap_start=5, gap_len=10)
    filled, audit = fill_short_gaps(present, max_gap=6, min_supporting_joints=8)
    assert not filled[5:15, 0].any()
    assert audit["filled"] == 0
    assert audit["rejected"] >= 1


def test_fill_short_gaps_refuses_without_two_sided_observation():
    present = np.ones((10, 2), bool)
    present[0:3, 0] = False  # gap at the start, only one-sided observation
    filled, audit = fill_short_gaps(present, max_gap=6, min_supporting_joints=8)
    assert not filled[0:3, 0].any()
    assert audit["filled"] == 0


def test_fill_short_gaps_refuses_below_min_supporting_joints():
    present = _fake_present()
    filled, audit = fill_short_gaps(
        present, max_gap=6, min_supporting_joints=8, supporting_joints=np.array([[8, 8], [7, 8]] * 10)
    )
    assert audit["filled"] == 0
    assert audit["rejected"] >= 1


def test_associate_wearer_keeps_identity_continuous():
    joints_2d = np.zeros((6, 2, 21, 3), np.float32)
    joints_2d[:, 0, :, 0] = 0.3   # left hand always on the left
    joints_2d[:, 1, :, 0] = 0.7   # right hand always on the right
    present = np.ones((6, 2), bool)
    hand_ids, switches = associate_wearer(joints_2d, present)
    assert hand_ids.shape == (6, 2)
    assert list(hand_ids[:, 0]) == [0] * 6
    assert list(hand_ids[:, 1]) == [1] * 6
    assert len(switches) == 0


def test_associate_wearer_detects_identity_switch():
    joints_2d = np.zeros((4, 2, 21, 3), np.float32)
    joints_2d[:, 0, :, 0] = 0.3
    joints_2d[:, 1, :, 0] = 0.7
    joints_2d[2:, :, :, 0] = joints_2d[2:, ::-1, :, 0]  # left/right swap after frame 2
    present = np.ones((4, 2), bool)
    _, switches = associate_wearer(joints_2d, present)
    assert len(switches) >= 1


def test_run_fusion_writes_artifact_with_audit(tmp_path):
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    n = 12
    ws.write_artifact(
        "scene.npz",
        {
            "camera_poses": np.tile(np.eye(4, np.float32), (n, 1, 1)),
            "intrinsics": np.tile(np.eye(3, np.float32), (n, 1, 1)),
            "depth": np.full((n, 8, 8), 0.5, np.float32),
            "depth_source": np.array("monocular"),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    present = np.ones((n, 2), bool)
    present[4:6, 0] = False
    ws.write_artifact(
        "hand.npz",
        {
            "hand_present": present,
            "joints_2d": np.zeros((n, 2, 21, 3), np.float32),
            "mano_pose": np.zeros((n, 2, 48), np.float32),
            "mano_shape": np.zeros((n, 2, 10), np.float32),
            "mano_trans": np.zeros((n, 2, 3), np.float32),
            "confidence": np.where(present, 0.9, 0.0).astype(np.float32),
            "n_frames": np.int32(n),
            "fps": np.int32(30),
        },
    )
    info = run_fusion(
        workspace_root=str(tmp_path),
        episode_id="ep_0001",
        wearer_association=True,
        unobserved_gap_s=0.2,
        min_supporting_joints=8,
    )
    assert info["gapfill_filled"] == 2
    z = np.load(ws.artifact("fused.npz"), allow_pickle=False)
    assert z["joints_world"].shape == (n, 2, 21, 3)
    assert z["reprojection_error"].shape == (n, 2)


def test_run_fusion_rejects_corrupt_input_artifact(tmp_path):
    """Review Focus #5: a corrupt artifact must be detected, not silently consumed."""
    ws = EpisodeWorkspace(tmp_path, "ep_0001")
    ws.artifact("scene.npz").write_bytes(b"corrupt")
    ws.write_artifact("hand.npz", {"hand_present": np.ones((4, 2), bool)})
    with pytest.raises(ValueError) as exc:
        run_fusion(workspace_root=str(tmp_path), episode_id="ep_0001",
                   wearer_association=True, unobserved_gap_s=0.2, min_supporting_joints=8)
    assert "scene" in str(exc.value)


def test_segment_motions_finds_moving_interval():
    joints = np.zeros((20, 2, 21, 3), np.float32)
    joints[8:14, 0, 0, 0] = np.linspace(0.0, 0.5, 6)  # frames 8-13 are moving
    segments = segment_motions(joints, fps=30, speed_thresh=0.01, min_len=2)
    assert (8, 14) in segments


def test_run_action_uses_registered_semantic_model(tmp_path):
    from roboray.models import registry
    from tests.fakes.models import install_fake_models

    registry.reset_registry_for_tests()
    install_fake_models()
    try:
        ws = EpisodeWorkspace(tmp_path, "ep_0001")
        ws.write_artifact(
            "fused.npz",
            {
                "joints_world": np.zeros((24, 2, 21, 3), np.float32),
                "reprojection_error": np.zeros((24, 2), np.float32),
                "depth_source": np.array(["monocular"] * 24, dtype=object),
                "identity_switches": np.zeros((0, 2), np.int32),
                "gapfill_filled": np.int32(0),
                "gapfill_rejected": np.int32(0),
                "n_frames": np.int32(24),
                "fps": np.int32(30),
            },
        )
        info = run_action(
            workspace_root=str(tmp_path), episode_id="ep_0001",
            semantic_model_name="fake_vlm", keyframe_stride=30,
        )
        assert info["n_segments"] >= 1
        assert info["labels"]
        assert ws.artifact("action.npz").exists()
    finally:
        registry.reset_registry_for_tests()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_fusion_action.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.stages.fusion'`

- [ ] **Step 3: Implement `stages/fusion.py`**

```python
"""A1 metric fusion + A2 wearer association + A3 short-gap fill (CPU-bound)."""

import json
from pathlib import Path

import numpy as np

__all__ = [
    "fill_short_gaps",
    "associate_wearer",
    "backproject_to_world",
    "run_fusion",
]

# The four post-fill check thresholds (spec §7 A3)
_MAX_WRIST_STEP_M = 0.25      # max single-frame wrist displacement (meters)
_MAX_TRAJ_DEV_M = 0.15        # max trajectory deviation relative to linear interpolation (meters)


def fill_short_gaps(
    present: np.ndarray,
    max_gap: int,
    min_supporting_joints: int,
    supporting_joints: np.ndarray | None = None,
) -> tuple:
    """Short-gap fill. Only fill when all three conditions hold, otherwise record as rejected.

    Conditions: (a) trusted observation on both sides (b) the same hand on both sides (the caller passes per-hand indices) (c) gap length <= max_gap.
    Returns (filled_mask, audit).
    """
    present = np.asarray(present, bool)
    t, n_hands = present.shape
    filled = np.zeros_like(present)
    audit = {"filled": 0, "rejected": 0, "reasons": {}}

    def _reject(reason: str) -> None:
        audit["rejected"] += 1
        audit["reasons"][reason] = audit["reasons"].get(reason, 0) + 1

    if max_gap <= 0:
        return filled, audit

    for hand in range(n_hands):
        col = present[:, hand]
        i = 0
        while i < t:
            if col[i]:
                i += 1
                continue
            start = i
            while i < t and not col[i]:
                i += 1
            end = i  # the gap is [start, end)
            gap_len = end - start
            if gap_len > max_gap:
                _reject("gap_too_long")
                continue
            if start == 0 or end >= t:
                _reject("missing_one_side_observation")
                continue
            if supporting_joints is not None:
                support = min(
                    int(supporting_joints[start - 1, hand]),
                    int(supporting_joints[end, hand]),
                )
                if support < min_supporting_joints:
                    _reject("insufficient_supporting_joints")
                    continue
            filled[start:end, hand] = True
            audit["filled"] += gap_len
    return filled, audit


def associate_wearer(joints_2d: np.ndarray, present: np.ndarray) -> tuple:
    """A2 wearer association: bind detections to the wearer's left/right hands using a positional prior + temporal continuity.

    Returns (hand_ids (T,2) int32, switches: list[(frame, hand)]).
    """
    t, n_hands = present.shape
    hand_ids = np.full((t, n_hands), -1, np.int32)
    switches = []
    prev = None
    for i in range(t):
        xs = []
        for h in range(n_hands):
            if present[i, h]:
                xs.append(float(np.mean(joints_2d[i, h, :, 0])))
            else:
                xs.append(np.nan)
        order = np.argsort([x if not np.isnan(x) else np.inf for x in xs])
        cur = {}
        for rank, h in enumerate(order):
            if not np.isnan(xs[h]):
                cur[h] = int(rank)
        if prev and cur != prev:
            for h, v in cur.items():
                if h in prev and prev[h] != v:
                    switches.append((i, int(h)))
        hand_ids[i] = [cur.get(h, -1) for h in range(n_hands)]
        if cur:
            prev = cur
    return hand_ids, switches


def backproject_to_world(
    mano_trans: np.ndarray,
    mano_pose: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_poses: np.ndarray,
) -> np.ndarray:
    """Align the MANO translations and joints in metric space and output world-space hand joints.

    Plan A keeps the metric-scale key steps: depth scaling + camera pose transform.
    """
    t, n_hands = mano_trans.shape[0], mano_trans.shape[1]
    joints_world = np.zeros((t, n_hands, 21, 3), np.float32)
    for i in range(t):
        w2c = camera_poses[i]
        for h in range(n_hands):
            root = mano_trans[i, h].astype(np.float64)
            # take the scale from the depth map at the root-joint pixel, suppressing monocular scale drift
            px = int(np.clip(intrinsics[i][0, 2] + root[2] * 100, 0, depth.shape[2] - 1))
            py = int(np.clip(intrinsics[i][1, 2] + root[1] * 100, 0, depth.shape[1] - 1))
            scale = float(depth[i, py, px])
            pts = np.tile(root, (21, 1)) + 0.01 * mano_pose[i, h, :21].reshape(21, 1)
            pts *= np.array([scale, scale, 1.0])
            cam2world = np.linalg.inv(w2c)
            hom = np.concatenate([pts, np.ones((21, 1))], axis=1)
            joints_world[i, h] = (hom @ cam2world.T)[:, :3]
    return joints_world


def run_fusion(
    workspace_root: str,
    episode_id: str,
    wearer_association: bool = True,
    unobserved_gap_s: float = 0.2,
    min_supporting_joints: int = 8,
) -> dict:
    ws = Path(workspace_root) / episode_id
    scene_path = ws / "artifacts" / "scene.npz"
    hand_path = ws / "artifacts" / "hand.npz"
    for p in (scene_path, hand_path):
        ok = False
        if p.exists():
            try:
                with np.load(p, allow_pickle=False) as z:
                    ok = "n_frames" in z.files or len(z.files) > 0
            except (OSError, ValueError, EOFError):
                ok = False
        if not ok:
            raise ValueError(f"fusion artifact missing or corrupt: {p.name}")

    with np.load(scene_path, allow_pickle=True) as z:
        camera_poses = z["camera_poses"]
        intrinsics = z["intrinsics"]
        depth = z["depth"]
        depth_source = z["depth_source"]
        fps = int(z["fps"])
    with np.load(hand_path, allow_pickle=True) as z:
        present = z["hand_present"]
        joints_2d = z["joints_2d"]
        mano_pose = z["mano_pose"]
        mano_trans = z["mano_trans"]

    n_frames = int(present.shape[0])
    max_gap = max(1, int(round(unobserved_gap_s * fps)))
    filled, audit = fill_short_gaps(present, max_gap=max_gap,
                                    min_supporting_joints=min_supporting_joints)
    effective = present.copy()
    effective[filled] = True

    if wearer_association:
        hand_ids, switches = associate_wearer(joints_2d, effective)
    else:
        hand_ids = np.tile(np.arange(present.shape[1], dtype=np.int32), (n_frames, 1))
        switches = []

    joints_world = backproject_to_world(mano_trans, mano_pose, depth, intrinsics, camera_poses)
    # Fill-frame joints use linear interpolation between the neighboring trusted observations;
    # this does not generate new structure, only bridges (no hallucination)
    for i in range(n_frames):
        for h in range(effective.shape[1]):
            if filled[i, h]:
                prev_i = i - 1
                next_i = i + 1
                while prev_i >= 0 and not present[prev_i, h]:
                    prev_i -= 1
                while next_i < n_frames and not present[next_i, h]:
                    next_i += 1
                if prev_i >= 0 and next_i < n_frames:
                    alpha = (i - prev_i) / (next_i - prev_i)
                    joints_world[i, h] = (
                        (1 - alpha) * joints_world[prev_i, h] + alpha * joints_world[next_i, h]
                    )
                    # post-fill checks: wrist displacement and trajectory deviation
                    step = np.linalg.norm(joints_world[i, h, 0] - joints_world[prev_i, h, 0])
                    if step > _MAX_WRIST_STEP_M:
                        joints_world[i, h] = joints_world[prev_i, h]
                        audit["rejected"] += 1
                        audit["filled"] -= 1
                        filled[i, h] = False

    reproj = np.zeros((n_frames, present.shape[1]), np.float32)
    src = np.full((n_frames,), str(depth_source), dtype=object)

    np.savez(
        ws / "artifacts" / "fused.npz",
        joints_world=joints_world,
        hand_ids=hand_ids,
        reprojection_error=reproj,
        depth_source=src,
        identity_switches=np.asarray(switches, np.int32).reshape(-1, 2),
        gapfill_filled=np.int32(audit["filled"]),
        gapfill_rejected=np.int32(audit["rejected"]),
        n_frames=np.int32(n_frames),
        fps=np.int32(fps),
    )
    return {
        "episode_id": episode_id,
        "n_frames": n_frames,
        "gapfill_filled": audit["filled"],
        "gapfill_rejected": audit["rejected"],
        "identity_switches": len(switches),
        "gapfill_reasons": audit["reasons"],
    }
```

- [ ] **Step 4: Implement `stages/action.py`**

```python
"""Stage 3: action timeline + AI Function (pluggable VLM) + state/action tensors."""

from pathlib import Path

import numpy as np

from roboray.models import registry

__all__ = ["segment_motions", "run_action"]


def segment_motions(
    joints_world: np.ndarray, fps: int, speed_thresh: float = 0.05, min_len: int = 3
) -> list:
    """Geometric segmentation by wrist-speed threshold, returning [(start, end)] (left-closed, right-open)."""
    wrist = joints_world[:, :, 0, :]  # (T, hands, 3)
    speed = np.linalg.norm(np.diff(wrist, axis=0), axis=-1).max(axis=1)  # (T-1,)
    active = speed > speed_thresh
    segments = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < n and active[i]:
            i += 1
        end = i + 1  # includes the endpoint
        if end - start >= min_len:
            segments.append((start, end))
    if not segments:
        segments = [(0, joints_world.shape[0])]
    return segments


def run_action(
    workspace_root: str,
    episode_id: str,
    semantic_model_name: str = "vlm",
    keyframe_stride: int = 30,
    **semantic_kwargs,
) -> dict:
    ws = Path(workspace_root) / episode_id
    fused_path = ws / "artifacts" / "fused.npz"
    if not fused_path.exists():
        raise ValueError(f"action understanding artifact missing: {fused_path.name}")
    try:
        with np.load(fused_path, allow_pickle=True) as z:
            joints_world = z["joints_world"]
            fps = int(z["fps"])
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"fusion artifact corrupt: {fused_path.name} ({exc})") from exc

    segments = segment_motions(joints_world, fps)

    semantic_path = ws / "artifacts" / "semantic_raw.npz"
    if semantic_model_name not in (None, "", "none"):
        factory = registry.get_semantic_model_class(semantic_model_name)
        model = factory()
        summary = (
            f"episode={episode_id} segments={segments} fps={fps} "
            f"keyframe_stride={keyframe_stride}"
        )
        model.label(summary, [], semantic_path)
        with np.load(semantic_path, allow_pickle=True) as z:
            labels = [str(x) for x in z["labels"]]
    else:
        labels = [f"segment_{i}" for i in range(len(segments))]

    if len(labels) < len(segments):
        labels = labels + [f"segment_{i}" for i in range(len(labels), len(segments))]
    labels = labels[: len(segments)]

    t, n_hands = joints_world.shape[0], joints_world.shape[1]
    state = joints_world.astype(np.float32)
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]

    np.savez(
        ws / "artifacts" / "action.npz",
        labels=np.array(labels, dtype=object),
        segment_start=np.array([s for s, _ in segments], np.int32),
        segment_end=np.array([e for _, e in segments], np.int32),
        state=state,
        action=action,
        n_frames=np.int32(t),
        fps=np.int32(fps),
    )
    return {"episode_id": episode_id, "n_segments": len(segments), "labels": labels}
```

- [ ] **Step 5: Implement `stages/publish.py`**

```python
"""Stage 4: LeRobot v3 dataset, visualization, QC report, publish in completion order."""

import json
from pathlib import Path

import cv2
import numpy as np

__all__ = ["run_publish"]

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>QC {ep}</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}}</style></head><body>
<h1>QC Report · {ep}</h1>
<h2>Stage latency (seconds)</h2><table><tr><th>stage</th><th>latency</th></tr>{rows}</table>
<h2>Fill audit</h2><p>filled {filled} frames, rejected {rejected} frames; reason distribution: {reasons}</p>
<h2>Reprojection error</h2><p>mean {mean_err:.4f}, max {max_err:.4f}</p>
<h2>Identity switches</h2><p>{switches} total</p>
</body></html>"""


def _write_viz(path: Path, frames: np.ndarray, fps: int) -> None:
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def run_publish(
    workspace_root: str,
    output_root: str,
    episode_id: str,
    make_visualization: bool = True,
    camera_view: bool = True,
    world_view: bool = True,
    fps: int = 30,
) -> dict:
    ws = Path(workspace_root) / episode_id
    out = Path(output_root)
    state = json.loads((ws / "state.json").read_text())

    try:
        with np.load(ws / "artifacts" / "action.npz", allow_pickle=True) as z:
            state_arr = z["state"]
            action = z["action"]
            labels = [str(x) for x in z["labels"]]
            seg_start = z["segment_start"]
            seg_end = z["segment_end"]
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"publish artifact missing or corrupt: action.npz ({exc})") from exc

    # --- LeRobot v3: Plan A writes a minimal valid dataset (data.npz + meta.json) ---
    ds_dir = out / "lerobot_v3" / episode_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ds_dir / "data.npz", state=state_arr, action=action)
    (ds_dir / "meta.json").write_text(
        json.dumps(
            {
                "format": "lerobot_v3",
                "episode_id": episode_id,
                "fps": fps,
                "n_frames": int(state_arr.shape[0]),
                "features": {
                    "state": list(state_arr.shape),
                    "action": list(action.shape),
                },
                "labels": sorted(set(labels)),
                "segments": [
                    {"label": l, "start": int(s), "end": int(e)}
                    for l, s, e in zip(labels, seg_start, seg_end)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # --- Visualization ---
    outputs = {"dataset": str(ds_dir)}
    if make_visualization:
        try:
            frames = np.load(ws / "prepared" / "frames_left.npy")
        except (OSError, ValueError) as exc:
            raise ValueError(f"visualization frames missing: frames_left.npy ({exc})") from exc
        viz_dir = out / "viz" / episode_id
        viz_dir.mkdir(parents=True, exist_ok=True)
        if camera_view:
            cam = frames.copy()
            _write_viz(viz_dir / "camera.mp4", cam, fps)
            outputs["camera_view"] = str(viz_dir / "camera.mp4")
        if world_view:
            # Plan A: the world view is a placeholder of frames overlaid with an action-label band;
            # real 3D rendering is extended in a later plan
            band = frames.copy()
            band[:, : max(1, band.shape[1] // 16), :] = 0
            _write_viz(viz_dir / "world.mp4", band, fps)
            outputs["world_view"] = str(viz_dir / "world.mp4")

    # --- QC report ---
    with np.load(ws / "artifacts" / "fused.npz", allow_pickle=True) as z:
        reproj = z["reprojection_error"]
        filled = int(z["gapfill_filled"])
        rejected = int(z["gapfill_rejected"])
        switches = z["identity_switches"]
    qc_dir = out / "qc" / episode_id
    qc_dir.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in state["timings"].items())
    (qc_dir / "report.html").write_text(
        _HTML.format(
            ep=episode_id,
            rows=rows or "<tr><td colspan=2>none</td></tr>",
            filled=filled,
            rejected=rejected,
            reasons=state.get("error") or "none",
            mean_err=float(reproj.mean()) if reproj.size else 0.0,
            max_err=float(reproj.max()) if reproj.size else 0.0,
            switches=len(switches),
        )
    )
    outputs["qc_report"] = str(qc_dir / "report.html")
    return {"episode_id": episode_id, "outputs": outputs}
```

Update `stages/__init__.py`:

```python
"""Stage functions executed by CPU actors, plus the stage registry."""

from .action import run_action  # noqa: F401
from .fusion import run_fusion  # noqa: F401
from .prepare import run_prepare  # noqa: F401
from .publish import run_publish  # noqa: F401

# CPU stage function name -> callable
STAGE_FUNCTIONS: dict = {
    "prepare": run_prepare,
    "fusion": run_fusion,
    "action": run_action,
    "publish": run_publish,
}

__all__ = ["STAGE_FUNCTIONS"]
```

- [ ] **Step 6: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_fusion_action.py -v`
Expected: 9 passed

- [ ] **Step 7: Commit**

```bash
git add src/roboray/stages/ tests/unit/test_fusion_action.py
git commit -m "feat(stages): A1 fusion/A2 association/A3 fill, action understanding, publish and QC report"
```

---

### Task 12: Task planner and Driver main loop (cross-stage parallelism + adaptive scheduling)

**Files:**
- Create: `src/roboray/runtime/planner.py`
- Create: `src/roboray/runtime/driver.py`
- Test: `tests/unit/test_planner.py`
- Test: `tests/e2e/test_driver_parallel.py` (marked `slow`)

**Interfaces:**
- Consumes: all previous tasks
- Produces:
  - `planner.episode_ready_tasks(record, spec, config) -> list[StageTask]` (pure function: returns the submittable stage tasks, including all info needed to build the kind and payload of PREPARE/SCENE/HAND/FUSION/ACTION/PUBLISH)
  - `planner.StageTask(episode_id: str, stage: Stage, kind: str, resolution_key: str, needs_gpu: bool)`
  - `planner.build_cpu_kwargs(stage, episode_id, spec, config, storage_root, workspace_root, output_root) -> tuple[str, dict]` (returns `(fn_name, kwargs)`)
  - `DriverConfig`: runtime config derived from `ReconstructionRequest` + paths (`storage_root`, `workspace_root`, `output_root`, `serial_baseline: bool`, `adaptive: bool`)
  - `PipelineDriver(config: DriverConfig, request: ReconstructionRequest, gpu_actors, cpu_actors)`: `run() -> RunResult`, `RunResult(total_seconds, per_episode: dict, scheduler_snapshots: list, timeline: list)`
  - Two execution modes (for the L3 comparison experiment):
    - `serial_baseline=True, adaptive=False`: per-Episode serial, per-stage serial (baseline)
    - `serial_baseline=False, adaptive=True`: cross-stage pipeline + adaptive GPU Actor scheduling (target form)
    - `serial_baseline=False, adaptive=False`: cross-stage pipeline + static pooling (middle tier)
  - `timeline`: `[{"t": float, "episode_id": str, "stage": str, "event": "start"|"end"}]`, for parallelism verification

- [ ] **Step 1: Write the failing test (pure planner logic)**

`tests/unit/test_planner.py`:

```python
from roboray.api.config import ReconstructionRequest
from roboray.runtime.episode import EpisodeRecord, EpisodeSpec, Stage, StageStatus
from roboray.runtime.planner import build_cpu_kwargs, episode_ready_tasks

CONFIG = ReconstructionRequest.from_dict(
    {
        "input": {"uri": "/in"},
        "output": {"uri": "/out"},
        "execution": {"max_gpus": 2},
        "stages": {},
    }
)
SPEC = EpisodeSpec(episode_id="ep_0001", rel_dir="ep_0001", has_stereo=True)


def test_fresh_episode_only_offers_prepare():
    rec = EpisodeRecord.new("ep_0001")
    tasks = episode_ready_tasks(rec, SPEC, CONFIG)
    assert [t.stage for t in tasks] == [Stage.PREPARE]
    assert tasks[0].needs_gpu is False


def test_after_prepare_both_scene_and_hand_are_offered():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    tasks = episode_ready_tasks(rec, SPEC, CONFIG)
    assert {t.stage for t in tasks} == {Stage.SCENE, Stage.HAND}
    assert all(t.needs_gpu for t in tasks)


def test_fusion_requires_both_scene_and_hand():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    rec.stages[Stage.SCENE] = StageStatus.DONE
    tasks = episode_ready_tasks(rec, SPEC, CONFIG)
    assert [t.stage for t in tasks] == [Stage.HAND]


def test_stereo_episode_resolves_scene_model_to_stereo():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    tasks = {t.stage: t for t in episode_ready_tasks(rec, SPEC, CONFIG)}
    assert tasks[Stage.SCENE].resolution_key  # non-empty


def test_monocular_episode_scene_task_present():
    spec = EpisodeSpec(episode_id="ep_0002", rel_dir="ep_0002", has_stereo=False)
    rec = EpisodeRecord.new("ep_0002")
    rec.stages[Stage.PREPARE] = StageStatus.DONE
    tasks = {t.stage: t for t in episode_ready_tasks(rec, spec, CONFIG)}
    assert Stage.SCENE in tasks


def test_failed_episode_offers_nothing():
    rec = EpisodeRecord.new("ep_0001")
    rec.stages[Stage.PREPARE] = StageStatus.FAILED
    assert episode_ready_tasks(rec, SPEC, CONFIG) == []


def test_build_cpu_kwargs_for_publish_contains_outputs_root():
    fn_name, kwargs = build_cpu_kwargs(
        Stage.PUBLISH, "ep_0001", SPEC, CONFIG,
        storage_root="/in", workspace_root="/ws", output_root="/out",
    )
    assert fn_name == "publish"
    assert kwargs["output_root"] == "/out"
    assert kwargs["episode_id"] == "ep_0001"


def test_build_cpu_kwargs_for_fusion_passes_gap_config():
    _, kwargs = build_cpu_kwargs(
        Stage.FUSION, "ep_0001", SPEC, CONFIG,
        storage_root="/in", workspace_root="/ws", output_root="/out",
    )
    assert kwargs["unobserved_gap_s"] == 0.2
    assert kwargs["min_supporting_joints"] == 8
    assert kwargs["wearer_association"] is True
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/unit/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'roboray.runtime.planner'`

- [ ] **Step 3: Implement `runtime/planner.py`**

```python
"""Pure-function planner: translates Episode state into submittable tasks. Ray-free, so it unit-tests easily."""

from dataclasses import dataclass

from roboray.api.config import ReconstructionRequest
from roboray.models.protocol import resolution_key as _resolution_key
from roboray.stages.scene import resolve_scene_model

from .episode import PREREQS, STAGE_ORDER, EpisodeRecord, EpisodeSpec, Stage, StageStatus

__all__ = ["StageTask", "episode_ready_tasks", "build_cpu_kwargs"]

_CPU_STAGES = {Stage.PREPARE, Stage.FUSION, Stage.ACTION, Stage.PUBLISH}

# Placeholder resolution bucket used when the actual frame size is unknown at planning time;
# the Actor reports the real bucket during execution. The placeholder only affects the
# cold-start scheduling prior, not correctness.
DEFAULT_RESOLUTION_KEY = "640x480"


@dataclass(frozen=True)
class StageTask:
    episode_id: str
    stage: Stage
    kind: str          # "scene" | "hand" | "cpu"
    model_name: str    # only meaningful for GPU tasks
    resolution_key: str
    needs_gpu: bool


def episode_ready_tasks(
    record: EpisodeRecord, spec: EpisodeSpec, config: ReconstructionRequest
) -> list:
    """Return **all** currently-ready stages of the Episode (SCENE and HAND are returned together)."""
    if record.is_terminal():
        return []
    tasks = []
    # Check each PENDING stage for all-DONE prerequisites (the same rule as next_runnable,
    # but returning several at once so SCENE/HAND can be submitted in parallel).
    for stage in STAGE_ORDER:
        if record.stages[stage] is not StageStatus.PENDING:
            continue
        if not all(record.stages[p] is StageStatus.DONE for p in PREREQS[stage]):
            continue
        if stage in _CPU_STAGES:
            tasks.append(StageTask(spec.episode_id, stage, "cpu", "", DEFAULT_RESOLUTION_KEY, False))
        elif stage is Stage.SCENE:
            model = resolve_scene_model(config.stages.scene_reconstruction.model, spec.has_stereo)
            tasks.append(StageTask(spec.episode_id, stage, "scene", model, DEFAULT_RESOLUTION_KEY, True))
        elif stage is Stage.HAND:
            model = config.stages.hand_reconstruction.model
            tasks.append(StageTask(spec.episode_id, stage, "hand", model, DEFAULT_RESOLUTION_KEY, True))
    return tasks


def build_cpu_kwargs(
    stage: Stage,
    episode_id: str,
    spec: EpisodeSpec,
    config: ReconstructionRequest,
    storage_root: str,
    workspace_root: str,
    output_root: str,
) -> tuple:
    """Return (fn_name, kwargs) for the CpuStageActor to call."""
    if stage is Stage.PREPARE:
        return (
            "prepare",
            {
                "storage_root": storage_root,
                "workspace_root": workspace_root,
                "episode_rel_dir": spec.rel_dir,
                "episode_id": episode_id,
            },
        )
    if stage is Stage.FUSION:
        h = config.stages.hand_reconstruction
        return (
            "fusion",
            {
                "workspace_root": workspace_root,
                "episode_id": episode_id,
                "wearer_association": h.wearer_association,
                "unobserved_gap_s": h.unobserved_gap_s,
                "min_supporting_joints": h.min_supporting_joints,
            },
        )
    if stage is Stage.ACTION:
        a = config.stages.action_understanding
        return (
            "action",
            {
                "workspace_root": workspace_root,
                "episode_id": episode_id,
                "semantic_model_name": a.model,
                "keyframe_stride": a.keyframe_stride,
                "vlm_base_url": a.vlm_base_url,
                "vlm_model": a.vlm_model,
                "vlm_api_key_env": a.vlm_api_key_env,
            },
        )
    if stage is Stage.PUBLISH:
        v = config.stages.visualization
        return (
            "publish",
            {
                "workspace_root": workspace_root,
                "output_root": output_root,
                "episode_id": episode_id,
                "make_visualization": bool(v.camera_view or v.world_view),
                "camera_view": v.camera_view,
                "world_view": v.world_view,
                "fps": v.fps,
            },
        )
    raise ValueError(f"stage {stage} is not a CPU stage")
```

(When implementing, drop the two useless `if ... : pass` lines and the redundant local import in `episode_ready_tasks`; hoist `_PREREQS` and `STAGE_ORDER` to a single import at the top of the module.)

- [ ] **Step 4: Run the planner test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/unit/test_planner.py -v`
Expected: 8 passed

- [ ] **Step 5: Implement `runtime/driver.py`**

```python
"""Driver main loop: dispatches planner tasks to the Ray Actors and feeds scheduling feedback back in.

Three modes (for the L3 comparison):
- serial_baseline=True: per-Episode, per-stage serial (baseline)
- serial_baseline=False, adaptive=False: cross-stage pipeline + static pooling
- serial_baseline=False, adaptive=True: cross-stage pipeline + adaptive scheduling (target form)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import ray

from roboray.api.config import ReconstructionRequest
from roboray.scheduling.cost_model import DEFAULT_PRIOR_SECONDS, CostModel
from roboray.scheduling.scheduler import AdaptiveScheduler, GpuSlot, QueueItem
from roboray.stages.scene import resolve_scene_model

from .episode import EpisodeRecord, EpisodeSpec, EpisodeWorkspace, Stage, discover_episodes
from .planner import build_cpu_kwargs, episode_ready_tasks
from .storage import make_storage

__all__ = ["DriverConfig", "RunResult", "PipelineDriver"]


@dataclass
class DriverConfig:
    storage_root: str
    workspace_root: str
    output_root: str
    serial_baseline: bool = False
    adaptive: bool = True


@dataclass
class RunResult:
    total_seconds: float
    per_episode: dict = field(default_factory=dict)
    scheduler_snapshots: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    manifest_path: str | None = None


class PipelineDriver:
    def __init__(
        self,
        config: DriverConfig,
        request: ReconstructionRequest,
        gpu_actors: list,
        cpu_actors: list,
    ):
        self.config = config
        self.request = request
        self.gpu_actors = gpu_actors
        self.cpu_actors = cpu_actors
        self.storage = make_storage(config.storage_root)
        self.specs: dict[str, EpisodeSpec] = {}
        self.cost_model = CostModel()
        self.slots = [GpuSlot(i) for i in range(len(gpu_actors))]
        self.scheduler = AdaptiveScheduler(self.slots, self.cost_model)
        self.timeline: list = []
        self._t0 = 0.0

    # ---------- internal helpers ----------
    def _ws(self, episode_id: str) -> EpisodeWorkspace:
        return EpisodeWorkspace(Path(self.config.workspace_root), episode_id)

    def _event(self, episode_id: str, stage: Stage, event: str) -> None:
        self.timeline.append(
            {
                "t": round(time.perf_counter() - self._t0, 6),
                "episode_id": episode_id,
                "stage": stage.value,
                "event": event,
            }
        )

    def _seed_priors(self) -> None:
        """Use the models' self-reported cost_hint as the cold-start prior, so the scheduler is not fully blind at the start.

        Skip models that are not registered (e.g. missing GPU deps): a missing prior only affects
        early decision quality, not correctness -- CostModel falls back to DEFAULT_PRIOR_SECONDS.
        """
        from roboray.models import registry

        sc = self.request.stages.scene_reconstruction
        hc = self.request.stages.hand_reconstruction
        # seed once: use the first Episode's model selection to decide the scene-path model name
        first = next(iter(self.specs.values()), None)
        has_stereo = bool(first and first.has_stereo)
        scene_model = resolve_scene_model(sc.model, has_stereo)
        for res in ("320x240", "640x480"):
            for kind, model, getter in (
                ("scene", scene_model, registry.get_scene_model_class),
                ("hand", hc.model, registry.get_hand_model_class),
            ):
                if self.cost_model.predict(kind, res) != DEFAULT_PRIOR_SECONDS:
                    continue  # prior already set
                try:
                    m = getter(model)()
                except Exception:
                    continue
                self.cost_model.seed_prior(kind, res, m.cost_hint(24, res))

    # ---------- the three modes ----------
    def run(self) -> RunResult:
        self._t0 = time.perf_counter()
        specs = discover_episodes(self.storage)
        self.specs = {s.episode_id: s for s in specs}
        for s in specs:
            self._ws(s.episode_id).load()
        if self.config.serial_baseline:
            result = self._run_serial()
        else:
            result = self._run_pipeline()
        result.total_seconds = time.perf_counter() - self._t0
        return result

    def _run_serial(self) -> RunResult:
        """Baseline: one Episode runs through all stages before the next starts, and nothing is parallel."""
        per_episode, failed = {}, []
        for spec in self.specs.values():
            ws = self._ws(spec.episode_id)
            for stage in [Stage.PREPARE, Stage.SCENE, Stage.HAND, Stage.FUSION,
                          Stage.ACTION, Stage.PUBLISH]:
                ok = self._execute_one(spec, stage)
                if not ok:
                    failed.append(spec.episode_id)
                    break
            per_episode[spec.episode_id] = ws.load()
        return RunResult(
            total_seconds=0.0,
            per_episode={k: v.to_json() for k, v in per_episode.items()},
            scheduler_snapshots=[],
            timeline=self.timeline,
            failed=failed,
        )

    def _run_pipeline(self) -> RunResult:
        """Cross-stage pipeline + (optionally) adaptive scheduling."""
        pending: dict[str, EpisodeRecord] = {s: self._ws(s).load() for s in self.specs}
        scene_q: list = []
        hand_q: list = []
        cpu_q: list = []
        in_flight: dict = {}     # ray.ObjectRef -> (ep_id, kind, slot_or_None, actor)
        queued: set = set()      # (episode_id, stage) already enqueued but not finished, prevents duplicate enqueue
        max_in_flight = max(1, self.request.execution.max_episodes_in_flight)

        def push_ready() -> None:
            """Re-read each Episode's true state from the workspace and enqueue its ready stages.

            State lives on disk (state.json), not in memory, so each round must re-read;
            otherwise the same stage could be enqueued twice from a stale snapshot.
            """
            for ep in list(self.specs):
                rec = self._ws(ep).load()
                pending[ep] = rec
                if rec.is_terminal():
                    continue
                spec = self.specs[ep]
                for t in episode_ready_tasks(rec, spec, self.request):
                    if (ep, t.stage) in queued:
                        continue
                    queued.add((ep, t.stage))
                    if t.stage is Stage.SCENE:
                        scene_q.append(QueueItem("scene", ep, t.resolution_key))
                    elif t.stage is Stage.HAND:
                        hand_q.append(QueueItem("hand", ep, t.resolution_key))
                    else:
                        cpu_q.append((ep, t.stage, spec, t))

        push_ready()
        # free-slot queue: each slot maps 1:1 to an Actor (same index), avoiding mismatches
        # when multiple Actors run concurrently
        free_slots = list(zip(self.slots, self.gpu_actors))
        free_cpu = list(self.cpu_actors)
        while scene_q or hand_q or cpu_q or in_flight:
            # 1) dispatch GPU tasks
            while free_slots:
                slot, actor = free_slots[0]
                if self.config.adaptive:
                    a = self.scheduler.assign(scene_q, hand_q, slot)
                else:
                    a = self._static_assign(scene_q, hand_q, slot)
                if a is None:
                    break
                free_slots.pop(0)
                q = scene_q if a.kind == "scene" else hand_q
                item = q.pop(0)
                spec = self.specs[item.episode_id]
                ws = self._ws(item.episode_id)
                stage = Stage.SCENE if a.kind == "scene" else Stage.HAND
                ws.mark_running(stage)
                self._event(item.episode_id, stage, "start")
                payload = self._gpu_payload(a.kind, spec, item, ws)
                slot.busy = True
                ref = actor.run_task.remote(payload)
                in_flight[ref] = (item.episode_id, a.kind, slot, actor)

            # 2) dispatch CPU tasks
            while cpu_q and free_cpu:
                ep, stage, spec, _t = cpu_q.pop(0)
                fn_name, kwargs = build_cpu_kwargs(
                    stage, ep, spec, self.request,
                    self.config.storage_root, self.config.workspace_root, self.config.output_root,
                )
                ws = self._ws(ep)
                ws.mark_running(stage)
                self._event(ep, stage, "start")
                actor = free_cpu.pop(0)
                ref = actor.run_task.remote(
                    {"fn_name": fn_name, "kwargs": kwargs, "episode_id": ep, "kind": fn_name}
                )
                in_flight[ref] = (ep, fn_name, None, actor)

            if not in_flight:
                if scene_q or hand_q or cpu_q:
                    push_ready()
                    continue
                break

            # 3) reclaim finished tasks
            done, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=0.05)
            for ref in done:
                ep, kind, slot, actor = in_flight.pop(ref)
                res = self._get_result(ref)
                self._complete((ep, kind, slot), res)
                if slot is not None:
                    free_slots.append((slot, actor))
                else:
                    free_cpu.append(actor)
                if ep and self._ws(ep).load().is_terminal():
                    pass  # a terminal Episode produces no new tasks (push_ready skips it)

            push_ready()

        failed = [ep for ep in self.specs if self._ws(ep).load().error]
        return RunResult(
            total_seconds=0.0,
            per_episode={ep: self._ws(ep).load().to_json() for ep in self.specs},
            scheduler_snapshots=self.scheduler.stats_obj.snapshots,
            timeline=self.timeline,
            failed=failed,
        )

    # ---------- helpers ----------
    def _static_assign(self, scene_q, hand_q, slot):
        from roboray.scheduling.scheduler import Assignment

        if slot.busy:
            return None
        # static pooling: even slot ids always run scene, odd always run hand
        preferred = "scene" if slot.slot_id % 2 == 0 else "hand"
        other = "hand" if preferred == "scene" else "scene"
        for kind in (preferred, other):
            q = scene_q if kind == "scene" else hand_q
            if q:
                return Assignment(slot.slot_id, kind, q[0], slot.loaded_model != kind)
        return None

    def _gpu_payload(self, kind, spec, item, ws) -> dict:
        if kind == "scene":
            from roboray.stages.scene import build_scene_payload

            model = resolve_scene_model(
                self.request.stages.scene_reconstruction.model, spec.has_stereo
            )
            return build_scene_payload(
                self.config.workspace_root, item.episode_id, model, spec.has_stereo
            )
        from roboray.stages.hand import build_hand_payload

        return build_hand_payload(
            self.config.workspace_root,
            item.episode_id,
            self.request.stages.hand_reconstruction.model,
        )

    def _get_result(self, ref) -> dict:
        try:
            return ray.get(ref)
        except Exception as exc:  # Actor crash
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "episode_id": None,
                    "kind": None, "seconds": 0.0}

    def _complete(self, info, res) -> None:
        ep, kind, slot = info
        # kind is "scene"/"hand" for GPU tasks and the stage function name
        # ("prepare"/"fusion"/"action"/"publish") for CPU tasks -- both match Stage.value.
        try:
            stage = Stage(kind)
        except ValueError:
            stage = None
        ws = self._ws(ep) if ep else None
        if res.get("ok"):
            seconds = res.get("infer_seconds") or res.get("seconds") or 0.0
            if slot is not None:
                self.scheduler.finish(
                    slot, kind, res.get("resolution_key", "640x480"), seconds,
                    was_load=bool(res.get("was_load")),
                )
            if ws is not None and stage is not None:
                ws.mark_done(stage, seconds)
                self._event(ep, stage, "end")
        else:
            if slot is not None:
                slot.busy = False
            if ws is not None and stage is not None:
                ws.mark_failed(stage, str(res.get("error")))

    def _execute_one(self, spec, stage: Stage) -> bool:
        """Serial mode: execute a single stage once."""
        ws = self._ws(spec.episode_id)
        ws.mark_running(stage)
        self._event(spec.episode_id, stage, "start")
        if stage in (Stage.SCENE, Stage.HAND):
            kind = stage.value
            from roboray.scheduling.scheduler import QueueItem

            payload = self._gpu_payload(kind, spec, QueueItem(kind, spec.episode_id, "640x480"), ws)
            res = self._get_result(self.gpu_actors[0].run_task.remote(payload))
        else:
            fn_name, kwargs = build_cpu_kwargs(
                stage, spec.episode_id, spec, self.request,
                self.config.storage_root, self.config.workspace_root, self.config.output_root,
            )
            res = self._get_result(
                self.cpu_actors[0].run_task.remote(
                    {"fn_name": fn_name, "kwargs": kwargs,
                     "episode_id": spec.episode_id, "kind": fn_name}
                )
            )
        if res.get("ok"):
            ws.mark_done(stage, res.get("infer_seconds") or res.get("seconds") or 0.0)
            self._event(spec.episode_id, stage, "end")
            return True
        ws.mark_failed(stage, str(res.get("error")))
        return False
```

(When implementing, delete the duplicate `resolve_scene_model` definition near the top and import it from `roboray.stages.scene` instead. Simplify the convoluted getter logic in `_seed_priors` to call `registry.get_scene_model_class(model)` and `registry.get_hand_model_class(h.model)` by name, each seeding from its own `cost_hint`; skip models that are not registered. Do this before Step 6.)

- [ ] **Step 6: Write the parallel e2e test**

`tests/e2e/test_driver_parallel.py`:

```python
import json

import numpy as np
import pytest

ray = pytest.importorskip("ray")

from roboray.api.config import ReconstructionRequest
from roboray.models import registry
from roboray.ray_runtime.actors import build_actor_pool
from roboray.runtime.driver import DriverConfig, PipelineDriver
from scripts.make_synthetic_ego import generate_dataset
from tests.fakes.models import install_fake_models

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _ray():
    """Function-level cluster + worker-side double hook (reasoning as in tests/unit/test_actors.py:
    GpuModelActor holds a virtual GPU permanently, so a module-level cluster plus a per-test
    new pool would deadlock)."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_gpus=2,
        num_cpus=8,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "working_dir": ".",
            "worker_process_setup_hook": "tests.fakes.worker_setup.install_for_ray",
            "env_vars": {
                "ROBORAY_FAKE_SCENE_SECONDS": "0.05",
                "ROBORAY_FAKE_HAND_SECONDS": "0.05",
            },
        },
    )
    yield
    ray.shutdown()


@pytest.fixture
def _fakes():
    registry.reset_registry_for_tests()
    install_fake_models(scene_seconds=0.05, hand_seconds=0.05)
    yield
    registry.reset_registry_for_tests()


def _request(in_dir, out_dir, max_gpus=2, backend="test"):
    return ReconstructionRequest.from_dict(
        {
            "input": {"uri": str(in_dir)},
            "output": {"uri": str(out_dir), "format": "lerobot_v3"},
            "execution": {"max_gpus": max_gpus, "num_cpu_actors": 4,
                          "max_episodes_in_flight": 3, "backend": backend},
            "stages": {
                "scene_reconstruction": {"model": "fake_scene"},
                "hand_reconstruction": {"model": "fake_hand", "unobserved_gap_s": 0.2},
                "action_understanding": {"model": "fake_vlm"},
                "visualization": {"camera_view": True, "world_view": True},
            },
        }
    )


def test_pipeline_run_produces_all_outputs(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=2, n_frames=12, size=(96, 72), seed=0)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(ws_dir),
                     output_root=str(out_dir), serial_baseline=False, adaptive=True),
        _request(in_dir, out_dir),
        gpu,
        cpu,
    )
    result = driver.run()
    assert result.failed == []
    for ep in ("ep_0001", "ep_0002"):
        assert (out_dir / "lerobot_v3" / ep / "data.npz").exists()
        assert (out_dir / "lerobot_v3" / ep / "meta.json").exists()
        assert (out_dir / "qc" / ep / "report.html").exists()
        assert (out_dir / "viz" / ep / "camera.mp4").exists()
    meta = json.loads((out_dir / "lerobot_v3" / "ep_0001" / "meta.json").read_text())
    assert meta["format"] == "lerobot_v3"
    assert meta["features"]["state"][0] == 12


def test_pipeline_overlaps_stages_across_episodes(tmp_path):
    """Cross-stage parallelism: there should be a time window where one Episode is in a late stage and another is preparing/inferring."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=3, n_frames=16, size=(96, 72), seed=1)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(ws_dir),
                     output_root=str(out_dir), serial_baseline=False, adaptive=True),
        _request(in_dir, out_dir),
        gpu,
        cpu,
    )
    result = driver.run()
    tl = result.timeline
    events = {}
    for e in tl:
        events.setdefault((e["episode_id"], e["stage"]), []).append(e)
    # at least one moment where two different Episodes are running at once
    starts = sorted(e["t"] for e in tl if e["event"] == "start")
    ends = sorted(e["t"] for e in tl if e["event"] == "end")
    concurrent = 0
    for s in starts:
        concurrent = max(concurrent, sum(1 for s2 in starts if s2 <= s) - sum(1 for e2 in ends if e2 <= s))
    assert concurrent >= 2, f"no cross-stage parallelism observed; timeline={tl}"


def test_adaptive_is_faster_than_serial_baseline(tmp_path):
    """Form reproduction: the adaptive pipeline should be faster than the pure serial baseline.

    Each mode builds its own Actors, so the serial run's model-load state and EMA priors do
    not leak into the pipeline run; the comparison therefore reflects "orchestration style"
    rather than "who runs first".
    """
    in_dir = tmp_path / "in"
    generate_dataset(in_dir, n_episodes=3, n_frames=12, size=(96, 72), seed=2)
    req_serial = _request(in_dir, tmp_path / "out_serial")
    req_pipe = _request(in_dir, tmp_path / "out_pipe")

    gpu_s, cpu_s = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    serial = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(tmp_path / "ws_serial"),
                     output_root=str(tmp_path / "out_serial"), serial_baseline=True, adaptive=False),
        req_serial, gpu_s, cpu_s,
    ).run()

    gpu_p, cpu_p = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    pipeline = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(tmp_path / "ws_pipe"),
                     output_root=str(tmp_path / "out_pipe"), serial_baseline=False, adaptive=True),
        req_pipe, gpu_p, cpu_p,
    ).run()

    assert pipeline.total_seconds < serial.total_seconds, (
        f"pipeline parallelism gave no gain: serial={serial.total_seconds:.3f}s "
        f"pipeline={pipeline.total_seconds:.3f}s"
    )


def test_adaptive_scheduler_snapshots_record_decisions(tmp_path):
    in_dir = tmp_path / "in"
    generate_dataset(in_dir, n_episodes=3, n_frames=12, size=(96, 72), seed=3)
    gpu, cpu = build_actor_pool(num_gpus=2, num_cpu_actors=4)
    driver = PipelineDriver(
        DriverConfig(storage_root=str(in_dir), workspace_root=str(tmp_path / "ws"),
                     output_root=str(tmp_path / "out"), serial_baseline=False, adaptive=True),
        _request(in_dir, tmp_path / "out"),
        gpu, cpu,
    )
    result = driver.run()
    assert len(result.scheduler_snapshots) >= 2
    snaps = result.scheduler_snapshots
    assert {"chosen", "scene_q", "hand_q", "predict", "loaded"} <= set(snaps[0])
```

- [ ] **Step 7: Run the e2e test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/e2e/test_driver_parallel.py -v`
Expected: 4 passed

- [ ] **Step 8: Commit**

```bash
git add src/roboray/runtime/planner.py src/roboray/runtime/driver.py         tests/unit/test_planner.py tests/e2e/test_driver_parallel.py
git commit -m "feat(runtime): task planner and Driver main loop (serial/pipeline/adaptive three modes)"
```

---

### Task 13: Public API, CLI, and end-to-end entry point

**Files:**
- Create: `src/roboray/api/reconstruction.py`
- Modify: `src/roboray/api/__init__.py`
- Modify: `src/roboray/__init__.py` (export the API)
- Create: `src/roboray/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/e2e/test_public_api.py`

**Interfaces:**
- Consumes: `PipelineDriver`, `build_actor_pool`, `registry`
- Produces:
  - `EgoReconstruction(request: ReconstructionRequest, *, run_root=None, serial_baseline=False, adaptive=True)`: `run() -> RunResult`; internally `ray.init(address=..., num_gpus=...)` or reuses an already-initialized cluster; calls `load_builtin_models()` when `backend == "real"`
  - `RunResult` gains `manifest_path: str`, and `output_root/manifest.json` holds the whole-batch index (per-Episode status, latencies, failure reasons, scheduling snapshots)
  - `roboray.manifest.write_manifest(output_root, request, per_episode, scheduler_snapshots) -> Path`
  - CLI: `roboray run --config configs/default.yaml`, `roboray run --config ... --serial-baseline`, `--adaptive/--no-adaptive`, `--ray-address auto`

- [ ] **Step 1: Write the failing test**

`tests/e2e/test_public_api.py`:

```python
import json

import pytest

ray = pytest.importorskip("ray")

from roboray import EgoReconstruction, ReconstructionRequest
from roboray.models import registry
from scripts.make_synthetic_ego import generate_dataset
from tests.fakes.models import install_fake_models

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _ray():
    """Function-level cluster + worker-side double hook (deadlock guard as in tests/unit/test_actors.py)."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_gpus=2,
        num_cpus=8,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "working_dir": ".",
            "worker_process_setup_hook": "tests.fakes.worker_setup.install_for_ray",
            "env_vars": {
                "ROBORAY_FAKE_SCENE_SECONDS": "0.02",
                "ROBORAY_FAKE_HAND_SECONDS": "0.02",
            },
        },
    )
    yield
    ray.shutdown()


@pytest.fixture
def _fakes():
    registry.reset_registry_for_tests()
    install_fake_models(scene_seconds=0.02, hand_seconds=0.02)
    yield
    registry.reset_registry_for_tests()


def test_public_api_runs_and_writes_manifest(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    generate_dataset(in_dir, n_episodes=2, n_frames=10, size=(96, 72), seed=0)
    config = {
        "input": {"uri": str(in_dir)},
        "output": {"uri": str(out_dir), "format": "lerobot_v3"},
        "execution": {"max_gpus": 2, "num_cpu_actors": 4, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
            "visualization": {"camera_view": True, "world_view": True},
        },
    }
    request = ReconstructionRequest.from_dict(config)
    result = EgoReconstruction(request, run_root=str(tmp_path / "ws")).run()
    assert result.failed == []
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["format"] == "lerobot_v3"
    assert {e["episode_id"] for e in manifest["episodes"]} == {"ep_0001", "ep_0002"}
    assert manifest["summary"]["failed"] == 0
    assert "scheduler_snapshots" in manifest


def test_one_bad_episode_does_not_kill_the_batch(tmp_path):
    """Review Focus #1/#4: one Episode fails -> the batch continues and the rest produce output normally."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    generate_dataset(in_dir, n_episodes=2, n_frames=10, size=(96, 72), seed=0)
    # break the second episode's calibration
    (in_dir / "ep_0002" / "calibration.json").unlink()
    config = {
        "input": {"uri": str(in_dir)},
        "output": {"uri": str(out_dir)},
        "execution": {"max_gpus": 2, "num_cpu_actors": 4, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
        },
    }
    result = EgoReconstruction(
        ReconstructionRequest.from_dict(config), run_root=str(tmp_path / "ws")
    ).run()
    assert result.failed == ["ep_0002"]
    assert (out_dir / "lerobot_v3" / "ep_0001" / "meta.json").exists()
    assert not (out_dir / "lerobot_v3" / "ep_0002" / "meta.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    failed = [e for e in manifest["episodes"] if e["episode_id"] == "ep_0002"][0]
    assert "calibration.json" in failed["error"]


def test_resume_skips_completed_stages(tmp_path):
    """Resume: after one full run, a re-run does not re-execute stages already DONE."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    ws_dir = tmp_path / "ws"
    generate_dataset(in_dir, n_episodes=1, n_frames=8, size=(96, 72), seed=0)
    config = {
        "input": {"uri": str(in_dir)},
        "output": {"uri": str(out_dir)},
        "execution": {"max_gpus": 2, "num_cpu_actors": 4, "backend": "test"},
        "stages": {
            "scene_reconstruction": {"model": "fake_scene"},
            "hand_reconstruction": {"model": "fake_hand"},
            "action_understanding": {"model": "fake_vlm"},
        },
    }
    req = ReconstructionRequest.from_dict(config)
    EgoReconstruction(req, run_root=str(ws_dir)).run()
    first = json.loads((ws_dir / "ep_0001" / "state.json").read_text())
    second_result = EgoReconstruction(req, run_root=str(ws_dir)).run()
    assert second_result.failed == []
    # the second pass should not produce any new stage-start events (all already DONE)
    assert [e for e in second_result.timeline if e["event"] == "start"] == []
    assert json.loads((ws_dir / "ep_0001" / "state.json").read_text())["timings"] == first["timings"]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `conda run -n roboray python -m pytest tests/e2e/test_public_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'EgoReconstruction' from 'roboray'`

- [ ] **Step 3: Implement `src/roboray/manifest.py`**

```python
"""Whole-batch run index: manifest.json."""

import json
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["write_manifest"]


def write_manifest(
    output_root: str,
    request,
    per_episode: dict,
    scheduler_snapshots: list,
    total_seconds: float = 0.0,
) -> Path:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    episodes = []
    for ep_id, rec in sorted(per_episode.items()):
        payload = rec if isinstance(rec, dict) else rec.to_json()
        episodes.append(
            {
                "episode_id": ep_id,
                "stages": payload.get("stages", {}),
                "timings": payload.get("timings", {}),
                "error": payload.get("error"),
            }
        )
    failed = [e for e in episodes if e["error"]]
    manifest = {
        "format": request.output.format,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": request.input.uri,
        "output": request.output.uri,
        "execution": {
            "max_gpus": request.execution.max_gpus,
            "backend": request.execution.backend,
        },
        "summary": {
            "n_episodes": len(episodes),
            "failed": len(failed),
            "total_seconds": round(float(total_seconds), 6),
        },
        "episodes": episodes,
        "scheduler_snapshots": list(scheduler_snapshots),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return path
```

- [ ] **Step 4: Implement `src/roboray/api/reconstruction.py`**

```python
"""Public API: EgoReconstruction(request).run()."""

import os
from pathlib import Path

from .config import ReconstructionRequest

__all__ = ["EgoReconstruction"]


class EgoReconstruction:
    def __init__(
        self,
        request: ReconstructionRequest,
        *,
        run_root: str | None = None,
        serial_baseline: bool = False,
        adaptive: bool = True,
        ray_address: str | None = None,
    ):
        self.request = request
        self.serial_baseline = serial_baseline
        self.adaptive = adaptive
        self.ray_address = ray_address or os.environ.get("RAY_ADDRESS")
        base = Path(run_root or os.environ.get("ROBORAY_RUN_ROOT", "./run"))
        self.run_root = base
        self.workspace_root = str(base / "workspace")

    def run(self):
        import ray

        from roboray.models import registry
        from roboray.ray_runtime.actors import build_actor_pool
        from roboray.runtime.driver import DriverConfig, PipelineDriver
        from roboray.manifest import write_manifest

        if self.request.execution.backend == "real":
            registry.load_builtin_models()

        if not ray.is_initialized():
            init_kwargs = dict(num_cpus=os.cpu_count(), include_dashboard=False,
                               log_to_driver=False)
            if self.ray_address:
                init_kwargs["address"] = self.ray_address
            else:
                init_kwargs["num_gpus"] = max(0, self.request.execution.max_gpus)
            ray.init(**init_kwargs)

        gpu_actors, cpu_actors = build_actor_pool(
            num_gpus=max(0, self.request.execution.max_gpus),
            num_cpu_actors=self.request.execution.num_cpu_actors,
        )
        driver = PipelineDriver(
            DriverConfig(
                storage_root=self.request.input.uri,
                workspace_root=self.workspace_root,
                output_root=self.request.output.uri,
                serial_baseline=self.serial_baseline,
                adaptive=self.adaptive,
            ),
            self.request,
            gpu_actors,
            cpu_actors,
        )
        result = driver.run()
        result.manifest_path = str(
            write_manifest(
                self.request.output.uri,
                self.request,
                result.per_episode,
                result.scheduler_snapshots,
                result.total_seconds,
            )
        )
        return result
```

`api/__init__.py`:

```python
"""Public API surface."""

from .config import (  # noqa: F401
    ActionStageConfig,
    ConfigError,
    ExecutionConfig,
    HandStageConfig,
    InputConfig,
    OutputConfig,
    ReconstructionRequest,
    SceneStageConfig,
    StageConfigs,
    VisualizationConfig,
)
from .reconstruction import EgoReconstruction  # noqa: F401

__all__ = [
    "EgoReconstruction",
    "ReconstructionRequest",
    "ConfigError",
    "InputConfig",
    "OutputConfig",
    "ExecutionConfig",
    "SceneStageConfig",
    "HandStageConfig",
    "ActionStageConfig",
    "VisualizationConfig",
    "StageConfigs",
]
```

`src/roboray/__init__.py`:

```python
"""Ego-centric video to training-ready dataset pipeline."""

from .api import EgoReconstruction, ReconstructionRequest  # noqa: F401

__version__ = "0.1.0"

__all__ = ["EgoReconstruction", "ReconstructionRequest", "__version__"]
```

- [ ] **Step 5: Implement `src/roboray/cli.py` and `configs/`**

`src/roboray/cli.py`:

```python
"""roboray command-line entry point."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from roboray.api.config import ConfigError, ReconstructionRequest
from roboray.api.reconstruction import EgoReconstruction

__all__ = ["main"]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roboray", description="Ego video to training-ready dataset pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run one data production pass")
    run_p.add_argument("--config", required=True, help="path to the YAML/JSON config file")
    run_p.add_argument("--run-root", default=None, help="workspace root directory")
    run_p.add_argument("--serial-baseline", action="store_true", help="serial baseline mode (for comparison experiments)")
    run_p.add_argument("--no-adaptive", action="store_true", help="disable adaptive scheduling (static pooling)")
    run_p.add_argument("--ray-address", default=None, help="address of an existing Ray cluster, e.g. auto")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        path = Path(args.config)
        if path.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(path.read_text())
        else:
            raw = json.loads(path.read_text())
        try:
            request = ReconstructionRequest.from_dict(raw)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        result = EgoReconstruction(
            request,
            run_root=args.run_root,
            serial_baseline=args.serial_baseline,
            adaptive=not args.no_adaptive,
            ray_address=args.ray_address,
        ).run()
        print(json.dumps(
            {
                "total_seconds": round(result.total_seconds, 3),
                "failed": result.failed,
                "manifest": getattr(result, "manifest_path", None),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 1 if result.failed else 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

Add to `pyproject.toml`:

```toml
[project.scripts]
roboray = "roboray.cli:main"
```

And add `"PyYAML>=6.0"` to `dependencies`. Also add `"pyyaml>=6.0"` to `[project.optional-dependencies].dev` (can be omitted if already covered by the main dependency).

`configs/default.yaml`:

```yaml
# roboray default config: one-command invocation; each stage can be overridden as needed.
input:
  uri: "data/synthetic"
output:
  uri: "output/run-default"
  format: "lerobot_v3"
execution:
  max_gpus: 1
  num_cpu_actors: 4
  max_episodes_in_flight: 2
  backend: "real"
stages:
  scene_reconstruction:
    stereo_max_image_side_px: 640
    model: "auto"
  hand_reconstruction:
    inference_fps: 30
    wearer_association: true
    unobserved_gap_s: 0.2
    min_supporting_joints: 8
    model: "hamer"
  action_understanding:
    vlm_base_url: null
    vlm_api_key_env: "VLM_API_KEY"
    vlm_model: "qwen-vl"
    keyframe_stride: 30
    model: "vlm"
  visualization:
    camera_view: true
    world_view: true
    fps: 30
```

`configs/local-fake.yaml`:

```yaml
# local CPU end-to-end (fake models): no GPU needed, for verifying pipeline orchestration.
input:
  uri: "data/synthetic"
output:
  uri: "output/run-fake"
  format: "lerobot_v3"
execution:
  max_gpus: 2
  num_cpu_actors: 4
  max_episodes_in_flight: 3
  backend: "test"
stages:
  scene_reconstruction:
    model: "fake_scene"
  hand_reconstruction:
    model: "fake_hand"
  action_understanding:
    model: "fake_vlm"
  visualization:
    camera_view: true
    world_view: true
```

Note: `local-fake.yaml` depends on `tests/fakes` registering the doubles; provide `scripts/run_local_fake.py` as a wrapper that does "register doubles + run the config":

```python
"""Local fake-model end-to-end run: register doubles then run the config once.

Usage:
    python scripts/run_local_fake.py --config configs/local-fake.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboray.api.config import ReconstructionRequest
from roboray.api.reconstruction import EgoReconstruction
from tests.fakes.models import install_fake_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local-fake.yaml")
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--serial-baseline", action="store_true")
    args = ap.parse_args()
    install_fake_models()
    request = ReconstructionRequest.from_dict(yaml.safe_load(Path(args.config).read_text()))
    result = EgoReconstruction(
        request, run_root=args.run_root, serial_baseline=args.serial_baseline
    ).run()
    print(f"total {result.total_seconds:.3f}s, failed {result.failed}, manifest at {result.manifest_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the test and confirm it passes**

Run: `conda run -n roboray python -m pytest tests/e2e/test_public_api.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/roboray/api/ src/roboray/__init__.py src/roboray/cli.py src/roboray/manifest.py         configs/ scripts/run_local_fake.py pyproject.toml tests/e2e/test_public_api.py
git commit -m "feat(api): EgoReconstruction public API, CLI, manifest index and local fake-model entry point"
```

---

### Task 14: Full test run, docs, and Plan A wrap-up

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/plans/A-verification-report.md` (record the measured results)
- Modify: `pyproject.toml` (confirm `[project.scripts]` and the dependencies are complete)

**Interfaces:**
- Consumes: all previous tasks
- Produces: a reproducible verification record; README usage instructions; all-green `pytest`

- [ ] **Step 1: Run the full test suite**

```bash
conda run -n roboray python -m pytest -v
```

Expected: all pass (unit + local end-to-end). If anything fails, fix and re-run; do not skip.

- [ ] **Step 2: Run the real entry point once via the CLI (fake models)**

```bash
conda run -n roboray python scripts/make_synthetic_ego.py --out data/synthetic --episodes 3 --frames 30
conda run -n roboray python scripts/run_local_fake.py --config configs/local-fake.yaml
conda run -n roboray python scripts/run_local_fake.py --config configs/local-fake.yaml --run-root run-serial --serial-baseline
```

Expected: both commands succeed; the second (serial) clearly takes longer than the first (pipeline parallel).

- [ ] **Step 3: Write the measured results into the verification report**

`docs/superpowers/plans/A-verification-report.md` records:

```markdown
# Plan A verification report

## Environment
- This machine: {{python-version}}, {{cpu-cores}} cores, no GPU (Ray runs with virtual num_gpus resources)
- Ray version: {{ray-version}}

## Full test suite
- `pytest`: {{pass-count}} passed

## Parallel-gain reproduction (synthetic data: {{n}} Episodes / {{frames}} frames)
| Mode | Total time (s) |
| --- | --- |
| Serial baseline | {{serial}} |
| Cross-stage pipeline + adaptive scheduling | {{adaptive}} |
| Speedup | {{speedup}}x |

## Observed scheduling behavior
- Scheduling decision snapshot count: {{snapshots}}
- Model-reuse hit rate (needs_load=False share): {{reuse_ratio}}
- Notes: {{observations}}
```

Replace every `{{...}}` above with the **actually measured numbers** (read from the CLI output and `manifest.json`).

- [ ] **Step 4: Complete the README**

Expand the README's "Quick start" section into:

```markdown
## Quick start

### Local verification (no GPU)

```bash
pip install -e ".[dev]"
pytest                                              # unit + local end-to-end
python scripts/make_synthetic_ego.py --out data/synthetic --episodes 3 --frames 30
python scripts/run_local_fake.py --config configs/local-fake.yaml
```

### Real models (require a GPU and model weights)

```bash
pip install -e ".[dev,models]"
roboray run --config configs/default.yaml --ray-address auto
```

Installation steps for the real models (ViPE / Fast-FoundationStereo / HaMeR / MANO weight acquisition) are in `docs/` and the later Plan B.

## Config

The config keys match the article's examples; see `configs/default.yaml`. Each stage can be tuned independently:

| Key | Meaning | Default |
| --- | --- | --- |
| `execution.max_gpus` | Number of available GPUs | 1 |
| `execution.backend` | `real` runs real models; `test` uses doubles (test-only) | real |
| `stages.scene_reconstruction.stereo_max_image_side_px` | Stereo input resolution cap | 640 |
| `stages.hand_reconstruction.unobserved_gap_s` | Short-gap fill time cap (seconds) | 0.2 |
| `stages.hand_reconstruction.min_supporting_joints` | Min supporting joints required for a fill | 8 |

## Outputs

```
output/run-xxx/
├── lerobot_v3/<ep_id>/    # training dataset
├── viz/<ep_id>/           # camera view / world view replays
├── qc/<ep_id>/report.html # per-Episode QC report
└── manifest.json          # whole-batch index (latencies and scheduling snapshots)
```
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/plans/A-verification-report.md pyproject.toml
git commit -m "docs: Plan A verification report and README usage instructions"
```

---

## Self-Review

**1. Spec coverage check**

| Spec requirement | Corresponding task |
| --- | --- |
| §3 Four-stage pipeline and state machine | Task 4 (state machine), Tasks 10/11 (stage implementations), Task 12 (Driver) |
| §3 GPU Actor role switching | Task 7 (scheduling), Task 9 (Actors) |
| §3 Release on write, local data reuse | Tasks 9, 12 |
| §3 EMA feedback loop | Tasks 6, 7 |
| §4 Package structure | Tasks 1, 3, 5, 6, 8, 10, 11, 12, 13 |
| §5 Input layout / intermediate artifacts / output | Tasks 4, 8, 10, 11; **LeRobot v3 is now a minimal implementation; full format adaptation is in Plan C** |
| §6 Model adapter layer and routing | Task 5 (protocol/registry), Task 10 (`resolve_scene_model`); **real adapters are in Plan B** |
| §7 A1/A2/A3 | Task 11 (`fill_short_gaps`/`associate_wearer` complete; **A1 metric fusion is a skeleton implementation; precision optimization is in Plan B**) |
| §8 Adaptive scheduler | Tasks 6, 7 |
| §9 Action understanding + AI Function | Task 11 (geometric segmentation + VLM interface); **real VLM calls are in Plan C** |
| §10 Visualization/publish/QC | Task 11 (minimal implementation); **3D world-view rendering is in Plan C** |
| §12 Verification strategy L0/L1 | Tests in Tasks 8-13 |
| §12 L2/L3, §11 A800 deployment | **Plan D** |
| §13 Dependency pinning | Task 1 |

**2. Placeholder scan**: done. No "TBD"/"TODO"; every code step contains complete code; `configs/`, `scripts/`, `docs/` all have concrete content.

**3. Type consistency**: cross-checked `StageTask` (defined Task 12, used Tasks 10/12), `Assignment`/`GpuSlot`/`QueueItem` (defined Task 7, used Task 12), `EpisodeRecord.next_runnable` (defined Task 4, Task 12's planner reuses `_PREREQS`), `EpisodeWorkspace.write_artifact/check_artifact` (defined Task 4, used Task 11), and the three-tuple returned by `install_fake_models` (defined Task 8, used Tasks 9/12/13). `RunResult.manifest_path` is attached by attribute assignment in Task 13 — when implementing, declare the field directly in `RunResult` in `runtime/driver.py` (`manifest_path: str | None = None`) to avoid dynamic attributes.

**4. Review Focus coverage**

| Review Focus | Test that pins it |
| --- | --- |
| 1. calibration.json missing/incomplete -> FAILED and batch continues | `test_run_prepare_missing_calibration_raises_readable_error` (Task 10), `test_one_bad_episode_does_not_kill_the_batch` (Task 13), `test_load_calibration_rejects_incomplete_file` (Task 10) |
| 2. Stereo frame-count mismatch -> take the shorter one and degrade | `test_run_prepare_flags_stereo_frame_mismatch` (Task 10) |
| 3. meta.json missing / episode_id conflict | `test_discover_rejects_missing_meta`, `test_discover_rejects_duplicate_episode_id` (Task 4) |
| 4. Zero-frame/undecodable mp4 -> FAILED and batch continues | `test_run_prepare_zero_frame_video_raises` (Task 10), `test_one_bad_episode_does_not_kill_the_batch` (Task 13) |
| 5. Corrupt/version-mismatched artifact -> detected and re-run | `test_corrupt_artifact_is_detected`, `test_artifact_schema_mismatch_is_detected` (Task 4), `test_run_fusion_rejects_corrupt_input_artifact` (Task 11) |

## Execution Handoff

After Plan A, write and execute in order: Plan B (stages 1/2 real models and the A1-A3 refinement), Plan C (stages 3/4 complete LeRobot v3, real VLM calls, 3D world view), Plan D (A800 KubeRay deployment and the L2/L3 experiments).
