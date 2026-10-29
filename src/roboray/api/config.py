"""Reconstruction request schema, mirroring the article's public config."""

from dataclasses import MISSING, dataclass, fields, is_dataclass
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
    "ModelEnvsConfig",
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
    # "real" runs the real models (production default); "test" is only for stand-in
    # registration in local end-to-end tests.
    backend: str = "real"
    # Number of CPU actors in each node pool (same pool and node as the 1 GPU actor).
    cpu_actors_per_node: int = 2
    # Upper bound (seconds) to wait for a slot on the episode's owning node to free
    # up. On timeout the episode is counted as failed and left with a queryable
    # error instead of silently hanging — node affinity is hard: if no slot frees up,
    # it simply waits. Implemented as a no-progress cap on "the owning node's CPU
    # group stays busy": the main loop's ray.wait runs every ~0.05s, so the driver
    # maps seconds to a number of consecutive no-dispatch rounds (default 600s ->
    # 12000 rounds) as the convergence guard.
    node_wait_timeout_s: float = 600.0
    # Model worker-process mode: "resident" = the adapter holds a resident
    # ModelEnvSession and weights load once (propagated to workers via
    # ROBORAY_MODEL_WORKER); "oneshot" = one subprocess per inference (byte-for-byte
    # equivalent to the previous behavior). GpuModelActor.__init__ reads it in the
    # worker process.
    model_worker: str = "resident"
    # Three-level idling (spec §5.6 / §8): how long an actor idles before it unloads
    # proactively (returns VRAM while staying alive); 0 = never unload proactively
    # (only on role switch / teardown). Must increase with actor_idle_exit_s
    # (30 < 240 < 300 node reclamation), otherwise a node never sees actors exit.
    actor_idle_unload_s: float = 30.0
    # How long an actor idles before exiting (driver unloads first, then ray.kill);
    # 0 = never exit proactively.
    actor_idle_exit_s: float = 240.0


@dataclass(frozen=True)
class SceneStageConfig:
    stereo_max_image_side_px: int = 640
    model: str = "auto"


@dataclass(frozen=True)
class HandStageConfig:
    inference_fps: int = 30
    max_hands: int = 4  # detection-count cap (same source as the script's DEFAULT_MAX_HANDS)
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
    model: str = "openai_vlm"


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
class ModelEnvsConfig:
    """Interpreter and weights paths for each real model's separate conda env.
    None/empty = use the current env or a default location.

    ``lerobot_python`` / ``lerobot_script``: interpreter and script overrides for the
    LeRobot export env. Both are flattened by ``resolve_env_request`` into
    ``ROBORAY_LEROBOT_*`` and shipped to workers via runtime_env — **multi-node /
    remote clusters must configure these**, otherwise workers never receive them
    (local inheritance cannot be relied on). ``lerobot_script`` is test-only
    (empty/None = the real script in the repo).
    """

    vipe_python: Optional[str] = None
    ffs_python: Optional[str] = None
    hammer_python: Optional[str] = None
    vipe_weights: Optional[str] = None
    ffs_weights: Optional[str] = None
    hammer_weights: Optional[str] = None
    lerobot_python: Optional[str] = None
    lerobot_script: Optional[str] = None


@dataclass(frozen=True)
class ReconstructionRequest:
    input: InputConfig
    output: OutputConfig
    execution: ExecutionConfig = ExecutionConfig()
    stages: StageConfigs = StageConfigs()
    models: ModelEnvsConfig = ModelEnvsConfig()

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
        # Explicitly exclude bool: bool is an int subclass, but True/False are not
        # valid numeric config values.
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
        elif f.default is not MISSING:
            continue
        elif f.default_factory is not MISSING:
            continue
        else:
            raise ConfigError(f"{sub}: missing required key")
    return cls(**kwargs)


def _dump(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dump(getattr(obj, f.name)) for f in fields(obj)}
    return obj
