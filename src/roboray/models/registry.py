"""Model registry: production code resolves models by name, tests inject stand-ins."""

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
    """Register the production models. Factories use lazy imports: the actual import
    happens at instantiation time, so this module can be imported even where torch
    is not installed.
    """

    def _scene(name: str):
        def factory():
            from roboray.models.scene_models import (
                FastFoundationStereoModel,
                VipeSceneModel,
            )

            return VipeSceneModel() if name == "vipe" else FastFoundationStereoModel()

        return factory

    def _hand(name: str):
        def factory():
            from roboray.models.hand_models import HamerHandModel

            return HamerHandModel()

        return factory

    def _semantic(name: str):
        def factory():
            from roboray.models.semantic_models import OpenAIVlm

            return OpenAIVlm()

        return factory

    for name in ("vipe", "fast_foundation_stereo"):
        if name not in _SCENE:
            _SCENE[name] = _scene(name)
    if "hamer" not in _HAND:
        _HAND["hamer"] = _hand("hamer")
    if "openai_vlm" not in _SEMANTIC:
        _SEMANTIC["openai_vlm"] = _semantic("openai_vlm")
