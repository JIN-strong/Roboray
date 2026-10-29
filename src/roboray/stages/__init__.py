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
