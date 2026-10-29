"""EMA prediction of measured latency keyed by (model, resolution tier); the core of the feedback loop."""

__all__ = ["CostModel", "DEFAULT_PRIOR_SECONDS"]

DEFAULT_PRIOR_SECONDS = 1.0
DEFAULT_LOAD_PRIOR_SECONDS = 5.0


class CostModel:
    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
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

    def stats(self) -> dict:
        """Alias for snapshot() (tests access the snapshot via .stats())."""
        return self.snapshot()
