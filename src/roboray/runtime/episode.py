"""Episode discovery, per-episode workspace, and resumable stage state."""

import json
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
    "STATE_SCHEMA_VERSION",
    "PREREQS",
]

ARTIFACT_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1


class Stage(str, Enum):
    PREPARE = "prepare"
    SCENE = "scene"
    HAND = "hand"
    FUSION = "fusion"
    ACTION = "action"
    PUBLISH = "publish"


STAGE_ORDER = [Stage.PREPARE, Stage.SCENE, Stage.HAND, Stage.FUSION, Stage.ACTION, Stage.PUBLISH]

# Prerequisite stages for each stage (SCENE and HAND run in parallel after PREPARE)
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
    # Backfilled after prepare: the driver uses this to route (scene stereo
    # decision, resolution tier). Never reading workspace files is a hard
    # constraint - under multi-node the driver cannot read node-local workspaces.
    has_stereo: Optional[bool] = None
    resolution_key: Optional[str] = None
    # Which Ray node this episode is pinned to (written at assignment, validated
    # for node loss).
    assigned_node: Optional[str] = None
    # prepare's degraded info: returned via stage_result (the driver does not
    # read the workspace summary.json - it is unreadable under multi-node) and
    # consumed by the manifest.
    degraded: Optional[bool] = None
    degrade_reason: Optional[str] = None
    # prepare's alignment mode and timestamp-dropped frame count. Dropped
    # timestamps are not stereo degradation (has_stereo stays True); align_dropped
    # is tracked separately; defaults of None/0 keep old records compatible.
    alignment: Optional[str] = None
    align_dropped: Optional[int] = 0

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
        """Return the first runnable stage per STAGE_ORDER: prerequisites all DONE and itself PENDING.

        Returns None if any stage is RUNNING or FAILED (no other stage of the
        same episode is advanced concurrently).
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
            "state_schema_version": STATE_SCHEMA_VERSION,
            "stages": {s.value: v.value for s, v in self.stages.items()},
            "timings": dict(self.timings),
            "error": self.error,
            "has_stereo": self.has_stereo,
            "resolution_key": self.resolution_key,
            "assigned_node": self.assigned_node,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "alignment": self.alignment,
            "align_dropped": self.align_dropped,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "EpisodeRecord":
        if not isinstance(raw, dict):
            raise ValueError("state.json missing/corrupt: expected a JSON object")
        version = raw.get("state_schema_version", 1)
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"state.json schema version mismatch: file has {version}, "
                f"current supports {STATE_SCHEMA_VERSION}"
            )
        stages = raw.get("stages")
        if not isinstance(stages, dict) or "episode_id" not in raw:
            raise ValueError("state.json missing/corrupt: missing episode_id or stages")
        try:
            parsed = {Stage(k): StageStatus(v) for k, v in stages.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError(f"state.json missing/corrupt: invalid stages ({exc})") from exc
        return cls(
            episode_id=raw["episode_id"],
            stages=parsed,
            timings={k: float(v) for k, v in raw.get("timings", {}).items()},
            error=raw.get("error"),
            # Older state.json lacks these three fields: they must tolerate None,
            # otherwise re-running an existing workspace after an upgrade would
            # raise and discard completed stage state.
            has_stereo=raw.get("has_stereo"),
            resolution_key=raw.get("resolution_key"),
            assigned_node=raw.get("assigned_node"),
            degraded=raw.get("degraded"),
            degrade_reason=raw.get("degrade_reason"),
            # Older state.json lacks these two fields: tolerate None/0 (same
            # strategy as degraded).
            alignment=raw.get("alignment"),
            align_dropped=raw.get("align_dropped", 0),
        )

    def reconcile_for_new_run(self, live_nodes: set | None = None) -> list:
        """Align state for a new run: cross-run crashes/failures return to a recoverable state.

        - RUNNING -> PENDING: crash residue; no in-flight tasks can exist at run start.
        - FAILED -> PENDING: the cross-run retry semantics of "re-run resumes
          from the first unfinished stage".
        - **assigned_node no longer in live_nodes -> all stages back to PENDING**:
          the workspace is node-local, so if the node is gone the workspace is
          gone too. In that case we must **not** only flip RUNNING back to
          PENDING - that would leave a false "prepare DONE" and the next run
          would read a prepared/ that no longer exists.
        (Within a run, FAILED stays terminal and the driver keeps it as-is.)

        ``live_nodes=None`` means node validation is skipped (unchanged behavior
        for older callers). Returns the list of changes [(stage, from_status)]
        so the caller can persist as needed.
        """
        changes = []

        node_lost = (
            live_nodes is not None
            and self.assigned_node is not None
            and self.assigned_node not in live_nodes
        )
        if node_lost:
            for stage in STAGE_ORDER:
                if self.stages[stage] is not StageStatus.PENDING:
                    changes.append((stage, self.stages[stage]))
                    self.stages[stage] = StageStatus.PENDING
            self.assigned_node = None
            self.error = None
            return changes

        for stage in STAGE_ORDER:
            cur = self.stages[stage]
            if cur in (StageStatus.RUNNING, StageStatus.FAILED):
                self.stages[stage] = StageStatus.PENDING
                changes.append((stage, cur))
        if changes:
            self.error = None
        return changes

    def record_prepare(
        self,
        has_stereo: bool,
        width: int,
        height: int,
        degraded: bool = False,
        degrade_reason: Optional[str] = None,
        alignment: Optional[str] = None,
        align_dropped: int = 0,
    ) -> None:
        """Backfill prepare's routing metadata.

        ``resolution_key`` uses the same ``"<w>x<h>"`` format as
        ``planner.DEFAULT_RESOLUTION_KEY`` (see ``stages/scene.resolution_key``).
        ``degraded`` / ``degrade_reason`` also come from prepare's return value:
        the driver does not read the workspace ``prepared/summary.json``
        (unreadable under multi-node), so degraded info must travel back through
        the actor along with the metadata, consumed by the manifest.
        ``alignment`` / ``align_dropped``: alignment mode and timestamp-dropped
        frame count; defaults of None/0 keep old callers compatible.
        """
        self.has_stereo = bool(has_stereo)
        self.resolution_key = f"{int(width)}x{int(height)}"
        self.degraded = bool(degraded)
        self.degrade_reason = degrade_reason
        self.alignment = alignment
        self.align_dropped = int(align_dropped) if align_dropped else 0

    def undone_stages(self) -> list:
        """Stage names not yet done (list of values); any state other than is_done counts as undone."""
        return [s.value for s in STAGE_ORDER if self.stages.get(s) is not StageStatus.DONE]


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    rel_dir: str
    has_stereo: bool


class EpisodeDiscoveryError(ValueError):
    """Raised when the input dataset is structurally invalid."""


def discover_episodes(storage: Storage, rel: str = ".") -> list[EpisodeSpec]:
    """Discover episodes in the input dataset.

    Rule: each subdirectory containing a ``meta.json`` is one episode. A missing
    meta.json, an invalid meta.json, or a duplicate episode_id all raise
    immediately (fail fast rather than silently producing wrong data).
    """
    specs: list[EpisodeSpec] = []
    seen: dict[str, str] = {}
    for name in storage.listdir(rel):
        sub = f"{rel}/{name}" if rel not in (".", "") else name
        if not storage.is_dir(sub):
            continue  # a file cannot be an episode (README.md, thumbnails, etc.), skip
        if not storage.exists(f"{sub}/meta.json"):
            raise EpisodeDiscoveryError(f"{sub}/meta.json missing: directory is not a valid episode")
        try:
            meta = json.loads(storage.read_bytes(f"{sub}/meta.json").decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EpisodeDiscoveryError(f"{sub}/meta.json is not valid JSON: {exc}") from exc
        if not isinstance(meta, dict) or not isinstance(meta.get("episode_id"), str):
            raise EpisodeDiscoveryError(f"{sub}/meta.json missing string field episode_id")
        ep_id = meta["episode_id"]
        if ep_id in seen:
            raise EpisodeDiscoveryError(
                f"duplicate episode_id: {ep_id!r} appears in both {seen[ep_id]!r} and {sub!r}; "
                "outputs would overwrite each other, please fix the input."
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
    """A single episode's workspace: intermediates + stage state (the resumable-run unit)."""

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
        tmp.replace(self.state_file)  # atomic replace, avoids a half-written state

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

    def apply_and_save(
        self,
        rec: EpisodeRecord,
        stage: Stage,
        status: StageStatus,
        seconds: Optional[float] = None,
        error: Optional[str] = None,
    ) -> EpisodeRecord:
        """Apply the state transition to the **given** record and persist it, **without reading from disk**.

        Hard invariant: the driver process never reads workspace files -
        state.json is only cross-run persistence for resumable runs. The driver
        keeps the record authoritatively in memory and calls this method when
        advancing state: it both updates the passed-in record (in place) and
        writes the new state to disk (for the next run to resume), but **never
        re-reads**. Unlike ``mark_*`` (which call ``load()`` first for
        independent callers/tests), this method requires the caller to already
        hold the in-memory record.
        """
        rec.stages[stage] = status
        if seconds is not None:
            rec.timings[stage.value] = float(seconds)
        if status is StageStatus.FAILED:
            rec.error = error
        self._save(rec)
        return rec

    def save_record(self, rec: EpisodeRecord) -> None:
        """Save (rewrite) a full state; used for one-shot bulk state migrations like reconcile."""
        self._save(rec)

    def write_artifact(self, name: str, arrays: dict) -> Path:
        p = self.artifact(name)
        np.savez(p, __schema_version__=ARTIFACT_SCHEMA_VERSION, **arrays)
        return p

    def check_artifact(self, name: str) -> bool:
        """Validate that the artifact is readable and matches the schema version; corrupt or mismatched returns False (that stage must be re-run)."""
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
