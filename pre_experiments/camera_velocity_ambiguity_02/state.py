"""CVA02 monotonic phase state and immutable calibration policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import canonical_json_digest
from pre_experiments.common.contracts import atomic_write_json


class StudyPhase(str, Enum):
    INPUTS_VERIFIED = "INPUTS_VERIFIED"
    CALIBRATION_COMPLETE = "CALIBRATION_COMPLETE"
    POLICY_FROZEN = "POLICY_FROZEN"
    DEVELOPMENT_COMPLETE = "DEVELOPMENT_COMPLETE"
    DECISION_COMPLETE = "DECISION_COMPLETE"


_PHASES = tuple(StudyPhase)


@dataclass(frozen=True)
class StudyState:
    phase: StudyPhase

    @classmethod
    def initial(cls) -> "StudyState":
        return cls(StudyPhase.INPUTS_VERIFIED)

    def transition(self, target: StudyPhase) -> "StudyState":
        current_index = _PHASES.index(self.phase)
        target_index = _PHASES.index(target)
        if target_index != current_index + 1:
            raise ValueError("study phase transitions must advance exactly one frozen step")
        return StudyState(target)


@dataclass(frozen=True)
class FrozenPolicy:
    direction_cosine_max: float
    normalized_separation_min: float
    barrier_margin: float
    calibration_scenes: tuple[str, ...]
    calibration_pair_count: int
    protocol_digest: str
    input_digest: str
    git_commit: str
    policy_digest: str


def _policy_payload(policy: FrozenPolicy, *, include_digest: bool) -> dict[str, object]:
    payload = asdict(policy)
    payload["calibration_scenes"] = list(policy.calibration_scenes)
    if not include_digest:
        payload.pop("policy_digest")
    return payload


def fit_and_freeze_policy(
    path: Path,
    calibration_rows: Sequence[Mapping[str, object]],
    *,
    calibration_scenes: Sequence[str],
    protocol_digest: str,
    input_digest: str,
    git_commit: str,
) -> FrozenPolicy:
    """Fit once from exactly 10 scenes/80 primary rows and publish immutably."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"frozen policy already exists: {destination}")
    scenes = tuple(calibration_scenes)
    if len(scenes) != 10 or len(set(scenes)) != 10:
        raise ValueError("calibration policy requires exactly 10 unique scenes")
    if len(calibration_rows) != 80:
        raise ValueError("calibration policy requires exactly 80 primary pair rows")
    row_scenes = {str(row.get("scene")) for row in calibration_rows}
    if row_scenes != set(scenes) or any(row.get("route") != "primary" for row in calibration_rows):
        raise ValueError("calibration rows must be primary and match the exact scene set")
    if len({str(row.get("pair_id")) for row in calibration_rows}) != 80:
        raise ValueError("calibration pair identities must be unique")
    if any(sum(row.get("scene") == scene for row in calibration_rows) != 8 for scene in scenes):
        raise ValueError("each calibration scene must contribute exactly eight primary pairs")
    cosine = np.asarray([row["flattened_cosine"] for row in calibration_rows], dtype=np.float64)
    separation = np.asarray([row["normalized_separation"] for row in calibration_rows], dtype=np.float64)
    barriers = np.asarray([row["control_barrier"] for row in calibration_rows], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (cosine, separation, barriers)):
        raise ValueError("calibration threshold inputs must be finite")
    unsigned = {
        "direction_cosine_max": float(np.quantile(cosine, 0.25)),
        "normalized_separation_min": float(np.quantile(separation, 0.25)),
        "barrier_margin": float(max(0.0, np.quantile(barriers, 0.95))),
        "calibration_scenes": list(scenes),
        "calibration_pair_count": 80,
        "protocol_digest": protocol_digest,
        "input_digest": input_digest,
        "git_commit": git_commit,
    }
    policy = FrozenPolicy(
        direction_cosine_max=unsigned["direction_cosine_max"],
        normalized_separation_min=unsigned["normalized_separation_min"],
        barrier_margin=unsigned["barrier_margin"],
        calibration_scenes=scenes,
        calibration_pair_count=80,
        protocol_digest=protocol_digest,
        input_digest=input_digest,
        git_commit=git_commit,
        policy_digest=canonical_json_digest(unsigned),
    )
    atomic_write_json(destination, _policy_payload(policy, include_digest=True))
    return policy


def load_frozen_policy(
    path: Path,
    *,
    protocol_digest: str,
    input_digest: str,
    git_commit: str,
) -> FrozenPolicy:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid frozen policy") from error
    expected_fields = {field.name for field in FrozenPolicy.__dataclass_fields__.values()}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("frozen policy schema mismatch")
    try:
        policy = FrozenPolicy(
            direction_cosine_max=float(payload["direction_cosine_max"]),
            normalized_separation_min=float(payload["normalized_separation_min"]),
            barrier_margin=float(payload["barrier_margin"]),
            calibration_scenes=tuple(payload["calibration_scenes"]),
            calibration_pair_count=int(payload["calibration_pair_count"]),
            protocol_digest=str(payload["protocol_digest"]),
            input_digest=str(payload["input_digest"]),
            git_commit=str(payload["git_commit"]),
            policy_digest=str(payload["policy_digest"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("frozen policy value mismatch") from error
    unsigned = _policy_payload(policy, include_digest=False)
    if (
        canonical_json_digest(unsigned) != policy.policy_digest
        or policy.protocol_digest != protocol_digest
        or policy.input_digest != input_digest
        or policy.git_commit != git_commit
    ):
        raise ValueError("frozen policy digest or provenance mismatch")
    return policy


def apply_development_policy(
    policy: FrozenPolicy,
    rows: Sequence[Mapping[str, object]],
    development_scenes: Sequence[str],
    *,
    threshold_overrides: Mapping[str, float] | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Validate development membership; threshold overrides are forbidden."""
    del policy
    if threshold_overrides is not None:
        raise ValueError("development threshold overrides are forbidden")
    scenes = tuple(development_scenes)
    if len(scenes) != 40 or len(set(scenes)) != 40:
        raise ValueError("development requires exactly 40 unique scenes")
    if {str(row.get("scene")) for row in rows} != set(scenes):
        raise ValueError("development rows do not cover the exact scene set")
    if any(row.get("route") != "primary" for row in rows):
        raise ValueError("development policy applies only to primary rows")
    return tuple(rows)
