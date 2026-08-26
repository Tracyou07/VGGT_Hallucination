"""Strict prediction-only, resumable artifact contracts for CVA02."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np


PREDICTION_SCHEMA = "camera_velocity_ambiguity_02.prediction_completion.v1"
PREDICTION_MEMBERS = {
    "frame_ids",
    "normalized_camera_tokens",
    "pred_c2w_raw",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")


class IncompletePredictionArtifact(FileNotFoundError):
    """Raised when an NPZ has no authenticated completion sidecar."""


@dataclass(frozen=True)
class PredictionIdentity:
    """Frozen inputs that make an existing model output reusable."""

    run_id: str
    scene: str
    artifact_kind: str
    window_index: int | None
    frame_digest: str
    checkpoint_sha256: str
    git_commit: str
    preprocess: str
    camera_iterations: int
    protocol_digest: str


def _ordered_frame_ids(frame_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_ids)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("frame digest requires a non-empty frame ID vector")
    integer = values.astype(np.int64, copy=False)
    if not np.array_equal(values, integer):
        raise ValueError("frame digest requires integer frame IDs")
    if any(left >= right for left, right in zip(integer, integer[1:])):
        raise ValueError("frame digest requires strictly increasing unique frame IDs")
    return integer


def frame_digest(frame_ids: np.ndarray) -> str:
    """Hash ordered frame identity using canonical JSON, independent of platform."""
    integer = _ordered_frame_ids(frame_ids)
    canonical = json.dumps(
        [int(value) for value in integer], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def build_prediction_identity(
    *,
    run_id: str,
    scene: str,
    artifact_kind: str,
    window_index: int | None,
    frame_ids: np.ndarray,
    checkpoint_sha256: str,
    git_commit: str,
    protocol_digest: str,
    preprocess: str,
    camera_iterations: int,
) -> PredictionIdentity:
    """Validate and freeze all provenance needed for exact resume."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not isinstance(scene, str) or _SCENE_PATTERN.fullmatch(scene) is None:
        raise ValueError("scene must use ScanNet sceneNNNN_NN identity")
    if artifact_kind not in {"global", "local"}:
        raise ValueError("artifact_kind must be global or local")
    if artifact_kind == "global" and window_index is not None:
        raise ValueError("global artifacts cannot have a window_index")
    if artifact_kind == "local" and (
        isinstance(window_index, bool) or not isinstance(window_index, (int, np.integer)) or int(window_index) < 0
    ):
        raise ValueError("local artifacts require a non-negative window_index")
    if preprocess != "crop":
        raise ValueError("preprocess must be crop")
    if camera_iterations != 4:
        raise ValueError("camera_iterations must be 4")
    _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    _require_sha256(protocol_digest, "protocol_digest")
    if not isinstance(git_commit, str) or _COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a 40-character lowercase hexadecimal id")
    return PredictionIdentity(
        run_id=run_id,
        scene=scene,
        artifact_kind=artifact_kind,
        window_index=None if window_index is None else int(window_index),
        frame_digest=frame_digest(frame_ids),
        checkpoint_sha256=checkpoint_sha256,
        git_commit=git_commit,
        preprocess=preprocess,
        camera_iterations=camera_iterations,
        protocol_digest=protocol_digest,
    )


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(arrays) != PREDICTION_MEMBERS:
        raise ValueError(f"prediction NPZ members must be exactly {sorted(PREDICTION_MEMBERS)}")
    frame_ids = _ordered_frame_ids(np.asarray(arrays["frame_ids"]))
    tokens = np.asarray(arrays["normalized_camera_tokens"])
    poses = np.asarray(arrays["pred_c2w_raw"])
    count = len(frame_ids)
    if tokens.ndim != 2 or tokens.shape[0] != count or tokens.shape[1] < 1:
        raise ValueError("normalized_camera_tokens must have shape [frames, channels]")
    if poses.shape != (count, 4, 4):
        raise ValueError("pred_c2w_raw must have shape [frames, 4, 4]")
    if not np.issubdtype(tokens.dtype, np.number) or not np.issubdtype(poses.dtype, np.number):
        raise ValueError("prediction arrays must be numeric")
    if not np.isfinite(tokens).all() or not np.isfinite(poses).all():
        raise ValueError("prediction arrays must contain only finite values")
    return {
        "frame_ids": frame_ids.copy(),
        "normalized_camera_tokens": tokens.copy(),
        "pred_c2w_raw": poses.copy(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion_payload(identity: PredictionIdentity, artifact_sha256: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": PREDICTION_SCHEMA,
        **asdict(identity),
        "artifact_sha256": artifact_sha256,
        "members": sorted(PREDICTION_MEMBERS),
    }
    return payload


def save_completed_prediction(
    artifact_path: Path,
    completion_path: Path,
    arrays: Mapping[str, np.ndarray],
    identity: PredictionIdentity,
) -> None:
    """Publish the NPZ first and its completion marker last."""
    artifact_path = Path(artifact_path)
    completion_path = Path(completion_path)
    validated = _validate_arrays(arrays)
    if frame_digest(validated["frame_ids"]) != identity.frame_digest:
        raise ValueError("prediction frame digest differs from provenance")
    if artifact_path == completion_path:
        raise ValueError("artifact and completion paths must differ")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_tmp = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    completion_tmp = completion_path.with_suffix(completion_path.suffix + ".tmp")
    with artifact_tmp.open("wb") as handle:
        np.savez_compressed(handle, **validated)
    payload = _completion_payload(identity, _sha256_file(artifact_tmp))
    completion_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completion_path.unlink(missing_ok=True)
    artifact_tmp.replace(artifact_path)
    completion_tmp.replace(completion_path)


def _read_completion(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prediction completion sidecar: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("prediction completion sidecar must be an object")
    return payload


def load_completed_prediction(
    artifact_path: Path,
    completion_path: Path,
    expected_identity: PredictionIdentity,
) -> dict[str, np.ndarray]:
    """Load only a fully published artifact with exact identity and content hash."""
    artifact_path = Path(artifact_path)
    completion_path = Path(completion_path)
    if not artifact_path.is_file() or not completion_path.is_file():
        raise IncompletePredictionArtifact(
            f"prediction is incomplete without both artifact and sidecar: {artifact_path}"
        )
    payload = _read_completion(completion_path)
    artifact_sha256 = payload.get("artifact_sha256")
    _require_sha256(artifact_sha256, "artifact_sha256")
    expected_payload = _completion_payload(expected_identity, artifact_sha256)
    if payload != expected_payload:
        raise ValueError("prediction completion schema or provenance mismatch")
    if _sha256_file(artifact_path) != artifact_sha256:
        raise ValueError("artifact SHA-256 mismatch")

    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            loaded = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid prediction NPZ: {artifact_path}") from error
    validated = _validate_arrays(loaded)
    if frame_digest(validated["frame_ids"]) != expected_identity.frame_digest:
        raise ValueError("prediction frame digest differs from provenance")
    return validated
