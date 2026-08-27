from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from pre_experiments.variational_camera_latent.alpha_scan import DEFAULT_ALPHAS
from pre_experiments.variational_camera_latent.candidates import load_candidate_shard
from pre_experiments.variational_camera_latent.vrfm_residual_scan import (
    load_vrfm_residual_alpha_scan,
)

from .schema import load_long_context_shard


CANDIDATE_BINDING_SCHEMA = "variational_camera_selector.candidate_binding_manifest.v1"
PRIVILEGED_BINDING_SCHEMA = "variational_camera_selector.privileged_binding_manifest.v1"
_ROLES = {"train", "validation", "smoke"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PREDICTION_RECORD_MEMBERS = {
    "scene",
    "role",
    "long_context_path",
    "long_context_sha256",
    "source_sha256",
    "candidate_path",
    "candidate_sha256",
    "residual_prediction_path",
    "residual_prediction_sha256",
}
_PRIVILEGED_RECORD_MEMBERS = {
    "scene",
    "role",
    "path",
    "sha256",
    "prediction_sha256",
    "source_sha256",
    "candidate_sha256",
}


@dataclass(frozen=True)
class CandidateGroup:
    """All selectable states for one 50-frame overlap."""

    scene: str
    role: str
    overlap_index: int
    sample_id: str
    span_start: int
    global_tokens: np.ndarray
    x0: np.ndarray
    delta_tokens: np.ndarray
    alphas: np.ndarray
    z: np.ndarray
    sample_seeds: np.ndarray
    choice_ids: np.ndarray
    source_sha256: str
    candidate_sha256: str
    residual_prediction_sha256: str
    utilities: np.ndarray | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _normalize_roles(roles: Sequence[str] | None) -> frozenset[str]:
    selected = frozenset(_ROLES if roles is None else roles)
    if not selected or not selected.issubset(_ROLES):
        raise ValueError("roles must be a non-empty subset of train, validation, and smoke")
    return selected


def _validate_manifest_records(
    payload: Mapping[str, object],
    *,
    schema: str,
    members: set[str],
    label: str,
) -> list[dict[str, object]]:
    if payload.get("schema") != schema:
        raise ValueError(f"{label} schema does not match")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} must contain at least one record")
    normalized: list[dict[str, object]] = []
    scenes: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != members:
            raise ValueError(f"{label} record members do not match the schema")
        scene = record.get("scene")
        role = record.get("role")
        if not isinstance(scene, str) or not scene:
            raise ValueError(f"{label} scene must be a non-empty string")
        if scene in scenes:
            raise ValueError(f"{label} contains a duplicate scene")
        if role not in _ROLES:
            raise ValueError(f"{label} role is invalid")
        scenes.add(scene)
        normalized.append(record.copy())
    return normalized


def expand_candidate_grid(
    x0: np.ndarray,
    corrected: np.ndarray,
    z: np.ndarray,
    sample_seeds: np.ndarray,
    alphas: Sequence[float] | np.ndarray = DEFAULT_ALPHAS,
    *,
    sample_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand one no-op plus an alpha-major grid of scaled residuals."""
    x0_array = np.asarray(x0, dtype=np.float32)
    corrected_array = np.asarray(corrected, dtype=np.float32)
    z_array = np.asarray(z, dtype=np.float32)
    seeds_array = np.asarray(sample_seeds, dtype=np.int64)
    alpha_values = np.asarray(alphas, dtype=np.float32)
    if x0_array.shape != (50, 2048):
        raise ValueError("x0 must have shape [50, 2048]")
    if corrected_array.ndim != 3 or corrected_array.shape[1:] != (50, 2048):
        raise ValueError("corrected candidates must have shape [samples, 50, 2048]")
    samples = corrected_array.shape[0]
    if z_array.ndim != 2 or z_array.shape[0] != samples or z_array.shape[1] < 1:
        raise ValueError("z must have shape [samples, z_dim]")
    if seeds_array.shape != (samples,) or len(set(seeds_array.tolist())) != samples:
        raise ValueError("sample seeds must be a unique integer vector [samples]")
    expected_alphas = np.asarray(DEFAULT_ALPHAS, dtype=np.float32)
    if alpha_values.shape != expected_alphas.shape or not np.array_equal(
        alpha_values, expected_alphas
    ):
        raise ValueError("alphas must exactly match the frozen selector grid")
    if not sample_id:
        raise ValueError("sample_id must be non-empty")

    nonzero = alpha_values[1:]
    residual = corrected_array - x0_array[None]
    scaled = (nonzero[:, None, None, None] * residual[None]).reshape(
        len(nonzero) * samples, 50, 2048
    )
    delta_tokens = np.concatenate(
        (np.zeros((1, 50, 2048), dtype=np.float32), scaled), axis=0
    )
    choice_alphas = np.concatenate(
        (np.zeros(1, dtype=np.float32), np.repeat(nonzero, samples))
    )
    choice_z = np.concatenate(
        (np.zeros((1, z_array.shape[1]), dtype=np.float32), np.tile(z_array, (len(nonzero), 1))),
        axis=0,
    )
    choice_seeds = np.concatenate(
        (np.asarray([-1], dtype=np.int64), np.tile(seeds_array, len(nonzero)))
    )
    ids = [f"{sample_id}:noop"]
    ids.extend(
        f"{sample_id}:seed_{int(seed)}:alpha_{float(alpha):g}"
        for alpha in nonzero
        for seed in seeds_array
    )
    choice_ids = np.asarray(ids, dtype="U160")
    if not all(
        np.isfinite(value).all() for value in (delta_tokens, choice_alphas, choice_z)
    ):
        raise ValueError("expanded candidate grid contains non-finite values")
    return delta_tokens, choice_alphas, choice_z, choice_seeds, choice_ids


class PredictionCandidateDataset:
    """Prediction-only groups; this API has no label-sidecar argument."""

    def __init__(
        self,
        binding_manifest: Path,
        *,
        roles: Sequence[str] | None = None,
    ) -> None:
        payload = _read_json(binding_manifest, "candidate binding manifest")
        if payload.get("alphas") != list(DEFAULT_ALPHAS):
            raise ValueError("candidate binding alpha grid does not match the frozen grid")
        records = _validate_manifest_records(
            payload,
            schema=CANDIDATE_BINDING_SCHEMA,
            members=_PREDICTION_RECORD_MEMBERS,
            label="candidate binding manifest",
        )
        selected_roles = _normalize_roles(roles)
        self.records = [record for record in records if str(record["role"]) in selected_roles]
        if not self.records:
            raise ValueError("candidate binding manifest has no records for the selected roles")
        self.scenes = tuple(str(record["scene"]) for record in self.records)
        self.roles = tuple(str(record["role"]) for record in self.records)
        self.binding_manifest = Path(binding_manifest)
        self._cache_index: int | None = None
        self._cache_long: dict[str, np.ndarray] | None = None
        self._cache_candidate: dict[str, np.ndarray] | None = None
        for record in self.records:
            self._validate_prediction_record(record)

    def __len__(self) -> int:
        return 8 * len(self.records)

    def _validate_prediction_record(self, record: Mapping[str, object]) -> None:
        for field in (
            "long_context_sha256",
            "source_sha256",
            "candidate_sha256",
            "residual_prediction_sha256",
        ):
            _validate_digest(record[field], field)
        long_path = Path(str(record["long_context_path"]))
        candidate_path = Path(str(record["candidate_path"]))
        residual_path = Path(str(record["residual_prediction_path"]))
        if _sha256_file(long_path) != record["long_context_sha256"]:
            raise ValueError("long-context SHA-256 does not match its binding record")
        if _sha256_file(candidate_path) != record["candidate_sha256"]:
            raise ValueError("candidate SHA-256 does not match its binding record")
        if _sha256_file(residual_path) != record["residual_prediction_sha256"]:
            raise ValueError("residual prediction SHA-256 does not match its binding record")
        long_context = load_long_context_shard(long_path)
        candidate = load_candidate_shard(candidate_path)
        if str(long_context["scene"]) != record["scene"]:
            raise ValueError("long-context scene does not match its binding record")
        if str(long_context["role"]) != record["role"]:
            raise ValueError("long-context role does not match its binding record")
        if str(long_context["source_shard_sha256"]) != record["source_sha256"]:
            raise ValueError("long-context source digest does not match its binding record")
        if str(long_context["candidate_shard_sha256"]) != record["candidate_sha256"]:
            raise ValueError("long-context candidate digest does not match its binding record")
        if candidate["z"].shape[1] != 32:
            raise ValueError("selector requires exactly 32 VRFM directions per overlap")
        if not np.array_equal(candidate["source_sample_ids"], long_context["source_sample_ids"]):
            raise ValueError("candidate and long-context sample IDs do not match")
        if not np.array_equal(candidate["span_starts"], long_context["span_starts"]):
            raise ValueError("candidate and long-context span starts do not match")
        if not np.array_equal(candidate["source_long_tokens"], long_context["overlap_long_tokens"]):
            raise ValueError("candidate and long-context source tokens do not match")

    def _load_record(
        self, record_index: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self._cache_index != record_index:
            record = self.records[record_index]
            self._cache_long = load_long_context_shard(Path(str(record["long_context_path"])))
            self._cache_candidate = load_candidate_shard(Path(str(record["candidate_path"])))
            self._cache_index = record_index
        assert self._cache_long is not None and self._cache_candidate is not None
        return self._cache_long, self._cache_candidate

    def __getitem__(self, index: int) -> CandidateGroup:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index, overlap = divmod(index, 8)
        record = self.records[record_index]
        long_context, candidate = self._load_record(record_index)
        delta, alphas, z, seeds, choice_ids = expand_candidate_grid(
            long_context["overlap_long_tokens"][overlap],
            candidate["corrected_camera_tokens"][overlap],
            candidate["z"][overlap],
            candidate["sample_seeds"][overlap],
            sample_id=str(long_context["source_sample_ids"][overlap]),
        )
        return CandidateGroup(
            scene=str(record["scene"]),
            role=str(record["role"]),
            overlap_index=overlap,
            sample_id=str(long_context["source_sample_ids"][overlap]),
            span_start=int(long_context["span_starts"][overlap]),
            global_tokens=long_context["global_camera_tokens"].copy(),
            x0=long_context["overlap_long_tokens"][overlap].copy(),
            delta_tokens=delta,
            alphas=alphas,
            z=z,
            sample_seeds=seeds,
            choice_ids=choice_ids,
            source_sha256=str(record["source_sha256"]),
            candidate_sha256=str(record["candidate_sha256"]),
            residual_prediction_sha256=str(record["residual_prediction_sha256"]),
        )


def _validate_join_arrays(
    group: CandidateGroup,
    prediction: Mapping[str, np.ndarray],
    sidecar: Mapping[str, np.ndarray],
) -> None:
    overlap = group.overlap_index
    scene = str(prediction["source_sample_ids"][0]).split(":", 1)[0]
    if scene != group.scene:
        raise ValueError("prediction scene does not match candidate group")
    if str(prediction["source_shard_sha256"]) != group.source_sha256:
        raise ValueError("prediction source SHA-256 does not match candidate group")
    if str(prediction["candidate_shard_sha256"]) != group.candidate_sha256:
        raise ValueError("prediction candidate SHA-256 does not match candidate group")
    if str(sidecar["prediction_sha256"]) != group.residual_prediction_sha256:
        raise ValueError("privileged prediction SHA-256 does not match candidate group")
    if not np.array_equal(prediction["source_sample_ids"], sidecar["source_sample_ids"]):
        raise ValueError("prediction and privileged sample IDs do not match")
    if str(prediction["source_sample_ids"][overlap]) != group.sample_id:
        raise ValueError("group sample ID does not match joined artifacts")
    if not np.array_equal(prediction["sample_seeds"], sidecar["sample_seeds"]):
        raise ValueError("prediction and privileged sample seeds do not match")
    if not np.array_equal(prediction["alphas"], sidecar["alphas"]):
        raise ValueError("prediction and privileged alpha grids do not match")
    if not np.array_equal(
        prediction["alphas"], np.asarray(DEFAULT_ALPHAS, dtype=np.float64)
    ):
        raise ValueError("joined alpha grid does not match the frozen selector grid")
    expected_seeds = np.concatenate(
        (
            np.asarray([-1], dtype=np.int64),
            np.tile(prediction["sample_seeds"][overlap], len(DEFAULT_ALPHAS) - 1),
        )
    )
    if not np.array_equal(group.sample_seeds, expected_seeds):
        raise ValueError("expanded group sample seeds do not match joined prediction")


def join_training_group(
    group: CandidateGroup,
    prediction: Mapping[str, np.ndarray],
    sidecar: Mapping[str, np.ndarray],
) -> CandidateGroup:
    """Attach utilities only after every prediction/label binding validates."""
    _validate_join_arrays(group, prediction, sidecar)
    overlap_relative = np.asarray(
        sidecar["relative_improvement"][group.overlap_index], dtype=np.float32
    )
    if overlap_relative.shape != (32, len(DEFAULT_ALPHAS)):
        raise ValueError("privileged utilities must have shape [32, 8] per overlap")
    utilities = np.concatenate(
        (
            np.zeros(1, dtype=np.float32),
            overlap_relative[:, 1:].T.reshape(-1),
        )
    )
    utilities[0] = 0.0
    if utilities.shape != (225,) or not np.isfinite(utilities).all():
        raise ValueError("joined utilities must be 225 finite values")
    return replace(group, utilities=utilities)


class SelectorTrainingDataset:
    """Training-only wrapper that joins the physically separate utility sidecars."""

    def __init__(
        self,
        prediction_manifest: Path,
        privileged_manifest: Path,
        *,
        roles: Sequence[str] = ("train",),
    ) -> None:
        self.prediction_dataset = PredictionCandidateDataset(
            prediction_manifest, roles=roles
        )
        payload = _read_json(privileged_manifest, "privileged binding manifest")
        records = _validate_manifest_records(
            payload,
            schema=PRIVILEGED_BINDING_SCHEMA,
            members=_PRIVILEGED_RECORD_MEMBERS,
            label="privileged binding manifest",
        )
        by_scene = {str(record["scene"]): record for record in records}
        self.privileged_records: list[dict[str, object]] = []
        self._prediction_cache_scene: str | None = None
        self._prediction_cache: dict[str, np.ndarray] | None = None
        self._sidecar_cache: dict[str, np.ndarray] | None = None
        for prediction_record in self.prediction_dataset.records:
            scene = str(prediction_record["scene"])
            sidecar_record = by_scene.get(scene)
            if sidecar_record is None:
                raise ValueError("privileged binding scene does not match prediction scenes")
            self._validate_training_record(prediction_record, sidecar_record)
            self.privileged_records.append(sidecar_record)
        self.scenes = self.prediction_dataset.scenes
        self.roles = self.prediction_dataset.roles

    def __len__(self) -> int:
        return len(self.prediction_dataset)

    def _validate_training_record(
        self,
        prediction_record: Mapping[str, object],
        sidecar_record: Mapping[str, object],
    ) -> None:
        if sidecar_record["scene"] != prediction_record["scene"]:
            raise ValueError("privileged scene does not match prediction scene")
        if sidecar_record["role"] != prediction_record["role"]:
            raise ValueError("privileged role does not match prediction role")
        for field in ("sha256", "prediction_sha256", "source_sha256", "candidate_sha256"):
            _validate_digest(sidecar_record[field], f"privileged {field}")
        if sidecar_record["prediction_sha256"] != prediction_record["residual_prediction_sha256"]:
            raise ValueError("privileged prediction digest does not match prediction binding")
        if sidecar_record["source_sha256"] != prediction_record["source_sha256"]:
            raise ValueError("privileged source digest does not match prediction binding")
        if sidecar_record["candidate_sha256"] != prediction_record["candidate_sha256"]:
            raise ValueError("privileged candidate digest does not match prediction binding")
        sidecar_path = Path(str(sidecar_record["path"]))
        if _sha256_file(sidecar_path) != sidecar_record["sha256"]:
            raise ValueError("privileged sidecar SHA-256 does not match its manifest")

        # Import the GT-bearing loader only in this training-only code path.
        from pre_experiments.variational_camera_latent.vrfm_residual_scan import (
            load_vrfm_residual_privileged,
        )

        prediction = load_vrfm_residual_alpha_scan(
            Path(str(prediction_record["residual_prediction_path"]))
        )
        sidecar = load_vrfm_residual_privileged(sidecar_path)
        probe = CandidateGroup(
            scene=str(prediction_record["scene"]),
            role=str(prediction_record["role"]),
            overlap_index=0,
            sample_id=str(prediction["source_sample_ids"][0]),
            span_start=int(prediction["span_starts"][0]),
            global_tokens=np.empty((0,), dtype=np.float32),
            x0=np.empty((0,), dtype=np.float32),
            delta_tokens=np.empty((0,), dtype=np.float32),
            alphas=np.empty((0,), dtype=np.float32),
            z=np.empty((0,), dtype=np.float32),
            sample_seeds=np.concatenate(
                (
                    np.asarray([-1], dtype=np.int64),
                    np.tile(prediction["sample_seeds"][0], len(DEFAULT_ALPHAS) - 1),
                )
            ),
            choice_ids=np.empty((0,), dtype="U1"),
            source_sha256=str(prediction_record["source_sha256"]),
            candidate_sha256=str(prediction_record["candidate_sha256"]),
            residual_prediction_sha256=str(prediction_record["residual_prediction_sha256"]),
        )
        _validate_join_arrays(probe, prediction, sidecar)

    def _load_join_arrays(
        self, record_index: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        scene = str(self.prediction_dataset.records[record_index]["scene"])
        if self._prediction_cache_scene != scene:
            prediction_record = self.prediction_dataset.records[record_index]
            sidecar_record = self.privileged_records[record_index]
            from pre_experiments.variational_camera_latent.vrfm_residual_scan import (
                load_vrfm_residual_privileged,
            )

            self._prediction_cache = load_vrfm_residual_alpha_scan(
                Path(str(prediction_record["residual_prediction_path"]))
            )
            self._sidecar_cache = load_vrfm_residual_privileged(
                Path(str(sidecar_record["path"]))
            )
            self._prediction_cache_scene = scene
        assert self._prediction_cache is not None and self._sidecar_cache is not None
        return self._prediction_cache, self._sidecar_cache

    def __getitem__(self, index: int) -> CandidateGroup:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = index // 8
        group = self.prediction_dataset[index]
        prediction, sidecar = self._load_join_arrays(record_index)
        return join_training_group(group, prediction, sidecar)
