"""Independent fixed-oracle replay and fail-closed Stage A-prime gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from numbers import Real
from pathlib import Path
import re

import numpy as np
import torch

from pre_experiments.camera_translation_hvrfm.artifacts import (
    _reject_symlink_components,
    load_bound_bundle_bytes,
)
from pre_experiments.camera_translation_hvrfm.data import (
    PublishedTranslationSample,
    calibration_role,
    validate_calibration_cohort,
)
from pre_experiments.camera_translation_hvrfm.geometry import (
    apply_translation_endpoint,
    prediction_scale,
)
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    FrozenOracle,
    apply_frozen_oracle,
)
from pre_experiments.variational_camera_latent.camera import pose_encoding_to_c2w


_FRAMES = 500
_ENDPOINTS = 4
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_BASELINE_CENTER_ATOL = 5e-6
_BASELINE_ROTATION_ATOL = 2e-5
_DIAGNOSTIC_ATOL = 1e-12
_CALIBRATION_SCENES = (
    "scene0000_00",
    "scene0013_02",
    "scene0029_01",
    "scene0084_01",
    "scene0121_01",
    "scene0207_01",
    "scene0280_00",
    "scene0325_01",
    "scene0675_00",
    "scene0691_00",
)


def decode_saved_oracle(quality: Mapping[str, np.ndarray]) -> FrozenOracle:
    """Construct the already-validated quality sidecar's saved oracle."""
    if not isinstance(quality, Mapping):
        raise ValueError("quality sidecar must be a mapping")
    try:
        return FrozenOracle(
            scene=str(quality["oracle_scene"]),
            frame_digest=str(quality["oracle_frame_digest"]),
            fit_count=int(quality["oracle_fit_count"]),
            scale=float(quality["oracle_scale"]),
            rotation=tuple(
                tuple(float(component) for component in row)
                for row in quality["oracle_rotation"]
            ),
            translation=tuple(
                float(component) for component in quality["oracle_translation"]
            ),
            rank=int(quality["oracle_rank"]),
            condition=float(quality["oracle_condition"]),
            transform_digest=str(quality["oracle_digest"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("quality sidecar has malformed saved-oracle fields") from error


def _sha256_file(path: Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"published artifact must be a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"could not hash published artifact: {source}") from error
    return digest.hexdigest()


def _canonical_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _relative_utility(
    baseline_rms: object, candidate_rms: object, *, name: str
) -> float:
    """Return relative RMS improvement with a strict baseline denominator."""
    try:
        baseline = _finite_float(baseline_rms, name=f"{name} baseline RMS")
    except ValueError as error:
        raise ValueError(
            f"{name} baseline RMS must be finite and positive"
        ) from error
    candidate = _finite_float(candidate_rms, name=f"{name} candidate RMS")
    if baseline <= 0.0:
        raise ValueError(f"{name} baseline RMS must be finite and positive")
    if candidate < 0.0:
        raise ValueError(f"{name} candidate RMS must be finite and nonnegative")
    return float((baseline - candidate) / baseline)


def _aggregate_scene_utilities(
    endpoint_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Aggregate four endpoint utilities using the frozen ratio-of-means rule."""
    if not isinstance(endpoint_rows, Sequence) or isinstance(
        endpoint_rows, (str, bytes)
    ):
        raise ValueError("endpoint rows must be a sequence")
    rows = list(endpoint_rows)
    if len(rows) != _ENDPOINTS or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("scene utility aggregation requires exactly four endpoint rows")
    covered = np.asarray(
        [
            _finite_float(row.get("covered_utility"), name="covered utility")
            for row in rows
        ],
        dtype=np.float64,
    )
    teacher = np.asarray(
        [
            _finite_float(
                row.get("teacher_covered_utility"),
                name="teacher covered utility",
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    full = np.asarray(
        [
            _finite_float(row.get("full_scene_utility"), name="full-scene utility")
            for row in rows
        ],
        dtype=np.float64,
    )
    mean_covered = float(np.mean(covered))
    mean_teacher = float(np.mean(teacher))
    mean_full = float(np.mean(full))
    if not math.isfinite(mean_teacher) or mean_teacher <= 0.0:
        raise ValueError("mean teacher covered utility must be finite and positive")
    retention = float(mean_covered / mean_teacher)
    if not math.isfinite(retention):
        raise ValueError("teacher retention must be finite")
    return {
        "mean_covered_utility": mean_covered,
        "mean_teacher_covered_utility": mean_teacher,
        "teacher_retention": retention,
        "mean_full_scene_utility": mean_full,
    }


def _validate_pose_stack(value: np.ndarray, *, name: str) -> np.ndarray:
    poses = np.asarray(value)
    if (
        poses.ndim != 3
        or poses.shape[-2:] != (4, 4)
        or poses.dtype != np.float64
        or not np.isfinite(poses).all()
    ):
        raise ValueError(f"{name} must be a finite float64 pose stack")
    if not np.allclose(
        poses[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must contain homogeneous poses")
    rotations = poses[:, :3, :3]
    gram = np.einsum("...ji,...jk->...ik", rotations, rotations)
    determinants = np.linalg.det(rotations)
    if (
        not np.allclose(gram, np.eye(3), atol=2e-6, rtol=0.0)
        or not np.allclose(determinants, 1.0, atol=2e-6, rtol=0.0)
        or np.any(determinants <= 0.0)
    ):
        raise ValueError(f"{name} must contain proper SO(3) rotations")
    return poses


def _so3_error_deg(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "...ij,...kj->...ik",
        candidate[..., :3, :3],
        reference[..., :3, :3],
    )
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.rad2deg(np.arccos(cosine)).astype(np.float64, copy=False)


def _so3_geodesic_delta_deg(
    candidate: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Return stable SO(3) deltas, with exact decoded equality mapped to zero."""
    candidate_rotations = np.asarray(candidate, dtype=np.float64)[..., :3, :3]
    reference_rotations = np.asarray(reference, dtype=np.float64)[..., :3, :3]
    if candidate_rotations.shape != reference_rotations.shape:
        raise ValueError("rotation delta inputs must have matching shapes")
    flat_candidate = candidate_rotations.reshape(-1, 3, 3)
    flat_reference = reference_rotations.reshape(-1, 3, 3)
    exact = np.all(
        flat_candidate.view(np.uint64) == flat_reference.view(np.uint64),
        axis=(1, 2),
    )
    angles = np.zeros(len(flat_candidate), dtype=np.float64)
    active = np.flatnonzero(~exact)
    if len(active):
        candidate_u, _, candidate_vh = np.linalg.svd(flat_candidate[active])
        reference_u, _, reference_vh = np.linalg.svd(flat_reference[active])
        projected_candidate = candidate_u @ candidate_vh
        projected_reference = reference_u @ reference_vh
        for projected, left, right in (
            (projected_candidate, candidate_u, candidate_vh),
            (projected_reference, reference_u, reference_vh),
        ):
            reflected = np.linalg.det(projected) < 0.0
            if np.any(reflected):
                left[reflected, :, -1] *= -1.0
                projected[reflected] = left[reflected] @ right[reflected]
        relative = projected_candidate @ np.swapaxes(projected_reference, -1, -2)
        sine = np.linalg.norm(
            relative - np.swapaxes(relative, -1, -2), axis=(1, 2)
        ) / (2.0 * math.sqrt(2.0))
        cosine = np.clip(
            (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
        angles[active] = np.rad2deg(
            np.arctan2(np.clip(sine, 0.0, 1.0), cosine)
        )
    return angles.reshape(candidate_rotations.shape[:-2])


def _rms_center_error(
    candidate: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray
) -> float:
    if mask.shape != (_FRAMES,) or mask.dtype != np.bool_ or not np.any(mask):
        raise ValueError("RMS mask must cover at least one of 500 frames")
    delta = candidate[mask, :3, 3] - ground_truth[mask, :3, 3]
    value = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("translation RMS must be finite and nonnegative")
    return value


def _diagnostic_match(
    actual: np.ndarray, recorded: np.ndarray, *, name: str, mask: np.ndarray | None = None
) -> None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(recorded)
    if mask is not None:
        left = left[mask]
        right = right[mask]
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{name} diagnostic is malformed")
    if not np.allclose(left, right, atol=_DIAGNOSTIC_ATOL, rtol=_DIAGNOSTIC_ATOL):
        raise ValueError(f"{name} diagnostic mismatch")


def _authenticate_published_sample(
    sample: PublishedTranslationSample,
) -> tuple[dict[str, Path], dict[str, bytes], dict[str, str]]:
    if not isinstance(sample, PublishedTranslationSample):
        raise ValueError("sample must be a PublishedTranslationSample")
    paths = {
        "long": Path(sample.long_path),
        "short": Path(sample.short_path),
        "quality": Path(sample.quality_path),
        "target": Path(sample.target_path),
    }
    expected = {
        "long": _canonical_digest(sample.long_sha256, name="published long digest"),
        "short": _canonical_digest(sample.short_sha256, name="published short digest"),
        "quality": _canonical_digest(
            sample.quality_sha256, name="published quality digest"
        ),
        "target": _canonical_digest(sample.target_sha256, name="published target digest"),
    }
    snapshots: dict[str, bytes] = {}
    actual: dict[str, str] = {}
    for name, path in paths.items():
        _reject_symlink_components(path)
        if not path.is_file():
            raise ValueError(f"published {name} artifact must be a regular file")
        try:
            with path.open("rb") as handle:
                snapshot = handle.read()
        except OSError as error:
            raise ValueError(f"could not snapshot published {name} artifact") from error
        snapshots[name] = snapshot
        actual[name] = hashlib.sha256(snapshot).hexdigest()
    for name in paths:
        if actual[name] != expected[name]:
            raise ValueError(f"published {name} digest mismatch")
    return paths, snapshots, actual


def _require_published_paths_unchanged(
    paths: Mapping[str, Path], authenticated_hashes: Mapping[str, str]
) -> None:
    for name, path in paths.items():
        if _sha256_file(path) != authenticated_hashes[name]:
            raise ValueError(
                f"published {name} artifact changed during evaluation"
            )


def _validate_sample_identity(
    sample: PublishedTranslationSample,
    bundle: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    long = bundle["long"]
    scene = str(long["scene"])
    sample_id = str(long["sample_id"])
    if sample.scene != scene or sample.sample_id != sample_id:
        raise ValueError("published sample identity does not match its bound bundle")
    if sample_id != f"{scene}:frames_500":
        raise ValueError("published sample ID is not canonical")
    if sample.role != calibration_role(scene):
        raise ValueError("published sample role does not match the frozen calibration split")


def evaluate_translation_sample(
    sample: PublishedTranslationSample,
) -> dict[str, object]:
    """Recompute one sample entirely from its four authenticated artifacts."""
    paths, snapshots, actual_hashes = _authenticate_published_sample(sample)
    bundle = load_bound_bundle_bytes(
        snapshots["long"],
        snapshots["short"],
        snapshots["target"],
        snapshots["quality"],
    )
    _validate_sample_identity(sample, bundle)
    long = bundle["long"]
    target = bundle["target"]
    quality = bundle["quality"]

    baseline_pose = long["baseline_pose_encoding"]
    endpoints = target["translation_endpoints"]
    coverage = target["coverage_mask"] != 0
    scale = float(long["prediction_scale"])
    positive_zero = [
        bool(np.all(endpoints[index].view(np.uint32)[~coverage[index]] == 0))
        for index in range(_ENDPOINTS)
    ]
    corrected_pose = apply_translation_endpoint(baseline_pose, endpoints, scale=scale)
    expected_tail = np.broadcast_to(baseline_pose[None], corrected_pose.shape)
    quaternion_equal = [
        corrected_pose[index, :, 3:7].tobytes(order="C")
        == expected_tail[index, :, 3:7].tobytes(order="C")
        for index in range(_ENDPOINTS)
    ]
    fov_equal = [
        corrected_pose[index, :, 7:9].tobytes(order="C")
        == expected_tail[index, :, 7:9].tobytes(order="C")
        for index in range(_ENDPOINTS)
    ]

    stacked_pose = np.concatenate((baseline_pose[None], corrected_pose), axis=0)
    try:
        with torch.no_grad():
            decoded_tensor = pose_encoding_to_c2w(torch.from_numpy(stacked_pose))
    except (RuntimeError, ValueError) as error:
        raise ValueError("baseline and corrected pose encodings could not be decoded") from error
    decoded = decoded_tensor.detach().to(device="cpu", dtype=torch.float64).numpy()
    if decoded.shape != (1 + _ENDPOINTS, _FRAMES, 4, 4):
        raise ValueError("pose decoder returned a noncanonical shape")
    decoded_baseline = _validate_pose_stack(decoded[0], name="decoded baseline")
    decoded_corrected = decoded[1:]
    for endpoint in range(_ENDPOINTS):
        _validate_pose_stack(
            decoded_corrected[endpoint], name=f"decoded endpoint {endpoint}"
        )

    stored_baseline = long["baseline_c2w"]
    baseline_center_error = float(
        np.max(
            np.linalg.norm(
                decoded_baseline[:, :3, 3] - stored_baseline[:, :3, 3], axis=1
            )
        )
        / scale
    )
    baseline_rotation_error = float(
        np.max(_so3_geodesic_delta_deg(decoded_baseline, stored_baseline))
    )
    if (
        not math.isfinite(baseline_center_error)
        or baseline_center_error > _BASELINE_CENTER_ATOL
        or not math.isfinite(baseline_rotation_error)
        or baseline_rotation_error > _BASELINE_ROTATION_ATOL
    ):
        raise ValueError("decoded baseline does not match the bound baseline C2W")

    oracle = decode_saved_oracle(quality)
    if oracle.scene != sample.scene or oracle.fit_count != _FRAMES:
        raise ValueError("saved FrozenOracle does not bind the complete sample")
    ground_truth = _validate_pose_stack(quality["gt_c2w"], name="ground truth")
    recomputed_gt_scale = prediction_scale(ground_truth)
    recorded_gt_scale = float(quality["gt_scene_scale"])
    if not math.isclose(
        recomputed_gt_scale,
        recorded_gt_scale,
        rel_tol=_DIAGNOSTIC_ATOL,
        abs_tol=_DIAGNOSTIC_ATOL,
    ):
        raise ValueError("saved GT scale diagnostic mismatch")

    baseline_aligned = apply_frozen_oracle(oracle, decoded_baseline)
    corrected_aligned = apply_frozen_oracle(
        oracle, decoded_corrected.reshape(_ENDPOINTS * _FRAMES, 4, 4)
    ).reshape(_ENDPOINTS, _FRAMES, 4, 4)
    teacher_raw = np.broadcast_to(
        stored_baseline[None], (_ENDPOINTS, _FRAMES, 4, 4)
    ).copy()
    teacher_raw[:, :, :3, 3] = target["teacher_centers_raw_filled"]
    teacher_aligned = apply_frozen_oracle(
        oracle, teacher_raw.reshape(_ENDPOINTS * _FRAMES, 4, 4)
    ).reshape(_ENDPOINTS, _FRAMES, 4, 4)

    baseline_translation_error = (
        np.linalg.norm(
            baseline_aligned[:, :3, 3] - ground_truth[:, :3, 3], axis=1
        )
        / recorded_gt_scale
    )
    baseline_rotation_error_deg = _so3_error_deg(baseline_aligned, ground_truth)
    _diagnostic_match(
        baseline_translation_error,
        quality["baseline_translation_error_normalized"],
        name="baseline translation",
    )
    _diagnostic_match(
        baseline_rotation_error_deg,
        quality["baseline_rotation_error_deg"],
        name="baseline rotation",
    )

    full_mask = np.ones(_FRAMES, dtype=np.bool_)
    baseline_full_rms = _rms_center_error(
        baseline_aligned, ground_truth, full_mask
    )
    if baseline_full_rms <= 0.0:
        raise ValueError("full-scene baseline RMS must be finite and positive")
    endpoint_rows: list[dict[str, object]] = []
    replayed_teacher_utilities = np.empty(_ENDPOINTS, dtype=np.float64)
    filled_centers = target["teacher_centers_raw_filled"]
    baseline_raw_centers = stored_baseline[:, :3, 3]
    for endpoint in range(_ENDPOINTS):
        mask = coverage[endpoint]
        if not np.any(mask):
            raise ValueError(f"endpoint {endpoint} must cover at least one frame")
        uncovered = ~mask
        if np.any(uncovered) and not np.array_equal(
            filled_centers[endpoint, uncovered], baseline_raw_centers[uncovered]
        ):
            raise ValueError("uncovered raw teacher centers must equal the baseline")

        teacher_translation_error = (
            np.linalg.norm(
                teacher_aligned[endpoint, :, :3, 3]
                - ground_truth[:, :3, 3],
                axis=1,
            )
            / recorded_gt_scale
        )
        teacher_rotation_error = _so3_error_deg(
            teacher_aligned[endpoint], ground_truth
        )
        _diagnostic_match(
            teacher_translation_error,
            quality["teacher_translation_error_normalized"][endpoint],
            name=f"endpoint {endpoint} teacher translation",
            mask=mask,
        )
        _diagnostic_match(
            teacher_rotation_error,
            quality["teacher_rotation_error_deg"][endpoint],
            name=f"endpoint {endpoint} teacher rotation",
            mask=mask,
        )

        baseline_covered_rms = _rms_center_error(
            baseline_aligned, ground_truth, mask
        )
        if baseline_covered_rms <= 0.0:
            raise ValueError(
                f"endpoint {endpoint} covered baseline RMS must be finite and positive"
            )
        corrected_covered_rms = _rms_center_error(
            corrected_aligned[endpoint], ground_truth, mask
        )
        teacher_covered_rms = _rms_center_error(
            teacher_aligned[endpoint], ground_truth, mask
        )
        corrected_full_rms = _rms_center_error(
            corrected_aligned[endpoint], ground_truth, full_mask
        )
        covered_utility = _relative_utility(
            baseline_covered_rms,
            corrected_covered_rms,
            name=f"endpoint {endpoint} covered utility",
        )
        teacher_utility = _relative_utility(
            baseline_covered_rms,
            teacher_covered_rms,
            name=f"endpoint {endpoint} teacher utility",
        )
        full_utility = _relative_utility(
            baseline_full_rms,
            corrected_full_rms,
            name=f"endpoint {endpoint} full-scene utility",
        )
        replayed_teacher_utilities[endpoint] = teacher_utility

        covered_error = float(
            np.max(
                np.linalg.norm(
                    decoded_corrected[endpoint, mask, :3, 3]
                    - filled_centers[endpoint, mask],
                    axis=1,
                )
            )
            / scale
        )
        uncovered_drift = 0.0
        if np.any(uncovered):
            uncovered_drift = float(
                np.max(
                    np.linalg.norm(
                        decoded_corrected[endpoint, uncovered, :3, 3]
                        - decoded_baseline[uncovered, :3, 3],
                        axis=1,
                    )
                )
                / scale
            )
        rotation_delta = float(
            np.max(
                _so3_geodesic_delta_deg(
                    decoded_corrected[endpoint], decoded_baseline
                )
            )
        )
        endpoint_rms = float(
            np.sqrt(
                np.mean(
                    np.sum(endpoints[endpoint].astype(np.float64) ** 2, axis=1)
                )
            )
        )
        numeric = (
            covered_utility,
            teacher_utility,
            full_utility,
            covered_error,
            uncovered_drift,
            rotation_delta,
            endpoint_rms,
        )
        endpoint_rows.append(
            {
                "endpoint_id": endpoint,
                "covered_utility": covered_utility,
                "teacher_covered_utility": teacher_utility,
                "full_scene_utility": full_utility,
                "covered_roundtrip_fraction": covered_error,
                "uncovered_drift_fraction": uncovered_drift,
                "rotation_delta_deg": rotation_delta,
                "quaternion_bytes_equal": quaternion_equal[endpoint],
                "fov_bytes_equal": fov_equal[endpoint],
                "uncovered_positive_zero": positive_zero[endpoint],
                "endpoint_rms": endpoint_rms,
                "coverage_fraction": float(np.mean(mask)),
                "all_finite": bool(np.isfinite(numeric).all()),
            }
        )

    if not np.allclose(
        replayed_teacher_utilities,
        quality["variant_utilities"],
        atol=_DIAGNOSTIC_ATOL,
        rtol=_DIAGNOSTIC_ATOL,
    ):
        raise ValueError("variant utility diagnostic mismatch")
    aggregates = _aggregate_scene_utilities(endpoint_rows)
    provenance = {
        "long_sha256": actual_hashes["long"],
        "short_sha256": actual_hashes["short"],
        "quality_sha256": actual_hashes["quality"],
        "target_sha256": actual_hashes["target"],
        "source_sha256": str(long["source_sha256"]),
        "checkpoint_sha256": str(long["checkpoint_sha256"]),
        "teacher_reference_sha256": str(target["teacher_reference_sha256"]),
        "git_commit": str(long["git_commit"]),
    }
    result = {
        "scene": sample.scene,
        "sample_id": sample.sample_id,
        "role": sample.role,
        "endpoint_count": _ENDPOINTS,
        "endpoint_ids": list(range(_ENDPOINTS)),
        "endpoints": endpoint_rows,
        **aggregates,
        "max_covered_roundtrip_fraction": float(
            max(row["covered_roundtrip_fraction"] for row in endpoint_rows)
        ),
        "max_uncovered_drift_fraction": float(
            max(row["uncovered_drift_fraction"] for row in endpoint_rows)
        ),
        "max_rotation_delta_deg": float(
            max(row["rotation_delta_deg"] for row in endpoint_rows)
        ),
        "quaternion_bytes_equal": bool(all(quaternion_equal)),
        "fov_bytes_equal": bool(all(fov_equal)),
        "uncovered_positive_zero": bool(all(positive_zero)),
        "all_finite": bool(all(row["all_finite"] for row in endpoint_rows)),
        "provenance": provenance,
    }
    _require_published_paths_unchanged(paths, actual_hashes)
    return result


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact Boolean")
    return value


def _same_summary(actual: float, recorded: object, *, name: str) -> float:
    value = _finite_float(recorded, name=name)
    if not math.isclose(actual, value, rel_tol=1e-15, abs_tol=1e-15):
        raise ValueError(f"scene {name} summary does not match endpoint evidence")
    return value


def _validated_scene_metrics(row: Mapping[str, object]) -> dict[str, object]:
    scene = row.get("scene")
    if not isinstance(scene, str) or scene not in _CALIBRATION_SCENES:
        raise ValueError("scene metrics contain a scene outside the exact calibration cohort")
    if row.get("sample_id") != f"{scene}:frames_500":
        raise ValueError("scene metrics sample ID mismatch")
    if row.get("role") != calibration_role(scene):
        raise ValueError("scene metrics calibration role mismatch")
    if type(row.get("endpoint_count")) is not int or row.get("endpoint_count") != 4:
        raise ValueError("scene metrics require exactly four endpoints")
    if row.get("endpoint_ids") != [0, 1, 2, 3]:
        raise ValueError("scene endpoint IDs must be exactly [0, 1, 2, 3]")
    endpoints_value = row.get("endpoints")
    if not isinstance(endpoints_value, Sequence) or isinstance(
        endpoints_value, (str, bytes)
    ):
        raise ValueError("scene endpoints must be a sequence")
    endpoints = list(endpoints_value)
    if len(endpoints) != 4 or any(not isinstance(value, Mapping) for value in endpoints):
        raise ValueError("scene metrics require exactly four endpoint rows")
    by_id: dict[int, Mapping[str, object]] = {}
    numeric_names = (
        "covered_utility",
        "teacher_covered_utility",
        "full_scene_utility",
        "covered_roundtrip_fraction",
        "uncovered_drift_fraction",
        "rotation_delta_deg",
        "endpoint_rms",
        "coverage_fraction",
    )
    boolean_names = (
        "quaternion_bytes_equal",
        "fov_bytes_equal",
        "uncovered_positive_zero",
        "all_finite",
    )
    for endpoint in endpoints:
        endpoint_id = endpoint.get("endpoint_id")
        if type(endpoint_id) is not int or endpoint_id not in range(4) or endpoint_id in by_id:
            raise ValueError("scene endpoint evidence must contain unique IDs 0 through 3")
        for name in numeric_names:
            value = _finite_float(endpoint.get(name), name=f"endpoint {name}")
            if name in {
                "covered_roundtrip_fraction",
                "uncovered_drift_fraction",
                "rotation_delta_deg",
                "endpoint_rms",
                "coverage_fraction",
            } and value < 0.0:
                raise ValueError(f"endpoint {name} must be nonnegative")
            if name == "coverage_fraction" and not 0.0 < value <= 1.0:
                raise ValueError("endpoint coverage fraction must be in (0, 1]")
        for name in boolean_names:
            _require_bool(endpoint.get(name), name=f"endpoint {name}")
        by_id[endpoint_id] = endpoint
    if set(by_id) != set(range(4)):
        raise ValueError("scene endpoint evidence must contain IDs 0 through 3")
    ordered = [by_id[index] for index in range(4)]
    aggregate = _aggregate_scene_utilities(ordered)
    for name in (
        "mean_covered_utility",
        "mean_teacher_covered_utility",
        "teacher_retention",
        "mean_full_scene_utility",
    ):
        _same_summary(aggregate[name], row.get(name), name=name)
    maxima = {
        "max_covered_roundtrip_fraction": max(
            float(endpoint["covered_roundtrip_fraction"]) for endpoint in ordered
        ),
        "max_uncovered_drift_fraction": max(
            float(endpoint["uncovered_drift_fraction"]) for endpoint in ordered
        ),
        "max_rotation_delta_deg": max(
            float(endpoint["rotation_delta_deg"]) for endpoint in ordered
        ),
    }
    for name, actual in maxima.items():
        _same_summary(actual, row.get(name), name=name)
    summary_booleans = {
        "quaternion_bytes_equal": all(
            bool(endpoint["quaternion_bytes_equal"]) for endpoint in ordered
        ),
        "fov_bytes_equal": all(bool(endpoint["fov_bytes_equal"]) for endpoint in ordered),
        "uncovered_positive_zero": all(
            bool(endpoint["uncovered_positive_zero"]) for endpoint in ordered
        ),
        "all_finite": all(bool(endpoint["all_finite"]) for endpoint in ordered),
    }
    for name, actual in summary_booleans.items():
        if _require_bool(row.get(name), name=f"scene {name}") is not actual:
            raise ValueError(f"scene {name} summary does not match endpoint evidence")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("scene provenance must be a mapping")
    for name in (
        "long_sha256",
        "short_sha256",
        "quality_sha256",
        "target_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "teacher_reference_sha256",
    ):
        _canonical_digest(provenance.get(name), name=f"scene provenance {name}")
    commit = provenance.get("git_commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("scene provenance Git commit is malformed")
    return {
        "scene": scene,
        **aggregate,
        **maxima,
        **summary_booleans,
    }


def _bind_scene_metrics_to_sample(
    row: Mapping[str, object], sample: PublishedTranslationSample
) -> None:
    if row.get("scene") != sample.scene:
        raise ValueError("scene metrics do not match their cohort scene")
    if row.get("sample_id") != sample.sample_id:
        raise ValueError("scene metrics sample ID does not match the cohort sample")
    if row.get("role") != sample.role:
        raise ValueError("scene metrics role does not match the cohort sample")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("scene provenance must be a mapping")
    for field in (
        "long_sha256",
        "short_sha256",
        "quality_sha256",
        "target_sha256",
    ):
        recorded = _canonical_digest(
            provenance.get(field), name=f"scene provenance {field}"
        )
        published = _canonical_digest(
            getattr(sample, field), name=f"cohort sample {field}"
        )
        if recorded != published:
            raise ValueError(
                f"scene provenance {field} does not match the cohort sample digest"
            )


def classify_stage_a_prime(
    scene_metrics: Sequence[Mapping[str, object]],
    *,
    cohort: Sequence[PublishedTranslationSample],
    physical_leakage_clean: bool,
) -> dict[str, object]:
    """Validate complete evidence and classify only the frozen ten-scene cohort."""
    validate_calibration_cohort(cohort)
    leakage = _require_bool(
        physical_leakage_clean, name="physical leakage audit"
    )
    if not isinstance(scene_metrics, Sequence) or isinstance(
        scene_metrics, (str, bytes)
    ):
        raise ValueError("scene metrics must be a sequence")
    rows = list(scene_metrics)
    if len(rows) != len(_CALIBRATION_SCENES) or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise ValueError("Stage A-prime requires the exact ten-scene calibration cohort")
    scenes = [row.get("scene") for row in rows]
    if len(set(scenes)) != len(rows) or set(scenes) != set(_CALIBRATION_SCENES):
        raise ValueError("scene metrics do not bind the exact ten-scene calibration cohort")
    cohort_by_scene = {sample.scene: sample for sample in cohort}
    validated = []
    for row in rows:
        sample = cohort_by_scene[str(row["scene"])]
        _bind_scene_metrics_to_sample(row, sample)
        validated.append(_validated_scene_metrics(row))
    retention = np.asarray(
        [row["teacher_retention"] for row in validated], dtype=np.float64
    )
    full_utility = np.asarray(
        [row["mean_full_scene_utility"] for row in validated], dtype=np.float64
    )
    covered_roundtrip = np.asarray(
        [row["max_covered_roundtrip_fraction"] for row in validated],
        dtype=np.float64,
    )
    uncovered_drift = np.asarray(
        [row["max_uncovered_drift_fraction"] for row in validated],
        dtype=np.float64,
    )
    rotation_delta = np.asarray(
        [row["max_rotation_delta_deg"] for row in validated], dtype=np.float64
    )
    mean_retention = float(np.mean(retention))
    mean_full = float(np.mean(full_utility))
    minimum_full = float(np.min(full_utility))
    positive_count = int(np.count_nonzero(full_utility > 0.0))
    gates = {
        "finite": bool(all(row["all_finite"] for row in validated)),
        "uncovered_positive_zero": bool(
            all(row["uncovered_positive_zero"] for row in validated)
        ),
        "quaternion_bytes_equal": bool(
            all(row["quaternion_bytes_equal"] for row in validated)
        ),
        "fov_bytes_equal": bool(all(row["fov_bytes_equal"] for row in validated)),
        "covered_roundtrip": float(np.max(covered_roundtrip)) < 1e-5,
        "uncovered_anchor": float(np.max(uncovered_drift)) < 1e-8,
        "rotation_guard": float(np.max(rotation_delta)) <= 1e-6,
        "teacher_retention": mean_retention >= 0.95,
        "positive_scene_count": positive_count == len(_CALIBRATION_SCENES),
        "positive_mean": mean_full > 0.0,
        "minimum_full_utility": minimum_full >= 0.0,
        "physical_leakage_clean": leakage is True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "classification": (
            "TRANSLATION_ENDPOINTS_READY"
            if not failed
            else "TRANSLATION_ENDPOINTS_FAILED"
        ),
        "failed_gates": failed,
        "gates": gates,
        "scene_count": len(_CALIBRATION_SCENES),
        "endpoint_count": len(_CALIBRATION_SCENES) * _ENDPOINTS,
        "mean_teacher_retention": mean_retention,
        "mean_full_scene_utility": mean_full,
        "minimum_full_scene_utility": minimum_full,
        "positive_scene_count": positive_count,
    }


__all__ = [
    "classify_stage_a_prime",
    "decode_saved_oracle",
    "evaluate_translation_sample",
]
