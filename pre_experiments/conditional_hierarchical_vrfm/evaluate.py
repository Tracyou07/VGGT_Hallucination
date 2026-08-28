"""Independent fixed-gauge metric replay and fail-closed Stage A gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    FrozenOracle,
    apply_frozen_oracle,
    fit_frozen_oracle,  # exported only so tests can prove evaluation never refits
)
from pre_experiments.conditional_hierarchical_vrfm.basis import (
    canonical_basis_sha256,
    temporal_dct_basis,
)


def _scalar(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar")
    return str(array)


def _same(left: object, right: object, name: str) -> None:
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise ValueError(f"cross-artifact {name} mismatch")


def _rms_centers(candidate: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    delta = candidate[mask, :3, 3] - gt[mask, :3, 3]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _relative_utility(baseline_rms: float, candidate_rms: float, name: str) -> float:
    if not np.isfinite((baseline_rms, candidate_rms)).all() or baseline_rms <= 0.0:
        raise ValueError(f"{name} baseline RMS must be finite and positive")
    return float((baseline_rms - candidate_rms) / baseline_rms)


def _mean_so3_error_deg(candidate: np.ndarray, gt: np.ndarray) -> float:
    relative = np.einsum("fij,fkj->fik", candidate[:, :3, :3], gt[:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(cosine))))


def _validate_so3(poses: np.ndarray, name: str) -> None:
    rotations = np.asarray(poses)[..., :3, :3]
    gram = np.einsum("...ji,...jk->...ik", rotations, rotations)
    identity = np.eye(3, dtype=np.float64)
    determinant = np.linalg.det(rotations)
    if (
        not np.isfinite(rotations).all()
        or not np.allclose(gram, identity, atol=1e-5, rtol=1e-5)
        or not np.allclose(determinant, 1.0, atol=1e-5, rtol=1e-5)
    ):
        raise ValueError(f"{name} must contain proper SO(3) rotations")


def _oracle(labels: Mapping[str, object]) -> FrozenOracle:
    return FrozenOracle(
        scene=_scalar(labels["oracle_scene"], "oracle_scene"),
        frame_digest=_scalar(labels["oracle_frame_digest"], "oracle_frame_digest"),
        fit_count=int(np.asarray(labels["oracle_fit_count"])),
        scale=float(np.asarray(labels["oracle_scale"])),
        rotation=tuple(tuple(float(v) for v in row) for row in np.asarray(labels["oracle_rotation"])),
        translation=tuple(float(v) for v in np.asarray(labels["oracle_translation"])),
        rank=int(np.asarray(labels["oracle_rank"])),
        condition=float(np.asarray(labels["oracle_condition"])),
        transform_digest=_scalar(labels["oracle_digest"], "oracle_digest"),
    )


def evaluate_latent_targets(
    long_context: Mapping[str, object],
    latent_targets: Mapping[str, object],
    teacher_labels: Mapping[str, object],
) -> dict[str, object]:
    """Replay every metric using the sidecar's single saved Sim(3), never fitting one."""
    scene = _scalar(long_context["scene"], "long_context.scene")
    for artifact, name in ((latent_targets, "target"), (teacher_labels, "teacher")):
        if _scalar(artifact["scene"], f"{name}.scene") != scene:
            raise ValueError(f"cross-artifact {name} scene mismatch")
        _same(long_context["frame_ids"], artifact["frame_ids"], f"{name} frame IDs")
    _same(long_context["source_sha256"], latent_targets["source_sha256"], "target source digest")
    _same(long_context["source_sha256"], teacher_labels["source_sha256"], "teacher source digest")
    _same(teacher_labels["checkpoint_sha256"], latent_targets["checkpoint_sha256"], "checkpoint digest")
    _same(teacher_labels["git_commit"], latent_targets["git_commit"], "Git commit")
    if _scalar(latent_targets["basis_sha256"], "basis_sha256") != canonical_basis_sha256():
        raise ValueError("target basis digest is not canonical")
    teacher_artifact_digest = teacher_labels.get("artifact_sha256")
    if not isinstance(teacher_artifact_digest, str) or len(teacher_artifact_digest) != 64:
        raise ValueError("teacher artifact digest is required for evaluation")
    if _scalar(latent_targets["teacher_sha256"], "teacher_sha256") != teacher_artifact_digest:
        raise ValueError("target teacher digest does not match teacher artifact")

    variant_ids = np.asarray(latent_targets["teacher_variant_ids"])
    if not np.array_equal(variant_ids, np.arange(4)):
        raise ValueError("target variant IDs must be exactly [0,1,2,3]")
    _same(latent_targets["teacher_window_masks"], teacher_labels["window_masks"], "window masks")
    coverage = np.asarray(teacher_labels["coverage_weights"], dtype=np.float64)
    _same(latent_targets["coverage_masks"], (coverage > 0.0).astype(np.uint8), "coverage masks")
    if coverage.shape != (4, 500) or not np.isfinite(coverage).all() or np.any(coverage < 0.0):
        raise ValueError("teacher coverage must be finite nonnegative [4,500]")

    baseline_raw = np.asarray(long_context["baseline_c2w"], dtype=np.float64)
    _same(baseline_raw, teacher_labels["baseline_c2w_raw"], "frozen baseline")
    corrected_raw = np.asarray(latent_targets["decoded_c2w_raw"], dtype=np.float64)
    gt = np.asarray(teacher_labels["gt_c2w"], dtype=np.float64)
    fused = np.asarray(teacher_labels["fused_c2w"], dtype=np.float64)
    coefficients = np.asarray(latent_targets["residual_coefficients"], dtype=np.float32)
    if corrected_raw.shape != (4, 500, 4, 4) or gt.shape != (500, 4, 4) or fused.shape != (4, 500, 4, 4):
        raise ValueError("evaluation pose shapes are invalid")
    if coefficients.shape != (4, 32, 2048):
        raise ValueError("evaluation coefficient shape is invalid")
    _validate_so3(baseline_raw, "frozen baseline")
    _validate_so3(gt, "ground truth")
    oracle = _oracle(teacher_labels)
    if oracle.scene != scene:
        raise ValueError("saved oracle scene mismatch")
    baseline = apply_frozen_oracle(oracle, baseline_raw)
    full = np.ones(500, dtype=bool)
    baseline_full_rms = _rms_centers(baseline, gt, full)
    baseline_rotation = _mean_so3_error_deg(baseline, gt)
    gt_scene_scale = float(np.asarray(teacher_labels["gt_scene_scale"]))
    if not np.isfinite(gt_scene_scale) or gt_scene_scale <= 0.0:
        raise ValueError("saved GT scene scale must be finite and positive")
    basis = temporal_dct_basis().numpy()
    variants: list[dict[str, float | int | bool]] = []
    teacher_utilities: list[float] = []
    for index in range(4):
        mask = coverage[index] > 0.0
        if not np.any(mask):
            raise ValueError("each teacher variant must cover frames")
        if not np.isfinite(corrected_raw[index]).all() or not np.isfinite(fused[index, mask]).all():
            raise ValueError("evaluation contains non-finite covered poses")
        _validate_so3(corrected_raw[index], "decoded target")
        _validate_so3(fused[index, mask], "covered teacher")
        corrected = apply_frozen_oracle(oracle, corrected_raw[index])
        baseline_covered_rms = _rms_centers(baseline, gt, mask)
        covered_utility = _relative_utility(
            baseline_covered_rms, _rms_centers(corrected, gt, mask), "covered utility"
        )
        teacher_utility = _relative_utility(
            baseline_covered_rms, _rms_centers(fused[index], gt, mask), "teacher utility"
        )
        teacher_utilities.append(teacher_utility)
        full_utility = _relative_utility(
            baseline_full_rms, _rms_centers(corrected, gt, full), "full-scene utility"
        )
        rotation_delta = _mean_so3_error_deg(corrected, gt) - baseline_rotation
        uncovered = ~mask
        drift = 0.0
        if np.any(uncovered):
            center_delta = corrected[uncovered, :3, 3] - baseline[uncovered, :3, 3]
            drift = float(np.sqrt(np.mean(np.sum(center_delta * center_delta, axis=1))) / gt_scene_scale)
        residual = basis @ coefficients[index]
        residual_rms = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)))
        values = (covered_utility, full_utility, rotation_delta, drift, residual_rms, teacher_utility)
        variants.append({
            "variant_id": index,
            "covered_utility": covered_utility,
            "full_scene_utility": full_utility,
            "rotation_delta_deg": rotation_delta,
            "uncovered_drift_fraction": drift,
            "residual_rms": residual_rms,
            "teacher_covered_utility": teacher_utility,
            "all_finite": bool(np.isfinite(values).all()),
        })
    denominator = float(np.mean(teacher_utilities))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("teacher-reference utility must be finite and positive")
    mean_covered = float(np.mean([row["covered_utility"] for row in variants]))
    return {
        "scene": scene,
        "variant_count": 4,
        "variants": variants,
        "mean_covered_utility": mean_covered,
        "mean_full_scene_utility": float(np.mean([row["full_scene_utility"] for row in variants])),
        "mean_rotation_delta_deg": float(np.mean([row["rotation_delta_deg"] for row in variants])),
        "uncovered_drift_fraction": float(max(row["uncovered_drift_fraction"] for row in variants)),
        "mean_residual_rms": float(np.mean([row["residual_rms"] for row in variants])),
        "teacher_retention": mean_covered / denominator,
        "all_finite": bool(all(row["all_finite"] for row in variants)),
    }


def classify_stage_a(
    scene_metrics: Sequence[Mapping[str, object]],
    *,
    expected_scenes: Sequence[str],
    prediction_contract_clean: bool,
) -> dict[str, object]:
    expected = tuple(expected_scenes)
    if len(expected) != 10 or len(set(expected)) != 10:
        raise ValueError("Stage A requires exactly ten unique expected scenes")
    rows = list(scene_metrics)
    actual = [str(row.get("scene", "")) for row in rows]
    if len(rows) != 10 or len(set(actual)) != 10 or set(actual) != set(expected):
        raise ValueError("scene metrics do not bind the exact expected scenes")
    numeric = (
        "teacher_retention", "mean_full_scene_utility", "mean_rotation_delta_deg",
        "uncovered_drift_fraction",
    )
    for row in rows:
        if row.get("variant_count") != 4 or type(row.get("all_finite")) is not bool:
            raise ValueError("scene metrics require four variants and a Boolean finite flag")
        for name in numeric:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
                raise ValueError("scene metrics must contain finite numeric gate inputs")
    retention = np.asarray([row["teacher_retention"] for row in rows], dtype=np.float64)
    utility = np.asarray([row["mean_full_scene_utility"] for row in rows], dtype=np.float64)
    rotation = np.asarray([row["mean_rotation_delta_deg"] for row in rows], dtype=np.float64)
    drift = np.asarray([row["uncovered_drift_fraction"] for row in rows], dtype=np.float64)
    gates = {
        "finite": all(bool(row["all_finite"]) for row in rows),
        "teacher_retention": float(np.mean(retention)) >= 0.70,
        "positive_mean": float(np.mean(utility)) > 0.0,
        "positive_scene_count": int(np.count_nonzero(utility > 0.0)) >= 8,
        "per_scene_harm": float(np.min(utility)) >= -0.01,
        "rotation_guard": float(np.mean(rotation)) <= 0.1,
        "uncovered_anchor": float(np.max(drift)) <= 0.005,
        "leakage_audit": prediction_contract_clean is True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "classification": "LATENT_TARGETS_READY" if not failed else "LATENT_LIFT_FAILED",
        "failed_gates": failed,
        "gates": gates,
        "scene_count": 10,
        "variant_count": 40,
    }
