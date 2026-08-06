"""Measure end-to-end Camera Head hidden-unit causal preferences."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.causal_preference import (
    GROUPS,
    activation_rms,
    central_output_jacobians,
    project_hidden_effects,
)
from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
    load_causal_scene_effects,
    save_causal_scene_effects,
)
from pre_experiments.camera_hidden_state_attribution.causal_analyze import (
    aggregate_scene_effects,
    freeze_causal_normalization,
    validate_frozen_causal_normalization,
    write_causal_numeric_summary,
)
from pre_experiments.common.contracts import atomic_write_json, read_git_commit
from pre_experiments.common.model_io import load_local_model, resolve_device
from pre_experiments.local_global_consistency.artifacts import (
    load_global_context,
)
from pre_experiments.local_global_consistency.split import load_split_manifest
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


ROOT = Path(__file__).resolve().parents[2]
AUTODL_TMP = Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp"))
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_ROOT", str(AUTODL_TMP / "results"))
)


def measure_scene_causal_effects(
    camera_head: torch.nn.Module,
    normalized_tokens: np.ndarray,
    device: torch.device,
    *,
    num_iterations: int = 4,
    basis_step: float = 1e-3,
    basis_batch_size: int = 2,
    basis_dimension_limit: int | None = None,
    direct_checks_per_iteration: int = 1,
    direct_relative_step: float = 1e-2,
    expected_pred_c2w_raw: np.ndarray | None = None,
    replay_tolerance: float = 5e-3,
) -> dict[str, np.ndarray]:
    """Measure one scene's standardized hidden effects on final camera outputs."""
    tokens = np.asarray(normalized_tokens, dtype=np.float32)
    if (
        tokens.ndim != 2
        or tokens.shape[0] < 2
        or not np.isfinite(tokens).all()
    ):
        raise ValueError("normalized_tokens must be finite with shape [frame, dim]")
    if num_iterations < 1:
        raise ValueError("num_iterations must be positive")
    if not np.isfinite(basis_step) or basis_step <= 0:
        raise ValueError("basis_step must be finite and positive")
    if basis_batch_size < 1:
        raise ValueError("basis_batch_size must be positive")
    if direct_checks_per_iteration < 0:
        raise ValueError("direct_checks_per_iteration must be non-negative")
    if not np.isfinite(direct_relative_step) or direct_relative_step <= 0:
        raise ValueError("direct_relative_step must be finite and positive")
    if not np.isfinite(replay_tolerance) or replay_tolerance <= 0:
        raise ValueError("replay_tolerance must be finite and positive")

    target_dim = int(camera_head.target_dim)
    measured_dimensions = (
        target_dim
        if basis_dimension_limit is None
        else int(basis_dimension_limit)
    )
    if not 1 <= measured_dimensions <= target_dim:
        raise ValueError(
            f"basis_dimension_limit must be between 1 and {target_dim}"
        )

    token_tensor = torch.from_numpy(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        baseline_poses, trace = camera_head.decode_pose_tokens(
            token_tensor,
            num_iterations=num_iterations,
            return_trace=True,
            trace_pose_tokens=True,
        )
    baseline_features = _pose_features(baseline_poses[-1])
    if expected_pred_c2w_raw is not None:
        replayed = _features_to_c2w(baseline_features)[0]
        expected = np.asarray(expected_pred_c2w_raw, dtype=np.float64)
        if (
            expected.shape != replayed.shape
            or not np.isfinite(expected).all()
            or not np.allclose(
                replayed,
                expected,
                atol=replay_tolerance,
                rtol=0,
            )
        ):
            maximum = (
                float(np.max(np.abs(replayed - expected)))
                if expected.shape == replayed.shape
                and np.isfinite(expected).all()
                else float("inf")
            )
            raise ValueError(
                f"Camera Head replay mismatch: max_abs={maximum:.6g}"
            )
    hidden = (
        torch.stack(trace["pose_branch_hidden_list"])[:, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    scales = activation_rms(hidden)

    feature_suffixes = {
        "camera_center": (3,),
        "rotation": (3, 3),
        "fov": (2,),
    }
    positive = {}
    negative = {}
    for name, suffix in feature_suffixes.items():
        baseline = baseline_features[name][0]
        shape = (num_iterations, target_dim, tokens.shape[0], *suffix)
        positive[name] = np.broadcast_to(
            baseline,
            shape,
        ).copy()
        negative[name] = np.broadcast_to(
            baseline,
            shape,
        ).copy()

    positions = [
        (iteration, basis)
        for iteration in range(num_iterations)
        for basis in range(measured_dimensions)
    ]
    for chunk in _chunks(positions, basis_batch_size):
        chunk_size = len(chunk)
        perturbation = torch.zeros(
            (num_iterations, 2 * chunk_size, target_dim),
            device=device,
            dtype=token_tensor.dtype,
        )
        for batch_index, (iteration, basis) in enumerate(chunk):
            perturbation[iteration, batch_index, basis] = basis_step
            perturbation[
                iteration, batch_index + chunk_size, basis
            ] = -basis_step
        expanded_tokens = token_tensor.expand(2 * chunk_size, -1, -1)
        with torch.no_grad():
            changed = camera_head.decode_pose_tokens(
                expanded_tokens,
                num_iterations=num_iterations,
                pose_delta_additive_perturbation=perturbation,
            )[-1]
        features = _pose_features(changed)
        for batch_index, (iteration, basis) in enumerate(chunk):
            for name in feature_suffixes:
                positive[name][iteration, basis] = features[name][batch_index]
                negative[name][iteration, basis] = features[name][
                    batch_index + chunk_size
                ]

    jacobians = central_output_jacobians(
        positive,
        negative,
        basis_step=basis_step,
    )
    weight = (
        camera_head.pose_branch.fc2.weight.detach().float().cpu().numpy()
    )
    projected = project_hidden_effects(
        jacobians,
        baseline_rotations=baseline_features["rotation"][0],
        output_weight=weight,
        activation_scales=scales,
    )
    measured_mask = np.zeros(
        (num_iterations, target_dim),
        dtype=bool,
    )
    measured_mask[:, :measured_dimensions] = True

    direct_positions = _select_direct_positions(
        projected,
        direct_checks_per_iteration,
    )
    direct = _measure_direct_positions(
        camera_head,
        token_tensor,
        device,
        num_iterations=num_iterations,
        positions=direct_positions,
        activation_scales=scales,
        projected=projected,
        relative_step=direct_relative_step,
        batch_size=basis_batch_size,
    )
    return {
        "activation_scale": scales,
        "translation_effect": projected["translation"],
        "rotation_effect_deg": projected["rotation"],
        "fov_effect": projected["fov"],
        "measured_basis_mask": measured_mask,
        **direct,
    }


def _pose_features(pose_encoding: torch.Tensor) -> dict[str, np.ndarray]:
    extrinsic, _ = pose_encoding_to_extri_intri(
        pose_encoding,
        build_intrinsics=False,
    )
    w2c_rotation = extrinsic[..., :3, :3]
    w2c_translation = extrinsic[..., :3, 3]
    c2w_rotation = w2c_rotation.transpose(-1, -2)
    camera_center = -torch.matmul(
        c2w_rotation,
        w2c_translation[..., None],
    ).squeeze(-1)
    tensors = {
        "camera_center": camera_center,
        "rotation": c2w_rotation,
        "fov": pose_encoding[..., 7:9],
    }
    return {
        name: value.detach().float().cpu().numpy()
        for name, value in tensors.items()
    }


def _features_to_c2w(features: dict[str, np.ndarray]) -> np.ndarray:
    rotation = np.asarray(features["rotation"], dtype=np.float64)
    center = np.asarray(features["camera_center"], dtype=np.float64)
    if (
        rotation.ndim != 4
        or rotation.shape[-2:] != (3, 3)
        or center.shape != rotation.shape[:2] + (3,)
    ):
        raise ValueError("invalid camera feature shapes")
    output = np.broadcast_to(
        np.eye(4, dtype=np.float64),
        rotation.shape[:2] + (4, 4),
    ).copy()
    output[..., :3, :3] = rotation
    output[..., :3, 3] = center
    return output


def _select_direct_positions(
    effects: dict[str, np.ndarray],
    per_iteration: int,
) -> list[tuple[int, int]]:
    if per_iteration == 0:
        return []
    iterations, hidden_dim = effects[GROUPS[0]].shape
    count = min(per_iteration, hidden_dim)
    group_scales = {
        group: max(
            float(np.quantile(effects[group], 0.9)),
            np.finfo(np.float64).eps,
        )
        for group in GROUPS
    }
    positions = []
    for iteration in range(iterations):
        score = sum(
            effects[group][iteration] / group_scales[group]
            for group in GROUPS
        )
        order = np.argsort(-score, kind="mergesort")
        positions.extend(
            (iteration, int(unit))
            for unit in order[:count]
        )
    return positions


def _measure_direct_positions(
    camera_head: torch.nn.Module,
    token_tensor: torch.Tensor,
    device: torch.device,
    *,
    num_iterations: int,
    positions: list[tuple[int, int]],
    activation_scales: np.ndarray,
    projected: dict[str, np.ndarray],
    relative_step: float,
    batch_size: int,
) -> dict[str, np.ndarray]:
    direct_iteration = np.asarray(
        [iteration for iteration, _ in positions],
        dtype=np.int64,
    )
    direct_unit = np.asarray(
        [unit for _, unit in positions],
        dtype=np.int64,
    )
    projected_values = {
        group: np.asarray(
            [projected[group][iteration, unit] for iteration, unit in positions],
            dtype=np.float64,
        )
        for group in GROUPS
    }
    measured_values = {
        group: np.empty(len(positions), dtype=np.float64)
        for group in GROUPS
    }
    hidden_dim = activation_scales.shape[1]
    offset = 0
    for chunk in _chunks(positions, batch_size):
        chunk_size = len(chunk)
        perturbation = torch.zeros(
            (num_iterations, 2 * chunk_size, hidden_dim),
            device=device,
            dtype=token_tensor.dtype,
        )
        for batch_index, (iteration, unit) in enumerate(chunk):
            amplitude = (
                relative_step * activation_scales[iteration, unit]
            )
            perturbation[iteration, batch_index, unit] = amplitude
            perturbation[
                iteration, batch_index + chunk_size, unit
            ] = -amplitude
        expanded_tokens = token_tensor.expand(2 * chunk_size, -1, -1)
        with torch.no_grad():
            changed = camera_head.decode_pose_tokens(
                expanded_tokens,
                num_iterations=num_iterations,
                hidden_additive_perturbation=perturbation,
            )[-1]
        features = _pose_features(changed)
        pair_effects = _direct_pair_effects(
            {
                name: values[:chunk_size]
                for name, values in features.items()
            },
            {
                name: values[chunk_size:]
                for name, values in features.items()
            },
            relative_step=relative_step,
        )
        for group in GROUPS:
            measured_values[group][offset : offset + chunk_size] = (
                pair_effects[group]
            )
        offset += chunk_size

    return {
        "direct_iteration": direct_iteration,
        "direct_unit": direct_unit,
        "direct_projected_translation": projected_values["translation"],
        "direct_measured_translation": measured_values["translation"],
        "direct_projected_rotation_deg": projected_values["rotation"],
        "direct_measured_rotation_deg": measured_values["rotation"],
        "direct_projected_fov": projected_values["fov"],
        "direct_measured_fov": measured_values["fov"],
    }


def _direct_pair_effects(
    positive: dict[str, np.ndarray],
    negative: dict[str, np.ndarray],
    *,
    relative_step: float,
) -> dict[str, np.ndarray]:
    denominator = 2.0 * relative_step
    translation = np.linalg.norm(
        positive["camera_center"] - negative["camera_center"],
        axis=-1,
    ).mean(axis=1) / denominator
    relative_rotation = np.matmul(
        np.swapaxes(positive["rotation"], -1, -2),
        negative["rotation"],
    )
    cosine = np.clip(
        (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    rotation = np.degrees(np.arccos(cosine)).mean(axis=1) / denominator
    fov = np.linalg.norm(
        positive["fov"] - negative["fov"],
        axis=-1,
    ).mean(axis=1) / denominator
    return {
        "translation": translation,
        "rotation": rotation,
        "fov": fov,
    }


def _chunks(
    values: Sequence[tuple[int, int]],
    size: int,
) -> Sequence[tuple[int, int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _select_stage_scenes(
    stage: str,
    split: dict[str, object],
    *,
    scene_limit: int,
) -> tuple[str, list[str]]:
    if stage not in {"smoke", "calibration", "holdout"}:
        raise ValueError("invalid causal preference stage")
    partition = "holdout" if stage == "holdout" else "calibration"
    key = f"{partition}_scenes"
    scenes = [str(scene) for scene in split.get(key, [])]
    if not scenes:
        raise ValueError(f"split contains no {partition} scenes")
    if stage == "smoke":
        scenes = scenes[:1]
    elif scene_limit:
        scenes = scenes[:scene_limit]
    return partition, scenes


def _measurement_config(
    args: argparse.Namespace,
    *,
    target_dim: int,
) -> dict[str, object]:
    measured_dimensions = args.basis_dimension_limit or target_dim
    if not 1 <= measured_dimensions <= target_dim:
        raise ValueError(
            f"basis_dimension_limit must be between 1 and {target_dim}"
        )
    return {
        "method": "centered_pose_delta_jacobian_projection",
        "num_iterations": int(args.num_iterations),
        "target_dim": int(target_dim),
        "measured_basis_dimensions": int(measured_dimensions),
        "basis_step": float(args.basis_step),
        "activation_scale": "per_scene_unit_rms_with_5pct_median_floor",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "calibration", "holdout"),
        required=True,
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_ROOT / "camera_hidden_causal_preference" / "results",
    )
    parser.add_argument("--frozen-normalization", type=Path)
    parser.add_argument("--run-dir-file", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--basis-step", type=float, default=1e-3)
    parser.add_argument("--basis-batch-size", type=int, default=2)
    parser.add_argument("--basis-dimension-limit", type=int, default=0)
    parser.add_argument(
        "--direct-checks-per-iteration",
        type=int,
        default=1,
    )
    parser.add_argument("--direct-relative-step", type=float, default=1e-2)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--replay-tolerance", type=float, default=5e-3)
    args = parser.parse_args(argv)
    if args.stage == "holdout" and args.frozen_normalization is None:
        parser.error("--frozen-normalization is required for holdout")
    if (
        args.num_iterations < 1
        or args.basis_step <= 0
        or args.basis_batch_size < 1
        or args.basis_dimension_limit < 0
        or args.direct_checks_per_iteration < 0
        or args.direct_relative_step <= 0
        or args.scene_limit < 0
        or args.replay_tolerance <= 0
    ):
        parser.error("numeric controls are outside their valid ranges")
    return args


def _scene_list() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "configs" / "fastvggt_scannet50.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    split = load_split_manifest(
        args.split_manifest.resolve(),
        _scene_list(),
    )
    partition, scenes = _select_stage_scenes(
        args.stage,
        split,
        scene_limit=args.scene_limit,
    )
    device = resolve_device(args.device)
    model = load_local_model(args.ckpt_dir.resolve())
    camera_head = model.camera_head.to(device).eval()
    del model

    measurement_config = _measurement_config(
        args,
        target_dim=int(camera_head.target_dim),
    )
    measured_dimensions = int(
        measurement_config["measured_basis_dimensions"]
    )
    full_partition_scenes = [
        str(scene)
        for scene in split[f"{partition}_scenes"]
    ]
    protocol_complete = (
        args.stage in {"calibration", "holdout"}
        and scenes == full_partition_scenes
        and measured_dimensions == int(camera_head.target_dim)
    )

    frozen: dict[str, object] | None = None
    if args.stage == "holdout":
        assert args.frozen_normalization is not None
        try:
            frozen_payload = json.loads(
                args.frozen_normalization.resolve().read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("invalid frozen normalization file") from error
        frozen = validate_frozen_causal_normalization(
            frozen_payload,
            split_digest=str(split["split_digest"]),
            calibration_scenes=[
                str(scene)
                for scene in split["calibration_scenes"]
            ],
            measurement_config=measurement_config,
        )

    invocation = {
        "stage": args.stage,
        "partition": partition,
        "source_run_dir": args.source_run_dir.resolve().as_posix(),
        "checkpoint_dir": args.ckpt_dir.resolve().as_posix(),
        "split_manifest": args.split_manifest.resolve().as_posix(),
        "split_digest": split["split_digest"],
        "scenes": scenes,
        "device": args.device,
        "measurement_config": measurement_config,
        "basis_batch_size": args.basis_batch_size,
        "direct_checks_per_iteration": args.direct_checks_per_iteration,
        "direct_relative_step": args.direct_relative_step,
        "replay_tolerance": args.replay_tolerance,
        "frozen_digest": (
            frozen.get("frozen_digest")
            if frozen is not None
            else None
        ),
    }
    commit = read_git_commit(ROOT)
    run_id = f"{commit[:7]}_{canonical_digest(invocation)[:12]}"
    run_dir = args.out_dir.resolve() / run_id

    scene_effects: list[dict[str, object]] = []
    for scene_index, scene in enumerate(scenes, start=1):
        scene_dir = run_dir / scene
        effect_path = scene_dir / "causal_unit_effects.npz"
        if effect_path.is_file():
            effects = load_causal_scene_effects(effect_path, scene)
        else:
            global_artifact = load_global_context(
                args.source_run_dir.resolve()
                / scene
                / "frames_500"
                / "context_diagnostics.npz"
            )
            measured = measure_scene_causal_effects(
                camera_head,
                global_artifact["normalized_camera_tokens"],
                device,
                num_iterations=args.num_iterations,
                basis_step=args.basis_step,
                basis_batch_size=args.basis_batch_size,
                basis_dimension_limit=measured_dimensions,
                direct_checks_per_iteration=(
                    args.direct_checks_per_iteration
                ),
                direct_relative_step=args.direct_relative_step,
                expected_pred_c2w_raw=global_artifact["pred_c2w_raw"],
                replay_tolerance=args.replay_tolerance,
            )
            save_causal_scene_effects(effect_path, measured)
            atomic_write_json(
                scene_dir / "complete.json",
                {
                    "run_id": run_id,
                    "scene": scene,
                    "measured_basis_dimensions": measured_dimensions,
                    "target_dim": int(camera_head.target_dim),
                    "direct_check_count": int(
                        len(measured["direct_iteration"])
                    ),
                },
            )
            effects = {"scene": scene, **measured}
        scene_effects.append(effects)
        print(
            f"[scene {scene_index}/{len(scenes)}] {scene}",
            flush=True,
        )

    analysis_complete = False
    if args.stage == "calibration" and protocol_complete:
        aggregate = aggregate_scene_effects(
            scene_effects,
            require_complete_basis=True,
        )
        frozen = freeze_causal_normalization(
            aggregate,
            split_digest=str(split["split_digest"]),
            calibration_scenes=scenes,
            measurement_config=measurement_config,
            quantile=0.9,
        )
        write_causal_numeric_summary(
            run_dir,
            scene_effects,
            output_weight=(
                camera_head.pose_branch.fc2.weight.detach()
                .float()
                .cpu()
                .numpy()
            ),
            partition=partition,
            frozen=frozen,
        )
        analysis_complete = True
    elif args.stage == "holdout" and measured_dimensions == int(
        camera_head.target_dim
    ):
        assert frozen is not None
        write_causal_numeric_summary(
            run_dir,
            scene_effects,
            output_weight=(
                camera_head.pose_branch.fc2.weight.detach()
                .float()
                .cpu()
                .numpy()
            ),
            partition=partition,
            frozen=frozen,
        )
        analysis_complete = protocol_complete

    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "study_name": "camera_hidden_causal_preference",
            "run_id": run_id,
            "git_commit": commit,
            "partition": partition,
            "split_digest": split["split_digest"],
            "protocol_complete": protocol_complete,
            "invocation": invocation,
            "metric_policy": (
                "prediction-only causal atlas; no GT metrics are computed"
            ),
        },
    )
    atomic_write_json(
        run_dir / "complete.json",
        {
            "run_id": run_id,
            "partition": partition,
            "scenes": scenes,
            "scene_count": len(scenes),
            "protocol_complete": protocol_complete,
            "analysis_complete": analysis_complete,
            "frozen_digest": (
                frozen.get("frozen_digest")
                if frozen is not None and analysis_complete
                else None
            ),
        },
    )
    if args.run_dir_file is not None:
        destination = args.run_dir_file.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{run_dir}\n", encoding="utf-8")
    print(f"[done] run={run_dir}")


if __name__ == "__main__":
    main()
