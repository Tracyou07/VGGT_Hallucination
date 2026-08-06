"""Replace frozen long-context Camera Head hidden units with short-context values."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.analyze import (
    intervention_metrics,
)
from pre_experiments.camera_hidden_state_attribution.replacement import (
    assemble_short_hidden,
    freeze_replacement_manifest,
    replacement_mask,
)
from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from pre_experiments.camera_hidden_state_attribution.replacement_analyze import (
    select_calibration_alpha,
    write_replacement_numeric_summary,
)
from pre_experiments.camera_hidden_state_attribution.replacement_artifacts import (
    load_replacement_scene,
    save_replacement_scene,
)
from pre_experiments.camera_hidden_state_attribution.run_study import (
    _validate_local_run,
    _verify_replay,
    replay_tokens,
)
from pre_experiments.common.contracts import atomic_write_json, read_git_commit
from pre_experiments.common.model_io import load_local_model, resolve_device
from pre_experiments.common.pose_metrics import align_pose_sequence
from pre_experiments.local_global_consistency.artifacts import (
    load_global_context,
    load_window_diagnostics,
)
from pre_experiments.local_global_consistency.split import load_split_manifest


ROOT = Path(__file__).resolve().parents[2]
AUTODL_TMP = Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp"))
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_ROOT", str(AUTODL_TMP / "results"))
)
DEFAULT_ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


def _aligned_metrics(
    pred_c2w_raw: np.ndarray,
    gt_c2w_raw: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    alignment = align_pose_sequence(
        np.linalg.inv(np.asarray(pred_c2w_raw, dtype=np.float64)),
        np.asarray(gt_c2w_raw, dtype=np.float64),
    )
    translation = np.asarray(
        alignment["translation_error_aligned"],
        dtype=np.float64,
    )
    rotation = np.asarray(
        alignment["rotation_error_deg_aligned"],
        dtype=np.float64,
    )
    metrics = {
        "aligned_translation_error_mean": float(translation.mean()),
        "aligned_translation_error_median": float(np.median(translation)),
        "aligned_translation_ate_rmse": float(
            np.sqrt(np.mean(np.square(translation)))
        ),
        "aligned_rotation_error_deg_mean": float(rotation.mean()),
        "aligned_rotation_error_deg_median": float(np.median(rotation)),
    }
    return metrics, translation, rotation


def run_scene_replacement(
    camera_head: torch.nn.Module,
    global_artifact: Mapping[str, np.ndarray],
    local_records: Sequence[Mapping[str, object]],
    frozen: Mapping[str, object],
    device: torch.device,
    *,
    scene: str,
    alphas: Sequence[float] = (1.0,),
    control_names: Sequence[str] | None = None,
    replay_tolerance: float = 5e-3,
) -> dict[str, object]:
    """Run baseline, selected replacement, and frozen control replacements."""
    global_ids = np.asarray(global_artifact["frame_ids"], dtype=np.int64)
    global_tokens = np.asarray(
        global_artifact["normalized_camera_tokens"],
        dtype=np.float32,
    )
    global_gt = np.asarray(global_artifact["gt_c2w_raw"], dtype=np.float64)
    if (
        global_ids.ndim != 1
        or global_tokens.ndim != 2
        or len(global_tokens) != len(global_ids)
        or global_gt.shape != (len(global_ids), 4, 4)
    ):
        raise ValueError("global replacement artifact has invalid shapes")

    baseline = replay_tokens(camera_head, global_tokens, device)
    _verify_replay(
        baseline["pred_c2w_raw"],
        np.asarray(global_artifact["pred_c2w_raw"]),
        scene=f"{scene}/global",
        tolerance=replay_tolerance,
    )
    global_index = {
        int(frame_id): index for index, frame_id in enumerate(global_ids)
    }
    short_windows = []
    for record in sorted(
        local_records,
        key=lambda item: int(item["window_index"]),
    ):
        artifact = record.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("local replacement record lacks an artifact")
        local_ids = np.asarray(artifact["frame_ids"], dtype=np.int64)
        try:
            indices = np.asarray(
                [global_index[int(frame_id)] for frame_id in local_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "local replacement frame is absent from global context"
            ) from error
        local_gt = np.asarray(artifact["gt_c2w_raw"], dtype=np.float64)
        if (
            local_gt.shape != (len(local_ids), 4, 4)
            or not np.allclose(local_gt, global_gt[indices], atol=1e-10, rtol=0)
        ):
            raise ValueError("local raw GT does not match global raw GT")
        local_replay = replay_tokens(
            camera_head,
            np.asarray(artifact["normalized_camera_tokens"], dtype=np.float32),
            device,
        )
        _verify_replay(
            local_replay["pred_c2w_raw"],
            np.asarray(artifact["pred_c2w_raw"]),
            scene=(
                f"{scene}/window_{int(record['window_index']):03d}"
            ),
            tolerance=replay_tolerance,
        )
        short_windows.append(
            {
                "window_index": int(record["window_index"]),
                "frame_ids": local_ids,
                "hidden": local_replay["hidden"],
            }
        )
    assembled = assemble_short_hidden(global_ids, short_windows)
    short_hidden = assembled["hidden"]
    iterations, _, hidden_dim = baseline["hidden"].shape
    if short_hidden.shape != baseline["hidden"].shape:
        raise ValueError("assembled short hidden does not match long hidden")

    alpha_values = tuple(float(alpha) for alpha in alphas)
    if (
        not alpha_values
        or any(
            not np.isfinite(alpha) or not 0.0 < alpha <= 1.0
            for alpha in alpha_values
        )
        or tuple(sorted(set(alpha_values))) != alpha_values
    ):
        raise ValueError("alphas must be unique, increasing, and in (0, 1]")

    control_sets = frozen.get("control_sets")
    if not isinstance(control_sets, Sequence):
        raise ValueError("frozen replacement controls are missing")
    if not control_sets or any(
        not isinstance(control, Mapping) for control in control_sets
    ):
        raise ValueError("invalid frozen replacement control")
    available_control_names = tuple(
        str(control.get("name", "")) for control in control_sets
    )
    if (
        any(not name for name in available_control_names)
        or len(set(available_control_names)) != len(available_control_names)
    ):
        raise ValueError("invalid frozen replacement control")
    requested_control_names = (
        (available_control_names[0],)
        if control_names is None
        else tuple(str(name) for name in control_names)
    )
    if (
        not requested_control_names
        or len(set(requested_control_names)) != len(requested_control_names)
        or any(
            name not in available_control_names
            for name in requested_control_names
        )
    ):
        raise ValueError("requested replacement controls are invalid")
    conditions = [("baseline", "baseline", 0.0, None)]
    for alpha in alpha_values:
        label = _alpha_label(alpha)
        conditions.append(
            (f"selected_a{label}", "selected", alpha, "selected")
        )
        conditions.extend(
            (
                f"{control_name}_a{label}",
                "control",
                alpha,
                control_name,
            )
            for control_name in requested_control_names
        )
    condition_names = [condition[0] for condition in conditions]

    replays = [baseline]
    masks = [np.zeros((iterations, hidden_dim), dtype=bool)]
    for _, _, alpha, set_name in conditions[1:]:
        assert set_name is not None
        mask = replacement_mask(
            frozen,
            set_name,
            iterations=iterations,
            hidden_dim=hidden_dim,
        )
        masks.append(mask)
        replays.append(
            replay_tokens(
                camera_head,
                global_tokens,
                device,
                hidden_replacement_values=short_hidden,
                hidden_replacement_mask=mask,
                hidden_replacement_alpha=alpha,
                trace_hidden=False,
            )
        )

    baseline_metrics, _, _ = _aligned_metrics(
        baseline["pred_c2w_raw"],
        global_gt,
    )
    rows = []
    translation_errors = []
    rotation_errors = []
    for condition_spec, replay, mask in zip(conditions, replays, masks):
        condition, family, alpha, _ = condition_spec
        metrics, translation, rotation = _aligned_metrics(
            replay["pred_c2w_raw"],
            global_gt,
        )
        if condition == "baseline":
            changes = {
                "camera_center_displacement_mean": 0.0,
                "rotation_change_deg_mean": 0.0,
                "fov_change_mean": 0.0,
            }
        else:
            changes = intervention_metrics(
                baseline["pred_c2w_raw"],
                replay["pred_c2w_raw"],
                baseline["pose_enc"],
                replay["pose_enc"],
            )
        rows.append(
            {
                "scene": scene,
                "condition": condition,
                "condition_family": family,
                "alpha": alpha,
                "replacement_count": int(mask.sum()),
                **metrics,
                "aligned_translation_error_delta": (
                    metrics["aligned_translation_error_mean"]
                    - baseline_metrics["aligned_translation_error_mean"]
                ),
                "aligned_rotation_error_deg_delta": (
                    metrics["aligned_rotation_error_deg_mean"]
                    - baseline_metrics["aligned_rotation_error_deg_mean"]
                ),
                **changes,
            }
        )
        translation_errors.append(translation)
        rotation_errors.append(rotation)

    return {
        "scene": scene,
        "condition_names": np.asarray(condition_names),
        "condition_family": np.asarray(
            [condition[1] for condition in conditions]
        ),
        "condition_alpha": np.asarray(
            [condition[2] for condition in conditions],
            dtype=np.float64,
        ),
        "replacement_count": np.asarray(
            [int(mask.sum()) for mask in masks],
            dtype=np.int64,
        ),
        "frame_ids": global_ids.copy(),
        "selected_window_index": assembled["selected_window_index"],
        "selected_boundary_distance": assembled[
            "selected_boundary_distance"
        ],
        "local_observation_count": assembled["observation_count"],
        "pred_c2w_raw": np.stack(
            [replay["pred_c2w_raw"] for replay in replays]
        ),
        "pose_enc": np.stack([replay["pose_enc"] for replay in replays]),
        "translation_error_aligned": np.stack(translation_errors),
        "rotation_error_deg_aligned": np.stack(rotation_errors),
        "rows": rows,
    }


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.8g}".replace(".", "p")


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read calibration table: {path}") from error
    if not rows:
        raise ValueError(f"calibration table is empty: {path}")
    return rows


def _validate_calibration_numeric_run(
    directory: Path,
    *,
    study_name: str,
    split_digest: str,
) -> dict[str, object]:
    metadata = _json_object(directory / "run_metadata.json")
    complete = _json_object(directory / "complete.json")
    if (
        metadata.get("run_id") != directory.name
        or metadata.get("study_name") != study_name
        or metadata.get("partition") != "calibration"
        or metadata.get("split_digest") != split_digest
        or metadata.get("protocol_complete") is not True
        or complete.get("run_id") != directory.name
        or complete.get("partition") != "calibration"
        or complete.get("protocol_complete") is not True
        or complete.get("analysis_complete") is not True
    ):
        raise ValueError(
            f"calibration numeric provenance mismatch: {directory}"
        )
    return metadata


def _freeze_from_calibration_dirs(
    attribution_dir: Path,
    causal_dir: Path,
    *,
    split_digest: str,
    calibration_scenes: Sequence[str],
    source_top_k: int,
    control_repeats: int,
    seed: int,
) -> dict[str, object]:
    attribution_dir = attribution_dir.resolve()
    causal_dir = causal_dir.resolve()
    attribution_metadata = _validate_calibration_numeric_run(
        attribution_dir,
        study_name="camera_hidden_state_attribution",
        split_digest=split_digest,
    )
    causal_metadata = _validate_calibration_numeric_run(
        causal_dir,
        study_name="camera_hidden_causal_preference",
        split_digest=split_digest,
    )
    frozen = freeze_replacement_manifest(
        _csv_rows(attribution_dir / "per_unit.csv"),
        _csv_rows(causal_dir / "per_position.csv"),
        split_digest=split_digest,
        calibration_scenes=calibration_scenes,
        source_top_k=source_top_k,
        control_repeats=control_repeats,
        seed=seed,
    )
    frozen.pop("frozen_digest")
    frozen["source_runs"] = {
        "attribution": str(attribution_metadata["run_id"]),
        "causal": str(causal_metadata["run_id"]),
    }
    frozen["source_git_commits"] = {
        "attribution": attribution_metadata.get("git_commit"),
        "causal": causal_metadata.get("git_commit"),
    }
    frozen["frozen_digest"] = canonical_digest(frozen)
    return frozen


def _validate_frozen_replacement(
    frozen: Mapping[str, object],
    *,
    split_digest: str,
    calibration_scenes: Sequence[str],
    require_selected_alpha: bool = False,
) -> dict[str, object]:
    try:
        value = json.loads(json.dumps(frozen))
    except (TypeError, ValueError) as error:
        raise ValueError("frozen replacement provenance mismatch") from error
    if not isinstance(value, dict):
        raise ValueError("frozen replacement provenance mismatch")
    digest = value.pop("frozen_digest", None)
    selected = value.get("selected")
    controls = value.get("control_sets")
    if (
        not isinstance(digest, str)
        or digest != canonical_digest(value)
        or value.get("schema_version") != 1
        or value.get("method") != "short_to_long_pose_hidden_replacement"
        or value.get("group") != "translation"
        or value.get("split_digest") != split_digest
        or value.get("calibration_scenes")
        != [str(scene) for scene in calibration_scenes]
        or not isinstance(selected, list)
        or value.get("selected_count") != len(selected)
        or not isinstance(controls, list)
        or value.get("control_repeats") != len(controls)
    ):
        raise ValueError("frozen replacement provenance mismatch")
    if require_selected_alpha:
        try:
            selected_alpha = float(value["selected_alpha"])
            alpha_grid = tuple(float(alpha) for alpha in value["alpha_grid"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "frozen replacement alpha provenance mismatch"
            ) from error
        if (
            not np.isfinite(selected_alpha)
            or not 0.0 < selected_alpha <= 1.0
            or selected_alpha not in alpha_grid
            or tuple(sorted(set(alpha_grid))) != alpha_grid
            or not value.get("alpha_selection_metric")
        ):
            raise ValueError("frozen replacement alpha provenance mismatch")
    value["frozen_digest"] = digest
    return value


def _finalize_alpha_selection(
    frozen: Mapping[str, object],
    selection: Mapping[str, object],
    *,
    alpha_grid: Sequence[float],
) -> dict[str, object]:
    """Attach calibration-only alpha selection and refresh provenance."""
    value = json.loads(json.dumps(frozen))
    value.pop("frozen_digest", None)
    value["alpha_grid"] = [float(alpha) for alpha in alpha_grid]
    value.update(selection)
    value["frozen_digest"] = canonical_digest(value)
    return value


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--alphas must be a comma-separated numeric grid"
        ) from error
    if (
        not alphas
        or any(
            not np.isfinite(alpha) or not 0.0 < alpha <= 1.0
            for alpha in alphas
        )
        or tuple(sorted(set(alphas))) != alphas
    ):
        raise argparse.ArgumentTypeError(
            "--alphas must be unique, increasing, and in (0, 1]"
        )
    return alphas


def _control_names_for_stage(
    frozen: Mapping[str, object],
    stage: str,
) -> tuple[str, ...]:
    """Use one control for selection and every frozen control for holdout."""
    if stage not in {"smoke", "calibration", "holdout"}:
        raise ValueError(f"unsupported replacement stage: {stage}")
    control_sets = frozen.get("control_sets")
    if not isinstance(control_sets, Sequence) or not control_sets:
        raise ValueError("frozen replacement controls are missing")
    names = tuple(
        str(control.get("name", ""))
        for control in control_sets
        if isinstance(control, Mapping)
    )
    if (
        len(names) != len(control_sets)
        or any(not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("invalid frozen replacement controls")
    return names if stage == "holdout" else names[:1]


def _scene_list() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "configs" / "fastvggt_scannet50.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _local_records(local_run: Path, scene: str) -> list[dict[str, object]]:
    paths = sorted(
        (local_run / scene).glob("window_*/window_diagnostics.npz")
    )
    if not paths:
        raise FileNotFoundError(f"no local windows found for {scene}")
    records = []
    for path in paths:
        try:
            window_index = int(path.parent.name.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid window directory: {path.parent}") from error
        records.append(
            {
                "window_index": window_index,
                "artifact": load_window_diagnostics(path),
            }
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "calibration", "holdout"),
        required=True,
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--attribution-calibration-dir", type=Path)
    parser.add_argument("--causal-calibration-dir", type=Path)
    parser.add_argument("--frozen-replacement", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_ROOT / "camera_hidden_replacement" / "results",
    )
    parser.add_argument("--run-dir-file", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--source-top-k", type=int, default=64)
    parser.add_argument("--control-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--replay-tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--alphas",
        type=_parse_alphas,
        default=DEFAULT_ALPHAS,
        help="comma-separated calibration interpolation grid",
    )
    args = parser.parse_args(argv)
    if args.stage == "holdout":
        if args.frozen_replacement is None:
            parser.error("--frozen-replacement is required for holdout")
    elif (
        args.attribution_calibration_dir is None
        or args.causal_calibration_dir is None
    ):
        parser.error(
            "calibration numeric directories are required for smoke/calibration"
        )
    if (
        args.source_top_k < 1
        or args.control_repeats < 1
        or args.scene_limit < 0
        or args.replay_tolerance <= 0
    ):
        parser.error("numeric controls are outside their valid ranges")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    split = load_split_manifest(
        args.split_manifest.resolve(),
        _scene_list(),
    )
    partition = (
        "holdout" if args.stage == "holdout" else "calibration"
    )
    all_scenes = [
        str(scene) for scene in split[f"{partition}_scenes"]
    ]
    scenes = all_scenes[:1] if args.stage == "smoke" else all_scenes
    if args.stage != "smoke" and args.scene_limit:
        scenes = scenes[: args.scene_limit]
    _validate_local_run(
        args.local_run_dir.resolve(),
        partition=partition,
        split_digest=str(split["split_digest"]),
    )

    if args.stage == "holdout":
        assert args.frozen_replacement is not None
        frozen = _validate_frozen_replacement(
            _json_object(args.frozen_replacement.resolve()),
            split_digest=str(split["split_digest"]),
            calibration_scenes=[
                str(scene) for scene in split["calibration_scenes"]
            ],
            require_selected_alpha=True,
        )
        run_alphas = (float(frozen["selected_alpha"]),)
    else:
        assert args.attribution_calibration_dir is not None
        assert args.causal_calibration_dir is not None
        frozen = _freeze_from_calibration_dirs(
            args.attribution_calibration_dir,
            args.causal_calibration_dir,
            split_digest=str(split["split_digest"]),
            calibration_scenes=[
                str(scene) for scene in split["calibration_scenes"]
            ],
            source_top_k=args.source_top_k,
            control_repeats=args.control_repeats,
            seed=args.seed,
        )
        run_alphas = args.alphas

    control_names = _control_names_for_stage(frozen, args.stage)
    invocation = {
        "stage": args.stage,
        "partition": partition,
        "source_run_dir": args.source_run_dir.resolve().as_posix(),
        "local_run_dir": args.local_run_dir.resolve().as_posix(),
        "checkpoint_dir": args.ckpt_dir.resolve().as_posix(),
        "split_manifest": args.split_manifest.resolve().as_posix(),
        "split_digest": split["split_digest"],
        "scenes": scenes,
        "device": args.device,
        "replay_tolerance": args.replay_tolerance,
        "unit_frozen_digest": frozen["frozen_digest"],
        "alphas": run_alphas,
        "control_names": control_names,
    }
    commit = read_git_commit(ROOT)
    run_id = f"{commit[:7]}_{canonical_digest(invocation)[:12]}"
    run_dir = args.out_dir.resolve() / run_id
    device = resolve_device(args.device)
    model = load_local_model(args.ckpt_dir.resolve())
    camera_head = model.camera_head.to(device).eval()
    del model

    results = []
    for scene_index, scene in enumerate(scenes, start=1):
        scene_dir = run_dir / scene
        artifact_path = scene_dir / "replacement_diagnostics.npz"
        if artifact_path.is_file():
            result = load_replacement_scene(artifact_path, scene)
        else:
            global_artifact = load_global_context(
                args.source_run_dir.resolve()
                / scene
                / "frames_500"
                / "context_diagnostics.npz"
            )
            result = run_scene_replacement(
                camera_head,
                global_artifact,
                _local_records(args.local_run_dir.resolve(), scene),
                frozen,
                device,
                scene=scene,
                alphas=run_alphas,
                control_names=control_names,
                replay_tolerance=args.replay_tolerance,
            )
            save_replacement_scene(artifact_path, result)
            atomic_write_json(
                scene_dir / "complete.json",
                {
                    "run_id": run_id,
                    "scene": scene,
                    "condition_names": result["condition_names"].tolist(),
                    "selected_count": frozen["selected_count"],
                    "unit_frozen_digest": invocation[
                        "unit_frozen_digest"
                    ],
                },
            )
        results.append(result)
        print(f"[scene {scene_index}/{len(scenes)}] {scene}", flush=True)

    summary = write_replacement_numeric_summary(
        run_dir,
        results,
        frozen,
        partition=partition,
    )
    if args.stage != "holdout":
        frozen = _finalize_alpha_selection(
            frozen,
            select_calibration_alpha(summary),
            alpha_grid=run_alphas,
        )
        summary = write_replacement_numeric_summary(
            run_dir,
            results,
            frozen,
            partition=partition,
        )
    protocol_complete = (
        args.stage in {"calibration", "holdout"}
        and scenes == all_scenes
    )
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "study_name": "camera_hidden_replacement",
            "run_id": run_id,
            "git_commit": commit,
            "partition": partition,
            "split_digest": split["split_digest"],
            "protocol_complete": protocol_complete,
            "invocation": invocation,
            "selection_policy": (
                "calibration-frozen attribution/causal top-k intersection"
            ),
            "replacement_policy": (
                "long Camera Head pose hidden interpolated toward the "
                "most-interior matched short-window hidden"
            ),
            "selected_alpha": frozen["selected_alpha"],
            "metric_policy": (
                "each prediction aligned independently; GT remains raw"
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
            "analysis_complete": protocol_complete,
            "frozen_digest": frozen["frozen_digest"],
        },
    )
    if args.run_dir_file is not None:
        destination = args.run_dir_file.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{run_dir}\n", encoding="utf-8")
    print(f"[done] run={run_dir}")


if __name__ == "__main__":
    main()
