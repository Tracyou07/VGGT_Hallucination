"""Fit and evaluate prediction-only scene-adaptive hidden interpolation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import json
import os
from pathlib import Path

from pre_experiments.camera_hidden_state_attribution.adaptive_alpha import (
    build_oracle_labels,
    build_scene_features,
    evaluate_leave_one_out,
    fit_frozen_selector,
    load_frozen_selector,
    predict_alpha,
)
from pre_experiments.camera_hidden_state_attribution.adaptive_replacement import (
    compare_adaptive_to_fixed,
    summarize_adaptive_rows,
)
from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from pre_experiments.camera_hidden_state_attribution.replacement_analyze import (
    build_replacement_rows,
)
from pre_experiments.camera_hidden_state_attribution.replacement_artifacts import (
    load_replacement_scene,
    save_replacement_scene,
)
from pre_experiments.camera_hidden_state_attribution.run_replacement import (
    _control_names_for_stage,
    _json_object,
    _local_records,
    _scene_list,
    _validate_frozen_replacement,
    run_scene_replacement,
)
from pre_experiments.camera_hidden_state_attribution.run_study import (
    _validate_local_run,
)
from pre_experiments.common.contracts import atomic_write_json, read_git_commit
from pre_experiments.common.model_io import load_local_model, resolve_device
from pre_experiments.local_global_consistency.artifacts import (
    load_global_context,
)
from pre_experiments.local_global_consistency.split import load_split_manifest


ROOT = Path(__file__).resolve().parents[2]
AUTODL_TMP = Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp"))
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_ROOT", str(AUTODL_TMP / "results"))
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read CSV artifact: {path}") from error
    if not rows:
        raise ValueError(f"CSV artifact is empty: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_calibration_bundle(
    score_rows: Sequence[Mapping[str, object]],
    replacement_rows: Sequence[Mapping[str, object]],
    *,
    expected_scenes: Sequence[str],
    split_digest: str,
    score_run_id: str,
    replacement_run_id: str,
) -> dict[str, object]:
    """Fit the final selector and a leakage-free calibration report."""
    expected = [str(scene) for scene in expected_scenes]
    features = build_scene_features(score_rows)
    if [row["scene"] for row in features] != sorted(expected):
        raise ValueError("calibration prediction score scenes mismatch")
    labels = build_oracle_labels(replacement_rows)
    if sorted(labels) != sorted(expected):
        raise ValueError("calibration replacement scenes mismatch")
    report = evaluate_leave_one_out(
        features,
        labels,
        replacement_rows,
        split_digest=split_digest,
        score_run_id=score_run_id,
        replacement_run_id=replacement_run_id,
    )
    frozen = fit_frozen_selector(
        features,
        labels,
        split_digest=split_digest,
        score_run_id=score_run_id,
        replacement_run_id=replacement_run_id,
    )
    return {
        "features": features,
        "labels": labels,
        "report": report,
        "frozen": frozen,
    }


def assign_holdout_alphas(
    frozen: Mapping[str, object],
    scene_features: Sequence[Mapping[str, object]],
    *,
    expected_scenes: Sequence[str],
) -> dict[str, float]:
    """Predict exactly one alpha for every declared holdout scene."""
    by_scene = {
        str(row.get("scene", "")): row for row in scene_features
    }
    expected = sorted(str(scene) for scene in expected_scenes)
    if (
        len(by_scene) != len(scene_features)
        or sorted(by_scene) != expected
        or set(frozen["calibration_scenes"]).intersection(expected)
    ):
        raise ValueError("holdout feature scenes mismatch")
    return {
        scene: predict_alpha(frozen, by_scene[scene])
        for scene in expected
    }


def _validated_run_metadata(
    directory: Path,
    *,
    partition: str,
    split_digest: str,
) -> dict[str, object]:
    metadata = _json_object(directory / "run_metadata.json")
    if (
        metadata.get("run_id") != directory.name
        or metadata.get("partition") != partition
        or metadata.get("split_digest") != split_digest
        or metadata.get("protocol_complete") is not True
    ):
        raise ValueError(f"source run provenance mismatch: {directory}")
    return metadata


def _write_holdout_summary(
    run_dir: Path,
    results: Sequence[Mapping[str, object]],
    *,
    selector: Mapping[str, object],
    frozen_replacement: Mapping[str, object],
    fixed_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    scene_rows = []
    frame_rows = []
    for result in results:
        result_scene_rows, result_frame_rows = build_replacement_rows(result)
        scene_rows.extend(result_scene_rows)
        frame_rows.extend(result_frame_rows)
    _write_csv(run_dir / "per_scene.csv", scene_rows)
    _write_csv(run_dir / "per_frame.csv", frame_rows)
    summary = {
        "partition": "holdout",
        "selector_digest": selector["selector_digest"],
        "frozen_replacement_digest": frozen_replacement["frozen_digest"],
        "selected_count": frozen_replacement["selected_count"],
        "configured_control_repeats": frozen_replacement["control_repeats"],
        **summarize_adaptive_rows(scene_rows),
        "fixed_alpha_comparison": compare_adaptive_to_fixed(
            scene_rows,
            fixed_rows,
        ),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "frozen_selector.json", dict(selector))
    atomic_write_json(
        run_dir / "frozen_replacement.json",
        dict(frozen_replacement),
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("calibration", "holdout"),
        required=True,
    )
    parser.add_argument("--score-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--replacement-calibration-dir", type=Path)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--frozen-replacement", type=Path)
    parser.add_argument("--fixed-holdout-dir", type=Path)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--local-run-dir", type=Path)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=AUTODL_TMP / "ckpt" / "VGGT-1B",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_ROOT / "camera_hidden_adaptive_alpha" / "results",
    )
    parser.add_argument("--run-dir-file", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--replay-tolerance", type=float, default=5e-3)
    args = parser.parse_args(argv)
    if args.stage == "calibration":
        if args.replacement_calibration_dir is None:
            parser.error(
                "--replacement-calibration-dir is required for calibration"
            )
    elif any(
        value is None
        for value in (
            args.selector,
            args.frozen_replacement,
            args.fixed_holdout_dir,
            args.source_run_dir,
            args.local_run_dir,
        )
    ):
        parser.error(
            "holdout requires --selector, --frozen-replacement, "
            "--fixed-holdout-dir, --source-run-dir, and --local-run-dir"
        )
    if args.replay_tolerance <= 0:
        parser.error("--replay-tolerance must be positive")
    return args


def _calibration_main(
    args: argparse.Namespace,
    split: Mapping[str, object],
    commit: str,
) -> Path:
    score_dir = args.score_run_dir.resolve()
    replacement_dir = args.replacement_calibration_dir.resolve()
    score_metadata = _validated_run_metadata(
        score_dir,
        partition="calibration",
        split_digest=str(split["split_digest"]),
    )
    replacement_metadata = _validated_run_metadata(
        replacement_dir,
        partition="calibration",
        split_digest=str(split["split_digest"]),
    )
    bundle = build_calibration_bundle(
        _read_csv(score_dir / "calibration_prediction_scores_per_frame.csv"),
        _read_csv(replacement_dir / "per_scene.csv"),
        expected_scenes=split["calibration_scenes"],
        split_digest=str(split["split_digest"]),
        score_run_id=str(score_metadata["run_id"]),
        replacement_run_id=str(replacement_metadata["run_id"]),
    )
    invocation = {
        "stage": "calibration",
        "score_run_dir": score_dir.as_posix(),
        "replacement_calibration_dir": replacement_dir.as_posix(),
        "split_digest": split["split_digest"],
        "selector_digest": bundle["frozen"]["selector_digest"],
    }
    run_id = f"{commit[:7]}_{canonical_digest(invocation)[:12]}"
    run_dir = args.out_dir.resolve() / run_id
    _write_csv(run_dir / "scene_features.csv", bundle["features"])
    _write_csv(run_dir / "loocv_per_scene.csv", bundle["report"]["rows"])
    _write_csv(
        run_dir / "oracle_alpha_per_scene.csv",
        [
            {"scene": scene, "oracle_alpha": alpha}
            for scene, alpha in sorted(bundle["labels"].items())
        ],
    )
    atomic_write_json(run_dir / "summary.json", bundle["report"])
    atomic_write_json(run_dir / "frozen_selector.json", bundle["frozen"])
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "study_name": "camera_hidden_adaptive_alpha",
            "run_id": run_id,
            "git_commit": commit,
            "partition": "calibration",
            "split_digest": split["split_digest"],
            "protocol_complete": True,
            "invocation": invocation,
        },
    )
    atomic_write_json(
        run_dir / "complete.json",
        {
            "run_id": run_id,
            "partition": "calibration",
            "scene_count": len(split["calibration_scenes"]),
            "protocol_complete": True,
            "analysis_complete": True,
            "selector_digest": bundle["frozen"]["selector_digest"],
        },
    )
    return run_dir


def _holdout_main(
    args: argparse.Namespace,
    split: Mapping[str, object],
    commit: str,
) -> Path:
    score_dir = args.score_run_dir.resolve()
    score_metadata = _validated_run_metadata(
        score_dir,
        partition="holdout",
        split_digest=str(split["split_digest"]),
    )
    selector = load_frozen_selector(
        args.selector.resolve(),
        expected_split_digest=str(split["split_digest"]),
    )
    features = build_scene_features(
        _read_csv(score_dir / "holdout_prediction_scores_per_frame.csv")
    )
    assignments = assign_holdout_alphas(
        selector,
        features,
        expected_scenes=split["holdout_scenes"],
    )
    frozen_replacement = _validate_frozen_replacement(
        _json_object(args.frozen_replacement.resolve()),
        split_digest=str(split["split_digest"]),
        calibration_scenes=split["calibration_scenes"],
        require_selected_alpha=True,
    )
    fixed_holdout_dir = args.fixed_holdout_dir.resolve()
    fixed_metadata = _validated_run_metadata(
        fixed_holdout_dir,
        partition="holdout",
        split_digest=str(split["split_digest"]),
    )
    if (
        fixed_metadata.get("study_name") != "camera_hidden_replacement"
        or float(fixed_metadata.get("selected_alpha", -1)) != 0.02
    ):
        raise ValueError("fixed holdout must be the frozen alpha=0.02 run")
    fixed_rows = _read_csv(fixed_holdout_dir / "per_scene.csv")
    control_names = _control_names_for_stage(
        frozen_replacement,
        "holdout",
    )
    _validate_local_run(
        args.local_run_dir.resolve(),
        partition="holdout",
        split_digest=str(split["split_digest"]),
    )
    invocation = {
        "stage": "holdout",
        "score_run_id": score_metadata["run_id"],
        "source_run_dir": args.source_run_dir.resolve().as_posix(),
        "local_run_dir": args.local_run_dir.resolve().as_posix(),
        "checkpoint_dir": args.ckpt_dir.resolve().as_posix(),
        "split_digest": split["split_digest"],
        "selector_digest": selector["selector_digest"],
        "frozen_replacement_digest": frozen_replacement["frozen_digest"],
        "fixed_holdout_run_id": fixed_metadata["run_id"],
        "scene_alphas": assignments,
        "control_names": control_names,
        "replay_tolerance": args.replay_tolerance,
        "device": args.device,
    }
    run_id = f"{commit[:7]}_{canonical_digest(invocation)[:12]}"
    run_dir = args.out_dir.resolve() / run_id
    device = resolve_device(args.device)
    model = load_local_model(args.ckpt_dir.resolve())
    camera_head = model.camera_head.to(device).eval()
    del model

    results = []
    scenes = [str(scene) for scene in split["holdout_scenes"]]
    for scene_index, scene in enumerate(scenes, start=1):
        scene_dir = run_dir / scene
        artifact_path = scene_dir / "adaptive_replacement_diagnostics.npz"
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
                frozen_replacement,
                device,
                scene=scene,
                alphas=(assignments[scene],),
                control_names=control_names,
                replay_tolerance=args.replay_tolerance,
            )
            save_replacement_scene(artifact_path, result)
            atomic_write_json(
                scene_dir / "complete.json",
                {
                    "run_id": run_id,
                    "scene": scene,
                    "selected_alpha": assignments[scene],
                    "condition_names": result["condition_names"].tolist(),
                    "selector_digest": selector["selector_digest"],
                },
            )
        results.append(result)
        print(f"[scene {scene_index}/{len(scenes)}] {scene}", flush=True)

    _write_holdout_summary(
        run_dir,
        results,
        selector=selector,
        frozen_replacement=frozen_replacement,
        fixed_rows=fixed_rows,
    )
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "study_name": "camera_hidden_adaptive_alpha",
            "run_id": run_id,
            "git_commit": commit,
            "partition": "holdout",
            "split_digest": split["split_digest"],
            "protocol_complete": True,
            "invocation": invocation,
            "feature_policy": "prediction-only scene medians",
            "metric_policy": (
                "each prediction aligned independently; GT remains raw"
            ),
        },
    )
    atomic_write_json(
        run_dir / "complete.json",
        {
            "run_id": run_id,
            "partition": "holdout",
            "scenes": scenes,
            "scene_count": len(scenes),
            "protocol_complete": True,
            "analysis_complete": True,
            "selector_digest": selector["selector_digest"],
            "frozen_replacement_digest": frozen_replacement["frozen_digest"],
        },
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    split = load_split_manifest(
        args.split_manifest.resolve(),
        _scene_list(),
    )
    commit = read_git_commit(ROOT)
    if args.stage == "calibration":
        run_dir = _calibration_main(args, split, commit)
    else:
        run_dir = _holdout_main(args, split, commit)
    if args.run_dir_file is not None:
        destination = args.run_dir_file.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{run_dir}\n", encoding="utf-8")
    print(f"[done] run={run_dir}")


if __name__ == "__main__":
    main()
