"""Analyze formal ScanNet-50 calibration or holdout local-window runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pre_experiments.common.contracts import atomic_write_json
from pre_experiments.local_global_consistency.artifacts import (
    load_global_context,
    load_window_diagnostics,
)
from pre_experiments.local_global_consistency.metrics import (
    apply_reliability,
    build_scene_rows,
    summarize_scores,
)
from pre_experiments.local_global_consistency.thresholds import (
    fit_frozen_thresholds,
)
from pre_experiments.local_global_consistency.windows import build_sliding_windows


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect_run_rows(run_dir: Path) -> dict[str, object]:
    """Load and validate every declared local window, then build scalar rows."""
    run_dir = run_dir.resolve()
    metadata = _json_object(run_dir / "run_metadata.json")
    invocation = metadata.get("invocation")
    if not isinstance(invocation, dict):
        raise ValueError("run metadata must contain invocation")
    source = Path(str(invocation["source_run_dir"]))
    scenes = invocation.get("scenes")
    if not isinstance(scenes, list) or not all(isinstance(scene, str) for scene in scenes):
        raise ValueError("run metadata invocation must contain scenes")
    try:
        window_length = int(invocation["window_length"])
        window_stride = int(invocation["window_stride"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run metadata must declare integer window parameters") from error

    all_observations = []
    all_overlaps = []
    all_scores = []
    all_validation = []
    for scene in scenes:
        global_artifact = load_global_context(
            source / scene / "frames_500" / "context_diagnostics.npz"
        )
        expected_windows = build_sliding_windows(
            global_artifact["frame_ids"],
            length=window_length,
            stride=window_stride,
        )
        window_records = []
        for directory in sorted((run_dir / scene).glob("window_*")):
            completion = _json_object(directory / "complete.json")
            if completion.get("run_id") != metadata.get("run_id"):
                raise ValueError(f"window run_id mismatch: {directory}")
            if completion.get("scene") != scene:
                raise ValueError(f"window scene mismatch: {directory}")
            if metadata.get("partition") is not None and completion.get(
                "partition"
            ) != metadata.get("partition"):
                raise ValueError(f"window partition mismatch: {directory}")
            if metadata.get("split_digest") is not None and completion.get(
                "split_digest"
            ) != metadata.get("split_digest"):
                raise ValueError(f"window split digest mismatch: {directory}")
            window_records.append(
                {
                    "index": completion["window_index"],
                    "start": completion["start"],
                    "stop": completion["stop"],
                    "artifact": load_window_diagnostics(
                        directory / "window_diagnostics.npz"
                    ),
                }
            )
        if not window_records:
            raise ValueError(f"no completed local windows for {scene}")
        actual_boundaries = [
            (int(record["index"]), int(record["start"]), int(record["stop"]))
            for record in window_records
        ]
        expected_boundaries = [
            (window.index, window.start, window.stop) for window in expected_windows
        ]
        if actual_boundaries != expected_boundaries:
            raise ValueError(f"incomplete or unexpected window set for {scene}")
        observations, overlaps, scores, validation = build_scene_rows(
            scene, global_artifact, window_records
        )
        all_observations.extend(observations)
        all_overlaps.extend(overlaps)
        all_scores.extend(scores)
        all_validation.extend(validation)

    return {
        "observations": all_observations,
        "overlaps": all_overlaps,
        "scores": all_scores,
        "validation": all_validation,
        "window_count": len(list(run_dir.glob("*/window_*"))),
    }


def _formal_metadata(
    metadata: dict[str, object],
    *,
    mode: str,
) -> tuple[dict[str, object], list[str]]:
    if mode not in {"calibration", "holdout"}:
        raise ValueError("analysis mode must be calibration or holdout")
    if metadata.get("partition") != mode:
        raise ValueError(f"{mode} analysis requires a {mode} run")
    if metadata.get("protocol_complete") is not True:
        raise ValueError("smoke or incomplete runs cannot produce formal analysis")
    invocation = metadata.get("invocation")
    if not isinstance(invocation, dict):
        raise ValueError("run metadata must contain invocation")
    for field in ("source_run_id", "split_digest", "partition", "protocol_complete"):
        if invocation.get(field) != metadata.get(field):
            raise ValueError(f"run metadata {field} disagrees with invocation")
    expected_protocol = {
        "window_length": 100,
        "window_stride": 50,
        "camera_iterations": 4,
        "preprocess_mode": "pad",
    }
    for field, expected in expected_protocol.items():
        if invocation.get(field) != expected:
            raise ValueError(
                f"formal analysis requires {field}={expected!r}, "
                f"found {invocation.get(field)!r}"
            )
    scenes = invocation.get("scenes")
    partition_scenes = invocation.get("partition_scenes")
    expected_count = 10 if mode == "calibration" else 40
    if (
        not isinstance(scenes, list)
        or len(scenes) != expected_count
        or len(set(scenes)) != expected_count
        or scenes != partition_scenes
    ):
        raise ValueError(
            f"formal {mode} analysis requires the complete ordered "
            f"{expected_count}-scene partition"
        )
    return invocation, scenes


def _calibration_analysis(
    run_dir: Path,
    metadata: dict[str, object],
    scenes: list[str],
    collected: dict[str, object],
) -> dict[str, object]:
    if collected.get("window_count") != 90:
        raise ValueError("formal calibration requires exactly 90 completed windows")
    score_rows = collected.get("scores")
    validation_rows = collected.get("validation")
    if not isinstance(score_rows, list) or not isinstance(validation_rows, list):
        raise ValueError("analysis row collection is invalid")
    provenance = {
        "calibration_scenes": scenes,
        "source_run_id": metadata.get("source_run_id"),
        "calibration_run_id": metadata.get("run_id"),
        "split_digest": metadata.get("split_digest"),
        "code_commit": metadata.get("git_commit"),
    }
    threshold_payload = fit_frozen_thresholds(score_rows, provenance)
    thresholds = threshold_payload["thresholds"]
    if not isinstance(thresholds, dict):
        raise ValueError("frozen threshold payload is invalid")
    scored = apply_reliability(score_rows, thresholds)
    summaries = summarize_scores(scored, validation_rows)
    outputs = {
        "calibration_prediction_scores_per_frame.csv": scored,
        "calibration_gt_validation_per_frame.csv": validation_rows,
        "calibration_summary.csv": summaries,
    }
    for filename, rows in outputs.items():
        _write_csv(run_dir / filename, rows)
    summary_payload = {
        "mode": "calibration",
        "run_id": metadata.get("run_id"),
        "source_run_id": metadata.get("source_run_id"),
        "split_digest": metadata.get("split_digest"),
        "threshold_digest": threshold_payload["threshold_digest"],
        "rows": summaries,
    }
    atomic_write_json(run_dir / "calibration_summary.json", summary_payload)
    atomic_write_json(
        run_dir / "frozen_reliability_thresholds.json", threshold_payload
    )
    completion = {
        "run_id": metadata.get("run_id"),
        "mode": "calibration",
        "partition": "calibration",
        "scenes": scenes,
        "window_count": collected["window_count"],
        "analysis_complete": True,
        "split_digest": metadata.get("split_digest"),
        "source_run_id": metadata.get("source_run_id"),
        "threshold_digest": threshold_payload["threshold_digest"],
    }
    atomic_write_json(run_dir / "complete.json", completion)
    return completion


def write_analysis(
    run_dir: Path,
    *,
    mode: str,
    thresholds_path: Path | None = None,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    metadata = _json_object(run_dir / "run_metadata.json")
    _, scenes = _formal_metadata(metadata, mode=mode)
    if mode == "calibration" and thresholds_path is not None:
        raise ValueError("calibration analysis fits its own frozen thresholds")
    if mode == "holdout":
        if thresholds_path is None:
            raise ValueError("holdout analysis requires an external threshold artifact")
        raise ValueError("holdout analysis is implemented in the holdout stage")
    atomic_write_json(
        run_dir / "complete.json",
        {
            "run_id": metadata.get("run_id"),
            "mode": mode,
            "analysis_complete": False,
            "status": "analysis_in_progress",
        },
    )
    collected = collect_run_rows(run_dir)
    return _calibration_analysis(run_dir, metadata, scenes, collected)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["calibration", "holdout"], required=True)
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args(argv)
    completion = write_analysis(
        args.run_dir,
        mode=args.mode,
        thresholds_path=args.thresholds,
    )
    print(
        f"[done] analysis={args.run_dir.resolve()} "
        f"mode={args.mode} windows={completion['window_count']}"
    )


if __name__ == "__main__":
    main()
