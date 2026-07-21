"""Analyze a completed Round 2A local-window run on CPU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pre_experiments.camera_iteration.contracts import atomic_write_json
from pre_experiments.local_global_consistency.artifacts import (
    load_global_context,
    load_window_diagnostics,
)
from pre_experiments.local_global_consistency.metrics import (
    apply_reliability,
    build_scene_rows,
    fit_reliability_thresholds,
    summarize_scores,
)
from pre_experiments.local_global_consistency.windows import build_sliding_windows


DEFAULT_STABLE_SCENES = {"scene0013_02", "scene0029_01"}


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
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


def write_analysis(
    run_dir: Path,
    *,
    stable_scenes: set[str] = DEFAULT_STABLE_SCENES,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    metadata = _json_object(run_dir / "run_metadata.json")
    atomic_write_json(
        run_dir / "complete.json",
        {
            "run_id": metadata.get("run_id"),
            "analysis_complete": False,
            "status": "analysis_in_progress",
        },
    )
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

    available_controls = stable_scenes.intersection(scenes)
    thresholds = (
        fit_reliability_thresholds(all_scores, stable_scenes=available_controls)
        if available_controls
        else None
    )
    scored = apply_reliability(all_scores, thresholds)
    summaries = summarize_scores(scored, all_validation)
    threshold_payload = {
        "fitted": thresholds is not None,
        "stable_scenes": sorted(available_controls),
        "thresholds": thresholds,
    }
    outputs = {
        "local_observations.csv": all_observations,
        "local_overlap_scores.csv": all_overlaps,
        "prediction_scores_per_frame.csv": scored,
        "gt_validation_per_frame.csv": all_validation,
        "local_global_summary.csv": summaries,
    }
    for filename, rows in outputs.items():
        _write_csv(run_dir / filename, rows)
    atomic_write_json(run_dir / "local_global_summary.json", summaries)
    atomic_write_json(run_dir / "reliability_thresholds.json", threshold_payload)
    completion = {
        "run_id": metadata.get("run_id"),
        "scenes": scenes,
        "window_count": len(list(run_dir.glob("*/window_*"))),
        "analysis_complete": True,
        "reliability_thresholds_fitted": thresholds is not None,
    }
    atomic_write_json(run_dir / "complete.json", completion)
    return completion


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stable-scenes", nargs="*", default=sorted(DEFAULT_STABLE_SCENES))
    args = parser.parse_args(argv)
    completion = write_analysis(args.run_dir, stable_scenes=set(args.stable_scenes))
    print(f"[done] analysis={args.run_dir.resolve()} windows={completion['window_count']}")


if __name__ == "__main__":
    main()
