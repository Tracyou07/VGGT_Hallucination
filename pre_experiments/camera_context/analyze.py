"""Analyze matched Camera frames across context lengths without a GPU."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from pre_experiments.camera_context.metrics import compare_contexts


def _load_context(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid context diagnostic artifact: {path}") from error


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty context table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_context_analysis(
    run_dir: Path,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Compare every ordered context pair per scene and write compact tables."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    frame_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    for scene_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        contexts: list[tuple[int, dict[str, np.ndarray]]] = []
        for selection_dir in scene_dir.iterdir():
            if not selection_dir.is_dir() or not selection_dir.name.startswith("frames_"):
                continue
            try:
                frame_count = int(selection_dir.name.removeprefix("frames_"))
            except ValueError:
                continue
            artifact_path = selection_dir / "context_diagnostics.npz"
            if artifact_path.is_file():
                contexts.append((frame_count, _load_context(artifact_path)))
        contexts.sort(key=lambda item: item[0])
        for (short_frames, short), (long_frames, long) in combinations(contexts, 2):
            rows, summary = compare_contexts(
                short,
                long,
                short_frames=short_frames,
                long_frames=long_frames,
            )
            frame_rows.extend({"scene": scene_dir.name, **row} for row in rows)
            summaries.append({"scene": scene_dir.name, **summary})

    if not frame_rows or not summaries:
        raise ValueError("run must contain at least two context artifacts for one scene")
    _write_csv(run_dir / "context_per_frame.csv", frame_rows)
    _write_csv(run_dir / "context_summary.csv", summaries)
    temporary_json = run_dir / "context_summary.json.tmp"
    temporary_json.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(run_dir / "context_summary.json")
    return frame_rows, summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    frame_rows, summaries = write_context_analysis(args.run_dir)
    print(f"per_frame_rows={len(frame_rows)}")
    print(f"summary_rows={len(summaries)}")
    print(f"results={args.run_dir.resolve()}")


if __name__ == "__main__":
    main()
