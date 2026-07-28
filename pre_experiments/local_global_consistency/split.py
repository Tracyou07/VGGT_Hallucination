"""Build and validate a raw-GT-only ScanNet-50 calibration/holdout split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pre_experiments.common.contracts import atomic_write_json
from pre_experiments.common.scannet import load_scene_frames, uniform_frame_ids
from pre_experiments.local_global_consistency.context_source import (
    load_context_frame_ids,
    validate_context_source_metadata,
)


FIXED_OBSERVED_SCENES = (
    "scene0000_00",
    "scene0013_02",
    "scene0029_01",
    "scene0691_00",
)
FEATURE_NAMES = (
    "cumulative_translation",
    "cumulative_rotation_deg",
    "p95_translation_step",
    "p95_rotation_step_deg",
)
SCHEMA_VERSION = 1


def _canonical_digest(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rotation_step_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def motion_features(raw_c2w: np.ndarray) -> dict[str, float]:
    """Summarize raw camera motion without reading prediction outcomes."""
    poses = np.asarray(raw_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("raw_c2w must have shape (N, 4, 4) with N >= 2")
    if not np.isfinite(poses).all():
        raise ValueError("raw_c2w must contain only finite values")
    rotations = poses[:, :3, :3]
    identity = np.eye(3)
    for rotation in rotations:
        if not np.allclose(rotation.T @ rotation, identity, atol=1e-5, rtol=0):
            raise ValueError("raw_c2w contains a non-orthonormal rotation")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0):
            raise ValueError("raw_c2w contains an invalid rotation determinant")
    translation_steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    rotation_steps = np.asarray(
        [
            _rotation_step_degrees(rotations[index], rotations[index + 1])
            for index in range(len(rotations) - 1)
        ],
        dtype=np.float64,
    )
    return {
        "cumulative_translation": float(np.sum(translation_steps)),
        "cumulative_rotation_deg": float(np.sum(rotation_steps)),
        "p95_translation_step": float(np.percentile(translation_steps, 95)),
        "p95_rotation_step_deg": float(np.percentile(rotation_steps, 95)),
    }


def _average_tie_percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    scenes = list(values)
    if len(scenes) < 2:
        raise ValueError("percentile ranking requires at least two scenes")
    array = np.asarray([values[scene] for scene in scenes], dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2.0
        ranks[order[start:stop]] = average_rank / (len(order) - 1)
        start = stop
    return {scene: float(ranks[index]) for index, scene in enumerate(scenes)}


def build_split_manifest(
    scene_ids: Sequence[str],
    raw_c2w_by_scene: Mapping[str, np.ndarray],
    *,
    source_run_id: str,
    seed: int = 33,
) -> dict[str, object]:
    """Create the deterministic ten-scene calibration and 40-scene holdout."""
    scenes = list(scene_ids)
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise ValueError("ScanNet-50 split requires exactly 50 unique scenes")
    if not all(isinstance(scene, str) and scene for scene in scenes):
        raise ValueError("scene IDs must be non-empty strings")
    if not set(FIXED_OBSERVED_SCENES).issubset(scenes):
        raise ValueError("all four fixed observed scenes must be present")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("source_run_id must be a non-empty string")
    if set(raw_c2w_by_scene) != set(scenes):
        raise ValueError("raw trajectory scene set must exactly match scene_ids")

    candidates = [scene for scene in scenes if scene not in FIXED_OBSERVED_SCENES]
    features = {
        scene: motion_features(raw_c2w_by_scene[scene]) for scene in candidates
    }
    ranks_by_feature = {
        name: _average_tie_percentile_ranks(
            {scene: features[scene][name] for scene in candidates}
        )
        for name in FEATURE_NAMES
    }
    difficulty = {
        scene: float(np.mean([ranks_by_feature[name][scene] for name in FEATURE_NAMES]))
        for scene in candidates
    }
    ordered_by_difficulty = sorted(candidates, key=lambda scene: (difficulty[scene], scene))
    strata: dict[str, str] = {}
    for stratum, members in zip(
        ("easy", "medium", "hard"),
        np.array_split(np.asarray(ordered_by_difficulty, dtype=object), 3),
    ):
        for scene in members.tolist():
            strata[str(scene)] = stratum

    selection_hashes = {
        scene: hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).hexdigest()
        for scene in candidates
    }
    selected: list[str] = []
    for stratum in ("easy", "medium", "hard"):
        members = [scene for scene in candidates if strata[scene] == stratum]
        selected.extend(sorted(members, key=lambda scene: selection_hashes[scene])[:2])
    selected_set = set(selected)
    calibration = list(FIXED_OBSERVED_SCENES) + [
        scene for scene in scenes if scene in selected_set
    ]
    holdout = [scene for scene in scenes if scene not in set(calibration)]

    scene_difficulty = {
        scene: {
            "features": features[scene],
            "percentile_ranks": {
                name: ranks_by_feature[name][scene] for name in FEATURE_NAMES
            },
            "difficulty_score": difficulty[scene],
            "stratum": strata[scene],
            "selection_hash": selection_hashes[scene],
            "selected_for_calibration": scene in selected_set,
        }
        for scene in candidates
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "seed": int(seed),
        "scene_order": scenes,
        "fixed_observed_scenes": list(FIXED_OBSERVED_SCENES),
        "calibration_scenes": calibration,
        "holdout_scenes": holdout,
        "new_calibration_scenes": [scene for scene in scenes if scene in selected_set],
        "new_calibration_strata": {scene: strata[scene] for scene in selected},
        "scene_difficulty": scene_difficulty,
    }
    payload["split_digest"] = _canonical_digest(payload)
    return payload


def load_split_manifest(path: Path, expected_scenes: Sequence[str]) -> dict[str, object]:
    """Load a split only when its complete identity and digest are valid."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid split manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("split manifest must be a JSON object")
    digest = payload.get("split_digest")
    unsigned = {key: value for key, value in payload.items() if key != "split_digest"}
    if not isinstance(digest, str) or digest != _canonical_digest(unsigned):
        raise ValueError("split manifest digest mismatch")

    expected = list(expected_scenes)
    if len(expected) != 50 or len(set(expected)) != 50:
        raise ValueError("expected scene list must contain 50 unique scenes")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema")
    if payload.get("scene_order") != expected:
        raise ValueError("split manifest scene order does not match the expected list")
    if payload.get("fixed_observed_scenes") != list(FIXED_OBSERVED_SCENES):
        raise ValueError("split manifest fixed observed scenes are invalid")
    calibration = payload.get("calibration_scenes")
    holdout = payload.get("holdout_scenes")
    if not isinstance(calibration, list) or not isinstance(holdout, list):
        raise ValueError("split manifest partitions must be lists")
    if len(calibration) != 10 or len(set(calibration)) != 10:
        raise ValueError("calibration partition must contain ten unique scenes")
    if len(holdout) != 40 or len(set(holdout)) != 40:
        raise ValueError("holdout partition must contain 40 unique scenes")
    if set(calibration).intersection(holdout) or set(calibration).union(holdout) != set(
        expected
    ):
        raise ValueError("calibration and holdout must exactly partition ScanNet-50")
    if not set(FIXED_OBSERVED_SCENES).issubset(calibration):
        raise ValueError("calibration is missing a fixed observed scene")
    new_scenes = payload.get("new_calibration_scenes")
    new_strata = payload.get("new_calibration_strata")
    if (
        not isinstance(new_scenes, list)
        or len(new_scenes) != 6
        or set(new_scenes) != set(calibration).difference(FIXED_OBSERVED_SCENES)
        or not isinstance(new_strata, dict)
        or set(new_strata) != set(new_scenes)
        or Counter(new_strata.values()) != Counter({"easy": 2, "medium": 2, "hard": 2})
    ):
        raise ValueError("new calibration scenes must contain two scenes per stratum")
    if not isinstance(payload.get("source_run_id"), str) or not payload["source_run_id"]:
        raise ValueError("split manifest source_run_id is missing")
    return payload


def _read_scene_list(path: Path) -> list[str]:
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"scene list contains duplicates: {path}")
    return scenes


def _build_from_paths(
    data_dir: Path,
    source_run_dir: Path,
    scenes: Sequence[str],
    seed: int,
) -> dict[str, object]:
    source_metadata = validate_context_source_metadata(source_run_dir, list(scenes))
    trajectories: dict[str, np.ndarray] = {}
    for scene in scenes:
        frame_ids = load_context_frame_ids(
            source_run_dir / scene / "frames_500" / "context_diagnostics.npz"
        )
        _, poses_by_id, valid_ids = load_scene_frames(data_dir, scene)
        expected_ids = uniform_frame_ids(valid_ids, 500)
        if not np.array_equal(frame_ids, np.asarray(expected_ids, dtype=np.int64)):
            raise ValueError(f"context frame IDs do not match raw ScanNet inputs: {scene}")
        trajectories[scene] = np.stack([poses_by_id[int(value)] for value in frame_ids])
    return build_split_manifest(
        scenes,
        trajectories,
        source_run_id=str(source_metadata["run_id"]),
        seed=seed,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args(argv)
    if args.validate is None:
        missing = [
            flag
            for flag, value in (
                ("--data-dir", args.data_dir),
                ("--source-run-dir", args.source_run_dir),
                ("--output", args.output),
            )
            if value is None
        ]
        if missing:
            parser.error(f"creation mode requires: {', '.join(missing)}")
    elif any(value is not None for value in (args.data_dir, args.source_run_dir, args.output)):
        parser.error("--validate cannot be combined with creation arguments")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scenes = _read_scene_list(args.scene_list.resolve())
    if args.validate is not None:
        manifest = load_split_manifest(args.validate.resolve(), scenes)
        print(
            f"[valid] calibration={len(manifest['calibration_scenes'])} "
            f"holdout={len(manifest['holdout_scenes'])} "
            f"digest={manifest['split_digest']}"
        )
        return
    manifest = _build_from_paths(
        args.data_dir.resolve(),
        args.source_run_dir.resolve(),
        scenes,
        args.seed,
    )
    atomic_write_json(args.output.resolve(), manifest)
    print(f"[done] split={args.output.resolve()} digest={manifest['split_digest']}")


if __name__ == "__main__":
    main()
