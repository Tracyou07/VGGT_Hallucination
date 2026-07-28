from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.local_global_consistency.split import (
    FIXED_OBSERVED_SCENES,
    build_split_manifest,
    load_split_manifest,
    main,
    motion_features,
)


def _trajectory(scale: float, frames: int = 12) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    for index in range(frames):
        angle = np.deg2rad(scale * index)
        cosine, sine = np.cos(angle), np.sin(angle)
        poses[index, :3, :3] = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        poses[index, :3, 3] = [scale * index, 0.0, 0.0]
    return poses


class MotionFeatureTest(unittest.TestCase):
    def test_motion_features_capture_translation_rotation_and_p95_steps(self):
        features = motion_features(_trajectory(2.0, frames=6))

        self.assertAlmostEqual(features["cumulative_translation"], 10.0)
        self.assertAlmostEqual(features["cumulative_rotation_deg"], 10.0)
        self.assertAlmostEqual(features["p95_translation_step"], 2.0)
        self.assertAlmostEqual(features["p95_rotation_step_deg"], 2.0)

    def test_motion_features_rejects_invalid_trajectory(self):
        for poses in (
            np.eye(4),
            np.repeat(np.eye(4)[None], 1, axis=0),
            np.full((2, 4, 4), np.nan),
        ):
            with self.subTest(shape=poses.shape):
                with self.assertRaises(ValueError):
                    motion_features(poses)


class SplitManifestTest(unittest.TestCase):
    def setUp(self):
        candidates = [f"scene{index:04d}_00" for index in range(100, 146)]
        self.scenes = list(FIXED_OBSERVED_SCENES) + candidates
        self.trajectories = {
            scene: _trajectory(float(index + 1) / 10.0)
            for index, scene in enumerate(self.scenes)
        }

    def test_builds_deterministic_leakage_controlled_ten_forty_split(self):
        first = build_split_manifest(
            self.scenes,
            self.trajectories,
            source_run_id="source-run",
            seed=33,
        )
        second = build_split_manifest(
            self.scenes,
            self.trajectories,
            source_run_id="source-run",
            seed=33,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["calibration_scenes"]), 10)
        self.assertEqual(len(first["holdout_scenes"]), 40)
        self.assertEqual(
            set(first["calibration_scenes"]) | set(first["holdout_scenes"]),
            set(self.scenes),
        )
        self.assertFalse(
            set(first["calibration_scenes"]) & set(first["holdout_scenes"])
        )
        self.assertTrue(
            set(FIXED_OBSERVED_SCENES).issubset(first["calibration_scenes"])
        )
        self.assertEqual(
            Counter(first["new_calibration_strata"].values()),
            {"easy": 2, "medium": 2, "hard": 2},
        )
        self.assertEqual(len(first["split_digest"]), 64)

    def test_load_manifest_rejects_tampering_or_scene_mismatch(self):
        manifest = build_split_manifest(
            self.scenes,
            self.trajectories,
            source_run_id="source-run",
            seed=33,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_split_manifest(path, self.scenes)
            self.assertEqual(loaded, manifest)

            manifest["holdout_scenes"].reverse()
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_split_manifest(path, self.scenes)

            path.write_text(json.dumps(loaded), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_split_manifest(path, list(reversed(self.scenes)))

    def test_split_cli_rejects_source_protocol_before_reading_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "source-run",
                        "invocation": {
                            "scenes": self.scenes,
                            "frame_counts": [100, 500],
                            "iterations": [4],
                            "sampling": "nested_uniform",
                            "preprocess_mode": "pad",
                            "save_context_diagnostics": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            scene_list = root / "scenes.txt"
            scene_list.write_text("\n".join(self.scenes) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frame_counts"):
                main(
                    [
                        "--data-dir",
                        str(root / "processed"),
                        "--scene-list",
                        str(scene_list),
                        "--source-run-dir",
                        str(source),
                        "--output",
                        str(root / "split.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
