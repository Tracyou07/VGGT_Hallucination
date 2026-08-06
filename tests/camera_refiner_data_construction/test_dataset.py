import tempfile
import unittest
from pathlib import Path

import numpy as np

from pre_experiments.camera_refiner_data_construction.artifacts import (
    save_scene_shard,
)
from pre_experiments.camera_refiner_data_construction.dataset import (
    build_dataset_manifest,
    validate_dataset_manifest,
    write_dataset_manifest,
)


def _scene_payload(scene: str) -> dict[str, object]:
    frame_count = 4
    candidates = 2
    return {
        "scene": scene,
        "frame_ids": np.arange(frame_count),
        "scales": np.array([100, 200, 300]),
        "candidate_names": np.array(["baseline", "candidate"]),
        "candidate_alpha": np.array([0.0, 0.02]),
        "candidate_beta": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "global_hidden": np.zeros((1, frame_count, 2), dtype=np.float32),
        "local_hidden": np.ones((3, 1, frame_count, 2), dtype=np.float32),
        "selected_window_index": np.zeros((3, frame_count), dtype=np.int64),
        "selected_boundary_distance": np.zeros((3, frame_count), dtype=np.int64),
        "selected_window_start": np.zeros((3, frame_count), dtype=np.int64),
        "selected_window_stop": np.array(
            [[100] * frame_count, [200] * frame_count, [300] * frame_count]
        ),
        "local_observation_count": np.ones((3, frame_count), dtype=np.int64),
        "pred_c2w_raw": np.tile(np.eye(4), (candidates, frame_count, 1, 1)),
        "pose_enc": np.zeros((candidates, frame_count, 9)),
        "gt_c2w_raw": np.tile(np.eye(4), (frame_count, 1, 1)),
        "translation_error_aligned": np.zeros((candidates, frame_count)),
        "rotation_error_deg_aligned": np.zeros((candidates, frame_count)),
        "hidden_displacement_rms": np.array([0.0, 0.01]),
        "camera_center_displacement_mean": np.array([0.0, 0.01]),
        "rotation_change_deg_mean": np.array([0.0, 0.01]),
        "fov_change_mean": np.array([0.0, 0.001]),
    }


class DatasetManifestTest(unittest.TestCase):
    def test_builds_and_validates_portable_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = {}
            for scene in ("scene_train", "scene_validation"):
                path = root / "shards" / f"{scene}.npz"
                save_scene_shard(path, _scene_payload(scene))
                shards[scene] = path
            roles = {
                "train": ["scene_train"],
                "validation": ["scene_validation"],
            }
            provenance = {
                "git_commit": "a" * 40,
                "checkpoint_digest": "checkpoint",
                "split_digest": "split",
                "frozen_policy_digest": "policy",
            }
            first = build_dataset_manifest(
                root,
                shards,
                roles,
                protected_holdout_scenes=["scene_holdout"],
                provenance=provenance,
            )
            second = build_dataset_manifest(
                root,
                shards,
                roles,
                protected_holdout_scenes=["scene_holdout"],
                provenance=provenance,
            )
            self.assertEqual(first["dataset_digest"], second["dataset_digest"])
            self.assertEqual(first["shards"][0]["path"], "shards/scene_train.npz")
            manifest_path = root / "dataset_manifest.json"
            write_dataset_manifest(manifest_path, first)

            report = validate_dataset_manifest(
                manifest_path,
                root,
                protected_holdout_scenes=["scene_holdout"],
            )

        self.assertEqual(report["scene_count"], 2)
        self.assertEqual(report["role_counts"], {"train": 1, "validation": 1})
        self.assertEqual(report["frame_count"], 8)

    def test_rejects_holdout_leakage_and_duplicate_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "scene_holdout.npz"
            save_scene_shard(path, _scene_payload("scene_holdout"))
            with self.assertRaisesRegex(ValueError, "holdout"):
                build_dataset_manifest(
                    root,
                    {"scene_holdout": path},
                    {"train": ["scene_holdout"]},
                    protected_holdout_scenes=["scene_holdout"],
                    provenance={
                        "git_commit": "a" * 40,
                        "checkpoint_digest": "checkpoint",
                        "split_digest": "split",
                        "frozen_policy_digest": "policy",
                    },
                )
            with self.assertRaisesRegex(ValueError, "multiple roles"):
                build_dataset_manifest(
                    root,
                    {"scene_holdout": path},
                    {
                        "calibration": ["scene_holdout"],
                        "validation": ["scene_holdout"],
                    },
                    protected_holdout_scenes=[],
                    provenance={
                        "git_commit": "a" * 40,
                        "checkpoint_digest": "checkpoint",
                        "split_digest": "split",
                        "frozen_policy_digest": "policy",
                    },
                )

    def test_detects_shard_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "scene_train.npz"
            save_scene_shard(path, _scene_payload("scene_train"))
            manifest = build_dataset_manifest(
                root,
                {"scene_train": path},
                {"train": ["scene_train"]},
                protected_holdout_scenes=[],
                provenance={
                    "git_commit": "a" * 40,
                    "checkpoint_digest": "checkpoint",
                    "split_digest": "split",
                    "frozen_policy_digest": "policy",
                },
            )
            manifest_path = root / "dataset_manifest.json"
            write_dataset_manifest(manifest_path, manifest)
            path.write_bytes(path.read_bytes() + b"tampered")

            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_dataset_manifest(
                    manifest_path,
                    root,
                    protected_holdout_scenes=[],
                )


if __name__ == "__main__":
    unittest.main()
