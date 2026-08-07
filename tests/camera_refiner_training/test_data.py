import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_refiner_training.data import (
    build_scene_windows,
    load_dataset_manifest,
    load_translation_units,
)


def poses(centers: np.ndarray) -> np.ndarray:
    value = np.tile(np.eye(4), (len(centers), 1, 1))
    value[:, :3, 3] = centers
    return value


class RefinerDataTest(unittest.TestCase):
    def test_strict_manifest_rejects_digest_or_shard_tampering(self):
        shard = self.root / "a.npz"
        shard.write_bytes(b"original")
        shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
        payload = {
            "schema_version": 1,
            "shards": [
                {
                    "scene": "scene_a",
                    "role": "train",
                    "path": "a.npz",
                    "sha256": shard_hash,
                }
            ],
        }
        payload["dataset_digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")

        load_dataset_manifest(manifest, self.root, roles={"train"})
        shard.write_bytes(b"changed")

        with self.assertRaisesRegex(ValueError, "checksum"):
            load_dataset_manifest(manifest, self.root, roles={"train"})
        payload["dataset_digest"] = "bad"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            load_dataset_manifest(manifest, self.root, roles={"train"})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_translation_manifest_is_authenticated_and_iteration_filtered(self):
        path = self.root / "units.json"
        payload = {
            "scores": {
                "translation": [
                    {"iteration": 0, "unit": 8, "score": 3.0},
                    {"iteration": 1, "unit": 9, "score": 2.0},
                    {"iteration": 0, "unit": 3, "score": 1.0},
                ]
            },
            "frozen_digest": "source-digest",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

        selection = load_translation_units(path, count=2, iteration=0)

        self.assertEqual(selection.indices, (8, 3))
        self.assertEqual(selection.iteration, 0)
        self.assertRegex(selection.digest, r"^[0-9a-f]{64}$")

    def test_manifest_filters_roles_and_resolves_shards_inside_root(self):
        (self.root / "a.npz").touch()
        (self.root / "b.npz").touch()
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "dataset_digest": "dataset-id",
                    "shards": [
                        {"scene": "scene_a", "role": "train", "path": "a.npz"},
                        {"scene": "scene_b", "role": "validation", "path": "b.npz"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        dataset = load_dataset_manifest(manifest, self.root, roles={"train"})

        self.assertEqual(dataset.digest, "dataset-id")
        self.assertEqual([entry.scene for entry in dataset.entries], ["scene_a"])
        self.assertEqual(dataset.entries[0].shard, (self.root / "a.npz").resolve())

    def test_scene_conditions_do_not_depend_on_gt(self):
        frame_ids = np.arange(100, dtype=np.int64)
        global_centers = np.stack(
            [np.linspace(0.0, 4.0, 100), np.sin(np.linspace(0, 2, 100)), np.zeros(100)],
            axis=1,
        )
        local_centers = global_centers * 2.0 + np.asarray([4.0, -2.0, 1.0])
        global_hidden = np.zeros((1, 100, 12), dtype=np.float32)
        local_hidden = np.zeros((1, 1, 100, 12), dtype=np.float32)
        global_hidden[0, :, 3] = np.linspace(0.0, 1.0, 100)
        global_hidden[0, :, 8] = 2.0
        local_hidden[0, 0, :, 3] = 1.0
        local_hidden[0, 0, :, 8] = 3.0
        shard = self.root / "scene_shard.npz"

        def write_shard(gt_offset: float) -> None:
            np.savez(
                shard,
                scene_name=np.asarray("scene_a"),
                frame_ids=frame_ids,
                scales=np.asarray([100]),
                global_hidden=global_hidden,
                local_hidden=local_hidden,
                selected_boundary_distance=np.minimum(frame_ids, 99 - frame_ids)[None],
                local_observation_count=np.ones((1, 100), dtype=np.int64),
                pred_c2w_raw=poses(global_centers)[None],
                gt_c2w_raw=poses(
                    global_centers
                    + np.stack(
                        [gt_offset * np.sin(np.linspace(0.0, 4.0, 100)), np.zeros(100), np.zeros(100)],
                        axis=1,
                    )
                ),
            )

        scene_dir = self.root / "scene_a"
        window_dir = scene_dir / "window_000"
        window_dir.mkdir(parents=True)
        np.savez(
            window_dir / "window_diagnostics.npz",
            frame_ids=frame_ids,
            pred_c2w_raw=poses(local_centers),
        )
        units_path = self.root / "units.json"
        units_path.write_text(
            json.dumps(
                {
                    "translation_units": [
                        {"iteration": 0, "unit": 8},
                        {"iteration": 0, "unit": 3},
                    ]
                }
            ),
            encoding="utf-8",
        )
        selection = load_translation_units(units_path, count=2, iteration=0)

        write_shard(0.5)
        first = build_scene_windows(shard, scene_dir, selection)
        write_shard(3.0)
        second = build_scene_windows(shard, scene_dir, selection)

        self.assertEqual(first.condition.shape, (1, 100, 19))
        np.testing.assert_allclose(first.condition, second.condition)
        self.assertFalse(np.allclose(first.target_residual, second.target_residual))
        np.testing.assert_array_equal(first.frame_ids[0], frame_ids)
        np.testing.assert_array_equal(first.global_c2w[:, :3, :3], np.tile(np.eye(3), (100, 1, 1)))


if __name__ == "__main__":
    unittest.main()
