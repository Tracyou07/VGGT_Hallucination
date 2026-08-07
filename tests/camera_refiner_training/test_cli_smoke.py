import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def poses(centers: np.ndarray) -> np.ndarray:
    value = np.tile(np.eye(4), (len(centers), 1, 1))
    value[:, :3, 3] = centers
    return value


class CliSmokeTest(unittest.TestCase):
    def test_train_then_infer_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            local = root / "local"
            dataset.mkdir()
            shards = []
            frame_ids = np.arange(0, 200, 2, dtype=np.int64)
            x = np.linspace(0.0, 5.0, 100)
            centers = np.stack([x, np.sin(x), np.zeros_like(x)], axis=1)
            for scene, role, offset in (
                ("train_scene", "train", 0.05),
                ("validation_scene", "validation", 0.08),
            ):
                shard = dataset / f"{scene}.npz"
                global_hidden = np.zeros((1, 100, 4), dtype=np.float32)
                local_hidden = np.ones((1, 1, 100, 4), dtype=np.float32)
                np.savez(
                    shard,
                    scene_name=np.asarray(scene),
                    frame_ids=frame_ids,
                    scales=np.asarray([100]),
                    global_hidden=global_hidden,
                    local_hidden=local_hidden,
                    selected_boundary_distance=np.minimum(np.arange(100), np.arange(99, -1, -1))[None],
                    local_observation_count=np.ones((1, 100), dtype=np.int64),
                    pred_c2w_raw=poses(centers)[None],
                    gt_c2w_raw=poses(centers + np.asarray([offset, 0.0, 0.0])),
                )
                window = local / scene / "window_000"
                window.mkdir(parents=True)
                np.savez(
                    window / "window_diagnostics.npz",
                    frame_ids=frame_ids,
                    pred_c2w_raw=poses(centers),
                )
                shards.append({"scene": scene, "role": role, "path": shard.name})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"dataset_digest": "smoke", "shards": shards}), encoding="utf-8")
            units = root / "units.json"
            units.write_text(json.dumps({"translation_units": [0, 1]}), encoding="utf-8")
            output = root / "train"
            common = [
                "--dataset-manifest", str(manifest),
                "--dataset-root", str(dataset),
                "--local-run-dir", str(local),
                "--frozen-units", str(units),
                "--device", "cpu",
                "--unit-count", "2",
            ]

            subprocess.run(
                [
                    sys.executable, "-m", "pre_experiments.camera_refiner_training.train",
                    *common,
                    "--out-dir", str(output),
                    "--model-kind", "deterministic",
                    "--epochs", "1",
                    "--hidden-size", "16",
                    "--depth", "1",
                    "--num-heads", "4",
                    "--diffusion-steps", "10",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            inference = root / "inference"
            subprocess.run(
                [
                    sys.executable, "-m", "pre_experiments.camera_refiner_training.infer",
                    *common,
                    "--checkpoint", str(output / "best.pt"),
                    "--out-dir", str(inference),
                    "--role", "validation",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            with np.load(inference / "validation_scene" / "refined_camera.npz") as result:
                np.testing.assert_array_equal(result["frame_ids"], frame_ids)
                np.testing.assert_array_equal(
                    result["refined_c2w"][:, :3, :3], result["baseline_c2w"][:, :3, :3]
                )
            summary = json.loads((inference / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["scene"], "validation_scene")
            self.assertEqual(summary[0]["rotation_max_abs_change"], 0.0)


if __name__ == "__main__":
    unittest.main()
