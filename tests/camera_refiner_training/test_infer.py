import unittest

import numpy as np
import torch

from pre_experiments.camera_refiner_training.data import SceneWindows
from pre_experiments.camera_refiner_training.geometry import SceneGauge
from pre_experiments.camera_refiner_training.infer import refine_scene, scene_frame_ids
from pre_experiments.camera_refiner_training.metrics import translation_metrics


def poses(centers: np.ndarray) -> np.ndarray:
    value = np.tile(np.eye(4), (len(centers), 1, 1))
    value[:, :3, 3] = centers
    return value


class ConstantRefiner(torch.nn.Module):
    def forward(self, noisy, condition, timestep):
        residual = torch.zeros_like(noisy)
        residual[..., 0] = 0.1
        confidence = torch.ones((*noisy.shape[:2], 1), device=noisy.device)
        return residual, confidence


class InferenceTest(unittest.TestCase):
    def test_scene_frame_ids_reconstruct_original_noncontiguous_sequence(self):
        windows = np.stack([np.arange(0, 200, 2), np.arange(100, 300, 2)])

        result = scene_frame_ids(windows, np.asarray([0, 50]), total_frames=150)

        np.testing.assert_array_equal(result, np.arange(0, 300, 2))

    def test_refinement_fuses_windows_and_preserves_rotation_exactly(self):
        centers = np.stack([np.linspace(0.0, 10.0, 150), np.zeros(150), np.zeros(150)], axis=1)
        global_c2w = poses(centers)
        gauge = SceneGauge.from_c2w(global_c2w)
        canonical = gauge.canonicalize(centers)
        scene = SceneWindows(
            scene="scene_a",
            condition=np.zeros((2, 100, 5), dtype=np.float32),
            target_residual=np.zeros((2, 100, 3), dtype=np.float32),
            global_centers=np.stack([canonical[:100], canonical[50:]]).astype(np.float32),
            frame_ids=np.stack([np.arange(100), np.arange(50, 150)]),
            starts=np.asarray([0, 50]),
            alignment_residual=np.zeros((2, 100), dtype=np.float32),
            global_c2w=global_c2w,
            gt_c2w_raw=global_c2w,
            gauge=gauge,
        )

        result = refine_scene(
            ConstantRefiner(),
            scene,
            condition_mean=torch.zeros(5),
            condition_std=torch.ones(5),
            model_kind="deterministic",
            diffusion_steps=20,
            sample_steps=5,
            seed=7,
            device=torch.device("cpu"),
        )

        np.testing.assert_array_equal(result.refined_c2w[:, :3, :3], global_c2w[:, :3, :3])
        expected = gauge.restore(canonical + np.asarray([0.1, 0.0, 0.0]))
        np.testing.assert_allclose(result.refined_c2w[:, :3, 3], expected, atol=1e-6)
        np.testing.assert_allclose(result.confidence, 1.0)

    def test_translation_metrics_align_prediction_and_keep_raw_metric(self):
        gt = np.stack([np.arange(8), np.sin(np.arange(8)), np.zeros(8)], axis=1).astype(float)
        prediction = gt * 2.0 + np.asarray([4.0, -3.0, 1.0])

        metrics = translation_metrics(poses(prediction), poses(gt))

        self.assertGreater(metrics["ate_raw"], 1.0)
        self.assertLess(metrics["ate_aligned"], 1e-8)


if __name__ == "__main__":
    unittest.main()
