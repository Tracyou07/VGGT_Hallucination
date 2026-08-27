from __future__ import annotations

import importlib
import inspect
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from pre_experiments.variational_camera_latent.source import save_source_shard


try:
    alpha_scan = importlib.import_module(
        "pre_experiments.variational_camera_latent.alpha_scan"
    )
except ModuleNotFoundError:
    alpha_scan = None


class AlphaScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "prediction_only" / "source" / "scene0000_00.npz"
        global_ids = np.arange(500, dtype=np.int64)
        global_tokens = np.zeros((500, 2048), dtype=np.float32)
        short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
        short_tokens[:, :50, 0] = 2.0
        short_tokens[:, 50:, 0] = 1.0
        short_ids = np.stack(
            [global_ids[start : start + 100] for start in range(0, 401, 50)]
        )
        global_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
        angle = np.linspace(0.0, 4.0 * np.pi, 500)
        global_c2w[:, 0, 3] = np.cos(angle)
        global_c2w[:, 1, 3] = np.sin(angle)
        global_c2w[:, 2, 3] = np.linspace(-1.0, 1.0, 500) ** 2
        overlap_c2w = np.stack(
            [global_c2w[start : start + 50] for start in range(50, 401, 50)]
        )
        save_source_shard(
            self.source,
            {
                "global_frame_ids": global_ids,
                "global_camera_tokens": global_tokens,
                "short_frame_ids": short_ids,
                "short_camera_tokens": short_tokens,
                "overlap_frame_ids": np.stack(
                    [global_ids[start : start + 50] for start in range(50, 401, 50)]
                ),
                "overlap_long_tokens": np.stack(
                    [global_tokens[start : start + 50] for start in range(50, 401, 50)]
                ),
                "overlap_left_tokens": short_tokens[:-1, 50:],
                "overlap_right_tokens": short_tokens[1:, :50],
                "span_starts": np.arange(0, 400, 50, dtype=np.int64),
                "sample_ids": np.asarray(
                    [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
                    dtype="U64",
                ),
                "global_pred_c2w": global_c2w,
                "overlap_long_c2w": overlap_c2w,
            },
        )
        self.prepared = self.root / "prepared" / "scene0000_00"
        self.prepared.mkdir(parents=True)
        np.save(self.prepared / "frame_ids.npy", global_ids, allow_pickle=False)
        np.save(self.prepared / "raw_gt_c2w.npy", global_c2w, allow_pickle=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _module(self):
        if alpha_scan is None:
            self.fail("alpha scan module is not implemented")
        return alpha_scan

    @staticmethod
    def _camera_head():
        class TokenTranslationHead:
            def decode_pose_tokens(self, tokens, *, num_iterations):
                raw = torch.zeros((*tokens.shape[:2], 9), device=tokens.device)
                raw[..., 0] = tokens[..., 0]
                raw[..., 3] = 1.0
                return [raw] * num_iterations

        return TokenTranslationHead()

    def test_prediction_scan_anchors_zero_and_reaches_both_short_endpoints(self) -> None:
        module = self._module()
        destination = (
            self.root / "prediction_only" / "alpha_scan" / "scene0000_00.npz"
        )

        module.generate_alpha_scan_candidates(
            self.source,
            destination,
            camera_head=self._camera_head(),
            checkpoint_sha256="a" * 64,
            device="cpu",
            alphas=(0.0, 0.1, 1.0),
        )
        arrays = module.load_alpha_scan_candidates(destination)

        np.testing.assert_array_equal(arrays["alphas"], [0.0, 0.1, 1.0])
        self.assertEqual(arrays["decoded_camera_raw"].shape, (8, 2, 3, 50, 9))
        self.assertEqual(arrays["decoded_camera_c2w"].shape, (8, 2, 3, 50, 4, 4))
        np.testing.assert_array_equal(
            arrays["decoded_camera_c2w"][:, 0, 0],
            np.load(self.source, allow_pickle=False)["overlap_long_c2w"],
        )
        np.testing.assert_allclose(arrays["decoded_camera_raw"][:, 0, 2, :, 0], 1.0)
        np.testing.assert_allclose(arrays["decoded_camera_raw"][:, 1, 2, :, 0], 2.0)
        self.assertFalse(
            any("gt" in name.lower() or "privileged" in name.lower() for name in arrays)
        )

    def test_prediction_scan_api_cannot_receive_privileged_data(self) -> None:
        module = self._module()
        parameters = inspect.signature(module.generate_alpha_scan_candidates).parameters

        self.assertNotIn("gt", parameters)
        self.assertNotIn("prepared", parameters)
        self.assertNotIn("privileged", parameters)

    def test_privileged_scan_is_separate_and_preserves_alpha_zero_baseline(self) -> None:
        module = self._module()
        prediction = (
            self.root / "prediction_only" / "alpha_scan" / "scene0000_00.npz"
        )
        sidecar = (
            self.root / "privileged_labels" / "alpha_scan" / "scene0000_00.npz"
        )
        module.generate_alpha_scan_candidates(
            self.source,
            prediction,
            camera_head=self._camera_head(),
            checkpoint_sha256="a" * 64,
            device="cpu",
            alphas=(0.0, 0.1, 1.0),
        )

        module.write_alpha_scan_privileged_sidecar(
            self.source,
            prediction,
            self.prepared,
            sidecar,
        )
        prediction_arrays = module.load_alpha_scan_candidates(prediction)
        privileged = module.load_alpha_scan_privileged(sidecar)

        self.assertNotIn("gt_c2w", prediction_arrays)
        self.assertIn("gt_c2w", privileged)
        self.assertTrue(sidecar.is_relative_to(self.root / "privileged_labels"))
        np.testing.assert_allclose(
            privileged["candidate_rms"][:, :, 0],
            np.repeat(privileged["baseline_rms"][:, None], 2, axis=1),
            atol=1e-12,
            rtol=0,
        )

    def test_summary_identifies_majority_small_step_rescues_as_overshoot(self) -> None:
        module = self._module()
        alphas = np.asarray([0.0, 0.1, 1.0], dtype=np.float64)
        relative = np.full((8, 2, 3), -0.2, dtype=np.float64)
        relative[:, :, 0] = 0.0
        relative[:5, 0, 1] = 0.2
        relative[:5, 0, 2] = -0.5

        summary = module.summarize_alpha_statistics(
            alphas,
            relative,
            min_improvement=0.01,
        )

        self.assertEqual(summary["useful_direction_count"], 5)
        self.assertEqual(summary["small_step_rescue_count"], 5)
        self.assertEqual(summary["diagnosis"], "DIRECTION_USEFUL_STEP_TOO_LARGE")


if __name__ == "__main__":
    unittest.main()
