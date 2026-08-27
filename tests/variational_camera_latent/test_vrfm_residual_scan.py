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
    residual_scan = importlib.import_module(
        "pre_experiments.variational_camera_latent.vrfm_residual_scan"
    )
except ModuleNotFoundError:
    residual_scan = None


class VrfmResidualScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "prediction_only" / "source" / "scene0000_00.npz"
        self.candidate = (
            self.root
            / "prediction_only"
            / "calibration_candidates"
            / "scene0000_00.npz"
        )
        global_ids = np.arange(500, dtype=np.int64)
        global_tokens = np.zeros((500, 2048), dtype=np.float32)
        short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
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
        sample_ids = np.asarray(
            [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
            dtype="U64",
        )
        span_starts = np.arange(0, 400, 50, dtype=np.int64)
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
                "span_starts": span_starts,
                "sample_ids": sample_ids,
                "global_pred_c2w": global_c2w,
                "overlap_long_c2w": overlap_c2w,
            },
        )
        corrected = np.zeros((8, 3, 50, 2048), dtype=np.float32)
        corrected[:, 0, :, 0] = 1.0
        corrected[:, 1, :, 0] = 2.0
        corrected[:, 2, :, 0] = 3.0
        self.candidate.parent.mkdir(parents=True)
        np.savez_compressed(
            self.candidate,
            z=np.zeros((8, 3, 3), dtype=np.float32),
            corrected_camera_tokens=corrected,
            latent_cluster_ids=np.tile(np.asarray([0, 1, 0]), (8, 1)),
            latent_cluster_centers=np.zeros((8, 2, 50, 2048), dtype=np.float32),
            source_long_tokens=np.zeros((8, 50, 2048), dtype=np.float32),
            source_sample_ids=sample_ids,
            span_starts=span_starts,
            sample_seeds=np.arange(24, dtype=np.int64).reshape(8, 3),
            checkpoint_sha256=np.asarray("b" * 64, dtype="U64"),
        )
        self.prepared = self.root / "prepared" / "scene0000_00"
        self.prepared.mkdir(parents=True)
        np.save(self.prepared / "frame_ids.npy", global_ids, allow_pickle=False)
        np.save(self.prepared / "raw_gt_c2w.npy", global_c2w, allow_pickle=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _module(self):
        if residual_scan is None:
            self.fail("VRFM residual alpha scan is not implemented")
        return residual_scan

    @staticmethod
    def _camera_head():
        class SequenceSensitiveHead:
            def decode_pose_tokens(self, tokens, *, num_iterations):
                raw = torch.zeros((*tokens.shape[:2], 9), device=tokens.device)
                sequence_mean = tokens[..., 0].mean(dim=1, keepdim=True)
                raw[..., 0] = tokens[..., 0] + sequence_mean
                raw[..., 3] = 1.0
                return [raw] * num_iterations

        return SequenceSensitiveHead()

    def test_prediction_scan_scales_each_vrfm_residual_in_full_context(self) -> None:
        module = self._module()
        destination = (
            self.root
            / "prediction_only"
            / "vrfm_residual_alpha_scan_full_context"
            / "scene0000_00.npz"
        )

        module.generate_vrfm_residual_alpha_scan(
            self.source,
            self.candidate,
            destination,
            camera_head=self._camera_head(),
            camera_head_checkpoint_sha256="a" * 64,
            producer_git_commit="c" * 40,
            device="cpu",
            alphas=(0.0, 0.5, 1.0),
            batch_size=2,
        )
        arrays = module.load_vrfm_residual_alpha_scan(destination)

        self.assertEqual(int(arrays["decode_context_frames"]), 500)
        self.assertEqual(arrays["decoded_camera_raw"].shape, (8, 3, 3, 50, 9))
        self.assertEqual(arrays["decoded_camera_c2w"].shape, (8, 3, 3, 50, 4, 4))
        np.testing.assert_allclose(arrays["decoded_camera_raw"][:, 0, 1, :, 0], 0.55)
        np.testing.assert_allclose(arrays["decoded_camera_raw"][:, 1, 1, :, 0], 1.10)
        np.testing.assert_allclose(arrays["decoded_camera_raw"][:, 2, 1, :, 0], 1.65)
        np.testing.assert_array_equal(
            arrays["decoded_camera_c2w"][:, :, 0],
            np.repeat(
                np.load(self.source, allow_pickle=False)["overlap_long_c2w"][:, None],
                3,
                axis=1,
            ),
        )
        np.testing.assert_array_equal(arrays["sample_seeds"], np.arange(24).reshape(8, 3))
        self.assertEqual(str(arrays["vrfm_checkpoint_sha256"]), "b" * 64)
        self.assertEqual(str(arrays["camera_head_checkpoint_sha256"]), "a" * 64)
        self.assertEqual(str(arrays["producer_git_commit"]), "c" * 40)
        self.assertFalse(
            any("gt" in name.lower() or "privileged" in name.lower() for name in arrays)
        )

    def test_prediction_api_cannot_receive_privileged_data(self) -> None:
        module = self._module()
        parameters = inspect.signature(
            module.generate_vrfm_residual_alpha_scan
        ).parameters

        self.assertNotIn("gt", parameters)
        self.assertNotIn("prepared", parameters)
        self.assertNotIn("privileged", parameters)

    def test_privileged_sidecar_is_separate_and_keeps_no_op_baseline(self) -> None:
        module = self._module()
        prediction = (
            self.root
            / "prediction_only"
            / "vrfm_residual_alpha_scan_full_context"
            / "scene0000_00.npz"
        )
        sidecar = (
            self.root
            / "privileged_labels"
            / "vrfm_residual_alpha_scan_full_context"
            / "scene0000_00.npz"
        )
        module.generate_vrfm_residual_alpha_scan(
            self.source,
            self.candidate,
            prediction,
            camera_head=self._camera_head(),
            camera_head_checkpoint_sha256="a" * 64,
            producer_git_commit="c" * 40,
            device="cpu",
            alphas=(0.0, 0.5, 1.0),
            batch_size=2,
        )

        module.write_vrfm_residual_privileged_sidecar(
            self.source,
            prediction,
            self.prepared,
            sidecar,
        )
        prediction_arrays = module.load_vrfm_residual_alpha_scan(prediction)
        privileged = module.load_vrfm_residual_privileged(sidecar)

        self.assertNotIn("gt_c2w", prediction_arrays)
        self.assertIn("gt_c2w", privileged)
        self.assertTrue(sidecar.is_relative_to(self.root / "privileged_labels"))
        np.testing.assert_allclose(
            privileged["candidate_rms"][:, :, 0],
            np.repeat(privileged["baseline_rms"][:, None], 3, axis=1),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_array_equal(privileged["best_alpha_index"], 0)
        np.testing.assert_array_equal(privileged["best_sample_index"], -1)
        np.testing.assert_array_equal(privileged["best_sample_seed"], -1)
        np.testing.assert_array_equal(privileged["best_latent_cluster_id"], -1)
        np.testing.assert_array_equal(privileged["accept_correction"], False)
        self.assertTrue(np.all(privileged["best_nonzero_sample_index"] >= 0))
        self.assertTrue(np.all(privileged["best_nonzero_alpha_index"] > 0))
        self.assertEqual(len(str(privileged["prepared_gt_sha256"])), 64)

    def test_summary_counts_vrfm_directions_and_small_step_rescues(self) -> None:
        module = self._module()
        alphas = np.asarray([0.0, 0.1, 1.0], dtype=np.float64)
        relative = np.full((8, 3, 3), -0.2, dtype=np.float64)
        relative[:, :, 0] = 0.0
        relative[:5, 0, 1] = 0.2
        relative[:5, 0, 2] = -0.5

        summary = module.summarize_vrfm_residual_statistics(
            alphas,
            relative,
            min_improvement=0.01,
        )

        self.assertEqual(summary["useful_overlap_count"], 5)
        self.assertEqual(summary["useful_direction_count"], 5)
        self.assertEqual(summary["small_step_rescue_overlap_count"], 5)
        self.assertEqual(summary["no_op_best_overlap_count"], 3)
        self.assertEqual(
            summary["diagnosis"],
            "VRFM_ORACLE_CANDIDATE_SET_CONTAINS_USEFUL_SMALL_STEP",
        )


if __name__ == "__main__":
    unittest.main()
