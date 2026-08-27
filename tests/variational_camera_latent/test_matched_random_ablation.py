from __future__ import annotations

import importlib
import inspect
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from pre_experiments.variational_camera_latent.source import save_source_shard
from pre_experiments.variational_camera_latent.vrfm_residual_scan import (
    generate_vrfm_residual_alpha_scan,
    load_vrfm_residual_privileged,
    write_vrfm_residual_privileged_sidecar,
)


try:
    matched_random = importlib.import_module(
        "pre_experiments.variational_camera_latent.matched_random_ablation"
    )
except ModuleNotFoundError:
    matched_random = None


class MatchedRandomAblationTests(unittest.TestCase):
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
        self.vrfm_prediction = (
            self.root
            / "prediction_only"
            / "vrfm_residual_alpha_scan_full_context"
            / "scene0000_00.npz"
        )

        global_ids = np.arange(500, dtype=np.int64)
        global_tokens = np.zeros((500, 2048), dtype=np.float32)
        short_tokens = np.zeros((9, 100, 2048), dtype=np.float32)
        short_ids = np.stack(
            [global_ids[start : start + 100] for start in range(0, 401, 50)]
        )
        sample_ids = np.asarray(
            [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
            dtype="U64",
        )
        span_starts = np.arange(0, 400, 50, dtype=np.int64)
        global_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 500, axis=0)
        global_c2w[:, 0, 3] = np.linspace(-1.0, 1.0, 500)
        global_c2w[:, 1, 3] = np.sin(np.linspace(0.0, 4.0 * np.pi, 500))
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
                "span_starts": span_starts,
                "sample_ids": sample_ids,
                "global_pred_c2w": global_c2w,
                "overlap_long_c2w": overlap_c2w,
            },
        )

        corrected = np.zeros((8, 3, 50, 2048), dtype=np.float32)
        corrected[:, 0, :, 0] = 1.0
        corrected[:, 1, :, 0] = 2.0
        # Sample two is deliberately the zero-norm edge case.
        self.candidate.parent.mkdir(parents=True)
        np.savez_compressed(
            self.candidate,
            z=np.arange(72, dtype=np.float32).reshape(8, 3, 3),
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
        if matched_random is None:
            self.fail("matched equal-norm random ablation is not implemented")
        return matched_random

    @staticmethod
    def _camera_head():
        class FullSequenceLengthHead:
            def decode_pose_tokens(self, tokens, *, num_iterations):
                raw = torch.zeros((*tokens.shape[:2], 9), device=tokens.device)
                raw[..., 0] = float(tokens.shape[1]) + tokens[..., 0]
                raw[..., 1] = torch.arange(
                    tokens.shape[1], device=tokens.device, dtype=tokens.dtype
                )[None]
                raw[..., 3] = 1.0
                return [raw] * num_iterations

        return FullSequenceLengthHead()

    def _write_vrfm_prediction(self) -> None:
        generate_vrfm_residual_alpha_scan(
            self.source,
            self.candidate,
            self.vrfm_prediction,
            camera_head=self._camera_head(),
            camera_head_checkpoint_sha256="a" * 64,
            producer_git_commit="c" * 40,
            device="cpu",
            alphas=(0.0, 0.5, 1.0),
            batch_size=2,
        )

    def _literal_residuals(self) -> np.ndarray:
        with np.load(self.candidate, allow_pickle=False) as archive:
            return (
                archive["corrected_camera_tokens"]
                - archive["source_long_tokens"][:, None]
            ).astype(np.float32)

    def test_direction_generation_is_deterministic_from_stable_pair_identity(self) -> None:
        module = self._module()
        residuals = self._literal_residuals()
        sample_ids = np.asarray(
            [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
            dtype="U64",
        )
        sample_seeds = np.arange(24, dtype=np.int64).reshape(8, 3)

        first = module.make_matched_random_directions(
            residuals,
            source_sample_ids=sample_ids,
            sample_seeds=sample_seeds,
            candidate_sha256="d" * 64,
            base_seed=20260827,
        )
        second = module.make_matched_random_directions(
            residuals,
            source_sample_ids=sample_ids,
            sample_seeds=sample_seeds,
            candidate_sha256="d" * 64,
            base_seed=20260827,
        )
        changed_seed = module.make_matched_random_directions(
            residuals,
            source_sample_ids=sample_ids,
            sample_seeds=sample_seeds,
            candidate_sha256="d" * 64,
            base_seed=20260828,
        )
        shared_transform_a = module.make_matched_random_directions(
            residuals,
            source_sample_ids=sample_ids,
            sample_seeds=sample_seeds,
            candidate_sha256="d" * 64,
            transform_identity_sha256="f" * 64,
            base_seed=20260827,
        )
        shared_transform_b = module.make_matched_random_directions(
            residuals,
            source_sample_ids=sample_ids,
            sample_seeds=sample_seeds,
            candidate_sha256="e" * 64,
            transform_identity_sha256="f" * 64,
            base_seed=20260827,
        )

        np.testing.assert_array_equal(
            first["random_direction_seeds"], second["random_direction_seeds"]
        )
        np.testing.assert_array_equal(
            first["random_direction_sha256"], second["random_direction_sha256"]
        )
        np.testing.assert_array_equal(
            first["random_residuals"], second["random_residuals"]
        )
        self.assertFalse(
            np.array_equal(
                first["random_direction_sha256"][:2, :2],
                changed_seed["random_direction_sha256"][:2, :2],
            )
        )
        np.testing.assert_array_equal(
            shared_transform_a["random_transform_sha256"],
            shared_transform_b["random_transform_sha256"],
        )
        np.testing.assert_array_equal(
            shared_transform_a["random_residuals"],
            shared_transform_b["random_residuals"],
        )

    def test_random_directions_match_each_vrfm_norm_and_preserve_zero_norm(self) -> None:
        module = self._module()
        residuals = self._literal_residuals()
        result = module.make_matched_random_directions(
            residuals,
            source_sample_ids=np.asarray(
                [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
                dtype="U64",
            ),
            sample_seeds=np.arange(24, dtype=np.int64).reshape(8, 3),
            candidate_sha256="d" * 64,
            base_seed=20260827,
        )

        vrfm_norms = np.linalg.norm(residuals.astype(np.float64), axis=(2, 3))
        random_norms = np.linalg.norm(
            result["random_residuals"].astype(np.float64), axis=(2, 3)
        )
        np.testing.assert_allclose(random_norms, vrfm_norms, rtol=1e-6, atol=1e-10)
        np.testing.assert_array_equal(result["random_residuals"][:, 2], 0.0)
        np.testing.assert_array_equal(result["random_residual_rms"][:, 2], 0.0)
        self.assertTrue(np.isfinite(result["cosine_to_vrfm"]).all())

    def test_shared_orthogonal_control_preserves_time_and_sample_geometry(self) -> None:
        """Catches independently randomizing samples or destroying temporal structure."""
        module = self._module()
        residuals = self._literal_residuals()
        result = module.make_matched_random_directions(
            residuals,
            source_sample_ids=np.asarray(
                [f"scene0000_00:overlap_{index:03d}" for index in range(8)],
                dtype="U64",
            ),
            sample_seeds=np.arange(24, dtype=np.int64).reshape(8, 3),
            candidate_sha256="d" * 64,
            base_seed=20260827,
        )

        original_rows = residuals[0].reshape(-1, 2048).astype(np.float64)
        random_rows = result["random_residuals"][0].reshape(-1, 2048).astype(
            np.float64
        )
        np.testing.assert_allclose(
            np.linalg.norm(random_rows, axis=1),
            np.linalg.norm(original_rows, axis=1),
            rtol=1e-6,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            random_rows @ random_rows.T,
            original_rows @ original_rows.T,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            result["random_direction_seeds"][0],
            np.repeat(result["random_direction_seeds"][0, 0], 3),
        )
        self.assertFalse(
            np.array_equal(result["random_residuals"][0, :2], residuals[0, :2])
        )

    def test_prediction_uses_same_three_samples_alpha_grid_and_full_g500_context(self) -> None:
        module = self._module()
        self._write_vrfm_prediction()
        destination = (
            self.root
            / "prediction_only"
            / "matched_random_ablation_full_context"
            / "scene0000_00.npz"
        )

        module.generate_matched_random_ablation(
            self.source,
            self.candidate,
            self.vrfm_prediction,
            destination,
            camera_head=self._camera_head(),
            camera_head_checkpoint_sha256="a" * 64,
            producer_git_commit="e" * 40,
            base_seed=20260827,
            device="cpu",
            batch_size=2,
        )
        arrays = module.load_matched_random_ablation(destination)

        np.testing.assert_array_equal(arrays["alphas"], [0.0, 0.5, 1.0])
        self.assertEqual(int(arrays["decode_context_frames"]), 500)
        self.assertEqual(arrays["decoded_camera_raw"].shape, (8, 3, 3, 50, 9))
        self.assertEqual(arrays["decoded_camera_c2w"].shape, (8, 3, 3, 50, 4, 4))
        np.testing.assert_array_equal(arrays["sample_seeds"], np.arange(24).reshape(8, 3))
        np.testing.assert_array_equal(arrays["decoded_camera_raw"][:, :, 0, :, 0], 500.0)
        expected_frame_indices = np.stack(
            [np.arange(start + 50, start + 100) for start in range(0, 400, 50)]
        )
        np.testing.assert_array_equal(
            arrays["decoded_camera_raw"][:, :, 0, :, 1],
            np.repeat(expected_frame_indices[:, None], 3, axis=1),
        )
        self.assertTrue(
            np.any(np.abs(arrays["decoded_camera_raw"][:, :2, -1, :, 0] - 500.0) > 1e-6)
        )
        np.testing.assert_array_equal(arrays["decoded_camera_raw"][:, 2, -1, :, 0], 500.0)
        with np.load(self.source, allow_pickle=False) as source:
            np.testing.assert_array_equal(
                arrays["decoded_camera_c2w"][:, :, 0],
                np.repeat(source["overlap_long_c2w"][:, None], 3, axis=1),
            )
        np.testing.assert_allclose(
            arrays["random_residual_rms"], arrays["vrfm_residual_rms"],
            rtol=1e-6, atol=1e-10,
        )
        self.assertFalse(
            any(
                forbidden in name.lower()
                for name in arrays
                for forbidden in ("gt", "privileged", "error", "quality", "depth")
            )
        )

    def test_prediction_api_cannot_receive_privileged_inputs(self) -> None:
        module = self._module()
        parameters = inspect.signature(module.generate_matched_random_ablation).parameters

        self.assertFalse(
            any(
                forbidden in name.lower()
                for name in parameters
                for forbidden in ("gt", "prepared", "privileged", "error", "quality")
            )
        )

    def test_privileged_sidecar_compares_random_and_vrfm_with_same_oracle(self) -> None:
        module = self._module()
        self._write_vrfm_prediction()
        random_prediction = (
            self.root
            / "prediction_only"
            / "matched_random_ablation_full_context"
            / "scene0000_00.npz"
        )
        vrfm_privileged = (
            self.root
            / "privileged_labels"
            / "vrfm_residual_alpha_scan_full_context"
            / "scene0000_00.npz"
        )
        paired_privileged = (
            self.root
            / "privileged_labels"
            / "matched_random_vs_vrfm"
            / "scene0000_00.npz"
        )
        module.generate_matched_random_ablation(
            self.source,
            self.candidate,
            self.vrfm_prediction,
            random_prediction,
            camera_head=self._camera_head(),
            camera_head_checkpoint_sha256="a" * 64,
            producer_git_commit="e" * 40,
            base_seed=20260827,
            device="cpu",
            batch_size=2,
        )
        write_vrfm_residual_privileged_sidecar(
            self.source,
            self.vrfm_prediction,
            self.prepared,
            vrfm_privileged,
        )

        module.write_matched_random_privileged_sidecar(
            self.source,
            random_prediction,
            self.vrfm_prediction,
            vrfm_privileged,
            self.prepared,
            paired_privileged,
        )
        prediction = module.load_matched_random_ablation(random_prediction)
        sidecar = module.load_matched_random_privileged(paired_privileged)
        upstream = load_vrfm_residual_privileged(vrfm_privileged)

        self.assertNotIn("gt_c2w", prediction)
        self.assertIn("gt_c2w", sidecar)
        np.testing.assert_array_equal(sidecar["oracle_digest"], upstream["oracle_digest"])
        np.testing.assert_allclose(
            sidecar["vrfm_candidate_rms"], upstream["candidate_rms"], atol=0, rtol=0
        )
        np.testing.assert_allclose(
            sidecar["random_candidate_rms"][:, :, 0],
            np.repeat(sidecar["baseline_rms"][:, None], 3, axis=1),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_array_equal(sidecar["vrfm_relative_improvement"][:, :, 0], 0.0)
        np.testing.assert_array_equal(sidecar["random_relative_improvement"][:, :, 0], 0.0)
        np.testing.assert_allclose(
            sidecar["paired_relative_advantage"],
            sidecar["vrfm_relative_improvement"]
            - sidecar["random_relative_improvement"],
            atol=1e-12,
            rtol=0,
        )

    def test_paired_summary_compares_the_same_sample_and_alpha_cells(self) -> None:
        module = self._module()
        alphas = np.asarray([0.0, 0.1, 1.0], dtype=np.float64)
        vrfm = np.zeros((8, 3, 3), dtype=np.float64)
        random = np.zeros((8, 3, 3), dtype=np.float64)
        vrfm[:6, :, 1:] = 0.2
        random[:6, :, 1:] = 0.1
        vrfm[6:, :, 1:] = 0.05
        random[6:, :, 1:] = 0.15

        summary = module.summarize_matched_random_statistics(
            alphas,
            vrfm,
            random,
            min_improvement=0.01,
        )

        self.assertEqual(summary["paired_nonzero_cell_count"], 48)
        self.assertEqual(summary["vrfm_paired_win_count"], 36)
        self.assertEqual(summary["random_paired_win_count"], 12)
        self.assertEqual(summary["paired_tie_count"], 0)
        self.assertAlmostEqual(summary["median_paired_relative_advantage"], 0.1)
        self.assertEqual(summary["vrfm_oracle_best_win_overlap_count"], 6)
        self.assertEqual(summary["random_oracle_best_win_overlap_count"], 2)
        self.assertEqual(summary["oracle_best_tie_overlap_count"], 0)
        self.assertEqual(
            summary["diagnosis"],
            "VRFM_DIRECTIONS_OUTPERFORM_MATCHED_RANDOM",
        )

    def test_report_keeps_scene_as_the_inference_unit_and_scopes_the_claim(self) -> None:
        module = self._module()
        self._write_vrfm_prediction()
        random_prediction = (
            self.root
            / "prediction_only"
            / "matched_random_ablation_full_context"
            / "scene0000_00.npz"
        )
        vrfm_privileged = (
            self.root
            / "privileged_labels"
            / "vrfm_residual_alpha_scan_full_context"
            / "scene0000_00.npz"
        )
        paired_privileged = (
            self.root
            / "privileged_labels"
            / "matched_random_vs_vrfm"
            / "scene0000_00.npz"
        )
        report_path = self.root / "reports" / "matched_random_pilot.json"
        module.generate_matched_random_ablation(
            self.source,
            self.candidate,
            self.vrfm_prediction,
            random_prediction,
            camera_head=self._camera_head(),
            camera_head_checkpoint_sha256="a" * 64,
            producer_git_commit="e" * 40,
            base_seed=20260827,
            device="cpu",
            batch_size=2,
        )
        write_vrfm_residual_privileged_sidecar(
            self.source,
            self.vrfm_prediction,
            self.prepared,
            vrfm_privileged,
        )
        module.write_matched_random_privileged_sidecar(
            self.source,
            random_prediction,
            self.vrfm_prediction,
            vrfm_privileged,
            self.prepared,
            paired_privileged,
        )

        report = module.write_matched_random_report(
            [paired_privileged],
            report_path,
            min_improvement=0.01,
        )

        self.assertEqual(
            report["schema"],
            "variational_camera_latent.matched_random_vs_vrfm_pilot_report.v1",
        )
        self.assertEqual(report["scene_count"], 1)
        self.assertEqual(report["inference_unit"], "scene")
        self.assertEqual(report["structured_null_replicate_count"], 1)
        self.assertIs(report["formal_training_attribution"], False)
        self.assertIs(report["oracle_upper_bound"], True)
        self.assertEqual(report["matched_oracle_budget"]["directions_per_overlap"], 3)
        self.assertEqual(report["matched_oracle_budget"]["nonzero_alpha_count"], 2)
        self.assertEqual(report["matched_oracle_budget"]["no_op_count_per_overlap"], 1)
        self.assertEqual(len(report["per_scene"]), 1)
        self.assertEqual(report["per_scene"][0]["scene"], "scene0000_00")
        self.assertIs(report["formal_scene_level_p_value_available"], False)
        self.assertNotIn("scene_level_exact_sign_flip_two_sided_p", report)
        self.assertEqual(report["diagnosis_basis"], "scene_level")
        self.assertEqual(
            report["scene_level_vrfm_win_count"]
            + report["scene_level_random_win_count"]
            + report["scene_level_tie_count"],
            1,
        )
        self.assertIn(
            report["diagnosis"],
            {
                "VRFM_DIRECTIONS_OUTPERFORM_MATCHED_RANDOM",
                "MATCHED_RANDOM_OUTPERFORMS_VRFM_DIRECTIONS",
                "VRFM_AND_MATCHED_RANDOM_TIED",
            },
        )
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8")), report
        )
        original_bytes = report_path.read_bytes()
        with self.assertRaisesRegex(
            ValueError, "existing matched-random report differs"
        ):
            module.write_matched_random_report(
                [paired_privileged],
                report_path,
                min_improvement=0.02,
            )
        self.assertEqual(report_path.read_bytes(), original_bytes)

    def test_oracle_best_comparison_includes_the_shared_noop(self) -> None:
        module = self._module()
        alphas = np.asarray([0.0, 0.1, 1.0], dtype=np.float64)
        vrfm = np.zeros((8, 3, 3), dtype=np.float64)
        random = np.zeros((8, 3, 3), dtype=np.float64)
        vrfm[:, :, 1:] = -0.1
        random[:, :, 1:] = -1.0

        summary = module.summarize_matched_random_statistics(
            alphas,
            vrfm,
            random,
            min_improvement=0.01,
        )

        self.assertEqual(summary["vrfm_oracle_best_win_overlap_count"], 0)
        self.assertEqual(summary["random_oracle_best_win_overlap_count"], 0)
        self.assertEqual(summary["oracle_best_tie_overlap_count"], 8)
        self.assertEqual(summary["median_oracle_best_relative_advantage"], 0.0)
        self.assertEqual(summary["diagnosis"], "VRFM_AND_MATCHED_RANDOM_TIED")


if __name__ == "__main__":
    unittest.main()
