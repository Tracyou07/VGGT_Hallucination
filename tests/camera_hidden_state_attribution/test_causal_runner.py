from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    load_causal_scene_effects,
    save_causal_scene_effects,
)
from pre_experiments.camera_hidden_state_attribution.run_causal_preference import (
    _measurement_config,
    _select_stage_scenes,
    measure_scene_causal_effects,
    parse_args,
)
from pre_experiments.camera_hidden_state_attribution.run_study import replay_tokens
from vggt.heads.camera_head import CameraHead


class CausalRunnerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.head = CameraHead(
            dim_in=16,
            trunk_depth=1,
            num_heads=4,
            mlp_ratio=2,
        ).eval()
        self.tokens = np.random.default_rng(19).normal(
            size=(3, 16)
        ).astype(np.float32)

    def test_measurement_returns_finite_iteration_unit_effects(self):
        effects = measure_scene_causal_effects(
            self.head,
            self.tokens,
            torch.device("cpu"),
            num_iterations=2,
            basis_step=1e-3,
            basis_batch_size=2,
            basis_dimension_limit=2,
            direct_checks_per_iteration=1,
            direct_relative_step=1e-2,
        )
        for name in (
            "activation_scale",
            "translation_effect",
            "rotation_effect_deg",
            "fov_effect",
        ):
            self.assertEqual(effects[name].shape, (2, 8))
            self.assertTrue(np.isfinite(effects[name]).all())
        self.assertEqual(effects["measured_basis_mask"].shape, (2, 9))
        self.assertEqual(int(effects["measured_basis_mask"].sum()), 4)
        np.testing.assert_array_equal(effects["direct_iteration"], [0, 1])
        self.assertEqual(effects["direct_unit"].shape, (2,))
        for name in (
            "direct_projected_translation",
            "direct_measured_translation",
            "direct_projected_rotation_deg",
            "direct_measured_rotation_deg",
            "direct_projected_fov",
            "direct_measured_fov",
        ):
            self.assertEqual(effects[name].shape, (2,))
            self.assertTrue(np.isfinite(effects[name]).all())

    def test_scene_artifact_round_trip_and_member_validation(self):
        effects = measure_scene_causal_effects(
            self.head,
            self.tokens,
            torch.device("cpu"),
            num_iterations=2,
            basis_step=1e-3,
            basis_batch_size=2,
            basis_dimension_limit=2,
            direct_checks_per_iteration=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "causal_unit_effects.npz"
            save_causal_scene_effects(path, effects)
            loaded = load_causal_scene_effects(path, "scene0000_00")
            self.assertEqual(loaded["scene"], "scene0000_00")
            for name, expected in effects.items():
                np.testing.assert_array_equal(loaded[name], expected)

            with np.load(path, allow_pickle=False) as archive:
                corrupted = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                    if name != "fov_effect"
                }
            np.savez(path, **corrupted)
            with self.assertRaisesRegex(ValueError, "members"):
                load_causal_scene_effects(path, "scene0000_00")

    def test_measurement_rejects_invalid_controls(self):
        with self.assertRaisesRegex(ValueError, "basis_step"):
            measure_scene_causal_effects(
                self.head,
                self.tokens,
                torch.device("cpu"),
                num_iterations=2,
                basis_step=0.0,
            )
        with self.assertRaisesRegex(ValueError, "basis_dimension_limit"):
            measure_scene_causal_effects(
                self.head,
                self.tokens,
                torch.device("cpu"),
                num_iterations=2,
                basis_dimension_limit=10,
            )

    def test_measurement_rejects_checkpoint_replay_mismatch(self):
        expected = replay_tokens(
            self.head,
            self.tokens,
            torch.device("cpu"),
        )["pred_c2w_raw"]
        expected[0, 0, 3] += 1.0
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            measure_scene_causal_effects(
                self.head,
                self.tokens,
                torch.device("cpu"),
                num_iterations=4,
                basis_dimension_limit=1,
                direct_checks_per_iteration=0,
                expected_pred_c2w_raw=expected,
                replay_tolerance=1e-4,
            )

    def test_cli_requires_frozen_holdout_and_has_numeric_defaults(self):
        common = [
            "--source-run-dir",
            "source",
            "--split-manifest",
            "split.json",
            "--ckpt-dir",
            "checkpoint",
        ]
        args = parse_args(["--stage", "calibration", *common])
        self.assertEqual(args.basis_step, 1e-3)
        self.assertEqual(args.basis_batch_size, 2)
        self.assertEqual(args.direct_checks_per_iteration, 1)
        self.assertEqual(args.num_iterations, 4)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--stage", "holdout", *common])

    def test_stage_scene_selection_and_measurement_provenance(self):
        split = {
            "calibration_scenes": ["c0", "c1"],
            "holdout_scenes": ["h0", "h1", "h2"],
        }
        self.assertEqual(
            _select_stage_scenes("smoke", split, scene_limit=0),
            ("calibration", ["c0"]),
        )
        self.assertEqual(
            _select_stage_scenes("calibration", split, scene_limit=1),
            ("calibration", ["c0"]),
        )
        self.assertEqual(
            _select_stage_scenes("holdout", split, scene_limit=2),
            ("holdout", ["h0", "h1"]),
        )

        args = parse_args(
            [
                "--stage",
                "calibration",
                "--source-run-dir",
                "source",
                "--split-manifest",
                "split.json",
                "--ckpt-dir",
                "checkpoint",
                "--basis-dimension-limit",
                "2",
            ]
        )
        self.assertEqual(
            _measurement_config(args, target_dim=9),
            {
                "method": "centered_pose_delta_jacobian_projection",
                "num_iterations": 4,
                "target_dim": 9,
                "measured_basis_dimensions": 2,
                "basis_step": 0.001,
                "activation_scale": "per_scene_unit_rms_with_5pct_median_floor",
            },
        )


if __name__ == "__main__":
    unittest.main()
