from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import torch

from pre_experiments.long_short_camera_head.pipeline import (
    DEFAULT_RESULT_ROOT,
    FORMAL_CALIBRATION_STEPS,
    FORMAL_CHECKPOINT_INTERVAL,
    FORMAL_LEARNING_RATE,
    FORMAL_PATIENCE,
    FORMAL_SMOKE_STEPS,
    REQUIRED_TEST_SUITES,
    formal_protocol,
    run_calibration,
    run_smoke,
    verify_completed_run,
)


class PipelineContractTests(unittest.TestCase):
    def test_default_result_root_is_under_vggt(self) -> None:
        self.assertEqual(
            DEFAULT_RESULT_ROOT,
            Path("/data/yjh/output/vggt/long_short_camera_head"),
        )

    def test_completion_verification_fails_closed_on_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_completed_run(Path(directory))

    def test_formal_protocol_is_locked_to_design(self) -> None:
        self.assertEqual(FORMAL_SMOKE_STEPS, 20)
        self.assertEqual(FORMAL_CALIBRATION_STEPS, 400)
        self.assertEqual(FORMAL_LEARNING_RATE, 2e-6)
        self.assertEqual(FORMAL_CHECKPOINT_INTERVAL, 25)
        self.assertEqual(FORMAL_PATIENCE, 100)
        protocol = formal_protocol()
        self.assertEqual(protocol["precision"], "bf16_autocast")
        self.assertEqual(protocol["optimizer"], "AdamW")
        self.assertEqual(protocol["batch_size"], 1)
        self.assertEqual(protocol["weight_decay"], 1e-4)
        self.assertEqual(protocol["gradient_clip_norm"], 1.0)
        self.assertEqual(
            REQUIRED_TEST_SUITES,
            (
                ("long_short_camera_head", "tests/long_short_camera_head"),
                ("variational_camera_latent", "tests/variational_camera_latent"),
                ("variational_camera_selector", "tests/variational_camera_selector"),
            ),
        )

    def test_formal_stages_reject_provisional_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "exactly 20"):
                run_smoke(
                    run_root=root,
                    checkpoint_dir=root,
                    device=torch.device("cpu"),
                    max_steps=4,
                    learning_rate=FORMAL_LEARNING_RATE,
                )
            with self.assertRaisesRegex(ValueError, "exactly 400"):
                run_calibration(
                    run_root=root,
                    checkpoint_dir=root,
                    variant="gt_only",
                    device=torch.device("cpu"),
                    max_steps=40,
                    learning_rate=FORMAL_LEARNING_RATE,
                    checkpoint_interval=FORMAL_CHECKPOINT_INTERVAL,
                    patience=FORMAL_PATIENCE,
                )

    def test_verifier_rejects_missing_formal_config_before_certifying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "manifests"
            target.mkdir(parents=True)
            records = [
                {
                    "scene": f"scene{i:04d}_00",
                    "role": "train" if i < 8 else "validation",
                }
                for i in range(10)
            ]
            (target / "data_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "long_short_camera_head.data_manifest.v1",
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "formal configuration"):
                verify_completed_run(root)


if __name__ == "__main__":
    unittest.main()
