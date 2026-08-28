from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pre_experiments.variational_camera_latent.alpha_scan import DEFAULT_ALPHAS
from pre_experiments.variational_camera_selector.dataset import CandidateGroup
from pre_experiments.variational_camera_selector import evaluate as selector_evaluate
from pre_experiments.variational_camera_selector import safety_gate
from pre_experiments.variational_camera_selector.evaluate import (
    load_score_shard,
    score_scene_candidates,
    summarize_calibration,
)
from pre_experiments.variational_camera_selector.model import CandidateRanker


class _FakePredictionDataset:
    def __init__(self, root: Path, *, frames: int = 3) -> None:
        self.scene = "scene0325_01"
        self.scenes = (self.scene,)
        self.roles = ("validation",)
        self.frames = frames
        self.binding_manifest = root / "candidate_binding_manifest.json"
        self.binding_manifest.write_text('{"fixture":"scores"}\n', encoding="utf-8")

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> CandidateGroup:
        nonzero = np.asarray(DEFAULT_ALPHAS[1:], dtype=np.float32)
        alphas = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.repeat(nonzero, 32))
        )
        seeds = np.concatenate(
            (np.asarray([-1], dtype=np.int64), np.tile(np.arange(32), 7))
        )
        z_raw = np.stack(
            (np.arange(32, dtype=np.float32), -np.arange(32, dtype=np.float32)),
            axis=1,
        )
        z = np.concatenate((np.zeros((1, 2), np.float32), np.tile(z_raw, (7, 1))))
        choice_ids = np.asarray(
            [f"{self.scene}:overlap_{index:03d}:choice_{choice}" for choice in range(225)],
            dtype="U96",
        )
        delta = np.zeros((225, self.frames, 2048), dtype=np.float32)
        delta[:, :, 0] = alphas[:, None]
        return CandidateGroup(
            scene=self.scene,
            role="validation",
            overlap_index=index,
            sample_id=f"{self.scene}:overlap_{index:03d}",
            span_start=index * 50,
            global_tokens=np.zeros((4, 2048), dtype=np.float32),
            x0=np.zeros((self.frames, 2048), dtype=np.float32),
            delta_tokens=delta,
            alphas=alphas,
            z=z,
            sample_seeds=seeds,
            choice_ids=choice_ids,
            source_sha256="a" * 64,
            candidate_sha256="b" * 64,
            residual_prediction_sha256="c" * 64,
        )


class SelectorEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = _FakePredictionDataset(self.root)
        self.checkpoint = self.root / "latest.pt"
        self.output = self.root / "scores.npz"
        self.selection_output = self.root / "selections.npz"
        torch.manual_seed(9)
        model_config = {"d_model": 4, "z_dim": 2, "input_dim": 2048, "span_count": 8}
        full = CandidateRanker(**model_config, include_global_context=True)
        residual = CandidateRanker(**model_config, include_global_context=False)
        torch.save(
            {
                "schema": "variational_camera_selector.training_checkpoint.v1",
                "completed_step": 3,
                "model_config": model_config,
                "full_context_model": full.state_dict(),
                "residual_only_model": residual.state_dict(),
                "config_digest": "d" * 64,
                "input_digest": "e" * 64,
            },
            self.checkpoint,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_score_api_has_no_privileged_parameter_and_output_has_no_labels(self) -> None:
        self.assertFalse(
            any(
                "privileged" in name.lower()
                for name in inspect.signature(score_scene_candidates).parameters
            )
        )
        path = score_scene_candidates(
            self.dataset,
            "scene0325_01",
            self.checkpoint,
            self.output,
            device="cpu",
        )
        arrays = load_score_shard(path)

        self.assertEqual(arrays["full_context_scores"].shape, (8, 225))
        self.assertEqual(arrays["residual_only_scores"].shape, (8, 225))
        self.assertEqual(arrays["choice_ids"].shape, (8, 225))
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in arrays
                for fragment in ("gt", "quality", "error", "depth", "utility")
            )
        )

    def test_selection_shard_materializes_selected_corrected_latents_without_labels(self) -> None:
        score_scene_candidates(
            self.dataset,
            "scene0325_01",
            self.checkpoint,
            self.output,
            device="cpu",
        )
        writer = getattr(selector_evaluate, "write_scene_selections", None)
        loader = getattr(selector_evaluate, "load_selection_shard", None)
        self.assertIsNotNone(writer, "selection materializer is missing")
        self.assertIsNotNone(loader, "selection loader is missing")
        assert writer is not None and loader is not None

        path = writer(
            _FakePredictionDataset(self.root, frames=50),
            "scene0325_01",
            self.output,
            self.selection_output,
        )
        arrays = loader(path)
        scores = load_score_shard(self.output)

        self.assertEqual(arrays["full_context_corrected_camera_tokens"].shape, (8, 50, 2048))
        self.assertEqual(arrays["residual_only_corrected_camera_tokens"].shape, (8, 50, 2048))
        for overlap in range(8):
            selected = int(scores["full_context_selected_indices"][overlap])
            expected = float(scores["alphas"][overlap, selected])
            np.testing.assert_allclose(
                arrays["full_context_corrected_camera_tokens"][overlap, :, 0],
                expected,
            )
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in arrays
                for fragment in ("gt", "quality", "error", "depth", "utility", "privileged")
            )
        )

    def test_gated_selection_fail_closes_without_loading_labels(self) -> None:
        # Catches a gate writer that needs GT at inference or emits the raw unsafe top-1.
        score_scene_candidates(
            self.dataset,
            "scene0325_01",
            self.checkpoint,
            self.output,
            device="cpu",
        )
        writer = getattr(safety_gate, "write_gated_scene_selection", None)
        loader = getattr(safety_gate, "load_gated_selection", None)
        self.assertIsNotNone(writer, "gated selection writer is missing")
        self.assertIsNotNone(loader, "gated selection loader is missing")
        assert writer is not None and loader is not None
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in inspect.signature(writer).parameters
                for fragment in ("gt", "utility", "privileged", "depth", "error")
            )
        )

        destination = self.root / "gated_selection.npz"
        path = writer(
            _FakePredictionDataset(self.root, frames=50),
            "scene0325_01",
            self.output,
            safety_gate.GatePolicy.fail_closed(),
            "f" * 64,
            destination,
        )
        arrays = loader(path)

        np.testing.assert_array_equal(arrays["selected_indices"], np.zeros(8, np.int64))
        self.assertFalse(arrays["gate_pass"].any())
        np.testing.assert_array_equal(
            arrays["corrected_camera_tokens"], np.zeros((8, 50, 2048), np.float32)
        )
        self.assertFalse(
            any(
                fragment in name.lower()
                for name in arrays
                for fragment in ("gt", "quality", "error", "depth", "utility", "privileged")
            )
        )

    def test_privileged_summary_uses_overlap_then_scene_units(self) -> None:
        fixtures = []
        for scene_index, scene in enumerate(("scene0325_01", "scene0675_00")):
            scores = np.arange(scene_index * 8, scene_index * 8 + 8, dtype=np.float64)
            utilities = 0.02 + 0.01 * scores
            fixtures.append(
                {
                    "scene": scene,
                    "full_context_utility": utilities,
                    "full_context_selected_score": scores,
                    "residual_only_utility": utilities / 2,
                    "residual_only_selected_score": -scores,
                    "random_utility": np.zeros(8),
                    "noop_utility": np.zeros(8),
                    "oracle_utility": utilities + 0.05,
                    "full_context_oracle_rank": np.ones(8, dtype=np.int64),
                    "full_context_spearman": np.full(8, 0.25),
                }
            )
        report = summarize_calibration(fixtures, random_seed=20260827)

        self.assertEqual(report["scene_count"], 2)
        self.assertEqual(report["overlap_count"], 16)
        self.assertEqual(report["inference_unit"], "overlap")
        self.assertEqual(report["aggregate_unit"], "scene")
        self.assertEqual(report["classification"], "LEARNABLE_SIGNAL")
        self.assertEqual(report["full_context"]["positive_over_1pct_count"], 16)
        self.assertEqual(len(report["per_scene"]), 2)
        self.assertIn("score_utility_calibration", report["full_context"])
        curve = report["full_context"]["score_utility_calibration"]
        self.assertEqual([row["count"] for row in curve], [4, 4, 4, 4])
        np.testing.assert_allclose(
            [row["observed_utility_mean"] for row in curve],
            [0.035, 0.075, 0.115, 0.155],
        )


if __name__ == "__main__":
    unittest.main()
