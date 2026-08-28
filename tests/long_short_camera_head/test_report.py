from __future__ import annotations

import unittest

from pre_experiments.long_short_camera_head.report import classify


def _row(scene: str, *, baseline: float, gt_only: float, long_short: float) -> dict[str, object]:
    return {
        "scene": scene,
        "baseline_rms": baseline,
        "gt_only_rms": gt_only,
        "long_short_rms": long_short,
        "baseline_rotation_deg": 1.0,
        "long_short_rotation_deg": 1.05,
    }


class ReportClassificationTests(unittest.TestCase):
    def test_promising_requires_long_short_to_beat_baseline_and_gt_only(self) -> None:
        report = classify(
            [_row("scene0325_01", baseline=1.0, gt_only=0.97, long_short=0.94),
             _row("scene0675_00", baseline=1.0, gt_only=0.98, long_short=0.96)],
            inference_leakage_audit=True,
        )
        self.assertEqual(report["classification"], "PROMISING")

    def test_report_fails_promising_when_one_scene_worsens_over_one_percent(self) -> None:
        report = classify(
            [_row("scene0325_01", baseline=1.0, gt_only=0.99, long_short=0.95),
             _row("scene0675_00", baseline=1.0, gt_only=1.00, long_short=1.02)],
            inference_leakage_audit=True,
        )
        self.assertNotEqual(report["classification"], "PROMISING")
        self.assertIn("per_scene_harm", report["failed_gates"])

    def test_nonfinite_or_leaky_result_is_invalid(self) -> None:
        rows = [_row("scene0325_01", baseline=1.0, gt_only=0.9, long_short=float("nan"))]
        self.assertEqual(
            classify(rows, inference_leakage_audit=True)["classification"], "INVALID"
        )
        rows[0]["long_short_rms"] = 0.8
        self.assertEqual(
            classify(rows, inference_leakage_audit=False)["classification"], "INVALID"
        )


if __name__ == "__main__":
    unittest.main()
