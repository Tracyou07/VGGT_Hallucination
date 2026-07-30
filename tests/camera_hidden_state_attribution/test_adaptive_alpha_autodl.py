import json
from pathlib import Path
import tempfile
import unittest

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from scripts.autodl.camera_hidden_state_attribution.export_adaptive_alpha import (
    CALIBRATION_FILES,
    export_adaptive_alpha,
)


ROOT = Path(__file__).resolve().parents[2]


class AdaptiveAlphaAutoDLTest(unittest.TestCase):
    def test_entrypoint_orders_calibration_holdout_and_export(self):
        text = (
            ROOT
            / "scripts"
            / "autodl"
            / "run_camera_hidden_adaptive_alpha.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(
            text.index("run_calibration"),
            text.index("run_holdout"),
        )
        self.assertLess(text.index("run_holdout"), text.index("run_export"))
        for name in (
            "CALIBRATION_SCORE_RUN_DIR",
            "HOLDOUT_SCORE_RUN_DIR",
            "REPLACEMENT_CALIBRATION_DIR",
            "FIXED_REPLACEMENT_HOLDOUT_DIR",
            "SOURCE_RUN_DIR",
            "HOLDOUT_LOCAL_RUN_DIR",
        ):
            self.assertIn(name, text)

    def test_export_authenticates_calibration_selector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            selector = {
                "schema_version": 1,
                "method": "scene_prediction_only_ridge_alpha_selector",
            }
            selector["selector_digest"] = canonical_digest(selector)
            payloads = {
                "run_metadata.json": {
                    "run_id": run.name,
                    "study_name": "camera_hidden_adaptive_alpha",
                    "partition": "calibration",
                },
                "complete.json": {
                    "run_id": run.name,
                    "partition": "calibration",
                    "protocol_complete": True,
                    "analysis_complete": True,
                    "selector_digest": selector["selector_digest"],
                },
                "frozen_selector.json": selector,
                "summary.json": {"scene_count": 10},
            }
            for name, payload in payloads.items():
                (run / name).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            for name in (
                "scene_features.csv",
                "loocv_per_scene.csv",
                "oracle_alpha_per_scene.csv",
            ):
                (run / name).write_text("scene,value\na,1\n", encoding="utf-8")

            destination = export_adaptive_alpha(run, root / "published")

            self.assertEqual(
                {path.name for path in destination.iterdir()},
                set(CALIBRATION_FILES),
            )


if __name__ == "__main__":
    unittest.main()
