import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from scripts.autodl.camera_hidden_state_attribution.export_replacement import (
    ALLOWED_FILES,
    export_replacement,
)


ROOT = Path(__file__).resolve().parents[2]


class HiddenReplacementAutoDLTest(unittest.TestCase):
    def test_git_allows_only_numeric_replacement_exports(self):
        def is_ignored(path: str) -> bool:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "check-ignore",
                    "--no-index",
                    "--quiet",
                    path,
                ],
                cwd=ROOT,
                check=False,
            )
            return result.returncode == 0

        self.assertFalse(
            is_ignored("results/camera_hidden_replacement/run/per_scene.csv")
        )
        self.assertFalse(
            is_ignored("results/camera_hidden_replacement/run/summary.json")
        )
        self.assertTrue(
            is_ignored(
                "results/camera_hidden_replacement/run/scene/replacement_diagnostics.npz"
            )
        )

    def test_entrypoint_requires_frozen_sources_and_ordered_stages(self):
        text = (
            ROOT / "scripts" / "autodl" / "run_camera_hidden_replacement.sh"
        ).read_text(encoding="utf-8")
        for name in (
            "SOURCE_RUN_DIR",
            "CALIBRATION_LOCAL_RUN_DIR",
            "HOLDOUT_LOCAL_RUN_DIR",
            "ATTRIBUTION_CALIBRATION_DIR",
            "CAUSAL_CALIBRATION_DIR",
            "SPLIT_MANIFEST",
            "CKPT_DIR",
        ):
            self.assertIn(name, text)
        self.assertLess(text.index("run_smoke"), text.index("run_calibration"))
        self.assertLess(text.index("run_calibration"), text.index("run_holdout"))
        self.assertLess(text.index("run_holdout"), text.index("run_export"))
        self.assertIn('[[ -d "$SOURCE_RUN_DIR" ]]', text)
        self.assertIn('[[ -d "$CALIBRATION_LOCAL_RUN_DIR" ]]', text)
        self.assertIn('[[ -d "$HOLDOUT_LOCAL_RUN_DIR" ]]', text)
        self.assertIn('cd "$ROOT"', text)

    def test_export_copies_only_complete_holdout_numeric_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "replacement_run"
            run.mkdir()
            frozen = {
                "schema_version": 1,
                "method": "short_to_long_pose_hidden_replacement",
            }
            frozen["frozen_digest"] = canonical_digest(frozen)
            payloads = {
                "run_metadata.json": {
                    "run_id": run.name,
                    "study_name": "camera_hidden_replacement",
                    "partition": "holdout",
                    "protocol_complete": True,
                },
                "complete.json": {
                    "run_id": run.name,
                    "partition": "holdout",
                    "protocol_complete": True,
                    "analysis_complete": True,
                    "frozen_digest": frozen["frozen_digest"],
                },
                "frozen_replacement.json": frozen,
                "summary.json": {"partition": "holdout"},
            }
            for name, payload in payloads.items():
                (run / name).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            (run / "per_scene.csv").write_text(
                "scene,condition\na,selected\n",
                encoding="utf-8",
            )
            (run / "per_frame.csv").write_text(
                "scene,condition,frame_id\na,selected,0\n",
                encoding="utf-8",
            )

            destination = export_replacement(run, root / "published")

            self.assertEqual(
                {path.name for path in destination.iterdir()},
                set(ALLOWED_FILES),
            )
            complete = payloads["complete.json"]
            complete["frozen_digest"] = "tampered"
            (run / "complete.json").write_text(
                json.dumps(complete),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                export_replacement(run, root / "published_tampered")
            complete["frozen_digest"] = frozen["frozen_digest"]
            (run / "complete.json").write_text(
                json.dumps(complete),
                encoding="utf-8",
            )
            (run / "raw_hidden.npz").write_bytes(b"raw")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                export_replacement(run, root / "published_raw")


if __name__ == "__main__":
    unittest.main()
