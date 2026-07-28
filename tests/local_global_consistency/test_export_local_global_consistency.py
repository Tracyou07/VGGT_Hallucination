import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.local_global_consistency.split import build_split_manifest
from pre_experiments.local_global_consistency.thresholds import (
    fit_frozen_thresholds,
)
from scripts.autodl.local_global_consistency.export_numeric_results import (
    CALIBRATION_FILES,
    HOLDOUT_FILES,
    export_numeric_results,
)


class LocalGlobalExportTest(unittest.TestCase):
    def make_sources(self, root: Path):
        scene_list = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "fastvggt_scannet50.txt"
        )
        scenes = [
            line.strip()
            for line in scene_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        trajectories = {}
        for index, scene in enumerate(scenes):
            poses = np.repeat(np.eye(4)[None], 3, axis=0)
            poses[:, 0, 3] = np.arange(3) * (index + 1)
            trajectories[scene] = poses
        split = build_split_manifest(
            scenes,
            trajectories,
            source_run_id="source-run",
        )
        split_path = root / "scannet50_local_global_split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")

        calibration = root / "abc1234_calibration"
        holdout = root / "abc1234_holdout"
        calibration.mkdir()
        holdout.mkdir()
        threshold_rows = [
            {
                "scene": scene,
                "local_local_token_cosine": float(index + 1),
                "local_local_pose_translation": float(index + 1),
                "local_local_pose_rotation_deg": float(index + 1),
            }
            for index, scene in enumerate(split["calibration_scenes"])
        ]
        thresholds = fit_frozen_thresholds(
            threshold_rows,
            {
                "calibration_scenes": split["calibration_scenes"],
                "source_run_id": "source-run",
                "calibration_run_id": calibration.name,
                "split_digest": split["split_digest"],
                "code_commit": "a" * 40,
            },
        )
        calibration_metadata = {
            "study_name": "local_global_consistency",
            "run_id": calibration.name,
            "git_commit": "a" * 40,
            "source_run_id": "source-run",
            "split_digest": split["split_digest"],
            "partition": "calibration",
            "protocol_complete": True,
        }
        holdout_metadata = {
            **calibration_metadata,
            "run_id": holdout.name,
            "partition": "holdout",
        }
        calibration_complete = {
            "run_id": calibration.name,
            "partition": "calibration",
            "scenes": split["calibration_scenes"],
            "analysis_complete": True,
            "split_digest": split["split_digest"],
            "source_run_id": "source-run",
            "threshold_digest": thresholds["threshold_digest"],
        }
        holdout_complete = {
            "run_id": holdout.name,
            "partition": "holdout",
            "scenes": split["holdout_scenes"],
            "analysis_complete": True,
            "split_digest": split["split_digest"],
            "source_run_id": "source-run",
            "threshold_digest": thresholds["threshold_digest"],
            "threshold_path": "/root/calibration/frozen_reliability_thresholds.json",
        }
        for directory, files, metadata, complete in (
            (
                calibration,
                CALIBRATION_FILES,
                calibration_metadata,
                calibration_complete,
            ),
            (holdout, HOLDOUT_FILES, holdout_metadata, holdout_complete),
        ):
            for name in files:
                if name == "run_metadata.json":
                    content = metadata
                elif name in {"complete.json", "holdout_complete.json"}:
                    content = complete
                elif name == "frozen_reliability_thresholds.json":
                    content = thresholds
                elif name == "holdout_aggregate_summary.json":
                    content = {
                        **holdout_complete,
                        "metrics": [],
                    }
                elif name.endswith(".json"):
                    content = []
                else:
                    (directory / name).write_text(
                        "scene,value\nscene0000_00,1\n", encoding="utf-8"
                    )
                    continue
                (directory / name).write_text(
                    json.dumps(content) + "\n", encoding="utf-8"
                )
            raw = directory / "scene0000_00" / "window_000"
            raw.mkdir(parents=True)
            (raw / "window_diagnostics.npz").write_bytes(b"external raw input")
            visualizations = directory / "visualizations"
            visualizations.mkdir()
            (visualizations / "plot.png").write_bytes(b"not published")
        return calibration, holdout, split_path, thresholds

    def test_exports_only_authenticated_numeric_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration, holdout, split_path, _ = self.make_sources(root)

            destination = export_numeric_results(
                calibration,
                holdout,
                split_path,
                root / "published",
            )

            self.assertEqual(
                destination.name,
                f"{calibration.name}__{holdout.name}",
            )
            self.assertEqual(
                {path.name for path in (destination / "calibration").iterdir()},
                set(CALIBRATION_FILES),
            )
            self.assertEqual(
                {path.name for path in (destination / "holdout").iterdir()},
                set(HOLDOUT_FILES),
            )
            self.assertTrue(
                (destination / "scannet50_local_global_split.json").is_file()
            )
            self.assertTrue((destination / "publish_manifest.json").is_file())
            self.assertFalse(any(path.suffix == ".npz" for path in destination.rglob("*")))
            self.assertFalse(any(path.suffix == ".png" for path in destination.rglob("*")))

    def test_rejects_unexpected_root_artifact_or_incomplete_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration, holdout, split_path, _ = self.make_sources(root)
            (holdout / "tokens.npz").write_bytes(b"forbidden")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                export_numeric_results(
                    calibration, holdout, split_path, root / "published"
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration, holdout, split_path, _ = self.make_sources(root)
            complete_path = holdout / "complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["analysis_complete"] = False
            complete_path.write_text(json.dumps(complete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "analysis"):
                export_numeric_results(
                    calibration, holdout, split_path, root / "published"
                )

    def test_rejects_missing_holdout_threshold_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration, holdout, split_path, _ = self.make_sources(root)
            complete_path = holdout / "holdout_complete.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            del complete["threshold_digest"]
            complete_path.write_text(json.dumps(complete), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "threshold"):
                export_numeric_results(
                    calibration, holdout, split_path, root / "published"
                )

    def test_rejects_missing_holdout_scene_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration, holdout, split_path, _ = self.make_sources(root)
            for name in ("complete.json", "holdout_complete.json"):
                path = holdout / name
                payload = json.loads(path.read_text(encoding="utf-8"))
                del payload["scenes"]
                path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "scenes"):
                export_numeric_results(
                    calibration, holdout, split_path, root / "published"
                )


if __name__ == "__main__":
    unittest.main()
