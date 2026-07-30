import json
from pathlib import Path
import tempfile
import unittest

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from scripts.autodl.camera_hidden_state_attribution.export_causal_preference import (
    ALLOWED_FILES,
    export_causal_preference,
)


class CausalAutoDLTest(unittest.TestCase):
    def test_export_copies_only_complete_authenticated_numeric_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "causal_run"
            run.mkdir()
            frozen = {
                "schema_version": 1,
                "method": "camera_hidden_causal_preference",
            }
            frozen_digest = canonical_digest(frozen)
            frozen["frozen_digest"] = frozen_digest
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "causal_run",
                        "study_name": "camera_hidden_causal_preference",
                        "partition": "holdout",
                        "protocol_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            (run / "complete.json").write_text(
                json.dumps(
                    {
                        "run_id": "causal_run",
                        "partition": "holdout",
                        "protocol_complete": True,
                        "analysis_complete": True,
                        "frozen_digest": frozen_digest,
                    }
                ),
                encoding="utf-8",
            )
            (run / "frozen_causal_normalization.json").write_text(
                json.dumps(frozen),
                encoding="utf-8",
            )
            (run / "summary.json").write_text(
                json.dumps({"partition": "holdout"}),
                encoding="utf-8",
            )
            (run / "per_position.csv").write_text(
                "iteration,unit\n0,0\n",
                encoding="utf-8",
            )
            (run / "direct_checks.csv").write_text(
                "scene,iteration,unit\ns,0,0\n",
                encoding="utf-8",
            )

            destination = export_causal_preference(
                run,
                root / "published",
            )
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                set(ALLOWED_FILES),
            )

            (run / "raw_basis.npz").write_bytes(b"raw")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                export_causal_preference(run, root / "published_raw")
            (run / "raw_basis.npz").unlink()

            complete = json.loads(
                (run / "complete.json").read_text(encoding="utf-8")
            )
            complete["protocol_complete"] = False
            (run / "complete.json").write_text(
                json.dumps(complete),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete formal"):
                export_causal_preference(
                    run,
                    root / "published_incomplete",
                )


if __name__ == "__main__":
    unittest.main()
