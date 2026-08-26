from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.artifacts import (
    PREDICTION_MEMBERS,
    IncompletePredictionArtifact,
    build_prediction_identity,
    frame_digest,
    load_completed_prediction,
    save_completed_prediction,
)


GIT_COMMIT = "1" * 40
CHECKPOINT_SHA256 = "2" * 64
PROTOCOL_DIGEST = "3" * 64


def _arrays() -> dict[str, np.ndarray]:
    frame_ids = np.asarray([2, 9, 14], dtype=np.int64)
    tokens = np.arange(18, dtype=np.float32).reshape(3, 6)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = [0.0, 1.0, 2.0]
    return {
        "frame_ids": frame_ids,
        "normalized_camera_tokens": tokens,
        "pred_c2w_raw": poses,
    }


def _identity(frame_ids: np.ndarray, *, run_id: str = "cva02-smoke") -> object:
    return build_prediction_identity(
        run_id=run_id,
        scene="scene0000_00",
        artifact_kind="global",
        window_index=None,
        frame_ids=frame_ids,
        checkpoint_sha256=CHECKPOINT_SHA256,
        git_commit=GIT_COMMIT,
        protocol_digest=PROTOCOL_DIGEST,
        preprocess="crop",
        camera_iterations=4,
    )


class PredictionArtifactTest(unittest.TestCase):
    def test_round_trip_is_prediction_only_and_exact(self) -> None:
        arrays = _arrays()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "global.npz"
            completion = root / "global.complete.json"
            save_completed_prediction(artifact, completion, arrays, _identity(arrays["frame_ids"]))

            loaded = load_completed_prediction(
                artifact, completion, _identity(arrays["frame_ids"])
            )
            self.assertEqual(set(loaded), PREDICTION_MEMBERS)
            self.assertNotIn("gt_c2w_raw", loaded)
            for name in PREDICTION_MEMBERS:
                np.testing.assert_array_equal(loaded[name], arrays[name])

            sidecar = json.loads(completion.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["frame_digest"], frame_digest(arrays["frame_ids"]))
            self.assertEqual(sidecar["preprocess"], "crop")
            self.assertEqual(sidecar["camera_iterations"], 4)
            self.assertEqual(sidecar["members"], sorted(PREDICTION_MEMBERS))

    def test_rejects_extra_missing_bad_shape_and_nonfinite_arrays(self) -> None:
        base = _arrays()
        cases = []
        extra = dict(base, gt_c2w_raw=base["pred_c2w_raw"])
        cases.append(extra)
        missing = dict(base)
        missing.pop("normalized_camera_tokens")
        cases.append(missing)
        bad_shape = dict(base, pred_c2w_raw=np.zeros((3, 3, 4)))
        cases.append(bad_shape)
        nonfinite = dict(base, normalized_camera_tokens=base["normalized_camera_tokens"].copy())
        nonfinite["normalized_camera_tokens"][0, 0] = np.nan
        cases.append(nonfinite)

        with tempfile.TemporaryDirectory() as temporary:
            for index, arrays in enumerate(cases):
                with self.subTest(index=index):
                    root = Path(temporary)
                    with self.assertRaises(ValueError):
                        save_completed_prediction(
                            root / f"{index}.npz",
                            root / f"{index}.json",
                            arrays,
                            _identity(base["frame_ids"]),
                        )

    def test_only_sidecar_marks_completion_and_tmp_files_never_count(self) -> None:
        arrays = _arrays()
        identity = _identity(arrays["frame_ids"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "global.npz"
            completion = root / "global.complete.json"
            with artifact.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            (root / "global.complete.json.tmp").write_text("{}", encoding="utf-8")

            with self.assertRaises(IncompletePredictionArtifact):
                load_completed_prediction(artifact, completion, identity)

    def test_resume_rejects_schema_provenance_hash_and_frame_mismatch(self) -> None:
        arrays = _arrays()
        identity = _identity(arrays["frame_ids"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "global.npz"
            completion = root / "global.complete.json"
            save_completed_prediction(artifact, completion, arrays, identity)

            for key, value in (
                ("schema", "wrong.schema"),
                ("run_id", "different-run"),
                ("protocol_digest", "4" * 64),
            ):
                original = json.loads(completion.read_text(encoding="utf-8"))
                changed = dict(original)
                changed[key] = value
                completion.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_completed_prediction(artifact, completion, identity)
                completion.write_text(json.dumps(original), encoding="utf-8")

            with artifact.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                load_completed_prediction(artifact, completion, identity)

        with self.assertRaisesRegex(ValueError, "frame digest"):
            _identity(np.asarray([2, 14, 9], dtype=np.int64))

    def test_identity_rejects_nonfrozen_model_settings_and_kind_contract(self) -> None:
        ids = _arrays()["frame_ids"]
        with self.assertRaisesRegex(ValueError, "preprocess must be crop"):
            build_prediction_identity(
                run_id="r",
                scene="scene0000_00",
                artifact_kind="global",
                window_index=None,
                frame_ids=ids,
                checkpoint_sha256=CHECKPOINT_SHA256,
                git_commit=GIT_COMMIT,
                protocol_digest=PROTOCOL_DIGEST,
                preprocess="resize",
                camera_iterations=4,
            )
        with self.assertRaisesRegex(ValueError, "camera_iterations must be 4"):
            build_prediction_identity(
                run_id="r",
                scene="scene0000_00",
                artifact_kind="local",
                window_index=0,
                frame_ids=ids,
                checkpoint_sha256=CHECKPOINT_SHA256,
                git_commit=GIT_COMMIT,
                protocol_digest=PROTOCOL_DIGEST,
                preprocess="crop",
                camera_iterations=3,
            )
        with self.assertRaisesRegex(ValueError, "global artifacts cannot have a window_index"):
            build_prediction_identity(
                run_id="r",
                scene="scene0000_00",
                artifact_kind="global",
                window_index=0,
                frame_ids=ids,
                checkpoint_sha256=CHECKPOINT_SHA256,
                git_commit=GIT_COMMIT,
                protocol_digest=PROTOCOL_DIGEST,
                preprocess="crop",
                camera_iterations=4,
            )


if __name__ == "__main__":
    unittest.main()
