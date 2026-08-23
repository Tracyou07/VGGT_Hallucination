import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib

import numpy as np
from PIL import Image

from pre_experiments.camera_solution_space_01.observation import (
    ObservationError,
    plan_observation,
    seal_observation,
    validate_observation,
)
from pre_experiments.camera_solution_space_01.contracts import sha256_file
from pre_experiments.camera_solution_space_01.sens_index import index_sens


def _matrix(value=1.0):
    return [value if index % 5 == 0 else 0.0 for index in range(16)]


def _jpeg(width=2, height=1):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (9, 20, 31)).save(
        stream, format="JPEG", quality=100, subsampling=0
    )
    return stream.getvalue()


def _sens_bytes(frame_count=120, jpeg_size=(2, 1)):
    color = _jpeg(*jpeg_size)
    depth = zlib.compress(np.array([[11, 12]], dtype="<u2").tobytes())
    payload = bytearray(struct.pack("<IQ", 4, 6) + b"sensor")
    for _ in range(4):
        payload.extend(struct.pack("<16f", *_matrix()))
    payload.extend(struct.pack("<iiIIIIfQ", 2, 1, 2, 1, 2, 1, 1000.0, frame_count))
    for frame_id in range(frame_count):
        payload.extend(struct.pack("<16f", *_matrix(float(frame_id + 1))))
        payload.extend(struct.pack("<QQQQ", 1000 + frame_id, 2000 + frame_id, len(color), len(depth)))
        payload.extend(color)
        payload.extend(depth)
    return bytes(payload)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.source = self.root / "scene.sens"
        self.source.write_bytes(_sens_bytes())
        self.index = index_sens(self.source)
        self.outputs = self.root / "observations"

    def _plan(self, eligibility):
        return plan_observation(
            self.source, self.index, "scene0000_00", "calibration", eligibility
        )

    def test_deterministically_chooses_lowest_explicit_eligible_window(self):
        plan = self._plan({2: True, 0: False, 1: False})
        self.assertEqual(plan["frame_ids"], [2, 17, 32, 47, 62, 77, 92, 107])
        self.assertEqual(plan["selection"]["chosen_start"], 2)
        self.assertEqual(plan["selection_version"], "fixed8_stride15_v1")
        self.assertEqual(plan["plan_id"], self._plan({2: True})["plan_id"])

    def test_sealing_consumes_planned_frames_without_resampling(self):
        plan = self._plan(lambda start, frame_ids: start == 3 and frame_ids[0] == 3)
        sealed = seal_observation(plan, self.source, self.outputs)
        manifest = json.loads((sealed / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [item["frame_id"] for item in manifest["ordered_model_input"]],
            [3, 18, 33, 48, 63, 78, 93, 108],
        )
        self.assertTrue((sealed / "rgb/000000.npy").is_file())
        self.assertFalse(any("match" in path.name for path in sealed.rglob("*")))

    def test_reads_only_selected_payload_ranges_and_preserves_source(self):
        plan = self._plan({4: True})
        selected = {4, 19, 34, 49, 64, 79, 94, 109}
        source_before = self.source.read_bytes()
        sealed = seal_observation(plan, self.source, self.outputs)
        audit = json.loads((sealed / "read_audit.json").read_text(encoding="utf-8"))
        expected_offsets = {
            offset
            for frame_id in selected
            for offset in (
                self.index.frames[frame_id].color_data_offset,
                self.index.frames[frame_id].depth_data_offset,
            )
        }
        self.assertEqual({entry["offset"] for entry in audit}, expected_offsets)
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_uint16_depth_round_trip_and_jpeg_dimensions_are_checked(self):
        plan = self._plan({0: True})
        sealed = seal_observation(plan, self.source, self.outputs)
        depth = np.load(sealed / "depth/000000.npy", allow_pickle=False)
        self.assertEqual(depth.dtype, np.dtype("uint16"))
        self.assertEqual(depth.tolist(), [[11, 12]])
        wrong = self.root / "wrong-size.sens"
        wrong.write_bytes(_sens_bytes(jpeg_size=(3, 1)))
        wrong_plan = plan_observation(wrong, index_sens(wrong), "scene", "evaluation", {0: True})
        with self.assertRaisesRegex(ObservationError, "JPEG dimensions"):
            seal_observation(wrong_plan, wrong, self.outputs)

    def test_invalid_existing_output_is_never_overwritten(self):
        plan = self._plan({0: True})
        target = self.outputs / plan["plan_id"]
        target.mkdir(parents=True)
        marker = target / "marker"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ObservationError, "existing output"):
            seal_observation(plan, self.source, self.outputs)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((target / "complete.json").exists())

    def test_idempotent_validation_and_tamper_extra_path_rejection(self):
        plan = self._plan({0: True})
        sealed = seal_observation(plan, self.source, self.outputs)
        self.assertEqual(validate_observation(sealed, self.source), plan["plan_id"])
        self.assertEqual(seal_observation(plan, self.source, self.outputs), sealed)
        rgb_path = sealed / "rgb/000000.npy"
        rgb_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ObservationError, "(size|hash)"):
            validate_observation(sealed, self.source)
        shutil.rmtree(sealed)
        sealed = seal_observation(plan, self.source, self.outputs)
        (sealed / "extra.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ObservationError, "extra"):
            validate_observation(sealed, self.source)
        (sealed / "extra.txt").unlink()
        manifest_path = sealed / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0]["path"] = "../outside"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        complete_path = sealed / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["manifest_sha256"] = sha256_file(manifest_path)
        complete_path.write_text(json.dumps(complete), encoding="utf-8")
        with self.assertRaisesRegex(ObservationError, "path"):
            validate_observation(sealed, self.source)


    def test_validation_rejects_changed_source_fingerprint(self):
        plan = self._plan({0: True})
        sealed = seal_observation(plan, self.source, self.outputs)
        self.source.write_bytes(self.source.read_bytes() + b"changed")
        with self.assertRaisesRegex(ObservationError, "source fingerprint"):
            validate_observation(sealed, self.source)

    def test_validation_rejects_schema_and_wrong_model_order(self):
        plan = self._plan({0: True})
        sealed = seal_observation(plan, self.source, self.outputs)
        complete_path = sealed / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["schema"] = "wrong"
        complete_path.write_text(json.dumps(complete), encoding="utf-8")
        with self.assertRaisesRegex(ObservationError, "schema"):
            validate_observation(sealed, self.source)
        shutil.rmtree(sealed)
        sealed = seal_observation(plan, self.source, self.outputs)
        manifest_path = sealed / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["ordered_model_input"][0], manifest["ordered_model_input"][1] = (
            manifest["ordered_model_input"][1],
            manifest["ordered_model_input"][0],
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        complete_path = sealed / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["manifest_sha256"] = sha256_file(manifest_path)
        complete_path.write_text(json.dumps(complete), encoding="utf-8")
        with self.assertRaisesRegex(ObservationError, "order"):
            validate_observation(sealed, self.source)


if __name__ == "__main__":
    unittest.main()
