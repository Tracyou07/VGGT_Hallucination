import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import unittest

from pre_experiments.camera_solution_space_01.contracts import (
    ContractError,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_hex,
    validate_schema,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_DIRECTORY = REPOSITORY_ROOT / "configs" / "camera_solution_space_01"
EXPECTED_CALIBRATION = [
    "scene0395_00", "scene0466_01", "scene0593_00", "scene0084_01",
    "scene0631_01", "scene0606_01", "scene0619_00", "scene0071_00",
    "scene0056_00", "scene0571_00", "scene0177_01", "scene0409_01",
]
EXPECTED_SCENES = (
    "scene0000_00", "scene0013_02", "scene0029_01", "scene0042_02", "scene0056_00",
    "scene0071_00", "scene0084_01", "scene0096_00", "scene0109_00", "scene0121_01",
    "scene0136_01", "scene0150_00", "scene0164_01", "scene0177_01", "scene0194_00",
    "scene0207_01", "scene0221_01", "scene0238_00", "scene0254_01", "scene0267_00",
    "scene0280_00", "scene0294_02", "scene0309_00", "scene0325_01", "scene0340_01",
    "scene0353_02", "scene0367_01", "scene0380_02", "scene0395_00", "scene0409_01",
    "scene0421_02", "scene0435_03", "scene0451_01", "scene0466_01", "scene0477_00",
    "scene0493_01", "scene0509_01", "scene0525_00", "scene0540_02", "scene0555_00",
    "scene0571_00", "scene0582_02", "scene0593_00", "scene0606_01", "scene0619_00",
    "scene0631_01", "scene0648_00", "scene0663_01", "scene0675_00", "scene0691_00",
)


class CanonicalJsonTests(unittest.TestCase):
    def test_canonical_json_is_compact_sorted_and_hashed(self):
        value = {"z": [3, {"b": False, "a": "é"}], "a": 1}
        expected = b'{"a":1,"z":[3,{"a":"\\u00e9","b":false}]}'
        self.assertEqual(canonical_json_bytes(value), expected)
        self.assertEqual(canonical_json_sha256(value), hashlib.sha256(expected).hexdigest())
        self.assertEqual(sha256_hex(expected), hashlib.sha256(expected).hexdigest())

    def test_rejects_non_finite_values_non_string_keys_and_unsupported_objects(self):
        values = ({"value": float("nan")}, {"value": float("inf")}, {1: "bad"}, {"bad": {1, 2}})
        for value in values:
            with self.subTest(value=repr(value)):
                with self.assertRaises(ContractError):
                    canonical_json_bytes(value)

    def test_rejects_non_dict_mappings_with_contract_error(self):
        with self.assertRaisesRegex(ContractError, "native dict"):
            canonical_json_bytes(MappingProxyType({"value": 1}))

    def test_schema_validation_requires_exact_schema_string(self):
        document = {"schema": "camera_solution_space/example/v1", "value": 1}
        validate_schema(document, "camera_solution_space/example/v1")
        with self.assertRaisesRegex(ContractError, "schema"):
            validate_schema(document, "camera_solution_space/example/v2")
        with self.assertRaisesRegex(ContractError, "schema"):
            validate_schema({"value": 1}, "camera_solution_space/example/v1")

    def test_hashes_are_lowercase_64_character_hex(self):
        self.assertRegex(sha256_hex(b"strict-contract"), r"^[0-9a-f]{64}$")


class ScanNetSplitTests(unittest.TestCase):
    def test_official_scene_list_and_frozen_split_reconstruct_exactly(self):
        scene_lines = (CONFIGURATION_DIRECTORY / "fastvggt_scannet50.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(tuple(scene_lines), EXPECTED_SCENES)

        split = json.loads((CONFIGURATION_DIRECTORY / "scannet50_split_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(split["schema"], "camera_solution_space_scannet50_split/v1")
        self.assertEqual(split["namespace"], "camera_solution_space_01:v1:")
        ordered = sorted(
            scene_lines,
            key=lambda scene_id: (hashlib.sha256((split["namespace"] + scene_id).encode("utf-8")).hexdigest(), scene_id),
        )
        self.assertEqual(split["calibration"], ordered[:12])
        self.assertEqual(split["evaluation"], ordered[12:])
        self.assertEqual(split["calibration"], EXPECTED_CALIBRATION)
        self.assertEqual(len(split["calibration"]), 12)
        self.assertEqual(len(split["evaluation"]), 38)


if __name__ == "__main__":
    unittest.main()
