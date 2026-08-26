from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from pre_experiments.camera_velocity_ambiguity_02.contracts import (
    EvidenceSource,
    ProtocolViolation,
    canonical_json_digest,
)
from pre_experiments.camera_velocity_ambiguity_02.protocol import load_protocol_v2


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scannet50_camera_velocity_ambiguity_02_split_v2.json"
PARENT_SPLIT = ROOT / "configs" / "scannet50_local_global_split.json"
SCENE_LIST = ROOT / "configs" / "fastvggt_scannet50.txt"
PARENT_DIGEST = "69c283245c4f220965e6fde3b96192de298e292eb8ca625c94851fe8932cdb8a"


def _load() -> object:
    return load_protocol_v2(
        CONFIG,
        parent_split_path=PARENT_SPLIT,
        scene_list_path=SCENE_LIST,
    )


class ProtocolV2Test(unittest.TestCase):
    def test_loads_frozen_identity_membership_and_evidence_sources(self) -> None:
        protocol = _load()
        parent = json.loads(PARENT_SPLIT.read_text(encoding="utf-8"))
        scenes = tuple(
            line.strip()
            for line in SCENE_LIST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

        self.assertEqual(protocol.name, "camera_velocity_ambiguity_02")
        self.assertEqual(protocol.schema_version, 2)
        self.assertEqual(protocol.parent_split_digest, PARENT_DIGEST)
        self.assertEqual(protocol.scene_order, scenes)
        self.assertEqual(protocol.calibration_scenes, tuple(parent["calibration_scenes"]))
        self.assertEqual(
            protocol.development_scenes,
            tuple(parent["holdout_scenes"]),
        )
        self.assertEqual(protocol.development_name, "development_evaluation")
        self.assertEqual(protocol.alphas, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(
            {source.value for source in EvidenceSource},
            {
                "PREDICTION_ONLY",
                "PRIVILEGED_GT",
                "OBSERVATION_RGBD",
                "PRESENTATION_ONLY",
            },
        )

    def test_derives_exact_frame_window_and_pair_counts(self) -> None:
        protocol = _load()

        self.assertEqual(protocol.frame_count("scene0000_00"), 500)
        self.assertEqual(protocol.frame_count("scene0150_00"), 430)
        with self.assertRaises(ProtocolViolation):
            protocol.frame_count("scene9999_99")

        self.assertEqual(protocol.window_length, 100)
        self.assertEqual(protocol.window_stride, 50)
        self.assertEqual(protocol.counts.scenes, 50)
        self.assertEqual(protocol.counts.global_runs, 50)
        self.assertEqual(protocol.counts.local_windows, 449)
        self.assertEqual(protocol.counts.adjacent_pairs, 399)
        self.assertEqual(protocol.counts.primary_pairs, 398)
        self.assertEqual(protocol.counts.secondary_pairs, 1)
        self.assertEqual(protocol.counts.calibration_primary_pairs, 80)
        self.assertEqual(protocol.counts.development_primary_pairs, 318)

    def test_rejects_changed_authenticated_config(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["windowing"]["length"] = 99

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolViolation, "config digest"):
                load_protocol_v2(
                    path,
                    parent_split_path=PARENT_SPLIT,
                    scene_list_path=SCENE_LIST,
                )

    def test_rejects_tampering_even_when_digest_is_recomputed(self) -> None:
        original = json.loads(CONFIG.read_text(encoding="utf-8"))

        mutations = {
            "parent split": lambda value: value["parent_split"].__setitem__(
                "digest", "0" * 64
            ),
            "scene order": lambda value: value["scene_order"].__setitem__(
                0, "scene9999_99"
            ),
            "calibration membership": lambda value: value["cohorts"][
                "calibration"
            ].__setitem__(0, value["cohorts"]["development_evaluation"][0]),
            "frame selection": lambda value: value["frame_selection"].__setitem__(
                "sampling", "round_stride"
            ),
            "declared count": lambda value: value["expected_counts"].__setitem__(
                "primary_pairs", 397
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    payload = copy.deepcopy(original)
                    mutate(payload)
                    unsigned = {
                        key: value
                        for key, value in payload.items()
                        if key != "config_digest"
                    }
                    payload["config_digest"] = canonical_json_digest(unsigned)
                    path = Path(directory) / f"{name.replace(' ', '_')}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ProtocolViolation):
                        load_protocol_v2(
                            path,
                            parent_split_path=PARENT_SPLIT,
                            scene_list_path=SCENE_LIST,
                        )


if __name__ == "__main__":
    unittest.main()
