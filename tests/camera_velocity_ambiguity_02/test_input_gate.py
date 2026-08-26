from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pre_experiments.camera_velocity_ambiguity_02.contracts import ProtocolViolation
from pre_experiments.camera_velocity_ambiguity_02.input_gate import (
    EXPECTED_INTEGRITY_STATEMENT,
    canonical_scene_list_digest,
    load_verified_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
SCENE_LIST = ROOT / "configs" / "fastvggt_scannet50.txt"
REMOTE_ROOT = "/data/yjh/share/datasets/ScanNet"
VERIFIED_AT = datetime(2026, 8, 26, 19, 5, tzinfo=timezone(timedelta(hours=8)))


def _scenes() -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in SCENE_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _scene_digest() -> str:
    return canonical_scene_list_digest(_scenes())


def _marker() -> dict[str, object]:
    assets: list[dict[str, object]] = []
    for index, scene in enumerate(_scenes(), start=1):
        assets.extend(
            (
                {
                    "key": f"{scene}:sens",
                    "scene": scene,
                    "kind": "sens",
                    "url": f"https://kaldir.vc.in.tum.de/scannet/v1/scans/{scene}/{scene}.sens",
                    "relative_path": f"raw_sens/scans/{scene}/{scene}.sens",
                    "bytes": index * 1000,
                    "sha256": f"{index:064x}",
                },
                {
                    "key": f"{scene}:ply",
                    "scene": scene,
                    "kind": "ply",
                    "url": f"https://kaldir.vc.in.tum.de/scannet/v2/scans/{scene}/{scene}_vh_clean_2.ply",
                    "relative_path": f"raw/scans/{scene}/{scene}_vh_clean_2.ply",
                    "bytes": index * 10,
                    "sha256": f"{index + 100:064x}",
                },
            )
        )
    return {
        "schema": "camera_solution_space_01.scannet50_verified_completion.v1",
        "verified_at": VERIFIED_AT.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "official_scene_list": (
            "https://raw.githubusercontent.com/mystorm16/FastVGGT/"
            "main/eval/scannet_50.yaml"
        ),
        "official_scene_list_sha256": _scene_digest(),
        "scene_count": 50,
        "asset_count": 100,
        "total_bytes": sum(int(asset["bytes"]) for asset in assets),
        "remote_root": REMOTE_ROOT,
        "integrity_statement": EXPECTED_INTEGRITY_STATEMENT,
        "assets": assets,
    }


def _write(path: Path, marker: dict[str, object]) -> None:
    path.write_text(json.dumps(marker, allow_nan=True), encoding="utf-8")


def _load(path: Path, *, now: datetime = VERIFIED_AT) -> object:
    return load_verified_inputs(
        path,
        expected_remote_root=REMOTE_ROOT,
        expected_scene_list_sha256=_scene_digest(),
        expected_scenes=_scenes(),
        now=now,
        max_age=timedelta(days=7),
    )


class InputGateTest(unittest.TestCase):
    def test_scene_digest_is_independent_of_checkout_line_endings(self) -> None:
        canonical = "".join(f"{scene}\n" for scene in _scenes()).encode("ascii")
        self.assertEqual(_scene_digest(), hashlib.sha256(canonical).hexdigest())
        self.assertEqual(
            _scene_digest(),
            "c1728acca7cabf68fcdd4eb63db858e1b1768d5beb77873b7bacd92cc1fe8788",
        )

    def test_loads_an_immutable_exact_100_asset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verified_completion.json"
            marker = _marker()
            _write(path, marker)

            verified = _load(path)

        self.assertEqual(verified.remote_root, REMOTE_ROOT)
        self.assertEqual(verified.scene_count, 50)
        self.assertEqual(verified.asset_count, 100)
        self.assertEqual(verified.total_bytes, marker["total_bytes"])
        self.assertEqual(len(verified.assets), 100)
        self.assertEqual(verified.assets[0].key, "scene0000_00:sens")
        self.assertRegex(verified.marker_sha256, r"^[0-9a-f]{64}$")
        with self.assertRaises(FrozenInstanceError):
            verified.remote_root = "/changed"  # type: ignore[misc]

    def test_rejects_missing_wrong_incomplete_or_stale_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            with self.assertRaises(ProtocolViolation):
                _load(missing)

            mutations = {
                "wrong root": lambda value: value.__setitem__("remote_root", "/wrong"),
                "missing asset": lambda value: value["assets"].pop(),
                "changed digest": lambda value: value.__setitem__(
                    "official_scene_list_sha256", "0" * 64
                ),
                "non finite": lambda value: value.__setitem__("total_bytes", float("nan")),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    marker = _marker()
                    mutate(marker)
                    path = root / f"{name.replace(' ', '_')}.json"
                    _write(path, marker)
                    with self.assertRaises(ProtocolViolation):
                        _load(path)

            stale = root / "stale.json"
            _write(stale, _marker())
            with self.assertRaisesRegex(ProtocolViolation, "stale"):
                _load(stale, now=VERIFIED_AT + timedelta(days=8))

    def test_rejects_asset_identity_size_and_hash_tampering(self) -> None:
        mutations = {
            "duplicate key": lambda asset: asset.__setitem__(
                "key", "scene0000_00:sens"
            ),
            "wrong path": lambda asset: asset.__setitem__("relative_path", "../escape"),
            "zero bytes": lambda asset: asset.__setitem__("bytes", 0),
            "bad hash": lambda asset: asset.__setitem__("sha256", "bad"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    marker = _marker()
                    mutate(marker["assets"][1])
                    if name == "zero bytes":
                        marker["total_bytes"] = sum(
                            int(asset["bytes"]) for asset in marker["assets"]
                        )
                    path = Path(directory) / f"{name.replace(' ', '_')}.json"
                    _write(path, marker)
                    with self.assertRaises(ProtocolViolation):
                        _load(path)


if __name__ == "__main__":
    unittest.main()
