"""Fail-closed ScanNet integrity gate that runs before any GPU loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Iterable, TypeVar

from pre_experiments.camera_velocity_ambiguity_02.contracts import ProtocolViolation


MARKER_SCHEMA = "camera_solution_space_01.scannet50_verified_completion.v1"
OFFICIAL_SCENE_LIST = (
    "https://raw.githubusercontent.com/mystorm16/FastVGGT/"
    "main/eval/scannet_50.yaml"
)
OFFICIAL_ASSET_ROOT = "https://kaldir.vc.in.tum.de/scannet"
EXPECTED_INTEGRITY_STATEMENT = (
    "Each final matched refreshed HTTPS Content-Length and local SHA-256 "
    "equaled the independently computed H20 SHA-256. The upstream server "
    "does not publish per-file cryptographic checksums."
)
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_MARKER_FIELDS = {
    "schema",
    "verified_at",
    "official_scene_list",
    "official_scene_list_sha256",
    "scene_count",
    "asset_count",
    "total_bytes",
    "remote_root",
    "integrity_statement",
    "assets",
}
EXPECTED_ASSET_FIELDS = {
    "key",
    "scene",
    "kind",
    "url",
    "relative_path",
    "bytes",
    "sha256",
}
DEFAULT_MAX_AGE = timedelta(days=7)


@dataclass(frozen=True)
class VerifiedAsset:
    """One local/H20-equal asset proven by the terminal verifier."""

    key: str
    scene: str
    kind: str
    url: str
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class VerifiedInputs:
    """Immutable identity consumed by later CPU and GPU stages."""

    marker_path: str
    marker_sha256: str
    verified_at: str
    official_scene_list_sha256: str
    remote_root: str
    scene_count: int
    asset_count: int
    total_bytes: int
    assets: tuple[VerifiedAsset, ...]


def canonical_scene_list_digest(scenes: Iterable[str]) -> str:
    """Hash the official ordered scene list with canonical LF endings."""
    values = tuple(scenes)
    if len(values) != 50 or len(set(values)) != 50:
        raise ProtocolViolation("scene list must contain exactly 50 unique scenes")
    if not all(isinstance(scene, str) and scene for scene in values):
        raise ProtocolViolation("scene IDs must be non-empty strings")
    try:
        payload = "".join(f"{scene}\n" for scene in values).encode("ascii")
    except UnicodeEncodeError as error:
        raise ProtocolViolation("scene IDs must be ASCII") from error
    return hashlib.sha256(payload).hexdigest()


def _reject_non_finite(value: object, label: str = "marker") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolViolation(f"{label} contains a non-finite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{label}[{index}]")


def _read_marker(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolViolation(f"invalid verified completion marker: {path}") from error
    if not isinstance(payload, dict) or set(payload) != EXPECTED_MARKER_FIELDS:
        raise ProtocolViolation("verified completion marker fields are not exact")
    _reject_non_finite(payload)
    return raw, payload


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ProtocolViolation("verified_at must be a timezone-aware timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError as error:
        raise ProtocolViolation("verified_at is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolViolation("verified_at must contain a timezone")
    return parsed


def _expected_assets(scenes: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    expected: list[tuple[str, ...]] = []
    for scene in scenes:
        expected.extend(
            (
                (
                    f"{scene}:sens",
                    scene,
                    "sens",
                    f"{OFFICIAL_ASSET_ROOT}/v1/scans/{scene}/{scene}.sens",
                    f"raw_sens/scans/{scene}/{scene}.sens",
                ),
                (
                    f"{scene}:ply",
                    scene,
                    "ply",
                    f"{OFFICIAL_ASSET_ROOT}/v2/scans/{scene}/{scene}_vh_clean_2.ply",
                    f"raw/scans/{scene}/{scene}_vh_clean_2.ply",
                ),
            )
        )
    return tuple(expected)


def load_verified_inputs(
    path: Path,
    *,
    expected_remote_root: str,
    expected_scene_list_sha256: str,
    expected_scenes: Iterable[str],
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> VerifiedInputs:
    """Authenticate the terminal ScanNet marker without touching GPU code."""
    marker_path = Path(path)
    raw, payload = _read_marker(marker_path)
    scenes = tuple(expected_scenes)
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise ProtocolViolation("expected scene list must contain 50 unique scenes")
    if payload["schema"] != MARKER_SCHEMA:
        raise ProtocolViolation("verified completion schema mismatch")
    if payload["official_scene_list"] != OFFICIAL_SCENE_LIST:
        raise ProtocolViolation("official scene-list URL mismatch")
    if payload["official_scene_list_sha256"] != expected_scene_list_sha256:
        raise ProtocolViolation("official scene-list digest mismatch")
    if payload["remote_root"] != expected_remote_root:
        raise ProtocolViolation("verified remote root mismatch")
    if payload["integrity_statement"] != EXPECTED_INTEGRITY_STATEMENT:
        raise ProtocolViolation("integrity statement mismatch")
    if payload["scene_count"] != 50 or payload["asset_count"] != 100:
        raise ProtocolViolation("verified completion must contain 50 scenes and 100 assets")

    verified_at = _parse_time(payload["verified_at"])
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ProtocolViolation("gate time must contain a timezone")
    if max_age <= timedelta(0):
        raise ProtocolViolation("max_age must be positive")
    age = current.astimezone(timezone.utc) - verified_at.astimezone(timezone.utc)
    if age < -timedelta(minutes=5):
        raise ProtocolViolation("verified completion timestamp is in the future")
    if age > max_age:
        raise ProtocolViolation("verified completion marker is stale")

    raw_assets = payload["assets"]
    expected_assets = _expected_assets(scenes)
    if not isinstance(raw_assets, list) or len(raw_assets) != len(expected_assets):
        raise ProtocolViolation("verified completion assets are incomplete")

    assets: list[VerifiedAsset] = []
    for index, (raw_asset, identity) in enumerate(zip(raw_assets, expected_assets)):
        if not isinstance(raw_asset, dict) or set(raw_asset) != EXPECTED_ASSET_FIELDS:
            raise ProtocolViolation(f"asset fields are not exact at index {index}")
        key, scene, kind, url, relative_path = identity
        if tuple(raw_asset[name] for name in ("key", "scene", "kind", "url", "relative_path")) != identity:
            raise ProtocolViolation(f"asset identity mismatch at index {index}")
        byte_count = raw_asset["bytes"]
        digest = raw_asset["sha256"]
        if type(byte_count) is not int or byte_count <= 0:
            raise ProtocolViolation(f"asset size is invalid: {key}")
        if not isinstance(digest, str) or HASH_PATTERN.fullmatch(digest) is None:
            raise ProtocolViolation(f"asset SHA-256 is invalid: {key}")
        assets.append(
            VerifiedAsset(
                key=key,
                scene=scene,
                kind=kind,
                url=url,
                relative_path=relative_path,
                bytes=byte_count,
                sha256=digest,
            )
        )

    total_bytes = sum(asset.bytes for asset in assets)
    if type(payload["total_bytes"]) is not int or payload["total_bytes"] != total_bytes:
        raise ProtocolViolation("verified total byte count mismatch")
    return VerifiedInputs(
        marker_path=str(marker_path.resolve()),
        marker_sha256=hashlib.sha256(raw).hexdigest(),
        verified_at=str(payload["verified_at"]),
        official_scene_list_sha256=expected_scene_list_sha256,
        remote_root=expected_remote_root,
        scene_count=50,
        asset_count=100,
        total_bytes=total_bytes,
        assets=tuple(assets),
    )


T = TypeVar("T")


def gate_gpu_launch(
    launch: Callable[[VerifiedInputs], T],
    marker_path: Path,
    **gate_kwargs: object,
) -> T:
    """Call a GPU/resource loader only after the marker authenticates."""
    verified = load_verified_inputs(marker_path, **gate_kwargs)
    return launch(verified)
