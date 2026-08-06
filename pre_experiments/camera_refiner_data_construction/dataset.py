"""Portable manifests for external multiscale Camera hidden shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from pre_experiments.camera_refiner_data_construction.artifacts import (
    SCHEMA_VERSION as SHARD_SCHEMA_VERSION,
    load_scene_shard,
)
from pre_experiments.common.contracts import atomic_write_json


DATASET_SCHEMA_VERSION = 1
ALLOWED_ROLES = {"train", "validation", "calibration", "holdout"}
PROVENANCE_FIELDS = {
    "git_commit",
    "checkpoint_digest",
    "split_digest",
    "frozen_policy_digest",
}
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_roles(
    scene_roles: Mapping[str, Sequence[str]],
    shard_scenes: set[str],
    protected_holdout_scenes: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    if not scene_roles or not set(scene_roles).issubset(ALLOWED_ROLES):
        raise ValueError(f"scene roles must use {sorted(ALLOWED_ROLES)}")
    normalized = {}
    scene_to_role = {}
    for role, role_scenes in scene_roles.items():
        scenes = sorted(str(scene) for scene in role_scenes)
        if not scenes or any(not scene for scene in scenes):
            raise ValueError(f"scene role {role} must be non-empty")
        if len(set(scenes)) != len(scenes):
            raise ValueError(f"scene role {role} contains duplicates")
        for scene in scenes:
            if scene in scene_to_role:
                raise ValueError(f"scene {scene} appears in multiple roles")
            scene_to_role[scene] = role
        normalized[role] = scenes
    if set(scene_to_role) != shard_scenes:
        raise ValueError("scene roles and shard scenes must match exactly")
    protected = {str(scene) for scene in protected_holdout_scenes}
    leaked = sorted(
        scene for scene, role in scene_to_role.items()
        if role == "train" and scene in protected
    )
    if leaked:
        raise ValueError(f"protected holdout scenes cannot be training data: {leaked}")
    return dict(sorted(normalized.items())), scene_to_role


def _validated_provenance(provenance: Mapping[str, object]) -> dict[str, str]:
    if set(provenance) != PROVENANCE_FIELDS:
        raise ValueError(f"dataset provenance must contain {sorted(PROVENANCE_FIELDS)}")
    normalized = {name: str(provenance[name]) for name in sorted(PROVENANCE_FIELDS)}
    if any(not value for value in normalized.values()) or _COMMIT_PATTERN.fullmatch(
        normalized["git_commit"]
    ) is None:
        raise ValueError("dataset provenance values are invalid")
    return normalized


def build_dataset_manifest(
    dataset_root: Path,
    shard_paths: Mapping[str, Path],
    scene_roles: Mapping[str, Sequence[str]],
    *,
    protected_holdout_scenes: Sequence[str],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Inspect external shards and build a deterministic portable manifest."""
    root = Path(dataset_root).resolve()
    if not shard_paths:
        raise ValueError("at least one dataset shard is required")
    shards = {str(scene): Path(path).resolve() for scene, path in shard_paths.items()}
    if any(not scene for scene in shards):
        raise ValueError("shard scene identities must be non-empty")
    roles, scene_to_role = _validated_roles(
        scene_roles,
        set(shards),
        protected_holdout_scenes,
    )
    normalized_provenance = _validated_provenance(provenance)
    entries = []
    for scene in sorted(shards):
        path = shards[scene]
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"shard is outside dataset root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"dataset shard is missing: {path}")
        shard = load_scene_shard(path, scene)
        entries.append(
            {
                "scene": scene,
                "role": scene_to_role[scene],
                "path": relative.as_posix(),
                "sha256": _sha256(path),
                "frame_count": int(len(shard["frame_ids"])),
                "scales": [int(value) for value in shard["scales"]],
                "shard_schema_version": SHARD_SCHEMA_VERSION,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "provenance": normalized_provenance,
        "scene_roles": roles,
        "protected_holdout_scenes": sorted(
            str(scene) for scene in protected_holdout_scenes
        ),
        "shards": entries,
    }
    manifest["dataset_digest"] = canonical_digest(manifest)
    return manifest


def write_dataset_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Write an authenticated dataset manifest without copying its shards."""
    value = dict(manifest)
    digest = value.pop("dataset_digest", None)
    if not isinstance(digest, str) or digest != canonical_digest(value):
        raise ValueError("dataset manifest digest is invalid")
    value["dataset_digest"] = digest
    atomic_write_json(Path(path), value)


def validate_dataset_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    protected_holdout_scenes: Sequence[str],
) -> dict[str, object]:
    """Validate manifest identity, roles, checksums, and every external shard."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid dataset manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid dataset manifest: {path}")
    value = dict(manifest)
    digest = value.pop("dataset_digest", None)
    if not isinstance(digest, str) or digest != canonical_digest(value):
        raise ValueError("dataset manifest digest is invalid")
    if value.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("dataset manifest schema is unsupported")
    if value.get("protected_holdout_scenes") != sorted(
        str(scene) for scene in protected_holdout_scenes
    ):
        raise ValueError("protected holdout provenance mismatch")
    provenance = value.get("provenance")
    roles = value.get("scene_roles")
    entries = value.get("shards")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(roles, Mapping)
        or not isinstance(entries, Sequence)
        or not entries
    ):
        raise ValueError("dataset manifest structure is invalid")
    _validated_provenance(provenance)
    entry_scenes = {
        str(entry.get("scene", ""))
        for entry in entries
        if isinstance(entry, Mapping)
    }
    normalized_roles, scene_to_role = _validated_roles(
        roles, entry_scenes, protected_holdout_scenes
    )
    root = Path(dataset_root).resolve()
    role_counts = {role: 0 for role in normalized_roles}
    frame_count = 0
    seen = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("dataset shard entry is invalid")
        scene = str(entry.get("scene", ""))
        if not scene or scene in seen:
            raise ValueError("dataset shard entries must have unique scenes")
        seen.add(scene)
        if entry.get("role") != scene_to_role[scene]:
            raise ValueError(f"dataset role mismatch for {scene}")
        relative = Path(str(entry.get("path", "")))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"dataset shard path escapes root: {relative}") from error
        if not candidate.is_file():
            raise FileNotFoundError(f"dataset shard is missing: {candidate}")
        if entry.get("sha256") != _sha256(candidate):
            raise ValueError(f"dataset shard checksum mismatch: {scene}")
        shard = load_scene_shard(candidate, scene)
        actual_frames = int(len(shard["frame_ids"]))
        if (
            entry.get("frame_count") != actual_frames
            or entry.get("scales") != [100, 200, 300]
            or entry.get("shard_schema_version") != SHARD_SCHEMA_VERSION
        ):
            raise ValueError(f"dataset shard metadata mismatch: {scene}")
        role_counts[scene_to_role[scene]] += 1
        frame_count += actual_frames
    return {
        "dataset_digest": digest,
        "scene_count": len(seen),
        "frame_count": frame_count,
        "role_counts": role_counts,
    }
