"""Build and validate a sequence-disjoint ScanNet adaptation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
STUDY_TYPE = "scannet_camera_refiner_adaptation"
SCENE_PATTERN = re.compile(r"scene[0-9]{4}_[0-9]{2}")
ROLE_ORDER = ("refiner_train", "validation", "selector_train")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_scenes(values: Iterable[str], *, label: str) -> list[str]:
    scenes = [str(value).strip() for value in values]
    if any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes):
        raise ValueError(f"{label} contains an invalid ScanNet scene ID")
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"{label} contains duplicate ScanNet scene IDs")
    return scenes


def _validated_role_counts(role_counts: Mapping[str, int]) -> dict[str, int]:
    if set(role_counts) != set(ROLE_ORDER):
        raise ValueError(f"role counts must contain exactly {list(ROLE_ORDER)}")
    normalized = {role: int(role_counts[role]) for role in ROLE_ORDER}
    if any(count < 1 for count in normalized.values()):
        raise ValueError("every scene role must contain at least one scene")
    return normalized


def _with_digest(value: Mapping[str, object], digest_name: str) -> dict[str, object]:
    result = dict(value)
    result[digest_name] = _canonical_digest(result)
    return result


def _validate_digest(value: Mapping[str, object], digest_name: str) -> None:
    payload = dict(value)
    digest = payload.pop(digest_name, None)
    if not isinstance(digest, str) or digest != _canonical_digest(payload):
        raise ValueError(f"{digest_name} is invalid")


def build_candidate_manifest(
    official_train_scenes: Iterable[str],
    *,
    excluded_scenes: Iterable[str],
    seed: int,
    role_counts: Mapping[str, int],
    min_frames: int,
) -> dict[str, object]:
    """Freeze a deterministic candidate order before any scene is downloaded."""
    train = _validated_scenes(official_train_scenes, label="official split")
    excluded = _validated_scenes(excluded_scenes, label="excluded split")
    counts = _validated_role_counts(role_counts)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    excluded_set = set(excluded)
    candidates = [scene for scene in train if scene not in excluded_set]
    target = sum(counts.values())
    if len(candidates) < target:
        raise ValueError(f"official split has only {len(candidates)} usable candidates")

    def selection_key(scene: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{scene}".encode("ascii")).hexdigest()
        return digest, scene

    candidates.sort(key=selection_key)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "study_type": STUDY_TYPE,
        "selection_seed": seed,
        "min_matching_frames": min_frames,
        "role_counts": counts,
        "target_scene_count": target,
        "official_train_scene_count": len(train),
        "official_train_digest": _canonical_digest(sorted(train)),
        "excluded_scenes": sorted(excluded),
        "excluded_scene_digest": _canonical_digest(sorted(excluded)),
        "candidate_scenes": candidates,
    }
    return _with_digest(payload, "manifest_digest")


def assign_scene_roles(
    accepted_scenes: Sequence[str], role_counts: Mapping[str, int]
) -> dict[str, list[str]]:
    """Assign whole scenes to roles in their frozen acceptance order."""
    scenes = _validated_scenes(accepted_scenes, label="accepted scenes")
    counts = _validated_role_counts(role_counts)
    expected = sum(counts.values())
    if len(scenes) != expected:
        raise ValueError(f"accepted scene list must contain exactly {expected} scenes")
    roles: dict[str, list[str]] = {}
    offset = 0
    for role in ROLE_ORDER:
        stop = offset + counts[role]
        roles[role] = scenes[offset:stop]
        offset = stop
    return roles


def _finite_pose_stems(pose_dir: Path) -> set[str]:
    if not pose_dir.is_dir():
        return set()
    stems: set[str] = set()
    for path in pose_dir.glob("*.txt"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            values = [float(value) for value in path.read_text(encoding="utf-8").split()]
        except (OSError, ValueError):
            continue
        if len(values) == 16 and all(math.isfinite(value) for value in values):
            stems.add(path.stem)
    return stems


def processed_scene_frame_count(scene_dir: Path) -> int:
    """Count frame IDs with a non-empty RGB image and finite 4x4 pose."""
    color_dir = Path(scene_dir) / "color"
    if not color_dir.is_dir():
        return 0
    image_stems = {
        path.stem
        for path in color_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stat().st_size > 0
    }
    return len(image_stems.intersection(_finite_pose_stems(Path(scene_dir) / "pose")))


def build_final_manifest(
    candidate_manifest: Mapping[str, object],
    accepted_scenes: Sequence[str],
    processed_root: Path,
) -> dict[str, object]:
    """Validate every accepted scene and freeze the adaptation split."""
    _validate_digest(candidate_manifest, "manifest_digest")
    if (
        candidate_manifest.get("schema_version") != SCHEMA_VERSION
        or candidate_manifest.get("study_type") != STUDY_TYPE
    ):
        raise ValueError("candidate manifest schema is unsupported")
    role_counts = candidate_manifest.get("role_counts")
    candidates = candidate_manifest.get("candidate_scenes")
    min_frames = candidate_manifest.get("min_matching_frames")
    if (
        not isinstance(role_counts, Mapping)
        or not isinstance(candidates, list)
        or not isinstance(min_frames, int)
    ):
        raise ValueError("candidate manifest structure is invalid")
    scenes = _validated_scenes(accepted_scenes, label="accepted scenes")
    expected = int(candidate_manifest["target_scene_count"])
    if len(scenes) != expected:
        raise ValueError(f"accepted scene list must contain exactly {expected} scenes")
    candidate_positions = {scene: index for index, scene in enumerate(candidates)}
    try:
        positions = [candidate_positions[scene] for scene in scenes]
    except KeyError as error:
        raise ValueError(f"accepted scene is not an official candidate: {error.args[0]}") from error
    if positions != sorted(positions):
        raise ValueError("accepted scenes do not follow the frozen candidate order")

    roles = assign_scene_roles(scenes, role_counts)
    scene_to_role = {
        scene: role for role, role_scenes in roles.items() for scene in role_scenes
    }
    root = Path(processed_root).resolve()
    entries = []
    for scene in scenes:
        frame_count = processed_scene_frame_count(root / scene)
        if frame_count < min_frames:
            raise ValueError(
                f"{scene} has {frame_count} matching frames; need at least {min_frames}"
            )
        entries.append(
            {
                "scene": scene,
                "role": scene_to_role[scene],
                "frame_count": frame_count,
                "relative_path": scene,
            }
        )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "study_type": STUDY_TYPE,
        "candidate_manifest_digest": candidate_manifest["manifest_digest"],
        "min_matching_frames": min_frames,
        "scene_count": len(scenes),
        "scene_roles": roles,
        "scenes": entries,
    }
    return _with_digest(payload, "dataset_digest")


def _read_scene_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON document: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if destination.is_file():
        if destination.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to replace a different frozen manifest: {destination}")
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-candidate-manifest")
    build.add_argument("--official-train", type=Path, required=True)
    build.add_argument("--exclude-scenes", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=33)
    build.add_argument("--min-frames", type=int, default=500)
    build.add_argument("--refiner-train-scenes", type=int, default=160)
    build.add_argument("--validation-scenes", type=int, default=20)
    build.add_argument("--selector-train-scenes", type=int, default=20)

    listing = commands.add_parser("list-candidates")
    listing.add_argument("--manifest", type=Path, required=True)

    count = commands.add_parser("processed-scene-frame-count")
    count.add_argument("--scene-dir", type=Path, required=True)

    final = commands.add_parser("finalize")
    final.add_argument("--candidate-manifest", type=Path, required=True)
    final.add_argument("--accepted-scenes", type=Path, required=True)
    final.add_argument("--processed-root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build-candidate-manifest":
        manifest = build_candidate_manifest(
            _read_scene_file(args.official_train),
            excluded_scenes=_read_scene_file(args.exclude_scenes),
            seed=args.seed,
            role_counts={
                "refiner_train": args.refiner_train_scenes,
                "validation": args.validation_scenes,
                "selector_train": args.selector_train_scenes,
            },
            min_frames=args.min_frames,
        )
        _write_json(args.output, manifest)
        print(args.output)
        return
    if args.command == "list-candidates":
        manifest = _read_json(args.manifest)
        _validate_digest(manifest, "manifest_digest")
        for scene in manifest.get("candidate_scenes", []):
            print(scene)
        return
    if args.command == "processed-scene-frame-count":
        print(processed_scene_frame_count(args.scene_dir))
        return
    if args.command == "finalize":
        manifest = build_final_manifest(
            _read_json(args.candidate_manifest),
            _read_scene_file(args.accepted_scenes),
            args.processed_root,
        )
        _write_json(args.output, manifest)
        print(args.output)
        return
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
