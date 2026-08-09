"""Build deterministic ordered CO3D clips from the downloaded RGB subset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .co3d_download import _canonical_image_path, _iter_jgz_rows


CLIP_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    category: str
    sequence_name: str
    role: str
    start_index: int
    temporal_stride: int
    frame_numbers: np.ndarray
    image_paths: tuple[Path, ...]
    gt_c2w: np.ndarray
    focal_length: np.ndarray
    principal_point: np.ndarray
    image_size: np.ndarray


@dataclass(frozen=True)
class ClipManifest:
    clips: tuple[ClipSpec, ...]
    digest: str
    clip_length: int
    source_selection_digest: str


@dataclass(frozen=True)
class _Frame:
    number: int
    relative_image_path: str
    c2w: np.ndarray
    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    image_size: tuple[int, int]


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def pytorch3d_viewpoint_to_c2w(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Convert a CO3D PyTorch3D row-vector camera to OpenCV-style c2w."""
    source_rotation = np.asarray(rotation, dtype=np.float64)
    source_translation = np.asarray(translation, dtype=np.float64)
    if source_rotation.shape != (3, 3) or source_translation.shape != (3,):
        raise ValueError("CO3D viewpoint must contain R [3,3] and T [3]")
    if not np.isfinite(source_rotation).all() or not np.isfinite(source_translation).all():
        raise ValueError("CO3D viewpoint must contain finite values")
    if not np.allclose(source_rotation.T @ source_rotation, np.eye(3), atol=1e-5):
        raise ValueError("CO3D viewpoint rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(source_rotation)), 1.0, abs_tol=1e-5):
        raise ValueError("CO3D viewpoint rotation must be proper")
    axis_flip = np.diag([-1.0, -1.0, 1.0])
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = source_rotation @ axis_flip
    c2w[:3, 3] = -(source_rotation @ source_translation)
    return c2w


def _finite_pair(value: object, name: str) -> tuple[float, float]:
    try:
        pair = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two finite values") from error
    if len(pair) != 2 or not np.isfinite(pair).all():
        raise ValueError(f"{name} must contain two finite values")
    return pair  # type: ignore[return-value]


def _image_size(image: Mapping[str, object]) -> tuple[int, int]:
    raw = image.get("size", (0, 0))
    try:
        values = tuple(int(item) for item in raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("CO3D image size is invalid") from error
    if len(values) != 2 or any(value < 0 for value in values):
        raise ValueError("CO3D image size is invalid")
    return values  # type: ignore[return-value]


def _parse_frame(
    row: Mapping[str, object],
    *,
    category: str,
    sequence_name: str,
    data_root: Path,
) -> _Frame:
    raw_number = row.get("frame_number")
    if isinstance(raw_number, bool):
        raise ValueError("CO3D frame number is invalid")
    try:
        number = int(raw_number)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("CO3D frame number is invalid") from error
    if float(raw_number) != number:  # type: ignore[arg-type]
        raise ValueError("CO3D frame number is invalid")
    image = row.get("image")
    viewpoint = row.get("viewpoint")
    if not isinstance(image, Mapping) or not isinstance(viewpoint, Mapping):
        raise ValueError("CO3D frame is missing image or viewpoint")
    raw_path = image.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("CO3D frame image path is invalid")
    relative = _canonical_image_path(raw_path, category, sequence_name)
    image_path = data_root.joinpath(*Path(relative).parts)
    if not image_path.is_file():
        raise FileNotFoundError(f"missing selected CO3D image: {image_path}")
    rotation = np.asarray(viewpoint.get("R"), dtype=np.float64)
    translation = np.asarray(viewpoint.get("T"), dtype=np.float64)
    return _Frame(
        number=number,
        relative_image_path=relative,
        c2w=pytorch3d_viewpoint_to_c2w(rotation, translation),
        focal_length=_finite_pair(viewpoint.get("focal_length"), "focal_length"),
        principal_point=_finite_pair(
            viewpoint.get("principal_point"), "principal_point"
        ),
        image_size=_image_size(image),
    )


def _selected_sequences(download: Mapping[str, object]) -> dict[str, dict[str, Mapping[str, object]]]:
    if download.get("schema_version") != 2 or download.get("dataset") != "CO3Dv2":
        raise ValueError("download manifest is not a CO3Dv2 schema-2 selection")
    selection_digest = download.get("selection_sha256")
    records = download.get("sequences")
    if not isinstance(selection_digest, str) or len(selection_digest) != 64:
        raise ValueError("download manifest selection digest is invalid")
    if not isinstance(records, list) or not records:
        raise ValueError("download manifest contains no selected sequences")
    selected: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("download sequence record is invalid")
        category = record.get("category")
        sequence_name = record.get("sequence_name")
        if not isinstance(category, str) or not isinstance(sequence_name, str):
            raise ValueError("download sequence identity is invalid")
        if sequence_name in selected[category]:
            raise ValueError(f"duplicate download sequence: {category}/{sequence_name}")
        selected[category][sequence_name] = record
    return dict(selected)


def _load_selected_frames(
    data_root: Path,
    category: str,
    selected: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, tuple[_Frame, ...]], list[dict[str, object]]]:
    frames: dict[str, list[_Frame]] = defaultdict(list)
    parse_errors: dict[str, str] = {}
    annotation_path = data_root / category / "frame_annotations.jgz"
    for row in _iter_jgz_rows(
        annotation_path, keys=("frames", "frame_annotations")
    ):
        sequence_name = row.get("sequence_name")
        if not isinstance(sequence_name, str) or sequence_name not in selected:
            continue
        if sequence_name in parse_errors:
            continue
        try:
            frames[sequence_name].append(
                _parse_frame(
                    row,
                    category=category,
                    sequence_name=sequence_name,
                    data_root=data_root,
                )
            )
        except (ValueError, FileNotFoundError) as error:
            parse_errors[sequence_name] = str(error)

    accepted = {}
    rejections = []
    for sequence_name, record in selected.items():
        sequence_frames = sorted(frames.get(sequence_name, ()), key=lambda item: item.number)
        expected_count = int(record.get("valid_frame_count", 0))
        numbers = [frame.number for frame in sequence_frames]
        reason = parse_errors.get(sequence_name)
        if reason is None and len(numbers) != len(set(numbers)):
            reason = "duplicate_frame_number"
        if reason is None and len(sequence_frames) != expected_count:
            reason = f"annotation_image_count_mismatch:{len(sequence_frames)}!={expected_count}"
        if reason is not None:
            rejections.append(
                {
                    "category": category,
                    "sequence_name": sequence_name,
                    "reason": reason,
                }
            )
            continue
        accepted[sequence_name] = tuple(sequence_frames)
    return accepted, rejections


def _role(category: str, sequence_name: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(
        f"{seed}\0{category}\0{sequence_name}\0split".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_fraction else "train"


def _clip_candidates(
    frames: Sequence[_Frame],
    *,
    category: str,
    sequence_name: str,
    clip_length: int,
    temporal_strides: Sequence[int],
    seed: int,
) -> list[tuple[int, int]]:
    candidates = []
    for stride in temporal_strides:
        last_start = len(frames) - 1 - (clip_length - 1) * stride
        candidates.extend((stride, start) for start in range(max(last_start + 1, 0)))
    candidates.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}\0{category}\0{sequence_name}\0{item[0]}\0{item[1]}".encode(
                "utf-8"
            )
        ).hexdigest()
    )
    return candidates


def build_clip_manifest(
    *,
    data_root: Path,
    download_manifest: Path,
    output_path: Path,
    clip_length: int,
    max_clips_per_sequence: int,
    temporal_strides: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> dict[str, object]:
    """Freeze ordered fixed-length clips without duplicating or looping frames."""
    root = Path(data_root).resolve()
    if clip_length < 2 or max_clips_per_sequence < 1:
        raise ValueError("clip length and clips per sequence must be positive")
    strides = tuple(int(value) for value in temporal_strides)
    if not strides or any(value < 1 for value in strides) or len(set(strides)) != len(strides):
        raise ValueError("temporal strides must be unique positive integers")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    download = json.loads(Path(download_manifest).read_text(encoding="utf-8"))
    if not isinstance(download, Mapping):
        raise ValueError("download manifest must be a JSON object")
    selected = _selected_sequences(download)
    clips: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    for category in sorted(selected):
        sequences, category_rejections = _load_selected_frames(
            root, category, selected[category]
        )
        rejections.extend(category_rejections)
        for sequence_name in sorted(sequences):
            frames = sequences[sequence_name]
            candidates = _clip_candidates(
                frames,
                category=category,
                sequence_name=sequence_name,
                clip_length=clip_length,
                temporal_strides=strides,
                seed=seed,
            )
            if not candidates:
                rejections.append(
                    {
                        "category": category,
                        "sequence_name": sequence_name,
                        "reason": "insufficient_frames",
                        "valid_frame_count": len(frames),
                        "required_frame_span": min(
                            1 + (clip_length - 1) * stride for stride in strides
                        ),
                    }
                )
                continue
            role = _role(category, sequence_name, seed, validation_fraction)
            for stride, start in candidates[:max_clips_per_sequence]:
                selected_frames = tuple(
                    frames[start + offset * stride] for offset in range(clip_length)
                )
                identity = f"{category}/{sequence_name}/s{stride}/i{start}"
                clip_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                clips.append(
                    {
                        "clip_id": clip_id,
                        "category": category,
                        "sequence_name": sequence_name,
                        "role": role,
                        "start_index": start,
                        "temporal_stride": stride,
                        "frame_numbers": [frame.number for frame in selected_frames],
                        "image_paths": [frame.relative_image_path for frame in selected_frames],
                        "gt_c2w": [frame.c2w.tolist() for frame in selected_frames],
                        "focal_length": [frame.focal_length for frame in selected_frames],
                        "principal_point": [frame.principal_point for frame in selected_frames],
                        "image_size": [frame.image_size for frame in selected_frames],
                    }
                )
    clips.sort(key=lambda item: (str(item["category"]), str(item["sequence_name"]), str(item["clip_id"])))
    rejections.sort(key=lambda item: (str(item["category"]), str(item["sequence_name"])))
    payload: dict[str, object] = {
        "schema_version": CLIP_MANIFEST_SCHEMA_VERSION,
        "dataset": "CO3Dv2",
        "study_type": "ordered_long_short_pose_clips",
        "source_download_manifest": Path(download_manifest).resolve().as_posix(),
        "source_selection_digest": str(download["selection_sha256"]),
        "clip_length": clip_length,
        "max_clips_per_sequence": max_clips_per_sequence,
        "temporal_strides": list(strides),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "clip_count": len(clips),
        "sequence_count": len({(item["category"], item["sequence_name"]) for item in clips}),
        "clips": clips,
        "rejections": rejections,
    }
    payload["manifest_digest"] = _canonical_digest(payload)
    _atomic_json(output_path, payload)
    return payload


def load_clip_manifest(path: Path, data_root: Path) -> ClipManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CLIP_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported clip manifest")
    declared = payload.pop("manifest_digest", None)
    if not isinstance(declared, str) or declared != _canonical_digest(payload):
        raise ValueError("clip manifest digest is invalid")
    records = payload.get("clips")
    clip_length = int(payload.get("clip_length", 0))
    if not isinstance(records, list) or clip_length < 2:
        raise ValueError("clip manifest records are invalid")
    root = Path(data_root).resolve()
    clips = []
    identities = set()
    sequence_roles: dict[tuple[str, str], str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("clip record must be an object")
        clip_id = str(record.get("clip_id", ""))
        category = str(record.get("category", ""))
        sequence_name = str(record.get("sequence_name", ""))
        role = str(record.get("role", ""))
        if not clip_id or clip_id in identities or role not in {"train", "validation"}:
            raise ValueError("clip identity or role is invalid")
        identities.add(clip_id)
        sequence_key = (category, sequence_name)
        previous_role = sequence_roles.setdefault(sequence_key, role)
        if previous_role != role:
            raise ValueError("one sequence appears in multiple roles")
        numbers = np.asarray(record.get("frame_numbers"), dtype=np.int64)
        paths = []
        for relative in record.get("image_paths", []):  # type: ignore[union-attr]
            candidate = (root / str(relative)).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ValueError("clip image escapes data root") from error
            if not candidate.is_file():
                raise FileNotFoundError(f"missing clip image: {candidate}")
            paths.append(candidate)
        gt_c2w = np.asarray(record.get("gt_c2w"), dtype=np.float64)
        focal = np.asarray(record.get("focal_length"), dtype=np.float32)
        principal = np.asarray(record.get("principal_point"), dtype=np.float32)
        image_size = np.asarray(record.get("image_size"), dtype=np.int64)
        if (
            numbers.shape != (clip_length,)
            or np.any(np.diff(numbers) <= 0)
            or len(paths) != clip_length
            or gt_c2w.shape != (clip_length, 4, 4)
            or focal.shape != (clip_length, 2)
            or principal.shape != (clip_length, 2)
            or image_size.shape != (clip_length, 2)
        ):
            raise ValueError(f"clip tensors have invalid shape: {clip_id}")
        numeric = (gt_c2w, focal, principal)
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError(f"clip contains non-finite values: {clip_id}")
        clips.append(
            ClipSpec(
                clip_id=clip_id,
                category=category,
                sequence_name=sequence_name,
                role=role,
                start_index=int(record.get("start_index", -1)),
                temporal_stride=int(record.get("temporal_stride", 0)),
                frame_numbers=numbers,
                image_paths=tuple(paths),
                gt_c2w=gt_c2w,
                focal_length=focal,
                principal_point=principal,
                image_size=image_size,
            )
        )
    return ClipManifest(
        clips=tuple(clips),
        digest=declared,
        clip_length=clip_length,
        source_selection_digest=str(payload.get("source_selection_digest", "")),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=100)
    parser.add_argument("--max-clips-per-sequence", type=int, default=1)
    parser.add_argument("--temporal-strides", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = build_clip_manifest(
        data_root=args.data_root,
        download_manifest=args.download_manifest,
        output_path=args.output,
        clip_length=args.clip_length,
        max_clips_per_sequence=args.max_clips_per_sequence,
        temporal_strides=args.temporal_strides,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        f"[manifest] clips={payload['clip_count']} "
        f"sequences={payload['sequence_count']} "
        f"rejections={len(payload['rejections'])} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
