"""Run frozen VGGT on ordered CO3D long clips and aligned short windows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Protocol, Sequence

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from .cache_schema import (
    load_sequence_shard,
    save_sequence_shard,
    sha256_file,
    write_sequence_manifest,
)
from .co3d_manifest import ClipManifest, ClipSpec, load_clip_manifest
from .geometry import align_pose_to_reference, select_consensus_short_pose


_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class CameraPrediction:
    activated_pose: np.ndarray
    raw_pose: np.ndarray | None = None
    hidden: np.ndarray | None = None
    camera_tokens: np.ndarray | None = None
    diagnostics: np.ndarray | None = None


class CameraRunner(Protocol):
    camera_iterations: int
    pose_projection: np.ndarray

    def predict(
        self, image_paths: tuple[Path, ...], *, trace: bool
    ) -> CameraPrediction: ...


class ClipBuildRejected(RuntimeError):
    """A deterministic clip-level geometry failure that may be recorded and skipped."""


def _atomic_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(destination)


def canonical_short_windows(
    *, frame_count: int, length: int, stride: int
) -> tuple[np.ndarray, ...]:
    """Return fixed windows that cover every long-clip frame, including the tail."""
    if frame_count < 2 or length < 2 or length > frame_count or stride < 1:
        raise ValueError("short-window dimensions are invalid")
    last_start = frame_count - length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return tuple(
        np.arange(start, start + length, dtype=np.int64) for start in starts
    )


def _activated_raw(raw_pose: np.ndarray) -> np.ndarray:
    result = np.asarray(raw_pose, dtype=np.float32).copy()
    result[..., 7:] = np.maximum(result[..., 7:], 0)
    return result


def _validate_prediction(
    prediction: CameraPrediction, frame_count: int, *, trace: bool
) -> None:
    pose = np.asarray(prediction.activated_pose)
    if pose.shape != (frame_count, 9) or not np.isfinite(pose).all():
        raise ValueError("camera prediction pose must have finite shape [S, 9]")
    if np.any(pose[:, 7:] < 0) or np.any(np.linalg.norm(pose[:, 3:7], axis=1) <= 1e-8):
        raise ValueError("camera prediction contains invalid activated values")
    if not trace:
        return
    if prediction.raw_pose is None:
        raise ValueError("long prediction is missing raw pose")
    raw = np.asarray(prediction.raw_pose)
    if raw.shape != pose.shape or not np.isfinite(raw).all():
        raise ValueError("long raw pose must have shape [S, 9]")
    if not np.allclose(pose, _activated_raw(raw), rtol=1e-5, atol=1e-6):
        raise ValueError("long activated pose does not match raw pose")
    expected = (
        ("hidden", prediction.hidden, (frame_count, 1024)),
        ("camera_tokens", prediction.camera_tokens, (frame_count, 2048)),
        (
            "diagnostics",
            prediction.diagnostics,
            (frame_count, prediction.diagnostics.shape[1])
            if prediction.diagnostics is not None and prediction.diagnostics.ndim == 2
            else (),
        ),
    )
    for name, value, shape in expected:
        if value is None:
            raise ValueError(f"long prediction is missing {name}")
        array = np.asarray(value)
        if tuple(array.shape) != shape or not np.isfinite(array).all():
            raise ValueError(f"long prediction {name} has invalid shape or values")
    assert prediction.diagnostics is not None
    if prediction.diagnostics.shape[1] < 1:
        raise ValueError("long prediction diagnostics must contain an iteration axis")


def _build_clip_sample(
    clip: ClipSpec,
    *,
    runner: CameraRunner,
    short_window: int,
    short_stride: int,
    feature_dtype: np.dtype,
) -> dict[str, np.ndarray | str | int]:
    frame_count = len(clip.frame_numbers)
    long_prediction = runner.predict(clip.image_paths, trace=True)
    _validate_prediction(long_prediction, frame_count, trace=True)
    short_indices = canonical_short_windows(
        frame_count=frame_count, length=short_window, stride=short_stride
    )
    aligned_poses = []
    translation_residuals = []
    rotation_residuals = []
    alignment_scales = []
    alignment_rotations = []
    alignment_translations = []
    for indices in short_indices:
        paths = tuple(clip.image_paths[int(index)] for index in indices)
        prediction = runner.predict(paths, trace=False)
        _validate_prediction(prediction, len(indices), trace=False)
        try:
            alignment = align_pose_to_reference(
                long_prediction.activated_pose[indices], prediction.activated_pose
            )
        except ValueError as error:
            raise ClipBuildRejected(f"short pose alignment failed: {error}") from error
        aligned_poses.append(alignment.aligned_pose)
        translation_residuals.append(alignment.translation_residual)
        rotation_residuals.append(alignment.rotation_residual_deg)
        alignment_scales.append(alignment.scale)
        alignment_rotations.append(alignment.rotation)
        alignment_translations.append(alignment.translation)
    try:
        consensus = select_consensus_short_pose(
            frame_count=frame_count,
            frame_indices=short_indices,
            aligned_poses=tuple(aligned_poses),
        )
    except ValueError as error:
        raise ClipBuildRejected(f"short pose consensus failed: {error}") from error
    assert long_prediction.raw_pose is not None
    assert long_prediction.hidden is not None
    assert long_prediction.camera_tokens is not None
    assert long_prediction.diagnostics is not None
    return {
        "scene_name": f"{clip.category}/{clip.sequence_name}",
        "clip_id": clip.clip_id,
        "long_hidden": np.asarray(long_prediction.hidden, dtype=feature_dtype),
        "camera_tokens": np.asarray(
            long_prediction.camera_tokens, dtype=feature_dtype
        ),
        "baseline_raw_pose": np.asarray(long_prediction.raw_pose, dtype=np.float32),
        "baseline_pose": np.asarray(long_prediction.activated_pose, dtype=np.float32),
        "short_pose": consensus.pose,
        "diagnostics": np.asarray(long_prediction.diagnostics, dtype=np.float32),
        "frame_ids": np.asarray(clip.frame_numbers, dtype=np.int64),
        "start": int(clip.start_index),
        "temporal_stride": int(clip.temporal_stride),
        "gt_c2w_raw": np.asarray(clip.gt_c2w, dtype=np.float64),
        "gt_focal_length": np.asarray(clip.focal_length, dtype=np.float32),
        "gt_principal_point": np.asarray(clip.principal_point, dtype=np.float32),
        "gt_image_size": np.asarray(clip.image_size, dtype=np.int64),
        "short_pose_observations": np.stack(aligned_poses).astype(np.float32),
        "short_frame_indices": np.stack(short_indices).astype(np.int64),
        "short_observation_count": consensus.observation_count,
        "selected_short_window": consensus.selected_window,
        "selected_boundary_distance": consensus.selected_boundary_distance,
        "short_alignment_translation_residual": np.stack(
            translation_residuals
        ).astype(np.float32),
        "short_alignment_rotation_residual_deg": np.stack(
            rotation_residuals
        ).astype(np.float32),
        "short_alignment_scale": np.asarray(alignment_scales, dtype=np.float64),
        "short_alignment_rotation": np.stack(alignment_rotations).astype(np.float64),
        "short_alignment_translation": np.stack(alignment_translations).astype(
            np.float64
        ),
    }


def _stack_samples(
    samples: Sequence[dict[str, np.ndarray | str | int]],
) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("cannot build an empty sequence shard")
    scalar_strings = {"scene_name": "scene_names", "clip_id": "clip_ids"}
    scalar_ints = {"start": "starts", "temporal_stride": "temporal_strides"}
    arrays: dict[str, np.ndarray] = {}
    for source, destination in scalar_strings.items():
        arrays[destination] = np.asarray([str(sample[source]) for sample in samples])
    for source, destination in scalar_ints.items():
        arrays[destination] = np.asarray(
            [int(sample[source]) for sample in samples], dtype=np.int64
        )
    excluded = set(scalar_strings) | set(scalar_ints)
    for name in samples[0]:
        if name in excluded:
            continue
        arrays[name] = np.stack([np.asarray(sample[name]) for sample in samples])
    return arrays


def _safe_component(value: str, description: str) -> str:
    if _SAFE_NAME.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


def _completed_record(
    marker_path: Path,
    shard_path: Path,
    *,
    build_digest: str,
    source_manifest_digest: str,
) -> dict[str, object] | None:
    if not marker_path.is_file() or not shard_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("build_digest") != build_digest
            or marker.get("source_manifest_digest") != source_manifest_digest
            or marker.get("sha256") != sha256_file(shard_path)
        ):
            return None
        arrays = load_sequence_shard(shard_path)
        sample_count = int(marker.get("sample_count", 0))
        if sample_count != len(arrays["long_hidden"]):
            return None
        return {
            "path": shard_path,
            "role": str(marker["role"]),
            "scene": str(marker["scene"]),
            "sample_count": sample_count,
            "rejections": list(marker.get("rejections", [])),
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _completed_rejection(
    marker_path: Path,
    *,
    build_digest: str,
    source_manifest_digest: str,
) -> list[dict[str, object]] | None:
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        rejections = marker.get("rejections")
        if (
            marker.get("build_digest") != build_digest
            or marker.get("source_manifest_digest") != source_manifest_digest
            or not isinstance(rejections, list)
            or not rejections
            or not all(isinstance(item, dict) for item in rejections)
        ):
            return None
        return rejections
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def build_cache(
    manifest: ClipManifest,
    *,
    output_dir: Path,
    runner: CameraRunner,
    short_window: int,
    short_stride: int,
    feature_dtype: str,
    build_digest: str,
) -> dict[str, object]:
    """Materialize one resumable authenticated shard per CO3D sequence."""
    if len(build_digest) != 64:
        raise ValueError("build_digest must be a SHA-256 value")
    if feature_dtype not in {"float16", "float32"}:
        raise ValueError("feature_dtype must be float16 or float32")
    dtype = np.dtype(feature_dtype)
    if short_window > manifest.clip_length:
        raise ValueError("short window cannot exceed the long clip")
    if not manifest.clips:
        raise ValueError("clip manifest contains no accepted clips")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    projection = np.asarray(runner.pose_projection, dtype=np.float32)
    if projection.shape != (9, 1024) or not np.isfinite(projection).all():
        raise ValueError("runner pose projection must have shape [9, 1024]")
    projection_path = root / "pose_projection.npy"
    if projection_path.is_file():
        existing = np.load(projection_path, allow_pickle=False)
        if not np.array_equal(existing, projection):
            raise ValueError("output directory belongs to another pose projection")
    else:
        _atomic_npy(projection_path, projection)

    grouped: dict[tuple[str, str], list[ClipSpec]] = defaultdict(list)
    for clip in manifest.clips:
        grouped[(clip.category, clip.sequence_name)].append(clip)
    shard_records = []
    all_rejections: list[dict[str, object]] = []
    for index, ((category, sequence_name), clips) in enumerate(
        sorted(grouped.items()), start=1
    ):
        roles = {clip.role for clip in clips}
        if len(roles) != 1:
            raise ValueError(f"sequence crosses roles: {category}/{sequence_name}")
        role = roles.pop()
        category_path = _safe_component(category, "category")
        sequence_path = _safe_component(sequence_name, "sequence name")
        shard_path = root / "shards" / role / category_path / f"{sequence_path}.npz"
        marker_path = shard_path.with_suffix(".complete.json")
        rejection_marker_path = shard_path.with_suffix(".rejected.json")
        record = _completed_record(
            marker_path,
            shard_path,
            build_digest=build_digest,
            source_manifest_digest=manifest.digest,
        )
        if record is None:
            cached_rejections = _completed_rejection(
                rejection_marker_path,
                build_digest=build_digest,
                source_manifest_digest=manifest.digest,
            )
            if cached_rejections is not None:
                all_rejections.extend(cached_rejections)
                print(
                    f"[resume-rejected {index}/{len(grouped)}] "
                    f"{category}/{sequence_name}",
                    flush=True,
                )
                continue
            samples = []
            sequence_rejections = []
            for clip in sorted(clips, key=lambda item: item.clip_id):
                try:
                    samples.append(
                        _build_clip_sample(
                            clip,
                            runner=runner,
                            short_window=short_window,
                            short_stride=short_stride,
                            feature_dtype=dtype,
                        )
                    )
                except ClipBuildRejected as error:
                    sequence_rejections.append(
                        {
                            "category": category,
                            "sequence_name": sequence_name,
                            "clip_id": clip.clip_id,
                            "reason": str(error),
                        }
                    )
            all_rejections.extend(sequence_rejections)
            if not samples:
                _atomic_json(
                    rejection_marker_path,
                    {
                        "schema_version": 1,
                        "build_digest": build_digest,
                        "source_manifest_digest": manifest.digest,
                        "rejections": sequence_rejections,
                    },
                )
                print(
                    f"[rejected {index}/{len(grouped)}] {category}/{sequence_name}",
                    flush=True,
                )
                continue
            save_sequence_shard(shard_path, _stack_samples(samples))
            rejection_marker_path.unlink(missing_ok=True)
            record = {
                "path": shard_path,
                "role": role,
                "scene": f"{category}/{sequence_name}",
                "sample_count": len(samples),
            }
            _atomic_json(
                marker_path,
                {
                    "schema_version": 1,
                    "build_digest": build_digest,
                    "source_manifest_digest": manifest.digest,
                    "sha256": sha256_file(shard_path),
                    "role": role,
                    "scene": record["scene"],
                    "sample_count": len(samples),
                    "rejections": sequence_rejections,
                },
            )
            status = "built"
        else:
            cached = record.get("rejections", [])
            if isinstance(cached, list):
                all_rejections.extend(
                    item for item in cached if isinstance(item, dict)
                )
            status = "resume"
        shard_records.append(record)
        print(
            f"[{status} {index}/{len(grouped)}] {category}/{sequence_name}",
            flush=True,
        )
    all_rejections.sort(
        key=lambda item: (
            str(item.get("category", "")),
            str(item.get("sequence_name", "")),
            str(item.get("clip_id", "")),
        )
    )
    _atomic_json(
        root / "cache_rejections.json",
        {
            "schema_version": 1,
            "build_digest": build_digest,
            "source_manifest_digest": manifest.digest,
            "rejection_count": len(all_rejections),
            "rejections": all_rejections,
        },
    )
    if not shard_records:
        raise RuntimeError("all CO3D clips were rejected; inspect cache_rejections.json")
    return write_sequence_manifest(
        root / "manifest.json",
        dataset_root=root,
        projection_path=projection_path,
        shard_records=tuple(shard_records),
        camera_iterations=int(runner.camera_iterations),
        source_manifest_digest=manifest.digest,
    )


def find_checkpoint(checkpoint_dir: Path) -> Path:
    """Resolve a local checkpoint directory or Hugging Face cache snapshot."""
    root = Path(checkpoint_dir).resolve()
    roots = [root]
    reference = root / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        if revision:
            roots.insert(0, root / "snapshots" / revision)
    if (root / "snapshots").is_dir():
        roots.extend(sorted((root / "snapshots").iterdir(), reverse=True))
    for candidate_root in roots:
        for name in ("model.safetensors", "pytorch_model.bin", "model.pt"):
            candidate = candidate_root / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no local VGGT checkpoint found under {root}")


def _load_pose_only_model(checkpoint_path: Path) -> VGGT:
    model = VGGT(enable_track=False, enable_point=False, enable_depth=False)
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover - environment failure only.
            raise RuntimeError("safetensors is required for this checkpoint") from error
        state = load_file(str(checkpoint_path), device="cpu")
    else:
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older supported torch releases.
            state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("VGGT checkpoint does not contain a state dictionary")
    incompatible = model.load_state_dict(state, strict=False)
    critical_missing = [
        name
        for name in incompatible.missing_keys
        if name.startswith(("aggregator.", "camera_head."))
    ]
    if critical_missing:
        raise ValueError(f"VGGT checkpoint misses camera-path weights: {critical_missing[:5]}")
    return model


class VGGTCameraRunner:
    """Camera-only frozen VGGT inference with final hidden-state tracing."""

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        device: str,
        preprocess_mode: str,
        camera_iterations: int,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if preprocess_mode not in {"crop", "pad"} or camera_iterations < 1:
            raise ValueError("preprocess mode or camera iterations are invalid")
        self.device = torch.device(device)
        self.preprocess_mode = preprocess_mode
        self.camera_iterations = camera_iterations
        self.checkpoint_path = find_checkpoint(checkpoint_dir)
        self.checkpoint_digest = sha256_file(self.checkpoint_path)
        self.model = _load_pose_only_model(self.checkpoint_path).to(self.device).eval()
        assert self.model.camera_head is not None
        self.pose_projection = (
            self.model.camera_head.pose_branch.fc2.weight.detach().float().cpu().numpy().copy()
        )

    def predict(
        self, image_paths: tuple[Path, ...], *, trace: bool
    ) -> CameraPrediction:
        images = load_and_preprocess_images(
            [str(path) for path in image_paths], mode=self.preprocess_mode
        ).to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(
                images,
                camera_num_iterations=self.camera_iterations,
                return_camera_trace=trace,
                camera_trace_pose_tokens=trace,
            )
        activated = output["pose_enc"][0].detach().float().cpu().numpy()
        if not trace:
            return CameraPrediction(activated_pose=activated)
        camera_trace = output["camera_trace"]
        raw = camera_trace["raw_pose_enc_list"][-1][0].detach().float().cpu().numpy()
        hidden = (
            camera_trace["pose_branch_hidden_list"][-1][0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        tokens = (
            camera_trace["normalized_camera_tokens"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        diagnostics = (
            camera_trace["delta_norm"][:, 0]
            .transpose(0, 1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        return CameraPrediction(
            activated_pose=activated,
            raw_pose=raw,
            hidden=hidden,
            camera_tokens=tokens,
            diagnostics=diagnostics,
        )


def _limited_manifest(manifest: ClipManifest, sequence_limit: int) -> ClipManifest:
    if sequence_limit == 0:
        return manifest
    identities = []
    for clip in manifest.clips:
        identity = (clip.category, clip.sequence_name)
        if identity not in identities:
            identities.append(identity)
    selected = set(identities[:sequence_limit])
    clips = tuple(
        clip
        for clip in manifest.clips
        if (clip.category, clip.sequence_name) in selected
    )
    digest = hashlib.sha256(
        f"{manifest.digest}\0{sequence_limit}\0{sorted(selected)}".encode("utf-8")
    ).hexdigest()
    return ClipManifest(
        clips=clips,
        digest=digest,
        clip_length=manifest.clip_length,
        source_selection_digest=manifest.source_selection_digest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-manifest", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--short-window", type=int, default=50)
    parser.add_argument("--short-stride", type=int, default=25)
    parser.add_argument("--camera-iterations", type=int, default=4)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="pad")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sequence-limit", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.short_window < 2 or args.short_stride < 1 or args.sequence_limit < 0:
        raise ValueError("window and sequence controls are invalid")
    source_manifest = load_clip_manifest(args.clip_manifest, args.data_root)
    manifest = _limited_manifest(source_manifest, args.sequence_limit)
    runner = VGGTCameraRunner(
        args.ckpt_dir,
        device=args.device,
        preprocess_mode=args.preprocess_mode,
        camera_iterations=args.camera_iterations,
    )
    invocation = {
        "source_manifest_digest": manifest.digest,
        "checkpoint_sha256": runner.checkpoint_digest,
        "short_window": args.short_window,
        "short_stride": args.short_stride,
        "camera_iterations": args.camera_iterations,
        "feature_dtype": args.feature_dtype,
        "preprocess_mode": args.preprocess_mode,
        "sequence_limit": args.sequence_limit,
    }
    build_digest = hashlib.sha256(
        json.dumps(invocation, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(
        args.out_dir / "run_metadata.json",
        {
            "schema_version": 1,
            "study_name": "co3d_full_hidden_sequence_cache",
            "build_digest": build_digest,
            "checkpoint_path": runner.checkpoint_path.as_posix(),
            "invocation": invocation,
        },
    )
    result = build_cache(
        manifest,
        output_dir=args.out_dir,
        runner=runner,
        short_window=args.short_window,
        short_stride=args.short_stride,
        feature_dtype=args.feature_dtype,
        build_digest=build_digest,
    )
    _atomic_json(
        args.out_dir / "complete.json",
        {
            "schema_version": 1,
            "build_digest": build_digest,
            "sample_count": result["sample_count"],
            "shard_count": len(result["shards"]),
            "manifest": "manifest.json",
        },
    )
    print(f"[done] cache={args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
