"""Prediction-gauge short-window teachers from authenticated legacy controls."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import os
from pathlib import Path
import re
from typing import Mapping
import zipfile

import numpy as np
import torch
from torch import nn

from pre_experiments.camera_translation_hvrfm.geometry import (
    _validate_baseline_c2w,
    _validate_so3,
    build_translation_endpoint,
    prediction_scale,
)
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    FrozenOracle,
    apply_frozen_oracle,
    fit_frozen_oracle,  # compatibility symbol; forbidden in the numeric builder
)
from pre_experiments.camera_velocity_ambiguity_02.geometry import (
    align_local_to_global,
)
from pre_experiments.conditional_hierarchical_vrfm.artifacts import (
    TEACHER_ARTIFACT_MEMBERS,
    load_teacher_artifact,
)
from pre_experiments.conditional_hierarchical_vrfm.teacher import (
    build_variant_window_masks as _build_legacy_variant_window_masks,
)
from pre_experiments.long_short_camera_head.data import (
    load_prepared_gt,  # compatibility symbol; forbidden in the numeric builder
)
from pre_experiments.variational_camera_latent.camera import (
    decode_camera_tokens,
    pose_encoding_to_c2w,
)
from pre_experiments.variational_camera_latent.schema import validate_source_shard


_FRAMES = 500
_WINDOWS = 9
_WINDOW_FRAMES = 100
_ENDPOINTS = 4
_TOKEN_WIDTH = 2048
_SOURCE_MEMBERS = frozenset(
    {
        "global_frame_ids",
        "global_camera_tokens",
        "short_frame_ids",
        "short_camera_tokens",
        "overlap_frame_ids",
        "overlap_long_tokens",
        "overlap_left_tokens",
        "overlap_right_tokens",
        "span_starts",
        "sample_ids",
        "global_pred_c2w",
        "overlap_long_c2w",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SCENE_RE = re.compile(r"scene\d{4}_\d{2}")
_HOMOGENEOUS_ATOL = 1e-10
_SO3_ATOL = 2e-6
_BASELINE_CENTER_NORMALIZED_ATOL = 5e-6
_BASELINE_ROTATION_ATOL = 2e-5
_WITNESS_CENTER_NORMALIZED_ATOL = 1e-5
_CONTROL_AUTHENTICATION_KEY = os.urandom(32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")
    return value


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    if ".." in Path(path).parts:
        raise ValueError("authenticated paths may not contain lexical parent traversal")
    current = _absolute_without_resolving(path)
    for candidate in (current, *current.parents):
        if candidate.is_symlink():
            raise ValueError(f"authenticated paths may not contain symlinks: {candidate}")


def _inspect_npz(path: Path, expected_members: frozenset[str], *, label: str) -> None:
    _reject_symlink_components(path)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular NPZ file")
    expected = {f"{name}.npy" for name in expected_members}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} contains duplicate ZIP members")
            if any(
                info.is_dir()
                or "/" in info.filename
                or "\\" in info.filename
                or info.filename in {".", ".."}
                for info in infos
            ):
                raise ValueError(f"{label} contains an unsafe ZIP member path")
            if len(names) != len(expected) or set(names) != expected:
                raise ValueError(f"{label} must use the exact schema")
    except ValueError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"invalid {label}: {path}") from error


def _authenticate_file(path: Path, expected_sha256: str, *, label: str) -> Path:
    expected = _canonical_digest(expected_sha256, name=f"expected {label} digest")
    source = Path(path)
    _reject_symlink_components(source)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file")
    actual = _sha256_file(source)
    if actual != expected:
        raise ValueError(f"{label} digest mismatch")
    return source


def _exact_array(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[object] | type[np.generic] | str,
) -> np.ndarray:
    expected_dtype = np.dtype(dtype)
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != expected_dtype
    ):
        raise ValueError(
            f"{name} must have exact shape {shape} and dtype {expected_dtype}"
        )
    result = value.copy()
    result.setflags(write=False)
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.setflags(write=False)
    return result


def _immutable_exact_array(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[object] | type[np.generic] | str,
) -> np.ndarray:
    validated = _exact_array(value, name=name, shape=shape, dtype=dtype)
    immutable_bytes = validated.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=validated.dtype).reshape(shape)


def _coverage_replay(window_weights: np.ndarray, window_masks: np.ndarray) -> np.ndarray:
    coverage = np.zeros((_ENDPOINTS, _FRAMES), dtype=np.float64)
    for endpoint in range(_ENDPOINTS):
        for window in range(_WINDOWS):
            if window_masks[endpoint, window] != 0:
                start = 50 * window
                coverage[endpoint, start : start + _WINDOW_FRAMES] += window_weights[
                    window
                ]
    return coverage


@dataclass(frozen=True, init=False)
class TeacherControls:
    """The only privileged values admitted to numeric teacher construction."""

    scene: str
    frame_ids: np.ndarray
    window_weights: np.ndarray
    window_masks: np.ndarray
    expected_coverage_weights: np.ndarray
    source_sha256: str
    checkpoint_sha256: str
    formal_label_sha256: str
    teacher_reference_sha256: str
    _authentication_tag: bytes = field(init=False, repr=False, compare=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise ValueError(
            "TeacherControls require an authenticated legacy reference"
        )

    def _validate_minted(self) -> None:
        if not isinstance(self.scene, str) or _SCENE_RE.fullmatch(self.scene) is None:
            raise ValueError("scene must use sceneNNNN_NN format")
        frames = _immutable_exact_array(
            self.frame_ids,
            name="frame_ids",
            shape=(_FRAMES,),
            dtype=np.int64,
        )
        if np.any(frames[1:] <= frames[:-1]):
            raise ValueError("frame_ids must be strictly increasing and unique")
        weights = _immutable_exact_array(
            self.window_weights,
            name="window_weights",
            shape=(_WINDOWS,),
            dtype=np.float64,
        )
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or np.any(weights > 1.0)
        ):
            raise ValueError("window_weights must be finite values in [0, 1]")
        masks = _immutable_exact_array(
            self.window_masks,
            name="window_masks",
            shape=(_ENDPOINTS, _WINDOWS),
            dtype=np.uint8,
        )
        if not np.isin(masks, (0, 1)).all():
            raise ValueError("window_masks must be binary uint8")
        if not np.array_equal(masks[0], (weights > 0.0).astype(np.uint8)):
            raise ValueError("window mask row zero must select every positive window")
        if len({row.tobytes() for row in masks}) != _ENDPOINTS:
            raise ValueError("the four registered window-mask rows must be unique")
        if np.any(masks[:, weights == 0.0] != 0):
            raise ValueError("zero-weight windows may not be selected")
        try:
            canonical_masks = _build_legacy_variant_window_masks(
                self.scene, weights
            ).astype(np.uint8, copy=False)
        except ValueError as error:
            raise ValueError("legacy window masks are not canonical") from error
        if not np.array_equal(masks, canonical_masks):
            raise ValueError(
                "window_masks must use the exact canonical legacy row order"
            )
        expected = _immutable_exact_array(
            self.expected_coverage_weights,
            name="expected_coverage_weights",
            shape=(_ENDPOINTS, _FRAMES),
            dtype=np.float64,
        )
        if not np.isfinite(expected).all() or np.any(expected < 0.0):
            raise ValueError("expected coverage must be finite and nonnegative")
        replay = _coverage_replay(weights, masks)
        if not np.array_equal(replay, expected):
            raise ValueError("legacy coverage replay does not match weights and masks")
        for name in (
            "source_sha256",
            "checkpoint_sha256",
            "formal_label_sha256",
            "teacher_reference_sha256",
        ):
            _canonical_digest(getattr(self, name), name=name)
        object.__setattr__(self, "frame_ids", frames)
        object.__setattr__(self, "window_weights", weights)
        object.__setattr__(self, "window_masks", masks)
        object.__setattr__(self, "expected_coverage_weights", expected)


@dataclass(frozen=True)
class _TeacherControlSnapshot:
    scene: str
    frame_ids: np.ndarray
    window_weights: np.ndarray
    window_masks: np.ndarray
    expected_coverage_weights: np.ndarray
    source_sha256: str
    checkpoint_sha256: str
    formal_label_sha256: str
    teacher_reference_sha256: str


def _control_authentication_tag(
    controls: TeacherControls | _TeacherControlSnapshot,
) -> bytes:
    digest = hmac.new(_CONTROL_AUTHENTICATION_KEY, digestmod=hashlib.sha256)
    for name in (
        "scene",
        "source_sha256",
        "checkpoint_sha256",
        "formal_label_sha256",
        "teacher_reference_sha256",
    ):
        encoded = getattr(controls, name).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    for name in (
        "frame_ids",
        "window_weights",
        "window_masks",
        "expected_coverage_weights",
    ):
        value = getattr(controls, name)
        descriptor = f"{name}:{value.dtype.str}:{value.shape}".encode("ascii")
        digest.update(len(descriptor).to_bytes(8, byteorder="big"))
        digest.update(descriptor)
        digest.update(value.tobytes(order="C"))
    return digest.digest()


def _require_authenticated_controls(controls: TeacherControls) -> None:
    try:
        actual = controls._authentication_tag
        expected = _control_authentication_tag(controls)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("controls are not authenticated") from error
    if not isinstance(actual, bytes) or not hmac.compare_digest(actual, expected):
        raise ValueError("controls are not authenticated")


def _snapshot_authenticated_controls(
    controls: TeacherControls,
) -> _TeacherControlSnapshot:
    _require_authenticated_controls(controls)
    authentication_tag = controls._authentication_tag
    snapshot = _TeacherControlSnapshot(
        scene=controls.scene,
        frame_ids=_readonly(controls.frame_ids),
        window_weights=_readonly(controls.window_weights),
        window_masks=_readonly(controls.window_masks),
        expected_coverage_weights=_readonly(
            controls.expected_coverage_weights
        ),
        source_sha256=controls.source_sha256,
        checkpoint_sha256=controls.checkpoint_sha256,
        formal_label_sha256=controls.formal_label_sha256,
        teacher_reference_sha256=controls.teacher_reference_sha256,
    )
    if not hmac.compare_digest(
        authentication_tag, _control_authentication_tag(snapshot)
    ):
        raise ValueError("controls changed while creating the authenticated snapshot")
    _require_authenticated_controls(controls)
    return snapshot


def _mint_teacher_controls(
    *,
    scene: str,
    frame_ids: np.ndarray,
    window_weights: np.ndarray,
    window_masks: np.ndarray,
    expected_coverage_weights: np.ndarray,
    source_sha256: str,
    checkpoint_sha256: str,
    formal_label_sha256: str,
    teacher_reference_sha256: str,
) -> TeacherControls:
    controls = object.__new__(TeacherControls)
    for name, value in (
        ("scene", scene),
        ("frame_ids", frame_ids),
        ("window_weights", window_weights),
        ("window_masks", window_masks),
        ("expected_coverage_weights", expected_coverage_weights),
        ("source_sha256", source_sha256),
        ("checkpoint_sha256", checkpoint_sha256),
        ("formal_label_sha256", formal_label_sha256),
        ("teacher_reference_sha256", teacher_reference_sha256),
    ):
        object.__setattr__(controls, name, value)
    controls._validate_minted()
    object.__setattr__(controls, "_authentication_tag", _control_authentication_tag(controls))
    return controls


@dataclass(frozen=True)
class RawGaugeTeacherSet:
    """Four center-only teachers expressed in the decoded long-prediction gauge."""

    baseline_pose_encoding: np.ndarray
    baseline_c2w: np.ndarray
    aligned_short_centers: np.ndarray
    coverage_weights: np.ndarray
    coverage_mask: np.ndarray
    raw_teacher_centers: np.ndarray
    filled_teacher_centers: np.ndarray
    translation_endpoints: np.ndarray
    prediction_scale: float

    def __post_init__(self) -> None:
        specifications = (
            ("baseline_pose_encoding", (_FRAMES, 9), np.float32),
            ("baseline_c2w", (_FRAMES, 4, 4), np.float64),
            ("aligned_short_centers", (_WINDOWS, _WINDOW_FRAMES, 3), np.float64),
            ("coverage_weights", (_ENDPOINTS, _FRAMES), np.float64),
            ("coverage_mask", (_ENDPOINTS, _FRAMES), np.uint8),
            ("raw_teacher_centers", (_ENDPOINTS, _FRAMES, 3), np.float64),
            ("filled_teacher_centers", (_ENDPOINTS, _FRAMES, 3), np.float64),
            ("translation_endpoints", (_ENDPOINTS, _FRAMES, 3), np.float32),
        )
        for name, shape, dtype in specifications:
            value = _exact_array(
                getattr(self, name), name=name, shape=shape, dtype=dtype
            )
            object.__setattr__(self, name, value)
        if not isinstance(self.prediction_scale, (float, np.floating)):
            raise ValueError("prediction_scale must be a float")
        if not np.isfinite(self.prediction_scale) or self.prediction_scale <= 0.0:
            raise ValueError("prediction_scale must be finite and positive")


def _load_reference(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    source = _authenticate_file(path, expected_sha256, label="teacher reference")
    _inspect_npz(source, frozenset(TEACHER_ARTIFACT_MEMBERS), label="teacher reference")
    try:
        return load_teacher_artifact(source)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"invalid authenticated teacher reference: {source}") from error


def load_teacher_controls(
    reference_path: Path,
    *,
    expected_sha256: str,
    expected_source_sha256: str,
    expected_checkpoint_sha256: str,
    expected_formal_label_sha256: str,
) -> TeacherControls:
    """Authenticate a legacy sidecar and extract only registered fusion controls."""
    source_digest = _canonical_digest(
        expected_source_sha256, name="expected source digest"
    )
    checkpoint_digest = _canonical_digest(
        expected_checkpoint_sha256, name="expected checkpoint digest"
    )
    formal_digest = _canonical_digest(
        expected_formal_label_sha256, name="expected formal-label digest"
    )
    reference_digest = _canonical_digest(
        expected_sha256, name="expected teacher-reference digest"
    )
    arrays = _load_reference(Path(reference_path), reference_digest)
    canonical_fields = {
        "scene": ((), np.dtype("U32")),
        "frame_ids": ((_FRAMES,), np.dtype(np.int64)),
        "window_weights": ((_WINDOWS,), np.dtype(np.float64)),
        "window_masks": ((_ENDPOINTS, _WINDOWS), np.dtype(np.uint8)),
        "coverage_weights": ((_ENDPOINTS, _FRAMES), np.dtype(np.float64)),
        "source_sha256": ((), np.dtype("U64")),
        "checkpoint_sha256": ((), np.dtype("U64")),
        "formal_label_sha256": ((), np.dtype("U64")),
    }
    for name, (shape, dtype) in canonical_fields.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"teacher reference {name} must use exact canonical shape and dtype"
            )
    bindings = {
        "source_sha256": source_digest,
        "checkpoint_sha256": checkpoint_digest,
        "formal_label_sha256": formal_digest,
    }
    for name, expected in bindings.items():
        if str(arrays[name]) != expected:
            raise ValueError(f"teacher reference {name} binding mismatch")
    return _mint_teacher_controls(
        scene=str(arrays["scene"]),
        frame_ids=arrays["frame_ids"],
        window_weights=arrays["window_weights"],
        window_masks=arrays["window_masks"],
        expected_coverage_weights=arrays["coverage_weights"],
        source_sha256=source_digest,
        checkpoint_sha256=checkpoint_digest,
        formal_label_sha256=formal_digest,
        teacher_reference_sha256=reference_digest,
    )


def _load_source(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    source = _authenticate_file(path, expected_sha256, label="source shard")
    _inspect_npz(source, _SOURCE_MEMBERS, label="source shard")
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in _SOURCE_MEMBERS}
    except (OSError, KeyError, EOFError, ValueError) as error:
        raise ValueError(f"invalid authenticated source shard: {source}") from error
    validate_source_shard(arrays)
    exact = {
        "global_frame_ids": ((500,), np.int64),
        "global_camera_tokens": ((500, 2048), np.float32),
        "short_frame_ids": ((9, 100), np.int64),
        "short_camera_tokens": ((9, 100, 2048), np.float32),
        "overlap_frame_ids": ((8, 50), np.int64),
        "overlap_long_tokens": ((8, 50, 2048), np.float32),
        "overlap_left_tokens": ((8, 50, 2048), np.float32),
        "overlap_right_tokens": ((8, 50, 2048), np.float32),
        "span_starts": ((8,), np.int64),
        "sample_ids": ((8,), np.dtype("U64")),
        "global_pred_c2w": ((500, 4, 4), np.float64),
        "overlap_long_c2w": ((8, 50, 4, 4), np.float64),
    }
    for name, (shape, dtype) in exact.items():
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"source shard {name} has a noncanonical dtype or shape")
    _validate_baseline_c2w(arrays["global_pred_c2w"])
    overlap = arrays["overlap_long_c2w"]
    _validate_pose_stack(overlap.reshape(-1, 4, 4), name="overlap_long_c2w")
    return arrays


def _scene_from_source(source: Mapping[str, np.ndarray]) -> str:
    sample_ids = source["sample_ids"]
    first = str(sample_ids[0])
    scene, separator, _ = first.partition(":")
    if separator != ":" or _SCENE_RE.fullmatch(scene) is None:
        raise ValueError("source sample IDs must bind one canonical scene")
    expected = np.asarray(
        [f"{scene}:overlap_{index:03d}" for index in range(8)], dtype="U64"
    )
    if not np.array_equal(sample_ids, expected):
        raise ValueError("source sample IDs must bind exactly eight ordered overlaps")
    return scene


def _validate_pose_stack(poses: np.ndarray, *, name: str) -> None:
    if (
        not isinstance(poses, np.ndarray)
        or poses.ndim != 3
        or poses.shape[-2:] != (4, 4)
        or poses.dtype != np.float64
        or not np.isfinite(poses).all()
    ):
        raise ValueError(f"{name} must be a finite float64 pose stack")
    if not np.allclose(
        poses[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=_HOMOGENEOUS_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must contain homogeneous poses")
    _validate_so3(poses[:, :3, :3], name=name, atol=_SO3_ATOL)


@contextmanager
def _frozen_camera_head(camera_head: nn.Module, device: torch.device):
    if not isinstance(camera_head, nn.Module):
        raise ValueError("camera_head must be an nn.Module")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested camera-head device is unavailable")
    tensors = list(camera_head.parameters()) + list(camera_head.buffers())
    for tensor in tensors:
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise ValueError("camera head tensors must be on the requested device")
    snapshots = [(tensor, tensor.detach().clone()) for tensor in tensors]
    modes = [(module, module.training) for module in camera_head.modules()]
    try:
        camera_head.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for tensor, snapshot in snapshots:
                tensor.copy_(snapshot)
        for module, training in modes:
            module.training = training


def _decode(
    camera_head: nn.Module, tokens: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    token_tensor = torch.from_numpy(tokens).to(device=device)
    raw = decode_camera_tokens(camera_head, token_tensor)
    c2w = pose_encoding_to_c2w(raw)
    raw_array = raw.detach().to(device="cpu", dtype=torch.float32).numpy()
    c2w_array = c2w.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(raw_array).all() or not np.isfinite(c2w_array).all():
        raise ValueError("Camera Head produced non-finite outputs")
    return raw_array, c2w_array


def _authenticate_baseline(decoded: np.ndarray, witness: np.ndarray) -> None:
    scale = prediction_scale(decoded)
    center_error = float(
        np.max(np.linalg.norm(decoded[:, :3, 3] - witness[:, :3, 3], axis=1))
        / scale
    )
    rotation_error = float(
        np.max(np.linalg.norm(decoded[:, :3, :3] - witness[:, :3, :3], axis=(1, 2)))
    )
    if (
        not np.isfinite(center_error)
        or center_error > _BASELINE_CENTER_NORMALIZED_ATOL
        or not np.isfinite(rotation_error)
        or rotation_error > _BASELINE_ROTATION_ATOL
    ):
        raise ValueError("decoded baseline does not match authenticated baseline witness")


def build_raw_gauge_teacher(
    source_path: Path,
    controls: TeacherControls,
    camera_head: nn.Module,
    *,
    expected_source_sha256: str,
    checkpoint_sha256: str,
    device: torch.device,
) -> RawGaugeTeacherSet:
    """Decode, align, and fuse using predictions plus authenticated weights/masks only."""
    if not isinstance(controls, TeacherControls):
        raise ValueError("controls must be authenticated TeacherControls")
    control_snapshot = _snapshot_authenticated_controls(controls)
    source_digest = _canonical_digest(
        expected_source_sha256, name="expected source digest"
    )
    checkpoint_digest = _canonical_digest(
        checkpoint_sha256, name="checkpoint_sha256"
    )
    if control_snapshot.source_sha256 != source_digest:
        raise ValueError("controls/source digest mismatch")
    if control_snapshot.checkpoint_sha256 != checkpoint_digest:
        raise ValueError("controls/checkpoint digest mismatch")
    source = _load_source(Path(source_path), source_digest)
    scene = _scene_from_source(source)
    if scene != control_snapshot.scene:
        raise ValueError("controls/source scene mismatch")
    if not np.array_equal(source["global_frame_ids"], control_snapshot.frame_ids):
        raise ValueError("controls/source frame IDs mismatch")

    with _frozen_camera_head(camera_head, device):
        long_raw, decoded_long_batch = _decode(
            camera_head, source["global_camera_tokens"][None], device
        )
    _require_authenticated_controls(controls)
    with _frozen_camera_head(camera_head, device):
        _, decoded_short = _decode(camera_head, source["short_camera_tokens"], device)
    _require_authenticated_controls(controls)
    baseline_pose_encoding = long_raw[0]
    decoded_long = decoded_long_batch[0]
    _validate_baseline_c2w(decoded_long)
    _validate_pose_stack(decoded_short.reshape(-1, 4, 4), name="decoded short poses")
    _authenticate_baseline(decoded_long, source["global_pred_c2w"])
    scale = prediction_scale(decoded_long)

    aligned_centers = np.full(
        (_WINDOWS, _WINDOW_FRAMES, 3), np.nan, dtype=np.float64
    )
    for window, start in enumerate(range(0, 401, 50)):
        alignment = align_local_to_global(
            decoded_long[start : start + _WINDOW_FRAMES],
            decoded_short[window],
            scene_scale=scale,
        )
        if not alignment.valid or alignment.aligned_c2w is None:
            unused = (
                control_snapshot.window_weights[window] == 0.0
                and not np.any(control_snapshot.window_masks[:, window])
            )
            if unused:
                continue
            reason = alignment.exclusion_reason or "unknown alignment failure"
            raise ValueError(f"selected short window {window} failed alignment: {reason}")
        _validate_pose_stack(alignment.aligned_c2w, name=f"aligned short window {window}")
        aligned_centers[window] = alignment.aligned_c2w[:, :3, 3]

    coverage = np.zeros((_ENDPOINTS, _FRAMES), dtype=np.float64)
    numerator = np.zeros((_ENDPOINTS, _FRAMES, 3), dtype=np.float64)
    for endpoint in range(_ENDPOINTS):
        for window in range(_WINDOWS):
            if control_snapshot.window_masks[endpoint, window] == 0:
                continue
            weight = control_snapshot.window_weights[window]
            if weight <= 0.0:
                raise AssertionError("validated selected window lost positive weight")
            centers = aligned_centers[window]
            if not np.isfinite(centers).all():
                raise ValueError(f"selected short window {window} has invalid aligned centers")
            start = 50 * window
            stop = start + _WINDOW_FRAMES
            coverage[endpoint, start:stop] += weight
            numerator[endpoint, start:stop] += weight * centers
    if not np.array_equal(
        coverage, control_snapshot.expected_coverage_weights
    ):
        raise ValueError("numeric coverage does not exactly replay legacy coverage")
    coverage_mask = (coverage > 0.0).astype(np.uint8)
    raw_centers = np.full((_ENDPOINTS, _FRAMES, 3), np.nan, dtype=np.float64)
    covered = coverage_mask != 0
    raw_centers[covered] = numerator[covered] / coverage[covered, None]
    if np.any(covered) and not np.isfinite(raw_centers[covered]).all():
        raise ValueError("covered fused teacher centers are non-finite")

    endpoints, filled, endpoint_scale = build_translation_endpoint(
        long_frame_ids=source["global_frame_ids"],
        teacher_frame_ids=control_snapshot.frame_ids,
        baseline_c2w=decoded_long,
        baseline_pose_encoding=baseline_pose_encoding,
        teacher_centers=raw_centers,
        coverage_mask=coverage_mask,
    )
    if np.asarray(endpoint_scale, dtype=np.float64).tobytes() != np.asarray(
        scale, dtype=np.float64
    ).tobytes():
        raise AssertionError("prediction-scale replay changed during endpoint construction")
    result = RawGaugeTeacherSet(
        baseline_pose_encoding=baseline_pose_encoding,
        baseline_c2w=decoded_long,
        aligned_short_centers=aligned_centers,
        coverage_weights=coverage,
        coverage_mask=coverage_mask,
        raw_teacher_centers=raw_centers,
        filled_teacher_centers=filled,
        translation_endpoints=endpoints,
        prediction_scale=float(scale),
    )
    _require_authenticated_controls(controls)
    return result


def _oracle_from_reference(arrays: Mapping[str, np.ndarray]) -> FrozenOracle:
    return FrozenOracle(
        scene=str(arrays["oracle_scene"]),
        frame_digest=str(arrays["oracle_frame_digest"]),
        fit_count=int(arrays["oracle_fit_count"]),
        scale=float(arrays["oracle_scale"]),
        rotation=tuple(
            tuple(float(component) for component in row)
            for row in arrays["oracle_rotation"]
        ),
        translation=tuple(float(component) for component in arrays["oracle_translation"]),
        rank=int(arrays["oracle_rank"]),
        condition=float(arrays["oracle_condition"]),
        transform_digest=str(arrays["oracle_digest"]),
    )


def verify_legacy_teacher_witness(
    teacher: RawGaugeTeacherSet,
    reference_path: Path,
    *,
    expected_sha256: str,
) -> None:
    """Forward-apply the registered oracle solely to authenticate legacy centers."""
    if not isinstance(teacher, RawGaugeTeacherSet):
        raise ValueError("teacher must be a RawGaugeTeacherSet")
    arrays = _load_reference(Path(reference_path), expected_sha256)
    if not np.array_equal(
        teacher.coverage_weights, np.asarray(arrays["coverage_weights"], dtype=np.float64)
    ):
        raise ValueError("legacy witness coverage does not match numeric teacher")
    _authenticate_baseline(
        teacher.baseline_c2w,
        np.asarray(arrays["baseline_c2w_raw"], dtype=np.float64),
    )
    oracle = _oracle_from_reference(arrays)
    fused = np.asarray(arrays["fused_c2w"], dtype=np.float64)
    normalizer = float(arrays["gt_scene_scale"])
    for endpoint in range(_ENDPOINTS):
        covered = teacher.coverage_mask[endpoint] != 0
        if not np.any(covered):
            continue
        raw = teacher.baseline_c2w[covered].copy()
        raw[:, :3, 3] = teacher.raw_teacher_centers[endpoint, covered]
        forward = apply_frozen_oracle(oracle, raw)
        error = float(
            np.max(
                np.linalg.norm(
                    forward[:, :3, 3] - fused[endpoint, covered, :3, 3], axis=1
                )
            )
            / normalizer
        )
        if not np.isfinite(error) or error > _WITNESS_CENTER_NORMALIZED_ATOL:
            raise ValueError("legacy forward oracle witness does not match raw teacher centers")


__all__ = [
    "RawGaugeTeacherSet",
    "TeacherControls",
    "build_raw_gauge_teacher",
    "load_teacher_controls",
    "verify_legacy_teacher_witness",
]
