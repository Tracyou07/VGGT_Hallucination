"""Selected-frame ScanNet extraction and registered RGB-D preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import zlib

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.contracts import (
    canonical_json_digest,
)
from pre_experiments.camera_velocity_ambiguity_02.frames import (
    FrameSelection,
    build_fastvggt_frame_selection,
)
from pre_experiments.camera_velocity_ambiguity_02.rgbd_gate import (
    RGBDObservations,
    build_rgbd_observations,
)
PREPARED_SCHEMA = "camera_velocity_ambiguity_02.prepared_scene.v1"
DEFAULT_OBSERVATION_HW = (120, 160)


@dataclass(frozen=True)
class PreparedScene:
    scene: str
    root: Path
    selection: FrameSelection
    raw_gt_c2w: np.ndarray
    observation_intrinsics: np.ndarray
    manifest_digest: str


def select_sensor_frame_ids(
    finite_pose_mask: np.ndarray, *, input_frames: int
) -> tuple[int, ...]:
    """Apply the exact FastVGGT frame rule to finite sensor-frame indices."""
    mask = np.asarray(finite_pose_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("finite_pose_mask must be a one-dimensional boolean array")
    finite_ids = tuple(int(value) for value in np.flatnonzero(mask))
    pseudo_images = tuple(Path(f"{value}.jpg") for value in finite_ids)
    selection = build_fastvggt_frame_selection(
        pseudo_images,
        finite_ids,
        input_frames=input_frames,
    )
    return selection.frame_ids


def _matrix4(name: str, value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return matrix


def observation_calibration(
    *,
    color_intrinsic: np.ndarray,
    depth_intrinsic: np.ndarray,
    color_extrinsic: np.ndarray,
    depth_extrinsic: np.ndarray,
    color_hw: tuple[int, int],
    depth_hw: tuple[int, int],
    observation_hw: tuple[int, int],
) -> np.ndarray:
    """Return registered depth intrinsics scaled to the observation raster."""
    color_k = _matrix4("color_intrinsic", color_intrinsic)
    depth_k = _matrix4("depth_intrinsic", depth_intrinsic)
    color_e = _matrix4("color_extrinsic", color_extrinsic)
    depth_e = _matrix4("depth_extrinsic", depth_extrinsic)
    if not np.allclose(color_e, depth_e, atol=1e-6, rtol=0):
        raise ValueError("color and depth streams are not registered")
    ch, cw = color_hw
    dh, dw = depth_hw
    oh, ow = observation_hw
    if min(ch, cw, dh, dw, oh, ow) < 1:
        raise ValueError("image dimensions must be positive")
    color_at_depth = color_k[:3, :3].copy()
    color_at_depth[0] *= dw / cw
    color_at_depth[1] *= dh / ch
    # ScanNet-50 contains two authenticated calibration profiles.  The second
    # differs by 0.413% in the resized vertical focal length because 968 color
    # rows map to a nominal 480-row registered depth raster.
    if not np.allclose(color_at_depth, depth_k[:3, :3], atol=0.5, rtol=5e-3):
        raise ValueError("color/depth intrinsics do not describe registered rasters")
    result = depth_k[:3, :3].copy()
    result[0] *= ow / dw
    result[1] *= oh / dh
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _decode_depth(sensor: Any, handle, frame_index: int) -> np.ndarray:
    if sensor.depth_compression_code != 1:
        raise ValueError("CVA02 requires ScanNet zlib_ushort depth compression")
    frame = sensor.frames[frame_index]
    handle.seek(frame.depth_offset)
    compressed = handle.read(frame.depth_size_bytes)
    if len(compressed) != frame.depth_size_bytes:
        raise EOFError("unexpected end of ScanNet depth payload")
    payload = zlib.decompress(compressed)
    expected = sensor.depth_height * sensor.depth_width
    depth_mm = np.frombuffer(payload, dtype="<u2")
    if depth_mm.size != expected:
        raise ValueError("decoded ScanNet depth has a wrong element count")
    return depth_mm.reshape(sensor.depth_height, sensor.depth_width)


def _manifest_path(root: Path) -> Path:
    return root / "prepared.json"


def _read_prepared(root: Path, expected: dict[str, object]) -> PreparedScene | None:
    path = _manifest_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    digest = payload.get("manifest_digest") if isinstance(payload, dict) else None
    unsigned = {key: value for key, value in payload.items() if key != "manifest_digest"} if isinstance(payload, dict) else {}
    if unsigned != expected or digest != canonical_json_digest(unsigned):
        return None
    frame_ids = tuple(int(value) for value in expected["frame_ids"])
    required = [
        *(root / "color" / f"{value}.jpg" for value in frame_ids),
        *(root / "rgbd_rgb" / f"{value}.jpg" for value in frame_ids),
        *(root / "depth" / f"{value}.npy" for value in frame_ids),
        *(root / "pose" / f"{value}.npy" for value in frame_ids),
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        return None
    selection = FrameSelection(
        frame_ids=frame_ids,
        image_paths=tuple(root / "color" / f"{value}.jpg" for value in frame_ids),
        pose_indices=tuple(range(len(frame_ids))),
    )
    gt = np.stack([np.load(root / "pose" / f"{value}.npy", allow_pickle=False) for value in frame_ids])
    return PreparedScene(
        scene=str(expected["scene"]),
        root=root,
        selection=selection,
        raw_gt_c2w=gt,
        observation_intrinsics=np.asarray(expected["observation_intrinsics"], dtype=np.float64),
        manifest_digest=str(digest),
    )


def prepare_selected_scene(
    sens_path: Path,
    output_dir: Path,
    *,
    scene: str,
    input_frames: int,
    sens_sha256: str,
    expected_bytes: int,
    observation_hw: tuple[int, int] = DEFAULT_OBSERVATION_HW,
) -> PreparedScene:
    """Extract only selected frames and publish a resumable scene manifest."""
    import cv2
    from scripts.autodl.scannet.sensreader_py3.SensorData import SensorData

    source = Path(sens_path)
    root = Path(output_dir)
    if not source.is_file() or source.stat().st_size != expected_bytes:
        raise ValueError("verified ScanNet .sens size changed before extraction")
    sensor = SensorData(str(source))
    finite = np.asarray(
        [np.isfinite(frame.camera_to_world).all() for frame in sensor.frames],
        dtype=np.bool_,
    )
    frame_ids = select_sensor_frame_ids(finite, input_frames=input_frames)
    intrinsics = observation_calibration(
        color_intrinsic=sensor.intrinsic_color,
        depth_intrinsic=sensor.intrinsic_depth,
        color_extrinsic=sensor.extrinsic_color,
        depth_extrinsic=sensor.extrinsic_depth,
        color_hw=(sensor.color_height, sensor.color_width),
        depth_hw=(sensor.depth_height, sensor.depth_width),
        observation_hw=observation_hw,
    )
    expected = {
        "schema": PREPARED_SCHEMA,
        "scene": scene,
        "sens_path": str(source.resolve()),
        "sens_sha256": sens_sha256,
        "sens_bytes": expected_bytes,
        "frame_ids": list(frame_ids),
        "observation_hw": list(observation_hw),
        "observation_intrinsics": intrinsics.tolist(),
        "depth_shift": float(sensor.depth_shift),
    }
    resumed = _read_prepared(root, expected)
    if resumed is not None:
        return resumed

    oh, ow = observation_hw
    with source.open("rb") as handle:
        for frame_id in frame_ids:
            frame = sensor.frames[frame_id]
            color = sensor._read_color(handle, frame)
            ok, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise OSError("failed to encode selected ScanNet color frame")
            _atomic_bytes(root / "color" / f"{frame_id}.jpg", encoded.tobytes())

            registered_rgb = cv2.resize(color, (ow, oh), interpolation=cv2.INTER_AREA)
            ok, encoded_small = cv2.imencode(
                ".jpg", registered_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not ok:
                raise OSError("failed to encode RGB-D observation frame")
            _atomic_bytes(root / "rgbd_rgb" / f"{frame_id}.jpg", encoded_small.tobytes())

            depth_mm = _decode_depth(sensor, handle, frame_id)
            depth_small = cv2.resize(depth_mm, (ow, oh), interpolation=cv2.INTER_NEAREST)
            depth_m = depth_small.astype(np.float32) / float(sensor.depth_shift)
            _atomic_array(root / "depth" / f"{frame_id}.npy", depth_m)
            _atomic_array(
                root / "pose" / f"{frame_id}.npy",
                np.asarray(frame.camera_to_world, dtype=np.float64),
            )

    payload = {**expected, "manifest_digest": canonical_json_digest(expected)}
    _atomic_bytes(
        _manifest_path(root),
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    result = _read_prepared(root, expected)
    if result is None:
        raise RuntimeError("prepared scene failed its completion validation")
    return result


def load_rgbd_observations(prepared: PreparedScene) -> RGBDObservations:
    """Load selected observation rasters with no GT pose fields."""
    import cv2

    rgb = []
    depth = []
    for frame_id in prepared.selection.frame_ids:
        image = cv2.imread(str(prepared.root / "rgbd_rgb" / f"{frame_id}.jpg"))
        if image is None:
            raise ValueError("prepared RGB-D color frame is unreadable")
        rgb.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0)
        depth.append(np.load(prepared.root / "depth" / f"{frame_id}.npy", allow_pickle=False))
    return build_rgbd_observations(
        {
            "frame_ids": np.asarray(prepared.selection.frame_ids, dtype=np.int64),
            "rgb": np.stack(rgb),
            "depth": np.stack(depth),
            "intrinsics": prepared.observation_intrinsics,
        }
    )
