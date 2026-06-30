"""Evaluate VGGT depth, camera pose, and point outputs on ScanNet scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from experiments.scannet_hallucination.resume import load_completed_metrics, selection_out_dir

try:
    from plyfile import PlyData
except Exception:  # pragma: no cover - reported at runtime if GT PLY is used.
    PlyData = None

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:  # pragma: no cover - torch checkpoints still work.
    load_safetensors = None


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_scene_list(path: Path, limit: int) -> list[str]:
    scenes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            scenes.append(item)
    return scenes[:limit] if limit > 0 else scenes


def frame_id(path: Path) -> int:
    return int(path.stem)


def get_sorted_image_paths(color_dir: Path) -> list[Path]:
    paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(color_dir.glob(suffix))
    return sorted(paths, key=frame_id)


def load_poses(pose_dir: Path) -> dict[int, np.ndarray]:
    poses = {}
    for path in sorted(pose_dir.glob("*.txt"), key=frame_id):
        try:
            pose = np.loadtxt(path, dtype=np.float64)
        except Exception:
            continue
        if pose.shape != (4, 4):
            continue
        if not np.isfinite(pose).all():
            continue
        poses[frame_id(path)] = pose
    return poses


def linspace_indices(total: int, count: int) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=np.int64)
    if count >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def regime_step_ids(valid_ids: list[int], count: int) -> list[int]:
    if len(valid_ids) <= count:
        return valid_ids
    first = valid_ids[0]
    remaining = valid_ids[1:]
    step = max(1, len(remaining) // (count - 1))
    return [first] + remaining[::step][: count - 1]


def make_frame_selections(
    valid_ids: list[int], frame_counts: list[int], sampling: str
) -> dict[int, list[int]]:
    selections: dict[int, list[int]] = {}
    max_count = max(frame_counts)
    if sampling == "nested_uniform":
        base = [valid_ids[i] for i in linspace_indices(len(valid_ids), max_count)]
        for count in frame_counts:
            selections[count] = [base[i] for i in linspace_indices(len(base), count)]
        return selections

    for count in frame_counts:
        if sampling == "prefix":
            selections[count] = valid_ids[:count]
        elif sampling == "uniform":
            selections[count] = [valid_ids[i] for i in linspace_indices(len(valid_ids), count)]
        elif sampling == "regime_step":
            selections[count] = regime_step_ids(valid_ids, count)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling}")
    return selections


def load_model(weights: str, ckpt_dir: Path, eval_native_points: bool) -> VGGT:
    if weights == "hub":
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        if not eval_native_points:
            model.point_head = None
        model.track_head = None
        return model

    model = VGGT(enable_track=False, enable_point=eval_native_points)
    if weights == "random":
        return model

    safetensors_path = ckpt_dir / "model.safetensors"
    torch_path = ckpt_dir / "model.pt"
    if safetensors_path.exists():
        if load_safetensors is None:
            raise RuntimeError("safetensors is not installed.")
        state_dict = load_safetensors(str(safetensors_path), device="cpu")
    elif torch_path.exists():
        state_dict = torch.load(torch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or model.pt found in {ckpt_dir}")
    model.load_state_dict(state_dict, strict=False)
    return model


def to_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    mats = np.tile(np.eye(4, dtype=np.float64), (extrinsic.shape[0], 1, 1))
    mats[:, :3, :4] = extrinsic
    return mats


def invert_poses(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(poses)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / len(src)
    u, s, vh = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vh
    var_src = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.sum(s * d) / max(var_src, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def evaluate_pose(pred_w2c: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    pred_c2w = invert_poses(pred_w2c)
    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    scale, rot_align, trans = umeyama(pred_centers, gt_centers)
    aligned_centers = scale * (pred_centers @ rot_align.T) + trans
    errors = np.linalg.norm(aligned_centers - gt_centers, axis=1)

    rot_errors = []
    for pred_pose, gt_pose in zip(pred_c2w, gt_c2w):
        aligned_rot = rot_align @ pred_pose[:3, :3]
        rot_errors.append(rotation_angle_deg(aligned_rot.T @ gt_pose[:3, :3]))

    rpe_rot = []
    rpe_trans = []
    for idx in range(1, len(gt_c2w)):
        pred_rel = np.linalg.inv(pred_c2w[idx - 1]) @ pred_c2w[idx]
        gt_rel = np.linalg.inv(gt_c2w[idx - 1]) @ gt_c2w[idx]
        rpe_rot.append(rotation_angle_deg(pred_rel[:3, :3].T @ gt_rel[:3, :3]))
        rpe_trans.append(abs(scale * np.linalg.norm(pred_rel[:3, 3]) - np.linalg.norm(gt_rel[:3, 3])))

    return {
        "pose_ate_rmse_aligned": float(np.sqrt(np.mean(errors**2))),
        "pose_ate_mean_aligned": float(np.mean(errors)),
        "pose_ate_max_aligned": float(np.max(errors)),
        "pose_are_mean_deg_aligned": float(np.mean(rot_errors)),
        "pose_are_max_deg_aligned": float(np.max(rot_errors)),
        "pose_rpe_rot_mean_deg": float(np.mean(rpe_rot)) if rpe_rot else 0.0,
        "pose_rpe_trans_mean_aligned": float(np.mean(rpe_trans)) if rpe_trans else 0.0,
        "pose_sim3_scale": float(scale),
    }


def resize_nearest(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32))
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)
    return resized > 0


def transform_intrinsic_for_resize(
    intrinsic: np.ndarray | None,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray | None:
    if intrinsic is None:
        return None
    output = intrinsic.copy().astype(np.float64)
    output[0, 0] *= scale_x
    output[1, 1] *= scale_y
    output[0, 2] = output[0, 2] * scale_x + offset_x
    output[1, 2] = output[1, 2] * scale_y + offset_y
    return output


def preprocess_depth_and_intrinsic(
    depth_path: Path,
    intrinsic: np.ndarray | None,
    mode: str,
    target_size: int,
    expected_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    valid = raw > 0
    depth = raw / 1000.0
    height, width = depth.shape

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        scale_x = new_width / width
        scale_y = new_height / height
        resized = resize_nearest(depth, new_width, new_height)
        resized_valid = resize_mask(valid, new_width, new_height)
        out = np.full((target_size, target_size), np.nan, dtype=np.float32)
        pad_top = (target_size - new_height) // 2
        pad_left = (target_size - new_width) // 2
        out[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
        valid_out = np.zeros((target_size, target_size), dtype=bool)
        valid_out[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized_valid
        out[~valid_out] = np.nan
        intr = transform_intrinsic_for_resize(intrinsic, scale_x, scale_y, pad_left, pad_top)
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        scale_x = new_width / width
        scale_y = new_height / height
        out = resize_nearest(depth, new_width, new_height)
        valid_out = resize_mask(valid, new_width, new_height)
        offset_y = 0
        if new_height > target_size:
            offset_y = -((new_height - target_size) // 2)
            start = -offset_y
            out = out[start : start + target_size, :]
            valid_out = valid_out[start : start + target_size, :]
        out[~valid_out] = np.nan
        intr = transform_intrinsic_for_resize(intrinsic, scale_x, scale_y, 0, offset_y)

    if out.shape != expected_hw:
        target_h, target_w = expected_hw
        old_h, old_w = out.shape
        valid_shape = np.isfinite(out)
        out = resize_nearest(np.nan_to_num(out, nan=0.0), target_w, target_h)
        valid_shape = resize_mask(valid_shape, target_w, target_h)
        out[~valid_shape] = np.nan
        if intr is not None:
            intr = transform_intrinsic_for_resize(intr, target_w / old_w, target_h / old_h, 0, 0)
    return out, intr


def load_intrinsic(scene_dir: Path) -> np.ndarray | None:
    intrinsic_dir = scene_dir / "intrinsic"
    for name in ("intrinsic_depth.txt", "intrinsic_color.txt"):
        path = intrinsic_dir / name
        if path.exists():
            matrix = np.loadtxt(path, dtype=np.float64)
            return matrix[:3, :3]
    return None


def load_gt_depths(
    scene_dir: Path,
    frame_ids: list[int],
    mode: str,
    target_size: int,
    expected_hw: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    depth_dir = scene_dir / "depth"
    if not depth_dir.exists():
        return None, None
    base_intrinsic = load_intrinsic(scene_dir)
    depths = []
    intrinsics = []
    for fid in frame_ids:
        path = depth_dir / f"{fid}.png"
        if not path.exists():
            return None, None
        depth, intrinsic = preprocess_depth_and_intrinsic(
            path, base_intrinsic, mode, target_size, expected_hw
        )
        depths.append(depth)
        if intrinsic is not None:
            intrinsics.append(intrinsic)
    depth_array = np.stack(depths, axis=0).astype(np.float32)
    intrinsic_array = np.stack(intrinsics, axis=0).astype(np.float64) if intrinsics else None
    return depth_array, intrinsic_array


def depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray | None) -> dict[str, float]:
    if gt_depth is None:
        return {"depth_valid_frames": 0.0}
    rows = []
    for pred, gt in zip(pred_depth, gt_depth):
        pred_2d = pred.squeeze()
        valid = np.isfinite(pred_2d) & np.isfinite(gt) & (pred_2d > 1e-6) & (gt > 1e-6)
        if valid.sum() < 100:
            continue
        pred_v = pred_2d[valid]
        gt_v = gt[valid]
        raw_absrel = np.mean(np.abs(pred_v - gt_v) / gt_v)
        raw_rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))
        scale = float(np.median(gt_v) / max(np.median(pred_v), 1e-6))
        aligned = pred_v * scale
        absrel = np.mean(np.abs(aligned - gt_v) / gt_v)
        rmse = np.sqrt(np.mean((aligned - gt_v) ** 2))
        ratio = np.maximum(aligned / gt_v, gt_v / np.maximum(aligned, 1e-6))
        rows.append((raw_absrel, raw_rmse, absrel, rmse, np.mean(ratio < 1.25), scale))
    if not rows:
        return {"depth_valid_frames": 0.0}
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "depth_absrel_raw": float(arr[:, 0].mean()),
        "depth_rmse_raw": float(arr[:, 1].mean()),
        "depth_absrel_aligned": float(arr[:, 2].mean()),
        "depth_rmse_aligned": float(arr[:, 3].mean()),
        "depth_delta1_aligned": float(arr[:, 4].mean()),
        "depth_median_scale": float(arr[:, 5].mean()),
        "depth_valid_frames": float(len(rows)),
    }


def load_gt_ply(scene: str, gt_ply_dir: Path, max_points: int, seed: int) -> np.ndarray | None:
    if PlyData is None:
        raise RuntimeError("plyfile is not installed.")
    path = gt_ply_dir / scene / f"{scene}_vh_clean_2.ply"
    if not path.exists():
        return None
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    return sample_points(points, max_points, np.random.default_rng(seed))


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points


def sample_grid_points(
    grid: np.ndarray,
    conf: np.ndarray | None,
    conf_thresh: float,
    points_per_frame: int,
    rng: np.random.Generator,
) -> np.ndarray:
    points = grid.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    if conf is not None:
        valid &= conf.reshape(-1) >= conf_thresh
    points = points[valid]
    if points.shape[0] > points_per_frame:
        idx = rng.choice(points.shape[0], size=points_per_frame, replace=False)
        points = points[idx]
    return points.astype(np.float32)


def collect_native_points(
    world_points: np.ndarray,
    world_conf: np.ndarray | None,
    conf_thresh: float,
    points_per_frame: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for idx in range(world_points.shape[0]):
        conf = world_conf[idx] if world_conf is not None else None
        chunks.append(sample_grid_points(world_points[idx], conf, conf_thresh, points_per_frame, rng))
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return sample_points(np.concatenate(chunks, axis=0), max_points, rng)


def collect_unprojected_points(
    depths: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_conf: np.ndarray | None,
    depth_conf_thresh: float,
    points_per_frame: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for idx in range(depths.shape[0]):
        depth = depths[idx].squeeze().copy()
        if depth_conf is not None:
            depth[depth_conf[idx] < depth_conf_thresh] = np.nan
        if not np.isfinite(depth).any():
            continue
        grid, _, _ = depth_to_world_coords_points(depth, extrinsics[idx], intrinsics[idx])
        chunks.append(sample_grid_points(grid, None, 0.0, points_per_frame, rng))
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return sample_points(np.concatenate(chunks, axis=0), max_points, rng)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (hom @ transform.T)[:, :3]


def bbox_scale_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    src_min, src_max = source.min(axis=0), source.max(axis=0)
    tgt_min, tgt_max = target.min(axis=0), target.max(axis=0)
    src_center = (src_min + src_max) / 2.0
    tgt_center = (tgt_min + tgt_max) / 2.0
    src_diag = np.linalg.norm(src_max - src_min)
    tgt_diag = np.linalg.norm(tgt_max - tgt_min)
    scale = float(tgt_diag / max(src_diag, 1e-8))
    return (source - src_center) * scale + tgt_center, scale


def chamfer_distance(pred: np.ndarray, gt: np.ndarray, max_dist: float) -> float:
    from scipy.spatial import cKDTree

    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return float("nan")
    pred_tree = cKDTree(pred)
    gt_tree = cKDTree(gt)
    d_pred, _ = gt_tree.query(pred, k=1, workers=-1)
    d_gt, _ = pred_tree.query(gt, k=1, workers=-1)
    return float(np.clip(d_pred, 0, max_dist).mean() + np.clip(d_gt, 0, max_dist).mean())


def point_metrics(
    name: str,
    points: np.ndarray,
    gt_points: np.ndarray | None,
    max_dist: float,
) -> dict[str, float]:
    prefix = f"pcd_{name}"
    if gt_points is None or points.shape[0] == 0:
        return {f"{prefix}_points": float(points.shape[0])}
    raw_cd = chamfer_distance(points, gt_points, max_dist)
    aligned, scale = bbox_scale_align(points, gt_points)
    aligned_cd = chamfer_distance(aligned, gt_points, max_dist)
    return {
        f"{prefix}_points": float(points.shape[0]),
        f"{prefix}_chamfer_raw": raw_cd,
        f"{prefix}_chamfer_aligned": aligned_cd,
        f"{prefix}_bbox_scale": float(scale),
    }


def write_json(path: Path, data: object) -> None:
    def clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2), encoding="utf-8")


def plot_trajectory(path: Path, pred_w2c: np.ndarray, gt_c2w: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt

        pred_c2w = invert_poses(pred_w2c)
        pred_centers = pred_c2w[:, :3, 3]
        gt_centers = gt_c2w[:, :3, 3]
        scale, rot, trans = umeyama(pred_centers, gt_centers)
        aligned = scale * (pred_centers @ rot.T) + trans
        plt.figure(figsize=(6, 5))
        plt.plot(gt_centers[:, 0], gt_centers[:, 2], label="gt", linewidth=2)
        plt.plot(aligned[:, 0], aligned[:, 2], label="pred aligned", linewidth=1)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as exc:
        print(f"[plot] warning: {exc}")


def run_inference(
    model: VGGT,
    image_paths: list[Path],
    device: torch.device,
    preprocess_mode: str,
) -> tuple[dict[str, np.ndarray], float, tuple[int, int]]:
    images = load_and_preprocess_images([str(p) for p in image_paths], mode=preprocess_mode)
    height, width = int(images.shape[2]), int(images.shape[3])
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predictions = model(images.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000.0

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], (height, width)
    )
    output = {
        "extrinsic": extrinsic[0].detach().float().cpu().numpy(),
        "intrinsic": intrinsic[0].detach().float().cpu().numpy(),
        "depth": predictions["depth"][0].detach().float().cpu().numpy(),
        "depth_conf": predictions["depth_conf"][0].detach().float().cpu().numpy(),
    }
    if "world_points" in predictions:
        output["world_points"] = predictions["world_points"][0].detach().float().cpu().numpy()
        output["world_points_conf"] = predictions["world_points_conf"][0].detach().float().cpu().numpy()
    del predictions, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, elapsed_ms, (height, width)


def evaluate_selection(
    args: argparse.Namespace,
    model: VGGT,
    scene: str,
    scene_dir: Path,
    selected_ids: list[int],
    image_by_id: dict[int, Path],
    poses_by_id: dict[int, np.ndarray],
    gt_points: np.ndarray | None,
    device: torch.device,
) -> dict[str, float | str]:
    image_paths = [image_by_id[fid] for fid in selected_ids]
    gt_c2w = np.stack([poses_by_id[fid] for fid in selected_ids], axis=0)
    output, inference_ms, image_hw = run_inference(model, image_paths, device, args.preprocess_mode)

    pred_w2c = to_homogeneous(output["extrinsic"])
    metrics: dict[str, float | str] = {
        "scene": scene,
        "frame_count_requested": float(len(selected_ids)),
        "frame_count_actual": float(len(selected_ids)),
        "inference_time_ms": float(inference_ms),
        "image_height": float(image_hw[0]),
        "image_width": float(image_hw[1]),
    }
    metrics.update(evaluate_pose(pred_w2c, gt_c2w))

    gt_depth, gt_intrinsic = load_gt_depths(
        scene_dir, selected_ids, args.preprocess_mode, args.target_size, image_hw
    )
    metrics.update(depth_metrics(output["depth"], gt_depth))

    first_gt_pose = gt_c2w[0]
    pred_cloud = collect_unprojected_points(
        output["depth"],
        output["extrinsic"],
        output["intrinsic"],
        output["depth_conf"],
        args.depth_conf_thresh,
        args.points_per_frame,
        args.max_pred_points,
        args.seed,
    )
    pred_cloud = apply_transform(pred_cloud, first_gt_pose)
    metrics.update(point_metrics("pred_depth_pred_pose", pred_cloud, gt_points, args.chamfer_max_dist))

    if args.eval_native_points and "world_points" in output:
        native_cloud = collect_native_points(
            output["world_points"],
            output.get("world_points_conf"),
            args.point_conf_thresh,
            args.points_per_frame,
            args.max_pred_points,
            args.seed + 1,
        )
        native_cloud = apply_transform(native_cloud, first_gt_pose)
        metrics.update(point_metrics("native_world_points", native_cloud, gt_points, args.chamfer_max_dist))

    if args.eval_counterfactuals and gt_depth is not None and gt_intrinsic is not None:
        gt_w2c = invert_poses(gt_c2w)[:, :3, :4]
        gt_depth_expanded = gt_depth[..., None]

        gt_depth_gt_pose = collect_unprojected_points(
            gt_depth_expanded,
            gt_w2c,
            gt_intrinsic,
            None,
            0.0,
            args.points_per_frame,
            args.max_pred_points,
            args.seed + 2,
        )
        metrics.update(point_metrics("gt_depth_gt_pose", gt_depth_gt_pose, gt_points, args.chamfer_max_dist))

        pred_depth_gt_pose = collect_unprojected_points(
            output["depth"],
            gt_w2c,
            output["intrinsic"],
            output["depth_conf"],
            args.depth_conf_thresh,
            args.points_per_frame,
            args.max_pred_points,
            args.seed + 3,
        )
        metrics.update(point_metrics("pred_depth_gt_pose", pred_depth_gt_pose, gt_points, args.chamfer_max_dist))

        gt_depth_pred_pose = collect_unprojected_points(
            gt_depth_expanded,
            output["extrinsic"],
            output["intrinsic"],
            None,
            0.0,
            args.points_per_frame,
            args.max_pred_points,
            args.seed + 4,
        )
        gt_depth_pred_pose = apply_transform(gt_depth_pred_pose, first_gt_pose)
        metrics.update(point_metrics("gt_depth_pred_pose", gt_depth_pred_pose, gt_points, args.chamfer_max_dist))

    out_dir = args.out_dir / scene / f"frames_{len(selected_ids)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "selected_frame_ids.json", selected_ids)
    np.savez_compressed(
        out_dir / "predicted_cameras.npz",
        frame_ids=np.asarray(selected_ids, dtype=np.int64),
        pred_w2c=pred_w2c.astype(np.float32),
        pred_intrinsic=output["intrinsic"].astype(np.float32),
        gt_c2w=gt_c2w.astype(np.float32),
    )
    if args.plot:
        plot_trajectory(out_dir / "trajectory.png", pred_w2c, gt_c2w)
    return metrics


def write_summary(rows: list[dict[str, float | str]], out_dir: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = ["scene", "frame_count_actual", "sampling", "inference_time_ms"]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_json(out_dir / "summary.json", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gt-ply-dir", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int, default=10)
    parser.add_argument("--frame-counts", type=int, nargs="+", default=[100, 300, 500, 1000])
    parser.add_argument("--sampling", choices=["prefix", "uniform", "nested_uniform", "regime_step"], default="prefix")
    parser.add_argument("--weights", choices=["local", "hub", "random"], default="local")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("ckpt/VGGT-1B"))
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--preprocess-mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth-conf-thresh", type=float, default=1.0)
    parser.add_argument("--point-conf-thresh", type=float, default=1.0)
    parser.add_argument("--chamfer-max-dist", type=float, default=0.5)
    parser.add_argument("--points-per-frame", type=int, default=2048)
    parser.add_argument("--max-pred-points", type=int, default=200000)
    parser.add_argument("--max-gt-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--eval-native-points", action="store_true")
    parser.add_argument("--eval-counterfactuals", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.gt_ply_dir = resolve_path(args.gt_ply_dir)
    args.scene_list = resolve_path(args.scene_list)
    args.ckpt_dir = resolve_path(args.ckpt_dir)
    args.out_dir = resolve_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    scenes = read_scene_list(args.scene_list, args.scene_limit)
    model = load_model(args.weights, args.ckpt_dir, args.eval_native_points).to(device).eval()

    all_rows: list[dict[str, float | str]] = []
    for scene in scenes:
        scene_dir = args.data_dir / scene
        color_dir = scene_dir / "color"
        pose_dir = scene_dir / "pose"
        image_paths = get_sorted_image_paths(color_dir)
        poses = load_poses(pose_dir)
        image_by_id = {frame_id(path): path for path in image_paths}
        valid_ids = sorted(set(image_by_id) & set(poses))
        if len(valid_ids) < min(args.frame_counts):
            print(f"[scene] skip {scene}: only {len(valid_ids)} valid frames")
            continue
        gt_points = load_gt_ply(scene, args.gt_ply_dir, args.max_gt_points, args.seed)
        selections = make_frame_selections(valid_ids, args.frame_counts, args.sampling)
        for requested_count, selected_ids in selections.items():
            if len(selected_ids) < 3:
                print(f"[scene] skip {scene}/{requested_count}: insufficient frames")
                continue
            existing = None
            if args.resume:
                metrics_path = selection_out_dir(
                    args.out_dir, scene, len(selected_ids)
                ) / "metrics.json"
                existing = load_completed_metrics(
                    metrics_path,
                    requested_count=requested_count,
                    sampling=args.sampling,
                )
            if existing is not None:
                print(f"[scene] resume skip {scene}/{requested_count}: {metrics_path}")
                all_rows.append(existing)
                write_summary(all_rows, args.out_dir)
                continue
            print(f"[scene] {scene} frames={len(selected_ids)} sampling={args.sampling}")
            try:
                row = evaluate_selection(
                    args,
                    model,
                    scene,
                    scene_dir,
                    selected_ids,
                    image_by_id,
                    poses,
                    gt_points,
                    device,
                )
                row["sampling"] = args.sampling
                row["frame_count_requested"] = float(requested_count)
                all_rows.append(row)
                write_summary(all_rows, args.out_dir)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[scene] ERROR {scene}/{requested_count}: {exc}")
            except Exception as exc:
                print(f"[scene] ERROR {scene}/{requested_count}: {exc}")

    write_summary(all_rows, args.out_dir)
    print(f"[summary] {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
