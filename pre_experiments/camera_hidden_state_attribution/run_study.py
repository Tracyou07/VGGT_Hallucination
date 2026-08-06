"""Replay saved Camera Head tokens for hidden-state attribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
    load_scene_statistics,
    save_scene_statistics,
)
from pre_experiments.camera_hidden_state_attribution.attribution import (
    contribution_drift,
    freeze_unit_sets,
    group_specificity,
)
from pre_experiments.camera_hidden_state_attribution.analyze import (
    intervention_metrics,
    unit_mask,
    write_numeric_summary,
)
from pre_experiments.common.contracts import atomic_write_json, read_git_commit
from pre_experiments.common.model_io import load_local_model, resolve_device
from pre_experiments.common.pose_metrics import to_homogeneous
from pre_experiments.common.pose_metrics import align_pose_sequence
from pre_experiments.local_global_consistency.artifacts import (
    load_global_context,
    load_window_diagnostics,
)
from pre_experiments.local_global_consistency.split import load_split_manifest
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


ROOT = Path(__file__).resolve().parents[2]
AUTODL_TMP = Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp"))
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_ROOT", str(AUTODL_TMP / "results"))
)


def replay_tokens(
    camera_head: torch.nn.Module,
    normalized_tokens: np.ndarray,
    device: torch.device,
    *,
    hidden_ablation_mask: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    tokens = np.asarray(normalized_tokens, dtype=np.float32)
    if tokens.ndim != 2 or len(tokens) < 2 or not np.isfinite(tokens).all():
        raise ValueError("normalized tokens must be finite with shape [S, hidden]")
    tensor = torch.from_numpy(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        poses, trace = camera_head.decode_pose_tokens(
            tensor,
            num_iterations=4,
            return_trace=True,
            trace_pose_tokens=True,
            hidden_ablation_mask=hidden_ablation_mask,
        )
    pose_enc = poses[-1]
    extrinsic, _ = pose_encoding_to_extri_intri(
        pose_enc, build_intrinsics=False
    )
    pred_w2c = to_homogeneous(
        extrinsic[0].detach().float().cpu().numpy()
    )
    return {
        "hidden": torch.stack(trace["pose_branch_hidden_list"])[
            :, 0
        ].detach().float().cpu().numpy(),
        "trunk": torch.stack(trace["trunk_output_list"])[
            :, 0
        ].detach().float().cpu().numpy(),
        "pose_delta": torch.stack(trace["pose_delta_list"])[
            :, 0
        ].detach().float().cpu().numpy(),
        "pose_enc": pose_enc[0].detach().float().cpu().numpy(),
        "pred_c2w_raw": np.linalg.inv(pred_w2c),
    }


def _verify_replay(
    replayed: np.ndarray,
    stored: np.ndarray,
    *,
    scene: str,
    tolerance: float,
) -> None:
    expected = np.asarray(stored, dtype=np.float64)
    if replayed.shape != expected.shape or not np.allclose(
        replayed, expected, atol=tolerance, rtol=0
    ):
        maximum = (
            float(np.max(np.abs(replayed - expected)))
            if replayed.shape == expected.shape
            else float("inf")
        )
        raise ValueError(
            f"Camera Head replay mismatch for {scene}: max_abs={maximum:.6g}"
        )


def collect_scene_statistics(
    camera_head: torch.nn.Module,
    global_artifact: dict[str, np.ndarray],
    local_artifacts: list[dict[str, np.ndarray]],
    device: torch.device,
    *,
    scene: str,
    replay_tolerance: float = 5e-3,
) -> dict[str, object]:
    global_replay = replay_tokens(
        camera_head, global_artifact["normalized_camera_tokens"], device
    )
    _verify_replay(
        global_replay["pred_c2w_raw"],
        global_artifact["pred_c2w_raw"],
        scene=f"{scene}/global",
        tolerance=replay_tolerance,
    )
    global_ids = np.asarray(global_artifact["frame_ids"], dtype=np.int64)
    id_to_index = {int(frame_id): index for index, frame_id in enumerate(global_ids)}
    matched_global = []
    matched_local = []
    boundary_global = {"edge": [], "interior": []}
    boundary_local = {"edge": [], "interior": []}
    for window_index, local in enumerate(local_artifacts):
        local_replay = replay_tokens(
            camera_head, local["normalized_camera_tokens"], device
        )
        _verify_replay(
            local_replay["pred_c2w_raw"],
            local["pred_c2w_raw"],
            scene=f"{scene}/window_{window_index:03d}",
            tolerance=replay_tolerance,
        )
        try:
            indices = [
                id_to_index[int(frame_id)]
                for frame_id in np.asarray(local["frame_ids"], dtype=np.int64)
            ]
        except KeyError as error:
            raise ValueError(f"local frame ID is absent from global context: {scene}") from error
        matched_global.append(global_replay["hidden"][:, indices])
        matched_local.append(local_replay["hidden"])
        positions = np.arange(len(indices))
        distance = np.minimum(positions, len(indices) - 1 - positions)
        for stratum, selector in (
            ("edge", distance < 10),
            ("interior", distance >= 25),
        ):
            if selector.any():
                boundary_global[stratum].append(
                    global_replay["hidden"][:, np.asarray(indices)[selector]]
                )
                boundary_local[stratum].append(
                    local_replay["hidden"][:, selector]
                )
    if not matched_local:
        raise ValueError(f"no local windows found for scene statistics: {scene}")
    global_hidden = np.concatenate(matched_global, axis=1)
    local_hidden = np.concatenate(matched_local, axis=1)
    weight = (
        camera_head.pose_branch.fc2.weight.detach().float().cpu().numpy()
    )
    boundary_drift = {}
    boundary_counts = {}
    for stratum in ("edge", "interior"):
        if boundary_global[stratum]:
            stratum_global = np.concatenate(boundary_global[stratum], axis=1)
            stratum_local = np.concatenate(boundary_local[stratum], axis=1)
            boundary_drift[stratum] = contribution_drift(
                stratum_global, stratum_local, weight
            )
            boundary_counts[stratum] = int(stratum_global.shape[1])
        else:
            boundary_drift[stratum] = {
                group: np.zeros_like(values)
                for group, values in contribution_drift(
                    global_hidden, local_hidden, weight
                ).items()
            }
            boundary_counts[stratum] = 0
    return {
        "scene": scene,
        "drift": contribution_drift(global_hidden, local_hidden, weight),
        "specificity": group_specificity(weight),
        "boundary_drift": boundary_drift,
        "boundary_counts": boundary_counts,
        "matched_observation_count": int(global_hidden.shape[1]),
    }


def run_global_interventions(
    camera_head: torch.nn.Module,
    global_artifact: dict[str, np.ndarray],
    frozen: dict[str, object],
    device: torch.device,
    *,
    scene: str,
) -> list[dict[str, object]]:
    """Ablate frozen units in global context and report component-wise effects."""
    tokens = global_artifact["normalized_camera_tokens"]
    baseline = replay_tokens(camera_head, tokens, device)
    baseline_alignment = align_pose_sequence(
        np.linalg.inv(baseline["pred_c2w_raw"]),
        global_artifact["gt_c2w_raw"],
    )
    baseline_translation = float(
        np.asarray(baseline_alignment["translation_error_aligned"]).mean()
    )
    iterations, _, hidden_dim = baseline["hidden"].shape
    rows = []
    for group in ("translation", "rotation", "fov"):
        for set_name in ("selected", "controls"):
            mask = unit_mask(
                frozen, group, set_name, iterations, hidden_dim
            )
            changed = replay_tokens(
                camera_head,
                tokens,
                device,
                hidden_ablation_mask=torch.from_numpy(mask).to(device),
            )
            metrics = intervention_metrics(
                baseline["pred_c2w_raw"],
                changed["pred_c2w_raw"],
                baseline["pose_enc"],
                changed["pose_enc"],
            )
            changed_alignment = align_pose_sequence(
                np.linalg.inv(changed["pred_c2w_raw"]),
                global_artifact["gt_c2w_raw"],
            )
            changed_translation = float(
                np.asarray(changed_alignment["translation_error_aligned"]).mean()
            )
            rows.append(
                {
                    "scene": scene,
                    "group": group,
                    "set": set_name,
                    **metrics,
                    "aligned_translation_error_mean": changed_translation,
                    "aligned_translation_error_delta": (
                        changed_translation - baseline_translation
                    ),
                }
            )
    return rows


def _scene_list() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "configs" / "fastvggt_scannet50.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _local_artifacts(local_run: Path, scene: str) -> list[dict[str, np.ndarray]]:
    paths = sorted((local_run / scene).glob("window_*/window_diagnostics.npz"))
    if not paths:
        raise FileNotFoundError(f"no local window artifacts found for {scene}")
    return [load_window_diagnostics(path) for path in paths]


def _validate_local_run(
    local_run: Path,
    *,
    partition: str,
    split_digest: str,
) -> None:
    try:
        metadata = json.loads(
            (local_run / "run_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid local run metadata: {local_run}") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("study_name") != "local_global_consistency"
        or metadata.get("partition") != partition
        or metadata.get("split_digest") != split_digest
    ):
        raise ValueError(
            f"local run provenance does not match {partition}/{split_digest}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "calibration", "holdout"), required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_ROOT / "camera_hidden_state_attribution" / "results",
    )
    parser.add_argument("--frozen-units", type=Path)
    parser.add_argument("--run-dir-file", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--replay-tolerance", type=float, default=5e-3)
    args = parser.parse_args(argv)
    if args.stage == "holdout" and args.frozen_units is None:
        parser.error("--frozen-units is required for holdout")
    if args.scene_limit < 0 or args.replay_tolerance <= 0:
        parser.error("scene-limit must be non-negative and tolerance positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    split = load_split_manifest(args.split_manifest.resolve(), _scene_list())
    partition = "calibration" if args.stage in {"smoke", "calibration"} else "holdout"
    _validate_local_run(
        args.local_run_dir.resolve(),
        partition=partition,
        split_digest=str(split["split_digest"]),
    )
    scenes = list(split[f"{partition}_scenes"])
    if args.stage == "smoke":
        scenes = scenes[:1]
    elif args.scene_limit:
        scenes = scenes[: args.scene_limit]
    invocation = {
        "stage": args.stage,
        "source_run_dir": args.source_run_dir.resolve().as_posix(),
        "local_run_dir": args.local_run_dir.resolve().as_posix(),
        "checkpoint_dir": args.ckpt_dir.resolve().as_posix(),
        "device": args.device,
        "split_digest": split["split_digest"],
        "scenes": scenes,
        "replay_tolerance": args.replay_tolerance,
    }
    commit = read_git_commit(ROOT)
    run_id = f"{commit[:7]}_{canonical_digest(invocation)[:12]}"
    run_dir = args.out_dir.resolve() / run_id
    device = resolve_device(args.device)
    model = load_local_model(args.ckpt_dir.resolve())
    camera_head = model.camera_head.to(device).eval()
    del model
    frozen = None
    if args.stage == "holdout":
        frozen = json.loads(args.frozen_units.read_text(encoding="utf-8"))
        digest = frozen.pop("frozen_digest", None)
        if (
            digest != canonical_digest(frozen)
            or frozen.get("split_digest") != split["split_digest"]
            or frozen.get("calibration_scenes") != split["calibration_scenes"]
        ):
            raise ValueError("frozen unit provenance mismatch")
        frozen["frozen_digest"] = digest

    scene_statistics = []
    for scene in scenes:
        scene_dir = run_dir / scene
        statistics_path = scene_dir / "unit_statistics.npz"
        if statistics_path.is_file():
            statistics = load_scene_statistics(statistics_path, scene)
        else:
            global_artifact = load_global_context(
                args.source_run_dir.resolve()
                / scene
                / "frames_500"
                / "context_diagnostics.npz"
            )
            statistics = collect_scene_statistics(
                camera_head,
                global_artifact,
                _local_artifacts(args.local_run_dir.resolve(), scene),
                device,
                scene=scene,
                replay_tolerance=args.replay_tolerance,
            )
            save_scene_statistics(statistics_path, statistics)
            atomic_write_json(
                scene_dir / "complete.json",
                {
                    "run_id": run_id,
                    "scene": scene,
                    "matched_observation_count": statistics["matched_observation_count"],
                },
            )
        scene_statistics.append(statistics)

    if args.stage == "calibration" and not args.scene_limit:
        frozen = freeze_unit_sets(scene_statistics, top_k=64, seed=33)
        frozen["split_digest"] = split["split_digest"]
        frozen["calibration_scenes"] = scenes
        frozen["frozen_digest"] = canonical_digest(frozen)
        atomic_write_json(run_dir / "frozen_units.json", frozen)

    protocol_complete = (
        args.stage in {"calibration", "holdout"} and args.scene_limit == 0
    )
    intervention_rows: list[dict[str, object]] = []
    if frozen is not None:
        for scene in scenes:
            global_artifact = load_global_context(
                args.source_run_dir.resolve()
                / scene
                / "frames_500"
                / "context_diagnostics.npz"
            )
            rows = run_global_interventions(
                camera_head, global_artifact, frozen, device, scene=scene
            )
            intervention_rows.extend(rows)
            atomic_write_json(
                run_dir / scene / "intervention_summary.json",
                {"scene": scene, "rows": rows},
            )
        write_numeric_summary(
            run_dir,
            frozen,
            intervention_rows,
            partition=partition,
            scene_statistics=scene_statistics,
        )
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "study_name": "camera_hidden_state_attribution",
            "run_id": run_id,
            "git_commit": commit,
            "partition": partition,
            "split_digest": split["split_digest"],
            "protocol_complete": protocol_complete,
            "invocation": invocation,
            "metric_policy": "prediction metrics aligned when used; GT always raw",
        },
    )
    atomic_write_json(
        run_dir / "complete.json",
        {
            "run_id": run_id,
            "partition": partition,
            "scenes": scenes,
            "scene_count": len(scenes),
            "protocol_complete": protocol_complete,
            "analysis_complete": bool(frozen is not None and protocol_complete),
            "frozen_digest": frozen.get("frozen_digest") if frozen else None,
        },
    )
    if args.run_dir_file is not None:
        args.run_dir_file.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.run_dir_file.resolve().write_text(f"{run_dir}\n", encoding="utf-8")
    print(f"[done] run={run_dir}")


if __name__ == "__main__":
    main()
