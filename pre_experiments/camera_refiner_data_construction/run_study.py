"""Replay frame-matched 100/200/300 Camera hidden candidates."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np
import torch

from pre_experiments.camera_hidden_state_attribution.analyze import (
    intervention_metrics,
)
from pre_experiments.camera_hidden_state_attribution.artifacts import (
    canonical_digest,
)
from pre_experiments.camera_hidden_state_attribution.replacement import (
    replacement_mask,
)
from pre_experiments.camera_hidden_state_attribution.run_study import (
    _verify_replay,
    replay_tokens,
)
from pre_experiments.camera_hidden_state_attribution.run_replacement import (
    _validate_frozen_replacement,
)
from pre_experiments.camera_refiner_data_construction.analyze import (
    freeze_candidate_policy,
    summarize_scene_shards,
    validate_frozen_policy,
    write_numeric_summary,
)
from pre_experiments.camera_refiner_data_construction.artifacts import (
    load_scene_shard,
    save_scene_shard,
)
from pre_experiments.camera_refiner_data_construction.protocol import (
    Candidate,
    LOCAL_SCALES,
    assemble_multiscale_hidden,
    default_mixture_candidates,
    default_pure_candidates,
    mix_local_hidden,
)
from pre_experiments.common.contracts import atomic_write_json, read_git_commit
from pre_experiments.common.model_io import load_local_model, resolve_device
from pre_experiments.common.pose_metrics import align_pose_sequence
from pre_experiments.local_global_consistency.artifacts import (
    load_global_context,
    load_window_diagnostics,
)
from pre_experiments.local_global_consistency.split import load_split_manifest


ROOT = Path(__file__).resolve().parents[2]
AUTODL_TMP = Path("/root/autodl-tmp")


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def validate_scale_runs(
    scale_run_dirs: Mapping[int, Path],
    *,
    partition: str,
    split_digest: str,
    source_run_id: str,
    expected_scenes: Sequence[str],
    run_scenes: Sequence[str] | None = None,
    require_complete: bool = True,
) -> dict[int, str]:
    """Authenticate complete 100/200/300 local-window runs."""
    if set(scale_run_dirs) != set(LOCAL_SCALES):
        raise ValueError(f"scale runs must contain exactly scales {LOCAL_SCALES}")
    if partition not in {"calibration", "holdout"}:
        raise ValueError("partition must be calibration or holdout")
    scenes = [str(scene) for scene in expected_scenes]
    selected_scenes = (
        scenes if run_scenes is None else [str(scene) for scene in run_scenes]
    )
    if not split_digest or not source_run_id or not scenes:
        raise ValueError("scale run provenance is incomplete")
    if not selected_scenes or any(scene not in scenes for scene in selected_scenes):
        raise ValueError("scale run scene selection is invalid")

    run_ids = {}
    for scale in LOCAL_SCALES:
        directory = Path(scale_run_dirs[scale]).resolve()
        metadata = _json_object(directory / "run_metadata.json")
        invocation = metadata.get("invocation")
        expected_pair = f"{scale}/{scale // 2}"
        if not isinstance(invocation, Mapping):
            raise ValueError(f"scale {expected_pair} invocation is missing")
        if (
            metadata.get("study_name") != "local_global_consistency"
            or metadata.get("partition") != partition
            or metadata.get("split_digest") != split_digest
            or metadata.get("source_run_id") != source_run_id
            or metadata.get("protocol_complete") is not require_complete
            or invocation.get("window_length") != scale
            or invocation.get("window_stride") != scale // 2
            or invocation.get("camera_iterations") != 4
            or invocation.get("preprocess_mode") != "pad"
            or invocation.get("partition_scenes") != scenes
            or invocation.get("scenes") != selected_scenes
        ):
            raise ValueError(
                f"scale run does not match required {expected_pair} protocol: "
                f"{directory}"
            )
        run_id = metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"scale {expected_pair} run_id is invalid")
        run_ids[scale] = run_id
    if len(set(run_ids.values())) != len(run_ids):
        raise ValueError("scale run identities must be distinct")
    return run_ids


def _aligned_errors(
    pred_c2w_raw: np.ndarray,
    gt_c2w_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alignment = align_pose_sequence(
        np.linalg.inv(np.asarray(pred_c2w_raw, dtype=np.float64)),
        np.asarray(gt_c2w_raw, dtype=np.float64),
    )
    return (
        np.asarray(alignment["translation_error_aligned"], dtype=np.float64),
        np.asarray(alignment["rotation_error_deg_aligned"], dtype=np.float64),
    )


def _local_hidden_records(
    camera_head: torch.nn.Module,
    records: Sequence[Mapping[str, object]],
    global_ids: np.ndarray,
    global_gt: np.ndarray,
    device: torch.device,
    *,
    scene: str,
    scale: int,
    replay_tolerance: float,
) -> list[dict[str, object]]:
    global_index = {int(frame_id): index for index, frame_id in enumerate(global_ids)}
    hidden_records = []
    for record in sorted(records, key=lambda item: int(item["window_index"])):
        artifact = record.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{scene}/scale_{scale} local artifact is missing")
        local_ids = np.asarray(artifact["frame_ids"], dtype=np.int64)
        try:
            indices = np.asarray(
                [global_index[int(frame_id)] for frame_id in local_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("local frame is absent from global context") from error
        local_gt = np.asarray(artifact["gt_c2w_raw"], dtype=np.float64)
        if local_gt.shape != (len(local_ids), 4, 4) or not np.allclose(
            local_gt,
            global_gt[indices],
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("local raw GT does not match global raw GT")
        replay = replay_tokens(
            camera_head,
            np.asarray(artifact["normalized_camera_tokens"], dtype=np.float32),
            device,
        )
        _verify_replay(
            replay["pred_c2w_raw"],
            np.asarray(artifact["pred_c2w_raw"]),
            scene=f"{scene}/scale_{scale}/window_{int(record['window_index']):03d}",
            tolerance=replay_tolerance,
        )
        hidden_records.append(
            {
                "window_index": int(record["window_index"]),
                "frame_ids": local_ids,
                "hidden": replay["hidden"],
            }
        )
    return hidden_records


def run_scene_candidates(
    camera_head: torch.nn.Module,
    global_artifact: Mapping[str, np.ndarray],
    scale_records: Mapping[int, Sequence[Mapping[str, object]]],
    frozen_units: Mapping[str, object],
    candidates: Sequence[Candidate],
    device: torch.device,
    *,
    scene: str,
    replay_tolerance: float = 5e-3,
) -> dict[str, object]:
    """Evaluate candidate hidden mixtures for one scene through Camera Head."""
    if set(scale_records) != set(LOCAL_SCALES):
        raise ValueError(f"scale records must contain exactly scales {LOCAL_SCALES}")
    candidate_list = tuple(candidates)
    if not candidate_list or len({item.name for item in candidate_list}) != len(
        candidate_list
    ):
        raise ValueError("candidates must be non-empty and uniquely identified")
    if replay_tolerance <= 0.0:
        raise ValueError("replay_tolerance must be positive")

    global_ids = np.asarray(global_artifact["frame_ids"], dtype=np.int64)
    global_tokens = np.asarray(
        global_artifact["normalized_camera_tokens"], dtype=np.float32
    )
    global_gt = np.asarray(global_artifact["gt_c2w_raw"], dtype=np.float64)
    if (
        global_ids.ndim != 1
        or global_tokens.ndim != 2
        or len(global_tokens) != len(global_ids)
        or global_gt.shape != (len(global_ids), 4, 4)
        or not np.isfinite(global_tokens).all()
        or not np.isfinite(global_gt).all()
    ):
        raise ValueError("global multiscale artifact has invalid shapes or values")

    baseline = replay_tokens(camera_head, global_tokens, device)
    _verify_replay(
        baseline["pred_c2w_raw"],
        np.asarray(global_artifact["pred_c2w_raw"]),
        scene=f"{scene}/global",
        tolerance=replay_tolerance,
    )
    assembled = assemble_multiscale_hidden(
        global_ids,
        {
            scale: _local_hidden_records(
                camera_head,
                scale_records[scale],
                global_ids,
                global_gt,
                device,
                scene=scene,
                scale=scale,
                replay_tolerance=replay_tolerance,
            )
            for scale in LOCAL_SCALES
        },
    )
    local_hidden = assembled["hidden"]
    global_hidden = np.asarray(baseline["hidden"], dtype=np.float32)
    if local_hidden.shape[1:] != global_hidden.shape:
        raise ValueError("local and global hidden tensors do not share a frame gauge")
    iterations, frame_count, hidden_dim = global_hidden.shape
    mask = replacement_mask(
        frozen_units,
        "selected",
        iterations=iterations,
        hidden_dim=hidden_dim,
    )
    if not mask.any():
        raise ValueError("frozen hidden selection is empty")

    replays = [baseline]
    hidden_displacement = [0.0]
    for candidate in candidate_list:
        mixed = mix_local_hidden(local_hidden, candidate.beta)
        delta = candidate.alpha * (mixed - global_hidden)
        selected_delta = delta[
            np.broadcast_to(mask[:, None, :], delta.shape)
        ]
        hidden_displacement.append(
            float(np.sqrt(np.mean(np.square(selected_delta, dtype=np.float64))))
        )
        replays.append(
            replay_tokens(
                camera_head,
                global_tokens,
                device,
                hidden_replacement_values=mixed,
                hidden_replacement_mask=mask,
                hidden_replacement_alpha=candidate.alpha,
                trace_hidden=False,
            )
        )

    translation_errors = []
    rotation_errors = []
    center_change = [0.0]
    rotation_change = [0.0]
    fov_change = [0.0]
    for index, replay in enumerate(replays):
        translation, rotation = _aligned_errors(
            replay["pred_c2w_raw"], global_gt
        )
        translation_errors.append(translation)
        rotation_errors.append(rotation)
        if index:
            changes = intervention_metrics(
                baseline["pred_c2w_raw"],
                replay["pred_c2w_raw"],
                baseline["pose_enc"],
                replay["pose_enc"],
            )
            center_change.append(float(changes["camera_center_displacement_mean"]))
            rotation_change.append(float(changes["rotation_change_deg_mean"]))
            fov_change.append(float(changes["fov_change_mean"]))

    return {
        "scene": scene,
        "frame_ids": global_ids.copy(),
        "scales": assembled["scales"],
        "candidate_names": np.asarray(
            ["baseline", *(item.name for item in candidate_list)]
        ),
        "candidate_alpha": np.asarray(
            [0.0, *(item.alpha for item in candidate_list)], dtype=np.float64
        ),
        "candidate_beta": np.asarray(
            [(0.0, 0.0, 0.0), *(item.beta for item in candidate_list)],
            dtype=np.float64,
        ),
        "global_hidden": global_hidden,
        "local_hidden": local_hidden,
        "selected_window_index": assembled["selected_window_index"],
        "selected_boundary_distance": assembled[
            "selected_boundary_distance"
        ],
        "selected_window_start": assembled["selected_window_start"],
        "selected_window_stop": assembled["selected_window_stop"],
        "local_observation_count": assembled["observation_count"],
        "pred_c2w_raw": np.stack([item["pred_c2w_raw"] for item in replays]),
        "pose_enc": np.stack([item["pose_enc"] for item in replays]),
        "gt_c2w_raw": global_gt.copy(),
        "translation_error_aligned": np.stack(translation_errors),
        "rotation_error_deg_aligned": np.stack(rotation_errors),
        "hidden_displacement_rms": np.asarray(hidden_displacement),
        "camera_center_displacement_mean": np.asarray(center_change),
        "rotation_change_deg_mean": np.asarray(rotation_change),
        "fov_change_mean": np.asarray(fov_change),
    }


def _parse_scale_run(value: str) -> tuple[int, Path]:
    try:
        scale_text, path_text = value.split("=", maxsplit=1)
        scale = int(scale_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "--scale-run must use SCALE=/absolute/run"
        ) from error
    if scale not in LOCAL_SCALES or not path_text:
        raise argparse.ArgumentTypeError(
            f"--scale-run scale must be one of {LOCAL_SCALES}"
        )
    return scale, Path(path_text)


def _scale_run_mapping(values: Sequence[tuple[int, Path]]) -> dict[int, Path]:
    result = {}
    for scale, path in values:
        if scale in result:
            raise ValueError(f"duplicate --scale-run for {scale}")
        result[scale] = path
    if set(result) != set(LOCAL_SCALES):
        raise ValueError(f"--scale-run must contain exactly scales {LOCAL_SCALES}")
    return result


def _scene_list() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "configs" / "fastvggt_scannet50.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _scale_records(run_dir: Path, scene: str) -> list[dict[str, object]]:
    paths = sorted((run_dir / scene).glob("window_*/window_diagnostics.npz"))
    if not paths:
        raise FileNotFoundError(f"no local windows found for {scene}: {run_dir}")
    records = []
    for path in paths:
        try:
            window_index = int(path.parent.name.rsplit("_", maxsplit=1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid local window directory: {path.parent}") from error
        records.append(
            {
                "window_index": window_index,
                "artifact": load_window_diagnostics(path),
            }
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "calibration", "holdout"), required=True
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument(
        "--scale-run", type=_parse_scale_run, action="append", required=True
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--frozen-units", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path)
    parser.add_argument(
        "--candidate-family", choices=("pure", "all"), default="pure"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=AUTODL_TMP / "camera_refiner_data_construction" / "results",
    )
    parser.add_argument("--run-dir-file", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--replay-tolerance", type=float, default=5e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--max-rotation-delta-deg", type=float, default=0.05)
    parser.add_argument("--max-fov-change", type=float, default=0.01)
    parser.add_argument("--min-improved-scene-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        args.scale_runs = _scale_run_mapping(args.scale_run)
    except ValueError as error:
        parser.error(str(error))
    if args.stage == "holdout" and args.frozen_policy is None:
        parser.error("--frozen-policy is required for holdout")
    if (
        args.scene_limit < 0
        or args.replay_tolerance <= 0.0
        or args.bootstrap_samples < 1
        or args.max_rotation_delta_deg < 0.0
        or args.max_fov_change < 0.0
        or not 0.0 <= args.min_improved_scene_fraction <= 1.0
    ):
        parser.error("numeric controls are outside their valid ranges")
    return args


def _candidate_from_policy(policy: Mapping[str, object]) -> Candidate:
    return Candidate(
        alpha=float(policy["selected_alpha"]),
        beta=tuple(float(value) for value in policy["selected_beta"]),  # type: ignore[arg-type]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_run = args.source_run_dir.resolve()
    source_metadata = _json_object(source_run / "run_metadata.json")
    source_run_id = source_metadata.get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise ValueError("source Camera Context run_id is invalid")
    split = load_split_manifest(args.split_manifest.resolve(), _scene_list())
    partition = "holdout" if args.stage == "holdout" else "calibration"
    partition_scenes = [str(scene) for scene in split[f"{partition}_scenes"]]
    if args.stage == "smoke":
        scenes = partition_scenes[:1]
    elif args.scene_limit:
        scenes = partition_scenes[: args.scene_limit]
    else:
        scenes = partition_scenes
    protocol_complete = args.stage != "smoke" and scenes == partition_scenes
    scale_dirs = {scale: path.resolve() for scale, path in args.scale_runs.items()}
    scale_run_ids = validate_scale_runs(
        scale_dirs,
        partition=partition,
        split_digest=str(split["split_digest"]),
        source_run_id=source_run_id,
        expected_scenes=partition_scenes,
        run_scenes=scenes,
        require_complete=protocol_complete,
    )

    frozen_units_value = _json_object(args.frozen_units.resolve())
    frozen_units = _validate_frozen_replacement(
        frozen_units_value,
        split_digest=str(split["split_digest"]),
        calibration_scenes=[str(scene) for scene in split["calibration_scenes"]],
    )
    frozen_policy = None
    if args.stage == "holdout":
        assert args.frozen_policy is not None
        policy_value = _json_object(args.frozen_policy.resolve())
        calibration_scale_ids = {
            int(scale): str(run_id)
            for scale, run_id in policy_value.get("scale_run_ids", {}).items()
        }
        frozen_policy = validate_frozen_policy(
            policy_value,
            split_digest=str(split["split_digest"]),
            calibration_scenes=[str(scene) for scene in split["calibration_scenes"]],
            source_run_id=source_run_id,
            scale_run_ids=calibration_scale_ids,
        )
        candidates = (_candidate_from_policy(frozen_policy),)
    else:
        candidates = default_pure_candidates()
        if args.candidate_family == "all":
            candidates = candidates + default_mixture_candidates()

    commit = read_git_commit(ROOT)
    invocation = {
        "stage": args.stage,
        "partition": partition,
        "source_run_dir": source_run.as_posix(),
        "source_run_id": source_run_id,
        "scale_run_dirs": {
            str(scale): scale_dirs[scale].as_posix() for scale in LOCAL_SCALES
        },
        "scale_run_ids": {
            str(scale): scale_run_ids[scale] for scale in LOCAL_SCALES
        },
        "split_digest": split["split_digest"],
        "scenes": scenes,
        "candidate_names": [candidate.name for candidate in candidates],
        "hidden_frozen_digest": frozen_units.get("frozen_digest"),
        "candidate_frozen_digest": (
            frozen_policy.get("frozen_digest") if frozen_policy else None
        ),
        "replay_tolerance": args.replay_tolerance,
        "device": args.device,
        "seed": args.seed,
    }
    run_id = f"{commit[:7]}_{canonical_digest(invocation)[:12]}"
    run_dir = args.out_dir.resolve() / run_id
    atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "study_name": "camera_refiner_data_construction",
            "run_id": run_id,
            "git_commit": commit,
            "partition": partition,
            "split_digest": split["split_digest"],
            "protocol_complete": protocol_complete,
            "invocation": invocation,
            "metric_policy": "each prediction is aligned independently; GT remains raw",
        },
    )

    device = resolve_device(args.device)
    model = load_local_model(args.ckpt_dir.resolve())
    camera_head = model.camera_head.to(device).eval()
    del model
    shards = []
    for index, scene in enumerate(scenes, start=1):
        scene_dir = run_dir / scene
        shard_path = scene_dir / "scene_shard.npz"
        if shard_path.is_file():
            shard = load_scene_shard(shard_path, scene)
        else:
            global_artifact = load_global_context(
                source_run / scene / "frames_500" / "context_diagnostics.npz"
            )
            shard = run_scene_candidates(
                camera_head,
                global_artifact,
                {
                    scale: _scale_records(scale_dirs[scale], scene)
                    for scale in LOCAL_SCALES
                },
                frozen_units,
                candidates,
                device,
                scene=scene,
                replay_tolerance=args.replay_tolerance,
            )
            save_scene_shard(shard_path, shard)
            atomic_write_json(
                scene_dir / "complete.json",
                {
                    "run_id": run_id,
                    "scene": scene,
                    "frame_count": len(shard["frame_ids"]),
                    "candidate_names": shard["candidate_names"].tolist(),
                },
            )
        shards.append(shard)
        print(f"[scene {index}/{len(scenes)}] {scene}", flush=True)

    summary = summarize_scene_shards(
        shards,
        partition=partition,
        frozen_policy=frozen_policy,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_numeric_summary(run_dir, summary)
    frozen_output = None
    if args.stage == "calibration" and protocol_complete:
        frozen_output = freeze_candidate_policy(
            summary,
            split_digest=str(split["split_digest"]),
            calibration_scenes=partition_scenes,
            source_run_id=source_run_id,
            scale_run_ids=scale_run_ids,
            max_rotation_delta_deg=args.max_rotation_delta_deg,
            max_fov_change=args.max_fov_change,
            min_improved_scene_fraction=args.min_improved_scene_fraction,
        )
        atomic_write_json(run_dir / "frozen_candidate_policy.json", frozen_output)
    atomic_write_json(
        run_dir / "complete.json",
        {
            "run_id": run_id,
            "partition": partition,
            "scenes": scenes,
            "scene_count": len(scenes),
            "protocol_complete": protocol_complete,
            "analysis_complete": protocol_complete,
            "frozen_digest": (
                frozen_output.get("frozen_digest")
                if frozen_output
                else frozen_policy.get("frozen_digest") if frozen_policy else None
            ),
        },
    )
    if args.run_dir_file is not None:
        pointer = args.run_dir_file.resolve()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(f"{run_dir}\n", encoding="utf-8")
    print(f"[done] run={run_dir}")


if __name__ == "__main__":
    main()
