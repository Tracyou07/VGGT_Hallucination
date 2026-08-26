"""End-to-end calibration-first execution for the CVA02 phenomenon check."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from pre_experiments.camera_velocity_ambiguity_02.analyze import (
    analyze_pair_records,
    publish_scene_records,
)
from pre_experiments.camera_velocity_ambiguity_02.artifacts import (
    build_prediction_identity,
    load_completed_prediction,
)
from pre_experiments.camera_velocity_ambiguity_02.contracts import (
    canonical_json_digest,
)
from pre_experiments.camera_velocity_ambiguity_02.controls import (
    build_negative_controls,
)
from pre_experiments.camera_velocity_ambiguity_02.data import (
    PreparedScene,
    load_rgbd_observations,
    prepare_selected_scene,
)
from pre_experiments.camera_velocity_ambiguity_02.events import EventPolicy
from pre_experiments.camera_velocity_ambiguity_02.frames import build_protocol_windows
from pre_experiments.camera_velocity_ambiguity_02.frozen_oracle import (
    evaluate_with_frozen_oracle,
    fit_frozen_oracle,
)
from pre_experiments.camera_velocity_ambiguity_02.geometry import (
    build_pair_geometry,
    global_scene_scale,
)
from pre_experiments.camera_velocity_ambiguity_02.input_gate import (
    VerifiedInputs,
    canonical_scene_list_digest,
    load_verified_inputs,
)
from pre_experiments.camera_velocity_ambiguity_02.interpolation import (
    TranslationCandidate,
    assert_translation_curve_convex,
    build_translation_candidates,
    evaluate_translation_candidates,
)
from pre_experiments.camera_velocity_ambiguity_02.protocol import load_protocol_v2
from pre_experiments.camera_velocity_ambiguity_02.rgbd_gate import (
    RGBDConfig,
    RGBDObservations,
    evaluate_rgbd_path,
    freeze_observation_scale,
)
from pre_experiments.camera_velocity_ambiguity_02.state import (
    FrozenPolicy,
    fit_and_freeze_policy,
)
from pre_experiments.camera_velocity_ambiguity_02.units import build_overlap_units
from pre_experiments.common.contracts import atomic_write_json, read_git_commit
from pre_experiments.common.model_io import find_checkpoint, resolve_device


ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")


def endpoint_validities(
    global_rms: float, left_rms: float, right_rms: float
) -> tuple[bool, bool]:
    """Label whether each endpoint strictly improves the frozen-oracle baseline."""
    values = np.asarray([global_rms, left_rms, right_rms], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("endpoint RMS values must be finite and positive")
    tolerance = 1e-9 * max(1.0, global_rms)
    return left_rms < global_rms - tolerance, right_rms < global_rms - tolerance


def provisional_smoke_policy() -> EventPolicy:
    """Return a fixed diagnostic policy; it is never used as the frozen policy."""
    return EventPolicy(
        direction_cosine_max=0.0,
        normalized_separation_min=1e-4,
        barrier_margin=1e-5,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction(
    root: Path,
    context,
    *,
    scene: str,
    kind: str,
    window_index: int | None,
    frame_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    ids = np.asarray(frame_ids, dtype=np.int64)
    identity = build_prediction_identity(
        run_id=context.run_id,
        scene=scene,
        artifact_kind=kind,
        window_index=window_index,
        frame_ids=ids,
        checkpoint_sha256=context.checkpoint_sha256,
        git_commit=context.git_commit,
        protocol_digest=context.protocol_digest,
        preprocess=context.preprocess,
        camera_iterations=context.camera_iterations,
    )
    directory = root / ("global" if kind == "global" else f"local/window_{window_index:03d}")
    return load_completed_prediction(
        directory / "prediction.npz", directory / "complete.json", identity
    )


def _slice_observations(
    observations: RGBDObservations, indices: Sequence[int]
) -> RGBDObservations:
    index = np.asarray(indices, dtype=np.int64)
    return RGBDObservations(
        frame_ids=observations.frame_ids[index].copy(),
        rgb=observations.rgb[index].copy(),
        depth=observations.depth[index].copy(),
        intrinsics=observations.intrinsics[index].copy(),
    )


def _slice_candidates(
    candidates: Sequence[TranslationCandidate], indices: Sequence[int]
) -> tuple[TranslationCandidate, ...]:
    index = np.asarray(indices, dtype=np.int64)
    return tuple(
        TranslationCandidate(
            alpha=candidate.alpha,
            c2w=candidate.c2w[index].copy(),
            fov=None if candidate.fov is None else candidate.fov[index].copy(),
        )
        for candidate in candidates
    )


def _temporal_support(
    observations: RGBDObservations,
    candidates: Sequence[TranslationCandidate],
    frozen_scale,
    config: RGBDConfig,
) -> bool:
    count = len(observations.frame_ids)
    midpoint = count // 2
    if midpoint < 3 or count - midpoint < 3:
        return False
    results = []
    for indices in (range(0, midpoint), range(midpoint, count)):
        result = evaluate_rgbd_path(
            _slice_observations(observations, indices),
            _slice_candidates(candidates, indices),
            frozen_scale,
            config,
        )
        results.append(result)
    return all(
        result.valid and result.interior_barrier > config.flat_energy_tolerance
        for result in results
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"missing scene record layer: {path}") from error
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"scene record must be a JSON object: {path}")
        rows.append(value)
    return rows


def _load_completed_scene(root: Path, expected: Mapping[str, object]):
    completion = root / "scene_complete.json"
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload != expected:
        return None
    names = (
        "prediction_only.jsonl",
        "privileged_labels.jsonl",
        "rgbd_observation.jsonl",
        "decisions.jsonl",
        "controls.jsonl",
        "records_manifest.json",
    )
    if any(not (root / name).is_file() for name in names):
        return None
    return tuple(_read_jsonl(root / name) for name in names[:5])


def analyze_prepared_scene(
    prepared: PreparedScene,
    *,
    prediction_root: Path,
    analysis_root: Path,
    context,
    protocol,
    policy: EventPolicy,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Analyze one exact scene and keep all evidence layers physically separate."""
    completion = {
        "schema": "camera_velocity_ambiguity_02.scene_complete.v1",
        "run_id": context.run_id,
        "scene": prepared.scene,
        "protocol_digest": protocol.config_digest,
        "prepared_manifest_digest": prepared.manifest_digest,
        "git_commit": context.git_commit,
        "provisional_policy": asdict(policy),
    }
    resumed = _load_completed_scene(Path(analysis_root), completion)
    if resumed is not None:
        return resumed

    selection = prepared.selection
    windows = build_protocol_windows(
        selection,
        length=protocol.window_length,
        stride=protocol.window_stride,
    )
    units = build_overlap_units(
        prepared.scene,
        windows,
        primary_overlap=protocol.window_length - protocol.window_stride,
    )
    global_arrays = _prediction(
        Path(prediction_root),
        context,
        scene=prepared.scene,
        kind="global",
        window_index=None,
        frame_ids=selection.frame_ids,
    )
    global_c2w = global_arrays["pred_c2w_raw"]
    scene_scale = global_scene_scale(global_c2w)
    oracle = fit_frozen_oracle(
        prepared.scene,
        global_arrays["frame_ids"],
        global_c2w,
        prepared.raw_gt_c2w,
    )
    full_observations = load_rgbd_observations(prepared)
    rgbd_config = RGBDConfig()
    frozen_scale = freeze_observation_scale(
        full_observations, global_c2w, rgbd_config
    )

    local_arrays = {
        window.index: _prediction(
            Path(prediction_root),
            context,
            scene=prepared.scene,
            kind="local",
            window_index=window.index,
            frame_ids=window.frame_ids,
        )
        for window in windows
    }
    prediction_rows: list[dict[str, object]] = []
    privileged_rows: list[dict[str, object]] = []
    rgbd_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for unit in units:
        geometry = build_pair_geometry(
            unit,
            global_c2w=global_c2w,
            left_local_c2w=local_arrays[unit.left_window_index]["pred_c2w_raw"],
            right_local_c2w=local_arrays[unit.right_window_index]["pred_c2w_raw"],
            scene_scale=scene_scale,
        )
        metrics = geometry.metrics
        alignment_valid = geometry.left_alignment.valid and geometry.right_alignment.valid
        prediction_row = {
            "sample_id": unit.pair_id,
            "scene": prepared.scene,
            "pair_id": unit.pair_id,
            "route": unit.route,
            "alignment_valid": bool(alignment_valid),
            "direction_evaluable": bool(metrics.direction_evaluable) if metrics else False,
            "flattened_cosine": float(metrics.flattened_cosine) if metrics and metrics.flattened_cosine is not None else 1.0,
            "normalized_separation": float(metrics.normalized_rms_separation) if metrics else 0.0,
        }

        shared = np.asarray(unit.global_shared_indices, dtype=np.int64)
        global_shared = global_c2w[shared]
        gt_shared = prepared.raw_gt_c2w[shared]
        global_eval = evaluate_with_frozen_oracle(oracle, global_shared, gt_shared)
        global_rms = max(global_eval.rms_translation_error, np.finfo(np.float64).eps)
        left_rms = global_rms
        right_rms = global_rms
        left_valid = False
        right_valid = False
        rgbd_valid = False
        barrier = 0.0
        temporal = False

        if metrics is not None and metrics.direction_evaluable:
            candidates = build_translation_candidates(
                global_shared,
                metrics.left_residual,
                metrics.right_residual,
                alphas=protocol.alphas,
            )
            curve = evaluate_translation_candidates(oracle, candidates, gt_shared)
            assert_translation_curve_convex(curve)
            left_rms = max(float(curve.rms_l2[0]), np.finfo(np.float64).eps)
            right_rms = max(float(curve.rms_l2[-1]), np.finfo(np.float64).eps)
            left_valid, right_valid = endpoint_validities(
                global_rms, left_rms, right_rms
            )
            pair_observations = _slice_observations(full_observations, shared)
            rgbd_path = evaluate_rgbd_path(
                pair_observations, candidates, frozen_scale, rgbd_config
            )
            rgbd_valid = bool(rgbd_path.valid)
            barrier = float(rgbd_path.interior_barrier)
            temporal = _temporal_support(
                pair_observations, candidates, frozen_scale, rgbd_config
            ) if rgbd_valid else False

            controls = build_negative_controls(
                metrics.left_residual,
                metrics.right_residual,
                wrong_window_residual=np.roll(metrics.right_residual, 1, axis=0),
                scene_scale=scene_scale,
            )
            for name, control in sorted(controls.items()):
                control_rows.append(
                    {
                        "sample_id": unit.pair_id,
                        "control": name,
                        "alignment_valid": control.alignment_valid,
                        "direction_evaluable": control.metrics.direction_evaluable,
                        "flattened_cosine": control.metrics.flattened_cosine,
                        "normalized_separation": control.metrics.normalized_rms_separation,
                    }
                )

        privileged_row = {
            "sample_id": unit.pair_id,
            "left_endpoint_valid": bool(left_valid),
            "right_endpoint_valid": bool(right_valid),
            "global_rms": float(global_rms),
            "left_rms": float(left_rms),
            "right_rms": float(right_rms),
        }
        rgbd_row = {
            "sample_id": unit.pair_id,
            "rgbd_valid": bool(rgbd_valid),
            "interior_barrier": float(barrier),
            "temporal_support": bool(temporal),
        }
        prediction_rows.append(prediction_row)
        privileged_rows.append(privileged_row)
        rgbd_rows.append(rgbd_row)
        decision_rows.append(
            analyze_pair_records(prediction_row, privileged_row, rgbd_row, policy)
        )

    publish_scene_records(
        Path(analysis_root),
        scene=prepared.scene,
        prediction_rows=prediction_rows,
        privileged_rows=privileged_rows,
        rgbd_rows=rgbd_rows,
        decision_rows=decision_rows,
        control_rows=control_rows,
    )
    atomic_write_json(Path(analysis_root) / "scene_complete.json", completion)
    return prediction_rows, privileged_rows, rgbd_rows, decision_rows, control_rows


def _scene_asset(verified: VerifiedInputs, scene: str):
    matches = [asset for asset in verified.assets if asset.scene == scene and asset.kind == "sens"]
    if len(matches) != 1:
        raise ValueError(f"verified inputs do not contain exactly one .sens asset for {scene}")
    return matches[0]


def _calibration_rows(
    scene_results: Sequence[tuple[list[dict[str, object]], ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prediction_rows, _privileged, _rgbd, _decisions, _controls in scene_results:
        for row in prediction_rows:
            if row["route"] == "primary":
                rows.append(
                    {
                        "scene": row["scene"],
                        "pair_id": row["pair_id"],
                        "route": "primary",
                        "flattened_cosine": row["flattened_cosine"],
                        "normalized_separation": row["normalized_separation"],
                        "control_barrier": RGBDConfig().flat_energy_tolerance,
                    }
                )
    return rows


def _event_policy(policy: FrozenPolicy) -> EventPolicy:
    return EventPolicy(
        policy.direction_cosine_max,
        policy.normalized_separation_min,
        policy.barrier_margin,
    )


def _write_smoke_completion(
    run_root: Path, *, context, protocol, prepared: PreparedScene
) -> None:
    unsigned = {
        "schema": "camera_velocity_ambiguity_02.smoke_complete.v1",
        "run_id": context.run_id,
        "scene": prepared.scene,
        "protocol_digest": protocol.config_digest,
        "prepared_manifest_digest": prepared.manifest_digest,
        "git_commit": context.git_commit,
        "production_frame_count": len(prepared.selection.frame_ids),
        "global_runs": 1,
        "local_windows": len(build_protocol_windows(prepared.selection)),
    }
    atomic_write_json(
        run_root / "manifests" / "smoke_complete.json",
        {**unsigned, "completion_digest": canonical_json_digest(unsigned)},
    )


def run_stage(args: argparse.Namespace) -> Path:
    """Run one-scene smoke or exact ten-scene calibration on H20."""
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run_id contains unsupported characters")
    protocol = load_protocol_v2(
        args.protocol,
        parent_split_path=args.parent_split,
        scene_list_path=args.scene_list,
    )
    verified = load_verified_inputs(
        args.marker,
        expected_remote_root=str(args.data_root),
        expected_scene_list_sha256=canonical_scene_list_digest(protocol.scene_order),
        expected_scenes=protocol.scene_order,
    )
    expected_count = 1 if args.stage == "smoke" else 10
    if args.scene_limit != expected_count:
        raise ValueError(f"{args.stage} requires exact scene_limit={expected_count}")
    scenes = protocol.calibration_scenes[:expected_count]
    checkpoint = find_checkpoint(args.checkpoint_dir)
    checkpoint_sha = _sha256_file(checkpoint)
    commit = read_git_commit(ROOT)
    from pre_experiments.camera_velocity_ambiguity_02.predict import (
        PredictionContext,
        load_local_camera_model,
        run_scene_predictions,
    )

    context = PredictionContext(
        run_id=args.run_id,
        checkpoint_sha256=checkpoint_sha,
        git_commit=commit,
        protocol_digest=protocol.config_digest,
    )
    run_root = args.result_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_root / "manifests" / "run.json",
        {
            "schema": "camera_velocity_ambiguity_02.run.v1",
            "run_id": args.run_id,
            "stage": args.stage,
            "scenes": list(scenes),
            "protocol_digest": protocol.config_digest,
            "input_digest": verified.marker_sha256,
            "checkpoint_sha256": checkpoint_sha,
            "git_commit": commit,
            "device": args.device,
        },
    )
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("formal CVA02 execution requires H20 CUDA")
    model = None
    results = []
    prepared_scenes: list[PreparedScene] = []
    for index, scene in enumerate(scenes, start=1):
        print(f"[cva02] scene {index}/{len(scenes)} prepare {scene}", flush=True)
        asset = _scene_asset(verified, scene)
        sens_path = args.data_root / asset.relative_path
        prepared = prepare_selected_scene(
            sens_path,
            args.processed_root / scene,
            scene=scene,
            input_frames=protocol.frame_count(scene),
            sens_sha256=asset.sha256,
            expected_bytes=asset.bytes,
        )
        prepared_scenes.append(prepared)
        if model is None:
            print("[cva02] integrity/preparation passed; loading local VGGT checkpoint", flush=True)
            model = load_local_camera_model(args.checkpoint_dir, device)
        prediction_root = run_root / "predictions" / scene
        summary = run_scene_predictions(
            model=model,
            scene=scene,
            selection=prepared.selection,
            output_dir=prediction_root,
            context=context,
            device=device,
            window_length=protocol.window_length,
            window_stride=protocol.window_stride,
        )
        print(f"[cva02] {scene} prediction {summary}", flush=True)
        results.append(
            analyze_prepared_scene(
                prepared,
                prediction_root=prediction_root,
                analysis_root=run_root / "calibration" / scene,
                context=context,
                protocol=protocol,
                policy=provisional_smoke_policy(),
            )
        )
        print(f"[cva02] {scene} analysis complete", flush=True)
        if args.stage == "smoke":
            _write_smoke_completion(
                run_root, context=context, protocol=protocol, prepared=prepared
            )

    if args.stage == "calibration":
        policy_path = run_root / "frozen_policy" / "policy.json"
        if policy_path.exists():
            raise FileExistsError("refusing to overwrite an existing frozen policy")
        frozen = fit_and_freeze_policy(
            policy_path,
            _calibration_rows(results),
            calibration_scenes=scenes,
            protocol_digest=protocol.config_digest,
            input_digest=verified.marker_sha256,
            git_commit=commit,
        )
        final_policy = _event_policy(frozen)
        class_counts: dict[str, int] = {}
        for scene, scene_result in zip(scenes, results):
            prediction_rows, privileged_rows, rgbd_rows, _old, control_rows = scene_result
            decisions = [
                analyze_pair_records(prediction, privileged, rgbd, final_policy)
                for prediction, privileged, rgbd in zip(
                    prediction_rows, privileged_rows, rgbd_rows
                )
            ]
            for row in decisions:
                name = str(row["event_class"])
                class_counts[name] = class_counts.get(name, 0) + 1
            publish_scene_records(
                run_root / "calibration" / scene,
                scene=scene,
                prediction_rows=prediction_rows,
                privileged_rows=privileged_rows,
                rgbd_rows=rgbd_rows,
                decision_rows=decisions,
                control_rows=control_rows,
            )
        unsigned = {
            "schema": "camera_velocity_ambiguity_02.calibration_complete.v1",
            "run_id": args.run_id,
            "scene_count": len(scenes),
            "primary_pair_count": len(_calibration_rows(results)),
            "policy_digest": frozen.policy_digest,
            "class_counts": class_counts,
        }
        atomic_write_json(
            run_root / "manifests" / "calibration_complete.json",
            {**unsigned, "completion_digest": canonical_json_digest(unsigned)},
        )
    return run_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "calibration"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene-limit", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("/data/yjh/share/datasets/ScanNet"))
    parser.add_argument("--processed-root", type=Path, default=Path("/data/yjh/share/datasets/ScanNet/processed_cva02_v1"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/data/yjh/share/pretrained/VGGT-1B"))
    parser.add_argument("--result-root", type=Path, default=Path("/data/output/camera_velocity_ambiguity"))
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/scannet50_camera_velocity_ambiguity_02_split_v2.json")
    parser.add_argument("--parent-split", type=Path, default=ROOT / "configs/scannet50_local_global_split.json")
    parser.add_argument("--scene-list", type=Path, default=ROOT / "configs/fastvggt_scannet50.txt")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = parser.parse_args(argv)
    if args.marker is None:
        args.marker = args.data_root / "verified_completion.json"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_root = run_stage(args)
    print(f"[cva02] stage={args.stage} complete run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
