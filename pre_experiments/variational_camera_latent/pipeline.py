from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from pre_experiments.common.contracts import read_git_commit
from pre_experiments.common.model_io import find_checkpoint

from .alpha_scan import (
    DEFAULT_ALPHAS,
    generate_alpha_scan_candidates,
    load_alpha_scan_candidates,
    load_alpha_scan_privileged,
    write_alpha_scan_privileged_sidecar,
    write_alpha_scan_report,
)
from .candidates import (
    analyze_candidate_shard,
    generate_deterministic_candidates,
    generate_scene_candidates,
    load_candidate_shard,
)
from .contracts import SourceShardRecord
from .privileged import (
    load_privileged_sidecar,
    write_privileged_deterministic_sidecar,
    write_privileged_scene_sidecar,
)
from .report import summarize_run
from .source import build_scene_source_shard, load_source_shard, write_source_manifest
from .train import TrainConfig, train_models
from .vrfm_residual_scan import (
    generate_vrfm_residual_alpha_scan,
    load_vrfm_residual_alpha_scan,
    load_vrfm_residual_privileged,
    prepared_gt_sha256,
    write_vrfm_residual_privileged_sidecar,
    write_vrfm_residual_report,
)


ROOT = Path(__file__).resolve().parents[2]


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_completion(path: Path, payload: dict[str, object]) -> dict[str, object]:
    if "completion_digest" in payload:
        raise ValueError("unsigned completion payload cannot contain completion_digest")
    completed = {**payload, "completion_digest": _canonical_digest(payload)}
    _atomic_json(Path(path), completed)
    return completed


def load_exact_completion(
    path: Path, payload: dict[str, object]
) -> dict[str, object] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {**payload, "completion_digest": _canonical_digest(payload)}
    return existing if existing == expected else None


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _load_signed_completion(path: Path, *, schema: str) -> dict[str, object]:
    payload = _read_json(path)
    completion_digest = payload.pop("completion_digest", None)
    if payload.get("schema") != schema:
        raise ValueError(f"completion schema does not match: {path}")
    if completion_digest != _canonical_digest(payload):
        raise ValueError(f"completion digest does not match: {path}")
    return payload


def _source_scene_order(source_run: Path) -> list[str]:
    manifest = _read_json(source_run / "manifests" / "run.json")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) < 10 or any(not isinstance(value, str) for value in scenes):
        raise ValueError("authenticated source run does not name ten calibration scenes")
    if len(set(scenes[:10])) != 10:
        raise ValueError("authenticated source run scene identities are not unique")
    return list(scenes[:10])


def build_sources(args: argparse.Namespace) -> Path:
    if args.scene_limit != 10:
        raise ValueError("source stage requires exactly ten scenes")
    completion_source = args.source_run / "manifests" / "calibration_complete.json"
    if not completion_source.is_file():
        raise ValueError("authenticated CVA02 calibration completion is missing")
    source_run_digest = _sha256_file(completion_source)
    scenes = _source_scene_order(args.source_run)
    manifest_path = args.run_root / "manifests" / "source_manifest.json"
    completion_path = args.run_root / "manifests" / "source_complete.json"
    unsigned = {
        "schema": "variational_camera_latent.source_complete.v1",
        "stage": "source",
        "scenes": scenes,
        "source_run": str(args.source_run),
        "source_run_digest": source_run_digest,
        "git_commit": read_git_commit(ROOT),
    }
    if manifest_path.is_file():
        unsigned["source_manifest_sha256"] = _sha256_file(manifest_path)
    if load_exact_completion(completion_path, unsigned) is not None:
        return args.run_root
    if completion_path.exists():
        raise ValueError("existing source completion does not match requested provenance")
    records: list[SourceShardRecord] = []
    for index, scene in enumerate(scenes):
        destination = args.run_root / "prediction_only" / "source" / f"{scene}.npz"
        if destination.exists():
            raise ValueError(f"uncommitted source shard already exists: {destination}")
        record = build_scene_source_shard(
            args.source_run / "predictions" / scene,
            destination,
            role="train" if index < 8 else "validation",
        )
        records.append(record)
        print(f"[vrfm] source {index + 1}/10 {scene}", flush=True)
    write_source_manifest(
        manifest_path,
        dataset_root=args.run_root / "prediction_only" / "source",
        records=records,
        source_run_digest=source_run_digest,
    )
    unsigned["source_manifest_sha256"] = _sha256_file(manifest_path)
    write_completion(completion_path, unsigned)
    return args.run_root


def _source_manifest(run_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = run_root / "manifests" / "source_manifest.json"
    payload = _read_json(path)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 10 or any(not isinstance(row, dict) for row in records):
        raise ValueError("source manifest must contain exactly ten records")
    return payload, records


def _camera_head(checkpoint_dir: Path, device_name: str):
    from pre_experiments.camera_velocity_ambiguity_02.predict import load_local_camera_model

    device = torch.device(device_name)
    model = load_local_camera_model(checkpoint_dir, device)
    head = model.camera_head
    head.eval()
    del model
    return head


def _candidate_or_generate(
    source_path: Path,
    checkpoint_path: Path,
    destination: Path,
    *,
    args: argparse.Namespace,
    camera_head,
) -> dict[str, object]:
    checkpoint_sha = _sha256_file(checkpoint_path)
    if destination.is_file():
        arrays = load_candidate_shard(destination)
        if str(arrays["checkpoint_sha256"]) != checkpoint_sha:
            raise ValueError("existing candidate shard was produced by another checkpoint")
        record_sha = _sha256_file(destination)
        scene = str(arrays["source_sample_ids"][0]).split(":", 1)[0]
    else:
        record = generate_scene_candidates(
            source_path,
            checkpoint_path,
            destination,
            samples=args.samples,
            steps=args.heun_steps,
            seed=args.seed,
            device=args.device,
            camera_head=camera_head,
        )
        scene, record_sha = record.scene, record.sha256
    return {"scene": scene, "path": str(destination), "sha256": record_sha}


def _train_and_sample(args: argparse.Namespace, *, smoke: bool) -> Path:
    _, records = _source_manifest(args.run_root)
    expected_limit = 1 if smoke else 10
    if args.scene_limit != expected_limit:
        raise ValueError(f"{'smoke' if smoke else 'calibration'} requires scene_limit={expected_limit}")
    selected = records[:1] if smoke else records
    training_records = selected if smoke else records[:8]
    stage = "smoke" if smoke else "calibration"
    training_root = args.run_root / "training" / stage
    max_steps = args.smoke_steps if smoke else args.calibration_steps
    result = train_models(
        TrainConfig(
            source_paths=tuple(Path(row["path"]) for row in training_records),
            run_root=training_root,
            max_steps=max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            d_model=args.d_model,
            z_dim=args.z_dim,
            layers=args.layers,
            heads=args.heads,
            beta_max=args.beta_max,
            checkpoint_interval=args.checkpoint_interval,
            git_commit=read_git_commit(ROOT),
        )
    )
    head = _camera_head(args.checkpoint_dir, args.device)
    candidate_root = args.run_root / "prediction_only" / f"{stage}_candidates"
    deterministic_root = args.run_root / "prediction_only" / f"{stage}_deterministic"
    candidate_records: list[dict[str, object]] = []
    for index, row in enumerate(selected):
        source_path = Path(row["path"])
        scene = str(row["scene"])
        candidate_path = candidate_root / f"{scene}.npz"
        candidate_records.append(
            _candidate_or_generate(
                source_path,
                result.checkpoint_path,
                candidate_path,
                args=args,
                camera_head=head,
            )
        )
        deterministic_path = deterministic_root / f"{scene}.npz"
        if not deterministic_path.exists():
            generate_deterministic_candidates(
                source_path,
                result.checkpoint_path,
                deterministic_path,
                steps=args.heun_steps,
                device=args.device,
                camera_head=head,
            )
        print(f"[vrfm] {stage} candidates {index + 1}/{len(selected)} {scene}", flush=True)
    del head
    manifest_path = args.run_root / "manifests" / f"{stage}_prediction_manifest.json"
    manifest = {
        "schema": f"variational_camera_latent.{stage}_prediction_manifest.v1",
        "stage": stage,
        "scenes": [row["scene"] for row in selected],
        "checkpoint_path": str(result.checkpoint_path),
        "checkpoint_sha256": _sha256_file(result.checkpoint_path),
        "completed_step": result.completed_step,
        "samples": args.samples,
        "heun_steps": args.heun_steps,
        "records": candidate_records,
    }
    _atomic_json(manifest_path, manifest)
    completion_path = args.run_root / "manifests" / f"{stage}_complete.json"
    write_completion(
        completion_path,
        {
            "schema": f"variational_camera_latent.{stage}_complete.v1",
            "stage": stage,
            "scene_count": len(selected),
            "prediction_manifest_sha256": _sha256_file(manifest_path),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "source_manifest_sha256": _sha256_file(args.run_root / "manifests" / "source_manifest.json"),
            "git_commit": read_git_commit(ROOT),
        },
    )
    return args.run_root


def build_privileged_sidecars(args: argparse.Namespace) -> Path:
    if args.scene_limit != 10:
        raise ValueError("privileged stage requires exactly ten scenes")
    _, sources = _source_manifest(args.run_root)
    prediction = _read_json(args.run_root / "manifests" / "calibration_prediction_manifest.json")
    prediction_records = prediction.get("records")
    if not isinstance(prediction_records, list) or len(prediction_records) != 10:
        raise ValueError("calibration prediction manifest is incomplete")
    records: list[dict[str, object]] = []
    deterministic_records: list[dict[str, object]] = []
    best_improvements: list[float] = []
    deterministic_improvements: list[float] = []
    for index, (source, candidate) in enumerate(zip(sources, prediction_records)):
        scene = str(source["scene"])
        if candidate.get("scene") != scene:
            raise ValueError("source and candidate scene order mismatch")
        destination = args.run_root / "privileged_labels" / f"{scene}.npz"
        if destination.is_file():
            arrays = load_privileged_sidecar(destination)
            sha = _sha256_file(destination)
        else:
            record = write_privileged_scene_sidecar(
                Path(source["path"]),
                Path(candidate["path"]),
                args.prepared_root / scene,
                destination,
            )
            arrays, sha = load_privileged_sidecar(destination), record.sha256
        best_improvements.extend(np.max(arrays["relative_improvement"], axis=1).tolist())
        records.append({"scene": scene, "path": str(destination), "sha256": sha})
        deterministic_destination = (
            args.run_root / "privileged_labels" / "deterministic" / f"{scene}.npz"
        )
        if deterministic_destination.is_file():
            deterministic_arrays = load_privileged_sidecar(deterministic_destination)
            deterministic_sha = _sha256_file(deterministic_destination)
        else:
            deterministic_record = write_privileged_deterministic_sidecar(
                Path(source["path"]),
                args.run_root / "prediction_only" / "calibration_deterministic" / f"{scene}.npz",
                args.prepared_root / scene,
                deterministic_destination,
            )
            deterministic_arrays = load_privileged_sidecar(deterministic_destination)
            deterministic_sha = deterministic_record.sha256
        deterministic_improvements.extend(
            deterministic_arrays["relative_improvement"][:, 0].tolist()
        )
        deterministic_records.append(
            {"scene": scene, "path": str(deterministic_destination), "sha256": deterministic_sha}
        )
        print(f"[vrfm] privileged {index + 1}/10 {scene}", flush=True)
    manifest_path = args.run_root / "manifests" / "privileged_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "variational_camera_latent.privileged_manifest.v1",
            "scene_count": 10,
            "records": records,
            "deterministic_records": deterministic_records,
            "best_relative_improvements": best_improvements,
            "deterministic_relative_improvements": deterministic_improvements,
        },
    )
    write_completion(
        args.run_root / "manifests" / "privileged_complete.json",
        {
            "schema": "variational_camera_latent.privileged_complete.v1",
            "stage": "privileged",
            "scene_count": 10,
            "privileged_manifest_sha256": _sha256_file(manifest_path),
            "prediction_manifest_sha256": _sha256_file(
                args.run_root / "manifests" / "calibration_prediction_manifest.json"
            ),
        },
    )
    return args.run_root


def publish_report(args: argparse.Namespace) -> Path:
    calibration = _read_json(args.run_root / "manifests" / "calibration_prediction_manifest.json")
    records = calibration.get("records")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("calibration prediction manifest is incomplete")
    ratios: list[float] = []
    sensitivities: list[float] = []
    for row in records:
        arrays = load_candidate_shard(Path(row["path"]))
        analyses = analyze_candidate_shard(Path(row["path"]))["overlaps"]
        ratios.extend(float(value["one_to_two_sse_ratio"]) for value in analyses)
        delta = arrays["corrected_camera_tokens"] - arrays["source_long_tokens"][:, None]
        sensitivities.append(float(np.sqrt(np.mean(np.var(delta, axis=1)))))
    prediction_manifest = args.run_root / "manifests" / "prediction_manifest.json"
    _atomic_json(
        prediction_manifest,
        {
            "schema": "variational_camera_latent.prediction_manifest.v1",
            "scene_count": 10,
            "candidate_count": 10 * 8 * args.samples,
            "z_sensitivity": float(np.mean(sensitivities)),
            "median_one_to_two_sse_ratio": float(np.median(ratios)),
            "records": records,
        },
    )
    report_path = args.run_root / "reports" / "exploration_report.json"
    summarize_run(
        prediction_manifest,
        args.run_root / "manifests" / "privileged_manifest.json",
        report_path,
    )
    write_completion(
        args.run_root / "manifests" / "report_complete.json",
        {
            "schema": "variational_camera_latent.report_complete.v1",
            "stage": "report",
            "prediction_manifest_sha256": _sha256_file(prediction_manifest),
            "privileged_manifest_sha256": _sha256_file(
                args.run_root / "manifests" / "privileged_manifest.json"
            ),
            "report_sha256": _sha256_file(report_path),
        },
    )
    return args.run_root


def run_alpha_scan(args: argparse.Namespace) -> Path:
    """Diagnose whether short-window directions help at a smaller latent step."""
    if args.scene_limit != 10:
        raise ValueError("alpha-scan stage requires exactly ten scenes")
    if not (args.run_root / "verified_completion.json").is_file():
        raise ValueError("alpha scan requires a verified Phase 1 run")
    _, sources = _source_manifest(args.run_root)
    source_run_manifest = _read_json(args.source_run / "manifests" / "run.json")
    checkpoint_sha256 = source_run_manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("source run has no authenticated checkpoint digest")

    prediction_root = args.run_root / "prediction_only" / "alpha_scan_full_context"
    prediction_records: list[dict[str, object]] = []
    head = None
    for index, source in enumerate(sources):
        scene = str(source["scene"])
        source_path = Path(source["path"])
        destination = prediction_root / f"{scene}.npz"
        if destination.is_file():
            arrays = load_alpha_scan_candidates(destination)
            if str(arrays["checkpoint_sha256"]) != checkpoint_sha256:
                raise ValueError("existing alpha-scan candidate checkpoint does not match")
            if str(arrays["source_shard_sha256"]) != _sha256_file(source_path):
                raise ValueError("existing alpha-scan candidate source does not match")
        else:
            if head is None:
                head = _camera_head(args.checkpoint_dir, args.device)
            generate_alpha_scan_candidates(
                source_path,
                destination,
                camera_head=head,
                checkpoint_sha256=checkpoint_sha256,
                device=args.device,
                alphas=DEFAULT_ALPHAS,
            )
            arrays = load_alpha_scan_candidates(destination)
        source_arrays = load_source_shard(source_path)
        if not np.array_equal(
            arrays["source_sample_ids"], source_arrays["sample_ids"]
        ):
            raise ValueError("alpha-scan candidate sample IDs do not match source")
        prediction_records.append(
            {"scene": scene, "path": str(destination), "sha256": _sha256_file(destination)}
        )
        print(f"[vrfm] alpha-scan decode {index + 1}/10 {scene}", flush=True)
    if head is not None:
        del head

    prediction_manifest_path = (
        args.run_root / "manifests" / "alpha_scan_full_context_prediction_manifest.json"
    )
    _atomic_json(
        prediction_manifest_path,
        {
            "schema": "variational_camera_latent.alpha_scan_full_context_prediction_manifest.v1",
            "scene_count": 10,
            "overlap_count": 80,
            "alphas": list(DEFAULT_ALPHAS),
            "checkpoint_sha256": checkpoint_sha256,
            "records": prediction_records,
        },
    )

    privileged_root = args.run_root / "privileged_labels" / "alpha_scan_full_context"
    privileged_records: list[dict[str, object]] = []
    privileged_paths: list[Path] = []
    for index, (source, prediction) in enumerate(zip(sources, prediction_records)):
        scene = str(source["scene"])
        destination = privileged_root / f"{scene}.npz"
        if destination.is_file():
            arrays = load_alpha_scan_privileged(destination)
            source_arrays = load_source_shard(Path(source["path"]))
            if not np.array_equal(
                arrays["sample_ids"], source_arrays["sample_ids"]
            ):
                raise ValueError("existing alpha-scan sidecar sample IDs do not match")
        else:
            write_alpha_scan_privileged_sidecar(
                Path(source["path"]),
                Path(prediction["path"]),
                args.prepared_root / scene,
                destination,
            )
            load_alpha_scan_privileged(destination)
        privileged_paths.append(destination)
        privileged_records.append(
            {"scene": scene, "path": str(destination), "sha256": _sha256_file(destination)}
        )
        print(f"[vrfm] alpha-scan privileged {index + 1}/10 {scene}", flush=True)

    privileged_manifest_path = (
        args.run_root / "manifests" / "alpha_scan_full_context_privileged_manifest.json"
    )
    _atomic_json(
        privileged_manifest_path,
        {
            "schema": "variational_camera_latent.alpha_scan_full_context_privileged_manifest.v1",
            "scene_count": 10,
            "overlap_count": 80,
            "records": privileged_records,
        },
    )
    report_path = args.run_root / "reports" / "alpha_scan_full_context_report.json"
    report = write_alpha_scan_report(
        privileged_paths,
        report_path,
        min_improvement=args.alpha_min_improvement,
    )
    verified = {
        "schema": "variational_camera_latent.alpha_scan_full_context_verified_completion.v1",
        "scene_count": 10,
        "overlap_count": 80,
        "diagnosis": report["diagnosis"],
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "privileged_manifest_sha256": _sha256_file(privileged_manifest_path),
        "report_sha256": _sha256_file(report_path),
        "phase1_completion_sha256": _sha256_file(args.run_root / "verified_completion.json"),
        "git_commit": read_git_commit(ROOT),
    }
    write_completion(
        args.run_root / "alpha_scan_full_context_verified_completion.json", verified
    )
    return args.run_root


def run_vrfm_residual_alpha_scan(args: argparse.Namespace) -> Path:
    """Re-evaluate every existing VRFM direction in full 500-frame context."""
    if args.scene_limit != 10:
        raise ValueError("VRFM residual alpha scan requires exactly ten scenes")
    if args.residual_scan_batch_size < 1:
        raise ValueError("VRFM residual alpha scan batch size must be positive")
    if (
        not np.isfinite(args.alpha_min_improvement)
        or args.alpha_min_improvement <= 0.0
    ):
        raise ValueError("VRFM residual minimum improvement must be positive")
    phase1_completion = args.run_root / "verified_completion.json"
    phase1 = _load_signed_completion(
        phase1_completion,
        schema="variational_camera_latent.verified_completion.v1",
    )
    bound_artifacts = {
        "prediction_manifest_sha256": args.run_root
        / "manifests"
        / "prediction_manifest.json",
        "privileged_manifest_sha256": args.run_root
        / "manifests"
        / "privileged_manifest.json",
        "report_sha256": args.run_root / "reports" / "exploration_report.json",
    }
    for field, path in bound_artifacts.items():
        if phase1.get(field) != _sha256_file(path):
            raise ValueError(f"verified Phase 1 does not bind {path}")
    source_manifest, sources = _source_manifest(args.run_root)
    source_manifest_path = args.run_root / "manifests" / "source_manifest.json"
    source_completion = _load_signed_completion(
        args.run_root / "manifests" / "source_complete.json",
        schema="variational_camera_latent.source_complete.v1",
    )
    if source_completion.get("source_manifest_sha256") != _sha256_file(
        source_manifest_path
    ):
        raise ValueError("source completion does not bind the current source manifest")
    source_run_completion_path = args.source_run / "manifests" / "calibration_complete.json"
    source_run_completion_sha256 = _sha256_file(source_run_completion_path)
    if (
        Path(str(source_completion.get("source_run"))).resolve()
        != args.source_run.resolve()
        or source_completion.get("source_run_digest")
        != source_run_completion_sha256
        or source_manifest.get("source_run_digest")
        != source_run_completion_sha256
    ):
        raise ValueError("source shards are not bound to the requested CVA02 source run")
    candidate_manifest_path = (
        args.run_root / "manifests" / "calibration_prediction_manifest.json"
    )
    candidate_manifest = _read_json(candidate_manifest_path)
    candidate_records = candidate_manifest.get("records")
    if (
        not isinstance(candidate_records, list)
        or len(candidate_records) != 10
        or any(not isinstance(row, dict) for row in candidate_records)
    ):
        raise ValueError("calibration prediction manifest is incomplete")
    calibration_completion = _load_signed_completion(
        args.run_root / "manifests" / "calibration_complete.json",
        schema="variational_camera_latent.calibration_complete.v1",
    )
    if calibration_completion.get("prediction_manifest_sha256") != _sha256_file(
        candidate_manifest_path
    ) or calibration_completion.get("source_manifest_sha256") != _sha256_file(
        source_manifest_path
    ):
        raise ValueError("calibration completion does not bind current inputs")
    phase1_prediction = _read_json(
        args.run_root / "manifests" / "prediction_manifest.json"
    )
    if phase1_prediction.get("records") != candidate_records:
        raise ValueError("verified Phase 1 and calibration candidate records differ")
    source_run_manifest = _read_json(args.source_run / "manifests" / "run.json")
    if (
        source_run_manifest.get("schema") != "camera_velocity_ambiguity_02.run.v1"
        or source_run_manifest.get("run_id") != args.source_run.name
    ):
        raise ValueError("source run identity does not match its directory")
    camera_head_checkpoint_sha256 = source_run_manifest.get("checkpoint_sha256")
    if (
        not isinstance(camera_head_checkpoint_sha256, str)
        or len(camera_head_checkpoint_sha256) != 64
    ):
        raise ValueError("source run has no authenticated Camera Head checkpoint digest")
    actual_camera_checkpoint_sha256 = _sha256_file(
        find_checkpoint(args.checkpoint_dir)
    )
    if actual_camera_checkpoint_sha256 != camera_head_checkpoint_sha256:
        raise ValueError("local Camera Head checkpoint digest does not match source run")
    producer_git_commit = read_git_commit(ROOT)

    prediction_root = (
        args.run_root
        / "prediction_only"
        / "vrfm_residual_alpha_scan_full_context"
    )
    prediction_records: list[dict[str, object]] = []
    head = None
    expected_samples: int | None = None
    for index, (source_record, candidate_record) in enumerate(
        zip(sources, candidate_records)
    ):
        scene = str(source_record["scene"])
        if candidate_record.get("scene") != scene:
            raise ValueError("source and VRFM candidate scene order mismatch")
        source_path = Path(source_record["path"])
        candidate_path = Path(str(candidate_record["path"]))
        if _sha256_file(source_path) != source_record.get("sha256"):
            raise ValueError("source shard digest does not match its manifest")
        if _sha256_file(candidate_path) != candidate_record.get("sha256"):
            raise ValueError("VRFM candidate shard digest does not match its manifest")
        original_candidate = load_candidate_shard(candidate_path)
        samples = int(original_candidate["z"].shape[1])
        if expected_samples is None:
            expected_samples = samples
        elif samples != expected_samples:
            raise ValueError("VRFM candidate shards use different sample counts")
        destination = prediction_root / f"{scene}.npz"
        if destination.is_file():
            arrays = load_vrfm_residual_alpha_scan(destination)
            if not np.array_equal(
                arrays["alphas"], np.asarray(DEFAULT_ALPHAS, dtype=np.float64)
            ):
                raise ValueError("existing VRFM residual alpha grid does not match")
            if str(arrays["source_shard_sha256"]) != _sha256_file(source_path):
                raise ValueError("existing VRFM residual output source digest does not match")
            if str(arrays["candidate_shard_sha256"]) != _sha256_file(candidate_path):
                raise ValueError("existing VRFM residual output candidate digest does not match")
            if (
                str(arrays["vrfm_checkpoint_sha256"])
                != str(original_candidate["checkpoint_sha256"])
            ):
                raise ValueError("existing VRFM residual output checkpoint does not match")
            if (
                str(arrays["camera_head_checkpoint_sha256"])
                != camera_head_checkpoint_sha256
            ):
                raise ValueError("existing VRFM residual Camera Head checkpoint does not match")
            if str(arrays["producer_git_commit"]) != producer_git_commit:
                raise ValueError("existing VRFM residual output was made by another commit")
        else:
            if head is None:
                head = _camera_head(args.checkpoint_dir, args.device)
            generate_vrfm_residual_alpha_scan(
                source_path,
                candidate_path,
                destination,
                camera_head=head,
                camera_head_checkpoint_sha256=camera_head_checkpoint_sha256,
                producer_git_commit=producer_git_commit,
                device=args.device,
                alphas=DEFAULT_ALPHAS,
                batch_size=args.residual_scan_batch_size,
            )
            arrays = load_vrfm_residual_alpha_scan(destination)
        source_arrays = load_source_shard(source_path)
        if not np.array_equal(arrays["source_sample_ids"], source_arrays["sample_ids"]):
            raise ValueError("VRFM residual output sample IDs do not match source")
        prediction_records.append(
            {
                "scene": scene,
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "source_shard_sha256": _sha256_file(source_path),
                "candidate_shard_sha256": _sha256_file(candidate_path),
            }
        )
        print(
            f"[vrfm] residual-alpha full-context decode {index + 1}/10 {scene}",
            flush=True,
        )
    if head is not None:
        del head
    assert expected_samples is not None

    prediction_manifest_path = (
        args.run_root
        / "manifests"
        / "vrfm_residual_alpha_scan_full_context_prediction_manifest.json"
    )
    _atomic_json(
        prediction_manifest_path,
        {
            "schema": "variational_camera_latent.vrfm_residual_alpha_scan_full_context_prediction_manifest.v1",
            "scene_count": 10,
            "overlap_count": 80,
            "samples_per_overlap": expected_samples,
            "direction_count": 80 * expected_samples,
            "grid_cell_count": 80 * expected_samples * len(DEFAULT_ALPHAS),
            "unique_pose_candidate_count": 80
            * (1 + expected_samples * (len(DEFAULT_ALPHAS) - 1)),
            "alphas": list(DEFAULT_ALPHAS),
            "decode_context_frames": 500,
            "camera_iterations": 4,
            "producer_git_commit": producer_git_commit,
            "camera_head_checkpoint_sha256": camera_head_checkpoint_sha256,
            "source_manifest_sha256": _sha256_file(
                args.run_root / "manifests" / "source_manifest.json"
            ),
            "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
            "records": prediction_records,
        },
    )

    privileged_root = (
        args.run_root
        / "privileged_labels"
        / "vrfm_residual_alpha_scan_full_context"
    )
    privileged_records: list[dict[str, object]] = []
    privileged_paths: list[Path] = []
    for index, (source_record, prediction_record) in enumerate(
        zip(sources, prediction_records)
    ):
        scene = str(source_record["scene"])
        prediction_path = Path(str(prediction_record["path"]))
        destination = privileged_root / f"{scene}.npz"
        if destination.is_file():
            arrays = load_vrfm_residual_privileged(destination)
            if str(arrays["prediction_sha256"]) != _sha256_file(prediction_path):
                raise ValueError("existing VRFM residual sidecar prediction digest does not match")
            source_arrays = load_source_shard(Path(source_record["path"]))
            if not np.array_equal(
                arrays["source_sample_ids"], source_arrays["sample_ids"]
            ):
                raise ValueError("existing VRFM residual sidecar sample IDs do not match")
            if str(arrays["prepared_gt_sha256"]) != prepared_gt_sha256(
                args.prepared_root / scene
            ):
                raise ValueError("existing VRFM residual sidecar GT digest does not match")
        else:
            write_vrfm_residual_privileged_sidecar(
                Path(source_record["path"]),
                prediction_path,
                args.prepared_root / scene,
                destination,
            )
            load_vrfm_residual_privileged(destination)
        privileged_paths.append(destination)
        privileged_records.append(
            {"scene": scene, "path": str(destination), "sha256": _sha256_file(destination)}
        )
        print(f"[vrfm] residual-alpha privileged {index + 1}/10 {scene}", flush=True)

    privileged_manifest_path = (
        args.run_root
        / "manifests"
        / "vrfm_residual_alpha_scan_full_context_privileged_manifest.json"
    )
    _atomic_json(
        privileged_manifest_path,
        {
            "schema": "variational_camera_latent.vrfm_residual_alpha_scan_full_context_privileged_manifest.v1",
            "scene_count": 10,
            "overlap_count": 80,
            "samples_per_overlap": expected_samples,
            "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
            "records": privileged_records,
        },
    )

    report_path = (
        args.run_root
        / "reports"
        / "vrfm_residual_alpha_scan_full_context_report.json"
    )
    report = write_vrfm_residual_report(
        privileged_paths,
        report_path,
        min_improvement=args.alpha_min_improvement,
    )
    verified = {
        "schema": "variational_camera_latent.vrfm_residual_alpha_scan_full_context_verified_completion.v1",
        "scene_count": 10,
        "overlap_count": 80,
        "direction_count": 80 * expected_samples,
        "grid_cell_count": 80 * expected_samples * len(DEFAULT_ALPHAS),
        "unique_pose_candidate_count": 80
        * (1 + expected_samples * (len(DEFAULT_ALPHAS) - 1)),
        "diagnosis": report["diagnosis"],
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "privileged_manifest_sha256": _sha256_file(privileged_manifest_path),
        "report_sha256": _sha256_file(report_path),
        "phase1_completion_sha256": _sha256_file(phase1_completion),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
        "git_commit": producer_git_commit,
    }
    write_completion(
        args.run_root
        / "vrfm_residual_alpha_scan_full_context_verified_completion.json",
        verified,
    )
    return args.run_root


def verify_completed_run(run_root: Path) -> Path:
    run_root = Path(run_root)
    source = _read_json(run_root / "manifests" / "source_manifest.json")
    prediction = _read_json(run_root / "manifests" / "prediction_manifest.json")
    privileged = _read_json(run_root / "manifests" / "privileged_manifest.json")
    report = _read_json(run_root / "reports" / "exploration_report.json")
    if len(source.get("records", [])) != 10 or len(prediction.get("records", [])) != 10:
        raise ValueError("completed run must contain exactly ten source and candidate scenes")
    if (
        len(privileged.get("records", [])) != 10
        or len(privileged.get("deterministic_records", [])) != 10
        or prediction.get("candidate_count") != 2560
    ):
        raise ValueError("completed run has an invalid candidate or privileged count")
    for row in prediction["records"]:
        path = Path(row["path"])
        if not path.is_relative_to(run_root / "prediction_only") or _sha256_file(path) != row["sha256"]:
            raise ValueError("prediction candidate path or digest is invalid")
        load_candidate_shard(path)
    for row in privileged["records"]:
        path = Path(row["path"])
        if not path.is_relative_to(run_root / "privileged_labels") or _sha256_file(path) != row["sha256"]:
            raise ValueError("privileged path or digest is invalid")
        load_privileged_sidecar(path)
    for row in privileged["deterministic_records"]:
        path = Path(row["path"])
        if not path.is_relative_to(run_root / "privileged_labels") or _sha256_file(path) != row["sha256"]:
            raise ValueError("deterministic privileged path or digest is invalid")
        load_privileged_sidecar(path)
    if report.get("technically_complete") is not True or report.get("signal") not in {
        "PROMISING", "WEAK_SIGNAL", "NO_SIGNAL"
    }:
        raise ValueError("exploration report is incomplete")
    verified = {
        "schema": "variational_camera_latent.verified_completion.v1",
        "scene_count": 10,
        "overlap_count": 80,
        "candidate_count": 2560,
        "signal": report["signal"],
        "prediction_manifest_sha256": _sha256_file(run_root / "manifests" / "prediction_manifest.json"),
        "privileged_manifest_sha256": _sha256_file(run_root / "manifests" / "privileged_manifest.json"),
        "report_sha256": _sha256_file(run_root / "reports" / "exploration_report.json"),
    }
    write_completion(run_root / "verified_completion.json", verified)
    return run_root


def run_stage(args: argparse.Namespace) -> Path:
    args.run_root = Path(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "source":
        return build_sources(args)
    if args.stage == "smoke":
        return _train_and_sample(args, smoke=True)
    if args.stage == "calibration":
        return _train_and_sample(args, smoke=False)
    if args.stage == "privileged":
        return build_privileged_sidecars(args)
    if args.stage == "report":
        return publish_report(args)
    if args.stage == "alpha-scan":
        return run_alpha_scan(args)
    if args.stage == "vrfm-residual-alpha-scan":
        return run_vrfm_residual_alpha_scan(args)
    if args.stage == "verify":
        return verify_completed_run(args.run_root)
    raise ValueError(f"unsupported stage: {args.stage}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "source",
            "smoke",
            "calibration",
            "privileged",
            "report",
            "alpha-scan",
            "vrfm-residual-alpha-scan",
            "verify",
        ),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int, default=10)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("/data/yjh/output/camera_velocity_ambiguity/cva02_20260826T2319CST_7e6fd06"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("/data/yjh/share/datasets/ScanNet/processed_cva02_v1"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("/data/yjh/share/pretrained/VGGT-1B")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=100)
    parser.add_argument("--calibration-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--beta-max", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--heun-steps", type=int, default=16)
    parser.add_argument("--alpha-min-improvement", type=float, default=0.01)
    parser.add_argument("--residual-scan-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = run_stage(args)
    print(f"[vrfm] stage={args.stage} complete run_root={root}", flush=True)


if __name__ == "__main__":
    main()
