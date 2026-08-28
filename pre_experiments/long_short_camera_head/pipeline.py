from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np
import torch

from .data import (
    LongContextRecord,
    load_long_context,
    load_source_records,
    publish_long_context,
    sha256_file,
)
from .evaluate import (
    EVALUATION_SCHEMA,
    evaluate_prediction,
    load_prediction,
    run_long_only_inference,
)
from .labels import build_privileged_labels, load_privileged_labels
from .report import inference_signature_is_long_only, write_report
from .train import (
    TrainConfig,
    configure_trainable_scope,
    load_base_camera_head,
    load_training_example,
    run_training_loop,
    train_camera_head,
)


DEFAULT_RESULT_ROOT = Path("/data/yjh/output/vggt/long_short_camera_head")
MANIFEST_SCHEMA = "long_short_camera_head.data_manifest.v1"
VERIFIED_SCHEMA = "long_short_camera_head.verified_completion.v1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _digest(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_manifest(run_root: Path) -> dict[str, object]:
    manifest = _read_json(Path(run_root) / "manifests" / "data_manifest.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("data manifest schema mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("data manifest is incomplete: expected ten scenes")
    return manifest


def prepare_run(
    *,
    run_root: Path,
    source_run: Path,
    prepared_root: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> Path:
    """Publish ten strict long inputs and ten separate privileged sidecars."""
    run_root = Path(run_root).resolve()
    records = load_source_records(source_run)
    if sum(record.role == "train" for record in records) != 8 or sum(
        record.role == "validation" for record in records
    ) != 2:
        raise ValueError("source split must contain exactly eight train and two validation scenes")
    manifest_path = run_root / "manifests" / "data_manifest.json"
    if manifest_path.is_file():
        existing = _load_manifest(run_root)
        for row in existing["records"]:
            long_path = Path(str(row["long_context_path"]))
            privileged_path = Path(str(row["privileged_path"]))
            if sha256_file(long_path) != row["long_context_sha256"]:
                raise ValueError("existing long context digest mismatch")
            if sha256_file(privileged_path) != row["privileged_sha256"]:
                raise ValueError("existing privileged digest mismatch")
            load_long_context(long_path)
            load_privileged_labels(privileged_path)
        return manifest_path

    camera_head, checkpoint_sha256 = load_base_camera_head(checkpoint_dir)
    camera_head = camera_head.to(device).eval()
    rows: list[dict[str, object]] = []
    for record in records:
        long_path = run_root / "data" / "long_context" / f"{record.scene}.npz"
        privileged_path = run_root / "data" / "privileged_labels" / f"{record.scene}.npz"
        long_record: LongContextRecord = publish_long_context(record, long_path)
        privileged_record = build_privileged_labels(
            record.path,
            Path(prepared_root) / record.scene,
            camera_head,
            privileged_path,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
        )
        rows.append(
            {
                "scene": record.scene,
                "role": record.role,
                "source_path": str(record.path),
                "source_sha256": record.sha256,
                "long_context_path": str(long_record.path.resolve()),
                "long_context_sha256": long_record.sha256,
                "privileged_path": str(privileged_record.path.resolve()),
                "privileged_sha256": privileged_record.sha256,
                "teacher_frame_count": privileged_record.teacher_frame_count,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "git_revision": _git_revision(),
        "source_run": str(Path(source_run).resolve()),
        "source_manifest_sha256": sha256_file(
            Path(source_run) / "manifests" / "source_manifest.json"
        ),
        "prepared_root": str(Path(prepared_root).resolve()),
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "base_checkpoint_sha256": checkpoint_sha256,
        "records": rows,
    }
    _atomic_json(manifest_path, payload)
    return manifest_path


def _pairs_for_role(
    manifest: dict[str, object], role: str
) -> tuple[tuple[Path, Path], ...]:
    rows = manifest["records"]
    assert isinstance(rows, list)
    return tuple(
        (Path(str(row["long_context_path"])), Path(str(row["privileged_path"])))
        for row in rows
        if row["role"] == role
    )


def run_smoke(
    *,
    run_root: Path,
    checkpoint_dir: Path,
    device: torch.device,
    max_steps: int,
    learning_rate: float,
) -> Path:
    manifest = _load_manifest(run_root)
    pair = _pairs_for_role(manifest, "train")[0]
    example = load_training_example(*pair)
    model, base_digest = load_base_camera_head(checkpoint_dir)
    trainable_names = configure_trainable_scope(model)
    configuration = {
        "schema": "long_short_camera_head.smoke.v1",
        "scene": example.scene,
        "base_checkpoint_sha256": base_digest,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "trainable_parameter_names": list(trainable_names),
    }
    data_identity = {
        "long_context_sha256": sha256_file(pair[0]),
        "privileged_sha256": sha256_file(pair[1]),
    }
    result = run_training_loop(
        model=model,
        train_examples=(example,),
        validation_examples=(example,),
        run_root=Path(run_root) / "smoke",
        variant="long_short",
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        checkpoint_interval=max_steps,
        patience=max_steps,
        seed=20260828,
        device=device,
        config_digest=_digest(configuration),
        data_digest=_digest(data_identity),
    )
    values = (
        result.initial_training_loss,
        result.final_training_loss,
        result.best_validation_rms,
    )
    if not np.isfinite(values).all() or result.final_training_loss > result.initial_training_loss * 1.01:
        raise ValueError("smoke training failed to remain finite and reduce its objective")
    payload = {
        **configuration,
        **data_identity,
        "initial_training_loss": result.initial_training_loss,
        "final_training_loss": result.final_training_loss,
        "best_validation_rms": result.best_validation_rms,
        "completed_step": result.completed_step,
        "best_checkpoint_sha256": sha256_file(result.best_checkpoint),
    }
    completion = Path(run_root) / "smoke" / "completed.json"
    _atomic_json(completion, payload)
    return completion


def run_calibration(
    *,
    run_root: Path,
    checkpoint_dir: Path,
    variant: str,
    device: torch.device,
    max_steps: int,
    learning_rate: float,
    checkpoint_interval: int,
    patience: int,
) -> Path:
    manifest = _load_manifest(run_root)
    result = train_camera_head(
        TrainConfig(
            checkpoint_dir=checkpoint_dir,
            run_root=Path(run_root) / "training" / variant,
            variant=variant,
            train_pairs=_pairs_for_role(manifest, "train"),
            validation_pairs=_pairs_for_role(manifest, "validation"),
            max_steps=max_steps,
            learning_rate=learning_rate,
            checkpoint_interval=checkpoint_interval,
            patience=patience,
            device=device,
        )
    )
    completion = Path(run_root) / "training" / variant / "completed.json"
    _atomic_json(
        completion,
        {
            "schema": "long_short_camera_head.calibration_completion.v1",
            "variant": variant,
            "completed_step": result.completed_step,
            "initial_training_loss": result.initial_training_loss,
            "final_training_loss": result.final_training_loss,
            "best_validation_rms": result.best_validation_rms,
            "best_checkpoint_sha256": sha256_file(result.best_checkpoint),
        },
    )
    return completion


def run_evaluation(
    *,
    run_root: Path,
    checkpoint_dir: Path,
    variant: str,
    device: torch.device,
) -> Path:
    manifest = _load_manifest(run_root)
    checkpoint = Path(run_root) / "training" / variant / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"{variant} best checkpoint is missing")
    rows = manifest["records"]
    assert isinstance(rows, list)
    output_rows: list[dict[str, object]] = []
    for row in rows:
        if row["role"] != "validation":
            continue
        scene = str(row["scene"])
        prediction_path = Path(run_root) / "predictions" / variant / f"{scene}.npz"
        metrics_path = Path(run_root) / "evaluation" / variant / f"{scene}.json"
        prediction = run_long_only_inference(
            Path(str(row["long_context_path"])),
            checkpoint,
            checkpoint_dir,
            prediction_path,
            device,
        )
        evaluation = evaluate_prediction(
            prediction.path,
            Path(str(row["privileged_path"])),
            metrics_path,
        )
        output_rows.append(
            {
                "scene": scene,
                "prediction_sha256": prediction.sha256,
                "evaluation_sha256": sha256_file(evaluation.path),
            }
        )
    completion = Path(run_root) / "evaluation" / variant / "completed.json"
    _atomic_json(
        completion,
        {
            "schema": "long_short_camera_head.evaluation_completion.v1",
            "variant": variant,
            "checkpoint_sha256": sha256_file(checkpoint),
            "records": output_rows,
        },
    )
    return completion


def verify_completed_run(run_root: Path) -> Path:
    run_root = Path(run_root).resolve()
    try:
        manifest = _load_manifest(run_root)
    except ValueError as error:
        raise ValueError("run is incomplete: data manifest") from error
    rows = manifest["records"]
    assert isinstance(rows, list)
    roles = [str(row["role"]) for row in rows]
    if roles.count("train") != 8 or roles.count("validation") != 2:
        raise ValueError("run is incomplete: split mismatch")
    smoke_completion = run_root / "smoke" / "completed.json"
    if not smoke_completion.is_file():
        raise ValueError("run is incomplete: smoke gate")
    smoke = _read_json(smoke_completion)
    smoke_values = [
        float(smoke.get(name, float("nan")))
        for name in ("initial_training_loss", "final_training_loss", "best_validation_rms")
    ]
    if not np.isfinite(smoke_values).all():
        raise ValueError("run is incomplete: smoke metrics")
    artifact_rows: list[dict[str, str]] = []
    for row in rows:
        long_path = Path(str(row["long_context_path"]))
        privileged_path = Path(str(row["privileged_path"]))
        if sha256_file(long_path) != row["long_context_sha256"]:
            raise ValueError("run is incomplete: long context digest mismatch")
        if sha256_file(privileged_path) != row["privileged_sha256"]:
            raise ValueError("run is incomplete: privileged digest mismatch")
        long_context = load_long_context(long_path)
        labels = load_privileged_labels(privileged_path)
        if str(long_context["source_sha256"]) != str(labels["source_sha256"]):
            raise ValueError("run is incomplete: joined data source mismatch")
    validation_scenes = sorted(str(row["scene"]) for row in rows if row["role"] == "validation")
    for variant in ("gt_only", "long_short"):
        checkpoint = run_root / "training" / variant / "checkpoints" / "best.pt"
        completion = run_root / "training" / variant / "completed.json"
        evaluation_completion = run_root / "evaluation" / variant / "completed.json"
        if not checkpoint.is_file() or not completion.is_file() or not evaluation_completion.is_file():
            raise ValueError(f"run is incomplete: {variant} training/evaluation")
        checkpoint_digest = sha256_file(checkpoint)
        for scene in validation_scenes:
            prediction_path = run_root / "predictions" / variant / f"{scene}.npz"
            metrics_path = run_root / "evaluation" / variant / f"{scene}.json"
            prediction = load_prediction(prediction_path)
            metrics = _read_json(metrics_path)
            if metrics.get("schema") != EVALUATION_SCHEMA:
                raise ValueError("run is incomplete: evaluation schema mismatch")
            if str(prediction["checkpoint_sha256"]) != checkpoint_digest:
                raise ValueError("run is incomplete: prediction checkpoint mismatch")
            if metrics.get("checkpoint_sha256") != checkpoint_digest:
                raise ValueError("run is incomplete: evaluation checkpoint mismatch")
            artifact_rows.append(
                {
                    "variant": variant,
                    "scene": scene,
                    "checkpoint_sha256": checkpoint_digest,
                    "prediction_sha256": sha256_file(prediction_path),
                    "evaluation_sha256": sha256_file(metrics_path),
                }
            )
    if not inference_signature_is_long_only():
        raise ValueError("run is incomplete: inference leakage audit failed")
    report_path = write_report(run_root)
    report = _read_json(report_path)
    if report.get("classification") not in {
        "PROMISING",
        "NO_SOURCE_HEAD_SIGNAL",
        "HEAD_ONLY_INSUFFICIENT",
    }:
        raise ValueError("run is incomplete: report is invalid")
    nonempty_errors = [
        str(path)
        for path in (run_root / "logs").glob("*.err.log")
        if path.stat().st_size > 0
    ]
    if nonempty_errors:
        raise ValueError(f"run is incomplete: nonempty stderr logs: {nonempty_errors}")
    completion_payload = {
        "schema": VERIFIED_SCHEMA,
        "git_revision": manifest.get("git_revision"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "base_checkpoint_sha256": manifest.get("base_checkpoint_sha256"),
        "scene_count": 10,
        "train_scene_count": 8,
        "locked_replay_scene_count": 2,
        "classification": report["classification"],
        "report_sha256": sha256_file(report_path),
        "artifacts": artifact_rows,
        "inference_leakage_audit": True,
    }
    completion_path = run_root / "verified_completion.json"
    _atomic_json(completion_path, completion_payload)
    return completion_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "smoke", "calibration", "evaluate", "report", "verify"),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--variant", choices=("gt_only", "long_short"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    return parser


def _required(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required for this stage")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.stage == "prepare":
        path = prepare_run(
            run_root=args.run_root,
            source_run=_required(args.source_run, "--source-run"),
            prepared_root=_required(args.prepared_root, "--prepared-root"),
            checkpoint_dir=_required(args.checkpoint_dir, "--checkpoint-dir"),
            device=device,
        )
    elif args.stage == "smoke":
        path = run_smoke(
            run_root=args.run_root,
            checkpoint_dir=_required(args.checkpoint_dir, "--checkpoint-dir"),
            device=device,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
        )
    elif args.stage == "calibration":
        if args.variant is None:
            raise ValueError("--variant is required for calibration")
        path = run_calibration(
            run_root=args.run_root,
            checkpoint_dir=_required(args.checkpoint_dir, "--checkpoint-dir"),
            variant=args.variant,
            device=device,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            checkpoint_interval=args.checkpoint_interval,
            patience=args.patience,
        )
    elif args.stage == "evaluate":
        if args.variant is None:
            raise ValueError("--variant is required for evaluation")
        path = run_evaluation(
            run_root=args.run_root,
            checkpoint_dir=_required(args.checkpoint_dir, "--checkpoint-dir"),
            variant=args.variant,
            device=device,
        )
    elif args.stage == "report":
        path = write_report(args.run_root)
    else:
        path = verify_completed_run(args.run_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
