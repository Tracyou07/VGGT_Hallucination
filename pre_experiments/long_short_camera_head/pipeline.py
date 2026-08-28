from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
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
    run_long_only_inference_batch,
)
from .labels import build_privileged_labels, load_privileged_labels
from .losses import LossWeights
from .report import inference_signature_is_long_only, write_report
from .train import (
    CHECKPOINT_SCHEMA,
    TrainConfig,
    configure_trainable_scope,
    load_base_camera_head,
    load_camera_head_checkpoint,
    load_training_example,
    run_training_loop,
    train_camera_head,
)


DEFAULT_RESULT_ROOT = Path("/data/yjh/output/vggt/long_short_camera_head")
MANIFEST_SCHEMA = "long_short_camera_head.data_manifest.v1"
VERIFIED_SCHEMA = "long_short_camera_head.verified_completion.v1"
RUN_CONFIG_SCHEMA = "long_short_camera_head.formal_run_config.v1"
TEST_EVIDENCE_SCHEMA = "long_short_camera_head.test_evidence.v2"
FORMAL_SMOKE_STEPS = 20
FORMAL_CALIBRATION_STEPS = 400
FORMAL_LEARNING_RATE = 2e-6
FORMAL_WEIGHT_DECAY = 1e-4
FORMAL_CHECKPOINT_INTERVAL = 25
FORMAL_PATIENCE = 100
FORMAL_SEED = 20260828
REQUIRED_TEST_SUITES = (
    ("long_short_camera_head", "tests/long_short_camera_head"),
    ("variational_camera_latent", "tests/variational_camera_latent"),
    ("variational_camera_selector", "tests/variational_camera_selector"),
)


def formal_protocol() -> dict[str, object]:
    weights = LossWeights()
    return {
        "schema": "long_short_camera_head.formal_protocol.v1",
        "smoke_scene": "scene0000_00",
        "smoke_steps": FORMAL_SMOKE_STEPS,
        "calibration_steps": FORMAL_CALIBRATION_STEPS,
        "batch_size": 1,
        "precision": "bf16_autocast",
        "optimizer": "AdamW",
        "learning_rate": FORMAL_LEARNING_RATE,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "gradient_clip_norm": 1.0,
        "checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
        "patience": FORMAL_PATIENCE,
        "seed": FORMAL_SEED,
        "train_scene_count": 8,
        "evaluation_scene_count": 10,
        "locked_replay_scene_count": 2,
        "variants": ["gt_only", "long_short"],
        "loss_weights": {
            name: float(getattr(weights, name))
            for name in (
                "gt_translation",
                "relative_translation",
                "rotation",
                "anchor",
                "teacher",
            )
        },
    }


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


def _expected_run_config(
    run_root: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    protocol = formal_protocol()
    manifest_path = Path(run_root) / "manifests" / "data_manifest.json"
    return {
        "schema": RUN_CONFIG_SCHEMA,
        "git_revision": manifest.get("git_revision"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "base_checkpoint_sha256": manifest.get("base_checkpoint_sha256"),
        "data_manifest_sha256": sha256_file(manifest_path),
        "protocol": protocol,
        "protocol_sha256": _digest(protocol),
    }


def _ensure_run_config(run_root: Path, manifest: dict[str, object]) -> Path:
    path = Path(run_root) / "config.json"
    expected = _expected_run_config(run_root, manifest)
    if path.is_file():
        if _read_json(path) != expected:
            raise ValueError("existing formal run configuration does not match")
    else:
        _atomic_json(path, expected)
    return path


def _load_formal_config(run_root: Path, manifest: dict[str, object]) -> dict[str, object]:
    path = Path(run_root) / "config.json"
    if not path.is_file():
        raise ValueError("run is incomplete: formal configuration")
    config = _read_json(path)
    if config != _expected_run_config(run_root, manifest):
        raise ValueError("run is incomplete: formal configuration mismatch")
    return config


def _require_exact(value: object, expected: object, name: str) -> None:
    if value != expected:
        raise ValueError(f"formal {name} must be exactly {expected!r}")


def run_test_preflight(*, run_root: Path) -> Path:
    """Run and bind the focused suite before formal GPU work begins."""
    run_root = Path(run_root).resolve()
    manifest = _load_manifest(run_root)
    config = _load_formal_config(run_root, manifest)
    completion = run_root / "manifests" / "test_evidence.json"
    if completion.is_file():
        evidence = _read_json(completion)
        suites = evidence.get("suites")
        if (
            evidence.get("schema") == TEST_EVIDENCE_SCHEMA
            and evidence.get("training_git_revision") == config["git_revision"]
            and evidence.get("tested_git_revision") == _git_revision()
            and isinstance(suites, list)
            and [
                (str(row.get("name")), str(row.get("path")))
                for row in suites
                if isinstance(row, dict)
            ]
            == list(REQUIRED_TEST_SUITES)
            and all(
                isinstance(row, dict)
                and Path(str(row.get("log_path", ""))).is_file()
                and row.get("log_sha256")
                == sha256_file(Path(str(row.get("log_path", ""))))
                and int(row.get("returncode", -1)) == 0
                for row in suites
            )
        ):
            return completion
        preserved = run_root / "diagnostics" / "test_evidence_v1_preserved.json"
        preserved.parent.mkdir(parents=True, exist_ok=True)
        if not preserved.exists():
            shutil.copy2(completion, preserved)
    try:
        repository_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot locate repository for focused tests") from error
    suite_rows: list[dict[str, object]] = []
    for suite_name, suite_path in REQUIRED_TEST_SUITES:
        command = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            suite_path,
            "-p",
            "test_*.py",
        ]
        completed = subprocess.run(
            command,
            cwd=repository_root,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        combined = completed.stdout + completed.stderr
        log_path = run_root / "logs" / f"tests_{suite_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(combined, encoding="utf-8")
        match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
        test_count = int(match.group(1)) if match else 0
        if (
            completed.returncode != 0
            or test_count < 1
            or not re.search(r"^OK", combined, re.MULTILINE)
        ):
            raise ValueError(f"preflight test suite failed: {suite_name}")
        suite_rows.append(
            {
                "name": suite_name,
                "path": suite_path,
                "command": command,
                "returncode": completed.returncode,
                "test_count": test_count,
                "log_path": str(log_path),
                "log_sha256": sha256_file(log_path),
            }
        )
    evidence = {
        "schema": TEST_EVIDENCE_SCHEMA,
        "training_git_revision": config["git_revision"],
        "tested_git_revision": _git_revision(),
        "test_count": sum(int(row["test_count"]) for row in suite_rows),
        "suites": suite_rows,
    }
    _atomic_json(completion, evidence)
    return completion


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
        _ensure_run_config(run_root, existing)
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
    _ensure_run_config(run_root, payload)
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
    _require_exact(max_steps, FORMAL_SMOKE_STEPS, "smoke update count")
    _require_exact(learning_rate, FORMAL_LEARNING_RATE, "learning rate")
    manifest = _load_manifest(run_root)
    config = _load_formal_config(run_root, manifest)
    completion = Path(run_root) / "smoke" / "completed.json"
    if completion.is_file():
        return completion
    pair = _pairs_for_role(manifest, "train")[0]
    example = load_training_example(*pair)
    _require_exact(example.scene, "scene0000_00", "smoke scene")
    model, base_digest = load_base_camera_head(checkpoint_dir)
    trainable_names = configure_trainable_scope(model)
    configuration = {
        "schema": "long_short_camera_head.smoke.v1",
        "scene": example.scene,
        "base_checkpoint_sha256": base_digest,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "checkpoint_interval": max_steps,
        "patience": max_steps,
        "seed": FORMAL_SEED,
        "precision": "bf16_autocast",
        "formal_protocol_sha256": config["protocol_sha256"],
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
        weight_decay=FORMAL_WEIGHT_DECAY,
        checkpoint_interval=max_steps,
        patience=max_steps,
        seed=FORMAL_SEED,
        device=device,
        config_digest=_digest(configuration),
        data_digest=_digest(data_identity),
    )
    values = (
        result.initial_training_loss,
        result.final_training_loss,
        result.best_validation_rms,
    )
    if (
        not np.isfinite(values).all()
        or result.completed_step != FORMAL_SMOKE_STEPS
        or not result.final_training_loss < result.initial_training_loss
    ):
        raise ValueError("smoke training failed its strict finite/decreasing objective gate")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    try:
        checkpoint_payload = torch.load(
            result.best_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        reloaded = load_camera_head_checkpoint(
            result.best_checkpoint,
            checkpoint_dir,
            torch.device("cpu"),
        )
        checkpoint_model = checkpoint_payload.get("model")
        reload_exact = isinstance(checkpoint_model, dict) and all(
            name in checkpoint_model
            and torch.equal(value.cpu(), checkpoint_model[name].cpu())
            for name, value in reloaded.state_dict().items()
        )
    finally:
        if "reloaded" in locals():
            del reloaded
        if "checkpoint_payload" in locals():
            del checkpoint_payload
        gc.collect()
    if not reload_exact:
        raise ValueError("smoke checkpoint did not reload exactly")
    prediction_path = Path(run_root) / "smoke" / "long_only_prediction.npz"
    prediction_record = run_long_only_inference(
        pair[0],
        result.best_checkpoint,
        checkpoint_dir,
        prediction_path,
        device,
    )
    prediction = load_prediction(prediction_record.path)
    if prediction["pose_encoding"].shape != (500, 9) or not np.isfinite(
        prediction["pose_encoding"]
    ).all():
        raise ValueError("smoke long-only inference is not finite 500-frame output")
    payload = {
        **configuration,
        **data_identity,
        "initial_training_loss": result.initial_training_loss,
        "final_training_loss": result.final_training_loss,
        "best_validation_rms": result.best_validation_rms,
        "best_step": result.best_step,
        "completed_step": result.completed_step,
        "config_digest": _digest(configuration),
        "data_digest": _digest(data_identity),
        "best_checkpoint_sha256": sha256_file(result.best_checkpoint),
        "training_metrics_sha256": sha256_file(result.metrics_path),
        "validation_metrics_sha256": sha256_file(result.validation_metrics_path),
        "checkpoint_reload_exact": reload_exact,
        "long_only_prediction_path": str(prediction_record.path.resolve()),
        "long_only_prediction_sha256": prediction_record.sha256,
        "long_only_frame_count": int(prediction["frame_ids"].shape[0]),
    }
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
    _require_exact(max_steps, FORMAL_CALIBRATION_STEPS, "calibration update count")
    _require_exact(learning_rate, FORMAL_LEARNING_RATE, "learning rate")
    _require_exact(checkpoint_interval, FORMAL_CHECKPOINT_INTERVAL, "evaluation interval")
    _require_exact(patience, FORMAL_PATIENCE, "early-stopping patience")
    manifest = _load_manifest(run_root)
    _load_formal_config(run_root, manifest)
    completion = Path(run_root) / "training" / variant / "completed.json"
    if completion.is_file():
        return completion
    result = train_camera_head(
        TrainConfig(
            checkpoint_dir=checkpoint_dir,
            run_root=Path(run_root) / "training" / variant,
            variant=variant,
            train_pairs=_pairs_for_role(manifest, "train"),
            validation_pairs=_pairs_for_role(manifest, "validation"),
            max_steps=max_steps,
            learning_rate=learning_rate,
            weight_decay=FORMAL_WEIGHT_DECAY,
            checkpoint_interval=checkpoint_interval,
            patience=patience,
            seed=FORMAL_SEED,
            device=device,
        )
    )
    provenance_path = Path(run_root) / "training" / variant / "training_provenance.json"
    provenance = _read_json(provenance_path)
    _atomic_json(
        completion,
        {
            "schema": "long_short_camera_head.calibration_completion.v1",
            "variant": variant,
            "completed_step": result.completed_step,
            "initial_training_loss": result.initial_training_loss,
            "final_training_loss": result.final_training_loss,
            "best_validation_rms": result.best_validation_rms,
            "best_step": result.best_step,
            "best_checkpoint_sha256": sha256_file(result.best_checkpoint),
            "latest_checkpoint_sha256": sha256_file(result.latest_checkpoint),
            "training_metrics_sha256": sha256_file(result.metrics_path),
            "validation_metrics_sha256": sha256_file(result.validation_metrics_path),
            "training_provenance_sha256": sha256_file(provenance_path),
            "config_digest": provenance["config_digest"],
            "data_digest": provenance["data_digest"],
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
    _load_formal_config(run_root, manifest)
    checkpoint = Path(run_root) / "training" / variant / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"{variant} best checkpoint is missing")
    rows = manifest["records"]
    assert isinstance(rows, list)
    output_rows: list[dict[str, object]] = []
    long_paths = tuple(Path(str(row["long_context_path"])) for row in rows)
    prediction_paths = tuple(
        Path(run_root) / "predictions" / variant / f"{row['scene']}.npz"
        for row in rows
    )
    predictions = run_long_only_inference_batch(
        long_paths,
        checkpoint,
        checkpoint_dir,
        prediction_paths,
        device,
    )
    for row, prediction in zip(rows, predictions):
        scene = str(row["scene"])
        metrics_path = Path(run_root) / "evaluation" / variant / f"{scene}.json"
        evaluation = evaluate_prediction(
            prediction.path,
            Path(str(row["privileged_path"])),
            metrics_path,
        )
        output_rows.append(
            {
                "scene": scene,
                "role": "locked_replay" if row["role"] == "validation" else "train_diagnostic",
                "prediction_sha256": prediction.sha256,
                "evaluation_sha256": sha256_file(evaluation.path),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    if len(output_rows) != 10:
        raise ValueError("formal evaluation must cover all ten scenes")
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
    config = _load_formal_config(run_root, manifest)
    rows = manifest["records"]
    assert isinstance(rows, list)
    roles = [str(row["role"]) for row in rows]
    if roles.count("train") != 8 or roles.count("validation") != 2:
        raise ValueError("run is incomplete: split mismatch")
    train_scenes = [str(row["scene"]) for row in rows if row["role"] == "train"]
    validation_scenes = [str(row["scene"]) for row in rows if row["role"] == "validation"]
    all_scenes = [str(row["scene"]) for row in rows]
    if len(set(all_scenes)) != 10:
        raise ValueError("run is incomplete: duplicate scenes")

    evidence_path = run_root / "manifests" / "test_evidence.json"
    if not evidence_path.is_file():
        raise ValueError("run is incomplete: focused-test evidence")
    evidence = _read_json(evidence_path)
    evidence_suites = evidence.get("suites")
    expected_suite_identity = list(REQUIRED_TEST_SUITES)
    if (
        evidence.get("schema") != TEST_EVIDENCE_SCHEMA
        or evidence.get("training_git_revision") != config["git_revision"]
        or evidence.get("tested_git_revision") != _git_revision()
        or not isinstance(evidence_suites, list)
        or [
            (str(row.get("name")), str(row.get("path")))
            for row in evidence_suites
            if isinstance(row, dict)
        ]
        != expected_suite_identity
        or int(evidence.get("test_count", 0))
        != sum(int(row.get("test_count", 0)) for row in evidence_suites)
        or any(
            not isinstance(row, dict)
            or int(row.get("returncode", -1)) != 0
            or int(row.get("test_count", 0)) < 1
            or not Path(str(row.get("log_path", ""))).is_file()
            or row.get("log_sha256")
            != sha256_file(Path(str(row.get("log_path", ""))))
            for row in evidence_suites
        )
    ):
        raise ValueError("run is incomplete: compatibility-test evidence mismatch")

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
    for name, expected in (
        ("scene", "scene0000_00"),
        ("max_steps", FORMAL_SMOKE_STEPS),
        ("completed_step", FORMAL_SMOKE_STEPS),
        ("best_step", FORMAL_SMOKE_STEPS),
        ("learning_rate", FORMAL_LEARNING_RATE),
        ("weight_decay", FORMAL_WEIGHT_DECAY),
        ("checkpoint_interval", FORMAL_SMOKE_STEPS),
        ("patience", FORMAL_SMOKE_STEPS),
        ("seed", FORMAL_SEED),
        ("precision", "bf16_autocast"),
        ("checkpoint_reload_exact", True),
        ("long_only_frame_count", 500),
    ):
        if smoke.get(name) != expected:
            raise ValueError(f"run is incomplete: smoke {name} mismatch")
    if not float(smoke["final_training_loss"]) < float(smoke["initial_training_loss"]):
        raise ValueError("run is incomplete: smoke loss did not strictly decrease")
    if smoke.get("formal_protocol_sha256") != config["protocol_sha256"]:
        raise ValueError("run is incomplete: smoke protocol mismatch")

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

    smoke_pair = next(row for row in rows if row["scene"] == "scene0000_00")
    smoke_configuration = {
        "schema": "long_short_camera_head.smoke.v1",
        "scene": "scene0000_00",
        "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        "max_steps": FORMAL_SMOKE_STEPS,
        "learning_rate": FORMAL_LEARNING_RATE,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "checkpoint_interval": FORMAL_SMOKE_STEPS,
        "patience": FORMAL_SMOKE_STEPS,
        "seed": FORMAL_SEED,
        "precision": "bf16_autocast",
        "formal_protocol_sha256": config["protocol_sha256"],
        "trainable_parameter_names": smoke.get("trainable_parameter_names"),
    }
    smoke_data = {
        "long_context_sha256": smoke_pair["long_context_sha256"],
        "privileged_sha256": smoke_pair["privileged_sha256"],
    }
    if smoke.get("config_digest") != _digest(smoke_configuration) or smoke.get(
        "data_digest"
    ) != _digest(smoke_data):
        raise ValueError("run is incomplete: smoke identity digest mismatch")
    smoke_checkpoint = run_root / "smoke" / "checkpoints" / "best.pt"
    smoke_prediction_path = Path(str(smoke.get("long_only_prediction_path", "")))
    if (
        not smoke_checkpoint.is_file()
        or smoke.get("best_checkpoint_sha256") != sha256_file(smoke_checkpoint)
        or not smoke_prediction_path.is_file()
        or smoke.get("long_only_prediction_sha256") != sha256_file(smoke_prediction_path)
    ):
        raise ValueError("run is incomplete: smoke artifact digest mismatch")
    smoke_prediction = load_prediction(smoke_prediction_path)
    if str(smoke_prediction["checkpoint_sha256"]) != sha256_file(smoke_checkpoint):
        raise ValueError("run is incomplete: smoke prediction checkpoint mismatch")
    try:
        smoke_payload = torch.load(smoke_checkpoint, map_location="cpu", weights_only=False)
        reloaded = load_camera_head_checkpoint(
            smoke_checkpoint,
            Path(str(manifest["checkpoint_dir"])),
            torch.device("cpu"),
        )
        checkpoint_model = smoke_payload.get("model") if isinstance(smoke_payload, dict) else None
        exact_reload = isinstance(checkpoint_model, dict) and all(
            name in checkpoint_model and torch.equal(value.cpu(), checkpoint_model[name].cpu())
            for name, value in reloaded.state_dict().items()
        )
    finally:
        if "reloaded" in locals():
            del reloaded
        if "smoke_payload" in locals():
            del smoke_payload
        gc.collect()
    if not exact_reload:
        raise ValueError("run is incomplete: smoke checkpoint exact reload failed")

    completion_hashes: dict[str, str] = {"smoke": sha256_file(smoke_completion)}
    for variant in ("gt_only", "long_short"):
        checkpoint = run_root / "training" / variant / "checkpoints" / "best.pt"
        completion = run_root / "training" / variant / "completed.json"
        provenance_path = run_root / "training" / variant / "training_provenance.json"
        evaluation_completion = run_root / "evaluation" / variant / "completed.json"
        if (
            not checkpoint.is_file()
            or not completion.is_file()
            or not provenance_path.is_file()
            or not evaluation_completion.is_file()
        ):
            raise ValueError(f"run is incomplete: {variant} training/evaluation")
        training = _read_json(completion)
        provenance = _read_json(provenance_path)
        expected_config_payload = {
            "schema": "long_short_camera_head.train_config.v1",
            "variant": variant,
            "max_steps": FORMAL_CALIBRATION_STEPS,
            "learning_rate": FORMAL_LEARNING_RATE,
            "weight_decay": FORMAL_WEIGHT_DECAY,
            "checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
            "patience": FORMAL_PATIENCE,
            "seed": FORMAL_SEED,
            "precision": "bf16_autocast",
            "weights": config["protocol"]["loss_weights"],
            "train_scenes": train_scenes,
            "validation_scenes": validation_scenes,
            "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        }
        if any(provenance.get(name) != value for name, value in expected_config_payload.items()):
            raise ValueError(f"run is incomplete: {variant} formal provenance mismatch")
        if provenance.get("config_digest") != _digest(expected_config_payload):
            raise ValueError(f"run is incomplete: {variant} config digest mismatch")
        files = provenance.get("files")
        if not isinstance(files, list) or provenance.get("data_digest") != _digest(files):
            raise ValueError(f"run is incomplete: {variant} data digest mismatch")
        expected_files = [
            {
                "kind": "train" if row["role"] == "train" else "validation",
                "scene": row["scene"],
                "long_sha256": row["long_context_sha256"],
                "privileged_sha256": row["privileged_sha256"],
            }
            for role in ("train", "validation")
            for row in rows
            if row["role"] == role
        ]
        if files != expected_files:
            raise ValueError(f"run is incomplete: {variant} file provenance mismatch")
        if provenance.get("device") != "cuda" or int(
            provenance.get("trainable_parameter_count", 0)
        ) <= 0:
            raise ValueError(f"run is incomplete: {variant} training scope/device mismatch")
        checkpoint_digest = sha256_file(checkpoint)
        completed_step = int(training.get("completed_step", -1))
        best_step = int(training.get("best_step", -1))
        if (
            completed_step < FORMAL_CHECKPOINT_INTERVAL
            or completed_step > FORMAL_CALIBRATION_STEPS
            or completed_step % FORMAL_CHECKPOINT_INTERVAL != 0
            or best_step < FORMAL_CHECKPOINT_INTERVAL
            or best_step > completed_step
            or best_step % FORMAL_CHECKPOINT_INTERVAL != 0
            or training.get("config_digest") != provenance["config_digest"]
            or training.get("data_digest") != provenance["data_digest"]
            or training.get("best_checkpoint_sha256") != checkpoint_digest
            or training.get("training_provenance_sha256") != sha256_file(provenance_path)
        ):
            raise ValueError(f"run is incomplete: {variant} completion mismatch")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            not isinstance(checkpoint_payload, dict)
            or checkpoint_payload.get("schema") != CHECKPOINT_SCHEMA
            or int(checkpoint_payload.get("step", -1)) != best_step
            or checkpoint_payload.get("config_digest") != provenance["config_digest"]
            or checkpoint_payload.get("data_digest") != provenance["data_digest"]
        ):
            raise ValueError(f"run is incomplete: {variant} checkpoint metadata mismatch")
        del checkpoint_payload
        gc.collect()
        evaluation_manifest = _read_json(evaluation_completion)
        evaluation_records = evaluation_manifest.get("records")
        if (
            evaluation_manifest.get("variant") != variant
            or evaluation_manifest.get("checkpoint_sha256") != checkpoint_digest
            or not isinstance(evaluation_records, list)
            or len(evaluation_records) != 10
            or {str(record.get("scene")) for record in evaluation_records} != set(all_scenes)
        ):
            raise ValueError(f"run is incomplete: {variant} all-scene evaluation manifest")
        evaluation_by_scene = {
            str(record["scene"]): record for record in evaluation_records
        }
        for scene in all_scenes:
            prediction_path = run_root / "predictions" / variant / f"{scene}.npz"
            metrics_path = run_root / "evaluation" / variant / f"{scene}.json"
            prediction = load_prediction(prediction_path)
            metrics = _read_json(metrics_path)
            record = evaluation_by_scene[scene]
            if metrics.get("schema") != EVALUATION_SCHEMA:
                raise ValueError("run is incomplete: evaluation schema mismatch")
            if metrics.get("scene") != scene or str(prediction["scene"]) != scene:
                raise ValueError("run is incomplete: evaluation scene mismatch")
            if str(prediction["checkpoint_sha256"]) != checkpoint_digest:
                raise ValueError("run is incomplete: prediction checkpoint mismatch")
            if metrics.get("checkpoint_sha256") != checkpoint_digest:
                raise ValueError("run is incomplete: evaluation checkpoint mismatch")
            expected_role = "locked_replay" if scene in validation_scenes else "train_diagnostic"
            if (
                record.get("role") != expected_role
                or record.get("prediction_sha256") != sha256_file(prediction_path)
                or record.get("evaluation_sha256") != sha256_file(metrics_path)
            ):
                raise ValueError("run is incomplete: evaluation artifact digest mismatch")
            artifact_rows.append(
                {
                    "variant": variant,
                    "scene": scene,
                    "checkpoint_sha256": checkpoint_digest,
                    "prediction_sha256": sha256_file(prediction_path),
                    "evaluation_sha256": sha256_file(metrics_path),
                }
            )
        completion_hashes[f"training_{variant}"] = sha256_file(completion)
        completion_hashes[f"evaluation_{variant}"] = sha256_file(evaluation_completion)
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
    if len(report.get("scenes", [])) != 10 or len(report.get("locked_replay_scenes", [])) != 2:
        raise ValueError("run is incomplete: report scene coverage mismatch")
    nonempty_errors = [
        str(path)
        for path in (run_root / "logs").glob("*.err.log")
        if path.stat().st_size > 0
    ]
    if nonempty_errors:
        raise ValueError(f"run is incomplete: nonempty stderr logs: {nonempty_errors}")
    completion_payload = {
        "schema": VERIFIED_SCHEMA,
        "git_revision": config.get("git_revision"),
        "verifier_git_revision": _git_revision(),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "base_checkpoint_sha256": manifest.get("base_checkpoint_sha256"),
        "config_sha256": sha256_file(run_root / "config.json"),
        "data_manifest_sha256": sha256_file(
            run_root / "manifests" / "data_manifest.json"
        ),
        "test_evidence_sha256": sha256_file(evidence_path),
        "stage_completion_sha256": completion_hashes,
        "scene_count": 10,
        "train_scene_count": 8,
        "locked_replay_scene_count": 2,
        "classification": report["classification"],
        "report_sha256": sha256_file(report_path),
        "artifacts": artifact_rows,
        "inference_leakage_audit": True,
        "formal_protocol_sha256": config["protocol_sha256"],
    }
    completion_path = run_root / "verified_completion.json"
    _atomic_json(completion_path, completion_payload)
    return completion_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "preflight",
            "smoke",
            "calibration",
            "evaluate",
            "report",
            "verify",
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--variant", choices=("gt_only", "long_short"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=FORMAL_CALIBRATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=FORMAL_LEARNING_RATE)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=FORMAL_CHECKPOINT_INTERVAL
    )
    parser.add_argument("--patience", type=int, default=FORMAL_PATIENCE)
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
    elif args.stage == "preflight":
        path = run_test_preflight(run_root=args.run_root)
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
