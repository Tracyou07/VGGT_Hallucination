from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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
from .matched_random_ablation import (
    generate_matched_random_ablation,
    load_matched_random_ablation,
    load_matched_random_privileged,
    write_matched_random_privileged_sidecar,
    write_matched_random_report,
)
from .matched_random_ensemble import (
    FORMAL_REPLICATE_COUNT,
    build_matched_random_ensemble_plan,
    load_matched_random_ensemble_plan,
    summarize_matched_random_ensemble,
    write_matched_random_ensemble_plan,
)
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


def _require_clean_git_checkout(repo_root: Path) -> str:
    status = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=*",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("matched random provenance requires a clean git checkout")
    return read_git_commit(repo_root)


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


def _require_matched_random_sample_budget(
    candidate: dict[str, np.ndarray],
    vrfm_prediction: dict[str, np.ndarray],
    vrfm_privileged: dict[str, np.ndarray] | None = None,
) -> None:
    candidate_z = np.asarray(candidate.get("z"))
    prediction_z = np.asarray(vrfm_prediction.get("z"))
    if (
        candidate_z.ndim != 3
        or candidate_z.shape[:2] != (8, 32)
        or prediction_z.ndim != 3
        or prediction_z.shape[:2] != (8, 32)
    ):
        raise ValueError(
            "matched random ablation requires exactly 32 samples per overlap"
        )
    if vrfm_privileged is not None:
        candidate_rms = np.asarray(vrfm_privileged.get("candidate_rms"))
        if candidate_rms.shape != (8, 32, len(DEFAULT_ALPHAS)):
            raise ValueError(
                "matched random ablation requires exactly 32 samples per overlap"
            )


def _matched_random_transform_identity(
    *,
    source_manifest_sha256: str,
    candidate_manifest_sha256: str,
    vrfm_prediction_manifest_sha256: str,
) -> str:
    for value in (
        source_manifest_sha256,
        candidate_manifest_sha256,
        vrfm_prediction_manifest_sha256,
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("matched random transform inputs must be SHA-256 digests")
    return _canonical_digest(
        {
            "schema": "variational_camera_latent.matched_random_transform_identity.v1",
            "source_manifest_sha256": source_manifest_sha256,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "vrfm_prediction_manifest_sha256": vrfm_prediction_manifest_sha256,
        }
    )


_MATCHED_RANDOM_20Q_PLAN_RELATIVE = Path(
    "manifests/matched_random_20q_plan.json"
)
_MATCHED_RANDOM_20Q_PREDICTION_SCHEMA = (
    "variational_camera_latent.matched_random_20q_prediction_manifest.v1"
)
_MATCHED_RANDOM_20Q_PREDICTION_COMPLETION_SCHEMA = (
    "variational_camera_latent.matched_random_20q_prediction_complete.v1"
)
_MATCHED_RANDOM_20Q_PRIVILEGED_SCHEMA = (
    "variational_camera_latent.matched_random_20q_privileged_manifest.v1"
)
_MATCHED_RANDOM_20Q_REPLICATE_COMPLETION_SCHEMA = (
    "variational_camera_latent.matched_random_20q_replicate_verified_completion.v1"
)
_MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_SCHEMA = (
    "variational_camera_latent.matched_random_20q_aggregate_prediction_manifest.v1"
)
_MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_COMPLETION_SCHEMA = (
    "variational_camera_latent."
    "matched_random_20q_aggregate_prediction_complete.v1"
)
_MATCHED_RANDOM_20Q_AGGREGATE_PRIVILEGED_SCHEMA = (
    "variational_camera_latent.matched_random_20q_aggregate_privileged_manifest.v1"
)
_MATCHED_RANDOM_20Q_REPORT_SCHEMA = (
    "variational_camera_latent.matched_random_20q_report.v1"
)
_MATCHED_RANDOM_20Q_COMPLETION_SCHEMA = (
    "variational_camera_latent.matched_random_20q_verified_completion.v1"
)


def _matched_random_replicate_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < FORMAL_REPLICATE_COUNT:
        raise ValueError("matched random replicate index must be in [0, 20)")
    return value


def _write_exact_json(path: Path, payload: dict[str, object], *, label: str) -> None:
    path = Path(path)
    if path.is_file():
        if _read_json(path) != payload:
            raise ValueError(f"existing {label} differs")
        return
    if path.exists():
        raise ValueError(f"existing {label} is not a regular file")
    _atomic_json(path, payload)


def _require_contract_path(
    path: Path,
    expected: Path,
    *,
    run_root: Path,
    label: str,
    require_file: bool = False,
) -> Path:
    """Require one lexical run-root path without following an in-tree symlink."""
    root = Path(run_root).absolute()
    actual = Path(path).absolute()
    lexical_expected = Path(expected).absolute()
    if actual != lexical_expected or not actual.is_relative_to(root):
        raise ValueError(f"{label} is outside its path contract")
    relative = actual.relative_to(root)
    cursor = root
    symlinked = root.is_symlink()
    for component in relative.parts:
        cursor = cursor / component
        symlinked = symlinked or cursor.is_symlink()
    if symlinked:
        raise ValueError(f"{label} path contains a symlink")
    if require_file and (not actual.is_file() or actual.is_symlink()):
        raise ValueError(f"{label} is not a regular contract file")
    return actual


def _require_exact_directory_entries(
    directory: Path,
    expected_entries: Sequence[Path],
    *,
    entry_kind: str,
    label: str,
) -> None:
    """Reject missing, extra, symlinked, or wrongly typed contract entries."""
    root = Path(directory).absolute()
    if entry_kind not in {"file", "directory"}:
        raise ValueError("exact directory entry kind must be file or directory")
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be a regular directory")
    expected_list = [Path(path).absolute() for path in expected_entries]
    expected = set(expected_list)
    if len(expected) != len(expected_list) or any(
        path.parent != root for path in expected
    ):
        raise ValueError(f"{label} expected entries are invalid")
    observed = set(root.iterdir())
    if observed != expected:
        raise ValueError(f"{label} must contain exactly its contracted entries")
    predicate = Path.is_file if entry_kind == "file" else Path.is_dir
    if any(path.is_symlink() or not predicate(path) for path in observed):
        raise ValueError(f"{label} contains an invalid {entry_kind} entry")


def _seal_matched_random_aggregate_prediction_barrier(
    *,
    run_root: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Seal and re-read the all-prediction barrier before privileged access."""
    run_root = Path(run_root).absolute()
    replicates = manifest.get("replicates")
    if (
        manifest.get("schema") != _MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_SCHEMA
        or manifest.get("replicate_count") != FORMAL_REPLICATE_COUNT
        or manifest.get("scene_count") != 10
        or manifest.get("prediction_artifact_count") != 200
        or not isinstance(replicates, list)
        or len(replicates) != FORMAL_REPLICATE_COUNT
    ):
        raise ValueError("20-Q aggregate prediction manifest is invalid")
    completion_sha256s: list[str] = []
    for index, row in enumerate(replicates):
        if (
            not isinstance(row, dict)
            or row.get("replicate_index") != index
            or row.get("replicate_id") != f"formal_null_{index:03d}"
        ):
            raise ValueError("20-Q aggregate prediction replicate order differs")
        completion_sha256 = row.get("prediction_completion_sha256")
        if (
            not isinstance(completion_sha256, str)
            or len(completion_sha256) != 64
            or any(character not in "0123456789abcdef" for character in completion_sha256)
        ):
            raise ValueError("20-Q aggregate prediction completion digest is invalid")
        completion_sha256s.append(completion_sha256)

    manifest_path = run_root / "manifests" / "matched_random_20q_prediction_manifest.json"
    completion_path = (
        run_root
        / "manifests"
        / "matched_random_20q_prediction_verified_completion.json"
    )
    _require_contract_path(
        manifest_path,
        manifest_path,
        run_root=run_root,
        label="20-Q aggregate prediction manifest",
    )
    _require_contract_path(
        completion_path,
        completion_path,
        run_root=run_root,
        label="20-Q aggregate prediction completion",
    )
    _write_exact_json(
        manifest_path,
        manifest,
        label="20-Q aggregate prediction manifest",
    )
    manifest_sha256 = _sha256_file(manifest_path)
    unsigned = {
        "schema": _MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_COMPLETION_SCHEMA,
        "replicate_count": FORMAL_REPLICATE_COUNT,
        "scene_count": 10,
        "prediction_artifact_count": 200,
        "plan_sha256": manifest["plan_sha256"],
        "plan_digest": manifest["plan_digest"],
        "prediction_manifest_sha256": manifest_sha256,
        "replicate_prediction_completion_sha256s": completion_sha256s,
        "git_commit": manifest["producer_git_commit"],
    }
    if load_exact_completion(completion_path, unsigned) is None:
        if completion_path.exists():
            raise ValueError("existing 20-Q aggregate prediction completion differs")
        write_completion(completion_path, unsigned)
    if _load_signed_completion(
        completion_path,
        schema=_MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_COMPLETION_SCHEMA,
    ) != unsigned:
        raise ValueError("20-Q aggregate prediction completion differs after sealing")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "completion_path": completion_path,
        "completion_sha256": _sha256_file(completion_path),
    }


def _matched_random_20q_plan_path(run_root: Path) -> Path:
    return Path(run_root).resolve() / _MATCHED_RANDOM_20Q_PLAN_RELATIVE


def _load_matched_random_prediction_context(args: argparse.Namespace) -> dict[str, object]:
    """Validate prediction-only upstream state shared by all 20-Q stages."""
    if args.scene_limit != 10:
        raise ValueError("matched random 20-Q stages require exactly ten scenes")
    if args.matched_random_batch_size < 1:
        raise ValueError("matched random ablation batch size must be positive")

    run_root = Path(args.run_root).resolve()
    source_manifest_path = run_root / "manifests" / "source_manifest.json"
    _require_contract_path(
        source_manifest_path,
        source_manifest_path,
        run_root=run_root,
        label="20-Q source manifest",
        require_file=True,
    )
    source_manifest, source_records = _source_manifest(run_root)
    source_manifest_sha256 = _sha256_file(source_manifest_path)
    source_completion = _load_signed_completion(
        run_root / "manifests" / "source_complete.json",
        schema="variational_camera_latent.source_complete.v1",
    )
    source_run_completion_path = args.source_run / "manifests" / "calibration_complete.json"
    source_run_completion_sha256 = _sha256_file(source_run_completion_path)
    if (
        source_completion.get("source_manifest_sha256") != source_manifest_sha256
        or Path(str(source_completion.get("source_run"))).resolve()
        != args.source_run.resolve()
        or source_completion.get("source_run_digest") != source_run_completion_sha256
        or source_manifest.get("source_run_digest") != source_run_completion_sha256
    ):
        raise ValueError("20-Q source artifacts do not bind the requested source run")

    candidate_manifest_path = run_root / "manifests" / "calibration_prediction_manifest.json"
    _require_contract_path(
        candidate_manifest_path,
        candidate_manifest_path,
        run_root=run_root,
        label="20-Q candidate manifest",
        require_file=True,
    )
    candidate_manifest = _read_json(candidate_manifest_path)
    candidate_records = candidate_manifest.get("records")
    if (
        candidate_manifest.get("schema")
        != "variational_camera_latent.calibration_prediction_manifest.v1"
        or candidate_manifest.get("stage") != "calibration"
        or candidate_manifest.get("samples") != 32
        or not isinstance(candidate_records, list)
        or len(candidate_records) != 10
        or any(not isinstance(row, dict) for row in candidate_records)
    ):
        raise ValueError("20-Q calibration prediction manifest is incomplete")
    candidate_manifest_sha256 = _sha256_file(candidate_manifest_path)
    calibration_completion = _load_signed_completion(
        run_root / "manifests" / "calibration_complete.json",
        schema="variational_camera_latent.calibration_complete.v1",
    )
    if (
        calibration_completion.get("prediction_manifest_sha256")
        != candidate_manifest_sha256
        or calibration_completion.get("source_manifest_sha256")
        != source_manifest_sha256
    ):
        raise ValueError("20-Q calibration completion does not bind current inputs")
    phase1_prediction = _read_json(run_root / "manifests" / "prediction_manifest.json")
    if phase1_prediction.get("records") != candidate_records:
        raise ValueError("20-Q Phase 1 and calibration candidate records differ")

    vrfm_prediction_manifest_path = (
        run_root
        / "manifests"
        / "vrfm_residual_alpha_scan_full_context_prediction_manifest.json"
    )
    _require_contract_path(
        vrfm_prediction_manifest_path,
        vrfm_prediction_manifest_path,
        run_root=run_root,
        label="20-Q VRFM prediction manifest",
        require_file=True,
    )
    vrfm_prediction_manifest = _read_json(vrfm_prediction_manifest_path)
    vrfm_prediction_records = vrfm_prediction_manifest.get("records")
    if (
        vrfm_prediction_manifest.get("schema")
        != "variational_camera_latent.vrfm_residual_alpha_scan_full_context_prediction_manifest.v1"
        or vrfm_prediction_manifest.get("scene_count") != 10
        or vrfm_prediction_manifest.get("overlap_count") != 80
        or vrfm_prediction_manifest.get("samples_per_overlap") != 32
        or vrfm_prediction_manifest.get("direction_count") != 2560
        or vrfm_prediction_manifest.get("grid_cell_count") != 20480
        or vrfm_prediction_manifest.get("unique_pose_candidate_count") != 18000
        or vrfm_prediction_manifest.get("alphas") != list(DEFAULT_ALPHAS)
        or vrfm_prediction_manifest.get("decode_context_frames") != 500
        or vrfm_prediction_manifest.get("camera_iterations") != 4
        or vrfm_prediction_manifest.get("source_manifest_sha256")
        != source_manifest_sha256
        or vrfm_prediction_manifest.get("candidate_manifest_sha256")
        != candidate_manifest_sha256
        or not isinstance(vrfm_prediction_records, list)
        or len(vrfm_prediction_records) != 10
        or any(not isinstance(row, dict) for row in vrfm_prediction_records)
    ):
        raise ValueError("20-Q VRFM prediction manifest is invalid")
    vrfm_prediction_manifest_sha256 = _sha256_file(vrfm_prediction_manifest_path)
    upstream_vrfm_commit = vrfm_prediction_manifest.get("producer_git_commit")
    if (
        not isinstance(upstream_vrfm_commit, str)
        or len(upstream_vrfm_commit) != 40
        or any(character not in "0123456789abcdef" for character in upstream_vrfm_commit)
    ):
        raise ValueError("20-Q VRFM producer commit is invalid")

    source_run_manifest = _read_json(args.source_run / "manifests" / "run.json")
    camera_head_checkpoint_sha256 = source_run_manifest.get("checkpoint_sha256")
    if (
        source_run_manifest.get("schema") != "camera_velocity_ambiguity_02.run.v1"
        or source_run_manifest.get("run_id") != args.source_run.name
        or not isinstance(camera_head_checkpoint_sha256, str)
        or len(camera_head_checkpoint_sha256) != 64
        or vrfm_prediction_manifest.get("camera_head_checkpoint_sha256")
        != camera_head_checkpoint_sha256
        or _sha256_file(find_checkpoint(args.checkpoint_dir))
        != camera_head_checkpoint_sha256
    ):
        raise ValueError("20-Q Camera Head checkpoint provenance differs")

    scenes = [str(row.get("scene")) for row in source_records]
    if (
        len(set(scenes)) != 10
        or candidate_manifest.get("scenes") != scenes
        or _source_scene_order(args.source_run) != scenes
    ):
        raise ValueError("20-Q scene identities or order differ")
    transform_identity_sha256 = _matched_random_transform_identity(
        source_manifest_sha256=source_manifest_sha256,
        candidate_manifest_sha256=candidate_manifest_sha256,
        vrfm_prediction_manifest_sha256=vrfm_prediction_manifest_sha256,
    )
    producer_git_commit = _require_clean_git_checkout(ROOT)

    pilot_manifest_path = (
        run_root
        / "manifests"
        / "matched_random_ablation_full_context_prediction_manifest.json"
    )
    _require_contract_path(
        pilot_manifest_path,
        pilot_manifest_path,
        run_root=run_root,
        label="pilot reference prediction manifest",
        require_file=True,
    )
    pilot_manifest = _read_json(pilot_manifest_path)
    pilot_records = pilot_manifest.get("records")
    pilot_producer_git_commit = pilot_manifest.get("producer_git_commit")
    if (
        pilot_manifest.get("schema")
        != "variational_camera_latent.matched_random_ablation_full_context_prediction_manifest.v1"
        or pilot_manifest.get("scene_count") != 10
        or pilot_manifest.get("overlap_count") != 80
        or pilot_manifest.get("samples_per_overlap") != 32
        or pilot_manifest.get("direction_count") != 2560
        or pilot_manifest.get("grid_cell_count") != 20480
        or pilot_manifest.get("unique_pose_candidate_count") != 18000
        or pilot_manifest.get("alphas") != list(DEFAULT_ALPHAS)
        or pilot_manifest.get("decode_context_frames") != 500
        or pilot_manifest.get("camera_iterations") != 4
        or pilot_manifest.get("structured_null_replicate_count") != 1
        or pilot_manifest.get("same_transform_across_scenes") is not True
        or pilot_manifest.get("preserves_feature_row_gram_geometry") is not True
        or not isinstance(pilot_producer_git_commit, str)
        or len(pilot_producer_git_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in pilot_producer_git_commit
        )
        or pilot_manifest.get("camera_head_checkpoint_sha256")
        != camera_head_checkpoint_sha256
        or pilot_manifest.get("source_manifest_sha256") != source_manifest_sha256
        or pilot_manifest.get("candidate_manifest_sha256") != candidate_manifest_sha256
        or pilot_manifest.get("paired_vrfm_prediction_manifest_sha256")
        != vrfm_prediction_manifest_sha256
        or pilot_manifest.get("transform_identity_sha256")
        != transform_identity_sha256
        or not isinstance(pilot_manifest.get("transform_sha256"), str)
        or not isinstance(pilot_records, list)
        or len(pilot_records) != 10
        or any(not isinstance(row, dict) for row in pilot_records)
    ):
        raise ValueError("observed matched-random pilot prediction manifest is invalid")
    reference_transform_sha256 = str(pilot_manifest["transform_sha256"])
    expected_pilot_root = run_root / "prediction_only" / "matched_random_ablation_full_context"
    _require_contract_path(
        expected_pilot_root,
        expected_pilot_root,
        run_root=run_root,
        label="pilot reference prediction root",
    )
    _require_exact_directory_entries(
        expected_pilot_root,
        [expected_pilot_root / f"{scene}.npz" for scene in scenes],
        entry_kind="file",
        label="pilot reference predictions",
    )
    for index, (scene, record) in enumerate(zip(scenes, pilot_records)):
        pilot_path = Path(str(record.get("path")))
        source_path = Path(str(source_records[index].get("path")))
        candidate_path = Path(str(candidate_records[index].get("path")))
        vrfm_path = Path(str(vrfm_prediction_records[index].get("path")))
        upstream_paths = (
            (source_path, run_root / "prediction_only" / "source" / f"{scene}.npz"),
            (
                candidate_path,
                run_root
                / "prediction_only"
                / "calibration_candidates"
                / f"{scene}.npz",
            ),
            (
                vrfm_path,
                run_root
                / "prediction_only"
                / "vrfm_residual_alpha_scan_full_context"
                / f"{scene}.npz",
            ),
        )
        for path, expected in upstream_paths:
            _require_contract_path(
                path,
                expected,
                run_root=run_root,
                label="pilot reference upstream artifact",
                require_file=True,
            )
        for upstream_record, upstream_path in (
            (source_records[index], source_path),
            (candidate_records[index], candidate_path),
            (vrfm_prediction_records[index], vrfm_path),
        ):
            if upstream_record.get("sha256") != _sha256_file(upstream_path):
                raise ValueError("observed matched-random upstream digest differs")
        _require_contract_path(
            pilot_path,
            expected_pilot_root / f"{scene}.npz",
            run_root=run_root,
            label="pilot reference prediction artifact",
            require_file=True,
        )
        if (
            record.get("scene") != scene
            or record.get("sha256") != _sha256_file(pilot_path)
            or record.get("source_shard_sha256")
            != source_records[index].get("sha256")
            or record.get("candidate_shard_sha256")
            != candidate_records[index].get("sha256")
            or record.get("paired_vrfm_prediction_sha256")
            != vrfm_prediction_records[index].get("sha256")
        ):
            raise ValueError("observed matched-random pilot prediction record differs")
        source = load_source_shard(source_path)
        candidate = load_candidate_shard(candidate_path)
        vrfm = load_vrfm_residual_alpha_scan(vrfm_path)
        _require_matched_random_sample_budget(candidate, vrfm)
        arrays = load_matched_random_ablation(pilot_path)
        if (
            str(arrays["transform_identity_sha256"]) != transform_identity_sha256
            or str(arrays["transform_sha256"]) != reference_transform_sha256
            or str(arrays["source_shard_sha256"])
            != str(source_records[index].get("sha256"))
            or str(arrays["candidate_shard_sha256"])
            != str(candidate_records[index].get("sha256"))
            or str(arrays["paired_vrfm_prediction_sha256"])
            != str(vrfm_prediction_records[index].get("sha256"))
            or str(arrays["producer_git_commit"]) != pilot_producer_git_commit
            or str(arrays["camera_head_checkpoint_sha256"])
            != camera_head_checkpoint_sha256
            or int(arrays["base_seed"]) != pilot_manifest.get("base_seed")
            or str(arrays["vrfm_checkpoint_sha256"])
            != str(candidate["checkpoint_sha256"])
            or str(arrays["paired_vrfm_producer_git_commit"])
            != upstream_vrfm_commit
            or not np.array_equal(
                arrays["source_sample_ids"], source["sample_ids"]
            )
            or not np.array_equal(arrays["z"], candidate["z"])
            or not np.array_equal(
                arrays["sample_seeds"], candidate["sample_seeds"]
            )
            or not np.array_equal(
                arrays["latent_cluster_ids"], candidate["latent_cluster_ids"]
            )
            or not np.array_equal(vrfm["source_sample_ids"], source["sample_ids"])
            or not np.array_equal(vrfm["z"], candidate["z"])
            or not np.array_equal(
                vrfm["sample_seeds"], candidate["sample_seeds"]
            )
            or not np.array_equal(
                vrfm["latent_cluster_ids"], candidate["latent_cluster_ids"]
            )
            or str(vrfm["producer_git_commit"]) != upstream_vrfm_commit
            or vrfm_prediction_records[index].get("source_shard_sha256")
            != source_records[index].get("sha256")
            or vrfm_prediction_records[index].get("candidate_shard_sha256")
            != candidate_records[index].get("sha256")
        ):
            raise ValueError("observed matched-random pilot artifact differs")

    if _require_clean_git_checkout(ROOT) != producer_git_commit:
        raise ValueError("20-Q producer commit changed during context validation")
    return {
        "run_root": run_root,
        "scenes": scenes,
        "source_records": source_records,
        "candidate_records": candidate_records,
        "vrfm_prediction_records": vrfm_prediction_records,
        "source_manifest_sha256": source_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "vrfm_prediction_manifest_sha256": vrfm_prediction_manifest_sha256,
        "upstream_vrfm_commit": upstream_vrfm_commit,
        "camera_head_checkpoint_sha256": camera_head_checkpoint_sha256,
        "transform_identity_sha256": transform_identity_sha256,
        "reference_prediction_manifest_sha256": _sha256_file(pilot_manifest_path),
        "reference_transform_sha256": reference_transform_sha256,
        "producer_git_commit": producer_git_commit,
    }


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


def run_matched_random_ablation(args: argparse.Namespace) -> Path:
    """Run one shared orthogonal structured-null pilot against frozen VRFM outputs."""
    if args.scene_limit != 10:
        raise ValueError("matched random ablation requires exactly ten scenes")
    if args.matched_random_batch_size < 1:
        raise ValueError("matched random ablation batch size must be positive")
    if (
        isinstance(args.matched_random_seed, bool)
        or args.matched_random_seed < 0
        or args.matched_random_seed >= 2**63
    ):
        raise ValueError("matched random ablation seed must be in [0, 2**63)")
    if (
        not np.isfinite(args.alpha_min_improvement)
        or args.alpha_min_improvement <= 0.0
    ):
        raise ValueError("matched random minimum improvement must be positive")

    run_root = args.run_root.resolve()
    source_manifest_path = run_root / "manifests" / "source_manifest.json"
    source_manifest, source_records = _source_manifest(run_root)
    source_completion = _load_signed_completion(
        run_root / "manifests" / "source_complete.json",
        schema="variational_camera_latent.source_complete.v1",
    )
    source_manifest_sha256 = _sha256_file(source_manifest_path)
    if source_completion.get("source_manifest_sha256") != source_manifest_sha256:
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
        run_root / "manifests" / "calibration_prediction_manifest.json"
    )
    candidate_manifest = _read_json(candidate_manifest_path)
    candidate_records = candidate_manifest.get("records")
    if (
        candidate_manifest.get("schema")
        != "variational_camera_latent.calibration_prediction_manifest.v1"
        or candidate_manifest.get("stage") != "calibration"
        or candidate_manifest.get("samples") != 32
        or not isinstance(candidate_records, list)
        or len(candidate_records) != 10
        or any(not isinstance(row, dict) for row in candidate_records)
    ):
        raise ValueError("calibration prediction manifest is incomplete")
    candidate_manifest_sha256 = _sha256_file(candidate_manifest_path)
    calibration_completion = _load_signed_completion(
        run_root / "manifests" / "calibration_complete.json",
        schema="variational_camera_latent.calibration_complete.v1",
    )
    if (
        calibration_completion.get("prediction_manifest_sha256")
        != candidate_manifest_sha256
        or calibration_completion.get("source_manifest_sha256")
        != source_manifest_sha256
    ):
        raise ValueError("calibration completion does not bind current inputs")
    phase1_prediction = _read_json(run_root / "manifests" / "prediction_manifest.json")
    if phase1_prediction.get("records") != candidate_records:
        raise ValueError("verified Phase 1 and calibration candidate records differ")

    vrfm_prediction_manifest_path = (
        run_root
        / "manifests"
        / "vrfm_residual_alpha_scan_full_context_prediction_manifest.json"
    )
    vrfm_prediction_manifest = _read_json(vrfm_prediction_manifest_path)
    vrfm_prediction_records = vrfm_prediction_manifest.get("records")
    if (
        vrfm_prediction_manifest.get("schema")
        != "variational_camera_latent.vrfm_residual_alpha_scan_full_context_prediction_manifest.v1"
        or vrfm_prediction_manifest.get("scene_count") != 10
        or vrfm_prediction_manifest.get("overlap_count") != 80
        or vrfm_prediction_manifest.get("samples_per_overlap") != 32
        or vrfm_prediction_manifest.get("direction_count") != 2560
        or vrfm_prediction_manifest.get("grid_cell_count") != 20480
        or vrfm_prediction_manifest.get("unique_pose_candidate_count") != 18000
        or vrfm_prediction_manifest.get("alphas") != list(DEFAULT_ALPHAS)
        or vrfm_prediction_manifest.get("decode_context_frames") != 500
        or vrfm_prediction_manifest.get("camera_iterations") != 4
        or vrfm_prediction_manifest.get("source_manifest_sha256")
        != source_manifest_sha256
        or vrfm_prediction_manifest.get("candidate_manifest_sha256")
        != candidate_manifest_sha256
        or not isinstance(vrfm_prediction_records, list)
        or len(vrfm_prediction_records) != 10
        or any(not isinstance(row, dict) for row in vrfm_prediction_records)
    ):
        raise ValueError("verified VRFM prediction manifest is invalid")
    vrfm_prediction_manifest_sha256 = _sha256_file(vrfm_prediction_manifest_path)
    upstream_vrfm_commit = vrfm_prediction_manifest.get("producer_git_commit")
    if (
        not isinstance(upstream_vrfm_commit, str)
        or len(upstream_vrfm_commit) != 40
        or any(character not in "0123456789abcdef" for character in upstream_vrfm_commit)
    ):
        raise ValueError("VRFM prediction manifest producer commit is invalid")

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
        or vrfm_prediction_manifest.get("camera_head_checkpoint_sha256")
        != camera_head_checkpoint_sha256
    ):
        raise ValueError("source and VRFM Camera Head checkpoints do not match")
    if _sha256_file(find_checkpoint(args.checkpoint_dir)) != camera_head_checkpoint_sha256:
        raise ValueError("local Camera Head checkpoint digest does not match source run")

    scenes = [str(row.get("scene")) for row in source_records]
    if (
        len(set(scenes)) != 10
        or candidate_manifest.get("scenes") != scenes
        or _source_scene_order(args.source_run) != scenes
    ):
        raise ValueError("matched random scene identities or order do not match")
    transform_identity_sha256 = _matched_random_transform_identity(
        source_manifest_sha256=source_manifest_sha256,
        candidate_manifest_sha256=candidate_manifest_sha256,
        vrfm_prediction_manifest_sha256=vrfm_prediction_manifest_sha256,
    )
    producer_git_commit = _require_clean_git_checkout(ROOT)

    prediction_root = run_root / "prediction_only" / "matched_random_ablation_full_context"
    prediction_records: list[dict[str, object]] = []
    expected_transform_sha256: str | None = None
    head = None
    for index, records in enumerate(
        zip(
            source_records,
            candidate_records,
            vrfm_prediction_records,
        )
    ):
        source_record, candidate_record, vrfm_record = records
        scene = scenes[index]
        if any(row.get("scene") != scene for row in records):
            raise ValueError("matched random upstream scene order differs")
        source_path = Path(str(source_record.get("path")))
        candidate_path = Path(str(candidate_record.get("path")))
        vrfm_prediction_path = Path(str(vrfm_record.get("path")))
        expected_paths = (
            (source_path, run_root / "prediction_only" / "source" / f"{scene}.npz"),
            (
                candidate_path,
                run_root / "prediction_only" / "calibration_candidates" / f"{scene}.npz",
            ),
            (
                vrfm_prediction_path,
                run_root
                / "prediction_only"
                / "vrfm_residual_alpha_scan_full_context"
                / f"{scene}.npz",
            ),
        )
        if any(path.resolve() != expected.resolve() for path, expected in expected_paths):
            raise ValueError("matched random upstream artifact path is outside its contract")
        for row, path, label in (
            (source_record, source_path, "source"),
            (candidate_record, candidate_path, "candidate"),
            (vrfm_record, vrfm_prediction_path, "VRFM prediction"),
        ):
            if row.get("sha256") != _sha256_file(path):
                raise ValueError(f"{label} shard digest does not match its manifest")
        if (
            vrfm_record.get("source_shard_sha256") != source_record.get("sha256")
            or vrfm_record.get("candidate_shard_sha256")
            != candidate_record.get("sha256")
        ):
            raise ValueError("VRFM prediction record does not bind source and candidate")

        source = load_source_shard(source_path)
        candidate = load_candidate_shard(candidate_path)
        vrfm_prediction = load_vrfm_residual_alpha_scan(vrfm_prediction_path)
        _require_matched_random_sample_budget(
            candidate,
            vrfm_prediction,
        )
        if (
            not np.array_equal(candidate["source_sample_ids"], source["sample_ids"])
            or not np.array_equal(vrfm_prediction["source_sample_ids"], source["sample_ids"])
            or not np.array_equal(vrfm_prediction["z"], candidate["z"])
            or not np.array_equal(
                vrfm_prediction["sample_seeds"], candidate["sample_seeds"]
            )
            or not np.array_equal(
                vrfm_prediction["latent_cluster_ids"],
                candidate["latent_cluster_ids"],
            )
            or str(vrfm_prediction["producer_git_commit"])
            != upstream_vrfm_commit
        ):
            raise ValueError("VRFM scene artifacts are not paired to current inputs")

        destination = prediction_root / f"{scene}.npz"
        if destination.is_file():
            arrays = load_matched_random_ablation(destination)
        else:
            if head is None:
                head = _camera_head(args.checkpoint_dir, args.device)
            generate_matched_random_ablation(
                source_path,
                candidate_path,
                vrfm_prediction_path,
                destination,
                camera_head=head,
                camera_head_checkpoint_sha256=camera_head_checkpoint_sha256,
                producer_git_commit=producer_git_commit,
                base_seed=args.matched_random_seed,
                transform_identity_sha256=transform_identity_sha256,
                device=args.device,
                batch_size=args.matched_random_batch_size,
            )
            arrays = load_matched_random_ablation(destination)
        expected_pairs = {
            "source_shard_sha256": str(source_record.get("sha256")),
            "candidate_shard_sha256": str(candidate_record.get("sha256")),
            "paired_vrfm_prediction_sha256": str(vrfm_record.get("sha256")),
            "vrfm_checkpoint_sha256": str(candidate["checkpoint_sha256"]),
            "camera_head_checkpoint_sha256": camera_head_checkpoint_sha256,
            "paired_vrfm_producer_git_commit": upstream_vrfm_commit,
            "producer_git_commit": producer_git_commit,
            "transform_identity_sha256": transform_identity_sha256,
        }
        for field, expected in expected_pairs.items():
            if str(arrays[field]) != expected:
                raise ValueError(f"existing matched random prediction does not bind {field}")
        if (
            int(arrays["base_seed"]) != args.matched_random_seed
            or not np.array_equal(
                arrays["alphas"], np.asarray(DEFAULT_ALPHAS, dtype=np.float64)
            )
            or not np.array_equal(arrays["source_sample_ids"], source["sample_ids"])
            or not np.array_equal(arrays["z"], candidate["z"])
            or not np.array_equal(arrays["sample_seeds"], candidate["sample_seeds"])
            or not np.array_equal(
                arrays["latent_cluster_ids"], candidate["latent_cluster_ids"]
            )
        ):
            raise ValueError("existing matched random prediction metadata differs")
        transform_sha256 = str(arrays["transform_sha256"])
        if expected_transform_sha256 is None:
            expected_transform_sha256 = transform_sha256
        elif transform_sha256 != expected_transform_sha256:
            raise ValueError("matched random scenes do not share one control transform")
        prediction_records.append(
            {
                "scene": scene,
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "source_shard_sha256": str(source_record.get("sha256")),
                "candidate_shard_sha256": str(candidate_record.get("sha256")),
                "paired_vrfm_prediction_sha256": str(vrfm_record.get("sha256")),
            }
        )
        print(f"[vrfm] matched-random decode {index + 1}/10 {scene}", flush=True)
    if head is not None:
        del head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert expected_transform_sha256 is not None

    prediction_manifest_path = (
        run_root / "manifests" / "matched_random_ablation_full_context_prediction_manifest.json"
    )
    prediction_manifest_payload = {
        "schema": "variational_camera_latent.matched_random_ablation_full_context_prediction_manifest.v1",
        "scene_count": 10,
        "overlap_count": 80,
        "samples_per_overlap": 32,
        "direction_count": 2560,
        "grid_cell_count": 20480,
        "unique_pose_candidate_count": 18000,
        "alphas": list(DEFAULT_ALPHAS),
        "decode_context_frames": 500,
        "camera_iterations": 4,
        "structured_null_replicate_count": 1,
        "base_seed": args.matched_random_seed,
        "transform_identity_sha256": transform_identity_sha256,
        "transform_sha256": expected_transform_sha256,
        "same_transform_across_scenes": True,
        "preserves_feature_row_gram_geometry": True,
        "producer_git_commit": producer_git_commit,
        "camera_head_checkpoint_sha256": camera_head_checkpoint_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "paired_vrfm_prediction_manifest_sha256": vrfm_prediction_manifest_sha256,
        "records": prediction_records,
    }
    if prediction_manifest_path.is_file() and _read_json(
        prediction_manifest_path
    ) != prediction_manifest_payload:
        raise ValueError("existing matched random prediction manifest differs")
    _atomic_json(prediction_manifest_path, prediction_manifest_payload)

    # Prediction-only artifacts are now sealed.  Privileged validation starts
    # below and cannot influence the shared control transform or decoded poses.
    phase1_completion_path = run_root / "verified_completion.json"
    phase1 = _load_signed_completion(
        phase1_completion_path,
        schema="variational_camera_latent.verified_completion.v1",
    )
    phase1_bindings = {
        "prediction_manifest_sha256": run_root
        / "manifests"
        / "prediction_manifest.json",
        "privileged_manifest_sha256": run_root
        / "manifests"
        / "privileged_manifest.json",
        "report_sha256": run_root / "reports" / "exploration_report.json",
    }
    for field, path in phase1_bindings.items():
        if phase1.get(field) != _sha256_file(path):
            raise ValueError(f"verified Phase 1 does not bind {path}")

    vrfm_privileged_manifest_path = (
        run_root
        / "manifests"
        / "vrfm_residual_alpha_scan_full_context_privileged_manifest.json"
    )
    vrfm_privileged_manifest = _read_json(vrfm_privileged_manifest_path)
    vrfm_privileged_records = vrfm_privileged_manifest.get("records")
    if (
        vrfm_privileged_manifest.get("schema")
        != "variational_camera_latent.vrfm_residual_alpha_scan_full_context_privileged_manifest.v1"
        or vrfm_privileged_manifest.get("scene_count") != 10
        or vrfm_privileged_manifest.get("overlap_count") != 80
        or vrfm_privileged_manifest.get("samples_per_overlap") != 32
        or vrfm_privileged_manifest.get("prediction_manifest_sha256")
        != vrfm_prediction_manifest_sha256
        or not isinstance(vrfm_privileged_records, list)
        or len(vrfm_privileged_records) != 10
        or any(not isinstance(row, dict) for row in vrfm_privileged_records)
    ):
        raise ValueError("verified VRFM privileged manifest is invalid")
    vrfm_privileged_manifest_sha256 = _sha256_file(vrfm_privileged_manifest_path)

    vrfm_report_path = (
        run_root / "reports" / "vrfm_residual_alpha_scan_full_context_report.json"
    )
    vrfm_completion_path = (
        run_root / "vrfm_residual_alpha_scan_full_context_verified_completion.json"
    )
    vrfm_completion = _load_signed_completion(
        vrfm_completion_path,
        schema=(
            "variational_camera_latent."
            "vrfm_residual_alpha_scan_full_context_verified_completion.v1"
        ),
    )
    expected_vrfm_completion = {
        "scene_count": 10,
        "overlap_count": 80,
        "direction_count": 2560,
        "grid_cell_count": 20480,
        "unique_pose_candidate_count": 18000,
        "prediction_manifest_sha256": vrfm_prediction_manifest_sha256,
        "privileged_manifest_sha256": vrfm_privileged_manifest_sha256,
        "report_sha256": _sha256_file(vrfm_report_path),
        "phase1_completion_sha256": _sha256_file(phase1_completion_path),
        "candidate_manifest_sha256": candidate_manifest_sha256,
    }
    for field, expected in expected_vrfm_completion.items():
        if vrfm_completion.get(field) != expected:
            raise ValueError(f"verified VRFM completion does not bind {field}")
    if vrfm_completion.get("git_commit") != upstream_vrfm_commit:
        raise ValueError("verified VRFM producer commit does not match")

    privileged_root = run_root / "privileged_labels" / "matched_random_vs_vrfm"
    privileged_records: list[dict[str, object]] = []
    privileged_paths: list[Path] = []
    for index, records in enumerate(
        zip(
            source_records,
            candidate_records,
            prediction_records,
            vrfm_prediction_records,
            vrfm_privileged_records,
        )
    ):
        (
            source_record,
            candidate_record,
            random_record,
            vrfm_record,
            vrfm_privileged_record,
        ) = records
        scene = scenes[index]
        if any(row.get("scene") != scene for row in records):
            raise ValueError("matched random privileged scene order differs")
        source_path = Path(str(source_record["path"]))
        candidate_path = Path(str(candidate_record["path"]))
        random_prediction_path = Path(str(random_record["path"]))
        vrfm_prediction_path = Path(str(vrfm_record["path"]))
        vrfm_privileged_path = Path(str(vrfm_privileged_record["path"]))
        expected_vrfm_privileged_path = (
            run_root
            / "privileged_labels"
            / "vrfm_residual_alpha_scan_full_context"
            / f"{scene}.npz"
        )
        if vrfm_privileged_path.resolve() != expected_vrfm_privileged_path.resolve():
            raise ValueError("VRFM privileged artifact path is outside its contract")
        if vrfm_privileged_record.get("sha256") != _sha256_file(
            vrfm_privileged_path
        ):
            raise ValueError("VRFM privileged shard digest does not match its manifest")
        source = load_source_shard(source_path)
        candidate = load_candidate_shard(candidate_path)
        vrfm_prediction = load_vrfm_residual_alpha_scan(vrfm_prediction_path)
        vrfm_privileged = load_vrfm_residual_privileged(vrfm_privileged_path)
        _require_matched_random_sample_budget(
            candidate,
            vrfm_prediction,
            vrfm_privileged,
        )
        if (
            not np.array_equal(
                vrfm_privileged["source_sample_ids"], source["sample_ids"]
            )
            or str(vrfm_privileged["prediction_sha256"])
            != vrfm_record.get("sha256")
        ):
            raise ValueError("VRFM privileged artifact is not paired to its prediction")
        destination = privileged_root / f"{scene}.npz"
        if destination.is_file():
            arrays = load_matched_random_privileged(destination)
        else:
            write_matched_random_privileged_sidecar(
                source_path,
                random_prediction_path,
                vrfm_prediction_path,
                vrfm_privileged_path,
                args.prepared_root / scene,
                destination,
            )
            arrays = load_matched_random_privileged(destination)
        expected_sidecar = {
            "random_prediction_sha256": str(random_record["sha256"]),
            "vrfm_prediction_sha256": str(vrfm_record["sha256"]),
            "vrfm_privileged_sha256": str(vrfm_privileged_record["sha256"]),
            "prepared_gt_sha256": prepared_gt_sha256(args.prepared_root / scene),
        }
        for field, expected in expected_sidecar.items():
            if str(arrays[field]) != expected:
                raise ValueError(f"existing matched random sidecar does not bind {field}")
        if not np.array_equal(arrays["source_sample_ids"], source["sample_ids"]):
            raise ValueError("matched random sidecar sample IDs differ from source")
        privileged_paths.append(destination)
        privileged_records.append(
            {"scene": scene, "path": str(destination), "sha256": _sha256_file(destination)}
        )
        print(f"[vrfm] matched-random privileged {index + 1}/10 {scene}", flush=True)

    privileged_manifest_path = (
        run_root / "manifests" / "matched_random_vs_vrfm_privileged_manifest.json"
    )
    privileged_manifest_payload = {
        "schema": "variational_camera_latent.matched_random_vs_vrfm_privileged_manifest.v1",
        "scene_count": 10,
        "overlap_count": 80,
        "samples_per_overlap": 32,
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "paired_vrfm_prediction_manifest_sha256": vrfm_prediction_manifest_sha256,
        "paired_vrfm_privileged_manifest_sha256": vrfm_privileged_manifest_sha256,
        "records": privileged_records,
    }
    if privileged_manifest_path.is_file() and _read_json(
        privileged_manifest_path
    ) != privileged_manifest_payload:
        raise ValueError("existing matched random privileged manifest differs")
    _atomic_json(privileged_manifest_path, privileged_manifest_payload)

    report_path = run_root / "reports" / "matched_random_vs_vrfm_pilot_report.json"
    report = write_matched_random_report(
        privileged_paths,
        report_path,
        min_improvement=args.alpha_min_improvement,
    )
    if (
        report.get("scene_count") != 10
        or report.get("inference_unit") != "scene"
        or report.get("structured_null_replicate_count") != 1
        or report.get("formal_training_attribution") is not False
    ):
        raise ValueError("matched random pilot report overstates or miscounts evidence")
    paired_report = report.get("paired_comparison")
    if not isinstance(paired_report, dict) or not isinstance(
        paired_report.get("diagnosis"), str
    ):
        raise ValueError("matched random pilot report has no paired diagnosis")
    if (
        report.get("diagnosis_basis") != "scene_level"
        or not isinstance(report.get("diagnosis"), str)
    ):
        raise ValueError("matched random pilot report has no scene-level diagnosis")
    if _require_clean_git_checkout(ROOT) != producer_git_commit:
        raise ValueError("matched random producer commit changed during execution")

    completion_path = run_root / "matched_random_ablation_pilot_verified_completion.json"
    unsigned = {
        "schema": "variational_camera_latent.matched_random_ablation_pilot_verified_completion.v1",
        "scene_count": 10,
        "overlap_count": 80,
        "direction_count_per_arm": 2560,
        "grid_cell_count_per_arm": 20480,
        "structured_null_replicate_count": 1,
        "formal_training_attribution": False,
        "diagnosis": report["diagnosis"],
        "diagnosis_basis": "scene_level",
        "min_improvement": float(args.alpha_min_improvement),
        "base_seed": args.matched_random_seed,
        "transform_identity_sha256": transform_identity_sha256,
        "transform_sha256": expected_transform_sha256,
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "privileged_manifest_sha256": _sha256_file(privileged_manifest_path),
        "report_sha256": _sha256_file(report_path),
        "phase1_completion_sha256": _sha256_file(phase1_completion_path),
        "vrfm_completion_sha256": _sha256_file(vrfm_completion_path),
        "git_commit": producer_git_commit,
    }
    if load_exact_completion(completion_path, unsigned) is None:
        if completion_path.exists():
            raise ValueError("existing matched random completion differs")
        write_completion(completion_path, unsigned)
    return args.run_root


def _load_current_matched_random_20q_plan(
    context: dict[str, object],
) -> tuple[Path, dict[str, object], str]:
    run_root = Path(context["run_root"])
    plan_path = _matched_random_20q_plan_path(run_root)
    _require_contract_path(
        plan_path,
        run_root / _MATCHED_RANDOM_20Q_PLAN_RELATIVE,
        run_root=run_root,
        label="20-Q plan",
        require_file=True,
    )
    plan = load_matched_random_ensemble_plan(plan_path)
    expected = {
        "producer_git_commit": context["producer_git_commit"],
        "camera_head_checkpoint_sha256": context["camera_head_checkpoint_sha256"],
        "source_manifest_sha256": context["source_manifest_sha256"],
        "candidate_manifest_sha256": context["candidate_manifest_sha256"],
        "vrfm_prediction_manifest_sha256": context["vrfm_prediction_manifest_sha256"],
        "transform_identity_sha256": context["transform_identity_sha256"],
        "reference_prediction_manifest_sha256": context[
            "reference_prediction_manifest_sha256"
        ],
        "reference_transform_sha256": context["reference_transform_sha256"],
        "scenes": context["scenes"],
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"20-Q plan does not bind current {field}")
    return plan_path, plan, _sha256_file(plan_path)


def create_matched_random_ensemble_plan(args: argparse.Namespace) -> Path:
    context = _load_matched_random_prediction_context(args)
    plan = build_matched_random_ensemble_plan(
        master_seed=args.matched_random_master_seed,
        transform_identity_sha256=str(context["transform_identity_sha256"]),
        reference_prediction_manifest_sha256=str(
            context["reference_prediction_manifest_sha256"]
        ),
        reference_transform_sha256=str(context["reference_transform_sha256"]),
        producer_git_commit=str(context["producer_git_commit"]),
        camera_head_checkpoint_sha256=str(
            context["camera_head_checkpoint_sha256"]
        ),
        source_manifest_sha256=str(context["source_manifest_sha256"]),
        candidate_manifest_sha256=str(context["candidate_manifest_sha256"]),
        vrfm_prediction_manifest_sha256=str(
            context["vrfm_prediction_manifest_sha256"]
        ),
        scenes=list(context["scenes"]),
    )
    plan_path = _matched_random_20q_plan_path(Path(context["run_root"]))
    _require_contract_path(
        plan_path,
        Path(context["run_root"]) / _MATCHED_RANDOM_20Q_PLAN_RELATIVE,
        run_root=Path(context["run_root"]),
        label="20-Q plan",
    )
    write_matched_random_ensemble_plan(plan_path, plan)
    return Path(context["run_root"])


def _matched_random_20q_replicate(
    plan: dict[str, object], replicate_index: int
) -> dict[str, object]:
    rows = plan.get("replicates")
    if not isinstance(rows, list) or len(rows) != FORMAL_REPLICATE_COUNT:
        raise ValueError("20-Q plan replicate table is invalid")
    row = rows[replicate_index]
    if (
        not isinstance(row, dict)
        or row.get("replicate_index") != replicate_index
        or row.get("replicate_id") != f"formal_null_{replicate_index:03d}"
    ):
        raise ValueError("20-Q plan replicate order differs")
    return row


def _validate_matched_random_20q_prediction_artifact(
    arrays: dict[str, np.ndarray],
    *,
    context: dict[str, object],
    source: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    source_record: dict[str, object],
    candidate_record: dict[str, object],
    vrfm_record: dict[str, object],
    replicate_seed: int,
    expected_transform_sha256: str,
) -> None:
    expected_pairs = {
        "source_shard_sha256": str(source_record.get("sha256")),
        "candidate_shard_sha256": str(candidate_record.get("sha256")),
        "paired_vrfm_prediction_sha256": str(vrfm_record.get("sha256")),
        "vrfm_checkpoint_sha256": str(candidate["checkpoint_sha256"]),
        "camera_head_checkpoint_sha256": str(
            context["camera_head_checkpoint_sha256"]
        ),
        "paired_vrfm_producer_git_commit": str(context["upstream_vrfm_commit"]),
        "producer_git_commit": str(context["producer_git_commit"]),
        "transform_identity_sha256": str(context["transform_identity_sha256"]),
        "transform_sha256": expected_transform_sha256,
    }
    for field, expected in expected_pairs.items():
        if str(arrays[field]) != expected:
            raise ValueError(f"20-Q prediction does not bind {field}")
    if (
        int(arrays["base_seed"]) != replicate_seed
        or not np.array_equal(
            arrays["alphas"], np.asarray(DEFAULT_ALPHAS, dtype=np.float64)
        )
        or not np.array_equal(arrays["source_sample_ids"], source["sample_ids"])
        or not np.array_equal(arrays["z"], candidate["z"])
        or not np.array_equal(arrays["sample_seeds"], candidate["sample_seeds"])
        or not np.array_equal(
            arrays["latent_cluster_ids"], candidate["latent_cluster_ids"]
        )
    ):
        raise ValueError("20-Q prediction metadata differs")


def run_matched_random_prediction_replicate(args: argparse.Namespace) -> Path:
    replicate_index = _matched_random_replicate_index(
        args.matched_random_replicate_index
    )
    context = _load_matched_random_prediction_context(args)
    run_root = Path(context["run_root"])
    plan_path, plan, plan_sha256 = _load_current_matched_random_20q_plan(context)
    row = _matched_random_20q_replicate(plan, replicate_index)
    replicate_seed = int(row["replicate_seed"])
    expected_transform_sha256 = str(row["expected_transform_sha256"])
    prediction_root = run_root / str(row["prediction_root"])
    prediction_manifest_path = run_root / str(row["prediction_manifest_path"])
    prediction_completion_path = run_root / str(row["prediction_completion_path"])
    expected_prefix = f"replicate_{replicate_index:03d}"
    expected_prediction_root = (
        run_root
        / "prediction_only"
        / "matched_random_ablation_20q_full_context"
        / expected_prefix
    )
    expected_prediction_manifest_path = (
        run_root
        / "manifests"
        / "matched_random_20q"
        / f"{expected_prefix}_prediction_manifest.json"
    )
    expected_prediction_completion_path = (
        run_root
        / "manifests"
        / "matched_random_20q"
        / f"{expected_prefix}_prediction_complete.json"
    )
    _require_contract_path(
        prediction_root,
        expected_prediction_root,
        run_root=run_root,
        label="20-Q prediction root",
    )
    _require_contract_path(
        prediction_manifest_path,
        expected_prediction_manifest_path,
        run_root=run_root,
        label="20-Q prediction manifest",
    )
    _require_contract_path(
        prediction_completion_path,
        expected_prediction_completion_path,
        run_root=run_root,
        label="20-Q prediction completion",
    )

    scenes = list(context["scenes"])
    source_records = list(context["source_records"])
    candidate_records = list(context["candidate_records"])
    vrfm_records = list(context["vrfm_prediction_records"])
    records: list[dict[str, object]] = []
    head = None
    for scene_index, (scene, source_record, candidate_record, vrfm_record) in enumerate(
        zip(scenes, source_records, candidate_records, vrfm_records)
    ):
        if any(
            record.get("scene") != scene
            for record in (source_record, candidate_record, vrfm_record)
        ):
            raise ValueError("20-Q prediction upstream scene order differs")
        source_path = Path(str(source_record.get("path")))
        candidate_path = Path(str(candidate_record.get("path")))
        vrfm_path = Path(str(vrfm_record.get("path")))
        expected_paths = (
            (source_path, run_root / "prediction_only" / "source" / f"{scene}.npz"),
            (
                candidate_path,
                run_root / "prediction_only" / "calibration_candidates" / f"{scene}.npz",
            ),
            (
                vrfm_path,
                run_root
                / "prediction_only"
                / "vrfm_residual_alpha_scan_full_context"
                / f"{scene}.npz",
            ),
        )
        for path, expected in expected_paths:
            _require_contract_path(
                path,
                expected,
                run_root=run_root,
                label="20-Q prediction upstream artifact",
                require_file=True,
            )
        for record, path in (
            (source_record, source_path),
            (candidate_record, candidate_path),
            (vrfm_record, vrfm_path),
        ):
            if record.get("sha256") != _sha256_file(path):
                raise ValueError("20-Q prediction upstream digest differs")
        if (
            vrfm_record.get("source_shard_sha256") != source_record.get("sha256")
            or vrfm_record.get("candidate_shard_sha256")
            != candidate_record.get("sha256")
        ):
            raise ValueError("20-Q VRFM record does not bind source and candidate")
        source = load_source_shard(source_path)
        candidate = load_candidate_shard(candidate_path)
        vrfm = load_vrfm_residual_alpha_scan(vrfm_path)
        _require_matched_random_sample_budget(candidate, vrfm)
        if (
            not np.array_equal(candidate["source_sample_ids"], source["sample_ids"])
            or not np.array_equal(vrfm["source_sample_ids"], source["sample_ids"])
            or not np.array_equal(vrfm["z"], candidate["z"])
            or not np.array_equal(vrfm["sample_seeds"], candidate["sample_seeds"])
            or not np.array_equal(
                vrfm["latent_cluster_ids"], candidate["latent_cluster_ids"]
            )
            or str(vrfm["producer_git_commit"])
            != str(context["upstream_vrfm_commit"])
        ):
            raise ValueError("20-Q VRFM artifact is not paired to current inputs")

        destination = prediction_root / f"{scene}.npz"
        _require_contract_path(
            destination,
            expected_prediction_root / f"{scene}.npz",
            run_root=run_root,
            label="20-Q prediction artifact",
        )
        if destination.is_file():
            arrays = load_matched_random_ablation(destination)
        else:
            if head is None:
                head = _camera_head(args.checkpoint_dir, args.device)
            generate_matched_random_ablation(
                source_path,
                candidate_path,
                vrfm_path,
                destination,
                camera_head=head,
                camera_head_checkpoint_sha256=str(
                    context["camera_head_checkpoint_sha256"]
                ),
                producer_git_commit=str(context["producer_git_commit"]),
                base_seed=replicate_seed,
                transform_identity_sha256=str(context["transform_identity_sha256"]),
                device=args.device,
                batch_size=args.matched_random_batch_size,
            )
            arrays = load_matched_random_ablation(destination)
        _validate_matched_random_20q_prediction_artifact(
            arrays,
            context=context,
            source=source,
            candidate=candidate,
            source_record=source_record,
            candidate_record=candidate_record,
            vrfm_record=vrfm_record,
            replicate_seed=replicate_seed,
            expected_transform_sha256=expected_transform_sha256,
        )
        records.append(
            {
                "scene": scene,
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "source_shard_sha256": str(source_record.get("sha256")),
                "candidate_shard_sha256": str(candidate_record.get("sha256")),
                "paired_vrfm_prediction_sha256": str(vrfm_record.get("sha256")),
            }
        )
        print(
            f"[vrfm] 20-Q prediction {replicate_index + 1}/20 "
            f"scene {scene_index + 1}/10 {scene}",
            flush=True,
        )
    if head is not None:
        del head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _require_exact_directory_entries(
        prediction_root,
        [prediction_root / f"{scene}.npz" for scene in scenes],
        entry_kind="file",
        label=f"20-Q prediction replicate {replicate_index}",
    )

    manifest = {
        "schema": _MATCHED_RANDOM_20Q_PREDICTION_SCHEMA,
        "replicate_index": replicate_index,
        "replicate_id": str(row["replicate_id"]),
        "replicate_seed": replicate_seed,
        "scene_count": 10,
        "overlap_count": 80,
        "samples_per_overlap": 32,
        "direction_count": 2560,
        "grid_cell_count": 20480,
        "unique_pose_candidate_count": 18000,
        "alphas": list(DEFAULT_ALPHAS),
        "plan_sha256": plan_sha256,
        "plan_digest": plan["plan_digest"],
        "transform_identity_sha256": context["transform_identity_sha256"],
        "transform_sha256": expected_transform_sha256,
        "producer_git_commit": context["producer_git_commit"],
        "camera_head_checkpoint_sha256": context[
            "camera_head_checkpoint_sha256"
        ],
        "source_manifest_sha256": context["source_manifest_sha256"],
        "candidate_manifest_sha256": context["candidate_manifest_sha256"],
        "vrfm_prediction_manifest_sha256": context[
            "vrfm_prediction_manifest_sha256"
        ],
        "records": records,
    }
    _write_exact_json(
        prediction_manifest_path,
        manifest,
        label=f"20-Q replicate {replicate_index} prediction manifest",
    )
    if _require_clean_git_checkout(ROOT) != context["producer_git_commit"]:
        raise ValueError("20-Q producer commit changed during prediction")
    completion = {
        "schema": _MATCHED_RANDOM_20Q_PREDICTION_COMPLETION_SCHEMA,
        "replicate_index": replicate_index,
        "replicate_id": str(row["replicate_id"]),
        "replicate_seed": replicate_seed,
        "scene_count": 10,
        "prediction_manifest_sha256": _sha256_file(prediction_manifest_path),
        "plan_sha256": plan_sha256,
        "plan_digest": plan["plan_digest"],
        "transform_identity_sha256": context["transform_identity_sha256"],
        "transform_sha256": expected_transform_sha256,
        "git_commit": context["producer_git_commit"],
    }
    if load_exact_completion(prediction_completion_path, completion) is None:
        if prediction_completion_path.exists():
            raise ValueError("existing 20-Q prediction completion differs")
        write_completion(prediction_completion_path, completion)
    return run_root


def _validate_matched_random_prediction_replicates(
    context: dict[str, object],
    plan: dict[str, object],
    plan_sha256: str,
) -> list[dict[str, object]]:
    """Validate every sealed prediction before privileged state is observable."""
    run_root = Path(context["run_root"])
    scenes = list(context["scenes"])
    source_records = list(context["source_records"])
    candidate_records = list(context["candidate_records"])
    vrfm_records = list(context["vrfm_prediction_records"])
    validated: list[dict[str, object]] = []
    prediction_family_root = (
        run_root
        / "prediction_only"
        / "matched_random_ablation_20q_full_context"
    )
    _require_contract_path(
        prediction_family_root,
        prediction_family_root,
        run_root=run_root,
        label="20-Q prediction family root",
    )
    _require_exact_directory_entries(
        prediction_family_root,
        [
            prediction_family_root / f"replicate_{index:03d}"
            for index in range(FORMAL_REPLICATE_COUNT)
        ],
        entry_kind="directory",
        label="20-Q prediction family",
    )
    for replicate_index in range(FORMAL_REPLICATE_COUNT):
        row = _matched_random_20q_replicate(plan, replicate_index)
        replicate_seed = int(row["replicate_seed"])
        transform_sha256 = str(row["expected_transform_sha256"])
        prefix = f"replicate_{replicate_index:03d}"
        prediction_root = run_root / str(row["prediction_root"])
        manifest_path = run_root / str(row["prediction_manifest_path"])
        completion_path = run_root / str(row["prediction_completion_path"])
        _require_contract_path(
            prediction_root,
            prediction_family_root / prefix,
            run_root=run_root,
            label="20-Q prediction root",
        )
        _require_contract_path(
            manifest_path,
            run_root
            / "manifests"
            / "matched_random_20q"
            / f"{prefix}_prediction_manifest.json",
            run_root=run_root,
            label="20-Q prediction manifest",
            require_file=True,
        )
        _require_contract_path(
            completion_path,
            run_root
            / "manifests"
            / "matched_random_20q"
            / f"{prefix}_prediction_complete.json",
            run_root=run_root,
            label="20-Q prediction completion",
            require_file=True,
        )
        _require_exact_directory_entries(
            prediction_root,
            [prediction_root / f"{scene}.npz" for scene in scenes],
            entry_kind="file",
            label=f"20-Q prediction replicate {replicate_index}",
        )

        manifest = _read_json(manifest_path)
        records = manifest.get("records")
        expected_manifest = {
            "schema": _MATCHED_RANDOM_20Q_PREDICTION_SCHEMA,
            "replicate_index": replicate_index,
            "replicate_id": str(row["replicate_id"]),
            "replicate_seed": replicate_seed,
            "scene_count": 10,
            "overlap_count": 80,
            "samples_per_overlap": 32,
            "direction_count": 2560,
            "grid_cell_count": 20480,
            "unique_pose_candidate_count": 18000,
            "alphas": list(DEFAULT_ALPHAS),
            "plan_sha256": plan_sha256,
            "plan_digest": plan["plan_digest"],
            "transform_identity_sha256": context[
                "transform_identity_sha256"
            ],
            "transform_sha256": transform_sha256,
            "producer_git_commit": context["producer_git_commit"],
            "camera_head_checkpoint_sha256": context[
                "camera_head_checkpoint_sha256"
            ],
            "source_manifest_sha256": context["source_manifest_sha256"],
            "candidate_manifest_sha256": context["candidate_manifest_sha256"],
            "vrfm_prediction_manifest_sha256": context[
                "vrfm_prediction_manifest_sha256"
            ],
        }
        if (
            not isinstance(records, list)
            or len(records) != 10
            or any(not isinstance(record, dict) for record in records)
            or {key: value for key, value in manifest.items() if key != "records"}
            != expected_manifest
        ):
            raise ValueError(
                f"20-Q replicate {replicate_index} prediction manifest differs"
            )
        manifest_sha256 = _sha256_file(manifest_path)
        expected_completion = {
            "schema": _MATCHED_RANDOM_20Q_PREDICTION_COMPLETION_SCHEMA,
            "replicate_index": replicate_index,
            "replicate_id": str(row["replicate_id"]),
            "replicate_seed": replicate_seed,
            "scene_count": 10,
            "prediction_manifest_sha256": manifest_sha256,
            "plan_sha256": plan_sha256,
            "plan_digest": plan["plan_digest"],
            "transform_identity_sha256": context[
                "transform_identity_sha256"
            ],
            "transform_sha256": transform_sha256,
            "git_commit": context["producer_git_commit"],
        }
        completion = _load_signed_completion(
            completion_path,
            schema=_MATCHED_RANDOM_20Q_PREDICTION_COMPLETION_SCHEMA,
        )
        if completion != expected_completion:
            raise ValueError(
                f"20-Q replicate {replicate_index} prediction completion differs"
            )

        validated_records: list[dict[str, object]] = []
        for scene_index, (
            scene,
            source_record,
            candidate_record,
            vrfm_record,
            record,
        ) in enumerate(
            zip(
                scenes,
                source_records,
                candidate_records,
                vrfm_records,
                records,
            )
        ):
            prediction_path = Path(str(record.get("path")))
            expected_prediction_path = prediction_root / f"{scene}.npz"
            expected_record = {
                "scene": scene,
                "path": str(expected_prediction_path),
                "sha256": record.get("sha256"),
                "source_shard_sha256": str(source_record.get("sha256")),
                "candidate_shard_sha256": str(candidate_record.get("sha256")),
                "paired_vrfm_prediction_sha256": str(vrfm_record.get("sha256")),
            }
            _require_contract_path(
                prediction_path,
                expected_prediction_path,
                run_root=run_root,
                label="20-Q prediction artifact",
                require_file=True,
            )
            if (
                record != expected_record
                or record.get("sha256") != _sha256_file(prediction_path)
            ):
                raise ValueError(
                    f"20-Q replicate {replicate_index} prediction record differs"
                )
            source_path = Path(str(source_record.get("path")))
            candidate_path = Path(str(candidate_record.get("path")))
            vrfm_path = Path(str(vrfm_record.get("path")))
            expected_upstream_paths = (
                (
                    source_path,
                    run_root / "prediction_only" / "source" / f"{scene}.npz",
                ),
                (
                    candidate_path,
                    run_root
                    / "prediction_only"
                    / "calibration_candidates"
                    / f"{scene}.npz",
                ),
                (
                    vrfm_path,
                    run_root
                    / "prediction_only"
                    / "vrfm_residual_alpha_scan_full_context"
                    / f"{scene}.npz",
                ),
            )
            for path, expected in expected_upstream_paths:
                _require_contract_path(
                    path,
                    expected,
                    run_root=run_root,
                    label="20-Q prediction upstream artifact",
                    require_file=True,
                )
            for upstream_record, upstream_path in (
                (source_record, source_path),
                (candidate_record, candidate_path),
                (vrfm_record, vrfm_path),
            ):
                if upstream_record.get("sha256") != _sha256_file(upstream_path):
                    raise ValueError("20-Q prediction upstream digest differs")
            if (
                vrfm_record.get("source_shard_sha256")
                != source_record.get("sha256")
                or vrfm_record.get("candidate_shard_sha256")
                != candidate_record.get("sha256")
            ):
                raise ValueError("20-Q VRFM manifest pairing differs")
            source = load_source_shard(source_path)
            candidate = load_candidate_shard(candidate_path)
            vrfm = load_vrfm_residual_alpha_scan(vrfm_path)
            _require_matched_random_sample_budget(candidate, vrfm)
            if (
                source_record.get("scene") != scene
                or candidate_record.get("scene") != scene
                or vrfm_record.get("scene") != scene
                or not np.array_equal(
                    candidate["source_sample_ids"], source["sample_ids"]
                )
                or not np.array_equal(vrfm["source_sample_ids"], source["sample_ids"])
                or not np.array_equal(vrfm["z"], candidate["z"])
                or not np.array_equal(vrfm["sample_seeds"], candidate["sample_seeds"])
                or not np.array_equal(
                    vrfm["latent_cluster_ids"], candidate["latent_cluster_ids"]
                )
                or str(vrfm["producer_git_commit"])
                != context["upstream_vrfm_commit"]
            ):
                raise ValueError("20-Q prediction upstream pairing differs")
            arrays = load_matched_random_ablation(prediction_path)
            _validate_matched_random_20q_prediction_artifact(
                arrays,
                context=context,
                source=source,
                candidate=candidate,
                source_record=source_record,
                candidate_record=candidate_record,
                vrfm_record=vrfm_record,
                replicate_seed=replicate_seed,
                expected_transform_sha256=transform_sha256,
            )
            validated_records.append(dict(record))
            print(
                f"[vrfm] validate 20-Q prediction {replicate_index + 1}/20 "
                f"scene {scene_index + 1}/10 {scene}",
                flush=True,
            )
        validated.append(
            {
                "replicate_index": replicate_index,
                "replicate_id": str(row["replicate_id"]),
                "replicate_seed": replicate_seed,
                "transform_sha256": transform_sha256,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_sha256,
                "completion_path": completion_path,
                "completion_sha256": _sha256_file(completion_path),
                "records": validated_records,
            }
        )
    return validated


def _load_matched_random_privileged_context(
    args: argparse.Namespace,
    context: dict[str, object],
) -> dict[str, object]:
    """Load privileged upstream only after the prediction barrier is sealed."""
    run_root = Path(context["run_root"])
    phase1_completion_path = run_root / "verified_completion.json"
    _require_contract_path(
        phase1_completion_path,
        phase1_completion_path,
        run_root=run_root,
        label="Phase 1 completion",
        require_file=True,
    )
    phase1 = _load_signed_completion(
        phase1_completion_path,
        schema="variational_camera_latent.verified_completion.v1",
    )
    for field, path in {
        "prediction_manifest_sha256": run_root
        / "manifests"
        / "prediction_manifest.json",
        "privileged_manifest_sha256": run_root
        / "manifests"
        / "privileged_manifest.json",
        "report_sha256": run_root / "reports" / "exploration_report.json",
    }.items():
        if phase1.get(field) != _sha256_file(path):
            raise ValueError(f"verified Phase 1 does not bind {path}")

    vrfm_privileged_manifest_path = (
        run_root
        / "manifests"
        / "vrfm_residual_alpha_scan_full_context_privileged_manifest.json"
    )
    _require_contract_path(
        vrfm_privileged_manifest_path,
        vrfm_privileged_manifest_path,
        run_root=run_root,
        label="VRFM privileged manifest",
        require_file=True,
    )
    vrfm_privileged_manifest = _read_json(vrfm_privileged_manifest_path)
    vrfm_privileged_records = vrfm_privileged_manifest.get("records")
    if (
        vrfm_privileged_manifest.get("schema")
        != "variational_camera_latent.vrfm_residual_alpha_scan_full_context_privileged_manifest.v1"
        or vrfm_privileged_manifest.get("scene_count") != 10
        or vrfm_privileged_manifest.get("overlap_count") != 80
        or vrfm_privileged_manifest.get("samples_per_overlap") != 32
        or vrfm_privileged_manifest.get("prediction_manifest_sha256")
        != context["vrfm_prediction_manifest_sha256"]
        or not isinstance(vrfm_privileged_records, list)
        or len(vrfm_privileged_records) != 10
        or any(not isinstance(record, dict) for record in vrfm_privileged_records)
    ):
        raise ValueError("verified VRFM privileged manifest is invalid")
    vrfm_privileged_manifest_sha256 = _sha256_file(
        vrfm_privileged_manifest_path
    )

    vrfm_report_path = (
        run_root / "reports" / "vrfm_residual_alpha_scan_full_context_report.json"
    )
    vrfm_completion_path = (
        run_root / "vrfm_residual_alpha_scan_full_context_verified_completion.json"
    )
    for path, label in (
        (vrfm_report_path, "VRFM report"),
        (vrfm_completion_path, "VRFM completion"),
    ):
        _require_contract_path(
            path,
            path,
            run_root=run_root,
            label=label,
            require_file=True,
        )
    vrfm_completion = _load_signed_completion(
        vrfm_completion_path,
        schema=(
            "variational_camera_latent."
            "vrfm_residual_alpha_scan_full_context_verified_completion.v1"
        ),
    )
    expected_vrfm_completion = {
        "scene_count": 10,
        "overlap_count": 80,
        "direction_count": 2560,
        "grid_cell_count": 20480,
        "unique_pose_candidate_count": 18000,
        "prediction_manifest_sha256": context[
            "vrfm_prediction_manifest_sha256"
        ],
        "privileged_manifest_sha256": vrfm_privileged_manifest_sha256,
        "report_sha256": _sha256_file(vrfm_report_path),
        "phase1_completion_sha256": _sha256_file(phase1_completion_path),
        "candidate_manifest_sha256": context["candidate_manifest_sha256"],
    }
    for field, expected in expected_vrfm_completion.items():
        if vrfm_completion.get(field) != expected:
            raise ValueError(f"verified VRFM completion does not bind {field}")
    if vrfm_completion.get("git_commit") != context["upstream_vrfm_commit"]:
        raise ValueError("verified VRFM producer commit does not match")
    return {
        "phase1_completion_path": phase1_completion_path,
        "vrfm_completion_path": vrfm_completion_path,
        "vrfm_privileged_manifest_path": vrfm_privileged_manifest_path,
        "vrfm_privileged_manifest_sha256": vrfm_privileged_manifest_sha256,
        "vrfm_privileged_records": vrfm_privileged_records,
    }


def _load_observed_matched_random_pilot_reference(
    args: argparse.Namespace,
    context: dict[str, object],
    privileged: dict[str, object],
) -> dict[str, object]:
    """Validate the historical pilot and return its descriptive provenance."""
    run_root = Path(context["run_root"])
    scenes = list(context["scenes"])
    pilot_prediction_manifest_path = (
        run_root
        / "manifests"
        / "matched_random_ablation_full_context_prediction_manifest.json"
    )
    _require_contract_path(
        pilot_prediction_manifest_path,
        pilot_prediction_manifest_path,
        run_root=run_root,
        label="pilot reference prediction manifest",
        require_file=True,
    )
    pilot_prediction_manifest = _read_json(pilot_prediction_manifest_path)
    pilot_prediction_records = pilot_prediction_manifest.get("records")
    if (
        _sha256_file(pilot_prediction_manifest_path)
        != context["reference_prediction_manifest_sha256"]
        or not isinstance(pilot_prediction_records, list)
        or len(pilot_prediction_records) != 10
    ):
        raise ValueError("observed Q0 prediction manifest differs")

    pilot_privileged_manifest_path = (
        run_root
        / "manifests"
        / "matched_random_vs_vrfm_privileged_manifest.json"
    )
    _require_contract_path(
        pilot_privileged_manifest_path,
        pilot_privileged_manifest_path,
        run_root=run_root,
        label="pilot reference privileged manifest",
        require_file=True,
    )
    pilot_privileged_manifest = _read_json(pilot_privileged_manifest_path)
    pilot_privileged_records = pilot_privileged_manifest.get("records")
    expected_pilot_privileged_manifest = {
        "schema": "variational_camera_latent.matched_random_vs_vrfm_privileged_manifest.v1",
        "scene_count": 10,
        "overlap_count": 80,
        "samples_per_overlap": 32,
        "prediction_manifest_sha256": context[
            "reference_prediction_manifest_sha256"
        ],
        "paired_vrfm_prediction_manifest_sha256": context[
            "vrfm_prediction_manifest_sha256"
        ],
        "paired_vrfm_privileged_manifest_sha256": privileged[
            "vrfm_privileged_manifest_sha256"
        ],
    }
    if (
        not isinstance(pilot_privileged_records, list)
        or len(pilot_privileged_records) != 10
        or any(not isinstance(record, dict) for record in pilot_privileged_records)
        or {
            key: value
            for key, value in pilot_privileged_manifest.items()
            if key != "records"
        }
        != expected_pilot_privileged_manifest
    ):
        raise ValueError("observed Q0 privileged manifest differs")

    pilot_report_path = (
        run_root / "reports" / "matched_random_vs_vrfm_pilot_report.json"
    )
    pilot_completion_path = (
        run_root / "matched_random_ablation_pilot_verified_completion.json"
    )
    for path, label in (
        (pilot_report_path, "pilot reference report"),
        (pilot_completion_path, "pilot reference completion"),
    ):
        _require_contract_path(
            path,
            path,
            run_root=run_root,
            label=label,
            require_file=True,
        )
    pilot_completion = _load_signed_completion(
        pilot_completion_path,
        schema=(
            "variational_camera_latent."
            "matched_random_ablation_pilot_verified_completion.v1"
        ),
    )
    pilot_completion_bindings = {
        "scene_count": 10,
        "overlap_count": 80,
        "direction_count_per_arm": 2560,
        "grid_cell_count_per_arm": 20480,
        "structured_null_replicate_count": 1,
        "formal_training_attribution": False,
        "base_seed": pilot_prediction_manifest.get("base_seed"),
        "transform_identity_sha256": context["transform_identity_sha256"],
        "transform_sha256": context["reference_transform_sha256"],
        "prediction_manifest_sha256": context[
            "reference_prediction_manifest_sha256"
        ],
        "privileged_manifest_sha256": _sha256_file(
            pilot_privileged_manifest_path
        ),
        "report_sha256": _sha256_file(pilot_report_path),
        "phase1_completion_sha256": _sha256_file(
            Path(privileged["phase1_completion_path"])
        ),
        "vrfm_completion_sha256": _sha256_file(
            Path(privileged["vrfm_completion_path"])
        ),
        "git_commit": pilot_prediction_manifest.get("producer_git_commit"),
    }
    for field, expected in pilot_completion_bindings.items():
        if pilot_completion.get(field) != expected:
            raise ValueError(f"observed Q0 completion does not bind {field}")

    vrfm_records = list(context["vrfm_prediction_records"])
    vrfm_privileged_records = list(privileged["vrfm_privileged_records"])
    overlap_best = np.empty((10, 8), dtype=np.float64)
    pilot_sidecar_root = (
        run_root / "privileged_labels" / "matched_random_vs_vrfm"
    )
    _require_contract_path(
        pilot_sidecar_root,
        pilot_sidecar_root,
        run_root=run_root,
        label="pilot reference sidecar root",
    )
    _require_exact_directory_entries(
        pilot_sidecar_root,
        [pilot_sidecar_root / f"{scene}.npz" for scene in scenes],
        entry_kind="file",
        label="pilot reference sidecars",
    )
    for scene_index, (
        scene,
        prediction_record,
        privileged_record,
        vrfm_record,
        vrfm_privileged_record,
    ) in enumerate(
        zip(
            scenes,
            pilot_prediction_records,
            pilot_privileged_records,
            vrfm_records,
            vrfm_privileged_records,
        )
    ):
        path = Path(str(privileged_record.get("path")))
        expected_path = (
            run_root
            / "privileged_labels"
            / "matched_random_vs_vrfm"
            / f"{scene}.npz"
        )
        _require_contract_path(
            path,
            expected_path,
            run_root=run_root,
            label="pilot reference sidecar",
            require_file=True,
        )
        if (
            prediction_record.get("scene") != scene
            or privileged_record.get("scene") != scene
            or privileged_record.get("sha256") != _sha256_file(path)
        ):
            raise ValueError("observed Q0 privileged record differs")
        arrays = load_matched_random_privileged(path)
        expected_bindings = {
            "random_prediction_sha256": prediction_record.get("sha256"),
            "vrfm_prediction_sha256": vrfm_record.get("sha256"),
            "vrfm_privileged_sha256": vrfm_privileged_record.get("sha256"),
            "prepared_gt_sha256": prepared_gt_sha256(
                args.prepared_root / scene
            ),
        }
        for field, expected in expected_bindings.items():
            if str(arrays[field]) != expected:
                raise ValueError(f"observed Q0 sidecar does not bind {field}")
        overlap_best[scene_index] = np.maximum(
            0.0,
            np.max(
                arrays["random_relative_improvement"][:, :, 1:],
                axis=(1, 2),
            ),
        )
    return {
        "label": "pilot_reference",
        "score": float(np.mean(np.median(overlap_best, axis=1))),
        "included_in_formal_null": False,
        "included_in_p_values": False,
        "transform_sha256": context["reference_transform_sha256"],
        "prediction_manifest_sha256": context[
            "reference_prediction_manifest_sha256"
        ],
        "privileged_manifest_sha256": _sha256_file(
            pilot_privileged_manifest_path
        ),
        "report_sha256": _sha256_file(pilot_report_path),
        "completion_sha256": _sha256_file(pilot_completion_path),
    }


def finalize_matched_random_ensemble(args: argparse.Namespace) -> Path:
    context = _load_matched_random_prediction_context(args)
    run_root = Path(context["run_root"])
    plan_path, plan, plan_sha256 = _load_current_matched_random_20q_plan(context)

    # This is the fail-closed prediction barrier: every signed completion,
    # manifest, digest, schema, and shard is checked before any privileged API.
    validated = _validate_matched_random_prediction_replicates(
        context,
        plan,
        plan_sha256,
    )
    aggregate_prediction_manifest_path = (
        run_root / "manifests" / "matched_random_20q_prediction_manifest.json"
    )
    aggregate_prediction_manifest = {
        "schema": _MATCHED_RANDOM_20Q_AGGREGATE_PREDICTION_SCHEMA,
        "replicate_count": FORMAL_REPLICATE_COUNT,
        "scene_count": 10,
        "prediction_artifact_count": 200,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "plan_digest": plan["plan_digest"],
        "producer_git_commit": context["producer_git_commit"],
        "source_manifest_sha256": context["source_manifest_sha256"],
        "candidate_manifest_sha256": context["candidate_manifest_sha256"],
        "vrfm_prediction_manifest_sha256": context[
            "vrfm_prediction_manifest_sha256"
        ],
        "transform_identity_sha256": context["transform_identity_sha256"],
        "replicates": [
            {
                "replicate_index": record["replicate_index"],
                "replicate_id": record["replicate_id"],
                "replicate_seed": record["replicate_seed"],
                "transform_sha256": record["transform_sha256"],
                "prediction_manifest_path": str(record["manifest_path"]),
                "prediction_manifest_sha256": record["manifest_sha256"],
                "prediction_completion_path": str(record["completion_path"]),
                "prediction_completion_sha256": record["completion_sha256"],
            }
            for record in validated
        ],
    }
    prediction_barrier = _seal_matched_random_aggregate_prediction_barrier(
        run_root=run_root,
        manifest=aggregate_prediction_manifest,
    )
    if prediction_barrier["manifest_path"] != aggregate_prediction_manifest_path:
        raise ValueError("20-Q aggregate prediction barrier path differs")
    aggregate_prediction_manifest_sha256 = str(
        prediction_barrier["manifest_sha256"]
    )
    aggregate_prediction_completion_sha256 = str(
        prediction_barrier["completion_sha256"]
    )

    # Privileged labels and prepared GT become observable only below this line.
    privileged = _load_matched_random_privileged_context(args, context)
    pilot_reference = _load_observed_matched_random_pilot_reference(
        args,
        context,
        privileged,
    )
    observed_pilot_score = float(pilot_reference["score"])
    scenes = list(context["scenes"])
    source_records = list(context["source_records"])
    candidate_records = list(context["candidate_records"])
    vrfm_records = list(context["vrfm_prediction_records"])
    vrfm_privileged_records = list(privileged["vrfm_privileged_records"])
    identity_overlap_best = np.empty((10, 8), dtype=np.float64)
    random_overlap_best = np.empty(
        (FORMAL_REPLICATE_COUNT, 10, 8), dtype=np.float64
    )
    replicate_outputs: list[dict[str, object]] = []
    sidecar_family_root = (
        run_root
        / "privileged_labels"
        / "matched_random_vs_vrfm_20q"
    )
    _require_contract_path(
        sidecar_family_root,
        sidecar_family_root,
        run_root=run_root,
        label="20-Q privileged family root",
    )
    for validated_replicate in validated:
        replicate_index = int(validated_replicate["replicate_index"])
        plan_row = _matched_random_20q_replicate(plan, replicate_index)
        replicate_id = str(plan_row["replicate_id"])
        replicate_seed = int(plan_row["replicate_seed"])
        transform_sha256 = str(plan_row["expected_transform_sha256"])
        if (
            validated_replicate["replicate_id"] != replicate_id
            or validated_replicate["replicate_seed"] != replicate_seed
            or validated_replicate["transform_sha256"] != transform_sha256
        ):
            raise ValueError("20-Q validated replicate does not bind its plan row")
        prefix = f"replicate_{replicate_index:03d}"
        sidecar_root = sidecar_family_root / prefix
        _require_contract_path(
            sidecar_root,
            sidecar_family_root / prefix,
            run_root=run_root,
            label="20-Q privileged replicate root",
        )
        sidecar_paths: list[Path] = []
        sidecar_records: list[dict[str, object]] = []
        random_records = list(validated_replicate["records"])
        for scene_index, (
            scene,
            source_record,
            candidate_record,
            random_record,
            vrfm_record,
            vrfm_privileged_record,
        ) in enumerate(
            zip(
                scenes,
                source_records,
                candidate_records,
                random_records,
                vrfm_records,
                vrfm_privileged_records,
            )
        ):
            if any(
                record.get("scene") != scene
                for record in (
                    source_record,
                    candidate_record,
                    random_record,
                    vrfm_record,
                    vrfm_privileged_record,
                )
            ):
                raise ValueError("20-Q privileged scene order differs")
            source_path = Path(str(source_record["path"]))
            candidate_path = Path(str(candidate_record["path"]))
            random_prediction_path = Path(str(random_record["path"]))
            vrfm_prediction_path = Path(str(vrfm_record["path"]))
            vrfm_privileged_path = Path(str(vrfm_privileged_record["path"]))
            expected_vrfm_privileged_path = (
                run_root
                / "privileged_labels"
                / "vrfm_residual_alpha_scan_full_context"
                / f"{scene}.npz"
            )
            _require_contract_path(
                vrfm_privileged_path,
                expected_vrfm_privileged_path,
                run_root=run_root,
                label="20-Q VRFM privileged artifact",
                require_file=True,
            )
            if (
                vrfm_privileged_record.get("sha256")
                != _sha256_file(vrfm_privileged_path)
            ):
                raise ValueError("20-Q VRFM privileged record differs")
            source = load_source_shard(source_path)
            candidate = load_candidate_shard(candidate_path)
            vrfm_prediction = load_vrfm_residual_alpha_scan(
                vrfm_prediction_path
            )
            vrfm_privileged = load_vrfm_residual_privileged(
                vrfm_privileged_path
            )
            _require_matched_random_sample_budget(
                candidate,
                vrfm_prediction,
                vrfm_privileged,
            )
            if (
                not np.array_equal(
                    vrfm_privileged["source_sample_ids"], source["sample_ids"]
                )
                or str(vrfm_privileged["prediction_sha256"])
                != vrfm_record.get("sha256")
            ):
                raise ValueError("20-Q VRFM privileged artifact pairing differs")
            destination = sidecar_root / f"{scene}.npz"
            _require_contract_path(
                destination,
                sidecar_family_root / prefix / f"{scene}.npz",
                run_root=run_root,
                label="20-Q privileged sidecar",
            )
            if destination.is_file():
                arrays = load_matched_random_privileged(destination)
            else:
                write_matched_random_privileged_sidecar(
                    source_path,
                    random_prediction_path,
                    vrfm_prediction_path,
                    vrfm_privileged_path,
                    args.prepared_root / scene,
                    destination,
                )
                arrays = load_matched_random_privileged(destination)
            expected_sidecar = {
                "random_prediction_sha256": str(random_record["sha256"]),
                "vrfm_prediction_sha256": str(vrfm_record["sha256"]),
                "vrfm_privileged_sha256": str(
                    vrfm_privileged_record["sha256"]
                ),
                "prepared_gt_sha256": prepared_gt_sha256(
                    args.prepared_root / scene
                ),
            }
            for field, expected in expected_sidecar.items():
                if str(arrays[field]) != expected:
                    raise ValueError(f"20-Q sidecar does not bind {field}")
            if not np.array_equal(
                arrays["source_sample_ids"], source["sample_ids"]
            ):
                raise ValueError("20-Q sidecar sample IDs differ from source")
            identity_score = np.maximum(
                0.0,
                np.max(
                    arrays["vrfm_relative_improvement"][:, :, 1:],
                    axis=(1, 2),
                ),
            )
            random_score = np.maximum(
                0.0,
                np.max(
                    arrays["random_relative_improvement"][:, :, 1:],
                    axis=(1, 2),
                ),
            )
            if replicate_index == 0:
                identity_overlap_best[scene_index] = identity_score
            elif not np.array_equal(
                identity_overlap_best[scene_index], identity_score
            ):
                raise ValueError("20-Q identity score differs across replicates")
            random_overlap_best[replicate_index, scene_index] = random_score
            sidecar_sha256 = _sha256_file(destination)
            sidecar_paths.append(destination)
            sidecar_records.append(
                {
                    "scene": scene,
                    "path": str(destination),
                    "sha256": sidecar_sha256,
                    **expected_sidecar,
                }
            )
            print(
                f"[vrfm] 20-Q privileged {replicate_index + 1}/20 "
                f"scene {scene_index + 1}/10 {scene}",
                flush=True,
            )

        _require_exact_directory_entries(
            sidecar_root,
            [sidecar_root / f"{scene}.npz" for scene in scenes],
            entry_kind="file",
            label=f"20-Q privileged replicate {replicate_index}",
        )

        privileged_manifest_path = (
            run_root
            / "manifests"
            / "matched_random_20q"
            / f"{prefix}_privileged_manifest.json"
        )
        _require_contract_path(
            privileged_manifest_path,
            privileged_manifest_path,
            run_root=run_root,
            label="20-Q privileged manifest",
        )
        privileged_manifest_payload = {
            "schema": _MATCHED_RANDOM_20Q_PRIVILEGED_SCHEMA,
            "replicate_index": replicate_index,
            "replicate_id": replicate_id,
            "replicate_seed": replicate_seed,
            "transform_sha256": transform_sha256,
            "scene_count": 10,
            "sidecar_count": 10,
            "plan_sha256": plan_sha256,
            "plan_digest": plan["plan_digest"],
            "aggregate_prediction_manifest_sha256": (
                aggregate_prediction_manifest_sha256
            ),
            "aggregate_prediction_completion_sha256": (
                aggregate_prediction_completion_sha256
            ),
            "prediction_manifest_sha256": validated_replicate[
                "manifest_sha256"
            ],
            "prediction_completion_sha256": validated_replicate[
                "completion_sha256"
            ],
            "vrfm_prediction_manifest_sha256": context[
                "vrfm_prediction_manifest_sha256"
            ],
            "vrfm_privileged_manifest_sha256": privileged[
                "vrfm_privileged_manifest_sha256"
            ],
            "records": sidecar_records,
        }
        _write_exact_json(
            privileged_manifest_path,
            privileged_manifest_payload,
            label=f"20-Q replicate {replicate_index} privileged manifest",
        )
        report_path = (
            run_root
            / "reports"
            / "matched_random_vs_vrfm_20q"
            / f"{prefix}_report.json"
        )
        _require_contract_path(
            report_path,
            report_path,
            run_root=run_root,
            label="20-Q replicate report",
        )
        write_matched_random_report(
            sidecar_paths,
            report_path,
            min_improvement=args.alpha_min_improvement,
        )
        completion_path = (
            run_root
            / "manifests"
            / "matched_random_20q"
            / f"{prefix}_verified_completion.json"
        )
        _require_contract_path(
            completion_path,
            completion_path,
            run_root=run_root,
            label="20-Q replicate completion",
        )
        completion_payload = {
            "schema": _MATCHED_RANDOM_20Q_REPLICATE_COMPLETION_SCHEMA,
            "replicate_index": replicate_index,
            "replicate_id": replicate_id,
            "replicate_seed": replicate_seed,
            "transform_sha256": transform_sha256,
            "scene_count": 10,
            "sidecar_count": 10,
            "plan_sha256": plan_sha256,
            "plan_digest": plan["plan_digest"],
            "prediction_completion_sha256": validated_replicate[
                "completion_sha256"
            ],
            "aggregate_prediction_completion_sha256": (
                aggregate_prediction_completion_sha256
            ),
            "privileged_manifest_sha256": _sha256_file(
                privileged_manifest_path
            ),
            "report_sha256": _sha256_file(report_path),
            "git_commit": context["producer_git_commit"],
        }
        if load_exact_completion(completion_path, completion_payload) is None:
            if completion_path.exists():
                raise ValueError(
                    f"existing 20-Q replicate {replicate_index} completion differs"
                )
            write_completion(completion_path, completion_payload)
        replicate_outputs.append(
            {
                "replicate_index": replicate_index,
                "replicate_id": replicate_id,
                "replicate_seed": replicate_seed,
                "transform_sha256": transform_sha256,
                "privileged_manifest_path": str(privileged_manifest_path),
                "privileged_manifest_sha256": _sha256_file(
                    privileged_manifest_path
                ),
                "report_path": str(report_path),
                "report_sha256": _sha256_file(report_path),
                "completion_path": str(completion_path),
                "completion_sha256": _sha256_file(completion_path),
            }
        )

    _require_exact_directory_entries(
        sidecar_family_root,
        [
            sidecar_family_root / f"replicate_{index:03d}"
            for index in range(FORMAL_REPLICATE_COUNT)
        ],
        entry_kind="directory",
        label="20-Q privileged family",
    )
    aggregate_privileged_manifest_path = (
        run_root / "manifests" / "matched_random_20q_privileged_manifest.json"
    )
    _require_contract_path(
        aggregate_privileged_manifest_path,
        aggregate_privileged_manifest_path,
        run_root=run_root,
        label="20-Q aggregate privileged manifest",
    )
    aggregate_privileged_manifest = {
        "schema": _MATCHED_RANDOM_20Q_AGGREGATE_PRIVILEGED_SCHEMA,
        "replicate_count": FORMAL_REPLICATE_COUNT,
        "scene_count": 10,
        "sidecar_count": 200,
        "plan_sha256": plan_sha256,
        "plan_digest": plan["plan_digest"],
        "prediction_manifest_sha256": aggregate_prediction_manifest_sha256,
        "prediction_completion_sha256": (
            aggregate_prediction_completion_sha256
        ),
        "vrfm_privileged_manifest_sha256": privileged[
            "vrfm_privileged_manifest_sha256"
        ],
        "replicates": replicate_outputs,
    }
    _write_exact_json(
        aggregate_privileged_manifest_path,
        aggregate_privileged_manifest,
        label="20-Q aggregate privileged manifest",
    )
    summary = summarize_matched_random_ensemble(
        identity_overlap_best,
        random_overlap_best,
        replicate_indices=list(range(FORMAL_REPLICATE_COUNT)),
        expected_replicates=list(plan["replicates"]),
        actual_replicate_bindings=[
            {
                "replicate_index": record["replicate_index"],
                "replicate_id": record["replicate_id"],
                "replicate_seed": record["replicate_seed"],
                "transform_sha256": record["transform_sha256"],
            }
            for record in replicate_outputs
        ],
        observed_pilot_score=observed_pilot_score,
    )
    report_path = run_root / "reports" / "matched_random_vs_vrfm_20q_report.json"
    _require_contract_path(
        report_path,
        report_path,
        run_root=run_root,
        label="20-Q aggregate report",
    )
    report_payload = {
        "schema": _MATCHED_RANDOM_20Q_REPORT_SCHEMA,
        "replicate_count": FORMAL_REPLICATE_COUNT,
        "scene_count": 10,
        "inference_unit": "structured_null_transform",
        "randomization_unit": "structured_null_transform",
        "formal_training_attribution": False,
        "oracle_upper_bound": True,
        "pilot_reference": pilot_reference,
        "plan_sha256": plan_sha256,
        "plan_digest": plan["plan_digest"],
        "prediction_manifest_sha256": aggregate_prediction_manifest_sha256,
        "prediction_completion_sha256": (
            aggregate_prediction_completion_sha256
        ),
        "privileged_manifest_sha256": _sha256_file(
            aggregate_privileged_manifest_path
        ),
        "statistics": summary,
        "replicates": [
            {
                "replicate_index": record["replicate_index"],
                "replicate_id": record["replicate_id"],
                "replicate_seed": record["replicate_seed"],
                "transform_sha256": record["transform_sha256"],
                "report_path": record["report_path"],
                "report_sha256": record["report_sha256"],
            }
            for record in replicate_outputs
        ],
    }
    _write_exact_json(report_path, report_payload, label="20-Q aggregate report")
    if _require_clean_git_checkout(ROOT) != context["producer_git_commit"]:
        raise ValueError("20-Q producer commit changed during finalization")
    completion_path = run_root / "matched_random_ablation_20q_verified_completion.json"
    _require_contract_path(
        completion_path,
        completion_path,
        run_root=run_root,
        label="20-Q aggregate completion",
    )
    completion_payload = {
        "schema": _MATCHED_RANDOM_20Q_COMPLETION_SCHEMA,
        "replicate_count": FORMAL_REPLICATE_COUNT,
        "scene_count": 10,
        "prediction_artifact_count": 200,
        "privileged_sidecar_count": 200,
        "plan_sha256": plan_sha256,
        "plan_digest": plan["plan_digest"],
        "prediction_manifest_sha256": aggregate_prediction_manifest_sha256,
        "prediction_completion_sha256": (
            aggregate_prediction_completion_sha256
        ),
        "privileged_manifest_sha256": _sha256_file(
            aggregate_privileged_manifest_path
        ),
        "report_sha256": _sha256_file(report_path),
        "phase1_completion_sha256": _sha256_file(
            Path(privileged["phase1_completion_path"])
        ),
        "vrfm_completion_sha256": _sha256_file(
            Path(privileged["vrfm_completion_path"])
        ),
        "identity_score": summary["identity_score"],
        "pilot_reference_label": pilot_reference["label"],
        "pilot_reference_score": pilot_reference["score"],
        "pilot_reference_included_in_formal_null": pilot_reference[
            "included_in_formal_null"
        ],
        "pilot_reference_included_in_p_values": pilot_reference[
            "included_in_p_values"
        ],
        "pilot_reference_transform_sha256": pilot_reference[
            "transform_sha256"
        ],
        "pilot_reference_prediction_manifest_sha256": pilot_reference[
            "prediction_manifest_sha256"
        ],
        "pilot_reference_privileged_manifest_sha256": pilot_reference[
            "privileged_manifest_sha256"
        ],
        "pilot_reference_report_sha256": pilot_reference["report_sha256"],
        "pilot_reference_completion_sha256": pilot_reference[
            "completion_sha256"
        ],
        "p_identity_unusually_good": summary["p_identity_unusually_good"],
        "p_identity_unusually_bad": summary["p_identity_unusually_bad"],
        "p_identity_two_sided": summary["p_identity_two_sided"],
        "identity_rank_descending_best_tie": summary[
            "identity_rank_descending_best_tie"
        ],
        "identity_rank_descending_worst_tie": summary[
            "identity_rank_descending_worst_tie"
        ],
        "git_commit": context["producer_git_commit"],
    }
    if load_exact_completion(completion_path, completion_payload) is None:
        if completion_path.exists():
            raise ValueError("existing 20-Q aggregate completion differs")
        write_completion(completion_path, completion_payload)
    return run_root


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
    if args.stage == "matched-random-predict":
        _matched_random_replicate_index(args.matched_random_replicate_index)
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
    if args.stage == "matched-random-ablation":
        return run_matched_random_ablation(args)
    if args.stage == "matched-random-plan":
        return create_matched_random_ensemble_plan(args)
    if args.stage == "matched-random-predict":
        return run_matched_random_prediction_replicate(args)
    if args.stage == "matched-random-finalize":
        return finalize_matched_random_ensemble(args)
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
            "matched-random-ablation",
            "matched-random-plan",
            "matched-random-predict",
            "matched-random-finalize",
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
    parser.add_argument("--matched-random-batch-size", type=int, default=8)
    parser.add_argument("--matched-random-seed", type=int, default=20260827)
    parser.add_argument(
        "--matched-random-master-seed", type=int, default=2026082701
    )
    parser.add_argument("--matched-random-replicate-index", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = run_stage(args)
    print(f"[vrfm] stage={args.stage} complete run_root={root}", flush=True)


if __name__ == "__main__":
    main()
