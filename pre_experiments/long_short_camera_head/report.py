from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .evaluate import run_long_only_inference


REPORT_SCHEMA = "long_short_camera_head.report.v1"


def classify(
    rows: Iterable[dict[str, object]],
    *,
    inference_leakage_audit: bool,
) -> dict[str, object]:
    """Apply the frozen decision gates to locked-replay scene metrics."""
    materialized = [dict(row) for row in rows]
    required = {
        "scene",
        "baseline_rms",
        "gt_only_rms",
        "long_short_rms",
        "baseline_rotation_deg",
        "long_short_rotation_deg",
    }
    structurally_valid = bool(materialized) and all(required <= set(row) for row in materialized)
    values: list[float] = []
    if structurally_valid:
        try:
            values = [
                float(row[name])
                for row in materialized
                for name in required - {"scene"}
            ]
        except (TypeError, ValueError):
            structurally_valid = False
    finite = structurally_valid and all(math.isfinite(value) for value in values)
    if not finite or not inference_leakage_audit:
        return {
            "schema": REPORT_SCHEMA,
            "classification": "INVALID",
            "failed_gates": [
                name
                for name, passed in (
                    ("finite_complete_metrics", finite),
                    ("inference_leakage_audit", inference_leakage_audit),
                )
                if not passed
            ],
            "scenes": materialized,
        }

    baseline = np.asarray([float(row["baseline_rms"]) for row in materialized])
    gt_only = np.asarray([float(row["gt_only_rms"]) for row in materialized])
    long_short = np.asarray([float(row["long_short_rms"]) for row in materialized])
    if np.any(baseline <= 0.0):
        return {
            "schema": REPORT_SCHEMA,
            "classification": "INVALID",
            "failed_gates": ["positive_baseline"],
            "scenes": materialized,
        }
    utilities = (baseline - long_short) / baseline
    rotation_delta = np.mean(
        [
            float(row["long_short_rotation_deg"])
            - float(row["baseline_rotation_deg"])
            for row in materialized
        ]
    )
    gates = {
        "positive_mean_utility": float(np.mean(utilities)) > 0.0,
        "per_scene_harm": bool(np.all(utilities >= -0.01)),
        "beats_gt_only": float(np.mean(long_short)) < float(np.mean(gt_only)),
        "rotation_guard": float(rotation_delta) <= 0.1,
        "inference_leakage_audit": True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if not gates["positive_mean_utility"]:
        classification = "NO_SOURCE_HEAD_SIGNAL"
    elif failed:
        classification = "HEAD_ONLY_INSUFFICIENT"
    else:
        classification = "PROMISING"
    return {
        "schema": REPORT_SCHEMA,
        "classification": classification,
        "failed_gates": failed,
        "mean_baseline_rms": float(np.mean(baseline)),
        "mean_gt_only_rms": float(np.mean(gt_only)),
        "mean_long_short_rms": float(np.mean(long_short)),
        "mean_long_short_utility": float(np.mean(utilities)),
        "mean_rotation_increase_deg": float(rotation_delta),
        "gates": gates,
        "scenes": materialized,
    }


def inference_signature_is_long_only() -> bool:
    names = set(inspect.signature(run_long_only_inference).parameters)
    forbidden = {
        "gt",
        "prepared_root",
        "short_tokens",
        "privileged",
        "privileged_path",
        "teacher",
    }
    return not bool(names & forbidden)


def _read_metrics(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid evaluation metrics: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != "long_short_camera_head.evaluation.v1":
        raise ValueError(f"evaluation schema mismatch: {path}")
    return payload


def _scene_roles(run_root: Path) -> dict[str, str]:
    path = Path(run_root) / "manifests" / "data_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("data manifest is required for role-aware reporting") from error
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("data manifest contains no scene roles")
    roles: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("data manifest record is malformed")
        scene = str(record.get("scene", ""))
        role = str(record.get("role", ""))
        if not scene or role not in {"train", "validation"} or scene in roles:
            raise ValueError("data manifest scene role is malformed")
        roles[scene] = role
    return roles


def write_report(run_root: Path) -> Path:
    run_root = Path(run_root)
    roles = _scene_roles(run_root)
    gt_files = {
        path.stem: path
        for path in (run_root / "evaluation" / "gt_only").glob("scene*.json")
    }
    long_files = {
        path.stem: path
        for path in (run_root / "evaluation" / "long_short").glob("scene*.json")
    }
    if (
        not gt_files
        or set(gt_files) != set(long_files)
        or set(gt_files) != set(roles)
    ):
        raise ValueError("matched evaluation files are incomplete")
    rows: list[dict[str, object]] = []
    for scene in sorted(gt_files):
        gt = _read_metrics(gt_files[scene])
        long_short = _read_metrics(long_files[scene])
        if gt["scene"] != scene or long_short["scene"] != scene:
            raise ValueError("evaluation file name and scene identity disagree")
        if not math.isclose(
            float(gt["baseline_rms"]),
            float(long_short["baseline_rms"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("matched variants use inconsistent baseline metrics")
        rows.append(
            {
                "scene": scene,
                "role": "locked_replay" if roles[scene] == "validation" else "train_diagnostic",
                "baseline_rms": float(long_short["baseline_rms"]),
                "gt_only_rms": float(gt["predicted_rms"]),
                "long_short_rms": float(long_short["predicted_rms"]),
                "baseline_rotation_deg": float(long_short["baseline_rotation_deg"]),
                "gt_only_rotation_deg": float(gt["predicted_rotation_deg"]),
                "long_short_rotation_deg": float(long_short["predicted_rotation_deg"]),
                "gt_only_checkpoint_sha256": str(gt["checkpoint_sha256"]),
                "long_short_checkpoint_sha256": str(long_short["checkpoint_sha256"]),
            }
        )
    locked_rows = [row for row in rows if row["role"] == "locked_replay"]
    report = classify(
        locked_rows,
        inference_leakage_audit=inference_signature_is_long_only(),
    )
    report["locked_replay_scenes"] = locked_rows
    report["scenes"] = rows
    report_path = run_root / "reports" / "result.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)

    markdown = [
        "# Long–Short Camera Head Result",
        "",
        f"Classification: **{report['classification']}**",
        "",
        "| Scene | Role | Baseline RMS | GT-only RMS | Long–short RMS |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['scene']} | {row['role']} | {row['baseline_rms']:.6f} | "
            f"{row['gt_only_rms']:.6f} | {row['long_short_rms']:.6f} |"
        )
    markdown_path = run_root / "reports" / "result.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report_path
