"""Deterministic compact Stage A report publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


REPORT_SCHEMA = "conditional_hierarchical_vrfm.stage_a_report.v1"


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _markdown(payload: Mapping[str, object]) -> str:
    failed = payload.get("failed_gates", [])
    scene_metrics = payload.get("scene_metrics", [])
    lines = [
        "# Privileged latent lifting — Stage A",
        "",
        f"Classification: `{payload['classification']}`",
        "",
        f"Git commit: `{payload['git_commit']}`",
        "",
        "Failed gates: " + (", ".join(str(value) for value in failed) if failed else "none"),
        "",
        f"Scene rows: {len(scene_metrics) if isinstance(scene_metrics, list) else 0}",
        "",
    ]
    return "\n".join(lines)


def _publish_exact(path: Path, content: str) -> None:
    if path.exists():
        try:
            current = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"cannot read existing report: {path}") from error
        if current != content:
            raise ValueError("existing report does not match independently recomputed content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_stage_a_report(
    run_root: Path, payload: Mapping[str, object]
) -> tuple[Path, Path]:
    required = {
        "schema", "git_commit", "classification", "failed_gates", "scene_metrics",
        "provenance",
    }
    if set(payload) != required or payload.get("schema") != REPORT_SCHEMA:
        raise ValueError("Stage A report must use the exact schema")
    if payload.get("classification") not in {"LATENT_TARGETS_READY", "LATENT_LIFT_FAILED"}:
        raise ValueError("Stage A report classification is invalid")
    commit = payload.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("Stage A report Git commit is invalid")
    root = Path(run_root)
    json_path = root / "reports" / "stage_a.json"
    markdown_path = root / "reports" / "stage_a.md"
    _publish_exact(json_path, _canonical_json(payload))
    _publish_exact(markdown_path, _markdown(payload))
    return json_path, markdown_path
