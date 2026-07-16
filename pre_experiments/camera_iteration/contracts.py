"""Run identity, metadata, and output routing contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


STUDY_TYPE = "method_pre_experiment"
STUDY_NAME = "camera_iteration"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_output_root(path: Path, repo_root: Path) -> Path:
    """Resolve an output root while protecting repository-owned namespaces."""
    root = repo_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_absolute() and not resolved.is_relative_to(root):
        raise ValueError("external output root must be an absolute path")
    allowed = (root / "results" / "pre_experiments" / STUDY_NAME).resolve()

    if any(part.lower() == "scannet_hallucination" for part in resolved.parts):
        raise ValueError("method pre-experiment output cannot use phenomenon paths")
    if resolved.is_relative_to(root) and not resolved.is_relative_to(allowed):
        raise ValueError(f"repository output must be under {allowed}")
    return resolved


def _validate_commit(git_commit: str) -> None:
    if _COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a 40-character lowercase hexadecimal id")


def _canonical_invocation(invocation: dict[str, object]) -> str:
    return json.dumps(invocation, sort_keys=True, separators=(",", ":"))


def make_run_id(git_commit: str, invocation: dict[str, object]) -> str:
    """Build a stable ID from the code revision and effective invocation."""
    _validate_commit(git_commit)
    digest = hashlib.sha256(_canonical_invocation(invocation).encode("utf-8")).hexdigest()[:12]
    return f"{git_commit[:7]}_{digest}"


def build_run_metadata(
    git_commit: str,
    invocation: dict[str, object],
) -> dict[str, object]:
    """Record the effective invocation without substituting protocol defaults."""
    _validate_commit(git_commit)
    scenes = invocation.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("invocation must contain at least one scene")

    normalized_invocation = json.loads(_canonical_invocation(invocation))
    return {
        "study_type": STUDY_TYPE,
        "study_name": STUDY_NAME,
        "git_commit": git_commit,
        "run_id": make_run_id(git_commit, invocation),
        "invocation": normalized_invocation,
        "primary_metric_policy": "prediction metrics use aligned values",
    }


def read_git_commit(repo_root: Path) -> str:
    """Read and validate the full commit for a repository checkout."""
    result = subprocess.run(
        ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    _validate_commit(commit)
    return commit


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON completely before replacing the destination path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
