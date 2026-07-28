"""Shared metadata I/O contracts for method pre-experiments."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _validate_commit(git_commit: str) -> None:
    if _COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a 40-character lowercase hexadecimal id")


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
