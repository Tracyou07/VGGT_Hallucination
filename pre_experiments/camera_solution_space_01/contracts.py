"""Canonical serialization and digest helpers for fail-closed artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised when a value cannot satisfy an artifact contract."""


def _validate_json_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{path}: non-finite floating-point value is not allowed")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path}: mapping keys must be strings")
            _validate_json_value(child, f"{path}.{key}")
        return
    if isinstance(value, Mapping):
        raise ContractError(f"{path}: JSON objects must use native dict, got {type(value).__name__}")
    raise ContractError(f"{path}: unsupported canonical JSON value {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a JSON-native value."""
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_hex(data: bytes | bytearray | memoryview) -> str:
    """Return a lowercase SHA-256 digest for bytes-like data."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("SHA-256 input must be bytes-like")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an existing file without retaining its contents in memory."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ContractError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_schema(document: Mapping[str, Any], expected_schema: str) -> None:
    """Require an artifact to declare exactly the expected schema version."""
    if not isinstance(document, Mapping):
        raise ContractError("document must be a mapping with a schema field")
    if not isinstance(expected_schema, str) or not expected_schema:
        raise ContractError("expected schema must be a non-empty string")
    actual_schema = document.get("schema")
    if not isinstance(actual_schema, str) or actual_schema != expected_schema:
        raise ContractError(f"schema mismatch: expected {expected_schema!r}, got {actual_schema!r}")


def canonical_json_sha256(value: Any, expected_schema: str | None = None) -> str:
    """Validate an optional schema and hash canonical JSON bytes."""
    if expected_schema is not None:
        validate_schema(value, expected_schema)
    return sha256_hex(canonical_json_bytes(value))
