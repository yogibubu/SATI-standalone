"""Compatibility, atomic handoff and verified-transfer primitives for MATRIX."""

from __future__ import annotations

from datetime import datetime, timezone
import hmac
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .atomic_io import atomic_json_write
from .manifest import sha256_file
from .unit_coherence import validate_unit_handoff


TOOL_COMPATIBILITY_SCHEMA = "matrix.core.tool_compatibility.v1"
ATOMIC_HANDOFF_SCHEMA = "matrix.core.atomic_handoff.v1"
VERIFIED_TRANSFER_SCHEMA = "matrix.core.verified_transfer.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^\s*[vV]?(\d+(?:\.\d+)*)", str(value))
    return tuple(int(item) for item in match.group(1).split(".")) if match else ()


def _version_at_least(version: str, minimum: str) -> bool:
    actual = _version_tuple(version)
    required = _version_tuple(minimum)
    if not required:
        return True
    width = max(len(actual), len(required))
    return actual + (0,) * (width - len(actual)) >= required + (0,) * (
        width - len(required)
    )


def validate_tool_compatibility(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    *,
    handoff_schema: str,
    required_capabilities: Sequence[str] = (),
) -> dict[str, Any]:
    """Negotiate one explicit producer-to-consumer handoff contract."""

    output_schemas = {str(value) for value in producer.get("output_schemas", [])}
    input_schemas = {str(value) for value in consumer.get("input_schemas", [])}
    producer_capabilities = {
        str(value) for value in producer.get("capabilities", [])
    }
    consumer_capabilities = {
        str(value) for value in consumer.get("capabilities", [])
    }
    required = {str(value) for value in required_capabilities}
    issues = []
    if handoff_schema not in output_schemas:
        issues.append(f"producer does not emit {handoff_schema}")
    if handoff_schema not in input_schemas:
        issues.append(f"consumer does not accept {handoff_schema}")
    missing_producer = sorted(required - producer_capabilities)
    missing_consumer = sorted(required - consumer_capabilities)
    if missing_producer:
        issues.append("producer lacks capabilities: " + ", ".join(missing_producer))
    if missing_consumer:
        issues.append("consumer lacks capabilities: " + ", ".join(missing_consumer))
    minimum = str(consumer.get("minimum_producer_version", "")).strip()
    version = str(producer.get("version", "")).strip()
    if minimum and not _version_at_least(version, minimum):
        issues.append(
            f"producer version {version or '<unknown>'} is older than {minimum}"
        )
    return {
        "schema": TOOL_COMPATIBILITY_SCHEMA,
        "compatible": not issues,
        "producer": {
            "tool": str(producer.get("tool", "")),
            "version": version,
        },
        "consumer": {
            "tool": str(consumer.get("tool", "")),
            "version": str(consumer.get("version", "")),
        },
        "handoff_schema": str(handoff_schema),
        "required_capabilities": sorted(required),
        "issues": issues,
        "checked_at_utc": _now(),
    }


def commit_validated_handoff(
    receipt_path: Path | str,
    *,
    result_path: Path | str,
    artifacts: Sequence[Mapping[str, Any]],
    producer: str,
    consumer: str,
    expected_dimensions: Mapping[str, str] | None = None,
    expected_conventions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the atomic commit marker only after every artifact validates."""

    result = Path(result_path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"handoff result is missing: {result}")
    if not producer or not consumer:
        raise ValueError("handoff producer and consumer are required")
    records = []
    normalized = []
    for raw in artifacts:
        item = dict(raw)
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"handoff artifact is missing: {path}")
        digest = sha256_file(path)
        declared = str(item.get("sha256", "")).strip()
        if declared and declared != digest:
            raise ValueError(f"handoff checksum mismatch: {path}")
        normalized.append({**item, "path": str(path), "sha256": digest})
        records.append(
            {
                "role": str(item.get("role", "artifact")),
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    coherence = validate_unit_handoff(
        normalized,
        expected_dimensions=expected_dimensions,
        expected_conventions=expected_conventions,
    )
    if not coherence["valid"]:
        raise ValueError(
            "handoff unit/convention validation failed: "
            + "; ".join(item["reason"] for item in coherence["issues"])
        )
    receipt = {
        "schema": ATOMIC_HANDOFF_SCHEMA,
        "status": "committed",
        "producer": producer,
        "consumer": consumer,
        "result_path": str(result),
        "result_sha256": sha256_file(result),
        "artifacts": records,
        "coherence": coherence,
        "committed_at_utc": _now(),
    }
    destination = Path(receipt_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(destination, receipt)
    return receipt


def rsync_download_command(
    *,
    host: str,
    remote_path: str,
    partial_path: Path | str,
) -> tuple[str, ...]:
    """Build a resumable rsync download without invoking a local shell."""

    normalized_host = str(host).strip()
    normalized_remote = str(remote_path).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", normalized_host):
        raise ValueError("invalid remote host")
    if (
        not normalized_remote.startswith("/")
        or "\n" in normalized_remote
        or "\r" in normalized_remote
    ):
        raise ValueError("remote path must be an absolute single-line path")
    partial = Path(partial_path).expanduser().resolve()
    partial.parent.mkdir(parents=True, exist_ok=True)
    return (
        "rsync",
        "--partial",
        "--append-verify",
        "--protect-args",
        f"{normalized_host}:{normalized_remote}",
        str(partial),
    )


def promote_verified_download(
    partial_path: Path | str,
    destination: Path | str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Atomically expose a downloaded file only after checksum verification."""

    partial = Path(partial_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    expected = str(expected_sha256).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
    if not partial.is_file():
        raise FileNotFoundError(partial)
    observed = sha256_file(partial)
    if not hmac.compare_digest(observed, expected):
        raise ValueError(
            f"remote transfer checksum mismatch: expected {expected}, found {observed}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(target)
    return {
        "schema": VERIFIED_TRANSFER_SCHEMA,
        "status": "complete",
        "path": str(target),
        "sha256": observed,
        "size_bytes": target.stat().st_size,
        "verified_at_utc": _now(),
    }


__all__ = [
    "ATOMIC_HANDOFF_SCHEMA",
    "TOOL_COMPATIBILITY_SCHEMA",
    "VERIFIED_TRANSFER_SCHEMA",
    "commit_validated_handoff",
    "promote_verified_download",
    "rsync_download_command",
    "validate_tool_compatibility",
]
