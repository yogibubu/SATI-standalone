"""Canonical reports and contract attachment for ORACLE robustness evidence."""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from matrix_chem import (
    FrozenPerceptionHandoff,
    OracleSonicContract,
    PerceptionHistory,
    PerceptionNoiseAudit,
    oracle_sonic_contract_to_dict,
    perception_contract_to_dict,
    frozen_perception_handoff_from_dict,
    perception_history_from_dict,
    perception_noise_audit_from_dict,
    validate_oracle_sonic_contract,
)

from .perception_policy import chemical_perception_policy_manifest


ORACLE_PERCEPTION_REPORT_SCHEMA = "matrix.oracle.perception_robustness_report.v1"


def attach_perception_robustness(
    contract: OracleSonicContract,
    audit: PerceptionNoiseAudit,
    *,
    history: PerceptionHistory | None = None,
    handoff: FrozenPerceptionHandoff | None = None,
) -> OracleSonicContract:
    """Return a new immutable contract; never mutate or reconstruct chemistry."""

    accepted_audit = audit
    if handoff is not None:
        if handoff.state_hash != audit.reference_state_hash:
            raise ValueError("handoff does not reference the audited state")
        accepted_audit = replace(
            audit,
            symmetry_decision=handoff.symmetry_decision,
            handoff_status="FROZEN_EXPLOITATION",
            history_fingerprint=(history.fingerprint if history is not None else ""),
        )
    elif history is not None:
        accepted_audit = replace(audit, history_fingerprint=history.fingerprint)
    policy_hash = chemical_perception_policy_manifest()["sha256"]
    output = replace(
        contract,
        robustness_audit=accepted_audit,
        perception_history=history,
        perception_handoff=handoff,
        chemical_policy_sha256=policy_hash,
    )
    validate_oracle_sonic_contract(output)
    return output


def perception_robustness_report_document(
    audit: PerceptionNoiseAudit,
    *,
    history: PerceptionHistory | None = None,
    handoff: FrozenPerceptionHandoff | None = None,
) -> dict:
    policy = chemical_perception_policy_manifest()
    payload = {
        "schema": ORACLE_PERCEPTION_REPORT_SCHEMA,
        "owner": "ORACLE",
        "audit": perception_contract_to_dict(audit),
        "history": None if history is None else perception_contract_to_dict(history),
        "handoff": None if handoff is None else perception_contract_to_dict(handoff),
        "chemical_policy": policy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def write_perception_robustness_report(
    path: Path,
    audit: PerceptionNoiseAudit,
    *,
    history: PerceptionHistory | None = None,
    handoff: FrozenPerceptionHandoff | None = None,
) -> Path:
    target = Path(path)
    document = perception_robustness_report_document(
        audit, history=history, handoff=handoff
    )
    target.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def read_perception_robustness_report(
    path: Path,
) -> tuple[PerceptionNoiseAudit, PerceptionHistory | None, FrozenPerceptionHandoff | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != ORACLE_PERCEPTION_REPORT_SCHEMA or payload.get("owner") != "ORACLE":
        raise ValueError("unsupported ORACLE perception robustness report")
    expected = str(payload.get("sha256", ""))
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    observed = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected != observed:
        raise ValueError("ORACLE perception robustness report fingerprint mismatch")
    audit = perception_noise_audit_from_dict(payload["audit"])
    history = (
        None
        if payload.get("history") is None
        else perception_history_from_dict(payload["history"])
    )
    handoff = (
        None
        if payload.get("handoff") is None
        else frozen_perception_handoff_from_dict(payload["handoff"])
    )
    return audit, history, handoff


def write_perception_decision_csv(path: Path, audit: PerceptionNoiseAudit) -> Path:
    target = Path(path)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "decision_id",
                "family",
                "accepted_class",
                "competing_class",
                "raw_score",
                "normalized_score",
                "signed_margin",
                "stability_fraction",
                "worst_case_perturbation",
                "fallback",
                "provider",
                "provider_version",
            ),
        )
        writer.writeheader()
        for item in audit.decisions:
            record = asdict(item)
            writer.writerow({name: record.get(name) for name in writer.fieldnames})
    return target


def perception_robustness_human_lines(audit: PerceptionNoiseAudit) -> tuple[str, ...]:
    model = audit.noise_settings.model
    return (
        f"Status: {audit.status}",
        f"Noise: {model.category}/{model.representation}",
        f"Samples: {audit.noise_settings.perturbation_count}",
        f"Strict group: {audit.strict_group}",
        f"Proposed group: {audit.proposed_group}",
        f"Symmetry decision: {audit.symmetry_decision}",
        f"Handoff: {audit.handoff_status}",
        f"Ordinary perception runtime (s): {audit.ordinary_runtime_seconds:.6g}",
        f"Ensemble audit runtime (s): {audit.ensemble_runtime_seconds:.6g}",
    )


def oracle_sonic_contract_sha256(contract: OracleSonicContract) -> str:
    encoded = json.dumps(
        oracle_sonic_contract_to_dict(contract), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ORACLE_PERCEPTION_REPORT_SCHEMA",
    "attach_perception_robustness",
    "oracle_sonic_contract_sha256",
    "perception_robustness_human_lines",
    "perception_robustness_report_document",
    "read_perception_robustness_report",
    "write_perception_decision_csv",
    "write_perception_robustness_report",
]
