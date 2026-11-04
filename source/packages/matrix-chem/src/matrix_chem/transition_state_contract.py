"""Frozen ORACLE prescription for a transition state from one geometry.

The transport classes live in :mod:`matrix_chem` because ORACLE and SMITH
both depend on this package.  Chemical classification and chart selection are
owned exclusively by ORACLE; SMITH validates this document and executes the
prescribed chart policy without interpreting the category identifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from matrix_core import read_sectioned_lines, replace_section, section_content


ORACLE_TS_GEOMETRY_CONTRACT_SCHEMA = "matrix.oracle.transition_state_geometry.v1"
ORACLE_TS_GEOMETRY_CONTRACT_SECTION = "ORACLE_TRANSITION_STATE_GEOMETRY"
ORACLE_TS_GEOMETRY_CONTRACT_OWNER = "ORACLE"
TS_SOURCE_SINGLE_GEOMETRY = "SINGLE_GEOMETRY"
TS_CHART_MINIMUM_LIKE = "MINIMUM_LIKE"
TS_CHART_REACTIVE_PSEUDOBOND = "REACTIVE_PSEUDOBOND"
TS_CHART_REACTIVE_DISTANCE = "REACTIVE_DISTANCE"
TS_ENDPOINT_ROUTE_STATUS = "SINGLE_GEOMETRY_ONLY"


def transition_state_descriptor(
    contract: "OracleTransitionStateGeometryContract", name: str, default: str = ""
) -> str:
    """Return one ORACLE-owned execution descriptor from a TS contract."""

    return dict(contract.descriptors).get(str(name), str(default))


class OracleTransitionStateContractError(ValueError):
    """Raised when an ORACLE transition-state prescription is inconsistent."""


@dataclass(frozen=True)
class TransitionStateKernelEdge:
    atoms: tuple[int, int]
    role: str
    kind: str
    priority: int
    provenance: str


@dataclass(frozen=True)
class TransitionStatePseudobond:
    atoms: tuple[int, int]
    kind: str
    priority: int
    mandatory: bool
    provenance: str


@dataclass(frozen=True)
class OracleTransitionStateGeometryContract:
    schema: str
    owner: str
    source: str
    catalog_id: str
    catalog_version: str
    category_id: str
    chart_policy: str
    natoms: int
    topology_hash: str
    reaction_kernel: tuple[TransitionStateKernelEdge, ...]
    prescribed_pseudobonds: tuple[TransitionStatePseudobond, ...]
    descriptors: tuple[tuple[str, str], ...]
    endpoints_route_status: str
    provenance: str


def validate_oracle_transition_state_geometry_contract(
    contract: OracleTransitionStateGeometryContract,
) -> None:
    """Validate transport invariants without interpreting chemical categories."""

    if contract.schema != ORACLE_TS_GEOMETRY_CONTRACT_SCHEMA:
        raise OracleTransitionStateContractError(
            f"unsupported transition-state contract schema: {contract.schema}"
        )
    if contract.owner != ORACLE_TS_GEOMETRY_CONTRACT_OWNER:
        raise OracleTransitionStateContractError(
            "ORACLE must be the sole transition-state contract owner"
        )
    if contract.source != TS_SOURCE_SINGLE_GEOMETRY:
        raise OracleTransitionStateContractError(
            "only the explicitly implemented SINGLE_GEOMETRY TS source is supported"
        )
    if not all(
        (
            contract.catalog_id,
            contract.catalog_version,
            contract.category_id,
            contract.topology_hash,
            contract.provenance,
        )
    ):
        raise OracleTransitionStateContractError(
            "transition-state contract identity and provenance must be complete"
        )
    if contract.natoms < 2:
        raise OracleTransitionStateContractError(
            "transition-state contract must contain at least two atoms"
        )
    if contract.endpoints_route_status != TS_ENDPOINT_ROUTE_STATUS:
        raise OracleTransitionStateContractError(
            "the two-endpoint transition-state route is not part of the single-geometry contract"
        )
    if contract.chart_policy not in {
        TS_CHART_MINIMUM_LIKE,
        TS_CHART_REACTIVE_DISTANCE,
        TS_CHART_REACTIVE_PSEUDOBOND,
    }:
        raise OracleTransitionStateContractError(
            f"unsupported transition-state chart policy: {contract.chart_policy}"
        )

    kernel_pairs: set[tuple[tuple[int, int], str]] = set()
    for edge in contract.reaction_kernel:
        pair = _validate_pair(edge.atoms, contract.natoms, "reaction-kernel edge")
        role = str(edge.role).strip().upper()
        if role not in {"BREAKING", "FORMING"}:
            raise OracleTransitionStateContractError(
                f"unsupported reaction-kernel role: {edge.role}"
            )
        if pair != edge.atoms or not edge.kind or not edge.provenance or edge.priority < 0:
            raise OracleTransitionStateContractError(
                f"invalid reaction-kernel edge: {edge}"
            )
        key = (pair, role)
        if key in kernel_pairs:
            raise OracleTransitionStateContractError(
                f"duplicate reaction-kernel edge: {pair} {role}"
            )
        kernel_pairs.add(key)

    pseudobond_pairs: set[tuple[int, int]] = set()
    for record in contract.prescribed_pseudobonds:
        pair = _validate_pair(record.atoms, contract.natoms, "prescribed pseudobond")
        if (
            pair != record.atoms
            or pair in pseudobond_pairs
            or not record.kind
            or not record.provenance
            or record.priority < 0
            or not record.mandatory
        ):
            raise OracleTransitionStateContractError(
                f"invalid or duplicate prescribed pseudobond: {record}"
            )
        pseudobond_pairs.add(pair)

    if contract.chart_policy == TS_CHART_MINIMUM_LIKE:
        if contract.reaction_kernel or contract.prescribed_pseudobonds:
            raise OracleTransitionStateContractError(
                "MINIMUM_LIKE policy cannot prescribe a reactive kernel or pseudobonds"
            )
    elif contract.chart_policy == TS_CHART_REACTIVE_PSEUDOBOND and (
        not contract.reaction_kernel or not contract.prescribed_pseudobonds
    ):
        raise OracleTransitionStateContractError(
            "REACTIVE_PSEUDOBOND policy requires a kernel and exact pseudobonds"
        )
    elif contract.chart_policy == TS_CHART_REACTIVE_DISTANCE and (
        not contract.reaction_kernel or contract.prescribed_pseudobonds
    ):
        raise OracleTransitionStateContractError(
            "REACTIVE_DISTANCE policy requires a kernel without pseudobonds"
        )
    if any(not name or not value for name, value in contract.descriptors):
        raise OracleTransitionStateContractError(
            "transition-state descriptors must be non-empty key/value pairs"
        )
    if len({name for name, _value in contract.descriptors}) != len(contract.descriptors):
        raise OracleTransitionStateContractError(
            "transition-state descriptor names must be unique"
        )
    required_descriptors = {
        "SEPARATE_EXOCYCLIC_TORSIONS",
        "DISTANCE_ONLY_KERNEL_EDGES",
    }
    descriptor_names = {name for name, _value in contract.descriptors}
    if not required_descriptors.issubset(descriptor_names):
        missing = ",".join(sorted(required_descriptors - descriptor_names))
        raise OracleTransitionStateContractError(
            f"transition-state contract is missing execution descriptors: {missing}"
        )


def transition_state_geometry_contract_to_dict(
    contract: OracleTransitionStateGeometryContract,
) -> dict[str, Any]:
    validate_oracle_transition_state_geometry_contract(contract)
    return asdict(contract)


def transition_state_geometry_contract_from_dict(
    payload: dict[str, Any],
) -> OracleTransitionStateGeometryContract:
    try:
        contract = OracleTransitionStateGeometryContract(
            schema=str(payload["schema"]),
            owner=str(payload["owner"]),
            source=str(payload["source"]),
            catalog_id=str(payload["catalog_id"]),
            catalog_version=str(payload["catalog_version"]),
            category_id=str(payload["category_id"]),
            chart_policy=str(payload["chart_policy"]),
            natoms=int(payload["natoms"]),
            topology_hash=str(payload["topology_hash"]),
            reaction_kernel=tuple(
                TransitionStateKernelEdge(
                    atoms=tuple(int(atom) for atom in item["atoms"]),
                    role=str(item["role"]),
                    kind=str(item["kind"]),
                    priority=int(item["priority"]),
                    provenance=str(item["provenance"]),
                )
                for item in payload["reaction_kernel"]
            ),
            prescribed_pseudobonds=tuple(
                TransitionStatePseudobond(
                    atoms=tuple(int(atom) for atom in item["atoms"]),
                    kind=str(item["kind"]),
                    priority=int(item["priority"]),
                    mandatory=bool(item["mandatory"]),
                    provenance=str(item["provenance"]),
                )
                for item in payload["prescribed_pseudobonds"]
            ),
            descriptors=tuple(
                (str(name), str(value)) for name, value in payload["descriptors"]
            ),
            endpoints_route_status=str(payload["endpoints_route_status"]),
            provenance=str(payload["provenance"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OracleTransitionStateContractError(
            "invalid serialized transition-state contract payload"
        ) from exc
    validate_oracle_transition_state_geometry_contract(contract)
    return contract


def transition_state_geometry_contract_section_lines(
    contract: OracleTransitionStateGeometryContract,
) -> list[str]:
    payload = json.dumps(
        transition_state_geometry_contract_to_dict(contract),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    chunks = [payload[index : index + 4096] for index in range(0, len(payload), 4096)]
    return [
        f"SCHEMA {contract.schema}",
        f"OWNER {ORACLE_TS_GEOMETRY_CONTRACT_OWNER}",
        "ENCODING CANONICAL_JSON_UTF8",
        f"PAYLOAD_SHA256 {digest}",
        "[PAYLOAD]",
        *chunks,
    ]


def write_oracle_transition_state_geometry_contract(
    path: Path,
    contract: OracleTransitionStateGeometryContract,
) -> None:
    replace_section(
        Path(path),
        ORACLE_TS_GEOMETRY_CONTRACT_SECTION,
        transition_state_geometry_contract_section_lines(contract),
    )


def read_oracle_transition_state_geometry_contract(
    path: Path,
) -> OracleTransitionStateGeometryContract:
    content = section_content(
        read_sectioned_lines(Path(path)), ORACLE_TS_GEOMETRY_CONTRACT_SECTION
    )
    if not content:
        raise OracleTransitionStateContractError(
            f"missing #{ORACLE_TS_GEOMETRY_CONTRACT_SECTION} section"
        )
    metadata: dict[str, str] = {}
    chunks: list[str] = []
    in_payload = False
    for raw in content:
        text = raw.strip()
        if text == "[PAYLOAD]":
            in_payload = True
        elif in_payload:
            chunks.append(text)
        elif text:
            fields = text.split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
    if metadata.get("SCHEMA") != ORACLE_TS_GEOMETRY_CONTRACT_SCHEMA:
        raise OracleTransitionStateContractError(
            "unsupported serialized transition-state contract schema"
        )
    if metadata.get("OWNER") != ORACLE_TS_GEOMETRY_CONTRACT_OWNER:
        raise OracleTransitionStateContractError(
            "serialized transition-state contract is not ORACLE-owned"
        )
    payload_text = "".join(chunks)
    digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    if digest != metadata.get("PAYLOAD_SHA256"):
        raise OracleTransitionStateContractError(
            "transition-state contract payload fingerprint mismatch"
        )
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise OracleTransitionStateContractError(
            "invalid transition-state contract JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise OracleTransitionStateContractError(
            "transition-state contract payload must be an object"
        )
    return transition_state_geometry_contract_from_dict(payload)


def _validate_pair(
    atoms: tuple[int, int], natoms: int, label: str
) -> tuple[int, int]:
    if len(atoms) != 2:
        raise OracleTransitionStateContractError(f"{label} must contain two atoms")
    left, right = sorted((int(atoms[0]), int(atoms[1])))
    if left < 1 or right > natoms or left == right:
        raise OracleTransitionStateContractError(f"invalid {label}: {atoms}")
    return left, right


__all__ = [
    "ORACLE_TS_GEOMETRY_CONTRACT_OWNER",
    "ORACLE_TS_GEOMETRY_CONTRACT_SCHEMA",
    "ORACLE_TS_GEOMETRY_CONTRACT_SECTION",
    "OracleTransitionStateContractError",
    "OracleTransitionStateGeometryContract",
    "TS_CHART_MINIMUM_LIKE",
    "TS_CHART_REACTIVE_DISTANCE",
    "TS_CHART_REACTIVE_PSEUDOBOND",
    "TS_ENDPOINT_ROUTE_STATUS",
    "TS_SOURCE_SINGLE_GEOMETRY",
    "TransitionStateKernelEdge",
    "TransitionStatePseudobond",
    "read_oracle_transition_state_geometry_contract",
    "transition_state_geometry_contract_from_dict",
    "transition_state_geometry_contract_section_lines",
    "transition_state_geometry_contract_to_dict",
    "validate_oracle_transition_state_geometry_contract",
    "transition_state_descriptor",
    "write_oracle_transition_state_geometry_contract",
]
