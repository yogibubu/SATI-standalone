"""Central frozen-core, ECP, and core--valence policy for canonical QM work."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


QM_ELEMENT_POLICY_SCHEMA = "matrix.qm.element_policy.v1"
QM_ELEMENT_POLICY_ID = "matrix-qm-element-policy-v1"
QM_ELEMENT_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True)
class QMElementPolicy:
    payload: dict[str, Any]
    sha256: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)


@dataclass(frozen=True)
class AtomElementPolicy:
    atom_index: int
    atomic_number: int
    uses_ecp: bool
    core_label: str
    canonical_frozen_electrons: int | None
    applied_frozen_electrons: int
    cv_basis: str | None
    cv_basis_status: str
    intrinsic_cv: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenCoreDirective:
    representation: str
    spatial_orbital_count: int
    spatial_orbital_indices: tuple[int, ...]
    alpha_spatial_orbital_indices: tuple[int, ...]
    beta_spatial_orbital_indices: tuple[int, ...]
    backend_default_allowed: bool = False
    adapter_verification_required: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedElementPolicy:
    atoms: tuple[AtomElementPolicy, ...]
    frozen_core_required: bool
    total_applied_frozen_electrons: int
    frozen_core: FrozenCoreDirective
    any_ecp: bool
    core_valence_requested: bool
    core_valence_supported: bool
    core_valence_status: str
    unsupported_cv_atomic_numbers: tuple[int, ...]
    policy_id: str
    policy_version: str
    policy_sha256: str

    @property
    def cv_basis_by_atom(self) -> tuple[str | None, ...]:
        return tuple(atom.cv_basis for atom in self.atoms)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["atoms"] = [atom.to_dict() for atom in self.atoms]
        payload["frozen_core"] = self.frozen_core.to_dict()
        return payload


def load_qm_element_policy() -> QMElementPolicy:
    resource = resources.files("matrix_qm").joinpath("data/qm_element_policy.json")
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MATRIX QM element policy is unreadable") from exc
    validate_qm_element_policy(payload)
    return QMElementPolicy(
        payload=deepcopy(payload),
        sha256=hashlib.sha256(raw).hexdigest(),
        source="matrix_qm:data/qm_element_policy.json",
    )


def qm_element_policy_schema_path() -> Path:
    return Path(
        str(resources.files("matrix_qm").joinpath("schemas/qm-element-policy-v1.schema.json"))
    )


def resolve_element_policy(
    atomic_numbers: Sequence[int],
    *,
    ecp_mask: Sequence[bool] | None = None,
    frozen_core_required: bool,
    core_valence_requested: bool,
) -> ResolvedElementPolicy:
    """Resolve one molecular policy without consulting backend defaults."""

    numbers = tuple(int(value) for value in atomic_numbers)
    if not numbers:
        raise ValueError("element policy requires at least one atom")
    if any(value < 1 or value > 118 for value in numbers):
        raise ValueError("atomic numbers must lie between 1 and 118")
    if ecp_mask is None:
        ecp_flags = (False,) * len(numbers)
    else:
        ecp_flags = tuple(bool(value) for value in ecp_mask)
        if len(ecp_flags) != len(numbers):
            raise ValueError("ecp_mask must have one entry per atom")

    policy = load_qm_element_policy()
    atoms: list[AtomElementPolicy] = []
    unsupported_cv: set[int] = set()
    for atom_index, (atomic_number, uses_ecp) in enumerate(zip(numbers, ecp_flags)):
        if atomic_number > 36 and not uses_ecp:
            raise ValueError(
                "the canonical Karlsruhe element policy requires an approved ECP "
                f"for atomic number {atomic_number}"
            )
        frozen_range = _range_for_atomic_number(policy.payload["frozen_core_ranges"], atomic_number)
        if uses_ecp:
            canonical_frozen = None
            core_label = "ECP"
            applied_frozen = 0
        elif frozen_range is None:
            if frozen_core_required:
                raise ValueError(
                    "no approved all-electron frozen-core rule exists for "
                    f"atomic number {atomic_number}; an approved ECP is required"
                )
            canonical_frozen = None
            core_label = "outside_approved_all_electron_range"
            applied_frozen = 0
        else:
            canonical_frozen = int(frozen_range["frozen_electrons"])
            core_label = str(frozen_range["core_label"])
            applied_frozen = canonical_frozen if frozen_core_required else 0

        cv_range = _range_for_atomic_number(policy.payload["core_valence_ranges"], atomic_number)
        cv_status = (
            "ecp"
            if uses_ecp
            else ("outside_approved_range" if cv_range is None else str(cv_range["status"]))
        )
        cv_basis = None if uses_ecp or cv_range is None else cv_range.get("basis")
        intrinsic_cv = bool(cv_range and cv_range.get("intrinsic_cv"))
        if not uses_ecp and (cv_range is None or cv_status == "unsupported"):
            unsupported_cv.add(atomic_number)
        atoms.append(
            AtomElementPolicy(
                atom_index=atom_index,
                atomic_number=atomic_number,
                uses_ecp=uses_ecp,
                core_label=core_label,
                canonical_frozen_electrons=canonical_frozen,
                applied_frozen_electrons=applied_frozen,
                cv_basis=None if cv_basis is None else str(cv_basis),
                cv_basis_status=cv_status,
                intrinsic_cv=intrinsic_cv,
            )
        )

    total_frozen = sum(atom.applied_frozen_electrons for atom in atoms)
    if total_frozen % 2:
        raise RuntimeError("central frozen-core electron count is not spatially paired")
    frozen_count = total_frozen // 2
    frozen_indices = tuple(range(frozen_count))
    directive = FrozenCoreDirective(
        representation="lowest_energy_occupied_spatial_orbitals_zero_based",
        spatial_orbital_count=frozen_count,
        spatial_orbital_indices=frozen_indices,
        alpha_spatial_orbital_indices=frozen_indices,
        beta_spatial_orbital_indices=frozen_indices,
    )

    any_ecp = any(ecp_flags)
    if not core_valence_requested:
        cv_supported = False
        cv_status = "not_requested"
    elif any_ecp:
        cv_supported = False
        cv_status = "unsupported_any_ecp_disables_complete_molecular_cv"
    elif unsupported_cv:
        cv_supported = False
        cv_status = "unsupported_element_basis"
    else:
        cv_supported = True
        cv_status = "supported"

    return ResolvedElementPolicy(
        atoms=tuple(atoms),
        frozen_core_required=bool(frozen_core_required),
        total_applied_frozen_electrons=total_frozen,
        frozen_core=directive,
        any_ecp=any_ecp,
        core_valence_requested=bool(core_valence_requested),
        core_valence_supported=cv_supported,
        core_valence_status=cv_status,
        unsupported_cv_atomic_numbers=tuple(sorted(unsupported_cv)),
        policy_id=QM_ELEMENT_POLICY_ID,
        policy_version=QM_ELEMENT_POLICY_VERSION,
        policy_sha256=policy.sha256,
    )


def validate_qm_element_policy(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != QM_ELEMENT_POLICY_SCHEMA:
        raise RuntimeError("unsupported MATRIX QM element-policy schema")
    if (
        payload.get("policy_id") != QM_ELEMENT_POLICY_ID
        or payload.get("manifest_version") != QM_ELEMENT_POLICY_VERSION
        or payload.get("status") != "approved"
    ):
        raise RuntimeError("MATRIX QM element-policy identity changed")
    if payload.get("change_policy") != "new_manifest_version_and_explicit_approval":
        raise RuntimeError("MATRIX QM element-policy change contract changed")

    frozen = payload.get("frozen_core_ranges")
    if not isinstance(frozen, list):
        raise RuntimeError("MATRIX QM frozen-core ranges are missing")
    _validate_contiguous_ranges(frozen, expected_min=1, expected_max=36)
    observed = [
        (
            item["atomic_number_min"],
            item["atomic_number_max"],
            item["frozen_electrons"],
            item["core_label"],
        )
        for item in frozen
    ]
    if observed != [
        (1, 2, 0, "none"),
        (3, 10, 2, "[He]"),
        (11, 18, 10, "[Ne]"),
        (19, 36, 18, "[Ar]"),
    ]:
        raise RuntimeError("approved frozen-core partition changed")

    ecp = payload.get("ecp_contract", {})
    if ecp != {
        "karlsruhe_recommended_ecp_mandatory": True,
        "ecp_atom_additional_frozen_electrons": 0,
        "any_ecp_disables_complete_molecular_cv": True,
        "partial_cv_forbidden": True,
        "all_electron_beyond_atomic_number": 36,
        "beyond_range_without_ecp": "unsupported",
    }:
        raise RuntimeError("approved ECP and molecular CV contract changed")

    cv_ranges = payload.get("core_valence_ranges")
    if not isinstance(cv_ranges, list):
        raise RuntimeError("MATRIX QM core-valence ranges are missing")
    _validate_contiguous_ranges(cv_ranges, expected_min=1, expected_max=36)
    h_he = _range_for_atomic_number(cv_ranges, 1)
    if h_he is None or h_he.get("basis") != "cc-pVTZ" or h_he.get("status") != "companion":
        raise RuntimeError("H/He core-valence companion basis changed")
    for atomic_number in (19, 31, 36):
        record = _range_for_atomic_number(cv_ranges, atomic_number)
        if record is None or record.get("status") != "unsupported":
            raise RuntimeError("unsupported core-valence element map changed")

    translation = payload.get("backend_translation_contract", {})
    if translation != {
        "central_representation": "lowest_energy_occupied_spatial_orbitals_zero_based",
        "restricted_open_uses_same_spatial_core_for_alpha_and_beta": True,
        "backend_default_allowed": False,
        "adapter_must_verify_applied_orbitals": True,
    }:
        raise RuntimeError("central frozen-orbital translation contract changed")


def _validate_contiguous_ranges(
    records: Sequence[Mapping[str, Any]], *, expected_min: int, expected_max: int
) -> None:
    cursor = expected_min
    for record in records:
        lower = int(record.get("atomic_number_min", -1))
        upper = int(record.get("atomic_number_max", -1))
        if lower != cursor or upper < lower:
            raise RuntimeError("element-policy ranges overlap or contain a gap")
        cursor = upper + 1
    if cursor != expected_max + 1:
        raise RuntimeError("element-policy ranges do not cover the approved interval")


def _range_for_atomic_number(
    records: Sequence[Mapping[str, Any]], atomic_number: int
) -> Mapping[str, Any] | None:
    return next(
        (
            record
            for record in records
            if int(record["atomic_number_min"]) <= atomic_number <= int(record["atomic_number_max"])
        ),
        None,
    )


__all__ = [
    "QM_ELEMENT_POLICY_ID",
    "QM_ELEMENT_POLICY_SCHEMA",
    "QM_ELEMENT_POLICY_VERSION",
    "AtomElementPolicy",
    "FrozenCoreDirective",
    "QMElementPolicy",
    "ResolvedElementPolicy",
    "load_qm_element_policy",
    "qm_element_policy_schema_path",
    "resolve_element_policy",
    "validate_qm_element_policy",
]
