"""Machine-readable ARCHITECT/ZAFF build protocol metadata.

The protocol is additive: existing ``matrix.zaff.force_field.v1`` artifacts
remain valid, while new builds can attach this manifest to their metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


ZAFF_PROTOCOL_SCHEMA = "matrix.architect.zaff_force_field_protocol.v1"
ZAFF_PROTOCOL_NORMATIVE = True
_LEVEL_ORDER = {
    "ZAFF0": 0,
    "ZAFF1": 1,
    "ZAFF2": 2,
    "ZAFF3": 3,
    "ZAFF4": 4,
}


def available_zaff_levels(force_field: Any) -> tuple[ZaffFieldLevel, ...]:
    """Return the compiled ZAFF levels exposed by an ARCHITECT field."""

    # ZAFF0 is the rigid exploration projection.  A QM-Hessian-backed build
    # exposes both the fitted diagonal ZAFF1 field and the complete harmonic
    # ZAFF2 field.  Planned points alone never promote a field to ZAFF3.
    levels = [ZaffFieldLevel.ZERO]
    if (
        force_field.metadata.get("source_xyzin")
        and getattr(force_field,"hessian_hartree_per_bohr2",None) is not None
    ):
        levels.extend((ZaffFieldLevel.ONE, ZaffFieldLevel.TWO))
    model = getattr(force_field, "anharmonic_model", None)
    if str(getattr(model, "status", "")).upper() == "COMPILED":
        levels.append(ZaffFieldLevel.THREE)
    if getattr(force_field,"native_zaff4_runtime",False):
        certificate=force_field.metadata.get("zaff4_certificate",{})
        required=("parent_zaff3_qualified","local_derivatives_passed","reference_jet_preserved","independent_holdouts_passed","asymptotic_controls_passed","rank_support_verified")
        if certificate.get("status")=="qualified" and all(certificate.get(k) is True for k in required):
            levels=list(ZaffFieldLevel)
    return tuple(levels)


def select_zaff_level(
    force_field: Any, requested: ZaffFieldLevel | str | None = None
) -> ZaffFieldLevel:
    """Select a requested level or the highest level actually available."""

    available = available_zaff_levels(force_field)
    if requested is None or str(requested).strip() == "":
        return available[-1]
    value = requested.value if isinstance(requested, ZaffFieldLevel) else str(requested).strip()
    aliases = {
        "Z0": "ZAFF0", "FAST": "ZAFF0", "ZAFF-FAST": "ZAFF0",
        "Z1": "ZAFF1", "Z2": "ZAFF2", "Z3": "ZAFF3", "Z4": "ZAFF4",
    }
    value = aliases.get(value.upper(), value)
    try:
        selected = ZaffFieldLevel(value)
    except ValueError as exc:
        raise ValueError(f"unknown ZAFF level: {requested}") from exc
    if selected not in available:
        raise ValueError(
            f"requested {selected.value} is unavailable; available: {[x.value for x in available]}"
        )
    return selected


class GeometrySource(str, Enum):
    LCB26 = "LCB26"
    LCB25 = "LCB25"
    L1 = "L1"
    PL1 = "PL1"


class ZaffFieldLevel(str, Enum):
    ZERO = "ZAFF0"
    FAST = "ZAFF0"  # compatibility alias for the former public name
    ONE = "ZAFF1"
    TWO = "ZAFF2"
    THREE = "ZAFF3"
    FOUR = "ZAFF4"


@dataclass(frozen=True)
class ZaffProtocolManifest:
    """Validated provenance summary for one ARCHITECT field build."""

    molecule_id: str
    geometry_source: GeometrySource
    geometry_hash: str
    synthon_hash: str
    cm5_provenance: Mapping[str, Any]
    mayer_provenance: Mapping[str, Any]
    available_levels: tuple[ZaffFieldLevel, ...]
    skipped_stages: tuple[str, ...] = ()
    qm_geometry_available: bool = False
    qm_hessian_available: bool = False
    anharmonic_available: bool = False
    targeted_qm_calculations: tuple[Mapping[str, Any], ...] = ()
    schema: str = ZAFF_PROTOCOL_SCHEMA
    primitive_contract_hash: str = ""
    sonic_contract_hash: str = ""
    ring_contract_hash: str = ""
    acceptance_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != ZAFF_PROTOCOL_SCHEMA:
            raise ValueError(f"unsupported ZAFF protocol schema: {self.schema}")
        if (
            not self.molecule_id.strip()
            or not self.geometry_hash.strip()
            or not self.synthon_hash.strip()
        ):
            raise ValueError("ZAFF protocol requires molecule, geometry and synthon identifiers")
        diagnostics = dict(self.acceptance_diagnostics or {})
        if not self.primitive_contract_hash or not self.sonic_contract_hash:
            raise ValueError(
                "new ZAFF protocol manifests require primitive and SONIC contract hashes"
            )
        required_diagnostics = {
            "charge_conservation",
            "topology_synthon_identity",
            "derivative_checks",
        }
        if not required_diagnostics.issubset(diagnostics):
            raise ValueError("ZAFF acceptance diagnostics are incomplete")
        if not self.cm5_provenance or not self.mayer_provenance:
            raise ValueError("ZAFF protocol requires CM5 and Mayer provenance")
        if not str(self.mayer_provenance.get("total_source", "")).strip():
            raise ValueError("Mayer provenance requires an explicit total_source")
        if not str(self.mayer_provenance.get("component_source", "")).strip():
            raise ValueError("Mayer provenance requires an explicit component_source")
        levels = tuple(self.available_levels)
        if not levels or levels[0] is not ZaffFieldLevel.ZERO:
            raise ValueError("ZAFF0 must be the first available field level")
        level_order = tuple(_LEVEL_ORDER[level.value] for level in levels)
        if len(set(level_order)) != len(level_order) or level_order != tuple(sorted(level_order)):
            raise ValueError("ZAFF field levels must be unique and monotonically ordered")
        if (ZaffFieldLevel.ONE in levels or ZaffFieldLevel.TWO in levels) and not (
            self.qm_geometry_available or self.qm_hessian_available
        ):
            raise ValueError("ZAFF1 and ZAFF2 require QM geometry or Hessian provenance")
        if ZaffFieldLevel.TWO in levels and ZaffFieldLevel.ONE not in levels:
            raise ValueError("ZAFF2 requires ZAFF1")
        if ZaffFieldLevel.THREE in levels and not self.anharmonic_available:
            raise ValueError("ZAFF3 requires anharmonic provenance")
        if ZaffFieldLevel.THREE in levels and ZaffFieldLevel.TWO not in levels:
            raise ValueError("ZAFF3 requires ZAFF2")
        if ZaffFieldLevel.FOUR in levels:
            if ZaffFieldLevel.THREE not in levels:raise ValueError("ZAFF4 requires ZAFF3")
            required={"rank_support_verified","four_primitive_native_support","reference_jet_preserved","independent_holdouts_passed","asymptotic_controls_passed"}
            if not all(diagnostics.get(k) is True for k in required):raise ValueError("ZAFF4 promotion requires native support, invariant reference jet and independent validation")
        for calculation in self.targeted_qm_calculations:
            mode = str(calculation.get("geometry_mode", "")).upper()
            terms = calculation.get("terms", ())
            if mode not in {"L0_RIGID", "L1_RIGID", "RELAXED"}:
                raise ValueError("targeted QM geometry_mode must be L0_RIGID, L1_RIGID or RELAXED")
            if not terms:
                raise ValueError("targeted QM calculation must declare at least one fitted term")
            if not str(calculation.get("geometry_hash", "")).strip():
                raise ValueError("targeted QM calculation requires a frozen geometry_hash")
            if not calculation.get("baseline_subtraction"):
                raise ValueError("targeted QM calculation requires baseline_subtraction metadata")
            if "fixed_terms" not in calculation:
                raise ValueError("targeted QM calculation requires fixed_terms metadata")

    def to_dict(self) -> dict[str, Any]:
        parent = {
            ZaffFieldLevel.ZERO: None,
            ZaffFieldLevel.ONE: ZaffFieldLevel.ZERO,
            ZaffFieldLevel.TWO: ZaffFieldLevel.ONE,
            ZaffFieldLevel.THREE: ZaffFieldLevel.TWO,
            ZaffFieldLevel.FOUR: ZaffFieldLevel.THREE,
        }
        return {
            "schema": self.schema,
            "molecule_id": self.molecule_id,
            "geometry_source": self.geometry_source.value,
            "geometry_hash": self.geometry_hash,
            "synthon_hash": self.synthon_hash,
            "cm5_provenance": dict(self.cm5_provenance),
            "mayer_provenance": dict(self.mayer_provenance),
            "available_levels": [level.value for level in self.available_levels],
            "active_level": self.available_levels[-1].value,
            "level_lineage": [
                {
                    "level": level.value,
                    "extends": None if parent[level] is None else parent[level].value,
                }
                for level in self.available_levels
            ],
            "skipped_stages": list(self.skipped_stages),
            "qm_geometry_available": self.qm_geometry_available,
            "qm_hessian_available": self.qm_hessian_available,
            "anharmonic_available": self.anharmonic_available,
            "targeted_qm_calculations": [dict(item) for item in self.targeted_qm_calculations],
            "primitive_contract_hash": self.primitive_contract_hash,
            "sonic_contract_hash": self.sonic_contract_hash,
            "ring_contract_hash": self.ring_contract_hash,
            "acceptance_diagnostics": dict(self.acceptance_diagnostics or {}),
        }


__all__ = [
    "available_zaff_levels",
    "GeometrySource",
    "ZAFF_PROTOCOL_NORMATIVE",
    "ZAFF_PROTOCOL_SCHEMA",
    "ZaffFieldLevel",
    "select_zaff_level",
    "ZaffProtocolManifest",
]
