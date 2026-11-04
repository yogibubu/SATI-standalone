"""Canonical ORACLE classification of an initial Cartesian geometry.

The classification is deliberately about provenance and structural validity,
not about the electronic method that will consume the geometry.  This keeps
the decision identical for every LINK backend and leaves optimization policy
to the versioned MATRIX initialization protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from matrix_chem import MolecularGeometry, build_topology_objects
from matrix_chem.topology.elements import atomic_number


INITIAL_GEOMETRY_QUALITY_SCHEMA = "matrix.oracle.initial_geometry_quality.v1"

_GENERATED_SOURCE_KINDS = frozenset(
    {
        "smiles",
        "cxsmiles",
        "inchi",
        "switch_smiles",
        "generated",
    }
)
_CARTESIAN_SOURCE_KINDS = frozenset(
    {
        "xyz",
        "xyzin",
        "enriched_xyz",
        "geometry",
        "mol",
        "sdf",
        "mol2",
        "qm_geometry",
    }
)
_OVERRIDES = frozenset({"auto", "good", "requires_preoptimization"})


@dataclass(frozen=True)
class InitialGeometryQuality:
    """Auditable, backend-independent initial-geometry decision."""

    status: str
    source_kind: str
    override: str
    reasons: tuple[str, ...]
    atom_count: int
    topology_status: str
    schema: str = INITIAL_GEOMETRY_QUALITY_SCHEMA
    owner: str = "ORACLE"

    def __post_init__(self) -> None:
        if self.schema != INITIAL_GEOMETRY_QUALITY_SCHEMA:
            raise ValueError(f"unsupported initial-geometry schema: {self.schema}")
        if self.owner != "ORACLE":
            raise ValueError("initial-geometry quality must be owned by ORACLE")
        if self.status not in {"GOOD_UNCHANGED", "PREOPTIMIZE", "INVALID"}:
            raise ValueError(f"unsupported initial-geometry status: {self.status}")
        if self.override not in _OVERRIDES:
            raise ValueError(f"unsupported initial-geometry override: {self.override}")
        if self.atom_count <= 0:
            raise ValueError("initial-geometry quality requires at least one atom")
        if not self.reasons:
            raise ValueError("initial-geometry quality requires an explicit reason")

    @property
    def good(self) -> bool:
        return self.status == "GOOD_UNCHANGED"

    @property
    def requires_preoptimization(self) -> bool:
        return self.status == "PREOPTIMIZE"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["good"] = self.good
        payload["requires_preoptimization"] = self.requires_preoptimization
        payload["next_owner"] = (
            "KEYMAKER/LINK"
            if self.requires_preoptimization
            else "NONE"
            if self.status == "INVALID"
            else "LINK"
        )
        payload["required_handoff"] = (
            "KEYMAKER_CARTESIAN_BASIN_INITIALIZATION"
            if self.requires_preoptimization
            else "STOP_INVALID_GEOMETRY"
            if self.status == "INVALID"
            else "PRESERVE_GEOMETRY_AND_BUILD_INITIAL_HESSIAN"
        )
        return payload


def assess_initial_geometry_quality(
    geometry: MolecularGeometry,
    *,
    source_kind: str = "auto",
    preparation: Mapping[str, Any] | None = None,
    override: str = "auto",
) -> InitialGeometryQuality:
    """Classify whether the supplied Cartesian geometry must be preoptimized.

    ``auto`` preserves an explicitly supplied, structurally valid Cartesian
    geometry and preoptimizes a geometry generated from a line notation.  A
    failed ORACLE closure also requires preoptimization.  Explicit overrides
    change this policy decision but can never bypass basic validity or
    topology checks.
    """

    normalized_override = str(override).strip().lower().replace("-", "_")
    if normalized_override not in _OVERRIDES:
        raise ValueError(
            "initial-geometry override must be auto, good or requires_preoptimization"
        )
    normalized_source = _normalized_source_kind(geometry, source_kind)
    invalid_reasons = _basic_validity_reasons(geometry)
    topology_status = "NOT_REQUIRED_MONATOMIC"
    atom_count = len(geometry.atoms)
    if not invalid_reasons and atom_count > 1:
        try:
            numbers = tuple(int(atomic_number(symbol)) for symbol in geometry.atoms)
            build_topology_objects(geometry.coordinates_angstrom, numbers)
            topology_status = "ORACLE_PERCEPTION_PASS"
        except (TypeError, ValueError) as exc:
            invalid_reasons.append(f"ORACLE_TOPOLOGY_REJECTED:{exc}")
            topology_status = "ORACLE_PERCEPTION_FAIL"
    if invalid_reasons:
        return InitialGeometryQuality(
            status="INVALID",
            source_kind=normalized_source,
            override=normalized_override,
            reasons=tuple(invalid_reasons),
            atom_count=atom_count,
            topology_status=topology_status,
        )

    if normalized_override == "good":
        status = "GOOD_UNCHANGED"
        reasons = ("EXPLICIT_GOOD_GEOMETRY_OVERRIDE",)
    elif normalized_override == "requires_preoptimization":
        status = "PREOPTIMIZE"
        reasons = ("EXPLICIT_PREOPTIMIZATION_OVERRIDE",)
    else:
        policy_reasons: list[str] = []
        if normalized_source in _GENERATED_SOURCE_KINDS:
            policy_reasons.append("GENERATED_FROM_LINE_NOTATION")
        elif normalized_source not in _CARTESIAN_SOURCE_KINDS:
            policy_reasons.append("UNVERIFIED_GEOMETRY_PROVENANCE")
        if preparation is not None and preparation.get("closure_converged") is False:
            policy_reasons.append("ORACLE_INTERNAL_CLOSURE_NOT_CONVERGED")
        if policy_reasons:
            status = "PREOPTIMIZE"
            reasons = tuple(policy_reasons)
        else:
            status = "GOOD_UNCHANGED"
            reasons = ("VALID_EXPLICIT_CARTESIAN_GEOMETRY",)
    return InitialGeometryQuality(
        status=status,
        source_kind=normalized_source,
        override=normalized_override,
        reasons=reasons,
        atom_count=atom_count,
        topology_status=topology_status,
    )


def _normalized_source_kind(geometry: MolecularGeometry, source_kind: str) -> str:
    value = str(source_kind).strip().lower().replace("-", "_")
    if value and value != "auto":
        return value
    source_format = str(geometry.source_format).strip().lower().replace("-", "_")
    if source_format and source_format != "unknown":
        return source_format
    if geometry.source_path is not None:
        suffix = Path(geometry.source_path).suffix.lower()
        return {
            ".xyz": "xyz",
            ".xyzin": "xyzin",
            ".smi": "smiles",
            ".smiles": "smiles",
            ".mol": "mol",
            ".sdf": "sdf",
            ".mol2": "mol2",
        }.get(suffix, "unknown")
    return "unknown"


def _basic_validity_reasons(geometry: MolecularGeometry) -> list[str]:
    reasons: list[str] = []
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    atom_count = len(geometry.atoms)
    if atom_count <= 0:
        reasons.append("NO_ATOMS")
        return reasons
    if coordinates.shape != (atom_count, 3) or not np.all(np.isfinite(coordinates)):
        reasons.append("INVALID_CARTESIAN_COORDINATES")
        return reasons
    numbers = tuple(atomic_number(symbol) for symbol in geometry.atoms)
    if any(number is None or int(number) <= 0 for number in numbers):
        reasons.append("UNKNOWN_ELEMENT")
    if atom_count > 1:
        from matrix_chem.spatial_regions import SpatialRegions

        regions = SpatialRegions.build(coordinates, cell_size=0.1)
        if next(regions.candidate_pairs(1.0e-8), None) is not None:
            reasons.append("COINCIDENT_ATOMS")
    return reasons


__all__ = [
    "INITIAL_GEOMETRY_QUALITY_SCHEMA",
    "InitialGeometryQuality",
    "assess_initial_geometry_quality",
]
