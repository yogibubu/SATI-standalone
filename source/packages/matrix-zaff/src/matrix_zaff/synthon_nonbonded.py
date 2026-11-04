"""Discrete synthon typing for the definitive ZAFF non-bonded model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from .construction import (
    DirectionalExpPEInteraction,
    ResolvedDirectionalExpPEContacts,
    ResolvedExpPEPairTable,
)


SYNTHON_DESCRIPTOR_FIELDS = (
    "charge",
    "coordination",
    "electron_domains",
    "effective_radius",
    "polarizability",
    "covalency",
    "delocalization",
    "strain",
    "sigma_index",
    "pi_index",
    "pi_pi_index",
)
ZAFF_SYNTHON_CATALOG_SCHEMA = "matrix.zaff.synthon_catalog.v1"
ZAFF_SYNTHON_EXPPE_LIBRARY_SCHEMA = "matrix.zaff.synthon_exppe_library.v1"
ZAFF_DIRECTIONAL_EXPPE_LIBRARY_SCHEMA = "matrix.zaff.directional_exppe_library.v1"


@dataclass(frozen=True)
class SynthonTypeThresholds:
    """Maximum component differences defining one discrete atomic type."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if (
            len(values) != len(SYNTHON_DESCRIPTOR_FIELDS)
            or any(not isfinite(value) or value <= 0.0 for value in values)
        ):
            raise ValueError("synthon thresholds must be positive for every descriptor")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SynthonDescriptor:
    """Complete atomic synthon descriptor; charge is one component."""

    atomic_number: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if (
            int(self.atomic_number) < 1
            or len(values) != len(SYNTHON_DESCRIPTOR_FIELDS)
            or any(not isfinite(value) for value in values)
        ):
            raise ValueError("invalid complete atomic synthon descriptor")
        object.__setattr__(self, "atomic_number", int(self.atomic_number))
        object.__setattr__(self, "values", values)


def descriptors_from_atomic_synthons(
    synthons,
    *,
    intrinsic_cm5_charges_e: Sequence[float],
    intrinsic_coordination_numbers: Sequence[float] | None = None,
) -> tuple[SynthonDescriptor, ...]:
    """Extract fixed types from covalent fragment data and intrinsic CM5 only."""

    intrinsic = np.asarray(intrinsic_cm5_charges_e, dtype=float).reshape(-1)
    if intrinsic.shape != (int(synthons.natoms),) or np.any(~np.isfinite(intrinsic)):
        raise ValueError("intrinsic CM5 must contain one finite charge per synthon")
    coordination = (
        np.asarray(
            [len(synthons.neighbors[atom]) for atom in range(int(synthons.natoms))],
            dtype=float,
        )
        if intrinsic_coordination_numbers is None
        else np.asarray(intrinsic_coordination_numbers, dtype=float).reshape(-1)
    )
    if coordination.shape != intrinsic.shape or np.any(~np.isfinite(coordination)):
        raise ValueError("intrinsic covalent coordination must contain one value per atom")
    from matrix_chem.topology.continuous_graph import (
        connectivity_effective_covalent_radius,
    )
    output = []
    for atom in range(int(synthons.natoms)):
        output.append(
            SynthonDescriptor(
                atomic_number=int(synthons.Z[atom]),
                values=(
                    float(intrinsic[atom]),
                    float(coordination[atom]),
                    float(synthons.electron_domains(atom)),
                    float(
                        connectivity_effective_covalent_radius(
                            int(synthons.Z[atom]), float(coordination[atom])
                        )
                    ),
                    float(synthons.polarizability(atom)),
                    float(synthons.covalency(atom)),
                    float(synthons.delocalization(atom)),
                    float(synthons.strain(atom)),
                    float(synthons.sigma_index(atom)),
                    float(synthons.pi_index(atom)),
                    float(synthons.pi_pi_index(atom)),
                ),
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class FixedSynthonCM5Runtime:
    """Keep synthon types fixed while directional CM5 response changes charges."""

    type_indices: np.ndarray
    cm5_hydrogen_bond_model: object

    @classmethod
    def prepare(
        cls,
        synthons,
        catalog: "DiscreteSynthonTypeCatalog",
        cm5_hydrogen_bond_model: object,
    ) -> "FixedSynthonCM5Runtime":
        """Type only on the intrinsic CM5 baseline, never on runtime charges."""

        intrinsic = np.asarray(
            cm5_hydrogen_bond_model.intrinsic_charges_e, dtype=float
        )
        descriptors = descriptors_from_atomic_synthons(
            synthons,
            intrinsic_cm5_charges_e=intrinsic,
        )
        return cls(catalog.assign(descriptors), cm5_hydrogen_bond_model)

    @classmethod
    def prepare_charge_response(
        cls,
        synthons,
        catalog: "DiscreteSynthonTypeCatalog",
        cm5_charge_response_model: object,
    ) -> "FixedSynthonCM5Runtime":
        """General entry point; ``prepare`` remains the compatible H-bond API."""

        return cls.prepare(synthons, catalog, cm5_charge_response_model)

    @property
    def cm5_charge_response_model(self) -> object:
        """General name for the compatible stored response-model field."""

        return self.cm5_hydrogen_bond_model

    def __post_init__(self) -> None:
        indices = np.asarray(self.type_indices, dtype=np.int64).reshape(-1)
        intrinsic = np.asarray(
            self.cm5_hydrogen_bond_model.intrinsic_charges_e, dtype=float
        ).reshape(-1)
        if indices.shape != intrinsic.shape or np.any(indices < 0):
            raise ValueError("fixed synthon types and intrinsic CM5 charges disagree")
        object.__setattr__(self, "type_indices", indices)

    def evaluate(self, coordinates_angstrom: np.ndarray):
        """Return current CM5 charges while preserving the immutable type vector."""

        charge_result = self.cm5_hydrogen_bond_model.evaluate(coordinates_angstrom)
        return charge_result, self.type_indices


@dataclass(frozen=True)
class DiscreteSynthonTypeCatalog:
    """Deterministic threshold catalog used for precompiled pair lookup."""

    thresholds: SynthonTypeThresholds
    prototypes: tuple[SynthonDescriptor, ...]
    schema: str = ZAFF_SYNTHON_CATALOG_SCHEMA
    version: str = "1"

    def __post_init__(self) -> None:
        prototypes = tuple(self.prototypes)
        if not isinstance(self.thresholds, SynthonTypeThresholds):
            raise TypeError("synthon catalog thresholds are invalid")
        if any(not isinstance(item, SynthonDescriptor) for item in prototypes):
            raise TypeError("synthon catalog prototypes are invalid")
        if self.schema != ZAFF_SYNTHON_CATALOG_SCHEMA or not str(self.version).strip():
            raise ValueError("unsupported or unversioned ZAFF synthon catalog")
        object.__setattr__(self, "prototypes", prototypes)

    @classmethod
    def fit(
        cls,
        descriptors: Sequence[SynthonDescriptor],
        thresholds: SynthonTypeThresholds,
    ) -> "DiscreteSynthonTypeCatalog":
        ordered = sorted(
            descriptors,
            key=lambda item: (item.atomic_number, *item.values),
        )
        prototypes: list[SynthonDescriptor] = []
        scale = np.asarray(thresholds.values, dtype=float)
        for descriptor in ordered:
            compatible = [
                (
                    float(
                        np.linalg.norm(
                            (
                                np.asarray(descriptor.values)
                                - np.asarray(prototype.values)
                            )
                            / scale
                        )
                    ),
                    index,
                )
                for index, prototype in enumerate(prototypes)
                if prototype.atomic_number == descriptor.atomic_number
                and np.all(
                    np.abs(
                        np.asarray(descriptor.values)
                        - np.asarray(prototype.values)
                    )
                    <= scale
                )
            ]
            if not compatible:
                prototypes.append(descriptor)
        return cls(thresholds=thresholds, prototypes=tuple(prototypes))

    def assign(self, descriptors: Sequence[SynthonDescriptor]) -> np.ndarray:
        scale = np.asarray(self.thresholds.values, dtype=float)
        assignments = np.empty(len(descriptors), dtype=np.int64)
        for atom, descriptor in enumerate(descriptors):
            candidates = []
            for index, prototype in enumerate(self.prototypes):
                difference = np.asarray(descriptor.values) - np.asarray(prototype.values)
                if (
                    prototype.atomic_number == descriptor.atomic_number
                    and np.all(np.abs(difference) <= scale)
                ):
                    candidates.append(
                        (float(np.linalg.norm(difference / scale)), index)
                    )
            if not candidates:
                raise KeyError(
                    f"no ZAFF synthon type covers Z={descriptor.atomic_number}, "
                    f"descriptor={descriptor.values}"
                )
            assignments[atom] = min(candidates)[1]
        return assignments

    def to_record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "descriptor_fields": list(SYNTHON_DESCRIPTOR_FIELDS),
            "thresholds": list(self.thresholds.values),
            "prototypes": [
                {"atomic_number": item.atomic_number, "values": list(item.values)}
                for item in self.prototypes
            ],
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> "DiscreteSynthonTypeCatalog":
        if record.get("schema") != ZAFF_SYNTHON_CATALOG_SCHEMA:
            raise ValueError("unsupported ZAFF synthon catalog record")
        if tuple(record.get("descriptor_fields", ())) != SYNTHON_DESCRIPTOR_FIELDS:
            raise ValueError("ZAFF synthon descriptor fields changed incompatibly")
        raw_prototypes = record.get("prototypes")
        if not isinstance(raw_prototypes, list):
            raise ValueError("ZAFF synthon catalog has no prototype list")
        return cls(
            thresholds=SynthonTypeThresholds(tuple(record["thresholds"])),
            prototypes=tuple(
                SynthonDescriptor(
                    int(item["atomic_number"]), tuple(item["values"])
                )
                for item in raw_prototypes
            ),
            schema=str(record["schema"]),
            version=str(record.get("version", "")),
        )


@dataclass(frozen=True)
class SynthonExpPELibrary:
    """Square Exp-PE parameter matrices indexed by discrete synthon type."""

    catalog: DiscreteSynthonTypeCatalog
    epsilon_kcal_per_mol: np.ndarray
    r_min_angstrom: np.ndarray
    alpha: np.ndarray
    source: str
    schema: str = ZAFF_SYNTHON_EXPPE_LIBRARY_SCHEMA
    version: str = "1"

    def __post_init__(self) -> None:
        if (
            self.schema != ZAFF_SYNTHON_EXPPE_LIBRARY_SCHEMA
            or not str(self.version).strip()
            or not str(self.source).strip()
        ):
            raise ValueError("unsupported or unversioned ZAFF synthon Exp-PE library")
        count = len(self.catalog.prototypes)
        shape = (count, count)
        for name in ("epsilon_kcal_per_mol", "r_min_angstrom", "alpha"):
            values = np.asarray(getattr(self, name), dtype=float)
            invalid = (
                np.any(values <= 4.0)
                if name == "alpha"
                else np.any(values <= 0.0)
            )
            if (
                values.shape != shape
                or np.any(~np.isfinite(values))
                or invalid
                or not np.allclose(values, values.T)
            ):
                raise ValueError(f"synthon Exp-PE {name} must be a symmetric type matrix")
            values = values.copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @classmethod
    def from_uff_prior(
        cls,
        catalog: DiscreteSynthonTypeCatalog,
        *,
        version: str = "1",
    ) -> "SynthonExpPELibrary":
        """Populate a complete universal prior without inventing fitted data."""

        from .construction import compile_uff_exppe_pair_table

        numbers = np.asarray(
            [prototype.atomic_number for prototype in catalog.prototypes], dtype=int
        )
        table = compile_uff_exppe_pair_table(numbers, numbers)
        return cls(
            catalog=catalog,
            epsilon_kcal_per_mol=table.epsilon_kcal_per_mol,
            r_min_angstrom=table.r_min_angstrom,
            alpha=table.alpha,
            source="UFF universal prior compiled to ZAFF Exp-PE",
            version=str(version),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "source": self.source,
            "catalog": self.catalog.to_record(),
            "epsilon_kcal_per_mol": self.epsilon_kcal_per_mol.tolist(),
            "r_min_angstrom": self.r_min_angstrom.tolist(),
            "alpha": self.alpha.tolist(),
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> "SynthonExpPELibrary":
        if record.get("schema") != ZAFF_SYNTHON_EXPPE_LIBRARY_SCHEMA:
            raise ValueError("unsupported ZAFF synthon Exp-PE library record")
        return cls(
            catalog=DiscreteSynthonTypeCatalog.from_record(record["catalog"]),
            epsilon_kcal_per_mol=np.asarray(record["epsilon_kcal_per_mol"], dtype=float),
            r_min_angstrom=np.asarray(record["r_min_angstrom"], dtype=float),
            alpha=np.asarray(record["alpha"], dtype=float),
            source=str(record.get("source", "")),
            schema=str(record["schema"]),
            version=str(record.get("version", "")),
        )

    def resolve(
        self,
        candidate_descriptors: Sequence[SynthonDescriptor],
        environment_descriptors: Sequence[SynthonDescriptor],
    ) -> ResolvedExpPEPairTable:
        candidate_types = self.catalog.assign(candidate_descriptors)
        environment_types = self.catalog.assign(environment_descriptors)
        selection = np.ix_(candidate_types, environment_types)
        return ResolvedExpPEPairTable(
            epsilon_kcal_per_mol=self.epsilon_kcal_per_mol[selection],
            r_min_angstrom=self.r_min_angstrom[selection],
            alpha=self.alpha[selection],
            source=f"{self.source}; discrete complete-synthon type pairs",
        )


@dataclass(frozen=True)
class DirectionalExpPETypeRule:
    """One typed covalent-axis/terminal rule for H-bond or XB residuals."""

    kind: str
    anchor_type: int
    contact_type: int
    terminal_type: int
    epsilon_kcal_per_mol: float
    radial_scale_angstrom: float
    alpha: float
    angular_power: int
    charge_beta: float = 0.0
    synthon_beta: float = 0.0
    charge_product: float = 0.0
    synthon_score: float = 0.0

    def __post_init__(self) -> None:
        probe = DirectionalExpPEInteraction(
            kind=self.kind,
            anchor_side="candidate",
            anchor_index=0,
            contact_side="candidate",
            contact_index=1,
            terminal_side="environment",
            terminal_index=0,
            epsilon_kcal_per_mol=self.epsilon_kcal_per_mol,
            radial_scale_angstrom=self.radial_scale_angstrom,
            alpha=self.alpha,
            angular_power=self.angular_power,
            charge_beta=self.charge_beta,
            synthon_beta=self.synthon_beta,
            charge_product=self.charge_product,
            synthon_score=self.synthon_score,
        )
        if min(int(self.anchor_type), int(self.contact_type), int(self.terminal_type)) < 0:
            raise ValueError("directional Exp-PE type indices cannot be negative")
        object.__setattr__(self, "kind", probe.kind)
        object.__setattr__(self, "angular_power", probe.angular_power)


@dataclass(frozen=True)
class DirectionalExpPETypeLibrary:
    """Versioned typed residual library shared by every construction engine."""

    rules: tuple[DirectionalExpPETypeRule, ...]
    source: str
    type_namespace: str = "zaff_synthon_catalog"
    schema: str = ZAFF_DIRECTIONAL_EXPPE_LIBRARY_SCHEMA
    version: str = "1"

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if any(not isinstance(rule, DirectionalExpPETypeRule) for rule in rules):
            raise TypeError("directional Exp-PE library contains an invalid rule")
        if (
            self.schema != ZAFF_DIRECTIONAL_EXPPE_LIBRARY_SCHEMA
            or not str(self.version).strip()
            or not str(self.source).strip()
            or not str(self.type_namespace).strip()
        ):
            raise ValueError("unsupported or unversioned directional Exp-PE library")
        object.__setattr__(self, "rules", rules)

    def resolve_cross_fragment(
        self,
        candidate_types: Sequence[int],
        candidate_bonds: Sequence[tuple[int, int]],
        environment_types: Sequence[int],
        environment_bonds: Sequence[tuple[int, int]],
    ) -> ResolvedDirectionalExpPEContacts:
        """Resolve only cross-fragment contacts; covalent axes remain intrinsic."""

        candidate = np.asarray(candidate_types, dtype=int).reshape(-1)
        environment = np.asarray(environment_types, dtype=int).reshape(-1)

        def axes(types: np.ndarray, bonds: Sequence[tuple[int, int]]):
            output = []
            for raw_left, raw_right in bonds:
                left, right = int(raw_left), int(raw_right)
                if left == right or min(left, right) < 0 or max(left, right) >= len(types):
                    raise ValueError("directional library received an invalid covalent bond")
                output.extend(((left, right), (right, left)))
            return output

        interactions = []
        for anchor, contact in axes(candidate, candidate_bonds):
            for terminal in range(len(environment)):
                interactions.extend(
                    self._matching_interactions(
                        candidate[anchor],
                        candidate[contact],
                        environment[terminal],
                        "candidate",
                        anchor,
                        contact,
                        "environment",
                        terminal,
                    )
                )
        for anchor, contact in axes(environment, environment_bonds):
            for terminal in range(len(candidate)):
                interactions.extend(
                    self._matching_interactions(
                        environment[anchor],
                        environment[contact],
                        candidate[terminal],
                        "environment",
                        anchor,
                        contact,
                        "candidate",
                        terminal,
                    )
                )
        return ResolvedDirectionalExpPEContacts(
            tuple(interactions),
            source=(
                f"{self.source}; {self.type_namespace}; residual-only; "
                "cross-fragment contacts"
            ),
        )

    def _matching_interactions(
        self,
        anchor_type: int,
        contact_type: int,
        terminal_type: int,
        axis_side: str,
        anchor_index: int,
        contact_index: int,
        terminal_side: str,
        terminal_index: int,
    ) -> list[DirectionalExpPEInteraction]:
        return [
            DirectionalExpPEInteraction(
                kind=rule.kind,
                anchor_side=axis_side,
                anchor_index=anchor_index,
                contact_side=axis_side,
                contact_index=contact_index,
                terminal_side=terminal_side,
                terminal_index=terminal_index,
                epsilon_kcal_per_mol=rule.epsilon_kcal_per_mol,
                radial_scale_angstrom=rule.radial_scale_angstrom,
                alpha=rule.alpha,
                angular_power=rule.angular_power,
                charge_beta=rule.charge_beta,
                synthon_beta=rule.synthon_beta,
                charge_product=rule.charge_product,
                synthon_score=rule.synthon_score,
            )
            for rule in self.rules
            if (
                rule.anchor_type == int(anchor_type)
                and rule.contact_type == int(contact_type)
                and rule.terminal_type == int(terminal_type)
            )
        ]

    def to_record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "source": self.source,
            "type_namespace": self.type_namespace,
            "rules": [
                {
                    "kind": rule.kind,
                    "anchor_type": rule.anchor_type,
                    "contact_type": rule.contact_type,
                    "terminal_type": rule.terminal_type,
                    "epsilon_kcal_per_mol": rule.epsilon_kcal_per_mol,
                    "radial_scale_angstrom": rule.radial_scale_angstrom,
                    "alpha": rule.alpha,
                    "angular_power": rule.angular_power,
                    "charge_beta": rule.charge_beta,
                    "synthon_beta": rule.synthon_beta,
                    "charge_product": rule.charge_product,
                    "synthon_score": rule.synthon_score,
                }
                for rule in self.rules
            ],
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> "DirectionalExpPETypeLibrary":
        if record.get("schema") != ZAFF_DIRECTIONAL_EXPPE_LIBRARY_SCHEMA:
            raise ValueError("unsupported directional Exp-PE library record")
        return cls(
            rules=tuple(
                DirectionalExpPETypeRule(**item) for item in record.get("rules", ())
            ),
            source=str(record.get("source", "")),
            type_namespace=str(record.get("type_namespace", "")),
            schema=str(record["schema"]),
            version=str(record.get("version", "")),
        )
