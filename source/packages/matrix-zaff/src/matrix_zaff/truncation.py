"""Versioned truncation levels for compiled ZAFF internal-coordinate fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ZAFF_TRUNCATION_SCHEMA = "matrix.zaff.truncation.v1"
ZAFF_TRUNCATION_LEVELS = (
    "DIAGONAL_HARMONIC",
    "HARMONIC",
    "DIAGONAL_QFF",
    "SEMIDIAGONAL_QFF",
    "RIGID_TORSIONAL",
    "GEMINAL_XH_DIAGONAL",
    "PHYSICAL_DIAGONAL",
    "SELECTED_COUPLINGS",
    "FULL",
)


@dataclass(frozen=True)
class ZaffTruncationPolicy:
    """One nested, reproducible view of a fully compiled ZAFF field."""

    level: str = "FULL"
    maximum_diagonal_order: int | None = None
    include_offdiagonal_quadratic: bool | None = None
    include_semidiagonal_cubic: bool | None = None
    include_physical_diagonal: bool | None = None
    include_coupled_terms: bool | None = None
    include_large_amplitude_surfaces: bool | None = None
    included_coupling_identifiers: tuple[str, ...] = ()
    excluded_coupling_identifiers: tuple[str, ...] = ()
    included_coupling_forms: tuple[str, ...] = ()
    included_coordinate_pairs: tuple[tuple[str, str], ...] = ()
    excluded_coordinate_pairs: tuple[tuple[str, str], ...] = ()
    schema: str = ZAFF_TRUNCATION_SCHEMA

    def __post_init__(self) -> None:
        level = str(self.level).upper()
        if self.schema != ZAFF_TRUNCATION_SCHEMA:
            raise ValueError("unsupported ZAFF truncation schema")
        if level not in ZAFF_TRUNCATION_LEVELS and level != "CUSTOM":
            raise ValueError(f"unsupported ZAFF truncation level {self.level!r}")
        object.__setattr__(self, "level", level)
        if self.maximum_diagonal_order is not None and self.maximum_diagonal_order not in {2, 3, 4}:
            raise ValueError("ZAFF maximum diagonal order must be 2, 3 or 4")
        if level == "CUSTOM" and self.maximum_diagonal_order is None:
            raise ValueError("custom ZAFF truncation requires maximum_diagonal_order")
        included = tuple(str(value) for value in self.included_coupling_identifiers)
        excluded = tuple(str(value) for value in self.excluded_coupling_identifiers)
        forms = tuple(str(value) for value in self.included_coupling_forms)
        if set(included) & set(excluded):
            raise ValueError("a ZAFF coupling cannot be both included and excluded")
        object.__setattr__(self, "included_coupling_identifiers", included)
        object.__setattr__(self, "excluded_coupling_identifiers", excluded)
        object.__setattr__(self, "included_coupling_forms", forms)
        included_pairs = _canonical_pairs(self.included_coordinate_pairs)
        excluded_pairs = _canonical_pairs(self.excluded_coordinate_pairs)
        if set(included_pairs) & set(excluded_pairs):
            raise ValueError("a ZAFF coordinate pair cannot be both included and excluded")
        object.__setattr__(self, "included_coordinate_pairs", included_pairs)
        object.__setattr__(self, "excluded_coordinate_pairs", excluded_pairs)

    @classmethod
    def named(cls, level: str) -> "ZaffTruncationPolicy":
        name = str(level).upper()
        name = {
            "ZAFF0": "RIGID_TORSIONAL",
            "ZAFF_LEVEL0": "RIGID_TORSIONAL",
            "ZAFF1": "GEMINAL_XH_DIAGONAL",
            "ZAFF2": "PHYSICAL_DIAGONAL",
            "ZAFF3": "FULL",
        }.get(name, name)
        settings = {
            "DIAGONAL_HARMONIC": (2, False, False, False, False, False),
            "HARMONIC": (2, True, False, False, False, False),
            "DIAGONAL_QFF": (4, True, False, False, False, False),
            "SEMIDIAGONAL_QFF": (4, True, True, False, False, False),
            "RIGID_TORSIONAL": (2, False, False, False, False, True),
            "GEMINAL_XH_DIAGONAL": (4, False, False, True, False, True),
            "PHYSICAL_DIAGONAL": (4, True, False, True, False, True),
            "SELECTED_COUPLINGS": (4, True, True, True, True, True),
            "FULL": (4, True, True, True, True, True),
        }
        if name not in settings:
            raise ValueError(f"unsupported ZAFF truncation level {level!r}")
        order, offdiagonal_quadratic, semidiagonal, physical, coupled, large_amplitude = settings[name]
        return cls(
            level=name,
            maximum_diagonal_order=order,
            include_offdiagonal_quadratic=offdiagonal_quadratic,
            include_semidiagonal_cubic=semidiagonal,
            include_physical_diagonal=physical,
            include_coupled_terms=coupled,
            include_large_amplitude_surfaces=large_amplitude,
        )

    @classmethod
    def resolve(
        cls, value: "ZaffTruncationPolicy | Mapping[str, Any] | str | None"
    ) -> "ZaffTruncationPolicy":
        if value is None:
            return cls.named("FULL")
        if isinstance(value, cls):
            return value.expanded()
        if isinstance(value, str):
            return cls.named(value)
        return cls.from_dict(value).expanded()

    def expanded(self) -> "ZaffTruncationPolicy":
        if self.level == "CUSTOM":
            return self
        baseline = self.named(self.level)
        return ZaffTruncationPolicy(
            level=self.level,
            maximum_diagonal_order=(
                baseline.maximum_diagonal_order
                if self.maximum_diagonal_order is None
                else self.maximum_diagonal_order
            ),
            include_offdiagonal_quadratic=(
                baseline.include_offdiagonal_quadratic
                if self.include_offdiagonal_quadratic is None
                else self.include_offdiagonal_quadratic
            ),
            include_semidiagonal_cubic=(
                baseline.include_semidiagonal_cubic
                if self.include_semidiagonal_cubic is None
                else self.include_semidiagonal_cubic
            ),
            include_physical_diagonal=(
                baseline.include_physical_diagonal
                if self.include_physical_diagonal is None
                else self.include_physical_diagonal
            ),
            include_coupled_terms=(
                baseline.include_coupled_terms
                if self.include_coupled_terms is None
                else self.include_coupled_terms
            ),
            include_large_amplitude_surfaces=(
                baseline.include_large_amplitude_surfaces
                if self.include_large_amplitude_surfaces is None
                else self.include_large_amplitude_surfaces
            ),
            included_coupling_identifiers=self.included_coupling_identifiers,
            excluded_coupling_identifiers=self.excluded_coupling_identifiers,
            included_coupling_forms=self.included_coupling_forms,
            included_coordinate_pairs=self.included_coordinate_pairs,
            excluded_coordinate_pairs=self.excluded_coordinate_pairs,
        )

    def to_dict(self) -> dict[str, Any]:
        policy = self.expanded()
        return {
            "schema": policy.schema,
            "level": policy.level,
            "maximum_diagonal_order": policy.maximum_diagonal_order,
            "include_offdiagonal_quadratic": policy.include_offdiagonal_quadratic,
            "include_semidiagonal_cubic": policy.include_semidiagonal_cubic,
            "include_physical_diagonal": policy.include_physical_diagonal,
            "include_coupled_terms": policy.include_coupled_terms,
            "include_large_amplitude_surfaces": policy.include_large_amplitude_surfaces,
            "included_coupling_identifiers": list(policy.included_coupling_identifiers),
            "excluded_coupling_identifiers": list(policy.excluded_coupling_identifiers),
            "included_coupling_forms": list(policy.included_coupling_forms),
            "included_coordinate_pairs": [list(pair) for pair in policy.included_coordinate_pairs],
            "excluded_coordinate_pairs": [list(pair) for pair in policy.excluded_coordinate_pairs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffTruncationPolicy":
        return cls(
            schema=str(payload.get("schema", ZAFF_TRUNCATION_SCHEMA)),
            level=str(payload.get("level", "FULL")),
            maximum_diagonal_order=(None if payload.get("maximum_diagonal_order") is None else int(payload["maximum_diagonal_order"])),
            include_offdiagonal_quadratic=_optional_bool(payload.get("include_offdiagonal_quadratic")),
            include_semidiagonal_cubic=_optional_bool(payload.get("include_semidiagonal_cubic")),
            include_physical_diagonal=_optional_bool(payload.get("include_physical_diagonal")),
            include_coupled_terms=_optional_bool(payload.get("include_coupled_terms")),
            include_large_amplitude_surfaces=_optional_bool(payload.get("include_large_amplitude_surfaces")),
            included_coupling_identifiers=tuple(
                str(value) for value in payload.get("included_coupling_identifiers", ())
            ),
            excluded_coupling_identifiers=tuple(
                str(value) for value in payload.get("excluded_coupling_identifiers", ())
            ),
            included_coupling_forms=tuple(
                str(value) for value in payload.get("included_coupling_forms", ())
            ),
            included_coordinate_pairs=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in payload.get("included_coordinate_pairs", ())
            ),
            excluded_coordinate_pairs=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in payload.get("excluded_coordinate_pairs", ())
            ),
        )

    def allows_coordinate_pair(self, first: str, second: str) -> bool:
        pair = tuple(sorted((str(first), str(second))))
        policy = self.expanded()
        if pair in policy.excluded_coordinate_pairs:
            return False
        return not policy.included_coordinate_pairs or pair in policy.included_coordinate_pairs

    def allows_coupled_term(self, record: Mapping[str, Any]) -> bool:
        """Select a complete switched ZAFF coupling without altering its kernel."""

        policy = self.expanded()
        if not policy.include_coupled_terms:
            return False
        identifier = str(record.get("identifier", ""))
        form = str(dict(record.get("potential", {})).get("functional_form", ""))
        if identifier in policy.excluded_coupling_identifiers:
            return False
        if policy.included_coupling_identifiers and identifier not in policy.included_coupling_identifiers:
            return False
        if policy.included_coupling_forms and form not in policy.included_coupling_forms:
            return False
        return True


def zaff_truncation_contract() -> dict[str, Any]:
    """Return the immutable list of nested runtime truncation levels."""

    return {
        "schema": ZAFF_TRUNCATION_SCHEMA,
        "hierarchy_version": "2.0.0",
        "default_level": "FULL",
        "levels_low_to_high": list(ZAFF_TRUNCATION_LEVELS),
        "zoom_levels_low_to_high": [
            "ZAFF0",
            "ZAFF1",
            "ZAFF2",
            "ZAFF3",
        ],
        "zoom_level_mapping": {
            "ZAFF0": "RIGID_TORSIONAL",
            "ZAFF1": "GEMINAL_XH_DIAGONAL",
            "ZAFF2": "PHYSICAL_DIAGONAL",
            "ZAFF3": "FULL",
        },
        "harmonic_invariant": (
            "ZAFF2_AND_ZAFF3_RETAIN_THE_COMPLETE_FITTED_QUADRATIC_FIELD"
        ),
        "level0_contract": "RIGID_GEOMETRY_TORSIONAL_AND_RIGID_BODY_EXPLORATION",
        "level0_projection_owner": "LINK",
        "level1_contract": (
            "PHYSICAL_DIAGONAL_TERMS_PLUS_ONLY_DISSOCIATING_GEMINAL_XH_COUPLINGS"
        ),
        "level1_stretches": (
            "MORSE_R0_FROM_REFERENCE_K2_FROM_QM_MINUS_ELECTROSTATIC_MINUS_VDW_"
            "HESSIAN_DE_FROM_ZAFF_LIBRARY"
        ),
        "base_anharmonicity": (
            "DIAGONAL_CUBIC_AND_QUARTIC_DERIVATIVES_ARE_GENERATED_BY_PHYSICAL_FUNCTIONS"
        ),
        "refinement": (
            "ADDITIONAL_DIAGONAL_FUNCTION_TERMS_THEN_SELECTED_OR_FULL_HIGHER_ORDER_COUPLINGS"
        ),
        "zaff3_compactness": (
            "NATIVE_INVERSE_DISTANCE_COSINE_FOURIER_VARIABLES_REQUIRE_FEWER_MIXED_"
            "COUPLINGS_THAN_CARTESIAN_NORMAL_MODE_QFF"
        ),
        "hydrogen_bond_switch": "C2_QUINTIC_RADIAL_TIMES_ANGULAR_ENVELOPE",
        "transferable_hydrogen_policy": (
            "BICOORDINATED_COVALENT_DOMAIN_REPLACES_ORDINARY_HYDROGEN_BOND_TERM"
        ),
        "diagnostic_taylor_levels": [
            "DIAGONAL_HARMONIC",
            "HARMONIC",
            "DIAGONAL_QFF",
            "SEMIDIAGONAL_QFF",
        ],
        "quadratic_offdiagonal_terms_selectable": True,
        "anharmonic_offdiagonal_terms_selectable": True,
        "selected_terms_retain_complete_switch_kernels": True,
        "artifact_retains_complete_field": True,
    }


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _canonical_pairs(values: Any) -> tuple[tuple[str, str], ...]:
    pairs = tuple(tuple(str(item) for item in pair) for pair in values)
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError("ZAFF coordinate-pair selectors require two distinct identifiers")
    return tuple(tuple(sorted(pair)) for pair in pairs)


__all__ = [
    "ZAFF_TRUNCATION_LEVELS",
    "ZAFF_TRUNCATION_SCHEMA",
    "ZaffTruncationPolicy",
    "zaff_truncation_contract",
]
