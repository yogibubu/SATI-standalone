"""Curvilinear rovibrational corrections in nonredundant SONIC coordinates.

This module owns the representation-level result of the curvilinear
rovibrational calculation.  Geometry fitting remains in MORPHEUS and
electronic-structure execution remains in the corresponding backend adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import pi, sqrt
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from matrix_chem.physical_constants import Phy, get_physical_constants
def isotopic_masses_amu(*args, **kwargs):
    from importlib import import_module
    return import_module("matrix_rovib").isotopic_masses_amu(*args, **kwargs)


def rovibrational_kinematics_from_modes(*args, **kwargs):
    from importlib import import_module
    return import_module("matrix_rovib").rovibrational_kinematics_from_modes(*args, **kwargs)

from .isotopic_internal_qff import (
    NonredundantInternalCubicField,
    normal_cubic_from_internal_field,
)


CURVILINEAR_DELTABVIB_SCHEMA = "matrix.trinity.curvilinear-deltabvib.v1"

_CONSTANTS = get_physical_constants()
_AMU_KG = _CONSTANTS[Phy.TO_KG]
_ANG_PER_BOHR = _CONSTANTS[Phy.TO_ANG]
_M_PER_BOHR = _CONSTANTS[Phy.M_PER_B]
_CLIGHT_CM_S = _CONSTANTS[Phy.C_LIGHT]
_PLANCK_J_S = _CONSTANTS[Phy.PLANCK]
_PLANCK_AU = _PLANCK_J_S / (_AMU_KG * _M_PER_BOHR**2)
_J_PER_HARTREE = _CONSTANTS[Phy.HARTREE]


@dataclass(frozen=True)
class CurvilinearAlphaComponents:
    """Mode-resolved vibration--rotation constants in MHz.

    Each array has shape ``(nvib, 3)`` in the principal-axis order A, B, C.
    Separating the metric, Coriolis and potential-curvature contributions makes
    the result auditable and avoids hiding a Cartesian approximation behind a
    nominally curvilinear label.
    """

    metric_MHz: np.ndarray
    coriolis_MHz: np.ndarray
    potential_MHz: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(value, dtype=float)
            for name, value in (
                ("metric_MHz", self.metric_MHz),
                ("coriolis_MHz", self.coriolis_MHz),
                ("potential_MHz", self.potential_MHz),
            )
        }
        shape = arrays["metric_MHz"].shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("curvilinear alpha components must have shape (nvib, 3)")
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("all curvilinear alpha components must have the same shape")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("curvilinear alpha components must be finite")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def total_MHz(self) -> np.ndarray:
        return self.metric_MHz + self.coriolis_MHz + self.potential_MHz


@dataclass(frozen=True)
class CurvilinearDeltaBVibResult:
    """One isotope-specific curvilinear Delta Bvib result.

    ``delta_MHz`` follows the MORPHEUS correction convention: it is subtracted
    from experimental ground-state constants to obtain semiexperimental
    equilibrium constants.
    """

    label: str
    substitutions: Mapping[int, int]
    frequencies_cm1: tuple[float, ...]
    alpha: CurvilinearAlphaComponents
    delta_MHz: tuple[float, float, float]
    representation: str
    source: str
    excluded_modes: tuple[int, ...] = ()
    schema: str = CURVILINEAR_DELTABVIB_SCHEMA


@dataclass(frozen=True)
class CurvilinearIsotopologueState:
    """Mass-specific GF solution used to transform one internal F2/F3 field."""

    label: str
    substitutions: Mapping[int, int]
    masses_amu: np.ndarray
    frequencies_cm1: np.ndarray
    modes_mw: np.ndarray
    representation: str = "Ir"
    exclude_modes: tuple[int, ...] = ()


@dataclass(frozen=True)
class CurvilinearIsotopologueDefinition:
    """Chemical isotope substitutions from which TRINITY rebuilds the GF state."""

    label: str
    substitutions: Mapping[int, int]
    representation: str = "Ir"
    exclude_modes: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrinityDeltaBVibMethod:
    """Provenance and capability descriptor for one TRINITY DeltaBvib route."""

    name: str
    derivative_source: str
    normal_mode_basis: str
    supported_purposes: tuple[str, ...] = (
        "semiexperimental-structure",
        "spectroscopy",
        "independent-validation",
    )


SONIC_INTERNAL_DELTABVIB_METHOD = TrinityDeltaBVibMethod(
    "sonic-internal-field",
    "gradient-or-hessian-acquired-f2-f3",
    "sonic-gf",
)
CARTESIAN_GRADIENT_DELTABVIB_METHOD = TrinityDeltaBVibMethod(
    "cartesian-gradient-stencil", "analytic-gradients", "cartesian-normal-modes"
)
CARTESIAN_HESSIAN_DELTABVIB_METHOD = TrinityDeltaBVibMethod(
    "cartesian-hessian-stencil", "analytic-or-numerical-hessians", "cartesian-normal-modes"
)
SONIC_GRADIENT_DELTABVIB_METHOD = TrinityDeltaBVibMethod(
    "sonic-gradient-stencil", "analytic-gradients", "sonic-gf"
)
SONIC_HESSIAN_DELTABVIB_METHOD = TrinityDeltaBVibMethod(
    "sonic-hessian-stencil", "analytic-or-numerical-hessians", "sonic-gf"
)


@dataclass(frozen=True)
class CurvilinearDeltaBVibJob:
    """Reusable TRINITY calculation attached to one L0 SONIC field."""

    field: NonredundantInternalCubicField
    b_matrix: np.ndarray
    b_prime: np.ndarray
    symbols: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    source: str = "TRINITY nonredundant SONIC F2/F3 and isotope-specific GF"
    method: TrinityDeltaBVibMethod = SONIC_INTERNAL_DELTABVIB_METHOD

    def calculate(
        self,
        definitions: Sequence[CurvilinearIsotopologueDefinition],
        *,
        purpose: str = "semiexperimental-structure",
    ) -> tuple[CurvilinearDeltaBVibResult, ...]:
        if purpose not in self.method.supported_purposes:
            raise ValueError(f"TRINITY DeltaBvib method {self.method.name} does not support {purpose}")
        rows = curvilinear_deltabvib_from_definitions(
            self.field,
            self.b_matrix,
            self.b_prime,
            self.symbols,
            self.coordinates_angstrom,
            definitions,
            source=self.source,
        )
        return tuple(
            replace(
                row,
                source=(
                    f"TRINITY method={self.method.name} "
                    f"derivatives={self.method.derivative_source} "
                    f"modes={self.method.normal_mode_basis} purpose={purpose}; {row.source}"
                ),
            )
            for row in rows
        )


@dataclass(frozen=True)
class CallableDeltaBVibJob:
    """Adapter for gradient-, Hessian-, Cartesian- or SONIC-based TRINITY kernels."""

    method: TrinityDeltaBVibMethod
    calculator: Callable[[Sequence[CurvilinearIsotopologueDefinition]], Sequence[CurvilinearDeltaBVibResult]]

    def calculate(
        self,
        definitions: Sequence[CurvilinearIsotopologueDefinition],
        *,
        purpose: str = "spectroscopy",
    ) -> tuple[CurvilinearDeltaBVibResult, ...]:
        if purpose not in self.method.supported_purposes:
            raise ValueError(f"TRINITY DeltaBvib method {self.method.name} does not support {purpose}")
        return tuple(
            replace(
                row,
                source=(
                    f"TRINITY method={self.method.name} "
                    f"derivatives={self.method.derivative_source} "
                    f"modes={self.method.normal_mode_basis} purpose={purpose}; {row.source}"
                ),
            )
            for row in self.calculator(definitions)
        )


@dataclass(frozen=True)
class TrinityDeltaBVibService:
    """Select an available TRINITY provider from the requested downstream use."""

    providers: tuple[object, ...]

    def calculate(
        self,
        definitions: Sequence[CurvilinearIsotopologueDefinition] | str,
        substitutions: Mapping[int, int] | None = None,
        *,
        purpose: str = "semiexperimental-structure",
    ) -> tuple[CurvilinearDeltaBVibResult, ...] | object:
        if isinstance(definitions, str):
            if substitutions is None:
                raise ValueError("isotopologue substitutions are required with a label")
            if len(self.providers) != 1:
                raise ValueError(
                    "label-based TRINITY calculation requires exactly one persistent provider"
                )
            provider = self.providers[0]
            if not hasattr(provider, "acquisition"):
                raise TypeError("configured TRINITY provider does not support label-based calculation")
            return provider.calculate(definitions, substitutions)
        if substitutions is not None:
            raise TypeError("substitutions are valid only with a single isotopologue label")
        eligible = [
            provider
            for provider in self.providers
            if purpose in provider.method.supported_purposes
        ]
        if not eligible:
            raise ValueError(f"no configured TRINITY DeltaBvib method supports {purpose}")
        prefer_sonic = purpose == "semiexperimental-structure"
        eligible.sort(
            key=lambda provider: (
                0
                if (provider.method.normal_mode_basis == "sonic-gf") == prefer_sonic
                else 1,
                provider.method.name,
            )
        )
        return eligible[0].calculate(definitions, purpose=purpose)


def curvilinear_isotopologue_state_from_internal_field(
    field: NonredundantInternalCubicField,
    b_matrix,
    masses_amu,
    definition: CurvilinearIsotopologueDefinition,
) -> CurvilinearIsotopologueState:
    """Rebuild the mass-dependent Wilson GF problem and Cartesian mode vectors."""

    from matrix_gf import solve_wilson_gf

    b = np.asarray(b_matrix, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    dimension = 3 * masses.size
    if b.shape != (field.coordinate_count, dimension):
        raise ValueError("SONIC B matrix and isotopologue masses disagree")
    if np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
        raise ValueError("isotopologue masses must be positive and finite")
    inverse_mass = 1.0 / np.repeat(masses, 3)
    g_matrix = (b * inverse_mass[None, :]) @ b.T
    gf = solve_wilson_gf(field.harmonic_internal, g_matrix, scale_to_cm=True)
    if np.any(gf.frequencies_cm <= 0.0):
        raise ValueError("curvilinear DeltaBvib requires a stable isotopologue GF solution")
    g_values, g_vectors = np.linalg.eigh(0.5 * (g_matrix + g_matrix.T))
    if np.any(g_values <= 0.0):
        raise ValueError("isotopologue Wilson G matrix must be positive definite")
    g_half_inverse = (g_vectors * (1.0 / np.sqrt(g_values))) @ g_vectors.T
    b_mass_weighted = b / np.sqrt(np.repeat(masses, 3))[None, :]
    cartesian_columns = b_mass_weighted.T @ g_half_inverse @ gf.normal_modes
    modes = cartesian_columns.T.reshape((field.coordinate_count, masses.size, 3))
    return CurvilinearIsotopologueState(
        label=definition.label,
        substitutions=dict(definition.substitutions),
        masses_amu=masses,
        frequencies_cm1=np.asarray(gf.frequencies_cm, dtype=float),
        modes_mw=modes,
        representation=definition.representation,
        exclude_modes=definition.exclude_modes,
    )


def curvilinear_qcent_from_sonic_state(
    coordinates_angstrom,
    state: CurvilinearIsotopologueState,
):
    """Watson quartic distortion from isotope-specific internal Wilson GF modes.

    The principal-axis inertia derivatives use the mass-weighted Cartesian
    realization of the SONIC modes; the modes themselves come from the
    nonredundant internal GF problem, not a Cartesian Hessian diagonalization.
    """
    from importlib import import_module
    rovib = import_module("matrix_rovib")
    qcent = import_module("matrix_rovib.qcent")
    compute_qcent_from_inertia_derivatives = rovib.compute_qcent_from_inertia_derivatives
    didq_sym6 = rovib.didq_sym6
    inertia_tensor_amuA2 = qcent.inertia_tensor_amuA2

    masses = np.asarray(state.masses_amu, dtype=float).reshape(-1)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    modes = np.asarray(state.modes_mw, dtype=float)
    frequencies = np.asarray(state.frequencies_cm1, dtype=float)
    if xyz.shape != (masses.size, 3) or modes.shape != (frequencies.size, masses.size, 3):
        raise ValueError("SONIC QCENT geometry and GF modes have inconsistent dimensions")
    if not all(np.all(np.isfinite(value)) for value in (xyz, modes, masses, frequencies)):
        raise ValueError("SONIC QCENT requires finite geometry, masses and modes")
    overlap = modes.reshape((frequencies.size, -1)) @ modes.reshape((frequencies.size, -1)).T
    if not np.allclose(overlap, np.eye(frequencies.size), atol=1e-5):
        raise ValueError("SONIC QCENT requires orthonormal mass-weighted GF modes")
    centered = xyz - np.average(xyz, axis=0, weights=masses)
    moments, axes = np.linalg.eigh(inertia_tensor_amuA2(masses, centered))
    if np.min(moments) <= 1e-10:
        raise ValueError("SONIC QCENT currently requires a nonlinear principal-axis rotor")
    principal_xyz = centered @ axes
    principal_modes = modes @ axes
    derivatives = didq_sym6(masses, principal_xyz, principal_modes)
    return compute_qcent_from_inertia_derivatives(
        masses, principal_xyz, frequencies, derivatives,
        representation=state.representation,
    )


def curvilinear_deltabvib_from_alpha(
    label: str,
    substitutions: Mapping[int, int],
    frequencies_cm1,
    alpha: CurvilinearAlphaComponents,
    *,
    representation: str,
    source: str,
    exclude_modes: tuple[int, ...] = (),
    invert_imaginary_modes: bool = True,
) -> CurvilinearDeltaBVibResult:
    """Contract mode-resolved curvilinear alpha constants into Delta Bvib."""

    frequencies = np.asarray(frequencies_cm1, dtype=float)
    if frequencies.shape != (alpha.total_MHz.shape[0],):
        raise ValueError("one harmonic frequency is required for every SONIC normal mode")
    if representation not in {"Ir", "IIIr"}:
        raise ValueError("asymmetric-top representation must be Ir or IIIr")
    excluded = {int(mode) for mode in exclude_modes}
    if any(mode < 1 or mode > frequencies.size for mode in excluded):
        raise ValueError("excluded mode index lies outside the vibrational space")
    rows = alpha.total_MHz.copy()
    if invert_imaginary_modes:
        rows[frequencies < 0.0] *= -1.0
    if excluded:
        rows[np.asarray([index + 1 in excluded for index in range(frequencies.size)])] = 0.0
    delta = 0.5 * np.sum(rows, axis=0)
    return CurvilinearDeltaBVibResult(
        label=str(label),
        substitutions={int(key): int(value) for key, value in substitutions.items()},
        frequencies_cm1=tuple(float(value) for value in frequencies),
        alpha=alpha,
        delta_MHz=tuple(float(value) for value in delta),
        representation=representation,
        source=str(source),
        excluded_modes=tuple(sorted(excluded)),
    )


def curvilinear_alpha_from_internal_field(
    field: NonredundantInternalCubicField,
    b_matrix,
    b_prime,
    coordinates_angstrom,
    masses_amu,
    frequencies_cm1,
    modes_mw,
) -> CurvilinearAlphaComponents:
    """Calculate mode-resolved isotope-specific alpha constants from SONIC F2/F3.

    The potential sector is transformed from the mass-independent nonredundant
    internal field with the exact B/B-prime curvature term.  The metric and
    Coriolis sectors are rebuilt from the same isotope-specific GF modes in the
    principal-axis Eckart frame.  No Cartesian electronic cubic field is used.
    """

    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies_cm1, dtype=float).reshape(-1)
    modes = np.asarray(modes_mw, dtype=float)
    if modes.shape != (frequencies.size, masses.size, 3):
        raise ValueError("isotopologue modes must have shape nvib x natoms x 3")
    if frequencies.size != field.coordinate_count:
        raise ValueError("isotopologue GF mode count and internal field disagree")
    if np.any(frequencies <= 0.0) or not np.all(np.isfinite(frequencies)):
        raise ValueError("curvilinear DeltaBvib requires positive finite GF frequencies")
    cartesian_per_q = modes.reshape((frequencies.size, -1)).T / np.sqrt(
        np.repeat(masses, 3)
    )[:, None]
    _harmonic, cubic_qmw = normal_cubic_from_internal_field(
        field,
        b_matrix,
        b_prime,
        cartesian_per_q,
    )
    kinematics = rovibrational_kinematics_from_modes(
        coordinates_angstrom,
        masses,
        modes,
    )
    return _mode_resolved_alpha_components(
        frequencies,
        cubic_qmw,
        kinematics.inertia_principal_amu_bohr2,
        kinematics.inertia_derivatives_amu_sqrt_ang,
        kinematics.coriolis,
    )


def curvilinear_deltabvib_from_internal_field(
    field: NonredundantInternalCubicField,
    b_matrix,
    b_prime,
    coordinates_angstrom,
    state: CurvilinearIsotopologueState,
    *,
    source: str = "TRINITY nonredundant SONIC F2/F3 and isotope-specific GF",
) -> CurvilinearDeltaBVibResult:
    """Calculate one complete curvilinear Delta Bvib result for MORPHEUS."""

    alpha = curvilinear_alpha_from_internal_field(
        field,
        b_matrix,
        b_prime,
        coordinates_angstrom,
        state.masses_amu,
        state.frequencies_cm1,
        state.modes_mw,
    )
    return curvilinear_deltabvib_from_alpha(
        state.label,
        state.substitutions,
        state.frequencies_cm1,
        alpha,
        representation=state.representation,
        source=source,
        exclude_modes=state.exclude_modes,
    )


def curvilinear_deltabvib_from_semidiagonal_cubic(
    coordinates_angstrom,
    state: CurvilinearIsotopologueState,
    cubic_fjkk_qmw,
    *,
    source: str = "TRINITY isotope-specific semidiagonal gradient stencil",
) -> CurvilinearDeltaBVibResult:
    """Use only Phi(j,k,k), acquired in this isotopologue's own GF modes.

    DeltaBvib's potential term accesses no all-distinct cubic components.
    Sampling each isotopologue separately avoids inferring missing parent
    all-distinct components when isotope substitution rotates the modes.
    """

    frequencies = np.asarray(state.frequencies_cm1, dtype=float)
    cubic = np.asarray(cubic_fjkk_qmw, dtype=float)
    if cubic.shape != (len(frequencies), len(frequencies)) or not np.all(np.isfinite(cubic)):
        raise ValueError("isotopologue semidiagonal cubic field has invalid dimensions")
    masses = np.asarray(state.masses_amu, dtype=float)
    modes = np.asarray(state.modes_mw, dtype=float)
    kinematics = rovibrational_kinematics_from_modes(
        coordinates_angstrom, masses, modes,
    )
    alpha = _mode_resolved_alpha_components(
        frequencies,
        cubic,
        kinematics.inertia_principal_amu_bohr2,
        kinematics.inertia_derivatives_amu_sqrt_ang,
        kinematics.coriolis,
    )
    return curvilinear_deltabvib_from_alpha(
        state.label,
        state.substitutions,
        frequencies,
        alpha,
        representation=state.representation,
        source=source,
        exclude_modes=state.exclude_modes,
    )


def curvilinear_deltabvib_for_isotopologues(
    field: NonredundantInternalCubicField,
    b_matrix,
    b_prime,
    coordinates_angstrom,
    states: Sequence[CurvilinearIsotopologueState],
    *,
    source: str = "TRINITY nonredundant SONIC F2/F3 and isotope-specific GF",
) -> tuple[CurvilinearDeltaBVibResult, ...]:
    """Transform one mass-independent field and calculate every isotope correction."""

    labels = [str(state.label) for state in states]
    if not labels:
        raise ValueError("at least one isotopologue state is required")
    if len(set(labels)) != len(labels):
        raise ValueError("isotopologue labels must be unique")
    return tuple(
        curvilinear_deltabvib_from_internal_field(
            field,
            b_matrix,
            b_prime,
            coordinates_angstrom,
            state,
            source=source,
        )
        for state in states
    )


def curvilinear_deltabvib_from_definitions(
    field: NonredundantInternalCubicField,
    b_matrix,
    b_prime,
    symbols: Sequence[str],
    coordinates_angstrom,
    definitions: Sequence[CurvilinearIsotopologueDefinition],
    *,
    source: str = "TRINITY nonredundant SONIC F2/F3 and isotope-specific GF",
) -> tuple[CurvilinearDeltaBVibResult, ...]:
    """End-to-end isotope correction from one field and chemical substitutions."""

    requested = tuple(definitions)
    if not requested:
        raise ValueError("at least one isotopologue definition is required")
    states = tuple(
        curvilinear_isotopologue_state_from_internal_field(
            field,
            b_matrix,
            isotopic_masses_amu(tuple(symbols), dict(definition.substitutions)),
            definition,
        )
        for definition in requested
    )
    return curvilinear_deltabvib_for_isotopologues(
        field,
        b_matrix,
        b_prime,
        coordinates_angstrom,
        states,
        source=source,
    )


def curvilinear_deltabvib_to_dict(result: CurvilinearDeltaBVibResult) -> dict[str, object]:
    """Serialize the versioned, mode-resolved result without losing provenance."""

    return {
        "schema": result.schema,
        "label": result.label,
        "substitutions": {str(key): value for key, value in result.substitutions.items()},
        "frequencies_cm1": list(result.frequencies_cm1),
        "representation": result.representation,
        "source": result.source,
        "excluded_modes": list(result.excluded_modes),
        "alpha_components_MHz": {
            "metric": result.alpha.metric_MHz.tolist(),
            "coriolis": result.alpha.coriolis_MHz.tolist(),
            "potential": result.alpha.potential_MHz.tolist(),
            "total": result.alpha.total_MHz.tolist(),
        },
        "delta_MHz": list(result.delta_MHz),
    }


def curvilinear_deltabvib_from_dict(payload: Mapping[str, object]) -> CurvilinearDeltaBVibResult:
    if payload.get("schema") != CURVILINEAR_DELTABVIB_SCHEMA:
        raise ValueError("unsupported curvilinear DeltaBvib schema")
    components = payload.get("alpha_components_MHz")
    if not isinstance(components, Mapping):
        raise ValueError("curvilinear DeltaBvib payload lacks alpha components")
    alpha = CurvilinearAlphaComponents(
        metric_MHz=np.asarray(components["metric"], dtype=float),
        coriolis_MHz=np.asarray(components["coriolis"], dtype=float),
        potential_MHz=np.asarray(components["potential"], dtype=float),
    )
    substitutions = payload.get("substitutions", {})
    if not isinstance(substitutions, Mapping):
        raise ValueError("curvilinear isotope substitutions must be a mapping")
    stored_delta = tuple(float(value) for value in payload.get("delta_MHz", ()))
    if len(stored_delta) != 3:
        raise ValueError("curvilinear DeltaBvib payload requires three axis values")
    frequencies = tuple(float(value) for value in payload.get("frequencies_cm1", ()))
    contracted = curvilinear_deltabvib_from_alpha(
        str(payload.get("label", "")),
        {int(key): int(value) for key, value in substitutions.items()},
        frequencies,
        alpha,
        representation=str(payload.get("representation", "")),
        source=str(payload.get("source", "")),
        exclude_modes=tuple(int(value) for value in payload.get("excluded_modes", ())),
    )
    if not np.allclose(stored_delta, contracted.delta_MHz, rtol=1.0e-10, atol=1.0e-8):
        raise ValueError("serialized DeltaBvib disagrees with its mode-resolved alpha components")
    return contracted


def read_curvilinear_deltabvib_results(path: Path | str) -> tuple[CurvilinearDeltaBVibResult, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "curvilinear" in payload:
        rows = payload["curvilinear"]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = [payload]
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("curvilinear DeltaBvib file must contain an object or a list of objects")
    return tuple(curvilinear_deltabvib_from_dict(row) for row in rows)


def write_curvilinear_deltabvib_results(
    path: Path | str,
    results: Sequence[CurvilinearDeltaBVibResult],
) -> Path:
    """Write the versioned TRINITY artifact consumed by MORPHEUS."""

    rows = tuple(results)
    if not rows:
        raise ValueError("at least one curvilinear DeltaBvib result is required")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([curvilinear_deltabvib_to_dict(row) for row in rows], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target


def write_curvilinear_deltabvib_to_xyzin(
    path: Path | str,
    results: Sequence[CurvilinearDeltaBVibResult],
) -> Path:
    """Persist TRINITY corrections in the canonical XYZin isotopologue records."""

    from matrix_chem.isotopologues import (
        read_xyzin_isotopologue_records,
        write_xyzin_isotopologue_records,
    )

    target = Path(path)
    records = read_xyzin_isotopologue_records(target)
    rows = tuple(results)
    by_label = {row.label: row for row in rows}
    if not by_label:
        raise ValueError("at least one curvilinear DeltaBvib result is required")
    if len(by_label) != len(rows):
        raise ValueError("duplicate TRINITY isotopologue labels")
    known = {record.label for record in records}
    if not set(by_label).issubset(known):
        raise ValueError("TRINITY DeltaBvib labels are absent from XYZin #ISOTOPOLOGUES")
    updated = []
    for record in records:
        result = by_label.get(record.label)
        if result is None:
            updated.append(record)
            continue
        if dict(record.substitutions) != dict(result.substitutions):
            raise ValueError(f"isotope substitutions disagree for {record.label}")
        updated.append(
            replace(
                record,
                deltavib_MHz=tuple(float(value) for value in result.delta_MHz),
                deltavib_source=(
                    f"curvilinear SONIC DeltaBvib ({result.representation}); {result.source}"
                ),
                deltavib_convention="subtract",
            )
        )
    return write_xyzin_isotopologue_records(target, tuple(updated))


def _mode_resolved_alpha_components(
    frequencies_cm1: np.ndarray,
    cubic_qmw: np.ndarray,
    inertia_principal_amu_bohr2: np.ndarray,
    inertia_derivatives_amu_sqrt_ang: np.ndarray,
    coriolis: Mapping[str, np.ndarray],
) -> CurvilinearAlphaComponents:
    frequencies = np.asarray(frequencies_cm1, dtype=float)
    cubic = np.asarray(cubic_qmw, dtype=float)
    inertia = np.asarray(inertia_principal_amu_bohr2, dtype=float)
    didq = np.asarray(inertia_derivatives_amu_sqrt_ang, dtype=float)
    nvib = frequencies.size
    if cubic.shape not in {(nvib, nvib, nvib), (nvib, nvib)}:
        raise ValueError("target-isotope cubic field dimensions disagree with GF modes")
    if inertia.shape != (3, 3) or didq.shape != (6, nvib):
        raise ValueError("isotopologue inertia data have incompatible dimensions")
    tensors = np.zeros((3, 3, nvib), dtype=float)
    tensors[0, 0], tensors[1, 1], tensors[2, 2] = didq[:3]
    tensors[0, 1] = tensors[1, 0] = didq[3]
    tensors[0, 2] = tensors[2, 0] = didq[4]
    tensors[1, 2] = tensors[2, 1] = didq[5]
    tensors /= _ANG_PER_BOHR
    beq_MHz = np.asarray(
        [_rotational_constant_mhz(float(inertia[index, index])) for index in range(3)]
    )
    beq_cm = beq_MHz / (_CLIGHT_CM_S * 1.0e-6)
    axes = ("x", "y", "z")
    zeta = {axis: np.asarray(coriolis[axis], dtype=float) for axis in axes}
    if any(value.shape != (nvib, nvib) for value in zeta.values()):
        raise ValueError("isotopologue Coriolis matrices have incompatible dimensions")

    metric = np.zeros((nvib, 3), dtype=float)
    coriolis_rows = np.zeros((nvib, 3), dtype=float)
    potential = np.zeros((nvib, 3), dtype=float)
    for tau in range(3):
        if not np.isfinite(beq_cm[tau]) or abs(beq_cm[tau]) < 1.0e-14:
            continue
        scale = -(beq_cm[tau] ** 2) * _CLIGHT_CM_S * 1.0e-6
        for i, wi in enumerate(frequencies):
            raw_metric = 0.0
            for eta in range(3):
                moment = inertia[eta, eta]
                if abs(moment) >= 1.0e-14:
                    raw_metric -= 3.0 * tensors[tau, eta, i] ** 2 / (4.0 * wi * moment)
            raw_coriolis = 0.0
            raw_potential = 0.0
            for j, wj in enumerate(frequencies):
                raw_coriolis += (
                    zeta[axes[tau]][i, j] ** 2
                    * (wi - wj) ** 2
                    / (2.0 * wi * wj * (wi + wj))
                )
                reduced = _mass_weighted_cubic_to_dimensionless(
                    (cubic[j, i, i] if cubic.ndim == 3 else cubic[j, i]),
                    frequencies, (i, i, j)
                )
                raw_potential -= (
                    pi
                    * sqrt(_CLIGHT_CM_S / _PLANCK_AU)
                    * tensors[tau, tau, j]
                    * reduced
                    / (wj * sqrt(wj))
                )
            # Delta Bvib = 1/2 sum_i alpha_i.  Store mode-resolved alpha rows.
            metric[i, tau] = 2.0 * scale * raw_metric
            coriolis_rows[i, tau] = 2.0 * scale * raw_coriolis
            potential[i, tau] = 2.0 * scale * raw_potential
    return CurvilinearAlphaComponents(metric, coriolis_rows, potential)


def _mass_weighted_cubic_to_dimensionless(
    value: float,
    frequencies_cm1: np.ndarray,
    modes: tuple[int, int, int],
) -> float:
    hbar_au = _PLANCK_AU / (2.0 * pi)
    reduced = float(value)
    for mode in modes:
        reduced *= sqrt(
            hbar_au / (2.0 * pi * _CLIGHT_CM_S * float(frequencies_cm1[mode]))
        )
    reduced *= _J_PER_HARTREE
    return reduced / (_PLANCK_J_S * _CLIGHT_CM_S)


def _rotational_constant_mhz(moment_amu_bohr2: float) -> float:
    if abs(moment_amu_bohr2) < 1.0e-14:
        return 0.0
    return float(_PLANCK_AU / (8.0 * pi**2 * moment_amu_bohr2) * 1.0e-6)


__all__ = [
    "CURVILINEAR_DELTABVIB_SCHEMA",
    "CARTESIAN_GRADIENT_DELTABVIB_METHOD",
    "CARTESIAN_HESSIAN_DELTABVIB_METHOD",
    "CallableDeltaBVibJob",
    "CurvilinearAlphaComponents",
    "CurvilinearDeltaBVibJob",
    "CurvilinearDeltaBVibResult",
    "CurvilinearIsotopologueDefinition",
    "CurvilinearIsotopologueState",
    "SONIC_GRADIENT_DELTABVIB_METHOD",
    "SONIC_HESSIAN_DELTABVIB_METHOD",
    "SONIC_INTERNAL_DELTABVIB_METHOD",
    "TrinityDeltaBVibMethod",
    "TrinityDeltaBVibService",
    "curvilinear_alpha_from_internal_field",
    "curvilinear_qcent_from_sonic_state",
    "curvilinear_deltabvib_from_alpha",
    "curvilinear_deltabvib_from_internal_field",
    "curvilinear_deltabvib_from_semidiagonal_cubic",
    "curvilinear_deltabvib_for_isotopologues",
    "curvilinear_deltabvib_from_definitions",
    "curvilinear_isotopologue_state_from_internal_field",
    "curvilinear_deltabvib_from_dict",
    "curvilinear_deltabvib_to_dict",
    "read_curvilinear_deltabvib_results",
    "write_curvilinear_deltabvib_results",
    "write_curvilinear_deltabvib_to_xyzin",
]
