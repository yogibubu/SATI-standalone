"""Integrated autonomous GFN2 energy, properties, and gradient session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from functools import lru_cache
from importlib.resources import files
import json
import os
import sys
import threading
from time import perf_counter
from typing import Sequence

import numpy as np
from matrix_numerics import best_diagonalizer_backend

from ._array_cache import triangular_indices
from .parameter_profiles import GFN2OnsiteProfile
from .gaussian_scc import GaussianSCCProfile, wrap_scc_interaction
from .alpb import (
    ALPBParameters,
    alpb_fixed_charge_energy_gradient_hartree,
    alpb_scc_interaction_matrix_hartree,
)
from .alpb_hessian import (
    ALPBHessianOperator,
    alpb_geometry_hessian_columns_hartree,
    alpb_mixed_geometry_charge_hartree,
)

from .basis import (
    SparseAtomicMultipoleIntegrals,
    atomic_multipole_integral_derivatives,
    contracted_multipole_exponent_derivatives,
    contracted_multipole_exponent_center_derivatives,
    contracted_multipole_exponent_second_derivatives,
    prepare_native_integral_context,
    sparse_atomic_multipole_integrals,
)
from .dispersion import (
    D4Parameters,
    D4PreparedGeometry,
    D4Result,
    SparseD4Result,
    atm_three_body_energy_charge_derivative_dense,
    atm_three_body_charge_hessian_dense,
    atm_three_body_energy_gradient_dense,
    atm_three_body_energy_gradient_sparse,
    atm_three_body_energy_sparse,
    d4_coefficient_direction,
    d4_global_parameter_partials_dense,
    d4_global_parameter_charge_potential_direction_dense,
    d4_global_head_adjoint_direction_dense,
    d4_charge_direction,
    d4_charge_hessian,
    d4_charge_hessian_matvec,
    d4_charge_potential_radial_direction,
    d4_charge_potential_geometry_direction_dense,
    d4_fock_sparse,
    d4_two_body,
    d4_two_body_sparse,
    _packed_d4_reference_data,
    prepare_d4_geometry,
    prepare_d4_geometry_sparse,
)
from .d4_reference import load_d4_reference_models
from .electronic_state import (
    GFN2OrbitalBasis,
    GFN2OrbitalState,
    GFN2SparseProjectorState,
)
from .density_solver import (
    DensityBuildResult, HybridDensitySolver, SCFSolverPolicy,
    SparseDensityBuildResult,
)
from .density_solver import sparse_closed_shell_density
from .localized_solver import SparseSymmetricMatrix, sparse_newton_schulz_inverse_sqrt
from .sparse_diis import SparseDIIS
from .sparse_lmo import (
    build_sparse_lewis_orbital_seed, sparse_local_orbital_sweep,
    sparse_projector_from_local_orbitals, oracle_carbonyl_pi_bonds,
)
from .sparse_observables import LazyAtomicPairMatrix, sparse_mayer_bond_orders
from .hamiltonian import (
    AtomParameters,
    OrbitalParameters,
    SparseHamiltonianDerivatives,
    coordination_numbers,
    extended_hueckel_hamiltonian,
    extended_hueckel_hamiltonian_sparse,
    extended_hueckel_hamiltonian_derivative,
    extended_hueckel_hamiltonian_derivatives_sparse,
    repulsion_energy,
    repulsion_energy_sparse,
)
from .gradients import (
    VariationalGradientTerms,
    anisotropic_explicit_gradient,
    anisotropic_explicit_hessian_vector_product_sparse,
    assemble_variational_gradient,
    assemble_variational_gradient_sparse,
    atm_three_body_gradient,
    atm_three_body_gradient_sparse,
    atm_three_body_hessian_vector_product_sparse,
    atomic_multipole_derivatives,
    coordination_number_jacobian,
    coordination_number_directional_derivative_sparse,
    coordination_response_gradient_sparse,
    d4_two_body_hessian_vector_product_sparse,
    d4_coordination_number_jacobian,
    energy_weighted_density,
    energy_weighted_density_from_fock,
    energy_weighted_density_from_fock_sparse,
    frozen_charge_scc_gradient,
    frozen_d4_two_body_gradient,
    frozen_d4_two_body_gradient_sparse,
    repulsion_gradient,
    repulsion_gradient_sparse,
    shell_charge_derivatives,
)
from .fragment_guess import covalent_components
from .molecular_basis import MolecularBasis, build_molecular_basis, relocate_molecular_basis
from .locality import (
    LocalityPolicy, build_localized_domains, enclosing_pca_ellipsoid,
    localized_domains_from_atom_neighbors,
)
from .kernels import batched_d4_coordination
from .kernels import d4_coordination_sparse, gfn2_coordination_sparse
from .multipole import (
    AnisotropicGeometry,
    AnisotropicParameters,
    AtomicMultipoles,
    anisotropic_energy_and_moment_potentials,
    anisotropic_fock_sparse,
    anisotropic_fock_derivatives_sparse,
    anisotropic_fock_derivatives_directional_sparse,
    anisotropic_fock_directional_sparse,
    anisotropic_moment_potential_direction,
    cumulative_atomic_multipole_directional_derivative_sparse,
    cumulative_atomic_multipoles_sparse,
    dense_scc_fock,
    dense_scc_observables,
    prepare_anisotropic_geometry,
    prepare_anisotropic_geometry_sparse,
    prepare_cumulative_multipole_operators,
    prepare_pair_cumulative_multipole_operators,
    prepare_sparse_cumulative_multipole_operators,
    sparse_orthogonal_observables,
)
from .parameters import (
    ELEMENT_SYMBOLS, GFN2ElementParameters, GFN2ParameterSet, load_gfn2_parameters,
)
from .nn_xtb import NNXTBOverlay, _MATRIX_SHIFT_FACTORS
from .nn_xtb_parameters import (
    analytic_aes_head_adjoints,
    analytic_aes_parameter_moment_potential_direction,
    analytic_aes_head_adjoint_directional,
    analytic_h0_head_adjoints, analytic_repulsion_head_adjoints,
    analytic_rcov_head_adjoints,
    analytic_rcov_coordination_jacobian,
    analytic_scc_hardness_head_adjoints, analytic_scc_parameter_fock_direction,
    analytic_scc_head_adjoint_directional,
    analytic_third_order_head_adjoints,
    build_nn_parameter_state,
)


from .scc import (
    ShellPartition,
    electronic_entropy_energy,
    fermi_occupations,
    monopole_energy,
    monopole_fock_sparse,
    shell_interaction_operator,
    shell_charges,
    shell_charges_sparse,
    solve_generalized_symmetric,
    solve_generalized_symmetric_direct,
    spin_fermi_occupations,
    symmetric_orthogonalizer,
)
from .telemetry import GFN2Telemetry, SolverTelemetry, TELEMETRY_SCHEMA
from .symmetry import (
    DEFAULT_SYMMETRY_SMALL_SYSTEM_HOST_THRESHOLD,
    FrozenPointGroup,
    SymmetryAdaptedBasis,
    compile_symmetry_adapted_basis,
)


@lru_cache(maxsize=16384)
def _nn_exponent_integral_response(left, right, order):
    """Reuse exact immutable AO derivatives across NN Hessian directions.

    The frozen Gaussian keys include centers, exponents and contractions;
    displaced validation points consequently cannot reuse a stale integral.
    """
    routines = {
        "first": contracted_multipole_exponent_derivatives,
        "second": contracted_multipole_exponent_second_derivatives,
        "mixed": contracted_multipole_exponent_center_derivatives,
    }
    result = routines[order](left, right)
    for value in result:
        value.setflags(write=False)
    return result


def _peak_rss_bytes() -> int:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return 0


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_EXTREME_NATIVE_MAX_ORBITALS = int(
    os.environ.get("MATRIX_GFN2_EXTREME_MAX_ORBITALS", "34")
)
_QUAD_UPPER = np.triu_indices(3)


@dataclass(frozen=True)
class GFN2Result:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    charges_electron: np.ndarray
    bond_orders: np.ndarray | LazyAtomicPairMatrix
    density: np.ndarray | SparseSymmetricMatrix
    iterations: int
    converged: bool
    point_charge_gradient_hartree_per_bohr: np.ndarray
    hirshfeld_charges_electron: np.ndarray | None
    cm5_charges_electron: np.ndarray | None
    density_solver: str = "diagonalization"
    telemetry: GFN2Telemetry | None = None
    orbital_state: GFN2OrbitalState | None = None
    sparse_projector_state: GFN2SparseProjectorState | None = None


def _tblite_mixer_rms(
    input_shell: np.ndarray,
    input_multipoles,
    output_shell: np.ndarray,
    output_multipoles,
) -> float:
    """RMS fixed-point change in the TBLite GFN2 multipole mixer vector."""

    components = np.concatenate((
        np.asarray(output_shell - input_shell).ravel(),
        np.asarray(
            output_multipoles.dipoles_bohr - input_multipoles.dipoles_bohr
        ).ravel(),
        np.asarray(
            output_multipoles.quadrupoles_bohr2[:, _QUAD_UPPER[0], _QUAD_UPPER[1]]
            - input_multipoles.quadrupoles_bohr2[:, _QUAD_UPPER[0], _QUAD_UPPER[1]]
        ).ravel(),
    ))
    return float(np.linalg.norm(components) / np.sqrt(components.size))


@dataclass(frozen=True)
class GFN2EnergyGradientResult:
    """Memory-minimal result for batched Hessian and ML evaluations."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    iterations: int
    converged: bool
    density_solver: str


@dataclass(frozen=True)
class NNXTBSemiAnalyticHessian:
    """NN Hessian with explicit mixed response and directional SCC probes.

    Network Jacobians/curvature and fixed-parameter forces are analytic.
    The fixed-parameter and electronic parameter-response derivatives in this
    assembly are directional differences, so this is not a fully analytic
    coupled NN Hessian.
    """

    hessian_hartree_per_bohr2: np.ndarray
    fixed_parameter_hartree_per_bohr2: np.ndarray
    parameter_geometry_hartree_per_bohr2: np.ndarray
    head_adjoint_response_hartree_per_bohr2: np.ndarray
    network_curvature_hartree_per_bohr2: np.ndarray
    reciprocity_error_hartree_per_bohr2: float
    chain_rule_error_hartree_per_bohr2: float


@dataclass(frozen=True)
class GFN2SFTDAEnergyGradientResult:
    """Ground and spin-flip excited-state energies and analytic gradients."""

    ground_state: GFN2Result
    sf_tda_result: object
    z_vector_result: object
    excitation_energies_hartree: np.ndarray
    total_energies_hartree: np.ndarray
    excitation_gradients_hartree_per_bohr: np.ndarray
    total_gradients_hartree_per_bohr: np.ndarray

    def __post_init__(self) -> None:
        excitation = np.array(self.excitation_energies_hartree, dtype=float, copy=True)
        total = np.array(self.total_energies_hartree, dtype=float, copy=True)
        excitation_gradient = np.array(
            self.excitation_gradients_hartree_per_bohr, dtype=float, copy=True
        )
        total_gradient = np.array(
            self.total_gradients_hartree_per_bohr, dtype=float, copy=True
        )
        if excitation.ndim != 1 or total.shape != excitation.shape:
            raise ValueError("SF-TDA energy dimensions are inconsistent")
        expected = (len(excitation), *self.ground_state.gradient_hartree_per_bohr.shape)
        if excitation_gradient.shape != expected or total_gradient.shape != expected:
            raise ValueError("SF-TDA gradient dimensions are inconsistent")
        if not all(
            np.all(np.isfinite(item))
            for item in (excitation, total, excitation_gradient, total_gradient)
        ):
            raise ValueError("SF-TDA energies and gradients must be finite")
        for item in (excitation, total, excitation_gradient, total_gradient):
            item.setflags(write=False)
        object.__setattr__(self, "excitation_energies_hartree", excitation)
        object.__setattr__(self, "total_energies_hartree", total)
        object.__setattr__(
            self, "excitation_gradients_hartree_per_bohr", excitation_gradient
        )
        object.__setattr__(self, "total_gradients_hartree_per_bohr", total_gradient)


@dataclass(frozen=True)
class GFN2SFTDAInterstateResult:
    """A complete molecular SF-TDA manifold and one interstate coupling."""

    manifold: GFN2SFTDAEnergyGradientResult
    state_i: int
    state_j: int
    interstate_numerator_hartree_per_bohr: np.ndarray
    derivative_coupling_per_bohr: np.ndarray | None
    energy_gap_hartree: float
    gap_tolerance_hartree: float

    def __post_init__(self) -> None:
        numerator = np.array(
            self.interstate_numerator_hartree_per_bohr, dtype=float, copy=True
        )
        coupling = self.derivative_coupling_per_bohr
        if numerator.shape != self.manifold.ground_state.gradient_hartree_per_bohr.shape:
            raise ValueError("SF-TDA interstate numerator has the wrong molecular shape")
        if not np.all(np.isfinite(numerator)):
            raise ValueError("SF-TDA interstate numerator must be finite")
        if coupling is not None:
            coupling = np.array(coupling, dtype=float, copy=True)
            if coupling.shape != numerator.shape or not np.all(np.isfinite(coupling)):
                raise ValueError("SF-TDA derivative coupling has the wrong molecular shape")
            coupling.setflags(write=False)
        if self.state_i == self.state_j:
            raise ValueError("SF-TDA interstate result requires distinct states")
        numerator.setflags(write=False)
        object.__setattr__(self, "interstate_numerator_hartree_per_bohr", numerator)
        object.__setattr__(self, "derivative_coupling_per_bohr", coupling)


def _energy_gradient_result(result: GFN2Result) -> GFN2EnergyGradientResult:
    return GFN2EnergyGradientResult(
        result.energy_hartree,
        result.gradient_hartree_per_bohr,
        result.iterations,
        result.converged,
        result.density_solver,
    )


@dataclass(frozen=True)
class _OrbitalMetadata:
    atom_index: int
    shell: str
    slater_exponent: float


@dataclass(frozen=True)
class _PointChargeEmbedding:
    hamiltonian: np.ndarray
    hamiltonian_derivative_qm: np.ndarray | None
    hamiltonian_derivative_point_charges: np.ndarray
    atomic_potential: np.ndarray
    atomic_potential_derivative_qm: np.ndarray
    atomic_potential_derivative_point_charges: np.ndarray


@dataclass(frozen=True)
class _SparseSCFModel:
    basis: MolecularBasis
    domains: object
    integrals: object
    overlap: SparseSymmetricMatrix
    core: SparseSymmetricMatrix
    orthogonalizer: SparseSymmetricMatrix
    partition: ShellPartition
    atom_parameters: dict[int, AtomParameters]
    orbital_parameters: dict[tuple[int, str], OrbitalParameters]
    metadata: tuple[_OrbitalMetadata, ...]
    coordination: np.ndarray


def _point_charge_embedding(
    overlap: np.ndarray,
    overlap_derivative: np.ndarray | None,
    orbital_atom: Sequence[int],
    coordinates_bohr: np.ndarray,
    point_coordinates_bohr: np.ndarray,
    point_charges_electron: np.ndarray,
) -> _PointChargeEmbedding:
    """Mulliken-monopole electrostatic embedding and coordinate derivatives."""

    coordinates = np.asarray(coordinates_bohr, dtype=float)
    point_coordinates = np.asarray(point_coordinates_bohr, dtype=float)
    charges = np.asarray(point_charges_electron, dtype=float)
    assignment = np.asarray(tuple(orbital_atom), dtype=np.intp)
    natoms = len(coordinates)
    npoints = len(charges)
    potential = np.zeros(natoms)
    atomic_derivative = np.zeros((natoms, 3))
    point_derivative = np.zeros((npoints, natoms, 3))
    if npoints:
        displacement = coordinates[:, None, :] - point_coordinates[None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        if np.any(distance < 1.0e-8):
            raise ValueError("point charge coincides with a QM atom")
        inverse = 1.0 / distance
        potential = inverse @ charges
        point_derivative = (
            charges[None, :, None] * displacement / distance[:, :, None] ** 3
        ).transpose(1, 0, 2)
        atomic_derivative = -np.sum(point_derivative, axis=0)
    orbital_potential = potential[assignment]
    potential_sum = orbital_potential[:, None] + orbital_potential[None, :]
    hamiltonian = -0.5 * overlap * potential_sum
    orbital_point_derivative = point_derivative[:, assignment, :]
    point_hamiltonian_derivative = -0.5 * overlap[None, None, :, :] * np.moveaxis(
        orbital_point_derivative[:, :, None, :]
        + orbital_point_derivative[:, None, :, :],
        -1,
        1,
    )
    qm_hamiltonian_derivative = None
    if overlap_derivative is not None:
        qm_hamiltonian_derivative = -0.5 * overlap_derivative * potential_sum
        selected = np.eye(natoms, dtype=float)[:, assignment]
        orbital_atomic_derivative = (
            selected[:, :, None] * atomic_derivative[:, None, :]
        )
        qm_hamiltonian_derivative -= 0.5 * overlap[None, None, :, :] * np.moveaxis(
            orbital_atomic_derivative[:, :, None, :]
            + orbital_atomic_derivative[:, None, :, :],
            -1,
            1,
        )
    return _PointChargeEmbedding(
        hamiltonian,
        qm_hamiltonian_derivative,
        point_hamiltonian_derivative,
        potential,
        atomic_derivative,
        point_derivative,
    )


@lru_cache(maxsize=1)
def _quadrature_weights() -> tuple[float, ...]:
    frequencies = np.asarray(
        (1e-6, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 1.2,
         1.4, 1.6, 1.8, 2., 2.5, 3., 4., 5., 7.5, 10.)
    )
    weights = np.empty_like(frequencies)
    weights[0] = 0.5 * (frequencies[1] - frequencies[0])
    weights[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
    weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
    return tuple(map(float, weights))


@lru_cache(maxsize=1)
def _covalent_radii() -> np.ndarray:
    resource = files("matrix_gfn2").joinpath("data/covalent-radii.json")
    return np.asarray(json.loads(resource.read_text())["radii_bohr"], dtype=float)


@lru_cache(maxsize=1)
def _atomic_radii() -> np.ndarray:
    resource = files("matrix_gfn2").joinpath("data/atomic-radii.json")
    return np.asarray(json.loads(resource.read_text())["radii_bohr"], dtype=float)


def _d4_coordination(
    numbers: Sequence[int], coordinates: np.ndarray, radii: np.ndarray, en: np.ndarray
) -> np.ndarray:
    atom_radii = np.asarray([radii[number - 1] for number in numbers], dtype=float)
    return batched_d4_coordination(coordinates, atom_radii, en)


def _available_memory_budget_bytes() -> int:
    """Return the conservative memory budget for concurrent GFN2 workers."""

    configured = os.environ.get("MATRIX_MEMORY_BUDGET_GB")
    if configured is not None:
        try:
            value = float(configured)
        except ValueError as error:
            raise ValueError("MATRIX_MEMORY_BUDGET_GB must be numeric") from error
        if value <= 0.0:
            raise ValueError("MATRIX_MEMORY_BUDGET_GB must be positive")
        return int(value * (1024**3))
    try:
        physical = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        physical = 8 * 1024**3
    return int(0.50 * physical)


def _dense_route_memory_budget_bytes() -> int:
    available = _available_memory_budget_bytes()
    if os.environ.get("MATRIX_MEMORY_BUDGET_GB") is not None:
        return available
    return min(available, 16 * 1024**3)


def _integral_route_memory_bytes(norbitals: int, pair_count: int) -> tuple[int, int]:
    """Conservative peak working-set estimates for dense and local pair integrals."""

    dense_pairs = norbitals * (norbitals + 1) // 2
    return 1024 * dense_pairs, 1024 * pair_count + 768 * 1024**2


def _memory_limited_worker_count(requested: int, natoms: int) -> int:
    """Cap independent workers using the measured sparse working-set model."""

    per_worker = int((768.0 + 0.80 * natoms) * 1024**2)
    affordable = max(1, _available_memory_budget_bytes() // max(per_worker, 1))
    if natoms >= 1000:
        affordable = min(affordable, 4)
    return max(1, min(int(requested), int(affordable)))


_HESSIAN_PROCESS_OPERATOR = None


def _initialize_hessian_process(calculator_options, coordinates, response_options):
    """Give each process its own SCF state and matrix-free response operator."""

    global _HESSIAN_PROCESS_OPERATOR
    os.environ["MATRIX_GFN2_NATIVE_THREADS"] = "1"
    if calculator_options.get("alpb_solvent") is not None:
        # Each process owns an ALPB reverse-mode graph.  PyTorch's default
        # intra-op pool would otherwise multiply the worker count again.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        import torch
        torch.set_num_threads(1)
    calculator = GFN2Calculator(**calculator_options)
    result = calculator.evaluate(
        np.asarray(coordinates, dtype=float),
        compute_cm5=False,
        compute_bond_orders=False,
        retain_orbitals=True,
    )
    if result.orbital_state is None:
        raise RuntimeError("parallel analytic Hessian requires retained orbitals")
    _HESSIAN_PROCESS_OPERATOR = calculator.ground_state_hessian_operator(
        result.orbital_state, **response_options,
    )


def _hessian_process_column(coordinate):
    operator = _HESSIAN_PROCESS_OPERATOR
    if operator is None:
        raise RuntimeError("analytic Hessian process was not initialized")
    direction = np.zeros(operator.dimension)
    direction[coordinate] = 1.0
    explicit, response, residual = operator.components_matvec(direction)
    return coordinate, explicit, response, residual


def _hessian_process_columns_alpb(coordinates):
    """Share one ALPB forward-AD graph across several Hessian columns."""

    operator = _HESSIAN_PROCESS_OPERATOR
    if operator is None:
        raise RuntimeError("analytic Hessian process was not initialized")
    calculator = operator.calculator
    context = operator.context
    indices = np.asarray(coordinates, dtype=int)
    geometry_block = alpb_geometry_hessian_columns_hartree(
        np.asarray(context["coordinates"], dtype=float),
        np.asarray(calculator.atomic_numbers),
        np.asarray(context["kernel"].reference_multipoles.charges),
        calculator.alpb_parameters,
        indices,
    )
    results = []
    for local, coordinate in enumerate(indices):
        direction = np.zeros(operator.dimension)
        direction[coordinate] = 1.0
        explicit, response, residual = operator.components_matvec(
            direction,
            alpb_geometry_hvp=geometry_block[:, local].reshape(-1, 3),
        )
        results.append((coordinate, explicit, response, residual))
    return results


class GFN2Calculator:
    def __init__(
        self,
        atoms: Sequence[str],
        *,
        charge: int = 0,
        multiplicity: int = 1,
        temperature_kelvin: float = 300.0,
        scf_energy_tolerance_hartree: float | None = None,
        scf_wavefunction_tolerance_electron: float = 1.0e-6,
        scf_mixer_tolerance: float = 2.0e-5,
        scf_convergence_mode: str | None = None,
        maximum_scf_iterations: int | None = None,
        point_charges_electron: Sequence[float] | None = None,
        point_charge_coordinates_angstrom: np.ndarray | None = None,
        diagonalizer_backend: str | None = None,
        density_solver: str = "auto",
        dense_sp2_min_orbitals: int = 256,
        sparse_sp2_min_orbitals: int = 3000,
        sparse_sp2_maximum_fill_fraction: float = 0.25,
        orthogonalizer_method: str = "auto",
        symmetry_contract: FrozenPointGroup | None = None,
        symmetry_cache_capacity: int = 4,
        symmetry_small_system_host_threshold: int = DEFAULT_SYMMETRY_SMALL_SYSTEM_HOST_THRESHOLD,
        nn_model_path: str | os.PathLike[str] | None = None,
        nn_enabled: bool = False,
        onsite_profile: GFN2OnsiteProfile | None = None,
        electrostatic_profile: GaussianSCCProfile | None = None,
        alpb_solvent: str | None = None,
        device: str | None = None,
        mps_min_orbitals: int | None = None,
    ) -> None:
        if device is not None and device not in {"cpu", "mps", "auto"}:
            raise ValueError("GFN2 device must be cpu, mps or auto")
        if mps_min_orbitals is not None and int(mps_min_orbitals) < 1:
            raise ValueError("MPS crossover dimension must be positive")
        self.device = device
        self.mps_min_orbitals = mps_min_orbitals
        lookup = {symbol: index for index, symbol in enumerate(ELEMENT_SYMBOLS, 1)}
        try:
            self.atomic_numbers = tuple(lookup[str(atom)] for atom in atoms)
        except KeyError as exc:
            raise ValueError(f"unsupported GFN2 element: {exc.args[0]}") from exc
        self.atoms = tuple(atoms)
        if onsite_profile is not None and not isinstance(onsite_profile, GFN2OnsiteProfile):
            raise TypeError("onsite_profile must be a GFN2OnsiteProfile")
        if onsite_profile is not None:
            if any(i >= len(self.atoms) for i, _ in onsite_profile.atom_shifts_ev):
                raise ValueError("onsite profile refers to an atom outside the molecule")
            if not onsite_profile.is_identity and (nn_enabled or symmetry_contract is not None):
                raise ValueError("nonidentity onsite profiles require the native unsymmetrized model")
        self.onsite_profile = onsite_profile
        if electrostatic_profile is not None:
            if not isinstance(electrostatic_profile, GaussianSCCProfile):
                raise TypeError("electrostatic_profile must be a GaussianSCCProfile")
            if len(electrostatic_profile.atom_widths_bohr) != len(self.atoms):
                raise ValueError("Gaussian SCC profile atom count does not match the molecule")
            if not electrostatic_profile.is_identity and (nn_enabled or symmetry_contract is not None):
                raise ValueError("Gaussian SCC requires the native unsymmetrized model")
        self.electrostatic_profile = electrostatic_profile
        self._gaussian_scc_active = electrostatic_profile is not None and not electrostatic_profile.is_identity
        self.alpb_parameters = (
            None if alpb_solvent is None else ALPBParameters.gfn2(alpb_solvent)
        )
        self.alpb_solvent = (
            None if self.alpb_parameters is None else self.alpb_parameters.solvent
        )
        self.nn_enabled = bool(nn_enabled)
        self.nn_overlay = None
        if nn_model_path is not None:
            self.nn_overlay = NNXTBOverlay(nn_model_path)
        if self.nn_enabled and self.nn_overlay is None:
            raise ValueError("nn_enabled=True requires nn_model_path")
        if self.nn_overlay is not None and not self.nn_enabled:
            raise ValueError("nn_model_path requires nn_enabled=True")
        if self.nn_enabled and (symmetry_contract is not None or point_charges_electron is not None):
            raise ValueError("NN-xTB currently requires a molecular system without symmetry or embedding")
        if self.nn_enabled and self.alpb_parameters is not None:
            raise ValueError("NN-xTB with ALPB is not yet qualified")
        if symmetry_contract is not None:
            symmetry_contract.validate()
            if symmetry_contract.atom_symbols != self.atoms:
                raise ValueError("frozen symmetry contract atom frame differs from MATRIX-GFN2")
        self.symmetry_contract = symmetry_contract
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.temperature_kelvin = float(temperature_kelvin)
        self.scf_convergence_mode = (
            "tblite" if self.multiplicity == 1 else "density"
        ) if scf_convergence_mode is None else str(scf_convergence_mode).strip().lower()
        self.scf_energy_tolerance_hartree = float(
            (1.0e-6 if self.scf_convergence_mode == "tblite" else 1.0e-8)
            if scf_energy_tolerance_hartree is None else scf_energy_tolerance_hartree
        )
        self.scf_wavefunction_tolerance_electron = float(
            scf_wavefunction_tolerance_electron
        )
        self.scf_mixer_tolerance = float(scf_mixer_tolerance)
        self.maximum_scf_iterations = int(
            (250 if self.scf_convergence_mode == "tblite" else 500)
            if maximum_scf_iterations is None else maximum_scf_iterations
        )
        if self.scf_convergence_mode not in {"density", "tblite"}:
            raise ValueError("SCF convergence mode must be 'density' or 'tblite'")
        if self.scf_convergence_mode == "tblite" and self.multiplicity != 1:
            raise ValueError("TBLite-like convergence currently requires a closed shell")
        if self.scf_convergence_mode == "density" and self.multiplicity == 1:
            raise ValueError("closed-shell GFN2 always uses the TBLite SCF criterion")
        if not np.isfinite(self.scf_energy_tolerance_hartree) or self.scf_energy_tolerance_hartree <= 0.0:
            raise ValueError("SCF energy tolerance must be positive")
        if not np.isfinite(self.scf_wavefunction_tolerance_electron) or self.scf_wavefunction_tolerance_electron <= 0.0:
            raise ValueError("SCF wavefunction tolerance must be positive")
        if not np.isfinite(self.scf_mixer_tolerance) or self.scf_mixer_tolerance <= 0.0:
            raise ValueError("SCF mixer tolerance must be positive")
        if self.maximum_scf_iterations < 1:
            raise ValueError("maximum SCF iterations must be positive")
        if (point_charges_electron is None) != (
            point_charge_coordinates_angstrom is None
        ):
            raise ValueError("point-charge values and coordinates must be provided together")
        self.point_charges_electron = np.asarray(
            () if point_charges_electron is None else point_charges_electron,
            dtype=float,
        )
        point_coordinates = np.asarray(
            np.empty((0, 3))
            if point_charge_coordinates_angstrom is None
            else point_charge_coordinates_angstrom,
            dtype=float,
        )
        if point_coordinates.shape != (len(self.point_charges_electron), 3):
            raise ValueError("point-charge coordinates must have shape ncharges x 3")
        if not np.all(np.isfinite(self.point_charges_electron)) or not np.all(
            np.isfinite(point_coordinates)
        ):
            raise ValueError("point-charge embedding must be finite")
        self.point_charge_coordinates_bohr = point_coordinates * ANGSTROM_TO_BOHR
        self.point_charges_electron.setflags(write=False)
        self.point_charge_coordinates_bohr.setflags(write=False)
        self.diagonalizer_backend = (
            None if diagonalizer_backend is None else str(diagonalizer_backend)
        )
        self.solver_policy = SCFSolverPolicy(
            mode=density_solver,
            dense_sp2_min_orbitals=int(dense_sp2_min_orbitals),
            sparse_sp2_min_orbitals=int(sparse_sp2_min_orbitals),
            sparse_sp2_maximum_fill_fraction=float(
                sparse_sp2_maximum_fill_fraction
            ),
        )
        self.orthogonalizer_method = str(orthogonalizer_method)
        self.symmetry_cache_capacity = int(symmetry_cache_capacity)
        self.symmetry_small_system_host_threshold = int(symmetry_small_system_host_threshold)
        if self.symmetry_cache_capacity < 1:
            raise ValueError("symmetry cache capacity must be positive")
        if self.symmetry_small_system_host_threshold < 0:
            raise ValueError("symmetry small-system threshold must be non-negative")
        self.parameters = load_gfn2_parameters()
        # Geometry-independent tables are prepared once per calculator.  In
        # particular, do not slice the universal D4 database or rebuild the
        # element arrays on every SCF point (this is visible for small
        # molecules where the electronic kernel itself is only a few ms).
        self._element_values = tuple(
            self.parameters.elements[number] for number in self.atomic_numbers
        )
        self._d4_models = tuple(
            load_d4_reference_models()[number - 1]
            for number in self.atomic_numbers
        )
        self._d4_parameters = D4Parameters(_quadrature_weights())
        self._valence = tuple(
            sum(shell.reference_occupation for shell in element.shells)
            for element in self._element_values
        )
        self._electronegativities = np.asarray(
            [element.electronegativity for element in self._element_values],
            dtype=float,
        )
        self._base_multipole_radii = np.asarray(
            [element.multipole_radius_bohr for element in self._element_values],
            dtype=float,
        )
        self._valence_coordination = np.asarray(
            [element.multipole_valence_coordination for element in self._element_values],
            dtype=float,
        )
        for shared_table in (
            self._electronegativities,
            self._base_multipole_radii,
            self._valence_coordination,
        ):
            shared_table.setflags(write=False)
        self._basis_template = build_molecular_basis(
            self.atomic_numbers,
            np.zeros((len(self.atomic_numbers), 3), dtype=float),
            self.parameters,
        )
        self._minimum_primitive_exponent = min(
            primitive.exponent
            for orbital in self._basis_template.cartesian_orbitals
            for primitive in orbital.primitives
        )
        self._cartesian_atom_indices = np.asarray(
            [item.atom_index for item in self._basis_template.cartesian_orbitals],
            dtype=np.intp,
        )
        self._cartesian_atom_indices.setflags(write=False)
        covalent_radii = _covalent_radii()
        atomic_radii = _atomic_radii()
        self._atom_parameters = {
            number: AtomParameters(
                self.parameters.elements[number].repulsion_exponent,
                self.parameters.elements[number].effective_nuclear_charge,
                covalent_radii[number - 1],
                2.0 * atomic_radii[number - 1],
                self.parameters.elements[number].electronegativity,
            )
            for number in set(self.atomic_numbers)
        }
        self._atomic_number_array = np.asarray(self.atomic_numbers, dtype=np.int64)
        self._coordination_radii = covalent_radii[self._atomic_number_array - 1]
        self._repulsion_alpha = np.asarray(
            [self._atom_parameters[number].alpha for number in self.atomic_numbers],
            dtype=float,
        )
        self._effective_nuclear_charges = np.asarray(
            [
                self._atom_parameters[number].effective_nuclear_charge
                for number in self.atomic_numbers
            ],
            dtype=float,
        )
        for shared_table in (
            self._atomic_number_array,
            self._coordination_radii,
            self._repulsion_alpha,
            self._effective_nuclear_charges,
        ):
            shared_table.setflags(write=False)
        metadata: list[_OrbitalMetadata] = []
        orbital_parameters: dict[tuple[int, str], OrbitalParameters] = {}
        for atom, shell in zip(
            self._basis_template.spherical_atom,
            self._basis_template.spherical_shell,
            strict=True,
        ):
            item = next(
                candidate
                for candidate in self.parameters.elements[self.atomic_numbers[atom]].shells
                if candidate.label == shell
            )
            metadata.append(_OrbitalMetadata(atom, shell, item.slater_exponent))
            orbital_parameters[(self.atomic_numbers[atom], shell)] = OrbitalParameters(
                item.level_hartree,
                item.coordination_shift_hartree,
                item.polynomial,
            )
        self._orbital_metadata = tuple(metadata)
        self._orbital_parameters = orbital_parameters
        if onsite_profile is not None and not onsite_profile.is_identity:
            self._orbital_parameters = onsite_profile.apply(
                self.atomic_numbers, orbital_parameters,
            )
        self._partition = _shell_partition(
            self._basis_template, self.atomic_numbers, self.parameters
        )
        if self._gaussian_scc_active:
            self._partition = replace(self._partition, electrostatic_profile=electrostatic_profile)
        (
            d4_reference_coordination,
            d4_gaussian_count,
            d4_reference_charges,
            d4_references,
            d4_mask,
        ) = _packed_d4_reference_data(self._d4_models)
        self._small_native_scf_configuration = {
            "coordination_radii": self._coordination_radii,
            "orbital_atom": np.asarray(
                self._basis_template.spherical_atom, dtype=np.int64
            ),
            "cartesian_atom": np.asarray(
                [
                    orbital.atom_index
                    for orbital in self._basis_template.cartesian_orbitals
                ],
                dtype=np.int64,
            ),
            "orbital_shell": np.asarray(
                self._partition.orbital_shell, dtype=np.int64
            ),
            "reference_population": np.asarray(
                self._partition.reference_population, dtype=float
            ),
            "valence": self._valence,
            "atomic_numbers": self._atomic_number_array,
            "repulsion_alpha": self._repulsion_alpha,
            "effective_nuclear_charges": self._effective_nuclear_charges,
            "d4_coordination_radii": np.asarray(
                _covalent_radii()[self._atomic_number_array - 1], dtype=float
            ),
            "shell_atom": np.asarray(self._partition.shell_atom, dtype=np.int64),
            "shell_hardness": np.asarray(
                self._partition.hardness_hartree, dtype=float
            ),
            "third_order": np.asarray(
                self._partition.third_order_hartree, dtype=float
            ),
            "anis_base_radius": self._base_multipole_radii,
            "anis_valence_coordination": self._valence_coordination,
            "anis_dipole_xc": np.asarray(
                [item.dipole_xc for item in self._element_values], dtype=float
            ),
            "anis_quadrupole_xc": np.asarray(
                [item.quadrupole_xc for item in self._element_values], dtype=float
            ),
            "d4_radii": self._coordination_radii,
            "electronegativity": self._electronegativities,
            "d4_reference_coordination": d4_reference_coordination,
            "d4_gaussian_count": d4_gaussian_count,
            "d4_reference_charges": d4_reference_charges,
            "d4_references": d4_references,
            "d4_mask": d4_mask,
            "d4_effective": np.asarray(
                [model.effective_nuclear_charge for model in self._d4_models],
                dtype=float,
            ),
            "d4_hardness": np.asarray(
                [model.hardness for model in self._d4_models], dtype=float
            ),
            "d4_r2r4": np.asarray(
                [model.r2r4 for model in self._d4_models], dtype=float
            ),
            "d4_frequency_weights": np.asarray(
                self._d4_parameters.frequency_weights, dtype=float
            ),
            "anis_a3": 3.0,
            "anis_a5": 4.0,
            "s6": self._d4_parameters.s6,
            "s8": self._d4_parameters.s8,
            "d4_a1": self._d4_parameters.a1,
            "d4_a2": self._d4_parameters.a2,
            "s9": self._d4_parameters.s9,
        }
        self._symmetry_reference_coefficients: np.ndarray | None = None
        self._symmetry_basis_cache: OrderedDict[tuple[str, bytes, bytes, tuple[int, ...]], SymmetryAdaptedBasis] = OrderedDict()
        self._symmetry_cache_hits = 0
        self._symmetry_cache_misses = 0
        self._symmetry_cache_evictions = 0
        self._previous_density: np.ndarray | None = None
        self._older_density: np.ndarray | None = None
        self._previous_orthogonal_density: np.ndarray | None = None
        self._previous_coordinates_bohr: np.ndarray | None = None
        self._older_orthogonal_density: np.ndarray | None = None
        self._older_coordinates_bohr: np.ndarray | None = None
        self._previous_orbital_energies: np.ndarray | None = None
        self._previous_occupations: np.ndarray | None = None
        self._restart_fixed_point_contraction = 0.0
        self._previous_sparse_projector = None
        self._previous_sparse_projector_state = None
        self._previous_sparse_domains = None
        self._previous_sparse_coordinates_bohr: np.ndarray | None = None
        self._last_result_key: tuple[bytes, bool, bool] | None = None
        self._last_result: GFN2Result | None = None
        self._post_scf_pair_integrals = None
        self._post_scf_pair_domains = None
        self._localized_domain_cache: dict[
            tuple[tuple[int, ...], float, int], tuple[object, np.ndarray, float]
        ] = {}
        self._native_integral_contexts: dict[int, object] = {}
        self._active_native_context = None
        self._mps_context = None
        self._active_coordination = None
        self._active_fused_dense_context = False
        self._empty_bond_orders = np.empty((0, 0), dtype=float)
        self._empty_point_charge_gradient = np.empty((0, 3), dtype=float)
        # Batch clones share this bounded gate only for allocation-heavy
        # boundary phases.  Their SCF kernels remain fully concurrent.
        self._sparse_allocation_gate: threading.Semaphore | None = None
        configured_native_threads = os.environ.get("MATRIX_GFN2_NATIVE_THREADS", "1")
        try:
            self._native_thread_limit = max(1, int(configured_native_threads))
        except ValueError as error:
            raise ValueError("MATRIX_GFN2_NATIVE_THREADS must be an integer") from error

    def _batch_worker_clone(self) -> GFN2Calculator:
        """Share immutable tables while isolating all mutable SCF state."""

        worker = copy(self)
        worker._symmetry_reference_coefficients = None
        worker._symmetry_basis_cache = OrderedDict()
        worker._symmetry_cache_hits = 0
        worker._symmetry_cache_misses = 0
        worker._symmetry_cache_evictions = 0
        worker._previous_density = None
        worker._older_density = None
        worker._previous_orthogonal_density = None
        worker._previous_coordinates_bohr = None
        worker._older_orthogonal_density = None
        worker._older_coordinates_bohr = None
        worker._previous_orbital_energies = None
        worker._previous_occupations = None
        worker._restart_fixed_point_contraction = 0.0
        worker._previous_sparse_projector = None
        worker._previous_sparse_projector_state = None
        worker._previous_sparse_domains = None
        worker._previous_sparse_coordinates_bohr = None
        worker._last_result_key = None
        worker._last_result = None
        worker._localized_domain_cache = {}
        worker._native_integral_contexts = {}
        worker._active_native_context = None
        worker._mps_context = None
        worker._active_coordination = None
        worker._active_fused_dense_context = False
        worker._native_thread_limit = 1
        return worker

    def _hessian_process_options(self) -> dict:
        """Constructor inputs sufficient to reproduce an independent SCF state."""

        return {
            "atoms": self.atoms,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "temperature_kelvin": self.temperature_kelvin,
            "scf_energy_tolerance_hartree": self.scf_energy_tolerance_hartree,
            "scf_wavefunction_tolerance_electron": self.scf_wavefunction_tolerance_electron,
            "scf_mixer_tolerance": self.scf_mixer_tolerance,
            "scf_convergence_mode": self.scf_convergence_mode,
            "maximum_scf_iterations": self.maximum_scf_iterations,
            "point_charges_electron": self.point_charges_electron,
            "point_charge_coordinates_angstrom": (
                self.point_charge_coordinates_bohr / ANGSTROM_TO_BOHR
            ),
            "diagonalizer_backend": self.diagonalizer_backend,
            "density_solver": self.solver_policy.mode,
            "dense_sp2_min_orbitals": self.solver_policy.dense_sp2_min_orbitals,
            "sparse_sp2_min_orbitals": self.solver_policy.sparse_sp2_min_orbitals,
            "sparse_sp2_maximum_fill_fraction": (
                self.solver_policy.sparse_sp2_maximum_fill_fraction
            ),
            "orthogonalizer_method": self.orthogonalizer_method,
            "symmetry_contract": self.symmetry_contract,
            "symmetry_cache_capacity": self.symmetry_cache_capacity,
            "symmetry_small_system_host_threshold": (
                self.symmetry_small_system_host_threshold
            ),
            "alpb_solvent": self.alpb_solvent,
            "onsite_profile": self.onsite_profile,
            "electrostatic_profile": self.electrostatic_profile,
        }

    def _localized_domains(
        self,
        coordinates: np.ndarray,
        orbital_atom: Sequence[int],
        policy: LocalityPolicy,
    ):
        """Reuse an unchanged Verlet topology with a rigorous cutoff guard."""

        assignment = tuple(int(value) for value in orbital_atom)
        key = (
            assignment,
            float(policy.interaction_cutoff_bohr),
            int(policy.buffer_hops),
        )
        cached = self._localized_domain_cache.get(key)
        if cached is not None:
            domains, reference, boundary_margin = cached
            native_context = self._active_native_context
            if (
                native_context is not None
                and native_context.geometry_within_displacement(
                    coordinates, reference, 0.5 * boundary_margin
                )
            ):
                return domains
            maximum_displacement = (
                float("inf")
                if native_context is not None
                else float(
                    np.max(
                        np.linalg.norm(coordinates - reference, axis=1),
                        initial=0.0,
                    )
                )
            )
            # Any interatomic distance changes by at most twice the maximum
            # Cartesian displacement.  Below the stored nearest-boundary
            # margin the exact neighbor graph, and therefore every buffered
            # orbital pair, is unchanged.
            if 2.0 * maximum_displacement < boundary_margin:
                return domains
        domains = build_localized_domains(coordinates, assignment, policy)
        if cached is not None:
            cached_domains = cached[0]
            cached_left, cached_right = cached_domains.allowed_pair_arrays()
            new_left, new_right = domains.allowed_pair_arrays()
            if (
                np.array_equal(cached_left, new_left)
                and np.array_equal(cached_right, new_right)
            ):
                # The exact topology is unchanged.  Preserve object identity
                # so every persistent sparse/native context can be reused,
                # even when the O(N^2) boundary-margin audit is intentionally
                # disabled for very large systems.
                domains = cached_domains
        boundary_margin = 0.0
        if len(coordinates) <= 512:
            displacement = coordinates[:, None, :] - coordinates[None, :, :]
            distance = np.linalg.norm(displacement, axis=2)
            off_diagonal = ~np.eye(len(coordinates), dtype=bool)
            boundary_margin = float(
                np.min(
                    np.abs(distance[off_diagonal] - policy.interaction_cutoff_bohr),
                    initial=np.inf,
                )
            )
        self._localized_domain_cache[key] = (
            domains,
            np.asarray(coordinates, dtype=float).copy(),
            boundary_margin,
        )
        return domains

    def _model_nn_development(self, coordinates: np.ndarray):
        """Build the per-atom dense native NN Hamiltonian and integral state.

        This route deliberately avoids the resident C++ H0/SCF tables, which
        are keyed by element rather than atom. Parameter-response derivatives
        are contracted separately by the analytic NN gradient/Hessian path.
        """
        overlay = self._nn_development_overlay
        state = build_nn_parameter_state(
            self.atomic_numbers, self.parameters,
            overlay.matrix_parameter_shifts(
                self.atomic_numbers, coordinates,
                charge=float(self.charge), unpaired=float(self.multiplicity - 1),
            ),
        )
        self._active_nn_state = state
        self._element_values = state.elements
        self._base_multipole_radii = np.asarray(
            [value.multipole_radius_bohr for value in state.elements], dtype=float,
        )
        self._valence_coordination = np.asarray(
            [value.multipole_valence_coordination for value in state.elements], dtype=float,
        )
        self._d4_parameters = state.d4_parameters(self._d4_parameters)
        basis = build_molecular_basis(
            self.atomic_numbers, coordinates, self.parameters,
            per_atom_elements=state.basis_elements,
        )
        self._basis_template = basis
        self._orbital_metadata = tuple(
            _OrbitalMetadata(atom, shell, state.elements[atom].shells[
                next(index for index, item in enumerate(state.elements[atom].shells)
                     if item.label == shell)
            ].slater_exponent)
            for atom, shell in zip(
                basis.spherical_atom, basis.spherical_shell, strict=True,
            )
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=state.basis_elements,
            per_atom_third_scales=state.third_scales,
        )
        self._partition = partition
        policy = LocalityPolicy(
            interaction_cutoff_bohr=30.0,
            buffer_hops=0,
            truncation_threshold=1.0e-15,
        )
        spherical_domains = self._localized_domains(
            coordinates, basis.spherical_atom, policy,
        )
        cartesian_domains = self._localized_domains(
            coordinates,
            tuple(item.atom_index for item in basis.cartesian_orbitals),
            policy,
        )
        cartesian = sparse_atomic_multipole_integrals(
            basis.cartesian_orbitals,
            *cartesian_domains.allowed_pair_arrays(),
            threshold=0.0, native_threads=self._native_thread_limit,
        )
        pair_integrals = basis.transform_sparse_multipoles(
            cartesian, spherical_domains, threshold=0.0,
        )
        self._post_scf_pair_integrals = pair_integrals
        self._post_scf_pair_domains = spherical_domains
        size = basis.norbitals
        overlap = np.zeros((size, size), dtype=float)
        rows, columns = pair_integrals.rows, pair_integrals.columns
        overlap[rows, columns] = pair_integrals.overlap
        overlap[columns, rows] = pair_integrals.overlap
        coordination = coordination_numbers(
            self.atomic_numbers, coordinates, state.atoms,
        )
        self._active_coordination = coordination
        self._active_native_context = None
        self._active_native_model_prepared = False
        core = extended_hueckel_hamiltonian(
            overlap, self._orbital_metadata, self.atomic_numbers, coordinates,
            state.atoms, state.orbitals, global_scalings=state.globals,
        )
        return (
            basis, overlap, None, None, core, partition,
            state.atoms, state.orbitals, None,
            {"embedding_mode": "none", "compiled_adapter": None},
        )

    def _model(
        self,
        coordinates: np.ndarray,
        *,
        relocate_basis: bool = True,
    ):
        if getattr(self, "_nn_development_overlay", None) is not None:
            return self._model_nn_development(coordinates)
        atom_parameters = self._atom_parameters
        # The persistent native context now has a lower prefactor than the
        # legacy Python Cartesian-integral/transform route even for the
        # smallest molecular bases.  Use one implementation throughout the
        # dense regime: besides removing the old crossover discontinuity this
        # keeps integral construction, Cartesian-to-spherical transformation,
        # core assembly, ISCF, and post-SCF contractions in the same resident
        # allocation context.
        use_fused_context = True
        basis = (
            relocate_molecular_basis(self._basis_template, coordinates)
            if relocate_basis or not use_fused_context
            else self._basis_template
        )
        # Screen only exponentially local AO products; long-range SCC and D4
        # remain untouched.  Scale the radius with the most diffuse primitive
        # so the Gaussian exponent-distance envelope is constant.  Ordinary
        # organic bases reach the measured 8.5-bohr fast path, whereas alkali
        # and other diffuse shells automatically retain as much as 20 bohr.
        ao_pair_cutoff_bohr = (
            max(
                8.5,
                min(
                    20.0,
                    8.5 * np.sqrt(
                        0.16614291148416 / self._minimum_primitive_exponent
                    ),
                ),
            )
            if self._basis_template.norbitals <= 512
            else 20.0
        )
        if self._basis_template.norbitals <= 114:
            # Complete AO pairs correct the measured small-molecule model
            # discrepancy at manageable cost. Up to 34 AOs they also enable
            # the monolithic resident geometry/SCF/gradient kernel.
            ao_pair_cutoff_bohr = max(
                ao_pair_cutoff_bohr,
                float(np.linalg.norm(np.ptp(coordinates, axis=0))) + 1.0,
            )
        experimental_ao_cutoff = os.environ.get(
            "MATRIX_EXPERIMENTAL_AO_PAIR_CUTOFF_BOHR"
        )
        if experimental_ao_cutoff is not None:
            candidate = float(experimental_ao_cutoff)
            if not np.isfinite(candidate) or candidate <= 0.0:
                raise ValueError("experimental AO pair cutoff must be positive")
            ao_pair_cutoff_bohr = candidate
        integral_policy = LocalityPolicy(
            interaction_cutoff_bohr=ao_pair_cutoff_bohr,
            buffer_hops=0,
            truncation_threshold=self.solver_policy.sparse_truncation_threshold,
        )
        spherical_domains = self._localized_domains(
            coordinates, basis.spherical_atom, integral_policy
        )
        object.__setattr__(spherical_domains, "native_threads", self._native_thread_limit)
        cartesian_domains = self._localized_domains(
            coordinates,
            tuple(item.atom_index for item in basis.cartesian_orbitals),
            integral_policy,
        )
        context_key = id(cartesian_domains)
        native_integral_context = self._native_integral_contexts.get(context_key)
        if native_integral_context is None:
            native_integral_context = prepare_native_integral_context(
                basis.cartesian_orbitals,
                *cartesian_domains.allowed_pair_arrays(),
            )
            native_integral_context.configure_core(
                self._orbital_metadata,
                self.atomic_numbers,
                self._atom_parameters,
                self._orbital_parameters,
            )
            self._native_integral_contexts[context_key] = native_integral_context
        self._active_native_context = native_integral_context
        self._active_fused_dense_context = use_fused_context
        spherical_rows, spherical_columns = spherical_domains.allowed_pair_arrays()
        dense_geometry_fused = bool(
            use_fused_context
            and len(spherical_rows) == basis.norbitals * (basis.norbitals + 1) // 2
        )
        if dense_geometry_fused:
            native_integral_context.configure_spherical(
                basis.spherical_cartesian_offsets,
                basis.spherical_cartesian_indices,
                basis.spherical_cartesian_coefficients,
                spherical_rows,
                spherical_columns,
            )
            (
                native_rows,
                native_columns,
                transformed,
                left_derivatives,
                right_derivatives,
                overlap,
                core,
                coordination,
            ) = native_integral_context.prepare_dense_geometry(
                coordinates[self._cartesian_atom_indices],
                coordinates,
                self._coordination_radii,
                np.asarray(basis.spherical_atom, dtype=np.int64),
                np.asarray(self._partition.orbital_shell, dtype=np.int64),
                np.asarray(self._partition.reference_population, dtype=float),
                np.asarray(self._valence, dtype=float),
                self._native_thread_limit,
            )
            pair_integrals = SparseAtomicMultipoleIntegrals(
                size=basis.norbitals,
                rows=np.asarray(native_rows, dtype=np.int32),
                columns=np.asarray(native_columns, dtype=np.int32),
                overlap=transformed[:, 0],
                dipole=transformed[:, 1:4],
                second=transformed[:, 4:].reshape(-1, 3, 3),
                left_derivatives=left_derivatives,
                right_derivatives=right_derivatives,
                threshold_discarded_frobenius_norm=0.0,
                domain_complete=True,
            )
        elif use_fused_context:
            pair_integrals = basis.evaluate_sparse_multipoles(
                native_integral_context,
                spherical_domains,
                threshold=0.0,
                native_threads=self._native_thread_limit,
                centers_bohr=coordinates[self._cartesian_atom_indices],
            )
        else:
            cartesian = sparse_atomic_multipole_integrals(
                basis.cartesian_orbitals,
                *cartesian_domains.allowed_pair_arrays(),
                threshold=0.0,
                native_threads=self._native_thread_limit,
            )
            pair_integrals = basis.transform_sparse_multipoles(
                cartesian, spherical_domains, threshold=0.0
            )
        size = basis.norbitals
        rows, columns = pair_integrals.rows, pair_integrals.columns
        off_diagonal = rows != columns
        upper, lower = rows[off_diagonal], columns[off_diagonal]
        if not dense_geometry_fused:
            overlap = np.zeros((size, size), dtype=float)
            overlap[rows, columns] = pair_integrals.overlap
            overlap[lower, upper] = overlap[upper, lower]
        if len(self.point_charges_electron):
            dipole = np.zeros((3, size, size), dtype=float)
            second = np.zeros((3, 3, size, size), dtype=float)
            dipole[:, rows, columns] = pair_integrals.dipole.T
            second[:, :, rows, columns] = pair_integrals.second.transpose(1, 2, 0)
            dipole[:, lower, upper] = dipole[:, upper, lower]
            second[:, :, lower, upper] = second[:, :, upper, lower]
        else:
            dipole = None
            second = None
        self._post_scf_pair_integrals = pair_integrals
        self._post_scf_pair_domains = spherical_domains
        metadata = self._orbital_metadata
        orbital_parameters = self._orbital_parameters
        if not dense_geometry_fused:
            coordination = coordination_numbers(
                self.atomic_numbers, coordinates, atom_parameters
            )
        self._active_coordination = coordination
        bare_core = core if dense_geometry_fused else (
            native_integral_context.build_core(overlap, coordinates, coordination)
            if use_fused_context
            else extended_hueckel_hamiltonian(
                overlap,
                metadata,
                self.atomic_numbers,
                coordinates,
                atom_parameters,
                orbital_parameters,
            )
        )
        embedding = None
        if len(self.point_charges_electron) or self.symmetry_contract is not None:
            embedding = _point_charge_embedding(
                overlap,
                None,
                basis.spherical_atom,
                coordinates,
                self.point_charge_coordinates_bohr,
                self.point_charges_electron,
            )
            core = bare_core + embedding.hamiltonian
        else:
            core = bare_core
        self._active_native_model_prepared = dense_geometry_fused and core is bare_core
        symmetry_adapter = None
        symmetry_diagnostics: dict[str, object] = {
            "embedding_mode": "none" if not len(self.point_charges_electron) else "full-matrix",
            "embedding_orbits": (),
            "embedding_residual": 0.0,
            "full_matrix_reason": None,
            "compiled_adapter": None,
        }
        if self.symmetry_contract is not None:
            cache_key = (
                self.symmetry_contract.digest,
                np.asarray(coordinates, dtype="<f8").tobytes(),
                np.asarray(overlap, dtype="<f8").tobytes(),
                tuple(overlap.shape),
            )
            symmetry_adapter = self._symmetry_basis_cache.get(cache_key)
            if symmetry_adapter is None:
                self._symmetry_cache_misses += 1
                symmetry_adapter = compile_symmetry_adapted_basis(
                    self.symmetry_contract,
                    basis,
                    overlap,
                    coordinates / ANGSTROM_TO_BOHR,
                )
                if symmetry_adapter is not None:
                    self._symmetry_basis_cache[cache_key] = symmetry_adapter
                    self._symmetry_basis_cache.move_to_end(cache_key)
                    if len(self._symmetry_basis_cache) > self.symmetry_cache_capacity:
                        self._symmetry_basis_cache.popitem(last=False)
                        self._symmetry_cache_evictions += 1
            else:
                self._symmetry_cache_hits += 1
                self._symmetry_basis_cache.move_to_end(cache_key)
        if symmetry_adapter is not None:
            assert embedding is not None
            symmetry_diagnostics["compiled_adapter"] = symmetry_adapter
            core_residual = max(
                float(np.max(np.abs(rep.T @ bare_core @ rep - bare_core), initial=0.0))
                for rep in symmetry_adapter.representations
            )
            if core_residual > 5.0e-5 * max(
                1.0, float(np.max(np.abs(bare_core), initial=0.0))
            ):
                raise RuntimeError(
                    "MATRIX-GFN2 intrinsic Hamiltonian breaks the frozen "
                    f"{self.symmetry_contract.point_group} point group"
                )
            embedding_residual = max(
                float(
                    np.max(
                        np.abs(rep.T @ embedding.hamiltonian @ rep - embedding.hamiltonian),
                        initial=0.0,
                    )
                )
                for rep in symmetry_adapter.representations
            )
            embedding_scale = max(
                1.0, float(np.max(np.abs(embedding.hamiltonian), initial=0.0))
            )
            point_orbits = symmetry_adapter.embedding_orbits(
                self.point_charge_coordinates_bohr / ANGSTROM_TO_BOHR,
                self.point_charges_electron,
                origin_angstrom=np.mean(coordinates, axis=0) / ANGSTROM_TO_BOHR,
            )
            symmetry_diagnostics["embedding_residual"] = embedding_residual
            symmetry_diagnostics["embedding_orbits"] = () if point_orbits is None else point_orbits
            if embedding_residual > 5.0e-5 * embedding_scale or point_orbits is None:
                symmetry_diagnostics["embedding_mode"] = "symmetry-breaking-full-matrix"
                symmetry_diagnostics["full_matrix_reason"] = (
                    "external embedding does not transform as the frozen solute point group"
                )
                symmetry_adapter = None
            else:
                symmetry_diagnostics["embedding_mode"] = (
                    "none" if not len(self.point_charges_electron) else "symmetry-adapted"
                )
                overlap = symmetry_adapter.symmetrize_matrix(overlap)
                core = symmetry_adapter.symmetrize_matrix(core)
        partition = self._partition
        return (
            basis,
            overlap,
            dipole,
            second,
            core,
            partition,
            atom_parameters,
            orbital_parameters,
            symmetry_adapter,
            symmetry_diagnostics,
        )

    def _model_sparse(
        self,
        coordinates: np.ndarray,
        domains=None,
        locality_policy: LocalityPolicy | None = None,
    ) -> _SparseSCFModel:
        """Build the localized SCF model without allocating an AO square."""

        covalent = _covalent_radii()
        atom_parameters = self._atom_parameters
        basis = relocate_molecular_basis(self._basis_template, coordinates)
        policy = locality_policy or LocalityPolicy(
            interaction_cutoff_bohr=20.0,
            buffer_hops=1,
            truncation_threshold=self.solver_policy.sparse_truncation_threshold
        )
        if domains is None:
            domains = self._localized_domains(
                coordinates, basis.spherical_atom, policy
            )
        object.__setattr__(domains, "native_threads", self._native_thread_limit)
        cartesian_assignment = tuple(
            item.atom_index for item in basis.cartesian_orbitals
        )
        cartesian_domains = (
            localized_domains_from_atom_neighbors(
                domains.atom_neighbors, cartesian_assignment,
            )
            if locality_policy is not None and domains is not None
            else self._localized_domains(
                coordinates, cartesian_assignment, policy,
            )
        )
        object.__setattr__(cartesian_domains, "native_threads", self._native_thread_limit)
        context_key = id(cartesian_domains)
        native_integral_context = self._native_integral_contexts.get(context_key)
        if native_integral_context is None:
            native_integral_context = prepare_native_integral_context(
                basis.cartesian_orbitals,
                *cartesian_domains.allowed_pair_arrays(),
            )
            self._native_integral_contexts[context_key] = native_integral_context
        # For a complete small/medium domain, truncation cannot reduce work
        # and only inflates the conservative Newton--Schulz error recursion.
        # Keep this algebra exact; true localized domains retain the audited
        # threshold and its propagated bound.
        local_threshold = (
            0.0
            if domains.fill_fraction == 1.0
            and cartesian_domains.fill_fraction == 1.0
            else self.solver_policy.sparse_truncation_threshold
        )
        integrals = basis.evaluate_sparse_multipoles(
            native_integral_context,
            domains,
            threshold=local_threshold,
            native_threads=self._native_thread_limit,
        )
        overlap = integrals.sparse_overlap()
        metadata = self._orbital_metadata
        orbital_parameters = self._orbital_parameters
        atomic_covalent = covalent[np.asarray(self.atomic_numbers) - 1]
        coordination, _ = gfn2_coordination_sparse(
            coordinates, atomic_covalent, cutoff_bohr=policy.interaction_cutoff_bohr
        )
        core = extended_hueckel_hamiltonian_sparse(
            overlap,
            metadata,
            self.atomic_numbers,
            coordinates,
            atom_parameters,
            orbital_parameters,
            coordination_override=coordination,
        )
        inverse = sparse_newton_schulz_inverse_sqrt(
            overlap,
            domains,
            tolerance=1.0e-8,
            truncation_threshold=local_threshold,
            input_error_bound=integrals.threshold_discarded_frobenius_norm,
        )
        if not inverse.converged:
            raise RuntimeError(
                "sparse overlap orthogonalization did not certify "
                f"(residual={inverse.residual:.3e}, bound={inverse.accumulated_error_bound:.3e})"
            )
        return _SparseSCFModel(
            basis,
            domains,
            integrals,
            overlap,
            core,
            inverse.inverse_sqrt,
            self._partition,
            atom_parameters,
            orbital_parameters,
            tuple(metadata),
            coordination,
        )

    def _single_point(
        self,
        coordinates_bohr: np.ndarray,
        *,
        compute_cm5: bool = False,
        compute_bond_orders: bool = True,
        density_policy: SCFSolverPolicy | None = None,
        retain_orbitals: bool = False,
        collect_telemetry: bool = True,
        energy_only: bool = False,
        nn_properties_only: bool = False,
        _mps_retry: bool = False,
    ) -> GFN2Result | float:
        total_started = perf_counter()
        self.last_accelerator_report = None
        mps_request = "cpu" if _mps_retry else (
            self.device or os.environ.get("MATRIX_GFN2_DEVICE", "cpu").strip().lower()
        )
        if mps_request not in {"cpu", "mps", "auto"}:
            raise ValueError("MATRIX_GFN2_DEVICE must be cpu, mps or auto")
        self._last_oracle_region_info = {"selected": False}
        self._last_oracle_region_state = ()
        self._last_oracle_fragment_proposals = ()
        self._last_fragment_orbital_state = ()
        self._last_fragment_scf_guess = {"selected": False}
        coordinates = np.asarray(coordinates_bohr, dtype=float)
        native_context = self._active_native_context
        extreme_orbitals = len(self._basis_template.spherical_atom)
        extreme_restart_compatible = bool(
            mps_request == "cpu"
            and native_context is not None
            and extreme_orbitals <= _EXTREME_NATIVE_MAX_ORBITALS
            and self._post_scf_pair_integrals is not None
            and self._post_scf_pair_integrals.domain_complete
            and self.multiplicity == 1
            and not compute_cm5
            and not compute_bond_orders
            and not retain_orbitals
            and not collect_telemetry
            and not len(self.point_charges_electron)
            and self.symmetry_contract is None
            and self._previous_density is not None
            and self._previous_density.shape
            == (extreme_orbitals, extreme_orbitals)
            and self._previous_coordinates_bohr is not None
            and self._previous_coordinates_bohr.shape == coordinates.shape
            and native_context.geometry_within_displacement(
                coordinates, self._previous_coordinates_bohr, 0.75
            )
        )
        if extreme_restart_compatible and not self._gaussian_scc_active:
            restart_density = self._previous_density
            if (
                self._older_density is not None
                and self._older_density.shape == restart_density.shape
                and self._older_coordinates_bohr is not None
                and self._older_coordinates_bohr.shape == coordinates.shape
            ):
                previous_step = (
                    self._previous_coordinates_bohr - self._older_coordinates_bohr
                ).ravel()
                current_step = (
                    coordinates - self._previous_coordinates_bohr
                ).ravel()
                denominator = float(previous_step @ previous_step)
                if denominator > np.finfo(float).tiny:
                    coefficient = float(current_step @ previous_step) / denominator
                    transverse = current_step - coefficient * previous_step
                    if (
                        -0.25 <= coefficient <= 2.0
                        and float(np.linalg.norm(transverse))
                        <= 0.25
                        * max(float(np.linalg.norm(current_step)), 1.0e-14)
                    ):
                        restart_density = restart_density + coefficient * (
                            restart_density - self._older_density
                        )
            cartesian_centers = coordinates[
                self._small_native_scf_configuration["cartesian_atom"]
            ]
            extreme = native_context.extreme_singlepoint(
                cartesian_centers,
                coordinates,
                self._small_native_scf_configuration,
                restart_density,
                electrons=int(
                    round(sum(self._partition.reference_population) - self.charge)
                ),
                multiplicity=self.multiplicity,
                temperature_kelvin=self.temperature_kelvin,
                wavefunction_tolerance=self.scf_wavefunction_tolerance_electron,
                energy_tolerance=self.scf_energy_tolerance_hartree,
                maximum_cycles=min(8, self.maximum_scf_iterations),
                mixer_tolerance=(
                    self.scf_mixer_tolerance
                    if self.scf_convergence_mode == "tblite" else -1.0
                ),
            )
            if extreme is not None:
                (
                    energy,
                    gradient,
                    charges,
                    density,
                    native_restart_density,
                    energies,
                    occupations,
                    iterations,
                    density_residual,
                    energy_residual,
                    mixer_residual,
                ) = extreme
                self._last_scf_trace = (
                    {
                        "cycle": int(iterations),
                        "density_residual": float(density_residual),
                        "energy_residual_hartree": float(energy_residual),
                        "mixer_residual_rms": float(mixer_residual),
                        "accepted_damping": 1.0,
                    },
                )
                self._older_density = self._previous_density
                self._previous_density = np.asarray(
                    native_restart_density, dtype=float
                ).copy()
                self._older_orthogonal_density = self._previous_orthogonal_density
                self._previous_orthogonal_density = None
                self._older_coordinates_bohr = self._previous_coordinates_bohr
                self._previous_coordinates_bohr = coordinates.copy()
                self._previous_orbital_energies = np.asarray(
                    energies, dtype=float
                ).copy()
                self._previous_occupations = np.asarray(
                    occupations, dtype=float
                ).copy()
                return GFN2Result(
                    float(energy),
                    np.asarray(gradient),
                    np.asarray(charges),
                    self._empty_bond_orders,
                    np.asarray(density),
                    int(iterations),
                    True,
                    self._empty_point_charge_gradient,
                    None,
                    None,
                    "diagonalization",
                    None,
                    None,
                )
        model_started = perf_counter()
        (
            basis,
            overlap,
            dipole,
            second,
            core,
            partition,
            atom_parameters,
            orbital_parameters,
            symmetry_adapter,
            symmetry_diagnostics,
        ) = self._model(
            coordinates,
            relocate_basis=bool(
                compute_cm5 or len(self.point_charges_electron)
            ),
        )
        model_wall_seconds = perf_counter() - model_started
        electrons = int(round(sum(partition.reference_population) - self.charge))
        initial_solver_started = perf_counter()
        oracle_seed = None
        oracle_plan = None
        if (
            os.environ.get("MATRIX_EXPERIMENTAL_ORACLE_LOCAL_GUESS") == "1"
            and self._previous_density is None
            and symmetry_adapter is None
            and self.charge == 0 and self.multiplicity == 1
        ):
            from matrix_oracle import perceive_orbital_regions
            from .oracle_region_guess import oracle_region_density_guess

            try:
                oracle_plan = perceive_orbital_regions(
                    coordinates / ANGSTROM_TO_BOHR,
                    np.asarray(self.atomic_numbers),
                )
                oracle_seed = oracle_region_density_guess(
                    core, overlap, np.asarray(basis.spherical_atom),
                    np.asarray(partition.orbital_shell),
                    np.asarray(partition.reference_population),
                    np.asarray(self.atomic_numbers), electrons,
                    self.temperature_kelvin, oracle_plan,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                self._last_oracle_region_info = {
                    "selected": False, "reason": str(exc),
                }
            else:
                self._last_oracle_region_state = oracle_seed[5]
                self._last_oracle_region_info = {
                    "selected": True,
                    **oracle_seed[4],
                    "topology_backend": oracle_plan.topology_backend,
                }
        restart_geometry_is_compatible = (
            symmetry_adapter is None
            and self._previous_density is not None
            and self._previous_density.shape == overlap.shape
            and self._previous_coordinates_bohr is not None
            and self._previous_coordinates_bohr.shape == coordinates.shape
            and self._active_native_context.geometry_within_displacement(
                coordinates, self._previous_coordinates_bohr, 0.75
            )
            and self._previous_orbital_energies is not None
            and self._previous_occupations is not None
        )
        # Long covalent systems benefit from the direct generalized LAPACK
        # driver even on the first geometry.  Keep compact systems and the
        # established small-molecule route on the orthogonalized solver.
        elongated_dense = bool(
            self.scf_convergence_mode == "tblite"
            and len(overlap) >= 128
            and enclosing_pca_ellipsoid(coordinates).aspect_ratio >= 2.0
        )
        cold_direct_eligible = elongated_dense and len(overlap) >= 600
        medium_covalent_chain = False
        if elongated_dense and len(overlap) < 600:
            medium_covalent_chain = len(covalent_components(
                self.atomic_numbers, coordinates, _covalent_radii()
            )) == 1
        direct_generalized = bool(
            (
                restart_geometry_is_compatible
                or oracle_seed is not None
                or cold_direct_eligible
                or (
                    os.environ.get("MATRIX_GFN2_EXPERIMENTAL_DIRECT_COLD") == "1"
                    and len(overlap) >= 128
                )
            )
            and symmetry_adapter is None
        )
        can_restart = bool(
            restart_geometry_is_compatible
            and (
                direct_generalized
                or (
                    self._previous_orthogonal_density is not None
                    and self._previous_orthogonal_density.shape == overlap.shape
                )
            )
        )
        diagonalizer_backend = self.diagonalizer_backend
        if (
            diagonalizer_backend is None
            and 128 <= len(overlap) < 600
            and os.environ.get("MATRIX_DIAGONALIZER_BACKEND", "auto").lower() == "auto"
        ):
            # GPU backend discovery imports accelerator runtimes on the
            # first small dense point, even though this route uses CPU LAPACK.
            diagonalizer_backend = "scipy"
        if symmetry_adapter is not None:
            adapted_orthogonalizer = np.zeros_like(symmetry_adapter.transformation)
            for block, overlap_block in zip(
                symmetry_adapter.blocks,
                symmetry_adapter.overlap_blocks,
                strict=True,
            ):
                block_orthogonalizer = symmetric_orthogonalizer(
                    overlap_block,
                    diagonalizer_backend=diagonalizer_backend,
                    method=self.orthogonalizer_method,
                )
                adapted_orthogonalizer[np.ix_(block, block)] = block_orthogonalizer
            orthogonalizer = (
                symmetry_adapter.transformation
                @ adapted_orthogonalizer
                @ symmetry_adapter.transformation.T
            )
        elif direct_generalized:
            # HybridDensitySolver does not use this placeholder in direct
            # generalized mode; avoiding S^-1/2 here is the intended resident
            # fast path.
            orthogonalizer = np.eye(len(overlap), dtype=float)
        else:
            orthogonalizer = symmetric_orthogonalizer(
                overlap,
                diagonalizer_backend=diagonalizer_backend,
                method=self.orthogonalizer_method,
            )
        if can_restart:
            energies = self._previous_orbital_energies.copy()
            occupations = self._previous_occupations.copy()
            restart_density = (
                self._previous_density
                if direct_generalized
                else self._previous_orthogonal_density
            )
            older_restart_candidate = (
                self._older_density
                if direct_generalized
                else self._older_orthogonal_density
            )
            if (
                older_restart_candidate is not None
                and older_restart_candidate.shape == overlap.shape
                and self._older_coordinates_bohr is not None
                and self._older_coordinates_bohr.shape == coordinates.shape
            ):
                previous_step = (
                    self._previous_coordinates_bohr - self._older_coordinates_bohr
                ).ravel()
                current_step = (coordinates - self._previous_coordinates_bohr).ravel()
                denominator = float(previous_step @ previous_step)
                if denominator > np.finfo(float).tiny:
                    coefficient = float(current_step @ previous_step) / denominator
                    transverse = current_step - coefficient * previous_step
                    alignment_error = float(np.linalg.norm(transverse))
                    if (
                        -0.25 <= coefficient <= 2.0
                        and alignment_error
                        <= 0.25 * max(float(np.linalg.norm(current_step)), 1.0e-14)
                    ):
                        previous_restart_density = (
                            self._previous_density
                            if direct_generalized
                            else self._previous_orthogonal_density
                        )
                        older_restart_density = (
                            self._older_density
                            if direct_generalized
                            else self._older_orthogonal_density
                        )
                        if older_restart_density is not None:
                            restart_density = (
                                previous_restart_density
                                + coefficient
                                * (
                                    previous_restart_density
                                    - older_restart_density
                                )
                            )
            restarted = (
                restart_density
                if direct_generalized
                else orthogonalizer @ restart_density @ orthogonalizer
            )
            density = 0.5 * (restarted + restarted.T)
            population = float(np.sum(density * overlap))
            if population > np.finfo(float).tiny:
                density *= electrons / population
        elif oracle_seed is not None:
            density, energies, coefficients, occupations = oracle_seed[:4]
        elif symmetry_adapter is None:
            if direct_generalized:
                energies, coefficients = solve_generalized_symmetric_direct(
                    core, overlap
                )
            else:
                energies, coefficients = solve_generalized_symmetric(
                    core,
                    overlap,
                    orthogonalizer=orthogonalizer,
                    diagonalizer_backend=diagonalizer_backend,
                )
            occupations = fermi_occupations(
                energies, electrons, multiplicity=self.multiplicity,
                temperature_kelvin=self.temperature_kelvin,
            )
            density = (coefficients * occupations) @ coefficients.T
        else:
            energies, coefficients = symmetry_adapter.solve(
                core,
                diagonalizer_backend=diagonalizer_backend,
                small_system_host_threshold=self.symmetry_small_system_host_threshold,
                reference_coefficients=self._symmetry_reference_coefficients,
            )
            self._symmetry_reference_coefficients = coefficients.copy()
            occupations = fermi_occupations(
                energies, electrons, multiplicity=self.multiplicity,
                temperature_kelvin=self.temperature_kelvin,
            )
            density = (coefficients * occupations) @ coefficients.T
        initial_solver_wall_seconds = perf_counter() - initial_solver_started
        active_density_policy = (
            self.solver_policy if density_policy is None else density_policy
        )
        domains = None
        if (
            active_density_policy.normalized_mode == "sparse-sp2"
            or len(overlap) >= active_density_policy.sparse_sp2_min_orbitals
        ):
            domains = self._localized_domains(
                coordinates,
                basis.spherical_atom,
                LocalityPolicy(),
            )
        density_builder = HybridDensitySolver(
            overlap=overlap,
            orthogonalizer=orthogonalizer,
            electrons=electrons,
            multiplicity=self.multiplicity,
            temperature_kelvin=self.temperature_kelvin,
            initial_energies=energies,
            initial_occupations=occupations,
            policy=active_density_policy,
            diagonalizer_backend=diagonalizer_backend,
            domains=domains,
            symmetry_adapter=symmetry_adapter,
            direct_generalized=direct_generalized,
            native_context=(
                self._active_native_context if direct_generalized else None
            ),
        )
        native_context = self._active_native_context
        assignment = basis.spherical_atom
        multipole_operators = None if getattr(
            self, "_active_native_model_prepared", False
        ) else (
            prepare_cumulative_multipole_operators(
                overlap, dipole, second, coordinates, assignment
            )
            if dipole is not None and second is not None
            else prepare_pair_cumulative_multipole_operators(
                self._post_scf_pair_integrals, coordinates, assignment
            )
        )
        valence = self._valence
        # The persistent electronic context is profitable even below the
        # integral/core crossover: small systems retain their specialized
        # model builder, then enter the same native resident ISCF library
        # path without forcing the slower generic integral setup.
        if native_context is not None and not getattr(
            self, "_active_native_model_prepared", False
        ):
            native_context.update_dense_model(
                core,
                overlap,
                multipole_operators,
                partition.orbital_shell,
                partition.reference_population,
                valence,
            )
        element_values = self._element_values
        coordination = self._active_coordination
        if coordination is None:
            raise RuntimeError("native GFN2 context lost its coordination state")
        base_multipole_radii = self._base_multipole_radii
        valence_coordination = self._valence_coordination
        d4_models = self._d4_models
        en = self._electronegativities
        d4_parameters = self._d4_parameters
        # The complete D4/ATM graph is both faster and substantially more
        # accurate while the electronic problem is still on its dense route.
        # Cross over only for larger systems, where packed local dispersion
        # avoids cubic ATM work and unbounded pair storage.
        sparse_dispersion = (
            len(coordinates) >= 64
            and getattr(self, "_active_nn_state", None) is None
        )
        small_native_scf = bool(
            native_context is not None and len(coordinates) < 21
            and self.alpb_parameters is None and not self._gaussian_scc_active
        )
        if small_native_scf:
            (
                interaction,
                effective_multipole_radii,
                anisotropic_geometry_values,
                d4_cn,
                d4_weights,
                d4_weight_derivatives,
                d4_rows,
                d4_columns,
                d4_ratio,
                d4_pair_factor,
            ) = native_context.prepare_small_scf(
                coordinates, coordination, self._small_native_scf_configuration
            )
            anisotropic_parameters = AnisotropicParameters(
                np.asarray(effective_multipole_radii),
                self._small_native_scf_configuration["anis_dipole_xc"],
                self._small_native_scf_configuration["anis_quadrupole_xc"],
            )
            anisotropic_geometry = AnisotropicGeometry(
                *(np.asarray(item) for item in anisotropic_geometry_values)
            )
            dense_ratio = np.zeros((len(coordinates), len(coordinates)))
            dense_pair_factor = np.zeros_like(dense_ratio)
            dense_ratio[d4_rows, d4_columns] = d4_ratio
            dense_ratio[d4_columns, d4_rows] = d4_ratio
            dense_pair_factor[d4_rows, d4_columns] = d4_pair_factor
            dense_pair_factor[d4_columns, d4_rows] = d4_pair_factor
            configuration = self._small_native_scf_configuration
            d4_prepared = D4PreparedGeometry(
                configuration["d4_frequency_weights"],
                np.asarray(d4_weights),
                np.asarray(d4_weight_derivatives),
                configuration["d4_reference_charges"],
                configuration["d4_references"],
                configuration["d4_mask"],
                configuration["d4_effective"],
                configuration["d4_hardness"],
                dense_ratio,
                dense_pair_factor,
            )
        else:
            nn_state = getattr(self, "_active_nn_state", None)
            interaction = shell_interaction_operator(
                coordinates, partition,
                alphaj=(2.0 if nn_state is None else nn_state.globals["alphaj"]),
            )
            if nn_state is None:
                effective_multipole_radii = base_multipole_radii + (
                    5.0 - base_multipole_radii
                ) / (
                    1.0
                    + np.exp(-4.0 * (coordination - valence_coordination - 1.2))
                )
                anisotropic_parameters = AnisotropicParameters(
                    effective_multipole_radii,
                    np.asarray([item.dipole_xc for item in element_values]),
                    np.asarray([item.quadrupole_xc for item in element_values]),
                )
            else:
                anisotropic_parameters = nn_state.anisotropic_parameters(coordination)
            anisotropic_geometry = (
                prepare_anisotropic_geometry_sparse(
                    coordinates, anisotropic_parameters, cutoff_bohr=64.0
                )
                if sparse_dispersion
                else prepare_anisotropic_geometry(
                    coordinates, anisotropic_parameters,
                ) if native_context is None else AnisotropicGeometry(
                    *native_context.anisotropic_geometry(
                        coordinates,
                        anisotropic_parameters.damping_radius_bohr,
                        anisotropic_parameters.a3,
                        anisotropic_parameters.a5,
                    )
                )
            )
        alpb_static_energy = 0.0
        if self.alpb_parameters is not None:
            if hasattr(interaction, "evaluate") and not isinstance(interaction, np.ndarray):
                raise ValueError(
                    "ALPB requires the dense SCC interaction route"
                )
            alpb_atomic_interaction, alpb_static_energy = (
                alpb_scc_interaction_matrix_hartree(
                    coordinates, np.asarray(self.atomic_numbers),
                    self.alpb_parameters,
                )
            )
            shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
            interaction = np.asarray(interaction, dtype=float) + (
                alpb_atomic_interaction[np.ix_(shell_atom, shell_atom)]
            )
        interaction = wrap_scc_interaction(interaction, coordinates, partition)
        if sparse_dispersion:
            d4_cn, _ = d4_coordination_sparse(
                coordinates,
                _covalent_radii()[np.asarray(self.atomic_numbers) - 1],
                en,
                cutoff_bohr=40.0,
            )
            d4_prepared = prepare_d4_geometry_sparse(
                coordinates, d4_cn, d4_models, d4_parameters,
                cutoff_bohr=50.0,
            )
        elif not small_native_scf:
            d4_cn = _d4_coordination(
                self.atomic_numbers, coordinates, _covalent_radii(), en
            )
            d4_prepared = prepare_d4_geometry(
                coordinates, d4_cn, d4_models, d4_parameters
            )
        native_scf_fused = bool(
            native_context is not None and isinstance(interaction, np.ndarray)
        )
        if native_scf_fused and not small_native_scf:
            if sparse_dispersion:
                d4_rows = np.asarray(d4_prepared.rows, dtype=np.int64)
                d4_columns = np.asarray(d4_prepared.columns, dtype=np.int64)
                d4_ratio = np.asarray(d4_prepared.ratio, dtype=float)
                d4_pair_factor = np.asarray(d4_prepared.pair_factor, dtype=float)
            else:
                d4_rows, d4_columns = triangular_indices(
                    len(coordinates), diagonal=False
                )
                d4_rows = np.asarray(d4_rows, dtype=np.int64)
                d4_columns = np.asarray(d4_columns, dtype=np.int64)
                d4_ratio = np.asarray(
                    d4_prepared.ratio[d4_rows, d4_columns], dtype=float
                )
                d4_pair_factor = np.asarray(
                    d4_prepared.pair_factor[d4_rows, d4_columns], dtype=float
                )
            native_context.configure_scf(
                {
                    "interaction": np.asarray(interaction, dtype=float),
                    "third_order": np.asarray(partition.third_order_hartree, dtype=float),
                    "anis_dipole_xc": np.asarray(anisotropic_parameters.dipole_xc, dtype=float),
                    "anis_quadrupole_xc": np.asarray(anisotropic_parameters.quadrupole_xc, dtype=float),
                    "anis_left": np.asarray(anisotropic_geometry.left, dtype=np.int64),
                    "anis_right": np.asarray(anisotropic_geometry.right, dtype=np.int64),
                    "anis_displacement": np.asarray(anisotropic_geometry.displacement, dtype=float),
                    "anis_f3": np.asarray(anisotropic_geometry.f3, dtype=float),
                    "anis_f5": np.asarray(anisotropic_geometry.f5, dtype=float),
                    "anis_distance2": np.asarray(anisotropic_geometry.distance2, dtype=float),
                    "d4_frequency_weights": np.asarray(d4_prepared.frequency_weights, dtype=float),
                    "d4_weights": np.asarray(d4_prepared.weights, dtype=float),
                    "d4_weight_derivatives": np.asarray(d4_prepared.weight_derivatives, dtype=float),
                    "d4_reference_charges": np.asarray(d4_prepared.reference_charges, dtype=float),
                    "d4_references": np.asarray(d4_prepared.references, dtype=float),
                    "d4_mask": np.asarray(d4_prepared.mask, dtype=float),
                    "d4_effective": np.asarray(d4_prepared.effective, dtype=float),
                    "d4_hardness": np.asarray(d4_prepared.hardness, dtype=float),
                    "d4_rows": d4_rows,
                    "d4_columns": d4_columns,
                    "d4_ratio": d4_ratio,
                    "d4_pair_factor": d4_pair_factor,
                }
            )
        def tblite_mixer_residual(
            input_density: np.ndarray, output_density: np.ndarray
        ) -> float:
            if native_scf_fused:
                # The Fock evaluation immediately preceding this fixed-point
                # step has already contracted the input density's multipoles.
                native_observables = native_context.last_scf_state()[0]
                input_shell = np.asarray(native_observables[0])
                input_multipoles = AtomicMultipoles(
                    np.asarray(native_observables[1]),
                    np.asarray(native_observables[2]),
                    np.asarray(native_observables[3]),
                )
            else:
                input_shell, input_multipoles = dense_scc_observables(
                    input_density, multipole_operators, partition.orbital_shell,
                    partition.reference_population, valence, native_context=native_context,
                )
            output_shell, output_multipoles = dense_scc_observables(
                output_density, multipole_operators, partition.orbital_shell,
                partition.reference_population, valence, native_context=native_context,
            )
            return _tblite_mixer_rms(
                input_shell, input_multipoles, output_shell, output_multipoles
            )

        converged = False
        fock_history: list[np.ndarray] = []
        error_history: list[np.ndarray] = []
        density_output_history: list[np.ndarray] = []
        density_error_history: list[np.ndarray] = []
        density_residuals: list[float] = []
        convergence_residuals: list[float] = []
        energy_residuals: list[float] = []
        accepted_dampings: list[float] = []
        sparse_diis = SparseDIIS(maximum_size=8) if domains is not None else None

        cached_scc_density: np.ndarray | None = None
        cached_scc_state = None
        cached_native_density: np.ndarray | None = None
        cached_native_state: tuple[np.ndarray, float] | None = None

        def native_state(candidate_density: np.ndarray) -> tuple[np.ndarray, float]:
            nonlocal cached_native_density, cached_native_state
            if candidate_density is cached_native_density and cached_native_state is not None:
                return cached_native_state
            if native_context is None:
                raise RuntimeError("native fused SCF context is unavailable")
            cached_native_density = candidate_density
            cached_native_state = native_context.scf_state_fock(candidate_density)
            return cached_native_state

        def scc_state(candidate_density: np.ndarray):
            nonlocal cached_scc_density, cached_scc_state
            if candidate_density is cached_scc_density:
                return cached_scc_state
            candidate_shell_q, candidate_multipoles = dense_scc_observables(
                candidate_density,
                multipole_operators,
                partition.orbital_shell,
                partition.reference_population,
                valence,
                native_context=native_context,
            )
            if hasattr(interaction, "evaluate"):
                scc_evaluation = interaction.evaluate(
                    candidate_shell_q, compute_gradient=False
                )
                quadratic_potential = scc_evaluation.potential
                quadratic_energy = scc_evaluation.energy_hartree
            else:
                quadratic_potential = interaction @ candidate_shell_q
                quadratic_energy = 0.5 * float(
                    candidate_shell_q @ quadratic_potential
                )
            shell_potential = quadratic_potential + np.asarray(
                partition.third_order_hartree
            ) * candidate_shell_q**2
            anisotropic_state = anisotropic_energy_and_moment_potentials(
                candidate_multipoles, coordinates, anisotropic_parameters,
                geometry=anisotropic_geometry,
            )
            candidate_d4 = (
                d4_two_body_sparse(
                    candidate_multipoles.charges, d4_prepared, d4_parameters,
                )
                if sparse_dispersion
                else d4_two_body(
                    coordinates, candidate_multipoles.charges, d4_cn,
                    d4_models, d4_parameters, prepared=d4_prepared,
                )
            )
            candidate_atm_energy = 0.0
            if getattr(self, "_active_nn_state", None) is not None and len(coordinates) >= 3:
                candidate_atm_energy, candidate_atm_charge = (
                    atm_three_body_energy_charge_derivative_dense(
                        coordinates, candidate_multipoles.charges, d4_cn,
                        d4_models, d4_parameters, candidate_d4,
                    )
                )
                candidate_d4 = replace(
                    candidate_d4,
                    charge_derivative_hartree=(
                        candidate_d4.charge_derivative_hartree + candidate_atm_charge
                    ),
                )
            candidate_energy = (
                float(np.sum(candidate_density * core))
                + monopole_energy(
                    candidate_shell_q, interaction, partition,
                    quadratic_energy_override=quadratic_energy,
                )
                + anisotropic_state[0]
                + candidate_d4.energy_hartree
                + candidate_atm_energy
            )
            cached_scc_density = candidate_density
            cached_scc_state = (
                candidate_shell_q,
                candidate_multipoles,
                candidate_d4,
                candidate_energy,
                anisotropic_state[1:],
                shell_potential,
            )
            return cached_scc_state

        def scc_energy(candidate_density: np.ndarray) -> float:
            if native_scf_fused:
                return native_state(candidate_density)[1]
            return float(scc_state(candidate_density)[3])

        if (
            os.environ.get("MATRIX_EXPERIMENTAL_NEUTRAL_FRAGMENT_SCF") == "1"
            or os.environ.get("MATRIX_EXPERIMENTAL_ORACLE_LOCAL_GUESS") == "1"
        ):
            if (
                not can_restart and symmetry_adapter is None
                and self.charge == 0 and self.multiplicity == 1
                and domains is None and oracle_seed is None
            ):
                from .fragment_scf_guess import neutral_fragment_scf_guess

                def embedded_fock(candidate_density: np.ndarray) -> tuple[np.ndarray, float]:
                    if native_scf_fused:
                        return native_state(candidate_density)
                    _, _, candidate_d4, candidate_energy, moment_potentials, shell_potential = (
                        scc_state(candidate_density)
                    )
                    qpot, mupot, thetapot = moment_potentials
                    candidate_fock = dense_scc_fock(
                        core, overlap, multipole_operators, partition.orbital_shell,
                        shell_potential, qpot, mupot, thetapot,
                        candidate_d4.charge_derivative_hartree,
                        native_context=native_context,
                    )
                    return candidate_fock, float(candidate_energy)

                fragment_started = perf_counter()
                try:
                    fragment_density, fragment_info = neutral_fragment_scf_guess(
                        core, overlap, np.asarray(basis.spherical_atom),
                        np.asarray(self.atomic_numbers), coordinates,
                        _covalent_radii(), np.asarray(valence),
                        self.temperature_kelvin, embedded_fock,
                        plan=oracle_plan,
                        maximum_cycles=int(os.environ.get(
                            "MATRIX_EXPERIMENTAL_FRAGMENT_PREP_CYCLES", "40"
                        )),
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    self._last_fragment_scf_guess = {
                        "selected": False,
                        "reason": str(exc),
                        "seconds": perf_counter() - fragment_started,
                    }
                else:
                    if fragment_info["converged"]:
                        density = fragment_density
                    self._last_fragment_scf_guess = {
                        "selected": bool(fragment_info["converged"]),
                        **fragment_info,
                        "seconds": perf_counter() - fragment_started,
                    }
                    if oracle_plan is not None and fragment_info["converged"]:
                        self._last_oracle_region_info = {
                            "selected": True,
                            "mode": "embedded_neutral_fragments",
                            "components": fragment_info["components"],
                            "topology_backend": oracle_plan.topology_backend,
                        }
            else:
                self._last_fragment_scf_guess = {
                    "selected": False,
                    "reason": "requires a cold neutral closed-shell dense system",
                }

        # A resident MPS phase is an explicit numerical accelerator. All its
        # density-dependent tensors remain on device; the established FP64
        # loop below certifies the finite-temperature state and its forces.
        mps_threshold = os.environ.get("MATRIX_GFN2_MPS_MIN_ORBITALS", "").strip()
        mps_minimum = self.mps_min_orbitals
        if mps_minimum is None and mps_threshold:
            mps_minimum = int(mps_threshold)
        if mps_minimum is not None and mps_minimum < 1:
            raise ValueError("MPS crossover dimension must be positive")
        if mps_request == "auto" and (mps_minimum is None or len(overlap) < mps_minimum):
            self.last_accelerator_report = {
                "requested": "auto", "actual": "cpu", "precision": "float64",
                "accepted_predictor": False,
                "selection_reason": "no qualified crossover" if mps_minimum is None else "below configured crossover",
            }
        if mps_request == "mps" or (mps_request == "auto" and mps_minimum is not None and len(overlap) >= mps_minimum):
            from .resident_mps import MPSExecution, ResidentSCC, cholesky_orthogonalizer

            execution = MPSExecution(requested=mps_request)
            device_scf = None
            try:
                if self.multiplicity != 1 or self.nn_enabled or symmetry_adapter is not None:
                    raise ValueError("resident MPS qualification requires a closed shell without NN or symmetry blocks")
                if domains is not None or not isinstance(interaction, np.ndarray):
                    raise ValueError("resident MPS qualification requires the dense SCC route")
                if density_builder.frontier_gap_hartree < 1e-2:
                    raise ValueError("small-gap/finite-occupation state requires the host finite-temperature solver")
                operators = multipole_operators
                if operators is None:
                    operators = prepare_pair_cumulative_multipole_operators(
                        self._post_scf_pair_integrals, coordinates, assignment,
                    )
                preparation = dict(
                    core=core, overlap=overlap, operators=operators,
                    partition=partition, valence=valence, interaction=interaction,
                    anisotropic_parameters=anisotropic_parameters,
                    anisotropic_geometry=anisotropic_geometry, d4_prepared=d4_prepared,
                )
                if self._mps_context is None:
                    self._mps_context = ResidentSCC(**preparation)
                else:
                    self._mps_context.update(**preparation)
                device_scf = self._mps_context
                execution = device_scf.report
                execution.requested = mps_request
                execution.orthogonalizer_kind = "cholesky" if direct_generalized else "symmetric"
                host_restart_density = density
                density = device_scf.run(
                    density,
                    cholesky_orthogonalizer(overlap)
                    if direct_generalized else orthogonalizer,
                    electrons,
                    max_iterations=min(60, self.maximum_scf_iterations),
                )
                cached_scc_density = cached_native_density = None
                cached_scc_state = cached_native_state = None
                if can_restart and native_scf_fused and os.environ.get("MATRIX_GFN2_MPS_SELECT_DENSITY", "1") != "0":
                    selection_started = perf_counter()
                    scores = []
                    # Evaluate restart last: it usually wins, and its native
                    # observables/Fock are then already current for SCF.
                    for candidate in (density, host_restart_density):
                        candidate_fock = (native_state(candidate)[0] if native_scf_fused
                                          else None)
                        if candidate_fock is None:
                            raise ValueError("density selection requires the native FP64 Fock")
                        product = (candidate_fock@candidate)@overlap
                        residual = float(np.max(abs(product-product.T)))
                        trace_error = abs(float(np.sum(candidate*overlap))-electrons)/len(overlap)
                        scores.append(max(residual, trace_error))
                    choose_host = np.isfinite(scores[1]) and (not np.isfinite(scores[0]) or scores[1] <= scores[0])
                    execution.density_seed_source = "cpu-restart" if choose_host else "gpu"
                    execution.gpu_seed_residual, execution.cpu_seed_residual = scores
                    execution.seed_selection_seconds = perf_counter()-selection_started
                    if choose_host:
                        density = host_restart_density
                    else:
                        # Restore native observables as well as the cached Fock.
                        cached_native_density = None
                        native_state(density)
            except (ImportError, RuntimeError, ValueError) as error:
                execution.fallback_reason = str(error)
                self._mps_context = None
            finally:
                # The calculator owns the reusable workspace. Batch clones
                # have independent contexts; reports contain no tensors.
                device_scf = None
            self.last_accelerator_report = execution.to_dict()
        iscf_started = perf_counter()
        native_resident_converged = False
        native_resident_started = False
        native_restart_density = None
        native_tblite_restart = bool(
            self.scf_convergence_mode == "tblite"
            and self.multiplicity == 1
            and 35 <= len(overlap) <= 200
            and density_builder.frontier_gap_hartree >= 1.0e-2
            and native_scf_fused and direct_generalized and can_restart
            and os.environ.get("MATRIX_GFN2_DISABLE_NATIVE_TBLITE_RESTART") != "1"
        )
        refinement_eligible = bool(
            self.last_accelerator_report is not None
            and self.last_accelerator_report.get("accepted_predictor")
            and len(overlap) >= 128 and not native_tblite_restart
        )
        if self.last_accelerator_report is not None:
            from .refinement_policy import select_refinement
            decision = select_refinement(
                os.environ.get("MATRIX_GFN2_MPS_REFINE_OCCUPIED", "auto"),
                len(overlap), eligible=refinement_eligible,
                qualified_domain=(self.charge == 0 and self.alpb_parameters is None
                                  and self.temperature_kelvin == 300
                                  and self.scf_convergence_mode == "tblite"
                                  and not len(self.point_charges_electron)
                                  and self.scf_energy_tolerance_hartree == 1e-10
                                  and self.scf_mixer_tolerance == 1e-8),
            )
            density_builder.iterative_refinement = decision.method == "occupied"
            density_builder.canonical_refinement = (
                decision.method == "canonical" and self.last_accelerator_report.get("accepted_predictor", False)
            )
            self.last_accelerator_report.update(
                fp64_refinement=decision.method,
                fp64_refinement_reason=decision.reason,
                fp64_refinement_calibration=decision.calibration,
            )
        density_builder.reuse_refinement_blocks = (
            os.environ.get("MATRIX_GFN2_MPS_REUSE_BLOCKS", "1") != "0"
        )
        if (density_builder.iterative_refinement and can_restart
                and self._mps_context is not None
                and self._mps_context._certified_host_orbitals is not None
                and os.environ.get("MATRIX_GFN2_MPS_TRANSPORT_ORBITALS", "1") != "0"):
            density_builder.seed_orbital_reference(*self._mps_context._certified_host_orbitals)
        if (
            (self.scf_convergence_mode == "density"
             and native_scf_fused and direct_generalized and can_restart)
            or native_tblite_restart
        ):
            native_input_density = density
            native_temperature = _annealed_scf_temperature(
                self.temperature_kelvin, 1, density_builder.frontier_gap_hartree
            )
            native_final_temperature = _annealed_scf_temperature(
                self.temperature_kelvin, 13, density_builder.frontier_gap_hartree
            )
            (
                native_certified,
                native_density,
                native_energies,
                native_coefficients,
                native_occupations,
                native_fock,
                native_initial_fock,
                native_restart_density,
                native_density_residual,
                native_energy_residual,
                native_final_energy,
                native_mixer_residual,
                native_cycles,
            ) = native_context.resident_diis(
                density,
                electrons=electrons,
                multiplicity=self.multiplicity,
                temperature_kelvin=native_temperature,
                wavefunction_tolerance=self.scf_wavefunction_tolerance_electron,
                energy_tolerance=self.scf_energy_tolerance_hartree,
                maximum_cycles=(
                    min(8, self.maximum_scf_iterations)
                    if native_temperature == native_final_temperature
                    and (len(overlap) <= 96
                         or native_tblite_restart)
                    else 1
                ),
                mixer_tolerance=(
                    self.scf_mixer_tolerance
                    if native_tblite_restart else -1.0
                ),
            )
            native_resident_converged = bool(
                native_certified and native_temperature == native_final_temperature
            )
            if int(native_cycles) <= 1 and len(coordinates) >= 21:
                # A single native Roothaan step does not contain enough
                # fixed-point history to replace the trajectory contraction
                # learned on preceding geometries.  Let the established
                # restart predictor consume this step below instead.
                native_restart_density = None
            # Whether certified or not, this is a complete variational ISCF
            # update.  Retain it as cycle one instead of discarding it and
            # rebuilding the same Fock/eigensystem in Python.  The returned
            # Fock belongs to the updated density, so the next guarded cycle
            # can continue without another state construction.
            native_resident_started = True
            iteration = int(native_cycles)
            density = np.asarray(native_density)
            energies = np.asarray(native_energies)
            coefficients = np.asarray(native_coefficients)
            occupations = np.asarray(native_occupations)
            effective_fock = np.asarray(native_fock)
            density_build = DensityBuildResult(
                density=density,
                solver="diagonalization",
                purification_iterations=0,
                used_fallback=False,
                idempotency_error=0.0,
                orbital_energies=energies,
                orbital_coefficients=coefficients,
            )
            density_residuals.append(float(native_density_residual))
            convergence_residuals.append(float(
                native_mixer_residual
                if native_tblite_restart
                else native_density_residual
            ))
            energy_residuals.append(float(native_energy_residual))
            accepted_dampings.append(1.0)
            density_output_history.append(density.copy())
            density_error_history.append(
                (density - native_input_density).ravel().copy()
            )
            initial_error = (
                native_initial_fock @ native_input_density @ overlap
                - overlap @ native_input_density @ native_initial_fock
            )
            _retain_best_diis_history(
                fock_history,
                error_history,
                np.asarray(native_initial_fock),
                initial_error,
                maximum_size=8,
            )
            cached_native_density = density
            cached_native_state = (effective_fock, float(native_final_energy))
            if native_resident_converged:
                converged = True
        current_scc_energy = (
            float(native_final_energy)
            if native_resident_converged
            else scc_energy(density)
        )
        scc_energies = [current_scc_energy]
        adaptive_damping = 0.25
        for iteration in (
            () if native_resident_converged
            else range(
                iteration + 1 if native_resident_started else 1,
                (min(24, self.maximum_scf_iterations)
                 if self.last_accelerator_report is not None
                 and self.last_accelerator_report.get("accepted_predictor")
                 else self.maximum_scf_iterations) + 1,
            )
        ):
            density_builder.temperature_kelvin = _annealed_scf_temperature(
                self.temperature_kelvin,
                iteration,
                density_builder.frontier_gap_hartree,
            )
            if native_scf_fused:
                fock, _ = native_state(density)
            else:
                shell_q, multipoles, d4, _, moment_potentials, shell_potential = scc_state(density)
                qpot, mupot, thetapot = moment_potentials
                fock = dense_scc_fock(
                    core, overlap, multipole_operators, partition.orbital_shell,
                    shell_potential, qpot, mupot, thetapot,
                    d4.charge_derivative_hartree,
                    native_context=native_context,
                )
            error = fock @ density @ overlap - overlap @ density @ fock
            if sparse_diis is None:
                _retain_best_diis_history(
                    fock_history,
                    error_history,
                    fock,
                    error,
                    maximum_size=8,
                )
                effective_fock = _diis_fock(fock_history, error_history)
            else:
                sparse_diis.append(
                    SparseSymmetricMatrix.from_dense(
                        fock,
                        domains,
                        threshold=self.solver_policy.sparse_truncation_threshold,
                    ),
                    _sparse_antisymmetric_residual(error, domains),
                )
                effective_fock = sparse_diis.extrapolate(
                    threshold=self.solver_policy.sparse_truncation_threshold
                ).to_dense()
            density_build = density_builder.solve(effective_fock)
            updated = density_build.density
            density_error = updated - density
            residual = float(np.max(np.abs(density_error)))
            convergence_residual = (
                tblite_mixer_residual(density, updated)
                if self.scf_convergence_mode == "tblite" else residual
            )
            convergence_tolerance = (
                self.scf_mixer_tolerance
                if self.scf_convergence_mode == "tblite"
                else self.scf_wavefunction_tolerance_electron
            )
            # A trajectory restart can already lie inside the wavefunction
            # convergence ball.  Damping such a fixed-point update forces
            # several unnecessary diagonalizations: the pre-update residual
            # is the certification that the full Roothaan density is safe to
            # test.  Accept it directly, then retain the independent energy
            # test below.  If that test fails, the ordinary safeguarded path
            # remains active on the next cycle.
            certified_fixed_point = (
                convergence_residual < convergence_tolerance
            )
            density_output_history.append(updated.copy())
            density_error_history.append(density_error.ravel().copy())
            if len(density_output_history) > 6:
                del density_output_history[0]
                del density_error_history[0]
            oscillating = bool(
                density_residuals and residual > 1.05 * density_residuals[-1]
            )
            target = (
                _anderson_density(density_output_history, density_error_history)
                if oscillating and len(density_output_history) >= 3
                else updated
            )
            if certified_fixed_point:
                adaptive_damping = 1.0
            elif can_restart and not oscillating:
                # The previous converged trajectory density is already on
                # the physical SCC branch.  A full downhill Roothaan/DIIS
                # step is therefore preferable to replaying the cold-start
                # damping ramp; the energy backtracking immediately below
                # remains the independent safety certificate.
                adaptive_damping = 1.0
            elif iteration < 3:
                adaptive_damping = 0.50
            elif density_residuals and residual < density_residuals[-1]:
                adaptive_damping = min(0.90, 1.35 * adaptive_damping)
            elif density_residuals:
                adaptive_damping = max(0.10, 0.60 * adaptive_damping)
            damping = adaptive_damping
            accepted_density = (
                updated
                if certified_fixed_point
                else (1.0 - damping) * density + damping * target
            )
            accepted_energy = scc_energy(accepted_density)
            # The initial core density and the self-consistent functional are
            # not comparable.  Once three SCC updates have established the
            # functional branch, backtrack any uphill step.
            accepted_extrapolation = (
                iteration < 4
                or accepted_energy <= current_scc_energy + 1.0e-12
                or (
                    medium_covalent_chain
                    and convergence_residuals
                    and convergence_residual < 0.5 * convergence_residuals[-1]
                    and accepted_energy - current_scc_energy
                    <= 2.0e-7 * len(coordinates)
                )
            )
            backtracks = 0
            while (
                iteration >= 4
                and not accepted_extrapolation
                and damping >= 1.0e-4
                and backtracks < 4
            ):
                damping *= 0.5
                candidate = (1.0 - damping) * density + damping * target
                candidate_energy = scc_energy(candidate)
                if candidate_energy <= current_scc_energy + 1.0e-12:
                    accepted_density = candidate
                    accepted_energy = candidate_energy
                    accepted_extrapolation = True
                    break
                backtracks += 1
            if not accepted_extrapolation:
                # The extrapolated subspace is not a descent direction.  Do
                # not keep reusing it with a vanishing step: discard the
                # contaminated DIIS/Anderson history and backtrack a plain
                # Roothaan update instead.  This is a standard DIIS restart,
                # and preserves the energy safeguard without stalling.
                fock_history[:] = [fock.copy()]
                error_history[:] = [error.ravel().copy()]
                if sparse_diis is not None:
                    sparse_diis.clear()
                    sparse_diis.append(
                        SparseSymmetricMatrix.from_dense(
                            fock,
                            domains,
                            threshold=self.solver_policy.sparse_truncation_threshold,
                        ),
                        _sparse_antisymmetric_residual(error, domains),
                    )
                density_output_history[:] = [updated.copy()]
                density_error_history[:] = [density_error.ravel().copy()]
                damping = min(0.20, adaptive_damping)
                while damping >= 1.0e-4:
                    candidate = (1.0 - damping) * density + damping * updated
                    candidate_energy = scc_energy(candidate)
                    if candidate_energy <= current_scc_energy + 1.0e-12:
                        accepted_density = candidate
                        accepted_energy = candidate_energy
                        break
                    damping *= 0.5
                if damping < 1.0e-4:
                    # At an energy stationary point, retain a bounded raw
                    # step so the commutator can still be reduced.
                    damping = min(0.02, adaptive_damping)
                    accepted_density = (1.0 - damping) * density + damping * updated
                    accepted_energy = scc_energy(accepted_density)
            energy_residual = abs(accepted_energy - current_scc_energy)
            density_residuals.append(residual)
            convergence_residuals.append(convergence_residual)
            energy_residuals.append(energy_residual)
            accepted_dampings.append(damping)
            density = accepted_density
            current_scc_energy = accepted_energy
            scc_energies.append(current_scc_energy)
            if (
                convergence_residual < convergence_tolerance
                and energy_residual < self.scf_energy_tolerance_hartree
                and density_builder.temperature_kelvin
                == _annealed_scf_temperature(
                    self.temperature_kelvin,
                    max(iteration, 13),
                    density_builder.frontier_gap_hartree,
                )
            ):
                if density_build.solver == "occupied-refinement":
                    # The intermediate projector avoids a full spectrum;
                    # independently restore canonical finite-T orbitals
                    # before declaring the electronic state converged.
                    certificate = density_builder._diagonalize(
                        effective_fock, used_fallback=False,
                    )
                    if native_scf_fused:
                        native_state(density)
                    canonical_residual = (
                        tblite_mixer_residual(density, certificate.density)
                        if self.scf_convergence_mode == "tblite"
                        else float(np.max(np.abs(certificate.density-density)))
                    )
                    canonical_energy = scc_energy(certificate.density)
                    canonical_change = abs(canonical_energy-current_scc_energy)
                    density_build = certificate
                    updated = certificate.density
                    if (canonical_residual >= convergence_tolerance
                            or canonical_change >= self.scf_energy_tolerance_hartree):
                        density = updated
                        current_scc_energy = canonical_energy
                        continue
                    convergence_residuals[-1] = max(convergence_residuals[-1], canonical_residual)
                    energy_residuals[-1] = max(energy_residuals[-1], canonical_change)
                density = updated
                converged = True
                break
        iscf_wall_seconds = perf_counter() - iscf_started
        self._last_scf_trace = tuple(
            {
                "cycle": (
                    index + int(native_cycles) - 1
                    if native_resident_started else index
                ),
                "density_residual": density_residual,
                "convergence_residual": convergence_residual,
                "convergence_metric": self.scf_convergence_mode,
                "energy_residual_hartree": energy_residual,
                "accepted_damping": damping,
            }
            for index, (
                density_residual, convergence_residual, energy_residual, damping
            ) in enumerate(
                zip(
                    density_residuals,
                    convergence_residuals,
                    energy_residuals,
                    accepted_dampings,
                    strict=True,
                ),
                1,
            )
        )
        if (not converged and self.last_accelerator_report is not None
                and self.last_accelerator_report.get("accepted_predictor")):
            report = dict(self.last_accelerator_report)
            report["accepted_predictor"] = False
            report["fallback_reason"] = "FP64 polishing did not converge; retried from the CPU initial state"
            self._mps_context = None
            retry = self._single_point(
                coordinates, compute_cm5=compute_cm5,
                compute_bond_orders=compute_bond_orders, density_policy=density_policy,
                retain_orbitals=retain_orbitals, collect_telemetry=collect_telemetry,
                energy_only=energy_only, nn_properties_only=nn_properties_only,
                _mps_retry=True,
            )
            self.last_accelerator_report = report
            if not isinstance(retry, float) and retry.telemetry is not None:
                retry = replace(retry, telemetry=replace(
                    retry.telemetry, accelerator_execution=report,
                    total_wall_seconds=perf_counter()-total_started,
                ))
            return retry
        post_scf_started = perf_counter()
        final_solver_started = perf_counter()
        can_reuse_final_orbitals = (
            converged
            and density_build.solver == "diagonalization"
            and density_build.orbital_energies is not None
            and density_build.orbital_coefficients is not None
        )
        if can_reuse_final_orbitals:
            energies = density_build.orbital_energies
            coefficients = density_build.orbital_coefficients
            if symmetry_adapter is not None:
                self._symmetry_reference_coefficients = coefficients.copy()
        elif symmetry_adapter is None:
            energies, coefficients = solve_generalized_symmetric(
                effective_fock,
                overlap,
                orthogonalizer=orthogonalizer,
                diagonalizer_backend=diagonalizer_backend,
            )
        else:
            energies, coefficients = symmetry_adapter.solve(
                effective_fock,
                diagonalizer_backend=diagonalizer_backend,
                small_system_host_threshold=self.symmetry_small_system_host_threshold,
                reference_coefficients=coefficients,
            )
            self._symmetry_reference_coefficients = coefficients.copy()
        if not native_resident_converged:
            occupations = fermi_occupations(
                energies,
                electrons,
                multiplicity=self.multiplicity,
                temperature_kelvin=self.temperature_kelvin,
            )
            density = (coefficients * occupations) @ coefficients.T
        final_solver_wall_seconds = perf_counter() - final_solver_started
        if (converged and self._mps_context is not None
                and self.last_accelerator_report is not None
                and self.last_accelerator_report.get("accepted_predictor")):
            self._mps_context.certify_host_density(density, (coefficients, energies, occupations))
        if self.last_accelerator_report is not None:
            self.last_accelerator_report["polishing_canonical_solves"] = density_builder.canonical_solves
            self.last_accelerator_report["occupied_refinement_steps"] = density_builder.refinement_steps
            self.last_accelerator_report["transported_orbital_reference"] = density_builder.transported_reference
            self.last_accelerator_report["refinement_block_reuse"] = density_builder.reuse_refinement_blocks
        if symmetry_adapter is not None:
            density = symmetry_adapter.symmetrize_matrix(density)
        # The native context already owns the complete SCC, multipolar, and D4
        # state builder.  Reuse it for the final density even when convergence
        # required guarded Python DIIS cycles; rebuilding the same observables
        # through separate NumPy kernels made the dense post-SCF cost grow
        # unnecessarily with molecular size.
        native_post_scf_state = bool(native_scf_fused and native_context is not None)
        anisotropic_potentials_override = None
        native_post_scf_energy = None
        if native_post_scf_state:
            native_state(density)
            (
                native_observables,
                native_anisotropic,
                native_d4,
                native_post_scf_energy,
            ) = native_context.last_scf_state()
            shell_q = np.asarray(native_observables[0])
            multipoles = AtomicMultipoles(
                np.asarray(native_observables[1]),
                np.asarray(native_observables[2]),
                np.asarray(native_observables[3]),
            )
            anisotropic_energy = float(native_anisotropic[0])
            anisotropic_potentials_override = tuple(
                np.asarray(item) for item in native_anisotropic[1:]
            )
            d4_energy, d4_charge, d4_coordination = native_d4[:3]
            pair_c6, pair_c8, coordination_left, coordination_right = native_d4[3:7]
            if sparse_dispersion:
                maximum_c6 = float(native_d4[7])
                maximum_ratio = float(np.max(d4_prepared.ratio, initial=0.0))
                radius = d4_prepared.cutoff_bohr
                tail_bound = d4_prepared.omitted_pairs * maximum_c6 * (
                    d4_parameters.s6 / radius**6
                    + d4_parameters.s8 * maximum_ratio / radius**8
                )
                d4 = SparseD4Result(
                    float(d4_energy), np.asarray(d4_charge),
                    np.asarray(d4_coordination), d4_prepared.rows,
                    d4_prepared.columns, np.asarray(pair_c6), np.asarray(pair_c8),
                    np.asarray(coordination_left), np.asarray(coordination_right),
                    float(tail_bound),
                )
            else:
                dense_c6 = np.zeros((len(coordinates), len(coordinates)))
                dense_c8 = np.zeros_like(dense_c6)
                dense_c6[d4_rows, d4_columns] = pair_c6
                dense_c6[d4_columns, d4_rows] = pair_c6
                dense_c8[d4_rows, d4_columns] = pair_c8
                dense_c8[d4_columns, d4_rows] = pair_c8
                d4 = D4Result(
                    float(d4_energy), np.asarray(d4_charge),
                    np.asarray(d4_coordination), dense_c6, dense_c8,
                    np.empty((0, 0, 0), dtype=float),
                )
        else:
            shell_q, multipoles = dense_scc_observables(
                density,
                multipole_operators,
                partition.orbital_shell,
                partition.reference_population,
                valence,
                native_context=native_context,
            )
            anisotropic_energy = anisotropic_energy_and_moment_potentials(
                multipoles, coordinates, anisotropic_parameters,
                geometry=anisotropic_geometry,
            )[0]
            d4 = (
                d4_two_body_sparse(
                    multipoles.charges, d4_prepared, d4_parameters,
                )
                if sparse_dispersion
                else d4_two_body(
                    coordinates, multipoles.charges, d4_cn, d4_models,
                    d4_parameters, prepared=d4_prepared,
                )
            )
            if getattr(self, "_active_nn_state", None) is not None and len(coordinates) >= 3:
                _, atm_charge_derivative = atm_three_body_energy_charge_derivative_dense(
                    coordinates, multipoles.charges, d4_cn, d4_models,
                    d4_parameters, d4,
                )
                d4 = replace(
                    d4,
                    charge_derivative_hartree=(
                        d4.charge_derivative_hartree + atm_charge_derivative
                    ),
                )
        if native_post_scf_energy is None:
            energy = float(np.sum(density * core))
            energy += monopole_energy(shell_q, interaction, partition)
            energy += anisotropic_energy + d4.energy_hartree
        else:
            energy = float(native_post_scf_energy)
        repulsion_gradient_override = None
        if native_context is not None and len(coordinates) <= 128:
            repulsion_value, repulsion_gradient_override = (
                native_context.repulsion_energy_gradient(
                    coordinates,
                    self._atomic_number_array,
                    self._repulsion_alpha,
                    self._effective_nuclear_charges,
                    24.0,
                )
            )
            energy += repulsion_value
        else:
            repulsion_value = repulsion_energy(
                self.atomic_numbers, coordinates, atom_parameters,
                kexp=(1.5 if getattr(self, "_active_nn_state", None) is None
                      else self._active_nn_state.globals["kexp"]),
                klight=(1.0 if getattr(self, "_active_nn_state", None) is None
                        else self._active_nn_state.globals["klight"]),
            )
            energy += repulsion_value
        if len(self.point_charges_electron):
            embedding = _point_charge_embedding(
                overlap,
                None,
                assignment,
                coordinates,
                self.point_charge_coordinates_bohr,
                self.point_charges_electron,
            )
            energy += float(np.asarray(valence) @ embedding.atomic_potential)
        neutral_d4 = None
        atm_gradient_override = None
        atm_value = 0.0
        if len(coordinates) >= 3:
            if sparse_dispersion:
                neutral_pairs = d4_two_body_sparse(
                    np.zeros(len(coordinates)), d4_prepared, d4_parameters,
                )
                atm = atm_three_body_energy_gradient_sparse(
                    coordinates, neutral_pairs,
                    cutoff_bohr=25.0,
                    s9=d4_parameters.s9,
                    a1=d4_parameters.a1,
                    a2=d4_parameters.a2,
                )
                energy += atm.energy_hartree
                atm_value = float(atm.energy_hartree)
                neutral_d4 = (neutral_pairs, atm)
            else:
                neutral_d4 = (
                    d4 if getattr(self, "_active_nn_state", None) is not None and not d4_parameters.neutral_atm
                    else d4_two_body(
                        coordinates, np.zeros(len(coordinates)), d4_cn,
                        d4_models, d4_parameters, prepared=d4_prepared,
                    )
                )
                if native_context is not None and len(coordinates) <= 128:
                    atm_energy, atm_gradient_override = (
                        native_context.atm_energy_gradient(
                            coordinates,
                            neutral_d4.c6,
                            neutral_d4.c8,
                            neutral_d4.c6_coordination_derivative,
                            _covalent_radii()[self._atomic_number_array - 1],
                            self._electronegativities,
                            s9=d4_parameters.s9,
                            a1=d4_parameters.a1,
                            a2=d4_parameters.a2,
                        )
                    )
                    energy += atm_energy
                    atm_value = float(atm_energy)
                else:
                    atm = atm_three_body_energy_gradient_dense(
                        coordinates,
                        neutral_d4,
                        d4_coordination_number_jacobian(
                            self.atomic_numbers,
                            coordinates,
                            _covalent_radii(),
                            self._electronegativities,
                        ),
                        s9=d4_parameters.s9,
                        a1=d4_parameters.a1,
                        a2=d4_parameters.a2,
                    )
                    energy += atm.energy_hartree
                    atm_value = float(atm.energy_hartree)
                    if atm.explicit_gradient_hartree_per_bohr is not None:
                        atm_gradient_override = (
                            atm.explicit_gradient_hartree_per_bohr
                        )
        # The restricted closed-shell channels are identical.  Reuse the
        # occupations already obtained for the final density instead of
        # repeating two finite-temperature chemical-potential searches in
        # the post-SCF analysis.  Open-shell states retain the exact separate
        # alpha/beta construction.
        if self.multiplicity == 1:
            alpha_occ = 0.5 * occupations
            beta_occ = alpha_occ
        else:
            alpha_occ, beta_occ = spin_fermi_occupations(
                energies,
                electrons,
                multiplicity=self.multiplicity,
                temperature_kelvin=self.temperature_kelvin,
            )
        energy += alpb_static_energy
        energy += electronic_entropy_energy(
            alpha_occ, beta_occ, self.temperature_kelvin
        )
        alpha_density = beta_density = None
        if compute_bond_orders:
            alpha_density = (coefficients * alpha_occ) @ coefficients.T
            beta_density = (coefficients * beta_occ) @ coefficients.T
            if symmetry_adapter is not None:
                alpha_density = symmetry_adapter.symmetrize_matrix(alpha_density)
                beta_density = symmetry_adapter.symmetrize_matrix(beta_density)
        bond_order_started = perf_counter()
        if not compute_bond_orders:
            bond_orders = self._empty_bond_orders
        elif domains is not None and density_builder.last_solver == "sparse-sp2":
            bond_orders = sparse_mayer_bond_orders(
                SparseSymmetricMatrix.from_dense(
                    density,
                    domains,
                    threshold=self.solver_policy.sparse_truncation_threshold,
                ),
                SparseSymmetricMatrix.from_dense(overlap, domains, threshold=0.0),
                assignment,
            )
        else:
            bond_orders = _bond_orders(
                density,
                overlap,
                assignment,
                spin_densities=(alpha_density, beta_density),
            )
        if compute_bond_orders and symmetry_adapter is not None:
            bond_orders = symmetry_adapter.expand_atomic_pairs(np.asarray(bond_orders))
        bond_order_wall_seconds = perf_counter() - bond_order_started
        if energy_only:
            if not converged:
                raise RuntimeError("GFN2 energy-only SCC did not converge")
            if getattr(self, "_active_nn_state", None) is not None:
                self._nn_development_components = {
                    "band": float(np.sum(density * core)),
                    "scc": float(monopole_energy(shell_q, interaction, partition)),
                    "aes": float(anisotropic_energy),
                    "d4_pair": float(d4.energy_hartree),
                    "d4_atm": atm_value,
                    "repulsion": float(repulsion_value),
                    "total": float(energy),
                }
            return float(energy)
        if nn_properties_only:
            if getattr(self, "_active_nn_state", None) is None:
                raise RuntimeError("NN property evaluation requires an active parameter state")
            if not converged:
                raise RuntimeError("NN-xTB SCC did not converge")
            return GFN2Result(
                float(energy), np.zeros_like(coordinates),
                np.asarray(multipoles.charges).copy(), bond_orders,
                np.asarray(density).copy(), iteration, True,
                self._empty_point_charge_gradient, None, None,
                density_builder.last_solver,
            )
        gradient_started = perf_counter()
        post_scf_explicit_gradient_override = None
        if (
            native_context is not None
            and len(coordinates) <= 128
            and not hasattr(d4, "rows")
            and not self._gaussian_scc_active
        ):
            radius_exponential = np.exp(
                -4.0 * (
                    coordination - self._valence_coordination - 1.2
                )
            )
            radius_cn_derivative = (
                4.0 * (5.0 - self._base_multipole_radii)
                * radius_exponential / (1.0 + radius_exponential) ** 2
            )
            post_scf_explicit_gradient_override = (
                native_context.post_scf_explicit_gradient(
                    coordinates,
                    shell_q,
                    np.asarray(partition.shell_atom, dtype=np.int64),
                    np.asarray(partition.hardness_hartree, dtype=float),
                    multipoles.charges,
                    multipoles.dipoles_bohr,
                    multipoles.quadrupoles_bohr2,
                    anisotropic_parameters.damping_radius_bohr,
                    self._coordination_radii,
                    radius_cn_derivative,
                    d4.c6,
                    d4.c8,
                    d4.coordination_derivative_hartree,
                    _covalent_radii()[self._atomic_number_array - 1],
                    self._electronegativities,
                    anisotropic_a3=anisotropic_parameters.a3,
                    anisotropic_a5=anisotropic_parameters.a5,
                    s6=d4_parameters.s6,
                    s8=d4_parameters.s8,
                    d4_a1=d4_parameters.a1,
                    d4_a2=d4_parameters.a2,
                )
            )
        weighted_override = (
            energy_weighted_density_from_fock(
                effective_fock, density, orthogonalizer
            )
            if density_builder.last_solver != "diagonalization"
            else None
        )
        if not len(self.point_charges_electron):
            weighted = (
                energy_weighted_density(coefficients, occupations, energies)
                if weighted_override is None else weighted_override
            )
            gradient = _analytic_gradient_pair_packed(
                density=density,
                weighted_density=weighted,
                basis=basis,
                coordinates=coordinates,
                atomic_numbers=self.atomic_numbers,
                atom_parameters=atom_parameters,
                orbital_parameters=orbital_parameters,
                partition=partition,
                shell_interaction=interaction,
                shell_q=shell_q,
                multipoles=multipoles,
                anisotropic_parameters=anisotropic_parameters,
                anisotropic_geometry=anisotropic_geometry,
                d4=d4,
                d4_parameters=d4_parameters,
                neutral_d4=neutral_d4,
                valence=valence,
                integrals=self._post_scf_pair_integrals,
                spherical_domains=self._post_scf_pair_domains,
                element_values=self._element_values,
                base_multipole_radii=self._base_multipole_radii,
                valence_coordination=self._valence_coordination,
                electronegativities=self._electronegativities,
                orbital_metadata=self._orbital_metadata,
                coordination_override=coordination,
                nn_global_scalings=(
                    None if getattr(self, "_active_nn_state", None) is None
                    else self._active_nn_state.globals
                ),
                native_context=self._active_native_context,
                repulsion_gradient_override=repulsion_gradient_override,
                atm_gradient_override=atm_gradient_override,
                post_scf_explicit_gradient_override=(
                    post_scf_explicit_gradient_override
                ),
                anisotropic_potentials_override=anisotropic_potentials_override,
            )
            point_charge_gradient = self._empty_point_charge_gradient
        else:
            gradient, point_charge_gradient = _analytic_gradient(
            density=density,
            coefficients=coefficients,
            occupations=occupations,
            orbital_energies=energies,
            basis=basis,
            overlap=overlap,
            dipole=dipole,
            second=second,
            coordinates=coordinates,
            atomic_numbers=self.atomic_numbers,
            atom_parameters=atom_parameters,
            orbital_parameters=orbital_parameters,
            partition=partition,
            shell_interaction=interaction,
            shell_q=shell_q,
            multipoles=multipoles,
            anisotropic_parameters=anisotropic_parameters,
            anisotropic_geometry=anisotropic_geometry,
            d4=d4,
            d4_cn=d4_cn,
            d4_parameters=d4_parameters,
            neutral_d4=neutral_d4,
            point_charge_coordinates=self.point_charge_coordinates_bohr,
            point_charges=self.point_charges_electron,
            valence=valence,
                weighted_density_override=weighted_override,
            )
        if self.alpb_parameters is not None:
            _, alpb_gradient = alpb_fixed_charge_energy_gradient_hartree(
                coordinates, np.asarray(self.atomic_numbers),
                np.asarray(multipoles.charges), self.alpb_parameters,
            )
            gradient += alpb_gradient
        if symmetry_adapter is not None:
            gradient = symmetry_adapter.expand_atomic_vectors(gradient)
        gradient_wall_seconds = perf_counter() - gradient_started
        analysis_started = perf_counter()
        hirshfeld = None
        cm5 = None
        reported_charges = multipoles.charges
        if symmetry_adapter is not None:
            reported_charges = symmetry_adapter.expand_atomic_scalars(reported_charges)
            if hirshfeld is not None:
                hirshfeld = symmetry_adapter.expand_atomic_scalars(hirshfeld)
            if cm5 is not None:
                cm5 = symmetry_adapter.expand_atomic_scalars(cm5)
        analysis_wall_seconds = perf_counter() - analysis_started
        if symmetry_adapter is not None:
            symmetry_solver_mode = "symmetry-blocked"
        elif self.symmetry_contract is not None:
            symmetry_solver_mode = "full-matrix-fallback"
        else:
            symmetry_solver_mode = "full-matrix"
        # Report the route actually used.  A compiled adapter can exist for a
        # frozen solute even when an asymmetric embedding has disabled the
        # block solve; exposing it here would falsely claim reduced storage.
        telemetry_adapter = symmetry_adapter
        block_storage = (
            sum(size * size for size in telemetry_adapter.block_dimensions)
            / float(len(overlap) ** 2)
            if telemetry_adapter is not None
            else 1.0
        )
        telemetry = None if not collect_telemetry else GFN2Telemetry(
            schema=TELEMETRY_SCHEMA,
            solver=SolverTelemetry(
                requested_solver=active_density_policy.normalized_mode,
                used_solver=density_builder.last_solver,
                requested_backend=(
                    "auto"
                    if self.diagonalizer_backend is None
                    else str(self.diagonalizer_backend)
                ),
                used_backend=density_build.product_backend,
                fallback_reason=density_build.fallback_reason,
                fallback_category=density_build.fallback_category,
                purification_iterations=density_build.purification_iterations,
                mixed_precision_iterations=density_build.mixed_precision_iterations,
                fp64_iterations=density_build.fp64_iterations,
                device_transfers=density_build.device_transfers,
                peak_device_bytes=density_build.peak_device_bytes,
                idempotency_error=density_build.idempotency_error,
            ),
            scc_iterations=iteration,
            final_density_residual=density_residuals[-1],
            final_energy_residual_hartree=energy_residuals[-1],
            maximum_density_residual=max(density_residuals),
            maximum_energy_residual_hartree=max(energy_residuals),
            converged=converged,
            accelerator_execution=self.last_accelerator_report,
            total_wall_seconds=perf_counter() - total_started,
            peak_rss_bytes=_peak_rss_bytes(),
            model_wall_seconds=model_wall_seconds,
            initial_solver_wall_seconds=initial_solver_wall_seconds,
            iscf_wall_seconds=iscf_wall_seconds,
            final_solver_wall_seconds=final_solver_wall_seconds,
            post_scf_wall_seconds=perf_counter() - post_scf_started,
            gradient_wall_seconds=gradient_wall_seconds,
            bond_order_wall_seconds=bond_order_wall_seconds,
            analysis_wall_seconds=analysis_wall_seconds,
            scc_energy_monotonic=bool(
                len(scc_energies) <= 4
                or np.all(np.diff(np.asarray(scc_energies[3:])) <= 1.0e-10)
            ),
            minimum_damping=min(accepted_dampings),
            symmetry_point_group=(
                self.symmetry_contract.point_group
                if self.symmetry_contract is not None
                else "C1"
            ),
            symmetry_operation_count=(
                len(self.symmetry_contract.operations)
                if self.symmetry_contract is not None
                else 1
            ),
            symmetry_atom_orbit_count=(
                0 if telemetry_adapter is None else len(telemetry_adapter.atom_orbits)
            ),
            symmetry_shell_orbit_count=(
                0 if telemetry_adapter is None else len(telemetry_adapter.shell_orbits)
            ),
            symmetry_block_dimensions=(
                () if telemetry_adapter is None else telemetry_adapter.block_dimensions
            ),
            symmetry_block_labels=(
                () if telemetry_adapter is None else telemetry_adapter.block_labels
            ),
            symmetry_block_batches=(
                () if telemetry_adapter is None else telemetry_adapter.block_batches
            ),
            symmetry_block_storage_fraction=block_storage,
            symmetry_overlap_invariance_error=(
                0.0 if telemetry_adapter is None else telemetry_adapter.overlap_invariance_error
            ),
            symmetry_geometry_error_angstrom=(
                0.0 if telemetry_adapter is None else telemetry_adapter.geometry_error_angstrom
            ),
            symmetry_fock_off_block_error=(
                0.0
                if symmetry_adapter is None
                else symmetry_adapter.off_block_error(effective_fock)
            ),
            symmetry_density_off_block_error=(
                0.0
                if symmetry_adapter is None
                else symmetry_adapter.off_block_error(density)
            ),
            symmetry_fock_commutator_error=(
                0.0
                if symmetry_adapter is None
                else symmetry_adapter.commutator_error(effective_fock)
            ),
            symmetry_density_commutator_error=(
                0.0
                if symmetry_adapter is None
                else symmetry_adapter.commutator_error(density)
            ),
            symmetry_irrep_degeneracy_error_hartree=(
                0.0
                if symmetry_adapter is None
                else symmetry_adapter.irrep_degeneracy_error(effective_fock)
            ),
            symmetry_embedding_mode=str(symmetry_diagnostics["embedding_mode"]),
            symmetry_solver_mode=symmetry_solver_mode,
            symmetry_embedding_orbits=tuple(symmetry_diagnostics["embedding_orbits"]),
            symmetry_embedding_residual=float(symmetry_diagnostics["embedding_residual"]),
            symmetry_full_matrix_reason=symmetry_diagnostics["full_matrix_reason"],
            symmetry_cache_hits=self._symmetry_cache_hits,
            symmetry_cache_misses=self._symmetry_cache_misses,
            symmetry_cache_evictions=self._symmetry_cache_evictions,
        )
        if telemetry is not None:
            telemetry.validate()
        restart_storage_density = (
            density
            if native_restart_density is None
            else np.asarray(native_restart_density)
        )
        contraction = self._restart_fixed_point_contraction
        if native_restart_density is None and len(density_error_history) >= 2:
            previous_fixed_point_error = density_error_history[-2]
            final_fixed_point_error = density_error_history[-1]
            denominator = float(
                previous_fixed_point_error @ previous_fixed_point_error
            )
            if denominator > np.finfo(float).tiny:
                contraction = float(
                    previous_fixed_point_error @ final_fixed_point_error
                ) / denominator
                if np.isfinite(contraction):
                    contraction = float(np.clip(contraction, -0.15, 0.15))
                    self._restart_fixed_point_contraction = contraction
        if (
            native_restart_density is None
            and density_error_history
            and -0.15 <= contraction <= 0.15
        ):
            final_fixed_point_error = density_error_history[-1]
            correction = contraction / (1.0 - contraction)
            restart_storage_density = density + correction * (
                final_fixed_point_error.reshape(density.shape)
            )
        self._older_density = self._previous_density
        self._previous_density = np.asarray(restart_storage_density).copy()
        self._older_orthogonal_density = self._previous_orthogonal_density
        self._older_coordinates_bohr = self._previous_coordinates_bohr
        if direct_generalized:
            self._previous_orthogonal_density = None
        else:
            orthogonal_coefficients = np.linalg.solve(orthogonalizer, coefficients)
            self._previous_orthogonal_density = (
                orthogonal_coefficients * occupations
            ) @ orthogonal_coefficients.T
            self._previous_orthogonal_density = 0.5 * (
                self._previous_orthogonal_density
                + self._previous_orthogonal_density.T
            )
        self._previous_coordinates_bohr = coordinates.copy()
        self._previous_orbital_energies = energies.copy()
        self._previous_occupations = occupations.copy()
        reported_density: np.ndarray | SparseSymmetricMatrix = density
        if domains is not None and density_builder.last_solver == "sparse-sp2":
            reported_density = SparseSymmetricMatrix.from_dense(
                density,
                domains,
                threshold=self.solver_policy.sparse_truncation_threshold,
            )
        orbital_state = None
        if retain_orbitals:
            primitive_offsets = [0]
            primitive_exponents: list[float] = []
            primitive_coefficients: list[float] = []
            cartesian_atom: list[int] = []
            cartesian_type: list[int] = []
            component_type = {
                (0, 0, 0): 1,
                (1, 0, 0): 2,
                (0, 1, 0): 3,
                (0, 0, 1): 4,
                (2, 0, 0): 5,
                (0, 2, 0): 6,
                (0, 0, 2): 7,
                (1, 1, 0): 8,
                (1, 0, 1): 9,
                (0, 1, 1): 10,
            }
            for orbital in basis.cartesian_orbitals:
                primitives = orbital.primitives
                primitive_exponents.extend(item.exponent for item in primitives)
                primitive_coefficients.extend(item.coefficient for item in primitives)
                primitive_offsets.append(len(primitive_exponents))
                cartesian_atom.append(orbital.atom_index)
                cartesian_type.append(component_type[orbital.angular])
            orbital_state = GFN2OrbitalState(
                model_id=("MATRIX-GFN2.v1" if self.onsite_profile is None or self.onsite_profile.is_identity
                          else "MATRIX-GFN2.v1.onsite." + self.onsite_profile.content_sha256)
                         + (".gaussian-scc." + self.electrostatic_profile.content_sha256 if self._gaussian_scc_active else ""),
                atomic_numbers=self._atomic_number_array,
                coordinates_bohr=coordinates,
                overlap=overlap,
                fock=effective_fock,
                coefficients=coefficients,
                orbital_energies_hartree=energies,
                alpha_occupations=alpha_occ,
                beta_occupations=beta_occ,
                ao_atom=np.asarray(basis.spherical_atom, dtype=np.int64),
                ao_shell=tuple(basis.spherical_shell),
                basis=GFN2OrbitalBasis(
                    cartesian_atom=cartesian_atom,
                    cartesian_type=cartesian_type,
                    primitive_offsets=primitive_offsets,
                    primitive_exponents=primitive_exponents,
                    primitive_coefficients=primitive_coefficients,
                    spherical_to_cartesian=basis.cartesian_to_spherical,
                ),
                total_energy_hartree=energy,
                electron_count=electrons,
                charge=self.charge,
                multiplicity=self.multiplicity,
            )
        return GFN2Result(
            energy, gradient, reported_charges, bond_orders,
            reported_density, iteration, converged, point_charge_gradient, hirshfeld, cm5,
            density_builder.last_solver, telemetry, orbital_state,
        )

    def _single_point_sparse(
        self,
        coordinates: np.ndarray,
        domains=None,
        *,
        locality_policy: LocalityPolicy | None = None,
        compute_bond_orders: bool = True,
        retain_projectors: bool = False,
    ) -> GFN2Result:
        """Closed-shell, gapped, end-to-end localized SCF route."""

        if self._gaussian_scc_active:
            raise NotImplementedError("Gaussian SCC is qualified on the dense molecular route")

        conservative_sparse_scf = (
            os.environ.get("MATRIX_GFN2_SPARSE_CONSERVATIVE_SCF") == "1"
        )
        experimental_sparse_lmo = (
            os.environ.get("MATRIX_GFN2_EXPERIMENTAL_SPARSE_LMO") == "1"
        )
        if self._sparse_allocation_gate is None:
            model = self._model_sparse(
                coordinates, domains=domains, locality_policy=locality_policy,
            )
        else:
            with self._sparse_allocation_gate:
                model = self._model_sparse(
                    coordinates, domains=domains, locality_policy=locality_policy,
                )
        basis, domains = model.basis, model.domains
        electrons = int(round(sum(model.partition.reference_population) - self.charge))
        if self.multiplicity != 1 or electrons % 2:
            raise RuntimeError("sparse GFN2 route requires an even closed shell")
        same_sparse_topology = self._previous_sparse_domains is domains
        if (
            not same_sparse_topology
            and self._previous_sparse_domains is not None
            and self._previous_sparse_domains.norbitals == domains.norbitals
        ):
            previous_left, previous_right = (
                self._previous_sparse_domains.allowed_pair_arrays()
            )
            current_left, current_right = domains.allowed_pair_arrays()
            same_sparse_topology = (
                np.array_equal(previous_left, current_left)
                and np.array_equal(previous_right, current_right)
            )
        sparse_restart = (
            not experimental_sparse_lmo
            and
            self._previous_sparse_projector is not None
            and self._previous_sparse_projector_state is not None
            and same_sparse_topology
            and self._previous_sparse_coordinates_bohr is not None
            and self._previous_sparse_coordinates_bohr.shape == coordinates.shape
            and float(np.max(np.linalg.norm(
                coordinates - self._previous_sparse_coordinates_bohr, axis=1
            ), initial=0.0)) <= 0.75
        )
        local_orbitals = None
        if experimental_sparse_lmo:
            from matrix_oracle import perceive_orbital_regions

            orthogonal_core, _ = model.orthogonalizer.congruence(
                model.core, domains,
                threshold=self.solver_policy.sparse_truncation_threshold,
            )
            plan = perceive_orbital_regions(
                coordinates / ANGSTROM_TO_BOHR,
                self._atomic_number_array,
            )
            lewis = build_sparse_lewis_orbital_seed(
                coordinates / ANGSTROM_TO_BOHR,
                basis.spherical_atom, basis.spherical_shell,
                plan.bonds, orthogonal_core, electrons // 2,
                site_charges=plan.electronegativity_charges,
                atomic_numbers=self.atomic_numbers,
                pi_bonds=oracle_carbonyl_pi_bonds(
                    plan, self.atomic_numbers,
                ),
            )
            local_orbitals = lewis.coefficients
            projector, _ = sparse_projector_from_local_orbitals(
                local_orbitals, electrons // 2, domains,
            )
            projector_state = None
            density_build = None
        elif sparse_restart:
            projector = self._previous_sparse_projector
            projector_state = self._previous_sparse_projector_state
            density_build = None
        else:
            density_build = sparse_closed_shell_density(
                model.core, model.orthogonalizer, domains, electrons // 2,
                tolerance=self.solver_policy.purification_tolerance,
                maximum_iterations=self.solver_policy.maximum_purification_iterations,
                truncation_threshold=self.solver_policy.sparse_truncation_threshold,
                build_ao_density=False,
            )
            if not density_build.converged:
                raise RuntimeError(
                    "initial sparse density did not certify "
                    f"(idempotency={density_build.idempotency_error:.3e}, "
                    f"bound={density_build.accumulated_truncation_bound:.3e})"
                )
            projector = density_build.orthogonal_projector
            projector_state = density_build.projector_state
        if not experimental_sparse_lmo:
            orthogonal_core, _ = model.orthogonalizer.congruence(
                model.core, domains,
                threshold=self.solver_policy.sparse_truncation_threshold,
            )
        interaction = shell_interaction_operator(coordinates, model.partition)
        operators = prepare_sparse_cumulative_multipole_operators(
            model.integrals, coordinates, basis.spherical_atom,
            compact=True,
        )
        valence = self._valence
        elements = self._element_values
        base_radius = self._base_multipole_radii
        valence_cn = self._valence_coordination
        effective_radius = base_radius + (5.0 - base_radius) / (
            1.0 + np.exp(-4.0 * (model.coordination - valence_cn - 1.2))
        )
        anisotropic_parameters = AnisotropicParameters(
            effective_radius,
            np.asarray([item.dipole_xc for item in elements]),
            np.asarray([item.quadrupole_xc for item in elements]),
        )
        anisotropic_geometry = prepare_anisotropic_geometry_sparse(
            coordinates, anisotropic_parameters, cutoff_bohr=64.0
        )
        en = self._electronegativities
        d4_cn, _ = d4_coordination_sparse(
            coordinates, _covalent_radii()[np.asarray(self.atomic_numbers) - 1], en,
            cutoff_bohr=40.0,
        )
        d4_models = self._d4_models
        d4_parameters = self._d4_parameters
        d4_prepared = prepare_d4_geometry_sparse(
            coordinates, d4_cn, d4_models, d4_parameters, cutoff_bohr=50.0
        )
        # A long sparse DIIS history duplicates large projector patterns.  For
        # large systems four vectors retain the useful local subspace while
        # keeping the working set bounded; small systems keep the fuller
        # history for difficult SCF branches.
        diis = SparseDIIS(
            maximum_size=4 if len(coordinates) >= 1000 else 8,
            store_residual_float32=len(coordinates) >= 1000,
            compact_indices=len(coordinates) >= 1000,
        )
        incremental_projector_updates = 0
        full_projector_rebuilds = int(not sparse_restart)
        incremental_projector_fallbacks = 0
        previous_energy = float("inf")
        converged = False
        sparse_scf_trace: list[dict[str, float | int | str]] = []
        cached_observables: tuple[np.ndarray, AtomicMultipoles] | None = None
        for iteration in range(1, self.maximum_scf_iterations + 1):
            if cached_observables is None:
                shell_q, multipoles = sparse_orthogonal_observables(
                    projector, model.orthogonalizer, domains, model.partition,
                    operators, valence,
                )
            else:
                shell_q, multipoles = cached_observables
                cached_observables = None
            if hasattr(interaction, "evaluate"):
                scc_evaluation = interaction.evaluate(
                    shell_q, compute_gradient=False
                )
                quadratic_potential = scc_evaluation.potential
                quadratic_energy = scc_evaluation.energy_hartree
            else:
                quadratic_potential = interaction @ shell_q
                quadratic_energy = 0.5 * float(shell_q @ quadratic_potential)
            anisotropic, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
                multipoles, coordinates, anisotropic_parameters,
                geometry=anisotropic_geometry,
            )
            d4 = d4_two_body_sparse(multipoles.charges, d4_prepared, d4_parameters)
            fock, _ = model.core.scaled_add(
                monopole_fock_sparse(
                    model.overlap,
                    shell_q,
                    interaction,
                    model.partition,
                    potential_override=quadratic_potential,
                )
            )
            fock, _ = fock.scaled_add(
                anisotropic_fock_sparse(
                    operators, qpot, mupot, thetapot, size=model.overlap.size
                )
            )
            fock, _ = fock.scaled_add(
                d4_fock_sparse(model.overlap, basis.spherical_atom, d4.charge_derivative_hartree)
            )
            effective_fock = fock
            if experimental_sparse_lmo:
                orthogonal_fock, fock_transform_error = (
                    model.orthogonalizer.congruence(
                        effective_fock, domains,
                        threshold=self.solver_policy.sparse_truncation_threshold,
                    )
                )
                for _ in range(2):
                    local_step = sparse_local_orbital_sweep(
                        local_orbitals, orthogonal_fock, electrons // 2,
                        maximum_rotation=0.1, coefficient_threshold=1.0e-8,
                    )
                    local_orbitals = local_step.coefficients
                self._last_sparse_lmo_coefficients = local_orbitals
                try:
                    local_projector, projector_error = sparse_projector_from_local_orbitals(
                        local_orbitals, electrons // 2, domains,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"local occupied projector failed at SCC cycle {iteration}: {exc}"
                    ) from exc
                updated_build = SparseDensityBuildResult(
                    SparseSymmetricMatrix(
                        local_projector.size,
                        np.empty(0, dtype=np.int32),
                        np.empty(0, dtype=np.int32), np.empty(0),
                    ),
                    local_projector, orthogonal_fock, 1,
                    local_step.orthogonality_error,
                    fock_transform_error + projector_error,
                    True, None, False, False,
                )
            else:
                updated_build = sparse_closed_shell_density(
                    effective_fock, model.orthogonalizer, domains, electrons // 2,
                    tolerance=self.solver_policy.purification_tolerance,
                    maximum_iterations=self.solver_policy.maximum_purification_iterations,
                    truncation_threshold=self.solver_policy.sparse_truncation_threshold,
                    projector_state=projector_state,
                    build_ao_density=False,
                )
            if not updated_build.converged:
                raise RuntimeError(
                    "sparse SP2 density did not certify "
                    f"(idempotency={updated_build.idempotency_error:.3e}, "
                    f"bound={updated_build.accumulated_truncation_bound:.3e})"
                )
            updated = updated_build.orthogonal_projector
            final_truncation_bound = updated_build.accumulated_truncation_bound
            final_orthogonal_fock = updated_build.orthogonal_fock
            projector_state = updated_build.projector_state
            incremental_projector_updates += int(
                updated_build.used_incremental_projector
            )
            incremental_projector_fallbacks += int(
                updated_build.incremental_fallback
            )
            full_projector_rebuilds += int(
                not updated_build.used_incremental_projector
            )
            residual_matrix, _ = updated.scaled_add(projector, other_scale=-1.0)
            residual = residual_matrix.maximum_absolute_difference(
                SparseSymmetricMatrix(
                    residual_matrix.size, np.empty(0, dtype=np.intp),
                    np.empty(0, dtype=np.intp), np.empty(0)
                )
            )
            if self.scf_convergence_mode == "tblite":
                updated_shell_q, updated_multipoles = sparse_orthogonal_observables(
                    updated, model.orthogonalizer, domains, model.partition,
                    operators, valence,
                )
                convergence_residual = _tblite_mixer_rms(
                    shell_q, multipoles, updated_shell_q, updated_multipoles
                )
                updated_anisotropic = anisotropic_energy_and_moment_potentials(
                    updated_multipoles, coordinates, anisotropic_parameters,
                    geometry=anisotropic_geometry,
                )[0]
                updated_d4 = d4_two_body_sparse(
                    updated_multipoles.charges, d4_prepared, d4_parameters
                )
                if hasattr(interaction, "evaluate"):
                    updated_quadratic_energy = interaction.evaluate(
                        updated_shell_q, compute_gradient=False
                    ).energy_hartree
                else:
                    updated_quadratic_energy = 0.5 * float(
                        updated_shell_q @ interaction @ updated_shell_q
                    )
                energy = 2.0 * orthogonal_core.frobenius_inner(updated)
                energy += monopole_energy(
                    updated_shell_q, interaction, model.partition,
                    quadratic_energy_override=updated_quadratic_energy,
                )
                energy += updated_anisotropic + updated_d4.energy_hartree
                convergence_tolerance = self.scf_mixer_tolerance
            else:
                convergence_residual = residual
                convergence_tolerance = self.scf_wavefunction_tolerance_electron
            if conservative_sparse_scf or experimental_sparse_lmo:
                target = updated
            else:
                diis.append(updated, residual_matrix)
                target = diis.extrapolate(threshold=0.0)
            restart_weight = (
                1.0 if experimental_sparse_lmo else
                0.3 if conservative_sparse_scf
                else (1.0 if sparse_restart else 0.85)
            )
            projector, mixing_discarded = projector.scaled_add(
                target,
                self_scale=1.0 - restart_weight,
                other_scale=restart_weight,
                threshold=self.solver_policy.sparse_truncation_threshold,
            )
            if (
                (conservative_sparse_scf or experimental_sparse_lmo)
                and self.scf_convergence_mode == "tblite"
                and mixing_discarded <= 1.0e-10
            ):
                keep = 0.7 if experimental_sparse_lmo else 1.0 - restart_weight
                new = 1.0 - keep
                cached_observables = (
                    keep * shell_q + new * updated_shell_q,
                    AtomicMultipoles(
                        keep * multipoles.charges
                        + new * updated_multipoles.charges,
                        keep * multipoles.dipoles_bohr
                        + new * updated_multipoles.dipoles_bohr,
                        keep * multipoles.quadrupoles_bohr2
                        + new * updated_multipoles.quadrupoles_bohr2,
                    ),
                )
            if self.scf_convergence_mode != "tblite":
                energy = 2.0 * orthogonal_core.frobenius_inner(projector)
                energy += monopole_energy(
                    shell_q,
                    interaction,
                    model.partition,
                    quadratic_energy_override=quadratic_energy,
                )
                energy += anisotropic + d4.energy_hartree
            energy_change = abs(energy - previous_energy)
            previous_energy = energy
            sparse_scf_trace.append({
                "cycle": iteration,
                "convergence_metric": self.scf_convergence_mode,
                "convergence_residual": convergence_residual,
                "energy_residual_hartree": energy_change,
                **({"orbital_residual": local_step.maximum_residual_after}
                   if experimental_sparse_lmo else {}),
            })
            if (
                convergence_residual < convergence_tolerance
                and energy_change < self.scf_energy_tolerance_hartree
                and (not experimental_sparse_lmo
                     or local_step.maximum_residual_after < 1.0e-7)
            ):
                projector = updated
                density_build = updated_build
                converged = True
                break
        if not converged:
            if experimental_sparse_lmo:
                self._last_scf_trace = tuple(sparse_scf_trace)
            raise RuntimeError("MATRIX-GFN2 sparse ISCF did not converge")
        self._last_scf_trace = tuple(sparse_scf_trace)
        # The final gradient only needs the converged projector and final Fock.
        # Release the iterative state before constructing derivative buffers.
        self._previous_sparse_projector = projector.copy()
        self._previous_sparse_projector_state = projector_state
        self._previous_sparse_domains = domains
        self._previous_sparse_coordinates_bohr = coordinates.copy()
        projector_state = None
        density_build = None
        del diis, updated_build, residual_matrix, target
        self._last_sparse_projector_statistics = {
            "incremental_updates": incremental_projector_updates,
            "full_rebuilds": full_projector_rebuilds,
            "incremental_fallbacks": incremental_projector_fallbacks,
        }
        shell_q, multipoles = sparse_orthogonal_observables(
            projector, model.orthogonalizer, domains, model.partition,
            operators, valence,
        )
        anisotropic = anisotropic_energy_and_moment_potentials(
            multipoles, coordinates, anisotropic_parameters, geometry=anisotropic_geometry
        )[0]
        d4 = d4_two_body_sparse(multipoles.charges, d4_prepared, d4_parameters)
        energy = 2.0 * orthogonal_core.frobenius_inner(projector)
        exact_scc = (
            interaction.evaluate_exact(shell_q, compute_gradient=False)
            if hasattr(interaction, "evaluate_exact")
            else None
        )
        energy += monopole_energy(
            shell_q,
            interaction,
            model.partition,
            quadratic_energy_override=(
                None if exact_scc is None else exact_scc.energy_hartree
            ),
        )
        energy += anisotropic + d4.energy_hartree
        energy += repulsion_energy_sparse(
            self.atomic_numbers, coordinates, model.atom_parameters, cutoff_bohr=24.0
        )
        neutral = d4_two_body_sparse(
            np.zeros(len(coordinates)), d4_prepared, d4_parameters
        )
        atm = atm_three_body_energy_gradient_sparse(
            coordinates, neutral,
            cutoff_bohr=50.0 if experimental_sparse_lmo else 25.0,
            s9=d4_parameters.s9, a1=d4_parameters.a1, a2=d4_parameters.a2,
        )
        energy += atm.energy_hartree
        def finish_sparse_gradient():
            doubled_projector = SparseSymmetricMatrix(
                projector.size, projector.rows, projector.columns,
                2.0 * projector.values,
                projector._native_topology,
            )
            density, _ = model.orthogonalizer.congruence(
                doubled_projector, domains,
                threshold=self.solver_policy.sparse_truncation_threshold,
            )
            weighted, _ = energy_weighted_density_from_fock_sparse(
                fock, projector, model.orthogonalizer, domains,
                threshold=self.solver_policy.sparse_truncation_threshold,
            )
            gradient = _analytic_gradient_pair_packed(
                density=density, weighted_density=weighted, basis=basis,
                coordinates=coordinates, atomic_numbers=self.atomic_numbers,
                atom_parameters=model.atom_parameters,
                orbital_parameters=model.orbital_parameters, partition=model.partition,
                shell_interaction=interaction, shell_q=shell_q, multipoles=multipoles,
                anisotropic_parameters=anisotropic_parameters,
                anisotropic_geometry=anisotropic_geometry, d4=d4,
                d4_parameters=d4_parameters, neutral_d4=(neutral, atm), valence=valence,
                integrals=model.integrals, spherical_domains=domains,
                element_values=self._element_values,
                base_multipole_radii=self._base_multipole_radii,
                valence_coordination=self._valence_coordination,
                electronegativities=self._electronegativities,
                native_context=self._active_native_context,
            )
            return density, gradient

        if self._sparse_allocation_gate is None:
            density, gradient = finish_sparse_gradient()
        else:
            with self._sparse_allocation_gate:
                density, gradient = finish_sparse_gradient()
        bond_orders = (
            sparse_mayer_bond_orders(density, model.overlap, basis.spherical_atom)
            if compute_bond_orders else np.empty((0, 0), dtype=float)
        )
        sparse_state = None
        if retain_projectors:
            domain_rows, domain_columns = domains.allowed_pair_arrays()
            sparse_state = GFN2SparseProjectorState(
                size=projector.size,
                occupied_orbitals=electrons // 2,
                projector_rows=projector.rows,
                projector_columns=projector.columns,
                projector_values=projector.values,
                fock_rows=final_orthogonal_fock.rows,
                fock_columns=final_orthogonal_fock.columns,
                fock_values=final_orthogonal_fock.values,
                domain_rows=domain_rows,
                domain_columns=domain_columns,
                ao_atom=np.asarray(basis.spherical_atom, dtype=np.int64),
                atomic_numbers=self._atomic_number_array,
                coordinates_bohr=coordinates,
                truncation_bound=float(final_truncation_bound),
            )
        return GFN2Result(
            float(energy), gradient, multipoles.charges, bond_orders, density,
            iteration, True, np.empty((0, 3)), None, None,
            "sparse-lmo" if experimental_sparse_lmo else "sparse-sp2", None,
            sparse_projector_state=sparse_state,
        )

    def density_response_kernel(
        self, state: GFN2OrbitalState
    ):
        """Build the analytic Fock--density Jacobian at a retained SCF state."""

        if tuple(int(value) for value in state.atomic_numbers) != self.atomic_numbers:
            raise ValueError("orbital state and MATRIX-GFN2 calculator use different atoms")
        coordinates = np.asarray(state.coordinates_bohr, dtype=float)
        (
            basis,
            overlap,
            dipole,
            second,
            _core,
            partition,
            _atom_parameters,
            _orbital_parameters,
            _symmetry_adapter,
            _symmetry_diagnostics,
        ) = self._model(coordinates, relocate_basis=True)
        if not np.allclose(overlap, state.overlap, atol=2.0e-10, rtol=0.0):
            raise ValueError("orbital state metric does not match the response geometry")
        operators = (
            prepare_cumulative_multipole_operators(
                overlap, dipole, second, coordinates, basis.spherical_atom
            )
            if dipole is not None and second is not None
            else prepare_pair_cumulative_multipole_operators(
                self._post_scf_pair_integrals, coordinates, basis.spherical_atom
            )
        )
        occupations = state.alpha_occupations + state.beta_occupations
        density = (state.coefficients * occupations) @ state.coefficients.T
        shell_q, multipoles = dense_scc_observables(
            density,
            operators,
            partition.orbital_shell,
            partition.reference_population,
            self._valence,
        )
        coordination = self._active_coordination
        if coordination is None:
            raise RuntimeError("MATRIX-GFN2 response lost its coordination state")
        nn_state = getattr(self, "_active_nn_state", None)
        if nn_state is None:
            effective_radii = self._base_multipole_radii + (
                5.0 - self._base_multipole_radii
            ) / (
                1.0
                + np.exp(
                    -4.0
                    * (coordination - self._valence_coordination - 1.2)
                )
            )
            anisotropic_parameters = AnisotropicParameters(
                effective_radii,
                np.asarray([item.dipole_xc for item in self._element_values]),
                np.asarray([item.quadrupole_xc for item in self._element_values]),
            )
        else:
            anisotropic_parameters = nn_state.anisotropic_parameters(coordination)
        anisotropic_geometry = prepare_anisotropic_geometry(
            coordinates, anisotropic_parameters
        )
        d4_parameters = self._d4_parameters
        if len(coordinates) >= 64 and nn_state is None:
            d4_cn, _ = d4_coordination_sparse(
                coordinates,
                _covalent_radii()[self._atomic_number_array - 1],
                self._electronegativities,
                cutoff_bohr=24.0,
            )
            d4_prepared = prepare_d4_geometry_sparse(
                coordinates,
                d4_cn,
                self._d4_models,
                d4_parameters,
                cutoff_bohr=24.0,
            )
        else:
            d4_cn = _d4_coordination(
                self.atomic_numbers,
                coordinates,
                _covalent_radii(),
                self._electronegativities,
            )
            d4_prepared = prepare_d4_geometry(
                coordinates, d4_cn, self._d4_models, d4_parameters
            )
        atm_charge_hessian = None
        if nn_state is not None and len(coordinates) >= 3:
            pair_d4 = d4_two_body(
                coordinates, multipoles.charges, d4_cn,
                self._d4_models, d4_parameters, prepared=d4_prepared,
            )
            atm_charge_hessian = atm_three_body_charge_hessian_dense(
                coordinates, multipoles.charges, d4_cn,
                self._d4_models, d4_parameters, pair_d4,
            )
        from .response import GFN2DensityResponseKernel

        response_interaction = shell_interaction_operator(
            coordinates, partition,
            alphaj=(2.0 if nn_state is None else nn_state.globals["alphaj"]),
        )
        if self.alpb_parameters is not None:
            atomic_interaction, _ = alpb_scc_interaction_matrix_hartree(
                coordinates, np.asarray(self.atomic_numbers), self.alpb_parameters,
            )
            shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
            response_interaction = np.asarray(response_interaction) + atomic_interaction[
                np.ix_(shell_atom, shell_atom)
            ]
        response_interaction = wrap_scc_interaction(response_interaction, coordinates, partition)
        return GFN2DensityResponseKernel(
            overlap=overlap,
            operators=operators,
            partition=partition,
            shell_interaction=response_interaction,
            reference_shell_charges=shell_q,
            reference_multipoles=multipoles,
            coordinates_bohr=coordinates,
            anisotropic_parameters=anisotropic_parameters,
            anisotropic_geometry=anisotropic_geometry,
            d4_prepared=d4_prepared,
            d4_parameters=d4_parameters,
            d4_models=self._d4_models,
            d4_coordination=d4_cn,
            atm_charge_hessian=atm_charge_hessian,
        )


    def _fixed_density_fock_for_validation(
        self, coordinates_bohr: np.ndarray, density: np.ndarray
    ) -> tuple[SparseSymmetricMatrix, SparseSymmetricMatrix]:
        """Materialize the physical Fock at fixed AO density for derivative tests."""

        coordinates = np.asarray(coordinates_bohr, dtype=float)
        (
            basis,
            overlap,
            dipole,
            second,
            core,
            partition,
            _atom_parameters,
            _orbital_parameters,
            _symmetry_adapter,
            _symmetry_diagnostics,
        ) = self._model(coordinates, relocate_basis=True)
        operators = (
            prepare_cumulative_multipole_operators(
                overlap, dipole, second, coordinates, basis.spherical_atom
            )
            if dipole is not None and second is not None
            else prepare_pair_cumulative_multipole_operators(
                self._post_scf_pair_integrals, coordinates, basis.spherical_atom
            )
        )
        shell_q, multipoles = dense_scc_observables(
            density,
            operators,
            partition.orbital_shell,
            partition.reference_population,
            self._valence,
        )
        interaction = shell_interaction_operator(coordinates, partition)
        quadratic = (
            interaction.evaluate(shell_q, compute_gradient=False).potential
            if hasattr(interaction, "evaluate")
            else np.asarray(interaction) @ shell_q
        )
        shell_potential = (
            quadratic + np.asarray(partition.third_order_hartree) * shell_q**2
        )
        coordination = self._active_coordination
        effective_radii = self._base_multipole_radii + (
            5.0 - self._base_multipole_radii
        ) / (1.0 + np.exp(-4.0 * (coordination - self._valence_coordination - 1.2)))
        anisotropic_parameters = AnisotropicParameters(
            effective_radii,
            np.asarray([item.dipole_xc for item in self._element_values]),
            np.asarray([item.quadrupole_xc for item in self._element_values]),
        )
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            multipoles, coordinates, anisotropic_parameters
        )
        d4_parameters = self._d4_parameters
        d4_cn = _d4_coordination(
            self.atomic_numbers,
            coordinates,
            _covalent_radii(),
            self._electronegativities,
        )
        prepared = prepare_d4_geometry(
            coordinates, d4_cn, self._d4_models, d4_parameters
        )
        d4 = d4_two_body(
            coordinates,
            multipoles.charges,
            d4_cn,
            self._d4_models,
            d4_parameters,
            prepared=prepared,
        )
        return (
            dense_scc_fock(
                core,
                overlap,
                operators,
                partition.orbital_shell,
                shell_potential,
                qpot,
                mupot,
                thetapot,
                d4.charge_derivative_hartree,
            ),
            overlap,
        )


    def density_response_coordinate_vjp(
        self,
        state: GFN2OrbitalState,
        response_density: np.ndarray | SparseSymmetricMatrix,
        response_weighted_density: np.ndarray | SparseSymmetricMatrix,
        *,
        _context: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Contract pair-packed coordinate kernels with a Z-vector density."""

        context = (
            self._prepare_density_response_gradient_context(state)
            if _context is None
            else _context
        )
        kernel = context["kernel"]
        coordinates = context["coordinates"]
        basis = context["basis"]
        partition = context["partition"]
        atom_parameters = context["atom_parameters"]
        orbital_parameters = context["orbital_parameters"]
        integrals = context["integrals"]
        domains = context["domains"]
        reference_density = context["reference_density"]
        sparse_operators = context["sparse_operators"]
        overlap_sparse = context["overlap_sparse"]
        shell_index = context["shell_index"]
        atom_index = context["atom_index"]
        pair_multiplicity = context["pair_multiplicity"]
        radius_derivative = context["radius_derivative"]
        reference_d4 = context["reference_d4"]
        zeros = context["zeros"]
        pz = (
            response_density
            if isinstance(response_density, SparseSymmetricMatrix)
            else SparseSymmetricMatrix.from_dense(
                np.asarray(response_density, dtype=float), domains, threshold=0.0
            )
        )
        wz = (
            response_weighted_density
            if isinstance(response_weighted_density, SparseSymmetricMatrix)
            else SparseSymmetricMatrix.from_dense(
                np.asarray(response_weighted_density, dtype=float),
                domains,
                threshold=0.0,
            )
        )
        if pz.size != integrals.size or wz.size != integrals.size:
            raise ValueError("sparse response matrices do not match the AO basis")

        delta_shell = shell_charges_sparse(pz, overlap_sparse, partition) - np.asarray(
            partition.reference_population
        )
        delta_multipoles = cumulative_atomic_multipoles_sparse(
            pz, sparse_operators, np.zeros(len(coordinates))
        )
        reference_shell = kernel.reference_shell_charges
        reference_multipoles = kernel.reference_multipoles
        interaction = kernel.shell_interaction
        if hasattr(interaction, "evaluate"):
            quadratic_delta = interaction.evaluate(
                delta_shell, compute_gradient=False
            ).potential
        else:
            quadratic_delta = np.asarray(interaction) @ delta_shell
        delta_shell_potential = quadratic_delta + (
            2.0
            * np.asarray(partition.third_order_hartree)
            * reference_shell
            * delta_shell
        )
        delta_d4_potential = d4_charge_hessian_matvec(
            reference_multipoles.charges,
            delta_multipoles.charges,
            kernel.d4_prepared,
        )
        if kernel.atm_charge_hessian is not None:
            delta_d4_potential += kernel.atm_charge_hessian @ delta_multipoles.charges
        _, delta_qpot, delta_mupot, delta_thetapot = (
            anisotropic_energy_and_moment_potentials(
                delta_multipoles,
                coordinates,
                kernel.anisotropic_parameters,
                geometry=kernel.anisotropic_geometry,
            )
        )
        camm_left, camm_right = anisotropic_fock_derivatives_sparse(
            integrals,
            sparse_operators,
            coordinates,
            delta_qpot,
            delta_mupot,
            delta_thetapot,
        )
        delta_scalar = (
            delta_shell_potential[shell_index] + delta_d4_potential[atom_index]
        )
        pair_scalar = -0.5 * (
            delta_scalar[integrals.rows] + delta_scalar[integrals.columns]
        )
        camm_left += pair_scalar[:, None] * integrals.left_derivatives[:, 0]
        camm_right += pair_scalar[:, None] * integrals.right_derivatives[:, 0]

        pair_values = reference_density.values_on(integrals.rows, integrals.columns)
        mixed = np.zeros_like(coordinates)
        left_contribution = (
            pair_multiplicity[:, None] * pair_values[:, None] * camm_left
        )
        right_contribution = (
            pair_multiplicity[:, None] * pair_values[:, None] * camm_right
        )
        np.add.at(mixed, atom_index[integrals.rows], left_contribution)
        np.add.at(mixed, atom_index[integrals.columns], right_contribution)

        nn_state = getattr(self, "_active_nn_state", None)
        alphaj = 2.0 if nn_state is None else nn_state.globals["alphaj"]
        def scc_explicit(charges):
            return frozen_charge_scc_gradient(
                coordinates, charges, partition,
                interaction=interaction, alphaj=alphaj,
            )
        mixed += (
            scc_explicit(reference_shell + delta_shell)
            - scc_explicit(reference_shell)
            - scc_explicit(delta_shell)
        )
        if self.alpb_parameters is not None:
            shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
            delta_atomic = np.bincount(
                shell_atom, weights=delta_shell, minlength=len(coordinates),
            )
            mixed_charge = self._alpb_mixed_geometry_charge(context)
            mixed += (mixed_charge @ delta_atomic).reshape(coordinates.shape)

        def add_multipoles(left, right):
            return AtomicMultipoles(
                left.charges + right.charges,
                left.dipoles_bohr + right.dipoles_bohr,
                left.quadrupoles_bohr2 + right.quadrupoles_bohr2,
            )

        def anisotropic_explicit(multipoles):
            active_nn = getattr(self, "_active_nn_state", None)
            coordination_radii = (
                self._coordination_radii if active_nn is None
                else np.asarray([value.covalent_radius_bohr for value in active_nn.atoms])
            )
            return anisotropic_explicit_gradient(
                multipoles,
                coordinates,
                kernel.anisotropic_parameters,
                None,
                geometry=kernel.anisotropic_geometry,
                coordination_radii_bohr=coordination_radii,
                radius_cn_derivative=radius_derivative,
            )

        mixed += (
            anisotropic_explicit(add_multipoles(reference_multipoles, delta_multipoles))
            - anisotropic_explicit(reference_multipoles)
            - anisotropic_explicit(delta_multipoles)
        )

        d4_direction = d4_charge_direction(
            reference_multipoles.charges,
            delta_multipoles.charges,
            kernel.d4_prepared,
        )
        displacement = coordinates[d4_direction.rows] - coordinates[d4_direction.columns]
        distance = np.linalg.norm(displacement, axis=1)
        ratio = d4_direction.c8_derivative / np.where(
            np.abs(d4_direction.c6_derivative) > 1.0e-300,
            d4_direction.c6_derivative,
            1.0,
        )
        ratio = np.where(
            np.abs(d4_direction.c6_derivative) > 1.0e-300,
            ratio,
            (
                kernel.d4_prepared.ratio
                if hasattr(kernel.d4_prepared, "rows")
                else kernel.d4_prepared.ratio[
                    d4_direction.rows, d4_direction.columns
                ]
            ),
        )
        cutoff = kernel.d4_parameters.a1 * np.sqrt(ratio) + kernel.d4_parameters.a2
        radial = (
            kernel.d4_parameters.s6
            * d4_direction.c6_derivative
            * 6.0
            * distance**5
            / (distance**6 + cutoff**6) ** 2
            + kernel.d4_parameters.s8
            * d4_direction.c8_derivative
            * 8.0
            * distance**7
            / (distance**8 + cutoff**8) ** 2
        )
        radial_vector = radial[:, None] * displacement / distance[:, None]
        np.add.at(mixed, d4_direction.rows, radial_vector)
        np.add.at(mixed, d4_direction.columns, -radial_vector)
        mixed += coordination_response_gradient_sparse(
            d4_direction.coordination_potential_derivative,
            coordinates,
            _covalent_radii()[self._atomic_number_array - 1],
            electronegativities=self._electronegativities,
        )
        if context.get("atm_charge_geometry") is not None:
            mixed += np.einsum("a,aij->ij", delta_multipoles.charges,
                               context["atm_charge_geometry"])

        return _analytic_gradient_pair_packed(
            density=pz,
            weighted_density=wz,
            basis=basis,
            coordinates=coordinates,
            atomic_numbers=self.atomic_numbers,
            atom_parameters=atom_parameters,
            orbital_parameters=orbital_parameters,
            partition=partition,
            shell_interaction=interaction,
            shell_q=reference_shell,
            multipoles=reference_multipoles,
            anisotropic_parameters=kernel.anisotropic_parameters,
            anisotropic_geometry=kernel.anisotropic_geometry,
            d4=reference_d4,
            d4_parameters=kernel.d4_parameters,
            neutral_d4=None,
            valence=self._valence,
            integrals=integrals,
            spherical_domains=domains,
            element_values=self._element_values,
            base_multipole_radii=self._base_multipole_radii,
            valence_coordination=self._valence_coordination,
            electronegativities=self._electronegativities,
            orbital_metadata=self._orbital_metadata,
            coordination_override=self._active_coordination,
            nn_global_scalings=(
                None if getattr(self, "_active_nn_state", None) is None
                else self._active_nn_state.globals
            ),
            native_context=self._active_native_context,
            repulsion_gradient_override=zeros,
            atm_gradient_override=zeros,
            post_scf_explicit_gradient_override=mixed,
        )

    def _alpb_mixed_geometry_charge(self, context: dict[str, object]) -> np.ndarray:
        """Cache the exact ALPB E_Rq block for all response contractions."""

        if self.alpb_parameters is None:
            raise RuntimeError("ALPB response requested without a solvent")
        cached = context.get("_alpb_mixed_geometry_charge")
        if cached is not None:
            return cached
        coordinates = np.asarray(context["coordinates"], dtype=float)
        charges = np.asarray(context["kernel"].reference_multipoles.charges)
        natoms = len(coordinates)
        # Forward mode carries one charge tangent per atom.  Bound its peak
        # tensor size on large systems; the batched reverse route remains exact.
        if natoms <= 256:
            mixed_charge = alpb_mixed_geometry_charge_hartree(
                coordinates, np.asarray(self.atomic_numbers), charges,
                self.alpb_parameters,
            )
            context["_alpb_mixed_geometry_charge"] = mixed_charge
            return mixed_charge
        mixed_charge = np.empty((3 * natoms, natoms))
        identity = np.eye(natoms)
        operator = self._alpb_hessian_operator(context)
        batch_size = max(1, min(32, 8192 // natoms))
        for start in range(0, natoms, batch_size):
            stop = min(start + batch_size, natoms)
            columns, _ = operator.matvec_batch(
                np.zeros((stop - start, natoms, 3)), identity[start:stop],
            )
            mixed_charge[:, start:stop] = columns.reshape(stop - start, -1).T
        context["_alpb_mixed_geometry_charge"] = mixed_charge
        return mixed_charge

    def _alpb_hessian_operator(self, context: dict[str, object]) -> ALPBHessianOperator:
        cached = context.get("_alpb_hessian_operator")
        if cached is not None:
            return cached
        if self.alpb_parameters is None:
            raise RuntimeError("ALPB Hessian operator requested without a solvent")
        operator = ALPBHessianOperator(
            np.asarray(context["coordinates"], dtype=float),
            np.asarray(self.atomic_numbers),
            np.asarray(context["kernel"].reference_multipoles.charges),
            self.alpb_parameters,
        )
        context["_alpb_hessian_operator"] = operator
        return operator

    def sparse_density_response_kernel(
        self,
        state: GFN2OrbitalState,
        *,
        _context: dict[str, object] | None = None,
    ):
        """Return the common-pattern pair-packed SCF Jacobian."""

        from .response import SparseGFN2DensityResponseKernel

        context = (
            self._prepare_density_response_gradient_context(state)
            if _context is None
            else _context
        )
        kernel = context["kernel"]
        return SparseGFN2DensityResponseKernel(
            overlap=context["overlap_sparse"],
            operators=context["sparse_operators"],
            partition=context["partition"],
            shell_interaction=kernel.shell_interaction,
            reference_shell_charges=kernel.reference_shell_charges,
            reference_multipoles=kernel.reference_multipoles,
            coordinates_bohr=context["coordinates"],
            anisotropic_parameters=kernel.anisotropic_parameters,
            anisotropic_geometry=kernel.anisotropic_geometry,
            d4_prepared=kernel.d4_prepared,
            orbital_atom=np.asarray(context["basis"].spherical_atom, dtype=np.int64),
        )

    def _prepare_density_response_gradient_context(
        self, state: GFN2OrbitalState
    ) -> dict[str, object]:
        """Prepare geometry-only response intermediates once for all roots."""

        kernel = self.density_response_kernel(state)
        coordinates = np.asarray(state.coordinates_bohr, dtype=float)
        (
            basis,
            _overlap,
            _dipole,
            _second,
            _core,
            partition,
            atom_parameters,
            orbital_parameters,
            _symmetry_adapter,
            _symmetry_diagnostics,
        ) = self._model(coordinates, relocate_basis=True)
        integrals = self._post_scf_pair_integrals
        domains = self._post_scf_pair_domains
        if integrals is None or domains is None:
            raise RuntimeError("pair-packed response context is unavailable")
        occupations = state.alpha_occupations + state.beta_occupations
        reference_density_array = (
            state.coefficients * occupations
        ) @ state.coefficients.T
        reference_density = SparseSymmetricMatrix.from_dense(
            reference_density_array, domains, threshold=0.0
        )
        sparse_operators = prepare_sparse_cumulative_multipole_operators(
            integrals, coordinates, basis.spherical_atom, compact=True
        )
        overlap_sparse = SparseSymmetricMatrix(
            integrals.size,
            integrals.rows,
            integrals.columns,
            integrals.overlap,
        )
        shell_index = np.asarray(partition.orbital_shell, dtype=np.intp)
        atom_index = np.asarray(basis.spherical_atom, dtype=np.intp)
        pair_multiplicity = np.where(integrals.rows == integrals.columns, 1.0, 2.0)
        nn_state = getattr(self, "_active_nn_state", None)
        aes_exponent = 4.0 if nn_state is None else nn_state.globals["aesexp"]
        aes_shift = 1.2 if nn_state is None else nn_state.globals["aesshift"]
        aes_max_radius = 5.0 if nn_state is None else nn_state.globals["aesrmax"]
        logistic_argument = -aes_exponent * (
            self._active_coordination - self._valence_coordination - aes_shift
        )
        logistic = np.exp(-np.logaddexp(0.0, logistic_argument))
        radius_derivative = (
            aes_exponent * (aes_max_radius - self._base_multipole_radii)
            * logistic * (1.0 - logistic)
        )

        reference_multipoles = kernel.reference_multipoles
        zeros = np.zeros_like(coordinates)
        reference_d4 = kernel._d4(reference_multipoles.charges)
        atm_charge_geometry = None
        pair_charge_radial = None
        if nn_state is not None:
            # Reuse the pair/ATM coefficient and HVP kernels to form E_qR.
            # The ATM term is charge dependent on the NN surface, unlike the
            # neutral post-SCF ATM convention of the baseline surface.
            n = len(coordinates)
            pair_charge_radial = np.zeros((n, n, 3))
            atm_charge_geometry = np.zeros_like(pair_charge_radial)
            pair = reference_d4
            rows, columns = triangular_indices(n, diagonal=False)
            sparse_pair = SparseD4Result(
                pair.energy_hartree, pair.charge_derivative_hartree,
                pair.coordination_derivative_hartree, rows, columns,
                pair.c6[rows, columns], pair.c8[rows, columns],
                pair.c6_coordination_derivative[rows, columns, rows],
                pair.c6_coordination_derivative[rows, columns, columns], 0.0,
            )
            parameters = kernel.d4_parameters
            atm_state = atm_three_body_energy_sparse(
                coordinates, sparse_pair, s9=parameters.s9,
                a1=parameters.a1, a2=parameters.a2,
            )
            displacement = coordinates[rows] - coordinates[columns]
            distance = np.linalg.norm(displacement, axis=1)
            ratio = kernel.d4_prepared.ratio[rows, columns]
            radius = parameters.a1 * np.sqrt(ratio) + parameters.a2
            for atom in range(n):
                dq = np.eye(n)[atom]
                coefficient = d4_coefficient_direction(
                    reference_multipoles.charges, dq, kernel.d4_coordination,
                    np.zeros(n), self._d4_models, kernel.d4_prepared,
                )
                radial = (parameters.s6 * coefficient.c6_derivative * 6 * distance**5
                          / (distance**6 + radius**6)**2
                          + parameters.s8 * coefficient.c8_derivative * 8 * distance**7
                          / (distance**8 + radius**8)**2
                          + parameters.s10 * (49. / 40.) * ratio**2
                          * coefficient.c6_derivative * 10 * distance**9
                          / (distance**10 + radius**10)**2)
                vector = radial[:, None] * displacement / distance[:, None]
                np.add.at(pair_charge_radial[atom], rows, vector)
                np.add.at(pair_charge_radial[atom], columns, -vector)
                atm_charge_geometry[atom] = atm_three_body_hessian_vector_product_sparse(
                    coordinates, zeros, sparse_pair, coefficient, atm_state,
                    _covalent_radii()[self._atomic_number_array - 1],
                    self._electronegativities, s9=(0.0 if parameters.neutral_atm else parameters.s9),
                    a1=parameters.a1, a2=parameters.a2,
                )
            if parameters.neutral_atm:
                atm_charge_geometry.fill(0.0)
            _, atm_potential = atm_three_body_energy_charge_derivative_dense(
                coordinates, reference_multipoles.charges, kernel.d4_coordination,
                self._d4_models, parameters, pair,
            )
            reference_d4 = replace(pair, charge_derivative_hartree=(
                pair.charge_derivative_hartree + atm_potential))
        return {
            "kernel": kernel,
            "coordinates": coordinates,
            "basis": basis,
            "partition": partition,
            "atom_parameters": atom_parameters,
            "orbital_parameters": orbital_parameters,
            "integrals": integrals,
            "domains": domains,
            "reference_density": reference_density,
            "sparse_operators": sparse_operators,
            "overlap_sparse": overlap_sparse,
            "shell_index": shell_index,
            "atom_index": atom_index,
            "pair_multiplicity": pair_multiplicity,
            "radius_derivative": radius_derivative,
            "reference_d4": reference_d4,
            "atm_charge_geometry": atm_charge_geometry,
            "pair_charge_radial": pair_charge_radial,
            "zeros": zeros,
        }

    def ground_state_derivative_matrices(
        self, state: GFN2OrbitalState
    ):
        """Materialize analytic fixed-density AO derivatives for CPSCF.

        This dense routine is the small-system qualification reference.  It
        reconstructs ``dF/dR`` from the already certified pair-packed
        coordinate VJP, so core, SCC, multipole, CN, and D4 terms have one
        derivative implementation rather than two drifting code paths.
        """

        from .ground_state_response import GroundStateDerivativeMatrices

        context = self._prepare_density_response_gradient_context(state)
        integrals = context["integrals"]
        basis = context["basis"]
        atom_index = np.asarray(basis.spherical_atom, dtype=np.intp)
        natoms = len(state.atomic_numbers)
        overlap_derivative = np.zeros((natoms, 3, state.nao, state.nao))
        for pair, (row, column) in enumerate(
            zip(integrals.rows, integrals.columns, strict=True)
        ):
            derivative = np.zeros((natoms, 3), dtype=float)
            derivative[atom_index[row]] += integrals.left_derivatives[pair, 0]
            derivative[atom_index[column]] += integrals.right_derivatives[pair, 0]
            overlap_derivative[:, :, row, column] = derivative
            overlap_derivative[:, :, column, row] = derivative

        fock_derivative = np.zeros_like(overlap_derivative)
        zero_weighted = np.zeros((state.nao, state.nao), dtype=float)
        for row in range(state.nao):
            for column in range(row + 1):
                seed = np.zeros((state.nao, state.nao), dtype=float)
                if row == column:
                    seed[row, column] = 1.0
                else:
                    seed[row, column] = 0.5
                    seed[column, row] = 0.5
                derivative = self.density_response_coordinate_vjp(
                    state, seed, zero_weighted, _context=context
                )
                fock_derivative[:, :, row, column] = derivative
                fock_derivative[:, :, column, row] = derivative
        return GroundStateDerivativeMatrices(overlap_derivative, fock_derivative)

    def _ground_state_directional_derivative_matrices_sparse(
        self,
        state: GFN2OrbitalState,
        direction_bohr: np.ndarray,
        *,
        context: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build pair-packed ``S'[v]`` and fixed-density ``F'[v]``.

        The same operator is shared by ground- and excited-state response.
        Point-charge embedding is differentiated with respect to the QM
        coordinates here, so an asymmetric solvent field never disappears
        from a directional response calculation.
        """

        direction = np.asarray(direction_bohr, dtype=float)
        coordinates = np.asarray(context["coordinates"], dtype=float)
        if direction.shape != coordinates.shape:
            raise ValueError("directional AO derivative requires natoms x 3")
        kernel = context["kernel"]
        basis = context["basis"]
        integrals = context["integrals"]
        density = context["reference_density"]
        sparse_operators = context["sparse_operators"]
        partition = context["partition"]
        atom_index = np.asarray(basis.spherical_atom, dtype=np.intp)
        left_atom = atom_index[integrals.rows]
        right_atom = atom_index[integrals.columns]
        overlap_direction = (
            np.einsum(
                "pk,pk->p", integrals.left_derivatives[:, 0], direction[left_atom]
            )
            + np.einsum(
                "pk,pk->p", integrals.right_derivatives[:, 0], direction[right_atom]
            )
        )
        overlap_sparse = SparseSymmetricMatrix(
            integrals.size,
            integrals.rows,
            integrals.columns,
            overlap_direction,
        )
        coordination = np.asarray(self._active_coordination, dtype=float)
        coordination_direction = coordination_number_directional_derivative_sparse(
            coordinates,
            direction,
            self._coordination_radii,
            cutoff_bohr=24.0,
        )
        h0 = extended_hueckel_hamiltonian_derivatives_sparse(
            integrals,
            self._orbital_metadata,
            self.atomic_numbers,
            coordinates,
            self._atom_parameters,
            self._orbital_parameters,
            coordination,
        )
        h0_direction = (
            np.einsum("pk,pk->p", h0.left, direction[left_atom])
            + np.einsum("pk,pk->p", h0.right, direction[right_atom])
            + h0.coordination_left * coordination_direction[left_atom]
            + h0.coordination_right * coordination_direction[right_atom]
        )
        multipole_direction = cumulative_atomic_multipole_directional_derivative_sparse(
            density,
            integrals,
            coordinates,
            direction,
            basis.spherical_atom,
        )
        exponential = np.exp(
            -4.0 * (coordination - self._valence_coordination - 1.2)
        )
        radius_derivative = (
            4.0
            * (5.0 - self._base_multipole_radii)
            * exponential
            / (1.0 + exponential) ** 2
        )
        reference_multipoles = kernel.reference_multipoles
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            reference_multipoles,
            coordinates,
            kernel.anisotropic_parameters,
            geometry=kernel.anisotropic_geometry,
        )
        qpot_dot, mupot_dot, thetapot_dot = anisotropic_moment_potential_direction(
            reference_multipoles,
            multipole_direction,
            coordinates,
            direction,
            kernel.anisotropic_parameters,
            radius_derivative * coordination_direction,
            geometry=kernel.anisotropic_geometry,
        )
        anisotropic_direction = anisotropic_fock_directional_sparse(
            integrals,
            sparse_operators,
            coordinates,
            direction,
            qpot,
            mupot,
            thetapot,
            qpot_dot,
            mupot_dot,
            thetapot_dot,
        )
        pair_density = density.values_on(integrals.rows, integrals.columns)
        shell_index = np.asarray(partition.orbital_shell, dtype=np.intp)
        left_shell = shell_index[integrals.rows]
        right_shell = shell_index[integrals.columns]
        shell_charge_direction = np.zeros(partition.nshells)
        np.add.at(
            shell_charge_direction,
            left_shell,
            -pair_density * overlap_direction,
        )
        off_diagonal = integrals.rows != integrals.columns
        np.add.at(
            shell_charge_direction,
            right_shell[off_diagonal],
            -pair_density[off_diagonal] * overlap_direction[off_diagonal],
        )
        shell_charge = np.asarray(kernel.reference_shell_charges, dtype=float)
        interaction = kernel.shell_interaction
        shell_potential = interaction @ shell_charge + np.asarray(
            partition.third_order_hartree
        ) * shell_charge**2
        if hasattr(interaction, "potential_direction"):
            quadratic_potential_direction = interaction.potential_direction(
                shell_charge, shell_charge_direction, direction
            )
        else:
            shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
            hardness = np.asarray(partition.hardness_hartree, dtype=float)
            displacement = (
                coordinates[shell_atom, None, :]
                - coordinates[np.newaxis, shell_atom, :]
            )
            relative = (
                direction[shell_atom, None, :]
                - direction[np.newaxis, shell_atom, :]
            )
            average = 0.5 * (hardness[:, None] + hardness[None, :])
            softened = np.einsum("ijk,ijk->ij", displacement, displacement) + average**-2
            gamma_direction = -np.einsum(
                "ijk,ijk->ij", displacement, relative
            ) * softened**-1.5
            quadratic_potential_direction = (
                interaction @ shell_charge_direction
                + gamma_direction @ shell_charge
            )
        shell_potential_direction = (
            quadratic_potential_direction
            + 2.0
            * np.asarray(partition.third_order_hartree)
            * shell_charge
            * shell_charge_direction
        )
        if self.alpb_parameters is not None:
            shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
            alpb_potential_direction = (
                self._alpb_mixed_geometry_charge(context).T @ direction.ravel()
            )
            shell_potential_direction += alpb_potential_direction[shell_atom]
        d4_radii = _covalent_radii()[self._atomic_number_array - 1]
        d4_coordination = _d4_coordination(
            self.atomic_numbers,
            coordinates,
            _covalent_radii(),
            self._electronegativities,
        )
        d4_coordination_direction = coordination_number_directional_derivative_sparse(
            coordinates,
            direction,
            d4_radii,
            electronegativities=self._electronegativities,
            cutoff_bohr=24.0,
        )
        d4_coefficient = d4_coefficient_direction(
            reference_multipoles.charges,
            multipole_direction.charges,
            d4_coordination,
            d4_coordination_direction,
            self._d4_models,
            kernel.d4_prepared,
        )
        d4 = context["reference_d4"]
        orbital_scalar = shell_potential[shell_index] + d4.charge_derivative_hartree[
            atom_index
        ]
        orbital_scalar_direction = (
            shell_potential_direction[shell_index]
            + d4_coefficient.charge_potential_derivative[atom_index]
            + d4_charge_potential_radial_direction(
                coordinates,
                direction,
                reference_multipoles.charges,
                kernel.d4_prepared,
                kernel.d4_parameters,
            )[atom_index]
        )
        scalar = -0.5 * (
            overlap_direction
            * (
                orbital_scalar[integrals.rows]
                + orbital_scalar[integrals.columns]
            )
            + integrals.overlap
            * (
                orbital_scalar_direction[integrals.rows]
                + orbital_scalar_direction[integrals.columns]
            )
        )
        embedding_direction = np.zeros_like(overlap_direction)
        if len(self.point_charges_electron):
            points = np.asarray(self.point_charge_coordinates_bohr, dtype=float)
            charges = np.asarray(self.point_charges_electron, dtype=float)
            displacement = coordinates[:, None, :] - points[None, :, :]
            distance = np.linalg.norm(displacement, axis=2)
            if np.any(distance < 1.0e-8):
                raise ValueError("point charge coincides with a QM atom")
            potential = (1.0 / distance) @ charges
            potential_direction = -np.sum(
                charges[None, :]
                * np.einsum("apk,ak->ap", displacement, direction)
                / distance**3,
                axis=1,
            )
            left_potential = potential[left_atom]
            right_potential = potential[right_atom]
            left_direction = potential_direction[left_atom]
            right_direction = potential_direction[right_atom]
            embedding_direction = -0.5 * (
                overlap_direction * (left_potential + right_potential)
                + integrals.overlap * (left_direction + right_direction)
            )
        fock_values = (
            h0_direction
            + anisotropic_direction.values
            + scalar
            + embedding_direction
        )
        fock_sparse = SparseSymmetricMatrix(
            integrals.size,
            integrals.rows,
            integrals.columns,
            fock_values,
        )
        return overlap_sparse, fock_sparse

    def _ground_state_directional_derivative_matrices(
        self,
        state: GFN2OrbitalState,
        direction_bohr: np.ndarray,
        *,
        context: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dense qualification boundary for the pair-packed derivative path."""

        overlap, fock = self._ground_state_directional_derivative_matrices_sparse(
            state, direction_bohr, context=context
        )
        return overlap.to_dense(), fock.to_dense()

    def ground_state_coupled_response(
        self,
        state: GFN2OrbitalState,
        *,
        degeneracy_tolerance_hartree: float = 1.0e-8,
    ):
        """Return analytic zero-temperature CPSCF response of the ground state."""

        from .ground_state_response import solve_ground_state_coupled_response

        derivatives = self.ground_state_derivative_matrices(state)
        return solve_ground_state_coupled_response(
            state,
            derivatives,
            self.density_response_kernel(state),
            degeneracy_tolerance_hartree=degeneracy_tolerance_hartree,
        )

    def ground_state_response_hessian(
        self,
        state: GFN2OrbitalState,
        *,
        degeneracy_tolerance_hartree: float = 1.0e-8,
    ):
        """Return the analytic orbital-response/Pulay Hessian block.

        The explicit fixed-density second nuclear derivative is intentionally
        not included.  Consequently this method cannot be used by vibrational
        analysis as a complete Hessian.
        """

        from .ground_state_response import GroundStateResponseHessian

        response = self.ground_state_coupled_response(
            state,
            degeneracy_tolerance_hartree=degeneracy_tolerance_hartree,
        )
        context = self._prepare_density_response_gradient_context(state)
        dimension = 3 * len(state.atomic_numbers)
        hessian = np.empty((dimension, dimension), dtype=float)
        for coordinate in range(dimension):
            atom, component = divmod(coordinate, 3)
            column = self.density_response_coordinate_vjp(
                state,
                response.density_derivative[atom, component],
                response.weighted_density_derivative[atom, component],
                _context=context,
            )
            hessian[:, coordinate] = column.reshape(-1)
        reciprocity = float(np.max(np.abs(hessian - hessian.T), initial=0.0))
        return GroundStateResponseHessian(
            hessian,
            reciprocity,
            float(np.max(response.residual_norms, initial=0.0)),
        )

    def ground_state_explicit_pair_hvp(
        self,
        state: GFN2OrbitalState,
        direction_bohr: np.ndarray,
        *,
        _context: dict[str, object] | None = None,
        _integral_directional_seconds: tuple[np.ndarray, np.ndarray] | None = None,
        _alpb_geometry_hvp: np.ndarray | None = None,
    ):
        """Apply the analytic explicit fixed-density R,R block."""

        from .gradients import (
            assemble_variational_hessian_vector_sparse,
            coordination_number_directional_derivative_sparse,
            coordination_response_gradient_sparse,
            coordination_response_hessian_vector_product_sparse,
            frozen_charge_scc_hessian_vector_product,
            frozen_d4_two_body_hessian_vector_product,
            frozen_d4_two_body_hessian_vector_product_sparse,
            repulsion_hessian_vector_product_sparse,
        )
        from .ground_state_response import GroundStateExplicitPairHVP
        from .hamiltonian import (
            extended_hueckel_hamiltonian_derivatives_sparse,
            extended_hueckel_hamiltonian_directional_derivatives_sparse,
        )

        direction = np.asarray(direction_bohr, dtype=float)
        coordinates = np.asarray(state.coordinates_bohr, dtype=float)
        if direction.shape != coordinates.shape or not np.all(np.isfinite(direction)):
            raise ValueError("explicit Hessian direction must have shape natoms x 3")
        context = (
            self._prepare_density_response_gradient_context(state)
            if _context is None
            else _context
        )
        kernel = context["kernel"]
        nn_state = getattr(self, "_active_nn_state", None)
        nn_globals = None if nn_state is None else nn_state.globals
        atom_parameters = context["atom_parameters"]
        orbital_parameters = context["orbital_parameters"]
        coordination_radii = np.asarray([
            value.covalent_radius_bohr for value in atom_parameters
        ]) if nn_state is not None else self._coordination_radii
        d4 = context["reference_d4"]
        parameters = kernel.d4_parameters
        basis = context["basis"]
        integrals = context["integrals"]
        domains = context["domains"]
        atom_index = np.asarray(basis.spherical_atom, dtype=np.intp)
        cartesian_centers = np.asarray(
            [item.center_bohr for item in basis.cartesian_orbitals], dtype=float
        )
        cartesian_direction = direction[
            [item.atom_index for item in basis.cartesian_orbitals]
        ]
        if self._active_native_context is None:
            raise RuntimeError("analytic AO Hessian requires the native integral context")
        self._active_native_context.configure_spherical(
            basis.spherical_cartesian_offsets,
            basis.spherical_cartesian_indices,
            basis.spherical_cartesian_coefficients,
            integrals.rows,
            integrals.columns,
        )
        if _integral_directional_seconds is None:
            integral_left_directional, integral_right_directional = (
                self._active_native_context.spherical_directional_seconds(
                    cartesian_centers,
                    np.zeros(3),
                    cartesian_direction,
                    domains.native_threads,
                )
            )
        else:
            integral_left_directional, integral_right_directional = (
                _integral_directional_seconds
            )
        coordination = np.asarray(self._active_coordination, dtype=float)
        coordination_direction = coordination_number_directional_derivative_sparse(
            coordinates,
            direction,
            coordination_radii,
            cutoff_bohr=24.0,
        )
        multipole_direction = cumulative_atomic_multipole_directional_derivative_sparse(
            context["reference_density"],
            integrals,
            coordinates,
            direction,
            basis.spherical_atom,
        )
        h0 = extended_hueckel_hamiltonian_derivatives_sparse(
            integrals,
            self._orbital_metadata,
            self.atomic_numbers,
            coordinates,
            atom_parameters,
            orbital_parameters,
            coordination,
            global_scalings=nn_globals,
        )
        h0_directional = extended_hueckel_hamiltonian_directional_derivatives_sparse(
            integrals,
            integral_left_directional[:, 0],
            integral_right_directional[:, 0],
            self._orbital_metadata,
            self.atomic_numbers,
            coordinates,
            direction,
            atom_parameters,
            orbital_parameters,
            coordination,
            coordination_direction,
            global_scalings=nn_globals,
        )
        aes_exponent = 4.0 if nn_globals is None else nn_globals["aesexp"]
        aes_shift = 1.2 if nn_globals is None else nn_globals["aesshift"]
        aes_max_radius = 5.0 if nn_globals is None else nn_globals["aesrmax"]
        logistic = np.exp(-np.logaddexp(
            0.0, -aes_exponent * (
                coordination - self._valence_coordination - aes_shift
            ),
        ))
        radius_derivative = (
            aes_exponent * (aes_max_radius - self._base_multipole_radii)
            * logistic * (1.0 - logistic)
        )
        radius_second = radius_derivative * aes_exponent * (1.0 - 2.0 * logistic)
        reference_multipoles = kernel.reference_multipoles
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            reference_multipoles,
            coordinates,
            kernel.anisotropic_parameters,
            geometry=kernel.anisotropic_geometry,
        )
        qpot_dot, mupot_dot, thetapot_dot = anisotropic_moment_potential_direction(
            reference_multipoles,
            multipole_direction,
            coordinates,
            direction,
            kernel.anisotropic_parameters,
            radius_derivative * coordination_direction,
            geometry=kernel.anisotropic_geometry,
        )
        camm_left_directional, camm_right_directional = (
            anisotropic_fock_derivatives_directional_sparse(
                integrals,
                context["sparse_operators"],
                coordinates,
                direction,
                integral_left_directional,
                integral_right_directional,
                qpot,
                mupot,
                thetapot,
                qpot_dot,
                mupot_dot,
                thetapot_dot,
            )
        )
        electronic_left_directional = h0_directional.left + camm_left_directional
        electronic_right_directional = h0_directional.right + camm_right_directional
        occupations = state.alpha_occupations + state.beta_occupations
        weighted_array = (
            state.coefficients * (occupations * state.orbital_energies_hartree)
        ) @ state.coefficients.T
        weighted = SparseSymmetricMatrix.from_dense(
            weighted_array, domains, threshold=0.0
        )
        density = context["reference_density"]
        pair_density = density.values_on(integrals.rows, integrals.columns)
        multiplicity = np.where(integrals.rows == integrals.columns, 1.0, 2.0)
        left_atom = atom_index[integrals.rows]
        right_atom = atom_index[integrals.columns]
        overlap_direction = (
            np.einsum(
                "pk,pk->p",
                integrals.left_derivatives[:, 0],
                direction[left_atom],
            )
            + np.einsum(
                "pk,pk->p",
                integrals.right_derivatives[:, 0],
                direction[right_atom],
            )
        )
        shell_index = np.asarray(context["partition"].orbital_shell, dtype=np.intp)
        left_shell = shell_index[integrals.rows]
        right_shell = shell_index[integrals.columns]
        shell_charge_direction = np.zeros(context["partition"].nshells)
        np.add.at(
            shell_charge_direction,
            left_shell,
            -pair_density * overlap_direction,
        )
        off_diagonal = integrals.rows != integrals.columns
        np.add.at(
            shell_charge_direction,
            right_shell[off_diagonal],
            -pair_density[off_diagonal] * overlap_direction[off_diagonal],
        )
        cn_coefficient = np.zeros(len(coordinates), dtype=float)
        cn_coefficient_direction = np.zeros_like(cn_coefficient)
        np.add.at(
            cn_coefficient,
            atom_index[integrals.rows],
            multiplicity * pair_density * h0.coordination_left,
        )
        np.add.at(
            cn_coefficient,
            atom_index[integrals.columns],
            multiplicity * pair_density * h0.coordination_right,
        )
        np.add.at(
            cn_coefficient_direction,
            atom_index[integrals.rows],
            multiplicity * pair_density * h0_directional.coordination_left,
        )
        np.add.at(
            cn_coefficient_direction,
            atom_index[integrals.columns],
            multiplicity * pair_density * h0_directional.coordination_right,
        )
        shell_charge = np.asarray(kernel.reference_shell_charges, dtype=float)
        shell_atom = np.asarray(context["partition"].shell_atom, dtype=np.intp)
        hardness = np.asarray(context["partition"].hardness_hartree, dtype=float)
        interaction = kernel.shell_interaction
        shell_potential = interaction @ shell_charge + np.asarray(
            context["partition"].third_order_hartree
        ) * shell_charge**2
        if hasattr(interaction, "potential_direction"):
            quadratic_potential_direction = interaction.potential_direction(
                shell_charge, shell_charge_direction, direction
            )
        else:
            shell_coordinates = coordinates[shell_atom]
            shell_displacement = (
                shell_coordinates[:, None, :] - shell_coordinates[None, :, :]
            )
            shell_direction = direction[shell_atom]
            shell_relative_direction = (
                shell_direction[:, None, :] - shell_direction[None, :, :]
            )
            shell_average = 0.5 * (hardness[:, None] + hardness[None, :])
            alphaj = 2.0 if nn_globals is None else nn_globals["alphaj"]
            shell_distance2 = np.einsum(
                "ijk,ijk->ij", shell_displacement, shell_displacement
            )
            shell_distance = np.sqrt(shell_distance2)
            softened = shell_distance**alphaj + shell_average**-alphaj
            gamma_direction = -np.einsum(
                "ijk,ijk->ij", shell_displacement, shell_relative_direction
            ) * np.where(
                shell_distance > 0.0,
                np.maximum(shell_distance, 1.0e-30) ** (alphaj - 2.0), 0.0,
            ) * softened ** (-1.0 / alphaj - 1.0)
            quadratic_potential_direction = (
                interaction @ shell_charge_direction
                + gamma_direction @ shell_charge
            )
        shell_potential_direction = (
            quadratic_potential_direction
            + 2.0
            * np.asarray(context["partition"].third_order_hartree)
            * shell_charge
            * shell_charge_direction
        )
        if self.alpb_parameters is not None:
            shell_atom = np.asarray(context["partition"].shell_atom, dtype=np.intp)
            alpb_potential_direction = (
                self._alpb_mixed_geometry_charge(context).T @ direction.ravel()
            )
            shell_potential_direction += alpb_potential_direction[shell_atom]
        h0_cn_product = coordination_response_hessian_vector_product_sparse(
            cn_coefficient,
            coordinates,
            direction,
            coordination_radii,
            cutoff_bohr=24.0,
        ) + coordination_response_gradient_sparse(
            cn_coefficient_direction,
            coordinates,
            coordination_radii,
            cutoff_bohr=24.0,
        )
        d4_radii = _covalent_radii()[self._atomic_number_array - 1]
        d4_coordination = _d4_coordination(
            self.atomic_numbers,
            coordinates,
            _covalent_radii(),
            self._electronegativities,
        )
        d4_coordination_direction = coordination_number_directional_derivative_sparse(
            coordinates,
            direction,
            d4_radii,
            electronegativities=self._electronegativities,
            cutoff_bohr=24.0,
        )
        d4_coefficient = d4_coefficient_direction(
            reference_multipoles.charges,
            multipole_direction.charges,
            d4_coordination,
            d4_coordination_direction,
            self._d4_models,
            kernel.d4_prepared,
        )
        orbital_scalar = shell_potential[shell_index] + d4.charge_derivative_hartree[
            atom_index
        ]
        radial_charge_potential_direction = (
            d4_charge_potential_radial_direction(
                coordinates,
                direction,
                reference_multipoles.charges,
                kernel.d4_prepared,
                parameters,
            ) if nn_state is None else np.zeros(len(coordinates))
        )
        orbital_scalar_direction = (
            shell_potential_direction[shell_index]
            + d4_coefficient.charge_potential_derivative[atom_index]
            + radial_charge_potential_direction[atom_index]
        )
        if nn_state is not None:
            dispersion_potential_direction = np.einsum(
                "aij,ij->a", context["pair_charge_radial"]
                + context["atm_charge_geometry"], direction,
            )
            if kernel.atm_charge_hessian is not None:
                dispersion_potential_direction += (
                    kernel.atm_charge_hessian @ multipole_direction.charges)
            orbital_scalar_direction += dispersion_potential_direction[atom_index]
        pair_scalar = -0.5 * (
            orbital_scalar[integrals.rows] + orbital_scalar[integrals.columns]
        )
        pair_scalar_direction = -0.5 * (
            orbital_scalar_direction[integrals.rows]
            + orbital_scalar_direction[integrals.columns]
        )
        electronic_left_directional += (
            pair_scalar[:, None] * integral_left_directional[:, 0]
            + pair_scalar_direction[:, None] * integrals.left_derivatives[:, 0]
        )
        electronic_right_directional += (
            pair_scalar[:, None] * integral_right_directional[:, 0]
            + pair_scalar_direction[:, None] * integrals.right_derivatives[:, 0]
        )
        ao_h0 = assemble_variational_hessian_vector_sparse(
            density,
            weighted,
            integrals.rows,
            integrals.columns,
            atom_index[integrals.rows],
            atom_index[integrals.columns],
            electronic_left_directional,
            electronic_right_directional,
            integral_left_directional[:, 0],
            integral_right_directional[:, 0],
            h0_cn_product,
            native_threads=domains.native_threads,
        )
        repulsion = repulsion_hessian_vector_product_sparse(
            self.atomic_numbers,
            coordinates,
            direction,
            atom_parameters,
            cutoff_bohr=24.0,
            kexp=(1.5 if nn_globals is None else nn_globals["kexp"]),
            klight=(1.0 if nn_globals is None else nn_globals["klight"]),
        )
        frozen_scc = frozen_charge_scc_hessian_vector_product(
            coordinates,
            direction,
            kernel.reference_shell_charges,
            context["partition"],
            interaction=kernel.shell_interaction,
            charge_direction=shell_charge_direction,
            alphaj=(2.0 if nn_globals is None else nn_globals["alphaj"]),
        )
        if self.alpb_parameters is not None:
            shell_atom = np.asarray(context["partition"].shell_atom, dtype=np.intp)
            atomic_charge_direction = np.bincount(
                shell_atom, weights=shell_charge_direction,
                minlength=len(coordinates),
            )
            if _alpb_geometry_hvp is None:
                alpb_explicit, _ = self._alpb_hessian_operator(context).matvec(
                    direction, atomic_charge_direction,
                )
            else:
                mixed = self._alpb_mixed_geometry_charge(context)
                alpb_explicit = (
                    np.asarray(_alpb_geometry_hvp, dtype=float)
                    + (mixed @ atomic_charge_direction).reshape(-1, 3)
                )
            frozen_scc += alpb_explicit
        d4_options = {
            "s6": parameters.s6,
            "s8": parameters.s8, "s10": parameters.s10,
            "a1": parameters.a1,
            "a2": parameters.a2,
        }
        if hasattr(d4, "rows"):
            frozen_d4 = frozen_d4_two_body_hessian_vector_product_sparse(
                coordinates, direction, d4, **d4_options
            )
        else:
            frozen_d4 = frozen_d4_two_body_hessian_vector_product(
                coordinates, direction, d4.c6, d4.c8, **d4_options
            )
        frozen_d4_cn = coordination_response_hessian_vector_product_sparse(
            d4.coordination_derivative_hartree,
            coordinates,
            direction,
            _covalent_radii()[self._atomic_number_array - 1],
            electronegativities=self._electronegativities,
            cutoff_bohr=24.0,
        )
        anisotropic = anisotropic_explicit_hessian_vector_product_sparse(
            reference_multipoles,
            multipole_direction,
            coordinates,
            direction,
            kernel.anisotropic_parameters,
            coordination_radii,
            radius_derivative,
            radius_second,
            coordination_direction,
            geometry=kernel.anisotropic_geometry,
        )
        def sparse_pairs(result, coefficient):
            if hasattr(result, "rows"):
                return result
            rows, columns = coefficient.rows, coefficient.columns
            return SparseD4Result(
                result.energy_hartree,
                result.charge_derivative_hartree,
                result.coordination_derivative_hartree,
                rows,
                columns,
                result.c6[rows, columns],
                result.c8[rows, columns],
                result.c6_coordination_derivative[rows, columns, rows],
                result.c6_coordination_derivative[rows, columns, columns],
                0.0,
            )

        pair_d4 = sparse_pairs(d4, d4_coefficient)
        full_d4 = d4_two_body_hessian_vector_product_sparse(
            coordinates,
            direction,
            pair_d4,
            d4_coefficient,
            d4_coordination_direction,
            d4_radii,
            self._electronegativities,
            **d4_options,
        )
        d4_coefficient_response = full_d4 - frozen_d4 - frozen_d4_cn
        if len(coordinates) >= 3:
            neutral_pairs = context.get("_hessian_neutral_pairs")
            if neutral_pairs is None:
                if hasattr(kernel.d4_prepared, "rows"):
                    neutral_pairs = d4_two_body_sparse(
                        np.zeros(len(coordinates)), kernel.d4_prepared, parameters
                    )
                else:
                    neutral_dense = d4_two_body(
                        coordinates,
                        np.zeros(len(coordinates)),
                        d4_coordination,
                        self._d4_models,
                        parameters,
                        prepared=kernel.d4_prepared,
                    )
                    neutral_coefficient_zero = d4_coefficient_direction(
                        np.zeros(len(coordinates)),
                        np.zeros(len(coordinates)),
                        d4_coordination,
                        np.zeros(len(coordinates)),
                        self._d4_models,
                        kernel.d4_prepared,
                    )
                    neutral_pairs = sparse_pairs(neutral_dense, neutral_coefficient_zero)
                context["_hessian_neutral_pairs"] = neutral_pairs
            neutral_coefficient = d4_coefficient_direction(
                np.zeros(len(coordinates)),
                np.zeros(len(coordinates)),
                d4_coordination,
                d4_coordination_direction,
                self._d4_models,
                kernel.d4_prepared,
            )
            if nn_state is not None and not parameters.neutral_atm:
                neutral_pairs = pair_d4
                neutral_coefficient = d4_coefficient
            atm_state = context.get("_hessian_atm_state")
            if atm_state is None or (nn_state is not None and not parameters.neutral_atm):
                atm_state = atm_three_body_energy_sparse(
                    coordinates,
                    neutral_pairs,
                    s9=parameters.s9,
                    a1=parameters.a1,
                    a2=parameters.a2,
                )
                if nn_state is None or parameters.neutral_atm:
                    context["_hessian_atm_state"] = atm_state
            atm = atm_three_body_hessian_vector_product_sparse(
                coordinates,
                direction,
                neutral_pairs,
                neutral_coefficient,
                atm_state,
                d4_radii,
                self._electronegativities,
                s9=parameters.s9,
                a1=parameters.a1,
                a2=parameters.a2,
            )
        else:
            atm = np.zeros_like(coordinates)
        return GroundStateExplicitPairHVP(
            ao_h0,
            repulsion,
            frozen_scc,
            frozen_d4,
            frozen_d4_cn,
            anisotropic,
            d4_coefficient_response,
            atm,
        )

    def ground_state_hessian(
        self,
        state: GFN2OrbitalState,
        *,
        degeneracy_tolerance_hartree: float = 1.0e-8,
    ):
        """Materialize the complete Hessian without a large dense CPSCF solve."""

        occupations = state.alpha_occupations + state.beta_occupations
        active_pairs = int(np.count_nonzero(np.triu(
            np.abs(occupations[:, None] - occupations[None, :]) > 1.0e-12,
            k=1,
        )))
        if active_pairs > 4096:
            return self.ground_state_hessian_matrix_free(
                state,
                degeneracy_tolerance_hartree=degeneracy_tolerance_hartree,
            )

        from .ground_state_response import GroundStateHessian

        response = self.ground_state_response_hessian(
            state,
            degeneracy_tolerance_hartree=degeneracy_tolerance_hartree,
        )
        dimension = 3 * len(state.atomic_numbers)
        explicit = np.empty((dimension, dimension), dtype=float)
        identity = np.eye(dimension)
        context = self._prepare_density_response_gradient_context(state)
        basis = context["basis"]
        integrals = context["integrals"]
        domains = context["domains"]
        if self._active_native_context is None and getattr(self, "_active_nn_state", None) is not None:
            cartesian_rows, cartesian_columns = triangular_indices(
                len(basis.cartesian_orbitals)
            )
            self._active_native_context = prepare_native_integral_context(
                basis.cartesian_orbitals, cartesian_rows, cartesian_columns,
            )
        cartesian_centers = np.asarray(
            [item.center_bohr for item in basis.cartesian_orbitals], dtype=float
        )
        cartesian_atom = np.asarray(
            [item.atom_index for item in basis.cartesian_orbitals], dtype=np.intp
        )
        self._active_native_context.configure_spherical(
            basis.spherical_cartesian_offsets,
            basis.spherical_cartesian_indices,
            basis.spherical_cartesian_coefficients,
            integrals.rows,
            integrals.columns,
        )
        coordinate_directions = identity.reshape(
            dimension, len(state.atomic_numbers), 3
        )
        # Keep the batched AO-second-derivative workspace bounded.  The full
        # Cartesian Hessian is O(N^2), but an unbounded [3N,npair,13,3] buffer
        # would introduce an avoidable cubic peak for larger molecules.
        bytes_per_direction = max(1, 2 * len(integrals.rows) * 13 * 3 * 8)
        batch_size = max(1, min(
            dimension, (128 * 1024**2) // bytes_per_direction
        ))
        for start in range(0, dimension, batch_size):
            stop = min(dimension, start + batch_size)
            directional_left, directional_right = (
                self._active_native_context.spherical_directional_seconds(
                    cartesian_centers,
                    np.zeros(3),
                    coordinate_directions[start:stop, cartesian_atom, :],
                    domains.native_threads,
                )
            )
            for local, coordinate in enumerate(range(start, stop)):
                explicit[:, coordinate] = self.ground_state_explicit_pair_hvp(
                    state,
                    identity[:, coordinate].reshape(-1, 3),
                    _context=context,
                    _integral_directional_seconds=(
                        directional_left[local],
                        directional_right[local],
                    ),
                ).total.reshape(-1)
        raw = explicit + response.response_hessian_hartree_per_bohr2
        reciprocity = float(np.max(np.abs(raw - raw.T), initial=0.0))
        hessian = 0.5 * (raw + raw.T)
        blocks = hessian.reshape(
            len(state.atomic_numbers), 3, len(state.atomic_numbers), 3
        )
        translational = float(
            np.max(np.abs(np.sum(blocks, axis=2)), initial=0.0)
        )
        return GroundStateHessian(
            hessian,
            0.5 * (explicit + explicit.T),
            0.5
            * (
                response.response_hessian_hartree_per_bohr2
                + response.response_hessian_hartree_per_bohr2.T
            ),
            reciprocity,
            translational,
            response.maximum_cpscf_residual,
        )

    def ground_state_hessian_matrix_free(
        self,
        state: GFN2OrbitalState,
        *,
        cpscf_tolerance: float = 1.0e-10,
        cpscf_maximum_iterations: int = 200,
        degeneracy_tolerance_hartree: float = 1.0e-8,
        workers: int = 1,
    ):
        """Assemble Cartesian columns using iterative CPSCF and batched AO seconds."""

        from .ground_state_response import GroundStateHessian

        if workers < 1:
            raise ValueError("analytic Hessian worker count must be positive")
        workers = _memory_limited_worker_count(workers, len(self.atoms))
        if workers > 1:
            dimension = 3 * len(state.atomic_numbers)
            explicit = np.empty((dimension, dimension), dtype=float)
            response = np.empty_like(explicit)
            maximum_residual = 0.0
            response_options = {
                "cpscf_tolerance": cpscf_tolerance,
                "cpscf_maximum_iterations": cpscf_maximum_iterations,
                "degeneracy_tolerance_hartree": degeneracy_tolerance_hartree,
            }
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_hessian_process,
                initargs=(
                    self._hessian_process_options(),
                    state.coordinates_bohr / ANGSTROM_TO_BOHR,
                    response_options,
                ),
            ) as executor:
                if self.alpb_parameters is None:
                    columns = executor.map(
                        _hessian_process_column, range(dimension), chunksize=2,
                    )
                else:
                    batch_size = max(1, min(16, 4096 // len(self.atoms)))
                    batches = (
                        tuple(range(start, min(start + batch_size, dimension)))
                        for start in range(0, dimension, batch_size)
                    )
                    columns = (
                        item for batch in executor.map(
                            _hessian_process_columns_alpb, batches,
                        ) for item in batch
                    )
                for coordinate, column_explicit, column_response, residual in columns:
                    explicit[:, coordinate] = column_explicit
                    response[:, coordinate] = column_response
                    maximum_residual = max(maximum_residual, residual)
            raw = explicit + response
            reciprocity = float(np.max(np.abs(raw - raw.T), initial=0.0))
            hessian = 0.5 * (raw + raw.T)
            blocks = hessian.reshape(len(state.atomic_numbers), 3, len(state.atomic_numbers), 3)
            translational = float(np.max(np.abs(np.sum(blocks, axis=2)), initial=0.0))
            return GroundStateHessian(
                hessian,
                0.5 * (explicit + explicit.T),
                0.5 * (response + response.T),
                reciprocity,
                translational,
                maximum_residual,
            )

        operator = self.ground_state_hessian_operator(
            state,
            cpscf_tolerance=cpscf_tolerance,
            cpscf_maximum_iterations=cpscf_maximum_iterations,
            degeneracy_tolerance_hartree=degeneracy_tolerance_hartree,
        )
        context = operator.context
        basis = context["basis"]
        integrals = context["integrals"]
        domains = context["domains"]
        if self._active_native_context is None:
            raise RuntimeError("analytic AO Hessian requires the native integral context")
        self._active_native_context.configure_spherical(
            basis.spherical_cartesian_offsets,
            basis.spherical_cartesian_indices,
            basis.spherical_cartesian_coefficients,
            integrals.rows,
            integrals.columns,
        )
        cartesian_centers = np.asarray(
            [item.center_bohr for item in basis.cartesian_orbitals], dtype=float
        )
        cartesian_atom = np.asarray(
            [item.atom_index for item in basis.cartesian_orbitals], dtype=np.intp
        )
        dimension = 3 * len(state.atomic_numbers)
        directions = np.eye(dimension).reshape(dimension, -1, 3)
        explicit = np.empty((dimension, dimension), dtype=float)
        response = np.empty_like(explicit)
        maximum_residual = 0.0
        bytes_per_direction = max(1, 2 * len(integrals.rows) * 13 * 3 * 8)
        batch_size = max(1, min(
            dimension, (128 * 1024**2) // bytes_per_direction
        ))
        for start in range(0, dimension, batch_size):
            stop = min(dimension, start + batch_size)
            directional_left, directional_right = (
                self._active_native_context.spherical_directional_seconds(
                    cartesian_centers,
                    np.zeros(3),
                    directions[start:stop, cartesian_atom, :],
                    domains.native_threads,
                )
            )
            for local, coordinate in enumerate(range(start, stop)):
                column_explicit, column_response, residual = (
                    operator.components_matvec(
                        directions[coordinate].reshape(-1),
                        integral_directional_seconds=(
                            directional_left[local], directional_right[local],
                        ),
                    )
                )
                explicit[:, coordinate] = column_explicit
                response[:, coordinate] = column_response
                maximum_residual = max(maximum_residual, residual)
        raw = explicit + response
        reciprocity = float(np.max(np.abs(raw - raw.T), initial=0.0))
        hessian = 0.5 * (raw + raw.T)
        blocks = hessian.reshape(
            len(state.atomic_numbers), 3, len(state.atomic_numbers), 3
        )
        translational = float(np.max(np.abs(np.sum(blocks, axis=2)), initial=0.0))
        return GroundStateHessian(
            hessian,
            0.5 * (explicit + explicit.T),
            0.5 * (response + response.T),
            reciprocity,
            translational,
            maximum_residual,
        )

    def ground_state_hessian_operator(
        self,
        state: GFN2OrbitalState,
        *,
        cpscf_tolerance: float = 1.0e-10,
        cpscf_maximum_iterations: int = 200,
        degeneracy_tolerance_hartree: float = 1.0e-8,
    ):
        """Return the primary matrix-free analytic Hessian operator."""

        from .ground_state_response import GroundStateHessianOperator

        context = self._prepare_density_response_gradient_context(state)
        return GroundStateHessianOperator(
            self,
            state,
            context,
            cpscf_tolerance=cpscf_tolerance,
            cpscf_maximum_iterations=cpscf_maximum_iterations,
            degeneracy_tolerance_hartree=degeneracy_tolerance_hartree,
        )

    def ground_state_vibrational_property_operator(
        self,
        state: GFN2OrbitalState,
        *,
        hessian_operator=None,
        magnetic_origin_bohr=None,
    ):
        """Return directional analytic IR/Raman property response."""

        from .vibrational_properties import GroundStateVibrationalPropertyOperator

        return GroundStateVibrationalPropertyOperator(
            self,
            state,
            hessian_operator=hessian_operator,
            magnetic_origin_bohr=magnetic_origin_bohr,
        )

    def density_response_coordinate_vjp_batch(
        self,
        state: GFN2OrbitalState,
        response_densities: Sequence[np.ndarray | SparseSymmetricMatrix],
        response_weighted_densities: Sequence[
            np.ndarray | SparseSymmetricMatrix
        ],
    ) -> np.ndarray:
        """Contract multiple roots while reusing one persistent geometry context."""

        if len(response_densities) != len(response_weighted_densities):
            raise ValueError("density-response root counts differ")
        context = self._prepare_density_response_gradient_context(state)
        return np.stack(
            tuple(
                self.density_response_coordinate_vjp(
                    state, density, weighted, _context=context
                )
                for density, weighted in zip(
                    response_densities, response_weighted_densities, strict=True
                )
            ),
            axis=0,
        )


    def sf_tda_orbital_relaxation_gradient(
        self,
        state: GFN2OrbitalState,
        result,
        *,
        selection_policy=None,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 200,
    ):
        """Return the matrix-free SF-TDA Z-vector and its gradient term.

        Each root is contracted directly with the resident pair-packed
        AO/CN/SCC/CAMM/D4 derivative kernels.  The returned gradient is the
        orbital-relaxation contribution to be added to the frozen-orbital
        SF-TDA derivative; no AO Fock-derivative tensor is materialized.
        """

        from .sf_tda_gradient import matrix_free_sf_tda_z_vector

        response = matrix_free_sf_tda_z_vector(
            state,
            result,
            self.density_response_kernel(state),
            selection_policy=selection_policy,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
        )
        gradients = self.density_response_coordinate_vjp_batch(
            state,
            response.response_density,
            response.response_weighted_density,
        )
        return response, gradients

    def evaluate_tda_two_photon_gradient(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        target_state: int = 0,
        spectral_range_ev: float = 7.0,
        maximum_configuration_energy_ev: float | None = None,
        response_tolerance: float = 1.0e-9,
        maximum_response_iterations: int = 200,
        damping_hartree: float = 0.0,
    ):
        """Return the complete analytic molecular TDA two-photon gradient."""

        from .stda import grimme_xtb_stda
        from .tda_two_photon import molecular_tda_two_photon_gradient

        ground = self.evaluate(
            coordinates_angstrom,
            retain_orbitals=True,
            compute_cm5=False,
            compute_bond_orders=False,
        )
        state = ground.orbital_state
        if state is None:
            raise RuntimeError("two-photon gradient requires retained orbitals")
        excited = grimme_xtb_stda(
            state,
            roots=100000,
            spectral_range_ev=float(spectral_range_ev),
            maximum_configuration_energy_ev=maximum_configuration_energy_ev,
        )
        return molecular_tda_two_photon_gradient(
            self,
            state,
            excited,
            int(target_state),
            spectral_range_ev=float(spectral_range_ev),
            maximum_configuration_energy_ev=maximum_configuration_energy_ev,
            tolerance=float(response_tolerance),
            maximum_iterations=int(maximum_response_iterations),
            damping_hartree=float(damping_hartree),
        )

    def closed_shell_stda_hessian(
        self,
        state: GFN2OrbitalState,
        *,
        excited=None,
        root: int = 0,
        roots: int = 6,
        spectral_range_ev: float = 7.0,
        maximum_configuration_energy_ev: float | None = None,
        response_tolerance: float = 1.0e-10,
    ):
        """Complete analytic molecular sTDA Hessian, with a zero-based root.

        Returns an operator with ``matvec``, ``materialize`` and ``project``.
        The selected configuration space is frozen. Gas-phase baseline
        closed-shell GFN2 and its ALPB solvent are supported; fixed external
        point charges, NN overlays and unresolved orbital degeneracies are
        rejected by this molecular reference implementation.
        """
        from .stda_hessian import MolecularSTDAHessian

        return MolecularSTDAHessian(
            self, state, excited=excited, root=root, roots=roots,
            spectral_range_ev=spectral_range_ev,
            maximum_configuration_energy_ev=maximum_configuration_energy_ev,
            response_tolerance=response_tolerance,
        )

    def sf_tda_hessian(
        self, state: GFN2OrbitalState, *, excited=None, root: int = 0,
        roots: int = 6, selection_policy=None, response_tolerance: float = 1.0e-10,
    ):
        """Complete molecular analytic SF-TDA Hessian for a zero-based root."""
        from .stda_hessian import MolecularSFTDAHessian

        return MolecularSFTDAHessian(
            self, state, excited=excited, root=root, roots=roots,
            selection_policy=selection_policy, response_tolerance=response_tolerance,
        )

    def evaluate_sf_tda(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        roots: int = 5,
        selection_policy=None,
        eigensolver_tolerance_hartree: float = 1.0e-9,
        z_vector_tolerance: float = 1.0e-10,
        maximum_subspace: int = 64,
        maximum_z_iterations: int = 200,
    ) -> GFN2SFTDAEnergyGradientResult:
        """Evaluate resident SF-TDA energies and complete analytic gradients."""

        from .sf_tda import (
            AtomicTransitionChargeResponse,
            SFTDASelectionPolicy,
            SpinFlipExcitationSpace,
            grimme_xtb_sf_tda,
            select_grimme_spin_flip_configurations,
        )
        from .sf_tda_gradient import (
            frozen_orbital_sf_tda_lagrangian,
            matrix_free_sf_tda_z_vector,
        )

        ground = self.evaluate(
            coordinates_angstrom,
            compute_cm5=False,
            compute_bond_orders=False,
            retain_orbitals=True,
        )
        if ground.orbital_state is None:
            raise RuntimeError("SF-TDA evaluation requires the resident orbital state")
        state = ground.orbital_state
        policy = selection_policy or SFTDASelectionPolicy()
        parent = SpinFlipExcitationSpace.from_grimme_window(state, policy)
        response = AtomicTransitionChargeResponse.grimme_xtb_spin_flip(state, parent)
        selected = select_grimme_spin_flip_configurations(parent, response, policy)
        root_count = min(int(roots), selected.size)
        if root_count < 1:
            raise ValueError("SF-TDA selected space contains no roots")
        excited = grimme_xtb_sf_tda(
            state,
            roots=root_count,
            tolerance_hartree=float(eigensolver_tolerance_hartree),
            maximum_subspace=int(maximum_subspace),
            selection_policy=policy,
        )
        frozen = frozen_orbital_sf_tda_lagrangian(
            state, excited, selection_policy=policy
        )
        z_vector = matrix_free_sf_tda_z_vector(
            state,
            excited,
            self.density_response_kernel(state),
            selection_policy=policy,
            tolerance=float(z_vector_tolerance),
            maximum_iterations=int(maximum_z_iterations),
        )
        response_density = frozen.response_density + z_vector.response_density
        response_weighted = (
            frozen.response_weighted_density + z_vector.response_weighted_density
        )
        excitation_gradient = frozen.explicit_kernel_gradient_hartree_per_bohr.copy()
        excitation_gradient += self.density_response_coordinate_vjp_batch(
            state, response_density, response_weighted
        )
        excitation_energy = excited.excitation_energies_hartree
        total_energy = float(ground.energy_hartree) + excitation_energy
        total_gradient = (
            np.asarray(ground.gradient_hartree_per_bohr)[None, :, :]
            + excitation_gradient
        )
        return GFN2SFTDAEnergyGradientResult(
            ground,
            excited,
            z_vector,
            excitation_energy,
            total_energy,
            excitation_gradient,
            total_gradient,
        )

    def sf_tda_coupling_from_manifold(
        self,
        manifold: GFN2SFTDAEnergyGradientResult,
        *,
        state_i: int,
        state_j: int,
        selection_policy=None,
        z_vector_tolerance: float = 1.0e-10,
        maximum_z_iterations: int = 200,
        gap_tolerance_hartree: float = 1.0e-7,
        reference_amplitudes: np.ndarray | None = None,
    ) -> GFN2SFTDAInterstateResult:
        """Contract one coupling from an already evaluated SF-TDA manifold.

        The off-diagonal derivative is obtained by polarizing the same
        pair-packed frozen and orbital-adjoint Lagrangians used for analytic
        state gradients.  It therefore adds no AO-square nuclear derivative
        tensor and retains the molecular FAST_TDA working-set contract.
        """

        from .sf_tda import SFTDAResult, SFTDASelectionPolicy
        from .sf_tda_coupling import sf_tda_interstate_coupling
        from .sf_tda_gradient import (
            frozen_orbital_sf_tda_lagrangian,
            matrix_free_sf_tda_z_vector,
        )

        first = int(state_i)
        second = int(state_j)
        if first == second or min(first, second) < 0:
            raise ValueError("SF-TDA coupling states must be distinct zero-based roots")
        policy = selection_policy or SFTDASelectionPolicy()
        excited = manifold.sf_tda_result
        amplitudes = np.asarray(excited.amplitudes, dtype=float)
        if max(first, second) >= len(amplitudes):
            raise IndexError("SF-TDA coupling state lies outside the evaluated manifold")
        phase_reference = reference_amplitudes
        if phase_reference is not None and np.asarray(phase_reference).shape != amplitudes.shape:
            phase_reference = None
        state = manifold.ground_state.orbital_state
        if state is None:
            raise RuntimeError("SF-TDA coupling requires the resident orbital state")

        class _MolecularInterstateKernel:
            size = amplitudes.shape[1]
            dimension = np.asarray(manifold.ground_state.gradient_hartree_per_bohr).size

            def gradient_vjp(_self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
                plus = np.asarray(left, dtype=float) + np.asarray(right, dtype=float)
                minus = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
                polarized = SFTDAResult(
                    np.zeros(2),
                    np.stack((plus, minus)),
                    np.zeros(2),
                    iterations=0,
                    converged=True,
                    solver_backend="analytic-polarization",
                )
                frozen = frozen_orbital_sf_tda_lagrangian(
                    state, polarized, selection_policy=policy
                )
                z_vector = matrix_free_sf_tda_z_vector(
                    state,
                    polarized,
                    self.density_response_kernel(state),
                    selection_policy=policy,
                    tolerance=float(z_vector_tolerance),
                    maximum_iterations=int(maximum_z_iterations),
                )
                density = frozen.response_density + z_vector.response_density
                weighted = (
                    frozen.response_weighted_density
                    + z_vector.response_weighted_density
                )
                derivatives = frozen.explicit_kernel_gradient_hartree_per_bohr.copy()
                derivatives += self.density_response_coordinate_vjp_batch(
                    state, density, weighted
                )
                return (0.25 * (derivatives[0] - derivatives[1])).reshape(-1)

        coupling = sf_tda_interstate_coupling(
            excited,
            _MolecularInterstateKernel(),
            first,
            second,
            gap_tolerance_hartree=float(gap_tolerance_hartree),
            reference_amplitudes=phase_reference,
        )
        shape = manifold.ground_state.gradient_hartree_per_bohr.shape
        derivative = coupling.derivative_coupling_per_bohr
        return GFN2SFTDAInterstateResult(
            manifold,
            first,
            second,
            coupling.interstate_numerator_hartree_per_bohr.reshape(shape),
            None if derivative is None else derivative.reshape(shape),
            coupling.energy_gap_hartree,
            coupling.gap_tolerance_hartree,
        )

    def evaluate_sf_tda_coupling(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        state_i: int,
        state_j: int,
        roots: int = 5,
        selection_policy=None,
        eigensolver_tolerance_hartree: float = 1.0e-9,
        z_vector_tolerance: float = 1.0e-10,
        maximum_subspace: int = 64,
        maximum_z_iterations: int = 200,
        gap_tolerance_hartree: float = 1.0e-7,
        reference_amplitudes: np.ndarray | None = None,
    ) -> GFN2SFTDAInterstateResult:
        """Evaluate a molecular SF-TDA manifold and one analytic coupling."""

        from .sf_tda import SFTDASelectionPolicy

        first = int(state_i)
        second = int(state_j)
        root_count = max(int(roots), first + 1, second + 1)
        policy = selection_policy or SFTDASelectionPolicy()
        manifold = self.evaluate_sf_tda(
            coordinates_angstrom,
            roots=root_count,
            selection_policy=policy,
            eigensolver_tolerance_hartree=float(eigensolver_tolerance_hartree),
            z_vector_tolerance=float(z_vector_tolerance),
            maximum_subspace=int(maximum_subspace),
            maximum_z_iterations=int(maximum_z_iterations),
        )
        return self.sf_tda_coupling_from_manifold(
            manifold,
            state_i=first,
            state_j=second,
            selection_policy=policy,
            z_vector_tolerance=float(z_vector_tolerance),
            maximum_z_iterations=int(maximum_z_iterations),
            gap_tolerance_hartree=float(gap_tolerance_hartree),
            reference_amplitudes=reference_amplitudes,
        )

    def evaluate_sf_tda_localized(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        roots: int = 5,
        selection_policy=None,
        locality_policy=None,
        rotation_policy=None,
        eigensolver_tolerance_hartree: float = 1.0e-9,
        z_vector_tolerance: float = 1.0e-10,
        maximum_subspace: int = 64,
        maximum_z_iterations: int = 200,
    ) -> GFN2SFTDAEnergyGradientResult:
        """Evaluate the certified localized SF-TDA graph and sparse adjoint.

        The localized route is deliberately fail-closed: a result is returned
        only when both the iterative eigensolver and the conservative omitted-
        coupling bound satisfy the requested locality contract.
        """

        from .sf_tda import SFTDASelectionPolicy
        from .sf_tda_local import LocalizedSFTDAPolicy, localized_sf_tda
        from .sf_tda_sparse_gradient import localized_sparse_sf_tda_lagrangian

        ground = self.evaluate(
            coordinates_angstrom,
            compute_cm5=False,
            compute_bond_orders=False,
            retain_orbitals=True,
        )
        if ground.orbital_state is None:
            raise RuntimeError("localized SF-TDA requires the resident orbital state")
        state = ground.orbital_state
        selection = selection_policy or SFTDASelectionPolicy()
        locality = locality_policy or LocalizedSFTDAPolicy()
        localized = localized_sf_tda(
            state,
            roots=int(roots),
            policy=locality,
            selection_policy=selection,
            tolerance_hartree=float(eigensolver_tolerance_hartree),
            maximum_subspace=int(maximum_subspace),
        )
        if not localized.qualified:
            raise RuntimeError(
                "localized SF-TDA truncation is not certified: "
                f"bound={localized.omitted_coupling_bound_hartree:.3e} hartree, "
                f"tolerance={locality.truncation_error_tolerance_hartree:.3e} hartree"
            )
        context = self._prepare_density_response_gradient_context(state)
        domains = context["domains"]
        overlap = context["overlap_sparse"]
        inverse = sparse_newton_schulz_inverse_sqrt(
            overlap,
            domains,
            tolerance=min(1.0e-10, float(z_vector_tolerance)),
            truncation_threshold=0.0,
        )
        if not inverse.converged:
            raise RuntimeError(
                "localized SF-TDA sparse orthogonalizer is uncertified: "
                f"residual={inverse.residual:.3e}"
            )
        response_kernel = self.sparse_density_response_kernel(state, _context=context)
        lagrangian = localized_sparse_sf_tda_lagrangian(
            state,
            localized.operator,
            localized.result.amplitudes,
            overlap,
            inverse.inverse_sqrt,
            domains,
            response_kernel,
            z_tolerance=float(z_vector_tolerance),
            maximum_z_iterations=int(maximum_z_iterations),
            rotation_policy=rotation_policy,
        )
        excitation_gradient = lagrangian.explicit_kernel_gradient_hartree_per_bohr.copy()
        excitation_gradient += np.stack(
            tuple(
                self.density_response_coordinate_vjp(
                    state,
                    lagrangian.response_density.root(root),
                    lagrangian.response_weighted_density.root(root),
                    _context=context,
                )
                for root in range(lagrangian.response_density.nroots)
            )
        )
        excitation = localized.result.excitation_energies_hartree
        total_energy = float(ground.energy_hartree) + excitation
        total_gradient = np.asarray(ground.gradient_hartree_per_bohr)[None] + excitation_gradient
        return GFN2SFTDAEnergyGradientResult(
            ground,
            localized,
            lagrangian.z_vector,
            excitation,
            total_energy,
            excitation_gradient,
            total_gradient,
        )

    def evaluate_nn_energy_development(
        self, coordinates_angstrom: np.ndarray,
        nn_model_path: str | os.PathLike[str] | NNXTBOverlay,
        *, return_components: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        """Evaluate the native NN energy with fresh geometry-dependent parameters.

        A fresh worker isolates geometry-dependent parameters and disables
        every element-keyed resident H0/SCF cache. This method is deliberately
        also used by the correctness-first ``nn_enabled`` calculator route.
        """
        if self.symmetry_contract is not None or len(self.point_charges_electron):
            raise ValueError("development NN energy requires a molecular system without embedding")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN-xTB energy coordinates must be finite natoms x 3")
        worker = self._batch_worker_clone()
        worker._nn_development_overlay = (
            nn_model_path if isinstance(nn_model_path, NNXTBOverlay)
            else NNXTBOverlay(nn_model_path)
        )
        energy = float(worker._single_point(
            coordinates * ANGSTROM_TO_BOHR,
            compute_bond_orders=False,
            collect_telemetry=False,
            energy_only=True,
        ))
        if return_components:
            return energy, worker._nn_development_components
        return energy

    def evaluate_nn_derivatives_development(
        self, coordinates_angstrom: np.ndarray,
        nn_model_path: str | os.PathLike[str] | NNXTBOverlay,
        *, order: int = 1, step_bohr: float = 1.0e-3,
    ) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        """Independent full-energy finite-difference audit of NN derivatives.

        Unlike a fixed-parameter differentiation, this recomputes neural heads,
        the Hamiltonian and the converged SCC state at every displacement. It
        is intentionally slower than the analytic gradient/Hessian route and
        is never called by the analytic Hessian entry points.
        """
        if order not in (1, 2):
            raise ValueError("NN development derivative order must be 1 or 2")
        if not np.isfinite(step_bohr) or step_bohr <= 0.0:
            raise ValueError("NN development derivative step must be positive and finite")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN-xTB derivative coordinates must be finite natoms x 3")
        overlay = nn_model_path if isinstance(nn_model_path, NNXTBOverlay) else NNXTBOverlay(nn_model_path)
        evaluator = self._batch_worker_clone()
        evaluator.scf_energy_tolerance_hartree = min(
            evaluator.scf_energy_tolerance_hartree, 1.0e-12,
        )
        evaluator.scf_wavefunction_tolerance_electron = min(
            evaluator.scf_wavefunction_tolerance_electron, 1.0e-10,
        )
        flat = coordinates.reshape(-1)
        delta_angstrom = step_bohr / ANGSTROM_TO_BOHR
        cache: dict[tuple[tuple[int, int], ...], float] = {}

        def displaced(*shifts: tuple[int, int]) -> float:
            key = tuple(sorted(shifts))
            if key not in cache:
                trial = flat.copy()
                for component, multiplier in shifts:
                    trial[component] += multiplier * delta_angstrom
                cache[key] = float(evaluator.evaluate_nn_energy_development(
                    trial.reshape(coordinates.shape), overlay,
                ))
            return cache[key]

        energy = displaced()
        dimension = flat.size
        gradient = np.empty(dimension, dtype=float)
        for component in range(dimension):
            gradient[component] = (
                displaced((component, 1)) - displaced((component, -1))
            ) / (2.0 * step_bohr)
        gradient = gradient.reshape(coordinates.shape)
        if order == 1:
            return energy, gradient
        hessian = np.empty((dimension, dimension), dtype=float)
        for row in range(dimension):
            hessian[row, row] = (
                displaced((row, 1)) - 2.0 * energy + displaced((row, -1))
            ) / step_bohr**2
            for column in range(row):
                value = (
                    displaced((row, 1), (column, 1))
                    - displaced((row, 1), (column, -1))
                    - displaced((row, -1), (column, 1))
                    + displaced((row, -1), (column, -1))
                ) / (4.0 * step_bohr**2)
                hessian[row, column] = hessian[column, row] = value
        return energy, gradient, hessian

    def _nn_frozen_parameter_worker(self, shifts: dict[str, np.ndarray]):
        """Prepare an unfused NN worker with fixed parameter shifts."""
        class FrozenShifts:
            def matrix_parameter_shifts(self, *_args, **_kwargs):
                return shifts

        worker = self._batch_worker_clone()
        worker.scf_energy_tolerance_hartree = min(
            worker.scf_energy_tolerance_hartree, 1.0e-12,
        )
        worker.scf_wavefunction_tolerance_electron = min(
            worker.scf_wavefunction_tolerance_electron, 1.0e-10,
        )
        worker.scf_mixer_tolerance = min(
            worker.scf_mixer_tolerance, 1.0e-8,
        )
        worker._nn_development_overlay = FrozenShifts()
        return worker

    def _nn_fixed_parameter_point(
        self, positions_bohr: np.ndarray, shifts: dict[str, np.ndarray],
        *, energy_only: bool = False, retain_orbitals: bool = False,
        return_worker: bool = False,
    ) -> GFN2Result | float:
        """Evaluate one NN-overlaid state with the parameter shifts frozen."""
        worker = self._nn_frozen_parameter_worker(shifts)
        result = worker._single_point(
            positions_bohr, energy_only=energy_only,
            compute_bond_orders=False, collect_telemetry=False,
            retain_orbitals=retain_orbitals,
        )
        return (result, worker) if return_worker else result

    def evaluate_nn_h0_parameter_cpscf(
        self, coordinates_angstrom: np.ndarray,
        head_direction: dict[str, np.ndarray],
    ):
        """Analytic SCC density response to basis-preserving H0 head shifts.

        ``head_direction`` is in MATRIX parameter-shift units. This is the
        parameter-response kernel needed by the NN mixed and second blocks,
        not by itself a complete NN Hessian.
        """
        from .ground_state_response import solve_ground_state_directional_response

        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN H0 CPSCF requires nn_enabled=True")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN H0 CPSCF coordinates must be finite natoms x 3")
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        worker = self._nn_frozen_parameter_worker(shifts)
        fixed = worker._single_point(
            positions, compute_bond_orders=False, collect_telemetry=False,
            retain_orbitals=True,
        )
        state = fixed.orbital_state
        if not fixed.converged or state is None:
            raise RuntimeError("NN H0 CPSCF requires a converged retained SCF state")
        parameters = build_nn_parameter_state(
            self.atomic_numbers, self.parameters, shifts,
        )
        _, fock_derivatives = analytic_h0_head_adjoints(
            self.atomic_numbers, positions, parameters,
            state.overlap, fixed.density, state.ao_atom, state.ao_shell,
            return_fock_derivatives=True,
        )
        source = np.zeros_like(state.fock)
        for name, direction in head_direction.items():
            if name not in fock_derivatives:
                raise ValueError(f"NN H0 CPSCF does not support head {name!r}")
            values = np.asarray(direction, dtype=float)
            if values.shape != fock_derivatives[name].shape[:2] or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid NN H0 CPSCF direction for {name!r}")
            source += np.einsum("ij,ijmn->mn", values, fock_derivatives[name])
        return solve_ground_state_directional_response(
            state, np.zeros_like(source), source,
            worker.density_response_kernel(state),
        )

    def evaluate_nn_scc_parameter_cpscf(
        self, coordinates_angstrom: np.ndarray,
        head_direction: dict[str, np.ndarray],
    ):
        """Analytic CPSCF response to hardness, exponent and cubic SCC heads.

        The head direction uses MATRIX shift units. This computes the
        coupled density response; explicit second parameter derivatives and
        geometry cross-derivatives are separate Hessian contributions.
        """
        from .ground_state_response import solve_ground_state_directional_response

        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN SCC CPSCF requires nn_enabled=True")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN SCC CPSCF coordinates must be finite natoms x 3")
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        worker = self._nn_frozen_parameter_worker(shifts)
        fixed = worker._single_point(
            positions, compute_bond_orders=False, collect_telemetry=False,
            retain_orbitals=True,
        )
        state = fixed.orbital_state
        if not fixed.converged or state is None:
            raise RuntimeError("NN SCC CPSCF requires a converged retained SCF state")
        parameters = build_nn_parameter_state(
            self.atomic_numbers, self.parameters, shifts,
        )
        basis = build_molecular_basis(
            self.atomic_numbers, positions, self.parameters,
            per_atom_elements=parameters.basis_elements,
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=parameters.basis_elements,
            per_atom_third_scales=parameters.third_scales,
        )
        labels = [None] * partition.nshells
        for orbital, index in enumerate(partition.orbital_shell):
            labels[index] = basis.spherical_shell[orbital]
        q_shell = shell_charges(fixed.density, state.overlap, partition)
        source = analytic_scc_parameter_fock_direction(
            positions, parameters, q_shell, partition.shell_atom, labels,
            state.overlap, partition.orbital_shell, head_direction,
        )
        return solve_ground_state_directional_response(
            state, np.zeros_like(source), source,
            worker.density_response_kernel(state),
        )

    def evaluate_nn_aes_d4_parameter_cpscf(
        self, coordinates_angstrom: np.ndarray,
        head_direction: dict[str, np.ndarray],
    ):
        """Coupled CPSCF density response to AES and D4 parameter heads.

        Mixed AES/D4 directions share the full GFN2 density-response kernel,
        so their SCC, multipole and dispersion orbital relaxation is coupled.
        This does not yet supply the explicit second parameter derivatives.
        """
        from .ground_state_response import solve_ground_state_directional_response

        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN AES/D4 CPSCF requires nn_enabled=True")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN AES/D4 CPSCF coordinates must be finite natoms x 3")
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        worker = self._nn_frozen_parameter_worker(shifts)
        fixed = worker._single_point(
            positions, compute_bond_orders=False, collect_telemetry=False,
            retain_orbitals=True,
        )
        state = fixed.orbital_state
        if not fixed.converged or state is None:
            raise RuntimeError("NN AES/D4 CPSCF requires a converged retained SCF state")
        parameters = build_nn_parameter_state(
            self.atomic_numbers, self.parameters, shifts,
        )
        if worker._post_scf_pair_integrals is None or worker._active_coordination is None:
            raise RuntimeError("NN AES/D4 CPSCF requires multipole integrals and coordination")
        basis = build_molecular_basis(
            self.atomic_numbers, positions, self.parameters,
            per_atom_elements=parameters.basis_elements,
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=parameters.basis_elements,
            per_atom_third_scales=parameters.third_scales,
        )
        operators = prepare_pair_cumulative_multipole_operators(
            worker._post_scf_pair_integrals, positions, basis.spherical_atom,
        )
        _, multipoles = dense_scc_observables(
            fixed.density, operators, partition.orbital_shell,
            partition.reference_population, worker._valence,
        )
        d4_names = {"global.a1", "global.a2", "global.s6", "global.s8", "global.s9", "global.s10"}
        aes_direction = {name: value for name, value in head_direction.items()
                         if name not in d4_names}
        qpot, mupot, thetapot = analytic_aes_parameter_moment_potential_direction(
            positions, worker._active_coordination, parameters,
            multipoles, aes_direction,
        )
        d4_direction = {}
        for name in d4_names:
            if name in head_direction:
                value = np.asarray(head_direction[name], dtype=float)
                if value.shape != (1, 1) or not np.all(np.isfinite(value)):
                    raise ValueError(f"invalid D4 direction for {name}")
                d4_direction[name[7:]] = float(value[0, 0])
        d4_qpot = np.zeros(len(positions))
        if d4_direction:
            d4_cn = _d4_coordination(
                self.atomic_numbers, positions, _covalent_radii(),
                worker._electronegativities,
            )
            d4_parameters = worker._d4_parameters
            prepared = prepare_d4_geometry(
                positions, d4_cn, worker._d4_models, d4_parameters,
            )
            pair = d4_two_body(
                positions, multipoles.charges, d4_cn,
                worker._d4_models, d4_parameters, prepared=prepared,
            )
            d4_qpot = d4_global_parameter_charge_potential_direction_dense(
                positions, multipoles.charges, d4_cn,
                worker._d4_models, d4_parameters, pair, d4_direction,
            )
        source = dense_scc_fock(
            np.zeros_like(state.fock), state.overlap, operators,
            partition.orbital_shell, np.zeros(partition.nshells),
            qpot, mupot, thetapot, d4_qpot,
        )
        return solve_ground_state_directional_response(
            state, np.zeros_like(source), source,
            worker.density_response_kernel(state),
        )

    def evaluate_nn_exponent_parameter_cpscf(
        self, coordinates_angstrom: np.ndarray,
        exponent_direction: np.ndarray,
        *, return_overlap_direction: bool = False,
        return_observable_direction: bool = False,
    ):
        """CPSCF response to a basis-moving ``shell.exp`` direction.

        The explicit Fock source includes H0, SCC, CAMM/AES and D4/ATM
        operator motion. The overlap source supplies the metric/Pulay part.
        """
        from .ground_state_response import solve_ground_state_directional_response

        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN exponent CPSCF requires nn_enabled=True")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        direction = np.array(exponent_direction, dtype=float, copy=True)
        natoms = len(self.atomic_numbers)
        if coordinates.shape != (natoms, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN exponent CPSCF coordinates must be finite natoms x 3")
        if direction.shape != (natoms, 4) or not np.all(np.isfinite(direction)):
            raise ValueError("NN exponent direction must be finite natoms x 4")
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        worker = self._nn_frozen_parameter_worker(shifts)
        fixed = worker._single_point(
            positions, compute_bond_orders=False, collect_telemetry=False,
            retain_orbitals=True,
        )
        state = fixed.orbital_state
        if not fixed.converged or state is None:
            raise RuntimeError("NN exponent CPSCF requires a converged retained SCF state")
        parameters = build_nn_parameter_state(
            self.atomic_numbers, self.parameters, shifts,
        )
        basis = build_molecular_basis(
            self.atomic_numbers, positions, self.parameters,
            per_atom_elements=parameters.basis_elements,
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=parameters.basis_elements,
            per_atom_third_scales=parameters.third_scales,
        )
        kernel = worker.density_response_kernel(state)
        slots = {
            (atom, shell.label): slot
            for atom, element in enumerate(parameters.elements)
            for slot, shell in enumerate(element.shells)
        }
        for atom, element in enumerate(parameters.elements):
            for slot, _shell in enumerate(element.shells):
                if f"atom {atom} shell {slot}: exp" in parameters.floor_events:
                    direction[atom, slot] = 0.0
        cartesian = basis.cartesian_orbitals
        cartesian_derivative = np.zeros((13, len(cartesian), len(cartesian)))
        for bra_index, bra in enumerate(cartesian):
            for ket_index in range(bra_index + 1):
                ket = cartesian[ket_index]
                bra_scale = direction[bra.atom_index, slots[bra.atom_index, bra.shell]]
                ket_scale = direction[ket.atom_index, slots[ket.atom_index, ket.shell]]
                if bra_scale == 0.0 and ket_scale == 0.0:
                    continue
                _, bra_derivative, ket_derivative = _nn_exponent_integral_response(
                    bra, ket, "first",
                )
                value = bra_scale * bra_derivative + ket_scale * ket_derivative
                cartesian_derivative[:, bra_index, ket_index] = value
                cartesian_derivative[:, ket_index, bra_index] = value
        transform = basis.cartesian_to_spherical
        moments_direction = np.asarray([
            transform.T @ component @ transform for component in cartesian_derivative
        ])
        ds = moments_direction[0]
        ao_atom = np.asarray(basis.spherical_atom, dtype=int)
        ao_zeta = np.asarray([
            next(shell.slater_exponent for shell in parameters.elements[atom].shells
                 if shell.label == label)
            for atom, label in zip(basis.spherical_atom, basis.spherical_shell, strict=True)
        ])
        ao_dzeta = np.asarray([
            direction[atom, slots[atom, label]]
            for atom, label in zip(basis.spherical_atom, basis.spherical_shell, strict=True)
        ])
        core_prefactor = extended_hueckel_hamiltonian(
            np.ones_like(state.overlap), worker._orbital_metadata,
            self.atomic_numbers, positions, parameters.atoms,
            parameters.orbitals, global_scalings=parameters.globals,
        )
        active = ao_atom[:, None] != ao_atom[None, :]
        core_prefactor[~active] = 0.0
        dh0 = core_prefactor * ds
        for left in range(basis.norbitals):
            for right in range(left):
                if not active[left, right]:
                    continue
                log_factor = (
                    0.25 * (ao_dzeta[left] / ao_zeta[left]
                            + ao_dzeta[right] / ao_zeta[right])
                    - 0.5 * (ao_dzeta[left] + ao_dzeta[right])
                    / (ao_zeta[left] + ao_zeta[right])
                ) * (2.0 * parameters.globals.get("wexp", 0.5))
                value = core_prefactor[left, right] * state.overlap[left, right] * log_factor
                dh0[left, right] += value
                dh0[right, left] += value
        rows, columns = triangular_indices(basis.norbitals)
        zero_derivatives = np.zeros((len(rows), 13, 3))
        derivative_integrals = SparseAtomicMultipoleIntegrals(
            basis.norbitals, rows, columns,
            ds[rows, columns],
            moments_direction[1:4, rows, columns].T,
            moments_direction[4:, rows, columns].T.reshape(-1, 3, 3),
            zero_derivatives, zero_derivatives,
        )
        derivative_operators = prepare_pair_cumulative_multipole_operators(
            derivative_integrals, positions, basis.spherical_atom,
        )
        qdot, moment_dot = dense_scc_observables(
            fixed.density, derivative_operators, partition.orbital_shell,
            np.zeros(partition.nshells), np.zeros(natoms),
        )
        q_shell = kernel.reference_shell_charges
        moments = kernel.reference_multipoles
        interaction = kernel.shell_interaction
        third = np.asarray(partition.third_order_hartree)
        shell_potential = interaction @ q_shell + third * q_shell**2
        shell_potential_dot = interaction @ qdot + 2.0 * third * q_shell * qdot
        aes_parameters = parameters.anisotropic_parameters(worker._active_coordination)
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            moments, positions, aes_parameters,
        )
        qpot_dot, mupot_dot, thetapot_dot = anisotropic_moment_potential_direction(
            moments, moment_dot, positions, np.zeros_like(positions),
            aes_parameters, np.zeros(natoms),
        )
        d4_potential = kernel._d4(moments.charges).charge_derivative_hartree.copy()
        if natoms >= 3:
            _, atm_potential = atm_three_body_energy_charge_derivative_dense(
                positions, moments.charges, kernel.d4_coordination,
                worker._d4_models, worker._d4_parameters, kernel._d4(moments.charges),
            )
            d4_potential += atm_potential
        d4_potential_dot = d4_charge_hessian(
            moments.charges, kernel.d4_prepared,
        ) @ moment_dot.charges
        if kernel.atm_charge_hessian is not None:
            d4_potential_dot += kernel.atm_charge_hessian @ moment_dot.charges
        direct_operator = dense_scc_fock(
            np.zeros_like(state.fock), ds, derivative_operators,
            partition.orbital_shell, shell_potential,
            qpot, mupot, thetapot, d4_potential,
        )
        potential_operator = dense_scc_fock(
            np.zeros_like(state.fock), state.overlap, kernel.operators,
            partition.orbital_shell, shell_potential_dot,
            qpot_dot, mupot_dot, thetapot_dot, d4_potential_dot,
        )
        source = dh0 + direct_operator + potential_operator
        response = solve_ground_state_directional_response(
            state, ds, source, kernel,
        )
        if return_observable_direction:
            return response, ds, moment_dot
        return (response, ds) if return_overlap_direction else response

    def _nn_expand_rcov_direction(self, coordinates_angstrom, head_direction):
        """Resolve CN motion into its exact H0-level and AES-valence sources."""
        direction = {name: np.array(value, dtype=float, copy=True) for name, value in head_direction.items()}
        cn_dot = np.zeros(len(self.atomic_numbers))
        if "atom.rcov" not in direction:
            return direction, cn_dot
        positions = np.asarray(coordinates_angstrom) * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        parameters = build_nn_parameter_state(self.atomic_numbers, self.parameters, shifts)
        jacobian, _ = analytic_rcov_coordination_jacobian(positions, parameters)
        radius_dot = direction.pop("atom.rcov")
        if radius_dot.shape != (len(self.atomic_numbers), 1):
            raise ValueError("invalid NN covalent-radius direction")
        cn_dot = jacobian @ radius_dot[:, 0]
        level_dot = direction.setdefault("shell.lev", np.zeros((len(positions), 4)))
        for atom, element in enumerate(parameters.elements):
            for slot, shell in enumerate(element.shells):
                level_dot[atom, slot] -= shell.coordination_shift_hartree * cn_dot[atom]
        direction["atom.mpvcn"] = direction.get("atom.mpvcn", np.zeros((len(positions), 1))) - cn_dot[:, None]
        return direction, cn_dot

    def evaluate_nn_parameter_cpscf(
        self, coordinates_angstrom: np.ndarray,
        head_direction: dict[str, np.ndarray],
        *, return_overlap_direction: bool = False,
        return_observable_direction: bool = False,
    ):
        """Linear coupled response to simultaneous H0/SCC/AES/D4/basis heads.

        Every component uses the same converged baseline and the complete
        Fock--density kernel. This exposes cross-family *orbital* response;
        it is not the explicit second derivative of the energy functional.
        """
        from .ground_state_response import GroundStateDirectionalResponse

        head_direction, _ = self._nn_expand_rcov_direction(
            coordinates_angstrom, head_direction,
        )
        h0_names = {"shell.lev", "shell.kcn", "shell.poly", "shell.h0exp", "atom.arad", "atom.en",
                    "global.ks", "global.kp", "global.kd", "global.ksd",
                    "global.kpd", "global.enscale", "global.wexp"}
        scc_names = {"atom.gam", "atom.gam3", "shell.lpar", "shell.third",
                     "global.alphaj"}
        aes_d4_names = {"atom.dpol", "atom.qpol", "atom.mprad", "atom.mpvcn",
                        "global.aesdmp3", "global.aesdmp5", "global.aesexp",
                        "global.aesshift", "global.aesrmax", "global.a1",
                        "global.a2", "global.s6", "global.s8", "global.s9", "global.s10"}
        supported = h0_names | scc_names | aes_d4_names | {"shell.exp"}
        unknown = set(head_direction) - supported
        if unknown:
            raise NotImplementedError(f"NN parameter CPSCF sources missing: {sorted(unknown)}")
        if not head_direction:
            raise ValueError("NN parameter CPSCF requires at least one head direction")
        responses = []
        h0 = {name: value for name, value in head_direction.items() if name in h0_names}
        scc = {name: value for name, value in head_direction.items() if name in scc_names}
        aes_d4 = {name: value for name, value in head_direction.items() if name in aes_d4_names}
        if h0:
            responses.append(self.evaluate_nn_h0_parameter_cpscf(coordinates_angstrom, h0))
        if scc:
            responses.append(self.evaluate_nn_scc_parameter_cpscf(coordinates_angstrom, scc))
        if aes_d4:
            responses.append(self.evaluate_nn_aes_d4_parameter_cpscf(coordinates_angstrom, aes_d4))
        overlap_direction = None
        moment_operator_direction = None
        if "shell.exp" in head_direction:
            exponent_response, overlap_direction, moment_operator_direction = self.evaluate_nn_exponent_parameter_cpscf(
                coordinates_angstrom, head_direction["shell.exp"],
                return_observable_direction=True,
            )
            responses.append(exponent_response)
        combined = GroundStateDirectionalResponse(
            *(sum((getattr(response, field) for response in responses))
              for field in (
                  "coefficient_derivative", "density_derivative",
                  "weighted_density_derivative", "fock_derivative",
                  "orbital_energy_derivative",
              )),
            sum(response.residual_norm for response in responses),
            sum(response.iterations for response in responses),
        )
        if return_overlap_direction:
            if overlap_direction is None:
                overlap_direction = np.zeros_like(combined.density_derivative)
            if return_observable_direction:
                return combined, overlap_direction, moment_operator_direction
            return combined, overlap_direction
        if return_observable_direction:
            return combined, moment_operator_direction
        return combined

    def evaluate_nn_scc_head_adjoint_response(
        self, coordinates_angstrom: np.ndarray,
        head_direction: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """SCC head-adjoint action including cross-family CPSCF response.

        Returns ``E_pp dp`` for the SCC heads plus their relaxed charge
        response to all supported H0/SCC/AES/D4/exponent perturbations.
        """
        response, ds = self.evaluate_nn_parameter_cpscf(
            coordinates_angstrom, head_direction,
            return_overlap_direction=True,
        )
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        fixed = self._nn_fixed_parameter_point(
            positions, shifts, retain_orbitals=True,
        )
        if not fixed.converged or fixed.orbital_state is None:
            raise RuntimeError("SCC head response requires a converged retained SCF state")
        state = build_nn_parameter_state(
            self.atomic_numbers, self.parameters, shifts,
        )
        basis = build_molecular_basis(
            self.atomic_numbers, positions, self.parameters,
            per_atom_elements=state.basis_elements,
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=state.basis_elements,
            per_atom_third_scales=state.third_scales,
        )
        labels = [None] * partition.nshells
        for orbital, index in enumerate(partition.orbital_shell):
            labels[index] = basis.spherical_shell[orbital]
        overlap = fixed.orbital_state.overlap
        q_shell = shell_charges(fixed.density, overlap, partition)
        orbital_population_direction = -np.sum(
            response.density_derivative * overlap + fixed.density * ds,
            axis=1,
        )
        qdot = np.zeros(partition.nshells)
        np.add.at(qdot, np.asarray(partition.orbital_shell, dtype=np.intp),
                  orbital_population_direction)
        scc_names = {"atom.gam", "atom.gam3", "shell.lpar", "shell.third",
                     "global.alphaj"}
        return analytic_scc_head_adjoint_directional(
            positions, state, q_shell, qdot, partition.shell_atom, labels,
            {name: value for name, value in head_direction.items()
             if name in scc_names},
        )

    def evaluate_nn_aes_d4_head_adjoint_response(
        self, coordinates_angstrom: np.ndarray,
        head_direction: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """AES/D4 head action including cross-family CPSCF moment response."""
        response, moment_operator_dot = self.evaluate_nn_parameter_cpscf(
            coordinates_angstrom, head_direction,
            return_observable_direction=True,
        )
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        fixed, worker = self._nn_fixed_parameter_point(
            positions, shifts, retain_orbitals=True, return_worker=True,
        )
        if not fixed.converged or fixed.orbital_state is None:
            raise RuntimeError("AES/D4 head response requires a converged retained SCF state")
        if worker._post_scf_pair_integrals is None or worker._active_coordination is None:
            raise RuntimeError("AES/D4 head response requires multipole integrals and CN")
        state = build_nn_parameter_state(
            self.atomic_numbers, self.parameters, shifts,
        )
        basis = build_molecular_basis(
            self.atomic_numbers, positions, self.parameters,
            per_atom_elements=state.basis_elements,
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=state.basis_elements,
            per_atom_third_scales=state.third_scales,
        )
        operators = prepare_pair_cumulative_multipole_operators(
            worker._post_scf_pair_integrals, positions, basis.spherical_atom,
        )
        _, moments = dense_scc_observables(
            fixed.density, operators, partition.orbital_shell,
            partition.reference_population, worker._valence,
        )
        _, density_moment_dot = dense_scc_observables(
            response.density_derivative, operators, partition.orbital_shell,
            np.zeros(partition.nshells), np.zeros(len(positions)),
        )
        if moment_operator_dot is None:
            moment_operator_dot = AtomicMultipoles(
                np.zeros(len(positions)), np.zeros((len(positions), 3)),
                np.zeros((len(positions), 3, 3)),
            )
        moment_dot = AtomicMultipoles(
            density_moment_dot.charges + moment_operator_dot.charges,
            density_moment_dot.dipoles_bohr + moment_operator_dot.dipoles_bohr,
            density_moment_dot.quadrupoles_bohr2 + moment_operator_dot.quadrupoles_bohr2,
        )
        aes_names = {"atom.dpol", "atom.qpol", "atom.mprad", "atom.mpvcn",
                     "global.aesdmp3", "global.aesdmp5", "global.aesexp",
                     "global.aesshift", "global.aesrmax"}
        result = analytic_aes_head_adjoint_directional(
            positions, worker._active_coordination, state, moments, moment_dot,
            {name: value for name, value in head_direction.items()
             if name in aes_names},
            coordination_direction=self._nn_expand_rcov_direction(
                coordinates_angstrom, head_direction,
            )[1],
        )
        d4_names = {"global.a1", "global.a2", "global.s6", "global.s8", "global.s9", "global.s10"}
        d4_direction = {
            name[7:]: float(np.asarray(value)[0, 0])
            for name, value in head_direction.items() if name in d4_names
        }
        d4_cn = _d4_coordination(
            self.atomic_numbers, positions, _covalent_radii(),
            worker._electronegativities,
        )
        d4_parameters = worker._d4_parameters
        prepared = prepare_d4_geometry(
            positions, d4_cn, worker._d4_models, d4_parameters,
        )
        pair = d4_two_body(
            positions, moments.charges, d4_cn,
            worker._d4_models, d4_parameters, prepared=prepared,
        )
        result.update({
            f"global.{name}": np.array([[value]])
            for name, value in d4_global_head_adjoint_direction_dense(
                positions, moments.charges, moment_dot.charges, d4_cn,
                worker._d4_models, d4_parameters, pair, prepared, d4_direction,
            ).items()
        })
        return result

    def evaluate_nn_exponent_head_adjoint_response(
        self, coordinates_angstrom: np.ndarray,
        head_direction: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Analytic ``E_exp,p dp`` with all stationary CPSCF/Pulay terms.

        This contracts the normalized STO-nG second exponent integrals with
        H0, SCC, AES and D4 operators. It is the fixed-geometry parameter
        Hessian action on ``shell.exp`` adjoints, not the ``E_exp,R`` block.
        """
        response, ds_direction, operator_moment_dot = self.evaluate_nn_parameter_cpscf(
            coordinates_angstrom, head_direction,
            return_overlap_direction=True, return_observable_direction=True,
        )
        head_direction, _ = self._nn_expand_rcov_direction(
            coordinates_angstrom, head_direction,
        )
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        positions = coordinates * ANGSTROM_TO_BOHR
        natoms = len(positions)
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        fixed, worker = self._nn_fixed_parameter_point(
            positions, shifts, retain_orbitals=True, return_worker=True,
        )
        orbital_state = fixed.orbital_state
        if not fixed.converged or orbital_state is None:
            raise RuntimeError("NN exponent adjoint response requires converged SCC")
        state = build_nn_parameter_state(self.atomic_numbers, self.parameters, shifts)
        basis = build_molecular_basis(
            self.atomic_numbers, positions, self.parameters,
            per_atom_elements=state.basis_elements,
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=state.basis_elements,
            per_atom_third_scales=state.third_scales,
        )
        kernel = worker.density_response_kernel(orbital_state)
        density = np.asarray(fixed.density)
        dp = response.density_derivative
        occupations = orbital_state.alpha_occupations + orbital_state.beta_occupations
        weighted = energy_weighted_density(
            orbital_state.coefficients, occupations,
            orbital_state.orbital_energies_hartree,
        )
        dw = response.weighted_density_derivative
        q_shell = kernel.reference_shell_charges
        moments = kernel.reference_multipoles
        operator_shell_dot = -np.bincount(
            np.asarray(partition.orbital_shell, dtype=np.intp),
            weights=np.sum(density * ds_direction, axis=1),
            minlength=partition.nshells,
        )
        density_shell_dot, density_moment_dot = dense_scc_observables(
            dp, kernel.operators, partition.orbital_shell,
            np.zeros(partition.nshells), np.zeros(natoms),
        )
        qdot = density_shell_dot + operator_shell_dot
        if operator_moment_dot is None:
            operator_moment_dot = AtomicMultipoles(
                np.zeros(natoms), np.zeros((natoms, 3)),
                np.zeros((natoms, 3, 3)),
            )
        moment_dot = AtomicMultipoles(
            density_moment_dot.charges + operator_moment_dot.charges,
            density_moment_dot.dipoles_bohr + operator_moment_dot.dipoles_bohr,
            density_moment_dot.quadrupoles_bohr2
            + operator_moment_dot.quadrupoles_bohr2,
        )
        interaction = kernel.shell_interaction
        third = np.asarray(partition.third_order_hartree)
        shell_potential = interaction @ q_shell + third * q_shell**2
        scc_names = {"atom.gam", "atom.gam3", "shell.lpar", "shell.third",
                     "global.alphaj"}
        shell_labels = [None] * partition.nshells
        for orbital, index in enumerate(partition.orbital_shell):
            shell_labels[index] = basis.spherical_shell[orbital]
        explicit_shell_dot = analytic_scc_parameter_fock_direction(
            positions, state, q_shell, partition.shell_atom, shell_labels,
            orbital_state.overlap, partition.orbital_shell,
            {name: value for name, value in head_direction.items()
             if name in scc_names}, return_shell_potential=True,
        )
        shell_potential_dot = (
            interaction @ qdot + 2.0 * third * q_shell * qdot
            + explicit_shell_dot
        )
        aes_parameters = state.anisotropic_parameters(worker._active_coordination)
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            moments, positions, aes_parameters,
        )
        moment_qdot, moment_mudot, moment_thetadot = anisotropic_moment_potential_direction(
            moments, moment_dot, positions, np.zeros_like(positions),
            aes_parameters, np.zeros(natoms),
        )
        aes_names = {"atom.dpol", "atom.qpol", "atom.mprad", "atom.mpvcn",
                     "global.aesdmp3", "global.aesdmp5", "global.aesexp",
                     "global.aesshift", "global.aesrmax"}
        aes_qdot, aes_mudot, aes_thetadot = analytic_aes_parameter_moment_potential_direction(
            positions, worker._active_coordination, state, moments,
            {name: value for name, value in head_direction.items()
             if name in aes_names},
        )
        qpot_dot = moment_qdot + aes_qdot
        mupot_dot = moment_mudot + aes_mudot
        thetapot_dot = moment_thetadot + aes_thetadot
        d4_cn = kernel.d4_coordination
        d4_parameters = worker._d4_parameters
        pair = kernel._d4(moments.charges)
        d4_potential = pair.charge_derivative_hartree.copy()
        if natoms >= 3:
            _, atm_potential = atm_three_body_energy_charge_derivative_dense(
                positions, moments.charges, d4_cn, worker._d4_models,
                d4_parameters, pair,
            )
            d4_potential += atm_potential
        d4_potential_dot = d4_charge_hessian(
            moments.charges, kernel.d4_prepared,
        ) @ moment_dot.charges
        if kernel.atm_charge_hessian is not None:
            d4_potential_dot += kernel.atm_charge_hessian @ moment_dot.charges
        d4_names = {"global.a1", "global.a2", "global.s6", "global.s8", "global.s9", "global.s10"}
        d4_direction = {
            name[7:]: float(np.asarray(value)[0, 0])
            for name, value in head_direction.items() if name in d4_names
        }
        if d4_direction:
            d4_potential_dot += d4_global_parameter_charge_potential_direction_dense(
                positions, moments.charges, d4_cn, worker._d4_models,
                d4_parameters, pair, d4_direction,
            )
        ao_atom = np.asarray(basis.spherical_atom)
        ao_labels = tuple(basis.spherical_shell)
        ao_zeta = np.asarray([
            next(shell.slater_exponent for shell in state.elements[atom].shells
                 if shell.label == label)
            for atom, label in zip(ao_atom, ao_labels, strict=True)
        ])
        shell_slot = {
            (atom, shell.label): (atom, slot)
            for atom, element in enumerate(state.elements)
            for slot, shell in enumerate(element.shells)
        }
        exponent_direction = np.array(
            head_direction.get("shell.exp", np.zeros((natoms, 4))), dtype=float, copy=True,
        )
        for atom, element in enumerate(state.elements):
            for slot, _shell in enumerate(element.shells):
                if f"atom {atom} shell {slot}: exp" in state.floor_events:
                    exponent_direction[atom, slot] = 0.0
        ao_dzeta = np.asarray([
            exponent_direction[shell_slot[(atom, label)]]
            for atom, label in zip(ao_atom, ao_labels, strict=True)
        ])
        core_prefactor = extended_hueckel_hamiltonian(
            np.ones_like(orbital_state.overlap), worker._orbital_metadata,
            self.atomic_numbers, positions, state.atoms, state.orbitals,
            global_scalings=state.globals,
        )
        active_h0 = ao_atom[:, None] != ao_atom[None, :]
        core_prefactor[~active_h0] = 0.0
        core_direction = np.zeros_like(core_prefactor)
        h0_names = {"shell.lev", "shell.kcn", "shell.poly", "shell.h0exp", "atom.arad", "atom.en",
                    "global.ks", "global.kp", "global.kd", "global.ksd",
                    "global.kpd", "global.enscale", "global.wexp"}
        _, h0_sources = analytic_h0_head_adjoints(
            self.atomic_numbers, positions, state, np.ones_like(orbital_state.overlap),
            np.zeros_like(density), orbital_state.ao_atom, orbital_state.ao_shell,
            return_fock_derivatives=True,
        )
        for name in h0_names & head_direction.keys():
            core_direction += np.einsum(
                "ij,ijab->ab", np.asarray(head_direction[name]), h0_sources[name],
            )
        for left in range(basis.norbitals):
            for right in range(left):
                if not active_h0[left, right]:
                    continue
                ln = (
                    0.25 * (ao_dzeta[left] / ao_zeta[left]
                            + ao_dzeta[right] / ao_zeta[right])
                    - 0.5 * (ao_dzeta[left] + ao_dzeta[right])
                    / (ao_zeta[left] + ao_zeta[right])
                ) * (2.0 * state.globals.get("wexp", 0.5))
                core_direction[left, right] += core_prefactor[left, right] * ln
                core_direction[right, left] += core_prefactor[left, right] * ln
        cartesian = basis.cartesian_orbitals
        ncart = len(cartesian)
        transform = basis.cartesian_to_spherical
        rows, columns = triangular_indices(basis.norbitals)
        zero_derivatives = np.zeros((len(rows), 13, 3))
        result = np.zeros((natoms, 4))
        for atom, label in shell_slot:
            key = (atom, label)
            if f"atom {atom} shell {shell_slot[key][1]}: exp" in state.floor_events:
                continue
            cart_first = np.zeros((13, ncart, ncart))
            cart_second = np.zeros_like(cart_first)
            for bra_index, bra in enumerate(cartesian):
                for ket_index in range(bra_index + 1):
                    ket = cartesian[ket_index]
                    bra_key = (bra.atom_index, bra.shell)
                    ket_key = (ket.atom_index, ket.shell)
                    bra_hit = bra_key == key
                    ket_hit = ket_key == key
                    if not bra_hit and not ket_hit:
                        continue
                    _, dbra, dket = _nn_exponent_integral_response(bra, ket, "first")
                    bb, bk, kk = _nn_exponent_integral_response(bra, ket, "second")
                    bra_dot = exponent_direction[shell_slot[bra_key]]
                    ket_dot = exponent_direction[shell_slot[ket_key]]
                    first = (dbra if bra_hit else 0.0) + (dket if ket_hit else 0.0)
                    second = (
                        (bra_dot * bb + ket_dot * bk if bra_hit else 0.0)
                        + (bra_dot * bk + ket_dot * kk if ket_hit else 0.0)
                    )
                    cart_first[:, bra_index, ket_index] = first
                    cart_first[:, ket_index, bra_index] = first
                    cart_second[:, bra_index, ket_index] = second
                    cart_second[:, ket_index, bra_index] = second
            first = np.asarray([transform.T @ matrix @ transform for matrix in cart_first])
            second = np.asarray([transform.T @ matrix @ transform for matrix in cart_second])
            ds = first[0]
            dds = second[0]
            h0 = core_prefactor * ds
            h0_dot = core_direction * ds + core_prefactor * dds
            for left in range(basis.norbitals):
                for right in range(left):
                    if not active_h0[left, right]:
                        continue
                    first_ln = second_ln = 0.0
                    for endpoint in (left, right):
                        if (ao_atom[endpoint], ao_labels[endpoint]) == key:
                            first_ln += (
                                0.25 / ao_zeta[endpoint]
                                - 0.5 / (ao_zeta[left] + ao_zeta[right])
                            ) * (2.0 * state.globals.get("wexp", 0.5))
                            second_ln += (
                                -0.25 * ao_dzeta[endpoint] / ao_zeta[endpoint]**2
                                + 0.5 * (ao_dzeta[left] + ao_dzeta[right])
                                / (ao_zeta[left] + ao_zeta[right])**2
                            ) * (2.0 * state.globals.get("wexp", 0.5))
                            second_ln += float(np.asarray(head_direction.get("global.wexp", [[0.0]]))[0, 0]) * (
                                0.5 / ao_zeta[endpoint] - 1.0 / (ao_zeta[left] + ao_zeta[right])
                            )
                    h0_value = core_prefactor[left, right] * orbital_state.overlap[left, right] * first_ln
                    h0_dot_value = (
                        core_direction[left, right] * orbital_state.overlap[left, right] * first_ln
                        + core_prefactor[left, right] * ds_direction[left, right] * first_ln
                        + core_prefactor[left, right] * orbital_state.overlap[left, right] * second_ln
                    )
                    h0[left, right] += h0_value
                    h0[right, left] += h0_value
                    h0_dot[left, right] += h0_dot_value
                    h0_dot[right, left] += h0_dot_value

            def derivative_operators(moments_direction):
                integrals = SparseAtomicMultipoleIntegrals(
                    basis.norbitals, rows, columns,
                    moments_direction[0, rows, columns],
                    moments_direction[1:4, rows, columns].T,
                    moments_direction[4:, rows, columns].T.reshape(-1, 3, 3),
                    zero_derivatives, zero_derivatives,
                )
                return prepare_pair_cumulative_multipole_operators(
                    integrals, positions, basis.spherical_atom,
                )

            first_operators = derivative_operators(first)
            second_operators = derivative_operators(second)
            q_first, moment_first = dense_scc_observables(
                density, first_operators, partition.orbital_shell,
                np.zeros(partition.nshells), np.zeros(natoms),
            )
            q_second_density, moment_second_density = dense_scc_observables(
                dp, first_operators, partition.orbital_shell,
                np.zeros(partition.nshells), np.zeros(natoms),
            )
            q_second_operator, moment_second_operator = dense_scc_observables(
                density, second_operators, partition.orbital_shell,
                np.zeros(partition.nshells), np.zeros(natoms),
            )
            q_second = q_second_density + q_second_operator
            moment_second = AtomicMultipoles(
                moment_second_density.charges + moment_second_operator.charges,
                moment_second_density.dipoles_bohr + moment_second_operator.dipoles_bohr,
                moment_second_density.quadrupoles_bohr2
                + moment_second_operator.quadrupoles_bohr2,
            )
            value = float(np.sum(dp * h0) + np.sum(density * h0_dot)
                          - np.sum(dw * ds) - np.sum(weighted * dds))
            value += float(shell_potential_dot @ q_first + shell_potential @ q_second)
            total_qpot = qpot + d4_potential
            total_qpot_dot = qpot_dot + d4_potential_dot
            value += float(total_qpot_dot @ moment_first.charges
                           + total_qpot @ moment_second.charges)
            value += float(np.sum(mupot_dot * moment_first.dipoles_bohr)
                           + np.sum(mupot * moment_second.dipoles_bohr))
            value += float(np.sum(thetapot_dot * moment_first.quadrupoles_bohr2)
                           + np.sum(thetapot * moment_second.quadrupoles_bohr2))
            result[shell_slot[key]] = value
        return result

    def evaluate_nn_exponent_head_geometry_response(
        self, coordinates_angstrom: np.ndarray,
        direction_bohr: np.ndarray,
    ) -> np.ndarray:
        """Analytic ``E_exp,R dR`` at frozen NN shifts, including CPSCF.

        The AO mixed derivatives are contracted with the existing CAMM
        coordinate-direction kernel. The electronic direction comes from the
        resident ground-state CPSCF solver; no displaced SCC is evaluated.
        """
        from .ground_state_response import solve_ground_state_directional_response

        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN exponent geometry response requires nn_enabled=True")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        direction = np.asarray(direction_bohr, dtype=float)
        natoms = len(self.atomic_numbers)
        if (coordinates.shape != (natoms, 3) or direction.shape != (natoms, 3)
                or not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(direction))):
            raise ValueError("NN exponent geometry direction must be finite natoms x 3")
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        fixed, worker = self._nn_fixed_parameter_point(
            positions, shifts, retain_orbitals=True, return_worker=True,
        )
        orbital_state = fixed.orbital_state
        if not fixed.converged or orbital_state is None:
            raise RuntimeError("NN exponent geometry response requires converged SCC")
        state = build_nn_parameter_state(self.atomic_numbers, self.parameters, shifts)
        context = worker._prepare_density_response_gradient_context(orbital_state)
        basis = context["basis"]
        partition = context["partition"]
        kernel = context["kernel"]
        integrals = context["integrals"]
        derivatives = worker.ground_state_derivative_matrices(orbital_state)
        ds = np.einsum("ab,abij->ij", direction, derivatives.overlap_per_bohr)
        df = np.einsum("ab,abij->ij", direction, derivatives.fock_hartree_per_bohr)
        response = solve_ground_state_directional_response(
            orbital_state, ds, df, kernel,
        )
        density = np.asarray(fixed.density)
        dp = response.density_derivative
        occupations = orbital_state.alpha_occupations + orbital_state.beta_occupations
        weighted = energy_weighted_density(
            orbital_state.coefficients, occupations,
            orbital_state.orbital_energies_hartree,
        )
        dw = response.weighted_density_derivative
        first_operators = kernel.operators
        q_shell = np.asarray(kernel.reference_shell_charges)
        moments = kernel.reference_multipoles
        shell_index = np.asarray(partition.orbital_shell, dtype=np.intp)
        q_density, moment_density = dense_scc_observables(
            dp, first_operators, shell_index,
            np.zeros(partition.nshells), np.zeros(natoms),
        )
        q_operator = -np.bincount(
            shell_index, weights=np.sum(density * ds, axis=1),
            minlength=partition.nshells,
        )
        qdot = q_density + q_operator
        moment_operator = cumulative_atomic_multipole_directional_derivative_sparse(
            context["reference_density"], integrals, positions, direction,
            basis.spherical_atom,
        )
        moment_dot = AtomicMultipoles(
            moment_density.charges + moment_operator.charges,
            moment_density.dipoles_bohr + moment_operator.dipoles_bohr,
            moment_density.quadrupoles_bohr2 + moment_operator.quadrupoles_bohr2,
        )
        third = np.asarray(partition.third_order_hartree)
        interaction = kernel.shell_interaction
        shell_potential = interaction @ q_shell + third * q_shell**2
        if hasattr(interaction, "potential_direction"):
            shell_dot = interaction.potential_direction(q_shell, qdot, direction)
        else:
            shell_atom = np.asarray(partition.shell_atom, dtype=np.intp)
            offset = positions[shell_atom, None, :] - positions[None, shell_atom, :]
            offset_dot = direction[shell_atom, None, :] - direction[None, shell_atom, :]
            hardness = np.asarray(partition.hardness_hartree)
            average = 0.5 * (hardness[:, None] + hardness[None, :])
            distance2 = np.einsum("ijk,ijk->ij", offset, offset)
            alphaj = state.globals["alphaj"]
            power = distance2 ** (0.5 * alphaj)
            radial = np.zeros_like(distance2)
            np.power(distance2, 0.5 * (alphaj - 2.0), out=radial,
                     where=distance2 > 0.0)
            interaction_dot = -(
                np.einsum("ijk,ijk->ij", offset, offset_dot)
                * radial * (power + average**-alphaj)**(-1.0 / alphaj - 1.0)
            )
            shell_dot = interaction @ qdot + interaction_dot @ q_shell
        shell_dot += 2.0 * third * q_shell * qdot
        cn_direction = coordination_number_directional_derivative_sparse(
            positions, direction,
            np.asarray([item.covalent_radius_bohr for item in state.atoms]),
            cutoff_bohr=max(24.0, float(np.max(np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=2,
            ))) + 1.0),
        )
        aes_parameters = state.anisotropic_parameters(worker._active_coordination)
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            moments, positions, aes_parameters,
        )
        aes_base = np.asarray([item.multipole_radius_bohr for item in state.elements])
        aes_valence = np.asarray([
            item.multipole_valence_coordination for item in state.elements
        ])
        aes_exponent = state.globals["aesexp"]
        aes_logistic = np.exp(-np.logaddexp(
            0.0, -aes_exponent * (
                worker._active_coordination - aes_valence - state.globals["aesshift"]
            ),
        ))
        aes_radius_direction = (
            aes_exponent * (state.globals["aesrmax"] - aes_base)
            * aes_logistic * (1.0 - aes_logistic) * cn_direction
        )
        qpot_dot, mupot_dot, thetapot_dot = anisotropic_moment_potential_direction(
            moments, moment_dot, positions, direction, aes_parameters,
            aes_radius_direction,
        )
        d4_cn = kernel.d4_coordination
        d4_cn_direction = coordination_number_directional_derivative_sparse(
            positions, direction,
            _covalent_radii()[self._atomic_number_array - 1],
            electronegativities=worker._electronegativities,
            cutoff_bohr=max(24.0, float(np.max(np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=2,
            ))) + 1.0),
        )
        pair = kernel._d4(moments.charges)
        d4_potential = pair.charge_derivative_hartree.copy()
        if natoms >= 3:
            _, atm = atm_three_body_energy_charge_derivative_dense(
                positions, moments.charges, d4_cn,
                worker._d4_models, worker._d4_parameters, pair,
            )
            d4_potential += atm
        d4_dot = d4_charge_potential_geometry_direction_dense(
            positions, direction, moments.charges, d4_cn, d4_cn_direction,
            worker._d4_models, worker._d4_parameters,
        )
        d4_dot += d4_charge_hessian(
            moments.charges, kernel.d4_prepared,
        ) @ moment_dot.charges
        if kernel.atm_charge_hessian is not None:
            d4_dot += kernel.atm_charge_hessian @ moment_dot.charges
        ao_atom = np.asarray(basis.spherical_atom, dtype=np.intp)
        ao_labels = tuple(basis.spherical_shell)
        zeta = np.asarray([
            next(shell.slater_exponent for shell in state.elements[atom].shells
                 if shell.label == label)
            for atom, label in zip(ao_atom, ao_labels, strict=True)
        ])
        core = extended_hueckel_hamiltonian(
            np.ones_like(orbital_state.overlap), worker._orbital_metadata,
            self.atomic_numbers, positions, state.atoms, state.orbitals,
            global_scalings=state.globals,
        )
        active_h0 = ao_atom[:, None] != ao_atom[None, :]
        core[~active_h0] = 0.0
        zero_integrals = replace(
            integrals, overlap=np.ones_like(integrals.overlap),
            left_derivatives=np.zeros_like(integrals.left_derivatives),
            right_derivatives=np.zeros_like(integrals.right_derivatives),
        )
        core_derivatives = extended_hueckel_hamiltonian_derivatives_sparse(
            zero_integrals, worker._orbital_metadata,
            self.atomic_numbers, positions, state.atoms, state.orbitals,
            worker._active_coordination, global_scalings=state.globals,
        )
        core_direction = np.zeros_like(core)
        for pair_index, (row, column) in enumerate(zip(
            integrals.rows, integrals.columns, strict=True,
        )):
            a, b = ao_atom[row], ao_atom[column]
            value = (
                np.dot(core_derivatives.left[pair_index], direction[a])
                + np.dot(core_derivatives.right[pair_index], direction[b])
                + core_derivatives.coordination_left[pair_index] * cn_direction[a]
                + core_derivatives.coordination_right[pair_index] * cn_direction[b]
            )
            core_direction[row, column] = core_direction[column, row] = value
        core_direction[~active_h0] = 0.0
        cartesian = basis.cartesian_orbitals
        transform = basis.cartesian_to_spherical
        size = basis.norbitals
        rows, columns = triangular_indices(size)
        dense_sparse = SparseSymmetricMatrix(size, rows, columns, density[rows, columns])
        slots = {
            (atom, shell.label): (atom, slot)
            for atom, element in enumerate(state.elements)
            for slot, shell in enumerate(element.shells)
        }
        result = np.zeros((natoms, 4))
        for key, slot in slots.items():
            if f"atom {key[0]} shell {slot[1]}: exp" in state.floor_events:
                continue
            cart_first = np.zeros((13, len(cartesian), len(cartesian)))
            cart_left = np.zeros((13, 3, len(cartesian), len(cartesian)))
            cart_right = np.zeros_like(cart_left)
            for bra_index, bra in enumerate(cartesian):
                for ket_index in range(bra_index + 1):
                    ket = cartesian[ket_index]
                    bra_hit = (bra.atom_index, bra.shell) == key
                    ket_hit = (ket.atom_index, ket.shell) == key
                    if not (bra_hit or ket_hit):
                        continue
                    _, dbra, dket = _nn_exponent_integral_response(bra, ket, "first")
                    b_l, b_r, k_l, k_r = _nn_exponent_integral_response(
                        bra, ket, "mixed",
                    )
                    first = (dbra if bra_hit else 0.0) + (dket if ket_hit else 0.0)
                    left = (b_l if bra_hit else 0.0) + (k_l if ket_hit else 0.0)
                    right = (b_r if bra_hit else 0.0) + (k_r if ket_hit else 0.0)
                    cart_first[:, bra_index, ket_index] = first
                    cart_first[:, ket_index, bra_index] = first
                    cart_left[:, :, bra_index, ket_index] = left
                    cart_left[:, :, ket_index, bra_index] = right
                    cart_right[:, :, bra_index, ket_index] = right
                    cart_right[:, :, ket_index, bra_index] = left
            first = np.asarray([transform.T @ matrix @ transform for matrix in cart_first])
            left = np.asarray([
                [transform.T @ matrix @ transform for matrix in component]
                for component in cart_left
            ])
            right = np.asarray([
                [transform.T @ matrix @ transform for matrix in component]
                for component in cart_right
            ])
            ds_exp = first[0]
            ds_exp_dot = np.zeros_like(ds_exp)
            for pair_index, (row, column) in enumerate(zip(rows, columns, strict=True)):
                a, b = ao_atom[row], ao_atom[column]
                value = (np.dot(left[0, :, row, column], direction[a])
                         + np.dot(right[0, :, row, column], direction[b]))
                ds_exp_dot[row, column] = ds_exp_dot[column, row] = value
            mixed_integrals = SparseAtomicMultipoleIntegrals(
                size, rows, columns, first[0, rows, columns],
                first[1:4, rows, columns].T,
                first[4:, rows, columns].T.reshape(-1, 3, 3),
                left[:, :, rows, columns].transpose(2, 0, 1),
                right[:, :, rows, columns].transpose(2, 0, 1),
            )
            exp_operators = prepare_pair_cumulative_multipole_operators(
                mixed_integrals, positions, basis.spherical_atom,
            )
            q_exp, moment_exp = dense_scc_observables(
                density, exp_operators, shell_index,
                np.zeros(partition.nshells), np.zeros(natoms),
            )
            q_exp_density, moment_exp_density = dense_scc_observables(
                dp, exp_operators, shell_index,
                np.zeros(partition.nshells), np.zeros(natoms),
            )
            q_exp_mixed = -np.bincount(
                shell_index, weights=np.sum(density * ds_exp_dot, axis=1),
                minlength=partition.nshells,
            )
            q_exp_dot = q_exp_density + q_exp_mixed
            moment_exp_mixed = cumulative_atomic_multipole_directional_derivative_sparse(
                dense_sparse, mixed_integrals, positions, direction,
                basis.spherical_atom,
            )
            moment_exp_dot = AtomicMultipoles(
                moment_exp_density.charges + moment_exp_mixed.charges,
                moment_exp_density.dipoles_bohr + moment_exp_mixed.dipoles_bohr,
                moment_exp_density.quadrupoles_bohr2
                + moment_exp_mixed.quadrupoles_bohr2,
            )
            h0 = core * ds_exp
            h0_dot = core_direction * ds_exp + core * ds_exp_dot
            for row in range(size):
                for column in range(row):
                    if not active_h0[row, column]:
                        continue
                    logarithmic = sum(
                        state.globals.get("wexp", 0.5) * (0.5 / zeta[endpoint]
                        - 1.0 / (zeta[row] + zeta[column]))
                        for endpoint in (row, column)
                        if (ao_atom[endpoint], ao_labels[endpoint]) == key
                    )
                    if logarithmic:
                        value = core[row, column] * orbital_state.overlap[row, column] * logarithmic
                        value_dot = (
                            core_direction[row, column] * orbital_state.overlap[row, column]
                            + core[row, column] * ds[row, column]
                        ) * logarithmic
                        h0[row, column] += value
                        h0[column, row] += value
                        h0_dot[row, column] += value_dot
                        h0_dot[column, row] += value_dot
            value = float(
                np.sum(dp * h0) + np.sum(density * h0_dot)
                - np.sum(dw * ds_exp) - np.sum(weighted * ds_exp_dot)
            )
            value += float(shell_dot @ q_exp + shell_potential @ q_exp_dot)
            value += float((qpot_dot + d4_dot) @ moment_exp.charges
                           + (qpot + d4_potential) @ moment_exp_dot.charges)
            value += float(np.sum(mupot_dot * moment_exp.dipoles_bohr)
                           + np.sum(mupot * moment_exp_dot.dipoles_bohr))
            value += float(np.sum(thetapot_dot * moment_exp.quadrupoles_bohr2)
                           + np.sum(thetapot * moment_exp_dot.quadrupoles_bohr2))
            result[slot] = value
        return result

    def _nn_parameter_response_state(
        self, coordinates_angstrom: np.ndarray,
        *, head_step_fraction: float = 1.0e-4,
    ) -> tuple[GFN2Result, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Fixed-parameter analytic force and stationary SCC head adjoints.

        Every active molecular head is differentiated at the converged density;
        an unknown head fails explicitly instead of triggering SCC probes.
        ``head_step_fraction`` is retained only for caller compatibility.
        """
        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN parameter response requires nn_enabled=True")
        if not np.isfinite(head_step_fraction) or head_step_fraction <= 0.0:
            raise ValueError("NN head step fraction must be positive and finite")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN-xTB coordinates must be finite natoms x 3")
        positions_bohr = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions_bohr,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )

        fixed, worker = self._nn_fixed_parameter_point(
            positions_bohr, shifts, energy_only=False, retain_orbitals=True,
            return_worker=True,
        )
        if not fixed.converged:
            raise RuntimeError("NN-xTB SCC did not converge for parameter response")
        state = build_nn_parameter_state(self.atomic_numbers, self.parameters, shifts)
        direct_adjoints = analytic_repulsion_head_adjoints(
            self.atomic_numbers, positions_bohr, state,
        )
        if fixed.orbital_state is None:
            raise RuntimeError("NN-xTB analytic head adjoints require retained orbitals")
        direct_adjoints.update(analytic_h0_head_adjoints(
            self.atomic_numbers, positions_bohr, state,
            fixed.orbital_state.overlap, fixed.density,
            fixed.orbital_state.ao_atom, fixed.orbital_state.ao_shell,
        ))
        basis = build_molecular_basis(
            self.atomic_numbers, positions_bohr, self.parameters,
            per_atom_elements=state.basis_elements,
        )
        partition = _shell_partition(
            basis, self.atomic_numbers, self.parameters,
            per_atom_elements=state.basis_elements,
            per_atom_third_scales=state.third_scales,
        )
        shell_labels = [None] * partition.nshells
        for orbital, index in enumerate(partition.orbital_shell):
            shell_labels[index] = basis.spherical_shell[orbital]
        q_shell = shell_charges(
            fixed.density, fixed.orbital_state.overlap, partition,
        )
        direct_adjoints.update(analytic_third_order_head_adjoints(
            self.atomic_numbers, state, q_shell,
            partition.shell_atom, shell_labels,
        ))
        direct_adjoints.update(analytic_scc_hardness_head_adjoints(
            self.atomic_numbers, positions_bohr, state, q_shell,
            partition.shell_atom, shell_labels,
        ))
        if worker._post_scf_pair_integrals is None or worker._active_coordination is None:
            raise RuntimeError("NN-xTB analytic AES adjoints require multipole integrals and CN")
        operators = prepare_pair_cumulative_multipole_operators(
            worker._post_scf_pair_integrals, positions_bohr, basis.spherical_atom,
        )
        _, multipoles = dense_scc_observables(
            fixed.density, operators, partition.orbital_shell,
            partition.reference_population, worker._valence,
        )
        direct_adjoints.update(analytic_aes_head_adjoints(
            positions_bohr, worker._active_coordination, state, multipoles,
        ))
        direct_adjoints["atom.rcov"] = analytic_rcov_head_adjoints(
            positions_bohr, state,
            direct_adjoints["shell.lev"], direct_adjoints["atom.mpvcn"],
        )
        d4_cn = _d4_coordination(
            self.atomic_numbers, positions_bohr,
            _covalent_radii(), worker._electronegativities,
        )
        d4_parameters = worker._d4_parameters
        d4_prepared = prepare_d4_geometry(
            positions_bohr, d4_cn, worker._d4_models, d4_parameters,
        )
        d4_pair = d4_two_body(
            positions_bohr, multipoles.charges, d4_cn,
            worker._d4_models, d4_parameters, prepared=d4_prepared,
        )
        d4_charge_potential = d4_pair.charge_derivative_hartree.copy()
        if len(positions_bohr) >= 3:
            _, atm_charge_potential = atm_three_body_energy_charge_derivative_dense(
                positions_bohr, multipoles.charges, d4_cn,
                worker._d4_models, d4_parameters, d4_pair,
            )
            d4_charge_potential += atm_charge_potential
        direct_adjoints.update({
            f"global.{name}": np.array([[value]])
            for name, value in d4_global_parameter_partials_dense(
                positions_bohr, d4_pair, d4_parameters, d4_prepared,
            ).items()
        })
        if "shell.exp" in shifts:
            direct_adjoints["shell.exp"] = _analytic_nn_exponent_head_adjoint(
                self.atomic_numbers, positions_bohr, state, basis, partition,
                fixed, worker, q_shell, multipoles, d4_charge_potential,
                shifts["shell.exp"].shape,
            )
        active = {head.name for head in self.nn_overlay.info.heads}
        missing = active - direct_adjoints.keys()
        if missing:
            raise NotImplementedError(
                f"analytic NN parameter adjoints missing: {sorted(missing)}"
            )
        adjoints = {
            head.name: direct_adjoints[head.name]
            * _MATRIX_SHIFT_FACTORS.get(head.name, 1.0)
            for head in self.nn_overlay.info.heads
        }
        return fixed, adjoints, shifts

    def evaluate_nn_gradient_parameter_response(
        self, coordinates_angstrom: np.ndarray,
        *, head_step_fraction: float = 1.0e-4,
    ) -> tuple[float, np.ndarray]:
        """Analytic coordinate gradient with converged-SCC NN head adjoints."""
        fixed, adjoints, _ = self._nn_parameter_response_state(
            coordinates_angstrom, head_step_fraction=head_step_fraction,
        )
        response = self.nn_overlay.parameter_shift_vjp(
            self.atomic_numbers, np.asarray(coordinates_angstrom) * ANGSTROM_TO_BOHR, adjoints,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        return fixed.energy_hartree, fixed.gradient_hartree_per_bohr + response

    def evaluate_nn_fixed_parameter_hessian(
        self, coordinates_angstrom: np.ndarray,
    ):
        """Analytic CPSCF Hessian with NN shifts frozen at this geometry.

        This is only ``E_RR`` in the Hessian of ``E(R, p(R))``. It excludes
        mixed geometry/parameter response, second parameter response and
        encoder curvature, so it must not be used as the total NN Hessian.
        """
        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("fixed-parameter NN Hessian requires nn_enabled=True")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN-xTB coordinates must be finite natoms x 3")
        positions = coordinates * ANGSTROM_TO_BOHR
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions,
            charge=float(self.charge), unpaired=float(self.multiplicity - 1),
        )
        worker = self._nn_frozen_parameter_worker(shifts)
        fixed = worker._single_point(
            positions, compute_bond_orders=False, collect_telemetry=False,
            retain_orbitals=True,
        )
        if not fixed.converged or fixed.orbital_state is None:
            raise RuntimeError("fixed-parameter NN Hessian requires a converged SCF state")
        return worker.ground_state_hessian(fixed.orbital_state)

    def evaluate_nn_repulsion_chain_hessian(
        self, coordinates_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Fully analytic NN-chain Hessian of the density-free pair repulsion.

        This is a separately testable *component* of the total NN Hessian;
        it does not include H0, SCC, AES or dispersion derivatives.
        """
        from .gradients import repulsion_hessian_vector_product_sparse
        from .nn_xtb_parameters import (
            analytic_repulsion_head_adjoints,
            analytic_repulsion_head_adjoint_directional,
        )

        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN repulsion Hessian requires nn_enabled=True")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        natoms = len(self.atomic_numbers)
        if coordinates.shape != (natoms, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN repulsion coordinates must be finite natoms x 3")
        positions = coordinates * ANGSTROM_TO_BOHR
        charge = float(self.charge)
        unpaired = float(self.multiplicity - 1)
        shifts = self.nn_overlay.matrix_parameter_shifts(
            self.atomic_numbers, positions, charge=charge, unpaired=unpaired,
        )
        state = build_nn_parameter_state(self.atomic_numbers, self.parameters, shifts)
        head_names = tuple(name for name in
            ("atom.repa", "atom.repb", "global.kexp", "global.klight") if name in shifts)
        adjoints = analytic_repulsion_head_adjoints(self.atomic_numbers, positions, state)
        scale = lambda values: {
            name: values[name] * _MATRIX_SHIFT_FACTORS.get(name, 1.0)
            for name in head_names
        }
        dimension = positions.size
        directions = np.eye(dimension).reshape((dimension, natoms, 3))
        mixed = [analytic_repulsion_head_adjoint_directional(
            self.atomic_numbers, positions, state, direction, {},
        ) for direction in directions]
        result = np.empty((dimension, dimension))
        cutoff = max(24.0, float(np.max(np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2,
        ))) + 1.0)
        for column, direction in enumerate(directions):
            raw = self.nn_overlay.parameter_shift_jvp(
                self.atomic_numbers, positions, direction,
                charge=charge, unpaired=unpaired,
            )
            dp = {name: raw[name] * _MATRIX_SHIFT_FACTORS.get(name, 1.0)
                  for name in head_names}
            parameter = analytic_repulsion_head_adjoint_directional(
                self.atomic_numbers, positions, state,
                np.zeros_like(direction), dp,
            )
            fixed = repulsion_hessian_vector_product_sparse(
                self.atomic_numbers, positions, direction, state.atoms,
                cutoff_bohr=cutoff, kexp=state.globals["kexp"],
                klight=state.globals["klight"],
            ).reshape(-1)
            explicit_mixed = np.array([
                sum(float(np.sum(mixed[row][name] * dp[name]))
                    for name in head_names)
                for row in range(dimension)
            ])
            response = {name: mixed[column][name] + parameter[name]
                        for name in head_names}
            network_response = self.nn_overlay.parameter_shift_vjp(
                self.atomic_numbers, positions, scale(response),
                charge=charge, unpaired=unpaired,
            ).reshape(-1)
            _, curvature = self.nn_overlay.parameter_shift_vjp(
                self.atomic_numbers, positions, scale(adjoints),
                charge=charge, unpaired=unpaired,
                hessian_direction=direction,
            )
            result[:, column] = (
                fixed + explicit_mixed + network_response + curvature.reshape(-1)
            )
        reciprocity = float(np.max(np.abs(result - result.T)))
        if reciprocity > 1.0e-8:
            raise ArithmeticError(
                f"analytic NN repulsion Hessian violates reciprocity: {reciprocity:.3e}"
            )
        return result

    def evaluate_nn_hessian_analytic(self, coordinates_angstrom: np.ndarray):
        """Complete analytic NN Hessian, including all CPSCF chain-rule blocks."""
        from .nn_xtb_hessian import analytic_nn_hessian

        return analytic_nn_hessian(self, coordinates_angstrom)

    def evaluate_nn_hessian_mixed_response(
        self, coordinates_angstrom: np.ndarray, *, step_bohr: float = 1.0e-3,
    ):
        """Compatibility entry point for the complete analytic NN Hessian.

        ``step_bohr`` is validated for compatibility but is not used. The
        displaced-SCC diagnostic is explicitly named ``..._audit``.
        """
        if not np.isfinite(step_bohr) or step_bohr <= 0.0:
            raise ValueError("NN Hessian compatibility step must be positive and finite")
        return self.evaluate_nn_hessian_analytic(coordinates_angstrom)

    def evaluate_nn_hessian_mixed_response_audit(
        self, coordinates_angstrom: np.ndarray, *, step_bohr: float = 1.0e-3,
    ) -> NNXTBSemiAnalyticHessian:
        """Assemble all four NN Hessian chain-rule blocks directionally.

        For ``E(R, p(R))`` the blocks are ``E_RR``, ``E_Rp p_R``,
        ``p_R^T (E_pR + E_pp p_R)``, and ``E_p p_RR``. The encoder
        Jacobian and curvature are analytic; the SCC parameter adjoints and
        their directional responses are evaluated by converged probes.
        """
        if not self.nn_enabled or self.nn_overlay is None:
            raise ValueError("NN mixed-response Hessian requires nn_enabled=True")
        if not np.isfinite(step_bohr) or step_bohr <= 0.0:
            raise ValueError("NN mixed-response Hessian step must be positive and finite")
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("NN-xTB coordinates must be finite natoms x 3")
        positions = coordinates * ANGSTROM_TO_BOHR
        fixed, adjoints, shifts = self._nn_parameter_response_state(coordinates)
        dimension = positions.size
        fixed_block = np.empty((dimension, dimension))
        parameter_block = np.empty_like(fixed_block)
        adjoint_block = np.empty_like(fixed_block)
        curvature_block = np.empty_like(fixed_block)
        chain_error = 0.0
        charge = float(self.charge)
        unpaired = float(self.multiplicity - 1)

        def network_vjp(position, weights):
            return self.nn_overlay.parameter_shift_vjp(
                self.atomic_numbers, position, weights,
                charge=charge, unpaired=unpaired,
            )

        for column in range(dimension):
            direction = np.zeros_like(positions)
            direction.reshape(-1)[column] = 1.0
            plus = positions + step_bohr * direction
            minus = positions - step_bohr * direction
            plus_fixed, plus_adjoints, _ = self._nn_parameter_response_state(
                plus / ANGSTROM_TO_BOHR,
            )
            minus_fixed, minus_adjoints, _ = self._nn_parameter_response_state(
                minus / ANGSTROM_TO_BOHR,
            )

            fixed_plus = self._nn_fixed_parameter_point(plus, shifts)
            fixed_minus = self._nn_fixed_parameter_point(minus, shifts)
            fixed_block[:, column] = (
                fixed_plus.gradient_hartree_per_bohr
                - fixed_minus.gradient_hartree_per_bohr
            ).reshape(-1) / (2.0 * step_bohr)

            raw_direction = self.nn_overlay.parameter_shift_jvp(
                self.atomic_numbers, positions, direction,
                charge=charge, unpaired=unpaired,
            )
            shift_direction = {
                name: raw * _MATRIX_SHIFT_FACTORS.get(name, 1.0)
                for name, raw in raw_direction.items()
            }
            shifted_plus = {
                name: value + step_bohr * shift_direction[name]
                for name, value in shifts.items()
            }
            shifted_minus = {
                name: value - step_bohr * shift_direction[name]
                for name, value in shifts.items()
            }
            parameter_plus = self._nn_fixed_parameter_point(positions, shifted_plus)
            parameter_minus = self._nn_fixed_parameter_point(positions, shifted_minus)
            parameter_block[:, column] = (
                parameter_plus.gradient_hartree_per_bohr
                - parameter_minus.gradient_hartree_per_bohr
            ).reshape(-1) / (2.0 * step_bohr)

            directional_adjoints = {
                name: (plus_adjoints[name] - minus_adjoints[name])
                / (2.0 * step_bohr)
                for name in adjoints
            }
            adjoint_block[:, column] = network_vjp(
                positions, directional_adjoints,
            ).reshape(-1)
            _, curvature = self.nn_overlay.parameter_shift_vjp(
                self.atomic_numbers, positions, adjoints,
                charge=charge, unpaired=unpaired,
                hessian_direction=direction,
            )
            curvature_block[:, column] = curvature.reshape(-1)

            direct_column = (
                plus_fixed.gradient_hartree_per_bohr
                + network_vjp(plus, plus_adjoints)
                - minus_fixed.gradient_hartree_per_bohr
                - network_vjp(minus, minus_adjoints)
            ).reshape(-1) / (2.0 * step_bohr)
            assembled = (
                fixed_block[:, column] + parameter_block[:, column]
                + adjoint_block[:, column] + curvature_block[:, column]
            )
            chain_error = max(chain_error, float(np.max(np.abs(assembled - direct_column))))

        raw_hessian = fixed_block + parameter_block + adjoint_block + curvature_block
        reciprocity = float(np.max(np.abs(raw_hessian - raw_hessian.T)))

        def symmetric(block):
            return 0.5 * (block + block.T)

        return NNXTBSemiAnalyticHessian(
            symmetric(raw_hessian), symmetric(fixed_block),
            symmetric(parameter_block), symmetric(adjoint_block),
            symmetric(curvature_block), reciprocity, chain_error,
        )

    def evaluate_energy(self, coordinates_angstrom: np.ndarray) -> float:
        """Return a resident dense SCC energy without the analytic gradient.

        This lightweight route is used for the two primitive-tangent X-H energy points
        in the frozen ZAFF two-Morse protocol.
        """
        if self.nn_enabled:
            return float(self.evaluate_nn_energy_development(
                coordinates_angstrom, self.nn_overlay,
            ))
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("GFN2 energy coordinates must be finite natoms x 3")
        result = self._single_point(
            coordinates * ANGSTROM_TO_BOHR,
            compute_bond_orders=False,
            retain_orbitals=False,
            collect_telemetry=True,
            energy_only=True,
        )
        return float(result)

    def evaluate(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        gradient_step_bohr: float | None = None,
        compute_cm5: bool = False,
        compute_bond_orders: bool = True,
        retain_orbitals: bool = False,
        retain_local_projectors: bool = False,
        collect_telemetry: bool = True,
    ) -> GFN2Result:
        if compute_cm5:
            raise ValueError(
                "MATRIX-GFN2 does not compute CM5: use its variational "
                "SCC atomic charges; CM5 is reserved for higher-level QM backends"
            )
        if self.nn_enabled:
            if retain_orbitals or retain_local_projectors:
                raise ValueError(
                    "NN-xTB orbital/projector retention is not supported by the "
                    "numerical-derivative route"
                )
            if gradient_step_bohr is not None and (
                not np.isfinite(gradient_step_bohr) or gradient_step_bohr <= 0.0
            ):
                raise ValueError("NN-xTB gradient step must be positive and finite")
            coordinates_nn = np.asarray(coordinates_angstrom, dtype=float)
            if coordinates_nn.shape != (len(self.atomic_numbers), 3) or not np.all(np.isfinite(coordinates_nn)):
                raise ValueError("NN-xTB coordinates must be finite natoms x 3")
            key = (
                np.ascontiguousarray(coordinates_nn).tobytes(),
                bool(compute_bond_orders),
                None if gradient_step_bohr is None else float(gradient_step_bohr),
            )
            if key == self._last_result_key and self._last_result is not None:
                return self._last_result
            worker = self._batch_worker_clone()
            worker.scf_energy_tolerance_hartree = min(
                worker.scf_energy_tolerance_hartree, 1.0e-12,
            )
            worker.scf_wavefunction_tolerance_electron = min(
                worker.scf_wavefunction_tolerance_electron, 1.0e-10,
            )
            worker.scf_mixer_tolerance = min(
                worker.scf_mixer_tolerance, 1.0e-8,
            )
            worker._nn_development_overlay = self.nn_overlay
            properties = worker._single_point(
                coordinates_nn * ANGSTROM_TO_BOHR,
                compute_bond_orders=compute_bond_orders,
                collect_telemetry=False,
                nn_properties_only=True,
            )
            if gradient_step_bohr is None:
                energy, gradient = self.evaluate_nn_gradient_parameter_response(
                    coordinates_nn,
                )
            else:
                energy, gradient = self.evaluate_nn_derivatives_development(
                    coordinates_nn, self.nn_overlay, order=1,
                    step_bohr=float(gradient_step_bohr),
                )
            result = replace(
                properties,
                energy_hartree=energy,
                gradient_hartree_per_bohr=gradient,
            )
            self._last_result_key = key
            self._last_result = result
            return result
        coordinates = np.asarray(coordinates_angstrom, dtype=float) * ANGSTROM_TO_BOHR
        result_key = (
            np.ascontiguousarray(coordinates).tobytes(),
            self.device or os.environ.get("MATRIX_GFN2_DEVICE", "cpu"),
            self.mps_min_orbitals or os.environ.get("MATRIX_GFN2_MPS_MIN_ORBITALS", ""),
            os.environ.get("MATRIX_GFN2_MPS_REFINE_OCCUPIED", "auto"),
            os.environ.get("MATRIX_GFN2_MPS_SELECT_DENSITY", "1"),
            os.environ.get("MATRIX_GFN2_MPS_TRANSPORT_ORBITALS", "1"),
            os.environ.get("MATRIX_GFN2_MPS_REUSE_BLOCKS", "1"),
            bool(compute_cm5),
            bool(compute_bond_orders),
            bool(retain_orbitals),
            bool(retain_local_projectors),
            bool(collect_telemetry),
        )
        # Geometry optimizers and property pipelines occasionally request the
        # same point twice (for example after a rejected line-search step).
        # Reusing the certified result avoids a complete SCF/gradient rebuild;
        # the result object is immutable by contract and all calculator state
        # is unchanged.
        if (
            gradient_step_bohr is None
            and self.symmetry_contract is None
            and not len(self.point_charges_electron)
            and result_key == self._last_result_key
        ):
            if self._last_result is None:  # defensive invariant guard
                raise RuntimeError("MATRIX-GFN2 result cache is inconsistent")
            return self._last_result
        sparse_requested = self.solver_policy.normalized_mode == "sparse-sp2"
        routing_domains = None
        sparse_automatic = False
        memory_forced_sparse = False
        dense_estimate_bytes = None
        sparse_estimate_bytes = None
        dense_memory_budget_bytes = None
        enclosing_ellipsoid = None
        sparse_shape_eligible = False
        if (
            self.solver_policy.normalized_mode == "auto"
            and self._basis_template.norbitals
            >= self.solver_policy.sparse_sp2_min_orbitals
        ):
            enclosing_ellipsoid = enclosing_pca_ellipsoid(coordinates)
            sparse_shape_eligible = enclosing_ellipsoid.aspect_ratio >= 2.0
            routing_domains = self._localized_domains(
                coordinates,
                self._basis_template.spherical_atom,
                LocalityPolicy(
                    interaction_cutoff_bohr=20.0,
                    buffer_hops=1,
                    truncation_threshold=(
                        self.solver_policy.sparse_truncation_threshold
                    ),
                ),
            )
            dense_estimate_bytes, sparse_estimate_bytes = _integral_route_memory_bytes(
                self._basis_template.norbitals, len(routing_domains.allowed_pairs)
            )
            dense_memory_budget_bytes = _dense_route_memory_budget_bytes()
            memory_forced_sparse = (
                dense_estimate_bytes > dense_memory_budget_bytes
                and sparse_estimate_bytes < dense_estimate_bytes
                and sparse_shape_eligible
            )
            sparse_automatic = (
                sparse_shape_eligible
                and (routing_domains.fill_fraction
                     <= self.solver_policy.sparse_sp2_maximum_fill_fraction
                     or memory_forced_sparse)
            )
        self._last_sparse_routing = {
            "orbitals": self._basis_template.norbitals,
            "enclosing_ellipsoid_axes_bohr": (
                None if enclosing_ellipsoid is None
                else tuple(float(axis) for axis in enclosing_ellipsoid.axes_bohr)
            ),
            "enclosing_ellipsoid_aspect_ratio": (
                None if enclosing_ellipsoid is None
                else enclosing_ellipsoid.aspect_ratio
            ),
            "sparse_shape_eligible": sparse_shape_eligible,
            "fill_fraction": (
                None if routing_domains is None else routing_domains.fill_fraction
            ),
            "selected": bool(sparse_automatic or sparse_requested),
            "memory_forced": memory_forced_sparse,
            "dense_estimate_bytes": dense_estimate_bytes,
            "sparse_estimate_bytes": sparse_estimate_bytes,
            "dense_memory_budget_bytes": dense_memory_budget_bytes,
        }
        sparse_eligible = (
            (sparse_requested or sparse_automatic)
            and self.symmetry_contract is None
            and not len(self.point_charges_electron)
            and self.alpb_parameters is None
            and not compute_cm5
            and self.multiplicity == 1
            and not retain_orbitals
        )
        if sparse_requested and not sparse_eligible:
            raise ValueError(
                "end-to-end sparse-sp2 requires a closed-shell calculation "
                "without frozen symmetry, point-charge embedding, ALPB, CM5 analysis, "
                "or canonical-orbital retention"
            )
        if (
            self.solver_policy.normalized_mode == "auto"
            and dense_estimate_bytes is not None
            and dense_memory_budget_bytes is not None
            and dense_estimate_bytes > dense_memory_budget_bytes
            and not sparse_eligible
        ):
            raise MemoryError(
                "dense GFN2 exceeds the configured memory budget, while the "
                "sparse projector is not certified for this geometry"
            )
        if (memory_forced_sparse and sparse_eligible
                and sparse_estimate_bytes > _available_memory_budget_bytes()):
            raise MemoryError("neither dense nor sparse GFN2 working set fits memory")
        if sparse_eligible:
            sparse_options = {"domains": routing_domains}
            if not compute_bond_orders:
                sparse_options["compute_bond_orders"] = False
            if retain_local_projectors:
                sparse_options["retain_projectors"] = True
            try:
                result = self._single_point_sparse(coordinates, **sparse_options)
            except (RuntimeError, FloatingPointError) as error:
                if (
                    not sparse_automatic
                    or "sparse" not in str(error)
                    or not (
                        "density did not certify" in str(error)
                        or "sparse SP2 became non-finite" in str(error)
                    )
                    or dense_estimate_bytes is None
                    or dense_memory_budget_bytes is None
                    or dense_estimate_bytes > dense_memory_budget_bytes
                ):
                    raise
                from .fragment_guess import covalent_components

                if len(covalent_components(
                    self.atomic_numbers, coordinates, _covalent_radii()
                )) != 1:
                    raise
                self._last_sparse_routing.update({
                    "selected": False,
                    "sparse_attempted": True,
                    "fallback": "dense_core_diagonalization",
                    "sparse_failure": str(error),
                })
                sparse_eligible = False
        if not sparse_eligible:
            density_policy = self.solver_policy
            if (
                self.solver_policy.normalized_mode == "auto"
                and self.symmetry_contract is None
            ):
                density_policy = replace(
                    self.solver_policy, mode="diagonalization"
                )
            if compute_bond_orders:
                single_point_options = {
                    "compute_cm5": compute_cm5,
                    "density_policy": density_policy,
                }
                if not collect_telemetry:
                    single_point_options["collect_telemetry"] = False
                if retain_orbitals:
                    single_point_options["retain_orbitals"] = True
                result = self._single_point(coordinates, **single_point_options)
            else:
                single_point_options = {
                    "compute_cm5": compute_cm5,
                    "compute_bond_orders": False,
                    "density_policy": density_policy,
                }
                if not collect_telemetry:
                    single_point_options["collect_telemetry"] = False
                if retain_orbitals:
                    single_point_options["retain_orbitals"] = True
                result = self._single_point(coordinates, **single_point_options)
        if not result.converged:
            raise RuntimeError("MATRIX-GFN2 ISCF did not converge")
        if gradient_step_bohr is not None:
            raise ValueError("gradient_step_bohr is obsolete: the gradient is analytic")
        self._last_result_key = result_key
        self._last_result = result
        return result

    def evaluate_independent(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        compute_cm5: bool = False,
        compute_bond_orders: bool = True,
        retain_orbitals: bool = False,
        collect_telemetry: bool = True,
    ) -> GFN2Result:
        """Evaluate a point without inheriting an earlier geometry's SCF state.

        Normal-mode derivative stencils amplify small restart-dependent SCC
        residuals. A fresh worker shares immutable model tables but has no
        density predictor, result cache, or mutable native integral context.
        """

        worker = self._batch_worker_clone()
        return worker.evaluate(
            coordinates_angstrom,
            compute_cm5=compute_cm5,
            compute_bond_orders=compute_bond_orders,
            retain_orbitals=retain_orbitals,
            collect_telemetry=collect_telemetry,
        )

    def evaluate_independent_hessian(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        workers: int = 1,
    ):
        """Evaluate a Hessian with the same independent SCF state.

        Both plain GFN2 and NN-xTB use their complete analytic route.
        """

        if workers < 1:
            raise ValueError("analytic Hessian worker count must be positive")
        if self.nn_enabled and workers != 1:
            raise ValueError("parallel analytic Hessian is not yet available with NN")
        if self.nn_enabled:
            worker = self._batch_worker_clone()
            result = worker.evaluate(
                coordinates_angstrom, compute_cm5=False,
                compute_bond_orders=False,
            )
            hessian = worker.evaluate_nn_hessian_analytic(coordinates_angstrom)
            return result, hessian
        worker = self._batch_worker_clone()
        result = worker.evaluate(
            coordinates_angstrom,
            compute_cm5=False,
            compute_bond_orders=False,
            retain_orbitals=True,
        )
        if result.orbital_state is None:
            raise RuntimeError("independent GFN2 Hessian requires retained orbitals")
        if workers > 1:
            return result, worker.ground_state_hessian_matrix_free(
                result.orbital_state, workers=workers,
            )
        return result, worker.ground_state_hessian(result.orbital_state)

    def evaluate_batch(
        self,
        coordinates_batch_angstrom: np.ndarray,
        *,
        workers: int | None = None,
        compute_cm5: bool = False,
        retain_electronic_state: bool = True,
    ) -> tuple[GFN2Result | GFN2EnergyGradientResult, ...]:
        """Evaluate independent structures concurrently, including on one GPU."""

        if compute_cm5:
            raise ValueError(
                "MATRIX-GFN2 does not compute CM5: use its variational "
                "SCC atomic charges; CM5 is reserved for higher-level QM backends"
            )

        batch = np.asarray(coordinates_batch_angstrom, dtype=float)
        if batch.ndim != 3 or batch.shape[1:] != (len(self.atoms), 3):
            raise ValueError("coordinate batch must have shape nstructures x natoms x 3")
        if not np.all(np.isfinite(batch)):
            raise ValueError("coordinate batch must be finite")
        selected_backend = self.diagonalizer_backend
        if selected_backend is None:
            angular_size = {"s": 1, "p": 3, "d": 5}
            orbital_count = sum(
                angular_size[shell.label[-1]]
                for number in self.atomic_numbers
                for shell in self.parameters.elements[number].shells
            )
            selected_backend = best_diagonalizer_backend(
                (orbital_count, orbital_count)
            ).name
        gpu = selected_backend in {
            "gpu", "cuda", "cupy", "torch-cuda", "torch-mps", "mps"
        }
        count = min(len(batch), 4 if gpu else 1) if workers is None else int(workers)
        if count < 1:
            raise ValueError("batch worker count must be positive")
        if not gpu:
            count = _memory_limited_worker_count(count, len(self.atoms))
        if count == 1 or len(batch) < 2:
            results = tuple(
                (
                    self.evaluate if retain_electronic_state
                    else self.evaluate_independent
                )(
                    coordinates, compute_cm5=compute_cm5,
                    compute_bond_orders=retain_electronic_state,
                )
                for coordinates in batch
            )
            if retain_electronic_state:
                return results
            compact = tuple(_energy_gradient_result(result) for result in results)
            self._last_result_key = None
            self._last_result = None
            self._previous_density = None
            self._older_density = None
            self._previous_orthogonal_density = None
            self._previous_coordinates_bohr = None
            self._older_orthogonal_density = None
            self._older_coordinates_bohr = None
            self._previous_orbital_energies = None
            self._previous_occupations = None
            self._restart_fixed_point_contraction = 0.0
            return compact
        # Two lanes stagger model/gradient allocation without serializing a
        # four-worker batch.  For a two-worker batch one lane is sufficient.
        allocation_gate = threading.BoundedSemaphore(max(1, count // 2))
        worker_calculators = tuple(
            self._batch_worker_clone() for _ in range(len(batch))
        )
        for worker in worker_calculators:
            worker._sparse_allocation_gate = allocation_gate

        def evaluate_one(
            worker: GFN2Calculator, coordinates: np.ndarray
        ) -> GFN2Result | GFN2EnergyGradientResult:
            evaluation_options = {
                "compute_cm5": compute_cm5,
                "compute_bond_orders": retain_electronic_state,
            }
            def finalize(
                result: GFN2Result,
            ) -> GFN2Result | GFN2EnergyGradientResult:
                if retain_electronic_state:
                    return result
                compact = _energy_gradient_result(result)
                # A lightweight batch must not leave the full density and lazy
                # observables reachable through the private per-worker cache.
                worker._last_result_key = None
                worker._last_result = None
                worker._previous_density = None
                worker._older_density = None
                worker._previous_orthogonal_density = None
                worker._previous_coordinates_bohr = None
                worker._older_orthogonal_density = None
                worker._older_coordinates_bohr = None
                worker._previous_orbital_energies = None
                worker._previous_occupations = None
                worker._restart_fixed_point_contraction = 0.0
                return compact

            if selected_backend in {"torch-cuda", "cuda"}:
                import torch

                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    result = worker.evaluate(coordinates, **evaluation_options)
                stream.synchronize()
                return finalize(result)
            if selected_backend == "cupy":
                import cupy as cp

                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    result = worker.evaluate(coordinates, **evaluation_options)
                stream.synchronize()
                return finalize(result)
            if selected_backend in {"torch-mps", "mps"}:
                # MPS has an implicit command queue; synchronizing through a
                # lightweight device query prevents host-side batch workers
                # from racing the allocator without forcing CPU copies.
                import torch

                result = worker.evaluate(coordinates, **evaluation_options)
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                return finalize(result)
            result = worker.evaluate(coordinates, **evaluation_options)
            return finalize(result)

        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = tuple(
                executor.submit(evaluate_one, worker, coordinates)
                for worker, coordinates in zip(
                    worker_calculators, batch, strict=True
                )
            )
            return tuple(future.result() for future in futures)


def _shell_partition(
    basis: MolecularBasis, numbers: Sequence[int], parameters: GFN2ParameterSet,
    *,
    per_atom_elements: Sequence[GFN2ElementParameters] | None = None,
    per_atom_third_scales: Sequence[dict[str, float]] | None = None,
) -> ShellPartition:
    if per_atom_elements is not None and len(per_atom_elements) != len(numbers):
        raise ValueError("per-atom element count differs from atom count")
    if per_atom_third_scales is not None and len(per_atom_third_scales) != len(numbers):
        raise ValueError("per-atom third-order scale count differs from atom count")
    shell_keys: list[tuple[int, str]] = []
    shell_lookup: dict[tuple[int, str], int] = {}
    orbital_shell = []
    for atom, label in zip(basis.spherical_atom, basis.spherical_shell, strict=True):
        key = (atom, label)
        shell_index = shell_lookup.get(key)
        if shell_index is None:
            shell_index = len(shell_keys)
            shell_lookup[key] = shell_index
            shell_keys.append(key)
        orbital_shell.append(shell_index)
    reference = []
    hardness = []
    third = []
    third_scale = {"s": 1.0, "p": 0.5, "d": 0.25}
    for atom, label in shell_keys:
        element = (
            parameters.elements[numbers[atom]]
            if per_atom_elements is None else per_atom_elements[atom]
        )
        shell = next(item for item in element.shells if item.label == label)
        reference.append(shell.reference_occupation)
        hardness.append(element.hubbard_hartree * shell.hubbard_scale)
        learned_scale = (
            1.0 if per_atom_third_scales is None
            else per_atom_third_scales[atom][label]
        )
        third.append(
            element.hubbard_derivative_hartree
            * third_scale[label[-1]] * learned_scale
        )
    return ShellPartition(
        tuple(orbital_shell), tuple(atom for atom, _ in shell_keys),
        tuple(reference), tuple(hardness), tuple(third)
    )


def _analytic_nn_exponent_head_adjoint(
    atomic_numbers, positions_bohr, parameter_state, basis, partition,
    fixed, worker, shell_q, multipoles, d4_charge_potential, head_shape,
) -> np.ndarray:
    """Stationary SCC/Pulay adjoint of every basis-moving Slater exponent.

    This contracts exact STO-nG exponent derivatives with the converged
    density and energy-weighted density.  It never perturbs an exponent or
    re-solves the SCF equations.
    """
    orbital_state = fixed.orbital_state
    if orbital_state is None:
        raise ValueError("NN exponent adjoint requires retained orbitals")
    density = np.asarray(fixed.density)
    occupations = (
        orbital_state.alpha_occupations + orbital_state.beta_occupations
    )
    weighted = energy_weighted_density(
        orbital_state.coefficients, occupations,
        orbital_state.orbital_energies_hartree,
    )
    shell_interaction = shell_interaction_operator(
        positions_bohr, partition, alphaj=parameter_state.globals["alphaj"],
    )
    shell_potential = shell_interaction @ shell_q + (
        np.asarray(partition.third_order_hartree) * shell_q**2
    )
    _, charge_potential, dipole_potential, quadrupole_potential = (
        anisotropic_energy_and_moment_potentials(
            multipoles, positions_bohr,
            parameter_state.anisotropic_parameters(worker._active_coordination),
        )
    )
    size = basis.norbitals
    overlap = orbital_state.overlap
    core_prefactor = extended_hueckel_hamiltonian(
        np.ones_like(overlap), worker._orbital_metadata,
        atomic_numbers, positions_bohr, parameter_state.atoms,
        parameter_state.orbitals, global_scalings=parameter_state.globals,
    )
    orbital_atom = np.asarray(basis.spherical_atom)
    zeta = np.asarray([
        next(shell.slater_exponent for shell in parameter_state.elements[atom].shells
             if shell.label == label)
        for atom, label in zip(basis.spherical_atom, basis.spherical_shell, strict=True)
    ])
    active_h0 = orbital_atom[:, None] != orbital_atom[None, :]
    core_prefactor[~active_h0] = 0.0
    cartesian = basis.cartesian_orbitals
    cartesian_size = len(cartesian)
    transform = basis.cartesian_to_spherical
    shell_keys = tuple(dict.fromkeys(
        (orbital.atom_index, orbital.shell) for orbital in cartesian
    ))
    shell_slot = {
        (atom, shell.label): (atom, slot)
        for atom, element in enumerate(parameter_state.elements)
        for slot, shell in enumerate(element.shells)
    }
    adjoint = np.zeros(head_shape)
    rows, columns = triangular_indices(size)
    zero_derivatives = np.zeros((len(rows), 13, 3))
    for atom, label in shell_keys:
        if f"atom {atom} shell {shell_slot[(atom, label)][1]}: exp" in parameter_state.floor_events:
            continue
        cartesian_derivative = np.zeros((13, cartesian_size, cartesian_size))
        for bra_index, bra in enumerate(cartesian):
            for ket_index in range(bra_index + 1):
                ket = cartesian[ket_index]
                if ((bra.atom_index, bra.shell) != (atom, label)
                        and (ket.atom_index, ket.shell) != (atom, label)):
                    continue
                _, bra_derivative, ket_derivative = (
                    _nn_exponent_integral_response(bra, ket, "first")
                )
                derivative = np.zeros(13)
                if (bra.atom_index, bra.shell) == (atom, label):
                    derivative += bra_derivative
                if (ket.atom_index, ket.shell) == (atom, label):
                    derivative += ket_derivative
                cartesian_derivative[:, bra_index, ket_index] = derivative
                cartesian_derivative[:, ket_index, bra_index] = derivative
        spherical = np.asarray([
            transform.T @ component @ transform
            for component in cartesian_derivative
        ])
        ds = spherical[0]
        dh0 = core_prefactor * ds
        for left in range(size):
            for right in range(left):
                if not active_h0[left, right]:
                    continue
                ln_factor = 0.0
                for endpoint in (left, right):
                    if (orbital_atom[endpoint], basis.spherical_shell[endpoint]) == (atom, label):
                        ln_factor += (
                            0.25 / zeta[endpoint]
                            - 0.5 / (zeta[left] + zeta[right])
                        ) * (2.0 * parameter_state.globals.get("wexp", 0.5))
                if ln_factor:
                    value = core_prefactor[left, right] * overlap[left, right] * ln_factor
                    dh0[left, right] += value
                    dh0[right, left] += value
        derivative_integrals = SparseAtomicMultipoleIntegrals(
            size, rows, columns, ds[rows, columns],
            spherical[1:4, rows, columns].T,
            spherical[4:, rows, columns].T.reshape(-1, 3, 3),
            zero_derivatives, zero_derivatives,
        )
        derivative_operators = prepare_pair_cumulative_multipole_operators(
            derivative_integrals, positions_bohr, basis.spherical_atom,
        )
        shell_derivative, moment_derivative = dense_scc_observables(
            density, derivative_operators, partition.orbital_shell,
            np.zeros(partition.nshells), np.zeros(len(atomic_numbers)),
        )
        value = float(np.sum(density * dh0) - np.sum(weighted * ds))
        value += float(shell_potential @ shell_derivative)
        value += float((charge_potential + d4_charge_potential) @ moment_derivative.charges)
        value += float(np.sum(dipole_potential * moment_derivative.dipoles_bohr))
        value += float(np.sum(quadrupole_potential * moment_derivative.quadrupoles_bohr2))
        adjoint[shell_slot[(atom, label)]] = value
    return adjoint


def _analytic_gradient(
    *,
    density: np.ndarray,
    coefficients: np.ndarray,
    occupations: np.ndarray,
    orbital_energies: np.ndarray,
    basis: MolecularBasis,
    overlap: np.ndarray,
    dipole: np.ndarray,
    second: np.ndarray,
    coordinates: np.ndarray,
    atomic_numbers: Sequence[int],
    atom_parameters: dict[int, AtomParameters],
    orbital_parameters: dict[tuple[int, str], OrbitalParameters],
    partition: ShellPartition,
    shell_interaction: np.ndarray,
    shell_q: np.ndarray,
    multipoles,
    anisotropic_parameters: AnisotropicParameters,
    anisotropic_geometry,
    d4,
    d4_cn: np.ndarray,
    d4_parameters: D4Parameters,
    neutral_d4,
    point_charge_coordinates: np.ndarray,
    point_charges: np.ndarray,
    valence: Sequence[float],
    weighted_density_override: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the stationary analytic GFN2 nuclear gradient."""

    del d4_cn  # its value is already represented by d4 and its Jacobian below
    cart_ds, cart_ddipole, cart_dsecond = atomic_multipole_integral_derivatives(
        basis.cartesian_orbitals
    )
    transform = basis.cartesian_to_spherical
    ds = transform.T @ cart_ds @ transform
    ddipole = transform.T @ cart_ddipole @ transform
    dsecond = transform.T @ cart_dsecond @ transform
    cn_jacobian = coordination_number_jacobian(
        atomic_numbers, coordinates, atom_parameters
    )
    metadata = tuple(
        _OrbitalMetadata(atom, shell, slater)
        for atom, shell, slater in zip(
            basis.spherical_atom,
            basis.spherical_shell,
            (
                next(
                    item.slater_exponent
                    for item in load_gfn2_parameters().elements[atomic_numbers[atom]].shells
                    if item.label == shell
                )
                for atom, shell in zip(
                    basis.spherical_atom, basis.spherical_shell, strict=True
                )
            ),
            strict=True,
        )
    )
    dh0 = extended_hueckel_hamiltonian_derivative(
        overlap,
        ds,
        metadata,
        atomic_numbers,
        coordinates,
        atom_parameters,
        orbital_parameters,
        cn_jacobian,
    )
    embedding = _point_charge_embedding(
        overlap,
        ds,
        basis.spherical_atom,
        coordinates,
        point_charge_coordinates,
        point_charges,
    )
    if embedding.hamiltonian_derivative_qm is not None:
        dh0 += embedding.hamiltonian_derivative_qm
    dq, dmu, dquadrupole = atomic_multipole_derivatives(
        density,
        overlap,
        ds,
        dipole,
        ddipole,
        second,
        dsecond,
        coordinates,
        basis.spherical_atom,
    )
    _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
        multipoles, coordinates, anisotropic_parameters,
        geometry=anisotropic_geometry,
    )
    moment_gradient = np.einsum("a,axk->xk", qpot, dq, optimize=False)
    moment_gradient += np.einsum("ac,acxk->xk", mupot, dmu, optimize=False)
    moment_gradient += np.einsum(
        "abc,abcxk->xk", thetapot, dquadrupole, optimize=False
    )
    base_radii = np.asarray(
        [load_gfn2_parameters().elements[number].multipole_radius_bohr for number in atomic_numbers]
    )
    valence_cn = np.asarray(
        [
            load_gfn2_parameters().elements[number].multipole_valence_coordination
            for number in atomic_numbers
        ]
    )
    coordination = coordination_numbers(atomic_numbers, coordinates, atom_parameters)
    exponential = np.exp(-4.0 * (coordination - valence_cn - 1.2))
    radius_cn_derivative = 4.0 * (5.0 - base_radii) * exponential / (1.0 + exponential) ** 2
    radius_jacobian = radius_cn_derivative[:, None, None] * cn_jacobian
    moment_gradient += anisotropic_explicit_gradient(
        multipoles, coordinates, anisotropic_parameters, radius_jacobian
    )

    shell_dq = shell_charge_derivatives(density, ds, partition)
    interaction = shell_interaction
    shell_potential = interaction @ shell_q + np.asarray(
        partition.third_order_hartree
    ) * shell_q**2
    explicit = frozen_charge_scc_gradient(
        coordinates, shell_q, partition, interaction=interaction
    )
    explicit += np.einsum("s,sxk->xk", shell_potential, shell_dq, optimize=False)
    explicit += moment_gradient

    en = np.asarray(
        [load_gfn2_parameters().elements[number].electronegativity for number in atomic_numbers]
    )
    d4_cn_jacobian = d4_coordination_number_jacobian(
        atomic_numbers, coordinates, _covalent_radii(), en
    )
    if hasattr(d4, "rows"):
        explicit += frozen_d4_two_body_gradient_sparse(
            coordinates, d4,
            s6=d4_parameters.s6, s8=d4_parameters.s8, s10=d4_parameters.s10,
            a1=d4_parameters.a1, a2=d4_parameters.a2,
        )
    else:
        explicit += frozen_d4_two_body_gradient(
            coordinates,
            d4.c6,
            d4.c8,
            s6=d4_parameters.s6,
            s8=d4_parameters.s8, s10=d4_parameters.s10,
            a1=d4_parameters.a1,
            a2=d4_parameters.a2,
        )
    explicit += np.einsum(
        "a,axk->xk", d4.coordination_derivative_hartree, d4_cn_jacobian, optimize=False
    )
    explicit += np.einsum(
        "a,axk->xk", d4.charge_derivative_hartree, dq, optimize=False
    )
    if neutral_d4 is not None and isinstance(neutral_d4, tuple):
        neutral_pairs, atm = neutral_d4
        explicit += atm_three_body_gradient_sparse(
            coordinates, neutral_pairs, atm, None,
            s9=d4_parameters.s9, a1=d4_parameters.a1,
            a2=d4_parameters.a2,
            coordination_radii_bohr=_covalent_radii()[
                np.asarray(atomic_numbers) - 1
            ],
            electronegativities=en,
        )
    elif neutral_d4 is not None:
        explicit += atm_three_body_gradient(
            coordinates,
            neutral_d4.c6,
            neutral_d4.c8,
            neutral_d4.c6_coordination_derivative,
            d4_cn_jacobian,
            s9=d4_parameters.s9,
            a1=d4_parameters.a1,
            a2=d4_parameters.a2,
        )
    explicit += repulsion_gradient(atomic_numbers, coordinates, atom_parameters)
    explicit += np.asarray(valence)[:, None] * embedding.atomic_potential_derivative_qm
    weighted = (
        energy_weighted_density(coefficients, occupations, orbital_energies)
        if weighted_density_override is None
        else np.asarray(weighted_density_override, dtype=float)
    )
    gradient = assemble_variational_gradient(
        density,
        weighted,
        VariationalGradientTerms(dh0, ds, (explicit,)),
    )
    point_gradient = np.einsum(
        "pkij,ij->pk",
        embedding.hamiltonian_derivative_point_charges,
        density,
        optimize=False,
    )
    point_gradient += np.einsum(
        "a,pak->pk",
        np.asarray(valence),
        embedding.atomic_potential_derivative_point_charges,
        optimize=False,
    )
    return gradient, point_gradient


def _analytic_gradient_pair_packed(
    *, density, weighted_density, basis, coordinates, atomic_numbers,
    atom_parameters, orbital_parameters, partition, shell_interaction,
    shell_q, multipoles, anisotropic_parameters, anisotropic_geometry,
    d4, d4_parameters, neutral_d4, valence, integrals=None, spherical_domains=None,
    element_values=None, base_multipole_radii=None, valence_coordination=None,
    electronegativities=None, orbital_metadata=None, coordination_override=None,
    nn_global_scalings=None,
    native_context=None,
    repulsion_gradient_override=None,
    atm_gradient_override=None,
    post_scf_explicit_gradient_override=None,
    anisotropic_potentials_override=None,
) -> np.ndarray:
    """Stationary gradient with no dense AO derivative tensor."""

    if integrals is None or spherical_domains is None:
        integral_policy = LocalityPolicy(interaction_cutoff_bohr=20.0, buffer_hops=0)
        spherical_domains = build_localized_domains(
            coordinates, basis.spherical_atom, integral_policy
        )
        cartesian_domains = build_localized_domains(
            coordinates,
            tuple(item.atom_index for item in basis.cartesian_orbitals),
            integral_policy,
        )
        cartesian = sparse_atomic_multipole_integrals(
            basis.cartesian_orbitals,
            *cartesian_domains.allowed_pair_arrays(),
            native_threads=spherical_domains.native_threads,
        )
        integrals = basis.transform_sparse_multipoles(
            cartesian, spherical_domains, threshold=0.0
        )
    use_dense_native_gradient = bool(
        native_context is not None
        and hasattr(native_context, "variational_gradient_dense")
        and not isinstance(density, SparseSymmetricMatrix)
        and not isinstance(weighted_density, SparseSymmetricMatrix)
        and partition.electrostatic_profile is None
    )
    density_sparse = None if use_dense_native_gradient else (
        density if isinstance(density, SparseSymmetricMatrix)
        else SparseSymmetricMatrix.from_dense(density, spherical_domains, threshold=0.0)
    )
    weighted_sparse = None if use_dense_native_gradient else (
        weighted_density
        if isinstance(weighted_density, SparseSymmetricMatrix)
        else SparseSymmetricMatrix.from_dense(
            weighted_density, spherical_domains, threshold=0.0
        )
    )
    if nn_global_scalings is None:
        atomic_radii = np.asarray(
            [atom_parameters[number].covalent_radius_bohr for number in atomic_numbers]
        )
    else:
        atomic_radii = np.asarray(
            [value.covalent_radius_bohr for value in atom_parameters]
        )
    if coordination_override is None:
        coordination, _ = gfn2_coordination_sparse(
            coordinates, atomic_radii, cutoff_bohr=24.0
        )
    else:
        coordination = np.asarray(coordination_override, dtype=float)
    if element_values is None:
        parameter_elements = load_gfn2_parameters().elements
        element_values = tuple(parameter_elements[number] for number in atomic_numbers)
    metadata = (
        tuple(orbital_metadata)
        if orbital_metadata is not None
        else tuple(
            _OrbitalMetadata(
                atom,
                shell,
                next(
                    item.slater_exponent
                    for item in element_values[atom].shells
                    if item.label == shell
                ),
            )
            for atom, shell in zip(
                basis.spherical_atom, basis.spherical_shell, strict=True
            )
        )
    )
    if native_context is not None and hasattr(native_context, "h0_derivatives"):
        h0 = SparseHamiltonianDerivatives(
            *native_context.h0_derivatives(
                integrals.rows,
                integrals.columns,
                integrals.overlap,
                integrals.left_derivatives[:, 0],
                integrals.right_derivatives[:, 0],
                coordinates,
                coordination,
            )
        )
    else:
        h0 = extended_hueckel_hamiltonian_derivatives_sparse(
            integrals, metadata, atomic_numbers, coordinates, atom_parameters,
            orbital_parameters, coordination,
            global_scalings=nn_global_scalings,
        )
    atom_index = np.asarray(basis.spherical_atom, dtype=np.intp)
    if anisotropic_potentials_override is None:
        _, qpot, mupot, thetapot = anisotropic_energy_and_moment_potentials(
            multipoles, coordinates, anisotropic_parameters,
            geometry=anisotropic_geometry,
        )
    else:
        qpot, mupot, thetapot = anisotropic_potentials_override
    camm_left, camm_right = anisotropic_fock_derivatives_sparse(
        integrals,
        None,
        coordinates,
        qpot,
        mupot,
        thetapot,
        orbital_atom=atom_index,
    )
    shell_potential = shell_interaction @ shell_q + np.asarray(
        partition.third_order_hartree
    ) * shell_q**2
    shell_index = np.asarray(partition.orbital_shell, dtype=np.intp)
    orbital_scalar = shell_potential[shell_index] + d4.charge_derivative_hartree[atom_index]
    pair_scalar = -0.5 * (
        orbital_scalar[integrals.rows] + orbital_scalar[integrals.columns]
    )
    # The persistent native contraction now reuses the pair pattern and avoids
    # the separate Python gather, ``add.at`` and variational-assembly passes.
    # This also lowers the fixed cost at one thread for small resident systems.
    use_fused_gradient = (
        native_context is not None
        and hasattr(native_context, "variational_gradient_sparse")
    )
    if use_fused_gradient:
        gradient_method = (
            native_context.variational_gradient_dense
            if use_dense_native_gradient
            else native_context.variational_gradient_sparse
        )
        electronic_gradient, cn_coefficient = gradient_method(
            density if use_dense_native_gradient else density_sparse,
            weighted_density if use_dense_native_gradient else weighted_sparse,
            integrals.rows,
            integrals.columns,
            atom_index[integrals.rows],
            atom_index[integrals.columns],
            h0.left,
            h0.right,
            camm_left,
            camm_right,
            pair_scalar,
            integrals.left_derivatives[:, 0],
            integrals.right_derivatives[:, 0],
            h0.coordination_left,
            h0.coordination_right,
            len(coordinates),
            spherical_domains.native_threads,
        )
        electronic_gradient = np.asarray(electronic_gradient, dtype=float)
        cn_coefficient = np.asarray(cn_coefficient, dtype=float)
    else:
        dh_left = h0.left
        dh_right = h0.right
        dh_left += camm_left
        dh_right += camm_right
        dh_left += pair_scalar[:, None] * integrals.left_derivatives[:, 0]
        dh_right += pair_scalar[:, None] * integrals.right_derivatives[:, 0]
        p = density_sparse.values_on(integrals.rows, integrals.columns)
        multiplicity = np.where(integrals.rows == integrals.columns, 1.0, 2.0)
        cn_coefficient = np.zeros(len(coordinates))
        np.add.at(
            cn_coefficient, atom_index[integrals.rows],
            multiplicity * p * h0.coordination_left,
        )
        np.add.at(
            cn_coefficient, atom_index[integrals.columns],
            multiplicity * p * h0.coordination_right,
        )
    explicit = coordination_response_gradient_sparse(
        cn_coefficient, coordinates, atomic_radii
    )
    base_radii = np.asarray(
        [element.multipole_radius_bohr for element in element_values]
        if base_multipole_radii is None else base_multipole_radii
    )
    valence_cn = np.asarray(
        [element.multipole_valence_coordination for element in element_values]
        if valence_coordination is None else valence_coordination
    )
    aes_exponent = 4.0 if nn_global_scalings is None else nn_global_scalings["aesexp"]
    aes_shift = 1.2 if nn_global_scalings is None else nn_global_scalings["aesshift"]
    aes_max_radius = 5.0 if nn_global_scalings is None else nn_global_scalings["aesrmax"]
    exponential = np.exp(-aes_exponent * (coordination - valence_cn - aes_shift))
    radius_derivative = (
        aes_exponent * (aes_max_radius - base_radii)
        * exponential / (1.0 + exponential) ** 2
    )
    en = np.asarray(
        [element.electronegativity for element in element_values]
        if electronegativities is None else electronegativities
    )
    d4_radii = _covalent_radii()[np.asarray(atomic_numbers) - 1]
    if post_scf_explicit_gradient_override is not None:
        explicit += np.asarray(post_scf_explicit_gradient_override, dtype=float)
    else:
        explicit += frozen_charge_scc_gradient(
            coordinates, shell_q, partition, interaction=shell_interaction,
            alphaj=(2.0 if nn_global_scalings is None else nn_global_scalings["alphaj"]),
        )
        explicit += anisotropic_explicit_gradient(
            multipoles, coordinates, anisotropic_parameters, None,
            geometry=anisotropic_geometry,
            coordination_radii_bohr=atomic_radii,
            radius_cn_derivative=radius_derivative,
        )
        if hasattr(d4, "rows"):
            explicit += frozen_d4_two_body_gradient_sparse(
                coordinates, d4,
                s6=d4_parameters.s6, s8=d4_parameters.s8, s10=d4_parameters.s10,
                a1=d4_parameters.a1, a2=d4_parameters.a2,
            )
        else:
            explicit += frozen_d4_two_body_gradient(
                coordinates, d4.c6, d4.c8,
                s6=d4_parameters.s6, s8=d4_parameters.s8, s10=d4_parameters.s10,
                a1=d4_parameters.a1, a2=d4_parameters.a2,
            )
        explicit += coordination_response_gradient_sparse(
            d4.coordination_derivative_hartree,
            coordinates, d4_radii, electronegativities=en,
        )
    if atm_gradient_override is not None:
        explicit += np.asarray(atm_gradient_override, dtype=float)
    elif neutral_d4 is not None and isinstance(neutral_d4, tuple):
        neutral_pairs, atm = neutral_d4
        explicit += atm_three_body_gradient_sparse(
            coordinates, neutral_pairs, atm, None,
            s9=d4_parameters.s9, a1=d4_parameters.a1, a2=d4_parameters.a2,
            coordination_radii_bohr=d4_radii,
            electronegativities=en,
        )
    elif neutral_d4 is not None:
        d4_cn_jacobian = d4_coordination_number_jacobian(
            atomic_numbers, coordinates, _covalent_radii(), en
        )
        explicit += atm_three_body_gradient(
            coordinates, neutral_d4.c6, neutral_d4.c8,
            neutral_d4.c6_coordination_derivative, d4_cn_jacobian,
            s9=d4_parameters.s9, a1=d4_parameters.a1, a2=d4_parameters.a2,
        )
    explicit += (
        repulsion_gradient_sparse(
            atomic_numbers, coordinates, atom_parameters, cutoff_bohr=24.0,
            kexp=(1.5 if nn_global_scalings is None else nn_global_scalings["kexp"]),
            klight=(1.0 if nn_global_scalings is None else nn_global_scalings["klight"]),
        )
        if repulsion_gradient_override is None
        else np.asarray(repulsion_gradient_override, dtype=float)
    )
    if use_fused_gradient:
        return electronic_gradient + explicit
    return assemble_variational_gradient_sparse(
        density_sparse, weighted_sparse,
        integrals.rows, integrals.columns,
        atom_index[integrals.rows], atom_index[integrals.columns],
        dh_left, dh_right,
        integrals.left_derivatives[:, 0], integrals.right_derivatives[:, 0],
        explicit,
        native_threads=spherical_domains.native_threads,
    )


def _diis_fock(
    focks: Sequence[np.ndarray], errors: Sequence[np.ndarray]
) -> np.ndarray:
    """Constrained Pulay extrapolation with bounded affine coefficients."""

    if len(focks) < 2:
        return focks[-1]
    count = len(focks)
    system = np.empty((count + 1, count + 1))
    system[:count, :count] = np.asarray(
        [[float(left @ right) for right in errors] for left in errors]
    )
    scale = max(float(np.max(np.abs(system[:count, :count]))), 1.0)
    system[:count, :count] += np.eye(count) * scale * 1.0e-12
    system[count, :count] = -1.0
    system[:count, count] = -1.0
    system[count, count] = 0.0
    rhs = np.zeros(count + 1)
    rhs[count] = -1.0
    try:
        coefficients = np.linalg.solve(system, rhs)[:count]
    except np.linalg.LinAlgError:
        return focks[-1]
    if (
        not np.all(np.isfinite(coefficients))
        or abs(float(np.sum(coefficients)) - 1.0) > 1.0e-8
        or np.max(np.abs(coefficients)) > 8.0
        or np.sum(np.abs(coefficients)) > 16.0
    ):
        return focks[-1]
    return np.sum(
        np.asarray([coefficient * fock for coefficient, fock in zip(coefficients, focks, strict=True)]),
        axis=0,
    )


def _sparse_antisymmetric_residual(error, domains) -> SparseSymmetricMatrix:
    """Pack one triangle of an antisymmetric SCF residual for DIIS metrics."""

    values = np.asarray(error, dtype=float)
    rows, columns = domains.allowed_pair_arrays()
    off_diagonal = rows != columns
    return SparseSymmetricMatrix(
        values.shape[0],
        np.asarray(rows[off_diagonal], dtype=np.intp),
        np.asarray(columns[off_diagonal], dtype=np.intp),
        np.asarray(values[rows[off_diagonal], columns[off_diagonal]], dtype=float),
    )


def _annealed_scf_temperature(
    target_kelvin: float,
    iteration: int,
    frontier_gap_hartree: float,
) -> float:
    """Return a short electronic-temperature homotopy ending at the target.

    The elevated temperatures are an SCF aid only.  Every calculation spends
    all iterations from cycle 13 onward at the user-requested temperature, so
    the converged density and reported entropy belong to the requested model.
    """

    target = float(target_kelvin)
    if target < 0.0 or not np.isfinite(target):
        raise ValueError("electronic temperature must be finite and nonnegative")
    cycle = int(iteration)
    if cycle < 1:
        raise ValueError("SCF iteration must be positive")
    gap = float(frontier_gap_hartree)
    if np.isnan(gap) or gap < 0.0:
        raise ValueError("frontier gap must be nonnegative")
    # A gap above 0.01 Eh (~0.27 eV) makes 300 K fractionality negligible.
    # Use the idempotent zero-temperature density in that regime.  Only
    # genuinely small-gap systems pay for electronic-temperature homotopy.
    if gap >= 1.0e-2:
        return 0.0
    if cycle <= 4:
        return max(target, 2000.0)
    if cycle <= 8:
        return max(target, 1000.0)
    if cycle <= 12:
        return max(target, 500.0)
    return target


def _retain_best_diis_history(
    focks: list[np.ndarray],
    errors: list[np.ndarray],
    fock: np.ndarray,
    error: np.ndarray,
    *,
    maximum_size: int,
) -> None:
    """Keep the current SCF point and the best older Pulay residuals.

    ``maximum_size`` is the maximum subspace dimension, not a FIFO window.
    At fixed geometry an older Fock with a smaller commutator residual remains
    more useful than a newer, poorer one.  The current point is retained even
    when its residual is the largest so that the subspace stays local to the
    active SCF branch.
    """

    if maximum_size < 2:
        raise ValueError("DIIS history size must be at least two")
    focks.append(np.asarray(fock, dtype=float).copy())
    errors.append(np.asarray(error, dtype=float).ravel().copy())
    if len(focks) != len(errors):
        raise ValueError("DIIS Fock and residual histories differ in length")
    if len(focks) <= maximum_size:
        return
    # The two newest points anchor the selected history to the current SCF
    # branch; the remaining slots are quality-selected across older cycles.
    worst_older = max(
        range(len(errors) - 2),
        key=lambda index: float(errors[index] @ errors[index]),
    )
    del focks[worst_older]
    del errors[worst_older]


def _anderson_density(
    outputs: Sequence[np.ndarray], residuals: Sequence[np.ndarray]
) -> np.ndarray:
    """Return a regularized, affine Anderson combination of density outputs."""

    if len(outputs) < 2:
        return outputs[-1]
    count = len(outputs)
    gram = np.asarray(
        [[float(left @ right) for right in residuals] for left in residuals]
    )
    scale = max(float(np.max(np.abs(gram))), 1.0)
    gram += np.eye(count) * scale * 1.0e-10
    system = np.empty((count + 1, count + 1))
    system[:count, :count] = gram
    system[count, :count] = 1.0
    system[:count, count] = 1.0
    system[count, count] = 0.0
    rhs = np.zeros(count + 1)
    rhs[count] = 1.0
    try:
        coefficients = np.linalg.solve(system, rhs)[:count]
    except np.linalg.LinAlgError:
        return outputs[-1]
    if (
        not np.all(np.isfinite(coefficients))
        or np.max(np.abs(coefficients)) > 4.0
        or np.sum(np.abs(coefficients)) > 8.0
    ):
        return outputs[-1]
    return np.sum(
        np.asarray(
            [
                coefficient * output
                for coefficient, output in zip(coefficients, outputs, strict=True)
            ]
        ),
        axis=0,
    )


def _bond_orders(
    density: np.ndarray,
    overlap: np.ndarray,
    assignment: Sequence[int],
    *,
    spin_densities: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate Mayer bond orders in the nonorthogonal AO basis."""

    natoms = max(assignment) + 1
    result = np.zeros((natoms, natoms))
    atom_projector = np.zeros((len(assignment), natoms), dtype=float)
    atom_projector[np.arange(len(assignment)), np.asarray(assignment)] = 1.0
    channels = (density,) if spin_densities is None else spin_densities
    scale = 1.0 if spin_densities is None else 2.0
    for channel in channels:
        product = channel @ overlap
        pair_contribution = scale * product * product.T
        result += atom_projector.T @ pair_contribution @ atom_projector
    np.fill_diagonal(result, 0.0)
    return 0.5 * (result + result.T)


__all__ = ["GFN2Calculator", "GFN2EnergyGradientResult", "GFN2Result"]
