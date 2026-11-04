"""Rho-axis-method contracts for one threefold internal rotor.

The module maps spectroscopic RAM parameters to the general DVR surface used
by LOCK.  Parameters multiply the operators conventionally printed in BELGI
tables; their signs are therefore translated explicitly to the Watson
operators used by :mod:`matrix_rovib.native_rotor`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import math
from typing import Mapping, Sequence

import numpy as np

try:
    from . import _lock_native
except ImportError:  # pragma: no cover - source checkouts may omit the extension
    _lock_native = None

from .lam_rotation import LAMRotationSurface
from .lam_rotation import _parent_assignments
from .native_rotor import (
    angular_momentum_operators,
    centrifugal_distortion_operators,
)
from .ram36_operators import (
    RAM36OperatorParameter,
    RAM36OperatorTerm,
    ram36_factorized_terms_many,
)


CM1_TO_MHZ = 29_979.2458
LOCK_RAM_ONE_TOP_SCHEMA = "matrix.lock.ram_one_top.v1"


@dataclass(frozen=True)
class RAMOneTopParameters:
    """BELGI-style RAM parameters for one threefold internal rotor, in MHz.

    ``rho`` is dimensionless.  The remaining floating-point fields have MHz
    units and multiply the operators listed in a conventional BELGI parameter
    table.  In particular, ``delta_J`` and ``delta_K`` are the positive or
    negative tabulated constants, not pre-signed native-kernel coefficients.
    """

    A: float
    B: float
    C: float
    F: float
    rho: float
    V3: float
    V6: float = 0.0
    D_ab: float = 0.0
    Delta_J: float = 0.0
    Delta_JK: float = 0.0
    Delta_K: float = 0.0
    delta_J: float = 0.0
    delta_K: float = 0.0
    k5: float = 0.0
    Fv: float = 0.0
    c2: float = 0.0
    Delta_ab: float = 0.0
    Lv: float = 0.0
    Gv: float = 0.0
    k4: float = 0.0
    Nv: float = 0.0
    HJ: float = 0.0
    HJK: float = 0.0
    HKJ: float = 0.0
    HK: float = 0.0
    h1: float = 0.0
    h2: float = 0.0
    h3: float = 0.0
    torsional_symmetry_order: int = 3

    def __post_init__(self) -> None:
        values = asdict(self)
        order = values.pop("torsional_symmetry_order")
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("RAM parameters must be finite")
        if self.F <= 0.0:
            raise ValueError("the RAM internal-rotation constant F must be positive")
        if int(order) != order or order < 1:
            raise ValueError("torsional_symmetry_order must be a positive integer")

    @classmethod
    def from_cm1(cls, **parameters: float) -> "RAMOneTopParameters":
        """Construct parameters from a BELGI table expressed in cm-1."""

        unknown = set(parameters) - set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if unknown:
            raise ValueError(f"unknown RAM parameters: {sorted(unknown)}")
        converted = {
            name: value
            if name in {"rho", "torsional_symmetry_order"}
            else float(value) * CM1_TO_MHZ
            for name, value in parameters.items()
        }
        return cls(**converted)

    def to_dict(self, *, units: str = "MHz") -> dict[str, object]:
        """Return a serializable parameter record in MHz or cm-1."""

        if units not in {"MHz", "cm-1"}:
            raise ValueError("RAM parameter units must be MHz or cm-1")
        values = asdict(self)
        if units == "cm-1":
            values = {
                name: value
                if name in {"rho", "torsional_symmetry_order"}
                else float(value) / CM1_TO_MHZ
                for name, value in values.items()
            }
        return {"schema": LOCK_RAM_ONE_TOP_SCHEMA, "units": units, "parameters": values}


@dataclass(frozen=True)
class RAMKContractedLevel:
    J: int
    index: int
    energy_MHz: float
    lam_symmetry: str
    torsional_state: int
    Ka: int
    Kc: int
    parent_purity: float


@dataclass(frozen=True)
class RAMKContractedBlock:
    J: int
    energies_MHz: np.ndarray
    levels: tuple[RAMKContractedLevel, ...]
    parameter_derivatives_MHz: Mapping[str, np.ndarray] = field(default_factory=dict)
    state_vectors_by_symmetry: Mapping[str, np.ndarray] = field(default_factory=dict)
    state_vector_indices_by_symmetry: Mapping[str, tuple[int, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RAMStateOverlapMapping:
    J: int
    symmetry: str
    reference_indices: tuple[int, ...]
    current_indices: tuple[int, ...]
    squared_overlaps: tuple[float, ...]
    minimum_squared_overlap: float
    mean_squared_overlap: float


@dataclass(frozen=True)
class RAMKContractedResult:
    blocks: Mapping[int, RAMKContractedBlock]
    k_truncation: int
    torsional_states_per_symmetry: int
    levels_per_symmetry: int | Mapping[int, int] | None
    tracking_steps: int
    native_acceleration: bool
    hermiticity_error_MHz: float


def ram_one_top_surface(
    parameters: RAMOneTopParameters,
    *,
    grid_size: int = 96,
    label: str = "internal rotation",
    metadata: Mapping[str, object] | None = None,
) -> LAMRotationSurface:
    """Map a one-top RAM Hamiltonian exactly onto a LOCK DVR surface.

    The implemented Hamiltonian is

    ``F (p_alpha-rho P_a)^2 + A P_a^2 + B P_b^2 + C P_c^2``

    plus the potential, centrifugal-distortion, and higher-order operators
    represented by :class:`RAMOneTopParameters`.
    """

    if grid_size < 3 or grid_size % parameters.torsional_symmetry_order:
        raise ValueError("grid_size must be divisible by the torsional symmetry order")
    q = np.arange(grid_size, dtype=float) * (2.0 * np.pi / grid_size)
    cos3 = 1.0 - np.cos(parameters.torsional_symmetry_order * q)
    cos6 = 1.0 - np.cos(2 * parameters.torsional_symmetry_order * q)
    potential = 0.5 * parameters.V3 * cos3 + 0.5 * parameters.V6 * cos6

    F = np.full((grid_size, 1, 1), parameters.F)
    rho = np.zeros((grid_size, 3, 1))
    rho[:, 0, 0] = parameters.rho

    rotational = np.zeros((grid_size, 3, 3))
    rotational[:, 0, 0] = parameters.A + parameters.k5 * cos3
    rotational[:, 1, 1] = parameters.B + parameters.c2 * cos3
    rotational[:, 2, 2] = parameters.C - parameters.c2 * cos3
    for axis in range(3):
        rotational[:, axis, axis] += parameters.Fv * cos3 + parameters.Nv * cos6
    rotational[:, 0, 1] = rotational[:, 1, 0] = parameters.D_ab

    # Native A-reduction operators 3 and 4 have the opposite sign to the
    # delta_J and delta_K operators printed in BELGI tables.
    quartic = np.asarray(
        [
            -parameters.Delta_J,
            -parameters.Delta_JK,
            -parameters.Delta_K,
            -parameters.delta_J,
            -parameters.delta_K,
        ]
    )
    quartic_surface = np.broadcast_to(quartic, (grid_size, 5)).copy()

    lam_p4 = np.full((grid_size, 1), parameters.k4)
    p2_rotational = np.zeros((grid_size, 1, 3, 3))
    p2_rotational[:, 0, 0, 0] = parameters.Gv
    p2_rotational[:, 0, 1, 1] = parameters.Gv
    p2_rotational[:, 0, 2, 2] = parameters.Gv
    p2_rotational[:, 0, 0, 1] = parameters.Delta_ab
    p2_rotational[:, 0, 1, 0] = parameters.Delta_ab
    p_rotational_cubic = np.zeros((grid_size, 1, 3))
    p_rotational_cubic[:, 0, 0] = parameters.Lv
    sextic = np.broadcast_to(
        np.asarray(
            [
                parameters.HJ,
                parameters.HJK,
                parameters.HKJ,
                parameters.HK,
                parameters.h1,
                parameters.h2,
                parameters.h3,
            ]
        ),
        (grid_size, 7),
    ).copy()

    return LAMRotationSurface(
        coordinates=(q,),
        potential_MHz=potential,
        F_MHz=F,
        rho=rho,
        R_MHz=rotational,
        periodic=(True,),
        labels=(label,),
        quartic_distortion_MHz=quartic_surface,
        sextic_distortion_MHz=sextic,
        distortion_reduction="A",
        lam_quartic_kinetic_MHz=lam_p4,
        p2_rotational_metric_MHz=p2_rotational,
        p_rotational_cubic_MHz=p_rotational_cubic,
        torsional_symmetry_order=parameters.torsional_symmetry_order,
        metadata={
            "representation": "rho-axis method",
            "ram_schema": LOCK_RAM_ONE_TOP_SCHEMA,
            **dict(metadata or {}),
        },
    )


def solve_ram_one_top_k_contracted(
    parameters: RAMOneTopParameters,
    *,
    j_values: Sequence[int] | range,
    k_truncation: int = 10,
    torsional_states_per_symmetry: int = 9,
    levels_per_symmetry: int | Mapping[int, int] | None = None,
    workers: int = 1,
    compute_parameter_derivatives: bool = False,
    parameter_derivative_names: Sequence[str] | None = None,
    contracted_basis_response: bool = True,
    tracking_steps: int = 0,
    return_state_vectors: bool = False,
    state_vector_energy_window_MHz: float | None = None,
    higher_order_terms: Sequence[RAM36OperatorTerm] = (),
    higher_order_parameters: Sequence[RAM36OperatorParameter] = (),
) -> RAMKContractedResult:
    """Solve a one-top RAM Hamiltonian with the BELGI two-step contraction.

    The first diagonalization uses ``exp(i(3k+sigma) alpha)`` functions and
    includes the complete diagonal-in-K torsional core
    ``F(p-rho K)^2 + V + k4 p^4``.  The second diagonalization projects every
    operator represented by :class:`RAMOneTopParameters` into the retained,
    K-dependent torsional functions.  A and E are independent blocks; the E-
    partner is exactly degenerate for this Cs Hamiltonian and is not repeated.
    Optional parameter derivatives use Hellmann-Feynman expectation values in
    the current contracted basis; a fit should rebuild the contraction after
    every accepted parameter step. ``levels_per_symmetry`` accepts either one
    count for every J block or an explicit ``{J: count}`` map.
    """
    values = tuple(sorted(set(int(J) for J in j_values)))
    if not values or values[0] < 0:
        raise ValueError("at least one nonnegative J value is required")
    primitive_size = 2 * int(k_truncation) + 1
    if k_truncation < 1:
        raise ValueError("k_truncation must be positive")
    if not 1 <= torsional_states_per_symmetry <= primitive_size:
        raise ValueError("torsional contraction must fit the Fourier basis")
    if isinstance(levels_per_symmetry, Mapping):
        requested_levels = {int(J): int(count) for J, count in levels_per_symmetry.items()}
        invalid_counts = any(count < 1 for count in requested_levels.values())
        if set(requested_levels) != set(values) or invalid_counts:
            raise ValueError(
                "levels_per_symmetry mapping must contain one positive count per requested J"
            )
    else:
        if levels_per_symmetry is not None and levels_per_symmetry < 1:
            raise ValueError("levels_per_symmetry must be positive")
        requested_levels = None
    if workers < 1:
        raise ValueError("workers must be positive")
    if tracking_steps < 0:
        raise ValueError("tracking_steps must be nonnegative")
    if state_vector_energy_window_MHz is not None and (
        not math.isfinite(state_vector_energy_window_MHz)
        or state_vector_energy_window_MHz < 0.0
    ):
        raise ValueError("state-vector energy window must be finite and nonnegative")
    requested_derivative_names = (
        None
        if parameter_derivative_names is None
        else tuple(dict.fromkeys(str(name) for name in parameter_derivative_names))
    )
    extra_terms = tuple(higher_order_terms)
    if not all(isinstance(term, RAM36OperatorTerm) for term in extra_terms):
        raise TypeError("higher_order_terms must contain RAM36OperatorTerm values")
    extra_parameters = tuple(higher_order_parameters)
    if not all(isinstance(item, RAM36OperatorParameter) for item in extra_parameters):
        raise TypeError("higher_order_parameters must contain RAM36OperatorParameter values")
    derivative_names = [item.derivative_name for item in extra_parameters]
    if len(derivative_names) != len(set(derivative_names)):
        raise ValueError("higher-order RAM36 parameter names must be unique")
    extra_terms += tuple(term for item in extra_parameters for term in item.terms())

    def diagonalize_sector(J: int, sigma: int, coupling_scale: float):
        hamiltonian, contractions = _ram_k_sector_hamiltonian(
            parameters,
            J,
            sigma=sigma,
            k_truncation=k_truncation,
            torsional_states=torsional_states_per_symmetry,
            coupling_scale=coupling_scale,
            higher_order_terms=extra_terms,
        )
        dimension = len(hamiltonian)
        requested = (
            requested_levels[J]
            if requested_levels is not None
            else levels_per_symmetry
        )
        retained = dimension if requested is None else min(int(requested), dimension)
        if retained == dimension:
            energies, eigenvectors = np.linalg.eigh(hamiltonian)
        else:
            from scipy.linalg import eigh

            energies, eigenvectors = eigh(
                hamiltonian,
                subset_by_index=(0, retained - 1),
                driver="evr",
                check_finite=False,
            )
        primitive_size = 2 * k_truncation + 1
        primitive_vectors = np.empty((primitive_size * (2 * J + 1), retained), dtype=complex)
        for k_index, contraction in enumerate(contractions):
            contracted_slice = slice(
                k_index * torsional_states_per_symmetry,
                (k_index + 1) * torsional_states_per_symmetry,
            )
            primitive_slice = slice(
                k_index * primitive_size,
                (k_index + 1) * primitive_size,
            )
            primitive_vectors[primitive_slice] = contraction @ eigenvectors[contracted_slice]
        return hamiltonian, contractions, energies, eigenvectors, primitive_vectors

    def solve_j(J: int):
        sector_results = []
        maximum_error = 0.0
        for sigma, symmetry in ((0, "A"), (1, "E+")):
            initial_scale = 0.0 if tracking_steps else 1.0
            hamiltonian, vectors, energies, eigenvectors, primitive_vectors = diagonalize_sector(
                J, sigma, initial_scale
            )
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(hamiltonian - hamiltonian.conj().T), initial=0.0)),
            )
            retained = eigenvectors.shape[1]
            rot = angular_momentum_operators(J, "Ir")
            reference_R = np.asarray(
                [
                    [parameters.A, parameters.D_ab, 0.0],
                    [parameters.D_ab, parameters.B, 0.0],
                    [0.0, 0.0, parameters.C],
                ]
            )
            # The assembled basis is K-major and its torsional functions vary
            # with K.  Project each K block onto the K=0 contraction before
            # assigning rigid-rotor parents; treating the coefficients as a
            # common direct-product basis swaps strongly mixed asymmetry
            # doublets.  Parent assignment expects torsion-major ordering.
            reference_contraction = vectors[J]
            referenced = np.empty_like(eigenvectors)
            for k_index, contraction in enumerate(vectors):
                block = slice(
                    k_index * torsional_states_per_symmetry,
                    (k_index + 1) * torsional_states_per_symmetry,
                )
                referenced[block] = (
                    reference_contraction.conj().T @ contraction @ eigenvectors[block]
                )
            reordered = (
                referenced.reshape((2 * J + 1, torsional_states_per_symmetry, retained))
                .transpose(1, 0, 2)
                .reshape((torsional_states_per_symmetry * (2 * J + 1), retained))
            )
            parents = _parent_assignments(J, reordered, rot, reference_R)
            if tracking_steps:
                from scipy.optimize import linear_sum_assignment

                tracked_parents = list(parents)
                previous_vectors = primitive_vectors
                for coupling_scale in np.linspace(0.0, 1.0, tracking_steps + 1)[1:]:
                    hamiltonian, vectors, energies, eigenvectors, primitive_vectors = (
                        diagonalize_sector(J, sigma, float(coupling_scale))
                    )
                    maximum_error = max(
                        maximum_error,
                        float(np.max(np.abs(hamiltonian - hamiltonian.conj().T), initial=0.0)),
                    )
                    rows, columns = linear_sum_assignment(
                        -(np.abs(previous_vectors.conj().T @ primitive_vectors) ** 2)
                    )
                    remapped: list[tuple[int, int, int, float] | None] = [None] * retained
                    for row, column in zip(rows, columns, strict=True):
                        parent = tracked_parents[int(row)]
                        remapped[int(column)] = (
                            parent[0],
                            parent[1],
                            parent[2],
                            float(
                                abs(
                                    np.vdot(
                                        previous_vectors[:, int(row)],
                                        primitive_vectors[:, int(column)],
                                    )
                                )
                                ** 2
                            ),
                        )
                    if any(parent is None for parent in remapped):
                        raise RuntimeError("adiabatic RAM tracking lost a contracted state")
                    tracked_parents = [parent for parent in remapped if parent is not None]
                    previous_vectors = primitive_vectors
                parents = tracked_parents
            derivatives = (
                _ram_k_parameter_derivatives(
                    parameters,
                    J,
                    sigma=sigma,
                    k_truncation=k_truncation,
                    contractions=vectors,
                    eigenvectors=eigenvectors,
                    include_basis_response=contracted_basis_response,
                    parameter_names=requested_derivative_names,
                    higher_order_terms=extra_terms,
                    higher_order_parameters=extra_parameters,
                )
                if compute_parameter_derivatives
                else {}
            )
            stored_indices = np.arange(len(energies), dtype=int)
            if return_state_vectors and state_vector_energy_window_MHz is not None:
                selected = np.zeros(len(energies), dtype=bool)
                if len(energies) > 1:
                    close = np.diff(energies) <= state_vector_energy_window_MHz
                    selected[:-1] |= close
                    selected[1:] |= close
                stored_indices = np.flatnonzero(selected)
            stored_vectors = (
                primitive_vectors[:, stored_indices] if return_state_vectors else None
            )
            if stored_vectors is not None:
                stored_vectors.setflags(write=False)
            sector_results.append(
                (
                    symmetry,
                    energies.real,
                    parents,
                    vectors,
                    derivatives,
                    stored_vectors,
                    tuple(int(value) for value in stored_indices),
                )
            )
        records = []
        for (
            symmetry,
            energies,
            parents,
            _vectors,
            derivatives,
            _state_vectors,
            _state_indices,
        ) in sector_results:
            records.extend(
                (float(energy), symmetry, parent, local_index, derivatives)
                for local_index, (energy, parent) in enumerate(zip(energies, parents, strict=True))
            )
        records.sort(key=lambda item: item[0])
        levels = tuple(
            RAMKContractedLevel(
                J=J,
                index=index,
                energy_MHz=energy,
                lam_symmetry=symmetry,
                torsional_state=parent[0],
                Ka=parent[1],
                Kc=parent[2],
                parent_purity=parent[3],
            )
            for index, (energy, symmetry, parent, _local_index, _derivatives) in enumerate(records)
        )
        derivative_names = tuple(sector_results[0][4])
        merged_derivatives = {
            name: np.asarray(
                [derivatives[name][local_index] for _, _, _, local_index, derivatives in records]
            )
            for name in derivative_names
        }
        return (
            J,
            RAMKContractedBlock(
                J=J,
                energies_MHz=np.asarray([item.energy_MHz for item in levels]),
                levels=levels,
                parameter_derivatives_MHz=merged_derivatives,
                state_vectors_by_symmetry={
                    symmetry: state_vectors
                    for (
                        symmetry,
                        _energies,
                        _parents,
                        _vectors,
                        _derivatives,
                        state_vectors,
                        _state_indices,
                    ) in sector_results
                    if state_vectors is not None
                },
                state_vector_indices_by_symmetry={
                    symmetry: state_indices
                    for (
                        symmetry,
                        _energies,
                        _parents,
                        _vectors,
                        _derivatives,
                        state_vectors,
                        state_indices,
                    ) in sector_results
                    if state_vectors is not None
                },
            ),
            maximum_error,
        )

    if workers == 1 or len(values) == 1:
        solved = tuple(solve_j(J) for J in values)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(int(workers), len(values)), thread_name_prefix="lock-ram-k"
        ) as pool:
            solved = tuple(pool.map(solve_j, values))
    return RAMKContractedResult(
        blocks={J: block for J, block, _error in solved},
        k_truncation=int(k_truncation),
        torsional_states_per_symmetry=int(torsional_states_per_symmetry),
        levels_per_symmetry=levels_per_symmetry,
        tracking_steps=int(tracking_steps),
        native_acceleration=_ram_native_available(),
        hermiticity_error_MHz=max(error for _J, _block, error in solved),
    )


def track_ram_states_by_overlap(
    reference: RAMKContractedResult,
    current: RAMKContractedResult,
    *,
    J: int,
    symmetry: str,
) -> RAMStateOverlapMapping:
    """Map retained RAM states between parameter points by wavefunction overlap."""

    if J not in reference.blocks or J not in current.blocks:
        raise ValueError(f"J={J} must be present in both RAM solutions")
    if symmetry not in {"A", "E+"}:
        raise ValueError("symmetry must be A or E+")
    left = reference.blocks[J].state_vectors_by_symmetry.get(symmetry)
    right = current.blocks[J].state_vectors_by_symmetry.get(symmetry)
    if left is None or right is None:
        raise ValueError("RAM overlap tracking requires return_state_vectors=True")
    if left.shape[0] != right.shape[0]:
        raise ValueError("RAM overlap tracking requires identical primitive bases")
    left_indices = reference.blocks[J].state_vector_indices_by_symmetry.get(
        symmetry, tuple(range(left.shape[1]))
    )
    right_indices = current.blocks[J].state_vector_indices_by_symmetry.get(
        symmetry, tuple(range(right.shape[1]))
    )
    if not len(left_indices) or not len(right_indices):
        return RAMStateOverlapMapping(
            J=int(J),
            symmetry=symmetry,
            reference_indices=(),
            current_indices=(),
            squared_overlaps=(),
            minimum_squared_overlap=1.0,
            mean_squared_overlap=1.0,
        )
    from scipy.optimize import linear_sum_assignment

    squared = np.abs(left.conj().T @ right) ** 2
    rows, columns = linear_sum_assignment(-squared)
    assigned = squared[rows, columns]
    return RAMStateOverlapMapping(
        J=int(J),
        symmetry=symmetry,
        reference_indices=tuple(int(left_indices[value]) for value in rows),
        current_indices=tuple(int(right_indices[value]) for value in columns),
        squared_overlaps=tuple(float(value) for value in assigned),
        minimum_squared_overlap=float(np.min(assigned)),
        mean_squared_overlap=float(np.mean(assigned)),
    )


def _ram_native_available() -> bool:
    return bool(
        _lock_native is not None
        and hasattr(_lock_native, "assemble_ram_factorized")
        and hasattr(_lock_native, "ram_factorized_expectation")
    )


@lru_cache(maxsize=512)
def _ram_operator_basis(J: int, sigma: int, k_truncation: int):
    """Cache parameter-independent Fourier and angular-momentum operators."""
    momenta = 3.0 * np.arange(-k_truncation, k_truncation + 1, dtype=float) + float(sigma)
    identity = np.eye(len(momenta), dtype=complex)
    p = np.diag(momenta).astype(complex)
    p2, p4 = p @ p, np.linalg.matrix_power(p, 4)
    c3 = identity.copy()
    c6 = identity.copy()
    for left in range(len(momenta)):
        for shift, target in ((1, c3), (2, c6)):
            right = left + shift
            if right < len(momenta):
                target[left, right] -= 0.5
                target[right, left] -= 0.5
    rot = angular_momentum_operators(J, "Ir")
    distortion = centrifugal_distortion_operators(J, "A", "Ir")
    for array in (identity, p, p2, p4, c3, c6, *rot, *distortion):
        array.setflags(write=False)
    return momenta, identity, p, p2, p4, c3, c6, rot, distortion


def _ram_k_sector_hamiltonian(
    parameters: RAMOneTopParameters,
    J: int,
    *,
    sigma: int,
    k_truncation: int,
    torsional_states: int,
    coupling_scale: float = 1.0,
    higher_order_terms: Sequence[RAM36OperatorTerm] = (),
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if not 0.0 <= coupling_scale <= 1.0:
        raise ValueError("RAM coupling_scale must be between zero and one")
    _momenta, identity, p, p2, p4, c3, c6, rot, distortion = _ram_operator_basis(
        J, sigma, k_truncation
    )
    Pa, Pb, Pc = rot
    Pa2, Pb2, Pc2 = Pa @ Pa, Pb @ Pb, Pc @ Pc
    total2 = Pa2 + Pb2 + Pc2
    higher_factors = ram36_factorized_terms_many(
        higher_order_terms,
        torsional_momentum=p,
        angular_momenta=rot,
    )
    potential = 0.5 * parameters.V3 * c3 + 0.5 * parameters.V6 * c6
    K_values = np.arange(-J, J + 1, dtype=float)
    contractions = []
    effective_rho = coupling_scale * parameters.rho
    for K in K_values:
        first = (
            parameters.F * ((p - effective_rho * K * identity) @ (p - effective_rho * K * identity))
            + potential
            + parameters.k4 * p4
        )
        _energies, vectors = np.linalg.eigh(0.5 * (first + first.conj().T))
        contractions.append(vectors[:, :torsional_states])
    contractions = tuple(contractions)

    dimension = (2 * J + 1) * torsional_states
    factors: list[tuple[float, np.ndarray, np.ndarray]] = []

    def add_factor(lam_operator: np.ndarray, rot_operator: np.ndarray, coefficient=1.0):
        if coefficient:
            factors.append((float(coefficient), lam_operator, rot_operator))

    add_factor(parameters.F * p2 + potential + parameters.k4 * p4, np.eye(2 * J + 1))
    add_factor(p, Pa, -2.0 * parameters.F * effective_rho)
    add_factor(identity, Pa2, parameters.F * effective_rho**2 + parameters.A)
    add_factor(identity, Pb2, parameters.B)
    add_factor(identity, Pc2, parameters.C)
    add_factor(identity, Pa @ Pb + Pb @ Pa, parameters.D_ab)
    add_factor(c3, Pa2, coupling_scale * parameters.k5)
    add_factor(c3, total2, coupling_scale * parameters.Fv)
    add_factor(c3, Pb2 - Pc2, coupling_scale * parameters.c2)
    add_factor(c6, total2, coupling_scale * parameters.Nv)
    add_factor(p2, total2, coupling_scale * parameters.Gv)
    add_factor(p2, Pa @ Pb + Pb @ Pa, coupling_scale * parameters.Delta_ab)
    add_factor(p, Pa @ total2, coupling_scale * parameters.Lv)

    coefficients = (
        -coupling_scale * parameters.Delta_J,
        -coupling_scale * parameters.Delta_JK,
        -coupling_scale * parameters.Delta_K,
        -coupling_scale * parameters.delta_J,
        -coupling_scale * parameters.delta_K,
        coupling_scale * parameters.HJ,
        coupling_scale * parameters.HJK,
        coupling_scale * parameters.HKJ,
        coupling_scale * parameters.HK,
        coupling_scale * parameters.h1,
        coupling_scale * parameters.h2,
        coupling_scale * parameters.h3,
    )
    for coefficient, operator in zip(coefficients, distortion[:len(coefficients)], strict=True):
        if coefficient:
            add_factor(identity, operator, coefficient)
    for coefficient, lam_operator, rot_operator in higher_factors:
        add_factor(lam_operator, rot_operator, coupling_scale * coefficient)
    if _ram_native_available():
        hamiltonian = np.asarray(
            _lock_native.assemble_ram_factorized(np.stack(contractions), factors),
            dtype=complex,
        )
    else:
        hamiltonian = np.zeros((dimension, dimension), dtype=complex)
        for coefficient, lam_operator, rot_operator in factors:
            rows, columns = np.nonzero(np.abs(rot_operator) > 1.0e-14)
            for left, right in zip(rows, columns, strict=True):
                block = contractions[left].conj().T @ lam_operator @ contractions[right]
                left_slice = slice(left * torsional_states, (left + 1) * torsional_states)
                right_slice = slice(right * torsional_states, (right + 1) * torsional_states)
                hamiltonian[left_slice, right_slice] += (
                    coefficient * rot_operator[left, right] * block
                )
    return 0.5 * (hamiltonian + hamiltonian.conj().T), contractions


def _ram_k_parameter_derivatives(
    parameters: RAMOneTopParameters,
    J: int,
    *,
    sigma: int,
    k_truncation: int,
    contractions: tuple[np.ndarray, ...],
    eigenvectors: np.ndarray,
    include_basis_response: bool = True,
    parameter_names: Sequence[str] | None = None,
    higher_order_terms: Sequence[RAM36OperatorTerm] = (),
    higher_order_parameters: Sequence[RAM36OperatorParameter] = (),
) -> dict[str, np.ndarray]:
    """Return analytic Hellmann-Feynman and contracted-basis derivatives."""
    _momenta, identity, p, p2, p4, c3, c6, rot, distortion = _ram_operator_basis(
        J, sigma, k_truncation
    )
    Pa, Pb, Pc = rot
    Pa2, Pb2, Pc2 = Pa @ Pa, Pb @ Pb, Pc @ Pc
    total2 = Pa2 + Pb2 + Pc2
    rot_identity = np.eye(2 * J + 1)
    cross_ab = Pa @ Pb + Pb @ Pa

    def expectation(lam_operator: np.ndarray, rot_operator: np.ndarray) -> np.ndarray:
        if _ram_native_available():
            return np.asarray(
                _lock_native.ram_factorized_expectation(
                    eigenvectors,
                    np.stack(contractions),
                    lam_operator,
                    rot_operator,
                ),
                dtype=float,
            )
        values = np.zeros(eigenvectors.shape[1], dtype=complex)
        rows, columns = np.nonzero(np.abs(rot_operator) > 1.0e-14)
        torsional_states = contractions[0].shape[1]
        for left, right in zip(rows, columns, strict=True):
            left_slice = slice(left * torsional_states, (left + 1) * torsional_states)
            right_slice = slice(right * torsional_states, (right + 1) * torsional_states)
            block = contractions[left].conj().T @ lam_operator @ contractions[right]
            operated = block @ eigenvectors[right_slice]
            values += rot_operator[left, right] * np.einsum(
                "ij,ij->j", eigenvectors[left_slice].conj(), operated, optimize=True
            )
        return values.real

    factor_groups: dict[str, tuple[tuple[float, np.ndarray, np.ndarray], ...]] = {
        "F": (
            (1.0, p2, rot_identity),
            (-2.0 * parameters.rho, p, Pa),
            (parameters.rho**2, identity, Pa2),
        ),
        "A": ((1.0, identity, Pa2),),
        "B": ((1.0, identity, Pb2),),
        "C": ((1.0, identity, Pc2),),
        "D_ab": ((1.0, identity, cross_ab),),
        "Delta_J": ((-1.0, identity, distortion[0]),),
        "Delta_JK": ((-1.0, identity, distortion[1]),),
        "Delta_K": ((-1.0, identity, distortion[2]),),
        "delta_J": ((-1.0, identity, distortion[3]),),
        "delta_K": ((-1.0, identity, distortion[4]),),
        "V3": ((0.5, c3, rot_identity),),
        "V6": ((0.5, c6, rot_identity),),
        "rho": (
            (-2.0 * parameters.F, p, Pa),
            (2.0 * parameters.F * parameters.rho, identity, Pa2),
        ),
        "k5": ((1.0, c3, Pa2),),
        "Fv": ((1.0, c3, total2),),
        "c2": ((1.0, c3, Pb2 - Pc2),),
        "Delta_ab": ((1.0, p2, cross_ab),),
        "Lv": ((1.0, p, Pa @ total2),),
        "Gv": ((1.0, p2, total2),),
        "k4": ((1.0, p4, rot_identity),),
        "Nv": ((1.0, c6, total2),),
        "HJ": ((1.0, identity, distortion[5]),),
        "HJK": ((1.0, identity, distortion[6]),),
        "HKJ": ((1.0, identity, distortion[7]),),
        "HK": ((1.0, identity, distortion[8]),),
        "h1": ((1.0, identity, distortion[9]),),
        "h2": ((1.0, identity, distortion[10]),),
        "h3": ((1.0, identity, distortion[11]),),
    }
    # Build every higher-order derivative in one pass.  The RAM36 registry
    # reuses a small set of momentum powers and torsional harmonics; doing one
    # pass lets ram36_factorized_terms_many cache them across all parameters.
    higher_derivative_terms = tuple(
        (parameter.derivative_name, parameter.derivative_terms())
        for parameter in higher_order_parameters
    )
    flattened_derivative_terms = tuple(
        term for _name, terms in higher_derivative_terms for term in terms
    )
    flattened_derivative_factors = ram36_factorized_terms_many(
        flattened_derivative_terms,
        torsional_momentum=p,
        angular_momenta=rot,
    )
    factor_offset = 0
    for name, terms in higher_derivative_terms:
        factor_count = 2 * len(terms)
        factor_groups[name] = flattened_derivative_factors[
            factor_offset : factor_offset + factor_count
        ]
        factor_offset += factor_count
    selected = tuple(factor_groups) if parameter_names is None else tuple(parameter_names)
    unknown = set(selected) - set(factor_groups)
    if unknown:
        raise ValueError(f"unknown RAM derivative parameters: {sorted(unknown)}")
    selected_groups = tuple(factor_groups[name] for name in selected)
    if _ram_native_available() and hasattr(_lock_native, "ram_factorized_expectations"):
        values = np.asarray(
            _lock_native.ram_factorized_expectations(
                eigenvectors,
                np.stack(contractions),
                selected_groups,
            ),
            dtype=float,
        )
        derivatives = {name: values[index] for index, name in enumerate(selected)}
    else:
        derivatives = {
            name: sum(
                coefficient * expectation(lam_operator, rot_operator)
                for coefficient, lam_operator, rot_operator in factor_groups[name]
            )
            for name in selected
        }
    response_selected = tuple(
        name for name in ("F", "V3", "V6", "rho", "k4") if name in selected
    )
    if include_basis_response and response_selected:
        responses = _ram_k_contracted_basis_response(
            parameters,
            J,
            sigma=sigma,
            k_truncation=k_truncation,
            contractions=contractions,
            eigenvectors=eigenvectors,
            higher_order_terms=higher_order_terms,
        )
        for name in response_selected:
            values = responses[name]
            derivatives[name] += values
    return derivatives


def _ram_k_contracted_basis_response(
    parameters: RAMOneTopParameters,
    J: int,
    *,
    sigma: int,
    k_truncation: int,
    contractions: tuple[np.ndarray, ...],
    eigenvectors: np.ndarray,
    higher_order_terms: Sequence[RAM36OperatorTerm] = (),
) -> dict[str, np.ndarray]:
    """Return Pulay-like derivatives from the K-dependent torsional contraction.

    The retained first-step eigenvectors respond through the discarded Fourier
    functions.  Rotations within the retained subspace are a gauge choice and
    cancel from second-step eigenvalues, so only retained--discarded couplings
    enter the response.
    """

    _momenta, identity, p, p2, p4, c3, c6, rot, distortion = _ram_operator_basis(
        J, sigma, k_truncation
    )
    torsional_states = contractions[0].shape[1]
    potential = 0.5 * parameters.V3 * c3 + 0.5 * parameters.V6 * c6

    response_names = ("F", "V3", "V6", "rho", "k4")
    contraction_derivatives: dict[str, list[np.ndarray]] = {name: [] for name in response_names}
    for K, retained_vectors in zip(np.arange(-J, J + 1, dtype=float), contractions, strict=True):
        shifted = p - parameters.rho * K * identity
        first = parameters.F * (shifted @ shifted) + potential + parameters.k4 * p4
        energies, full_vectors = np.linalg.eigh(0.5 * (first + first.conj().T))
        discarded = full_vectors[:, torsional_states:]
        discarded_energies = energies[torsional_states:]
        retained_energies = energies[:torsional_states]
        gaps = retained_energies[None, :] - discarded_energies[:, None]
        if gaps.size and float(np.min(np.abs(gaps))) < 1.0e-10:
            raise RuntimeError("RAM contraction response encountered a zero retained-state gap")
        first_derivatives = {
            "F": shifted @ shifted,
            "V3": 0.5 * c3,
            "V6": 0.5 * c6,
            "rho": -2.0 * parameters.F * K * p
            + 2.0 * parameters.F * parameters.rho * K**2 * identity,
            "k4": p4,
        }
        for name, derivative in first_derivatives.items():
            if discarded.shape[1]:
                coupling = discarded.conj().T @ derivative @ retained_vectors
                response = discarded @ (coupling / gaps)
            else:
                response = np.zeros_like(retained_vectors)
            contraction_derivatives[name].append(response)

    Pa, Pb, Pc = rot
    Pa2, Pb2, Pc2 = Pa @ Pa, Pb @ Pb, Pc @ Pc
    total2 = Pa2 + Pb2 + Pc2
    cross_ab = Pa @ Pb + Pb @ Pa
    rot_identity = np.eye(2 * J + 1)
    factors: list[tuple[float, np.ndarray, np.ndarray]] = [
        (1.0, parameters.F * p2 + potential + parameters.k4 * p4, rot_identity),
        (-2.0 * parameters.F * parameters.rho, p, Pa),
        (parameters.F * parameters.rho**2 + parameters.A, identity, Pa2),
        (parameters.B, identity, Pb2),
        (parameters.C, identity, Pc2),
        (parameters.D_ab, identity, cross_ab),
        (parameters.k5, c3, Pa2),
        (parameters.Fv, c3, total2),
        (parameters.c2, c3, Pb2 - Pc2),
        (parameters.Nv, c6, total2),
        (parameters.Gv, p2, total2),
        (parameters.Delta_ab, p2, cross_ab),
        (parameters.Lv, p, Pa @ total2),
        (-parameters.Delta_J, identity, distortion[0]),
        (-parameters.Delta_JK, identity, distortion[1]),
        (-parameters.Delta_K, identity, distortion[2]),
        (-parameters.delta_J, identity, distortion[3]),
        (-parameters.delta_K, identity, distortion[4]),
        (parameters.HJ, identity, distortion[5]),
        (parameters.HJK, identity, distortion[6]),
        (parameters.HKJ, identity, distortion[7]),
        (parameters.HK, identity, distortion[8]),
        (parameters.h1, identity, distortion[9]),
        (parameters.h2, identity, distortion[10]),
        (parameters.h3, identity, distortion[11]),
    ]
    factors = [term for term in factors if term[0] != 0.0]
    factors.extend(
        ram36_factorized_terms_many(
            higher_order_terms,
            torsional_momentum=p,
            angular_momenta=rot,
        )
    )
    if _ram_native_available() and hasattr(
        _lock_native, "ram_contracted_basis_responses"
    ):
        response_values = np.asarray(
            _lock_native.ram_contracted_basis_responses(
                eigenvectors,
                np.stack(contractions),
                np.asarray(
                    [contraction_derivatives[name] for name in response_names],
                    dtype=complex,
                ),
                factors,
            ),
            dtype=float,
        )
        return {
            name: response_values[index]
            for index, name in enumerate(response_names)
        }
    values_by_name: dict[str, np.ndarray] = {}
    for name in response_names:
        values = np.zeros(eigenvectors.shape[1], dtype=complex)
        derivatives = contraction_derivatives[name]
        for coefficient, lam_operator, rot_operator in factors:
            rows, columns = np.nonzero(np.abs(rot_operator) > 1.0e-14)
            for left, right in zip(rows, columns, strict=True):
                response_block = (
                    derivatives[left].conj().T @ lam_operator @ contractions[right]
                    + contractions[left].conj().T @ lam_operator @ derivatives[right]
                )
                left_slice = slice(left * torsional_states, (left + 1) * torsional_states)
                right_slice = slice(right * torsional_states, (right + 1) * torsional_states)
                operated = response_block @ eigenvectors[right_slice]
                values += (
                    coefficient
                    * rot_operator[left, right]
                    * np.einsum(
                        "ij,ij->j",
                        eigenvectors[left_slice].conj(),
                        operated,
                        optimize=False,
                    )
                )
        values_by_name[name] = values.real
    return values_by_name


def rho_axis_frame(
    rho_pas: Sequence[float],
    *,
    preferred_c_axis_pas: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return the right-handed PAS-to-RAM component transformation.

    Rows of the returned orthogonal matrix are the RAM unit axes expressed in
    PAS components.  The RAM ``a`` axis is parallel to ``rho_pas``; the ``c``
    axis is chosen as close as possible to ``preferred_c_axis_pas``.
    """

    rho_vector = np.asarray(rho_pas, dtype=float)
    preferred = np.asarray(preferred_c_axis_pas, dtype=float)
    if rho_vector.shape != (3,) or preferred.shape != (3,):
        raise ValueError("rho and preferred c axis must have three components")
    if not np.all(np.isfinite(rho_vector)) or not np.all(np.isfinite(preferred)):
        raise ValueError("rho and preferred c axis must be finite")
    norm = float(np.linalg.norm(rho_vector))
    if norm <= 0.0:
        raise ValueError("rho must be nonzero to define a rho-axis frame")
    a_axis = rho_vector / norm
    c_projection = preferred - np.dot(preferred, a_axis) * a_axis
    if np.linalg.norm(c_projection) < 1.0e-12:
        candidates = np.eye(3)
        preferred = candidates[np.argmin(np.abs(candidates @ a_axis))]
        c_projection = preferred - np.dot(preferred, a_axis) * a_axis
    c_axis = c_projection / np.linalg.norm(c_projection)
    b_axis = np.cross(c_axis, a_axis)
    b_axis /= np.linalg.norm(b_axis)
    frame = np.vstack((a_axis, b_axis, c_axis))
    if np.linalg.det(frame) < 0.0:
        frame[1] *= -1.0
    return frame


def vector_pas_to_ram(vector_pas: Sequence[float], frame: np.ndarray) -> np.ndarray:
    """Transform vector components from PAS to RAM."""

    vector = np.asarray(vector_pas, dtype=float)
    rotation = _validated_frame(frame)
    if vector.shape != (3,):
        raise ValueError("a PAS vector must have three components")
    return rotation @ vector


def tensor_pas_to_ram(tensor_pas: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Transform one or more Cartesian rank-two tensors from PAS to RAM."""

    tensor = np.asarray(tensor_pas, dtype=float)
    rotation = _validated_frame(frame)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("PAS tensors must end in a 3 by 3 Cartesian block")
    return np.einsum("ai,...ij,bj->...ab", rotation, tensor, rotation, optimize=True)


def _validated_frame(frame: np.ndarray) -> np.ndarray:
    rotation = np.asarray(frame, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("the PAS-to-RAM frame must be a finite 3 by 3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12):
        raise ValueError("the PAS-to-RAM frame must be orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-12):
        raise ValueError("the PAS-to-RAM frame must be right handed")
    return rotation


__all__ = [
    "CM1_TO_MHZ",
    "LOCK_RAM_ONE_TOP_SCHEMA",
    "RAMOneTopParameters",
    "ram_one_top_surface",
    "rho_axis_frame",
    "tensor_pas_to_ram",
    "vector_pas_to_ram",
]
