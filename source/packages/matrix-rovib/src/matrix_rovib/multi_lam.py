"""General product-DVR builders for coupled internal rotors and other LAMs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .lam_rotation import LAMRotationSurface, PermutationInversionSymmetry


@dataclass(frozen=True)
class GeneralLAMCoordinate:
    """One SMITH coordinate grid, periodic torsion or nonperiodic tunnelling path."""

    label: str
    points: tuple[float, ...]
    periodic: bool
    coordinate_kind: str = "generalized"

    def __post_init__(self) -> None:
        values = np.asarray(self.points, dtype=float)
        if not self.label.strip() or len(values) < 3 or not np.all(np.isfinite(values)):
            raise ValueError("a general LAM coordinate needs a label and three finite points")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("general LAM coordinate points must be strictly increasing")
        if not self.coordinate_kind.strip():
            raise ValueError("general LAM coordinate kind must be nonempty")
        if self.periodic:
            spacing = np.diff(values)
            if not np.allclose(spacing, spacing[0], rtol=1.0e-9, atol=1.0e-12):
                raise ValueError("periodic LAM grids must be uniform for the Fourier DVR")


def general_lam_surface(
    coordinates: Sequence[GeneralLAMCoordinate],
    *,
    potential_MHz: np.ndarray,
    F_MHz: np.ndarray,
    rho: np.ndarray,
    R_MHz: np.ndarray,
    dipole_au: np.ndarray | None = None,
    geometric_potential_MHz: np.ndarray | None = None,
    spin_tensor_surfaces_MHz: Mapping[str, np.ndarray] | None = None,
    permutation_inversion_symmetry: PermutationInversionSymmetry | None = None,
    metadata: Mapping[str, object] | None = None,
) -> LAMRotationSurface:
    """Build the production surface for arbitrary SMITH large-amplitude paths.

    All fields are representation-independent arrays evaluated on the product
    grid. This supports coupled internal rotors, inversion, proton transfer and
    other non-torsional tunnelling coordinates without molecule-specific code.
    """

    axes = tuple(coordinates)
    if not 1 <= len(axes) <= 4 or len({axis.label for axis in axes}) != len(axes):
        raise ValueError("general LAM surfaces require one to four unique coordinates")
    merged_metadata = {
        "representation": "general SMITH product DVR",
        "coordinate_kinds": [axis.coordinate_kind for axis in axes],
        **dict(metadata or {}),
    }
    return LAMRotationSurface(
        coordinates=tuple(np.asarray(axis.points, dtype=float) for axis in axes),
        potential_MHz=np.asarray(potential_MHz, dtype=float),
        F_MHz=np.asarray(F_MHz, dtype=float),
        rho=np.asarray(rho, dtype=float),
        R_MHz=np.asarray(R_MHz, dtype=float),
        periodic=tuple(axis.periodic for axis in axes),
        labels=tuple(axis.label for axis in axes),
        dipole_au=None if dipole_au is None else np.asarray(dipole_au, dtype=float),
        geometric_potential_MHz=(
            None
            if geometric_potential_MHz is None
            else np.asarray(geometric_potential_MHz, dtype=float)
        ),
        spin_tensor_surfaces_MHz=spin_tensor_surfaces_MHz,
        permutation_inversion_symmetry=permutation_inversion_symmetry,
        metadata=merged_metadata,
    )


@dataclass(frozen=True)
class InternalRotorLAM:
    """One periodic internal rotor in a multi-LAM rho-axis Hamiltonian."""

    label: str
    F_MHz: float
    rho_abc: tuple[float, float, float]
    V3_MHz: float
    V6_MHz: float = 0.0
    k4_MHz: float = 0.0
    symmetry_order: int = 3
    rotation_one_minus_cos_MHz: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("multi-LAM rotor labels must be nonempty")
        values = (self.F_MHz, self.V3_MHz, self.V6_MHz, self.k4_MHz, *self.rho_abc)
        if not all(math.isfinite(float(value)) for value in values) or self.F_MHz <= 0.0:
            raise ValueError("multi-LAM rotor constants must be finite and F positive")
        if int(self.symmetry_order) != self.symmetry_order or self.symmetry_order < 1:
            raise ValueError("multi-LAM symmetry orders must be positive integers")
        rotation = np.asarray(self.rotation_one_minus_cos_MHz, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation modulation must be a finite 3 by 3 matrix")
        if not np.allclose(rotation, rotation.T, atol=1.0e-12):
            raise ValueError("rotation modulation must be symmetric")


@dataclass(frozen=True)
class MultiLAMFourierTerm:
    """A scalar product-Fourier term over all LAM coordinates."""

    coefficient_MHz: float
    harmonics: tuple[int, ...]
    functions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.coefficient_MHz)):
            raise ValueError("multi-LAM Fourier coefficients must be finite")
        if len(self.harmonics) != len(self.functions) or not self.harmonics:
            raise ValueError("a Fourier term requires matching harmonics and functions")
        if any(int(value) != value or value < 0 for value in self.harmonics):
            raise ValueError("Fourier harmonics must be nonnegative integers")
        if any(value not in {"cos", "sin", "one_minus_cos"} for value in self.functions):
            raise ValueError("Fourier functions must be cos, sin, or one_minus_cos")


@dataclass(frozen=True)
class MultiLAMRotationTerm:
    """A product-Fourier modulation of the rotational metric."""

    matrix_MHz: tuple[tuple[float, float, float], ...]
    harmonics: tuple[int, ...]
    functions: tuple[str, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix_MHz, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("multi-LAM rotational terms require a finite 3 by 3 matrix")
        if not np.allclose(matrix, matrix.T, atol=1.0e-12):
            raise ValueError("multi-LAM rotational matrices must be symmetric")
        MultiLAMFourierTerm(1.0, self.harmonics, self.functions)


@dataclass(frozen=True)
class MultiLAMP2RotationTerm:
    """Fourier-modulated ``p_mu^2 J_a J_b`` coefficient surface."""

    lam_axis: int
    matrix_MHz: tuple[tuple[float, float, float], ...]
    harmonics: tuple[int, ...]
    functions: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.lam_axis, bool) or int(self.lam_axis) != self.lam_axis or self.lam_axis < 0:
            raise ValueError("p2-rotation terms require a nonnegative LAM-axis index")
        MultiLAMRotationTerm(self.matrix_MHz, self.harmonics, self.functions)


@dataclass(frozen=True)
class MultiLAMP1RotationCubicTerm:
    """Fourier-modulated ``p_mu J_a J^2`` coefficient surface."""

    lam_axis: int
    vector_MHz: tuple[float, float, float]
    harmonics: tuple[int, ...]
    functions: tuple[str, ...]

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector_MHz, dtype=float)
        if (
            isinstance(self.lam_axis, bool)
            or int(self.lam_axis) != self.lam_axis
            or self.lam_axis < 0
            or vector.shape != (3,)
            or not np.all(np.isfinite(vector))
        ):
            raise ValueError("p-rotation-cubic terms require a LAM axis and finite vector")
        MultiLAMFourierTerm(1.0, self.harmonics, self.functions)


@dataclass(frozen=True)
class MultiLAMP1RotationAnticommutatorTerm:
    """Fourier-modulated ``p_mu {J_a^2,J_b}/2`` interaction.

    The diagonal case ``a == b`` is ``p_mu J_a^3``.  Sums over ``a`` recover
    ``p_mu J_b J^2``.  The representation therefore covers the independent
    cubic torsion--rotation operators without molecule-specific branches.
    """

    lam_axis: int
    squared_axis: int
    linear_axis: int
    coefficient_MHz: float
    harmonics: tuple[int, ...]
    functions: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.lam_axis, bool)
            or int(self.lam_axis) != self.lam_axis
            or self.lam_axis < 0
            or self.squared_axis not in range(3)
            or self.linear_axis not in range(3)
            or not math.isfinite(float(self.coefficient_MHz))
        ):
            raise ValueError("p-rotation anticommutators require valid LAM and rotation axes")
        MultiLAMFourierTerm(1.0, self.harmonics, self.functions)


@dataclass(frozen=True)
class MultiTopRAMParameters:
    """Constant metrics and Fourier surfaces for one to four coupled tops."""

    A_MHz: float
    B_MHz: float
    C_MHz: float
    tops: tuple[InternalRotorLAM, ...]
    D_ab_MHz: float = 0.0
    kinetic_couplings_MHz: tuple[tuple[float, ...], ...] = ()
    potential_couplings: tuple[MultiLAMFourierTerm, ...] = ()
    rotation_couplings: tuple[MultiLAMRotationTerm, ...] = ()
    p2_rotation_couplings: tuple[MultiLAMP2RotationTerm, ...] = ()
    p1_rotation_cubic_couplings: tuple[MultiLAMP1RotationCubicTerm, ...] = ()
    p1_rotation_anticommutator_couplings: tuple[
        MultiLAMP1RotationAnticommutatorTerm, ...
    ] = ()
    quartic_distortion_MHz: tuple[float, ...] = (0.0,) * 5
    sextic_distortion_MHz: tuple[float, ...] = (0.0,) * 7
    distortion_reduction: str = "A"
    pi2_rotation_MHz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    nuclear_spin_weights: tuple[float, ...] = ()
    permutation_inversion_symmetry: PermutationInversionSymmetry | None = None

    def __post_init__(self) -> None:
        if not 1 <= len(self.tops) <= 4:
            raise ValueError("multi-top RAM supports one to four LAM coordinates")
        if not all(
            math.isfinite(float(value))
            for value in (self.A_MHz, self.B_MHz, self.C_MHz, self.D_ab_MHz)
        ):
            raise ValueError("multi-top rotational constants must be finite")
        rotational = np.asarray(
            [
                [self.A_MHz, self.D_ab_MHz, 0.0],
                [self.D_ab_MHz, self.B_MHz, 0.0],
                [0.0, 0.0, self.C_MHz],
            ]
        )
        if float(np.min(np.linalg.eigvalsh(rotational))) <= 0.0:
            raise ValueError("multi-top rotational metric must be positive definite")
        dimension = len(self.tops)
        if self.kinetic_couplings_MHz:
            coupling = np.asarray(self.kinetic_couplings_MHz, dtype=float)
            if coupling.shape != (dimension, dimension) or not np.all(np.isfinite(coupling)):
                raise ValueError("kinetic couplings must be a finite square LAM matrix")
            if not np.allclose(coupling, coupling.T, atol=1.0e-12):
                raise ValueError("kinetic couplings must be symmetric")
        if len(self.pi2_rotation_MHz) != 3 or not np.all(np.isfinite(self.pi2_rotation_MHz)):
            raise ValueError("pi2_rotation_MHz requires three finite coefficients")
        for term in (
            *self.potential_couplings,
            *self.rotation_couplings,
            *self.p2_rotation_couplings,
            *self.p1_rotation_cubic_couplings,
            *self.p1_rotation_anticommutator_couplings,
        ):
            if len(term.harmonics) != dimension:
                raise ValueError("every multi-LAM Fourier term must span all coordinates")
        if any(
            term.lam_axis >= dimension
            for term in (
                *self.p2_rotation_couplings,
                *self.p1_rotation_cubic_couplings,
                *self.p1_rotation_anticommutator_couplings,
            )
        ):
            raise ValueError("higher-order multi-LAM term references an absent coordinate")
        if len(self.quartic_distortion_MHz) != 5 or len(self.sextic_distortion_MHz) != 7:
            raise ValueError("multi-top distortion requires five quartic and seven sextic terms")
        if not np.all(np.isfinite((*self.quartic_distortion_MHz, *self.sextic_distortion_MHz))):
            raise ValueError("multi-top distortion constants must be finite")
        if self.distortion_reduction.upper() not in {"A", "S"}:
            raise ValueError("multi-top Watson reduction must be A or S")


@dataclass(frozen=True)
class LAMContractionPolicy:
    """Energy and symmetry controls for a J,K-dependent LAM contraction."""

    minimum_states: int = 12
    maximum_states: int | None = None
    states_per_symmetry_sector: int = 2
    energy_window_MHz: float = 100_000.0
    rotational_margin_MHz: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_states < 1 or self.states_per_symmetry_sector < 0:
            raise ValueError("LAM contraction state counts must be nonnegative and retain a state")
        if self.maximum_states is not None and self.maximum_states < self.minimum_states:
            raise ValueError("maximum_states must not be smaller than minimum_states")
        if self.energy_window_MHz < 0.0 or self.rotational_margin_MHz < 0.0:
            raise ValueError("LAM contraction energy margins must be nonnegative")


@dataclass(frozen=True)
class LAMContractionBlock:
    J: int
    K: int
    retained_indices: tuple[int, ...]
    energy_cutoff_MHz: float


@dataclass(frozen=True)
class RAMGridConvergenceStep:
    """One product-DVR refinement used to certify a multi-top calculation."""

    grid_sizes: tuple[int, ...]
    relative_energies_MHz: tuple[float, ...]
    maximum_change_MHz: float | None


@dataclass(frozen=True)
class RAMGridConvergenceResult:
    """Numerical certificate and converged surface for a multi-top RAM model."""

    surface: LAMRotationSurface
    steps: tuple[RAMGridConvergenceStep, ...]
    tolerance_MHz: float
    converged: bool

    @property
    def grid_sizes(self) -> tuple[int, ...]:
        return self.steps[-1].grid_sizes

    def require_convergence(self) -> "RAMGridConvergenceResult":
        if not self.converged:
            change = self.steps[-1].maximum_change_MHz
            raise RuntimeError(
                "multi-top product DVR did not converge: "
                f"last maximum relative-energy change is {change!r} MHz, "
                f"target {self.tolerance_MHz} MHz"
            )
        return self


def converge_ram_multi_top_grid(
    parameters: MultiTopRAMParameters,
    *,
    initial_grid_sizes: Sequence[int] | int = 9,
    maximum_grid_sizes: Sequence[int] | int = 45,
    comparison_states: int = 12,
    tolerance_MHz: float = 1.0e-3,
    consecutive: int = 1,
) -> RAMGridConvergenceResult:
    """Refine every periodic DVR until low torsional term values converge.

    Refinements add twice each rotor symmetry order. This preserves an odd
    number of Fourier points per physical period and avoids the unpaired
    Nyquist momentum that can split exactly degenerate E sectors on even
    grids. Energies are compared relative to the ground state so arbitrary
    zero-point shifts cannot produce a false failure.
    """

    from .lam_rotation import solve_lam_basis

    dimension = len(parameters.tops)
    orders = tuple(int(top.symmetry_order) for top in parameters.tops)

    def normalize(value: Sequence[int] | int, label: str) -> tuple[int, ...]:
        raw = (int(value),) * dimension if isinstance(value, int) else tuple(map(int, value))
        if len(raw) != dimension:
            raise ValueError(f"{label} requires one value per LAM coordinate")
        return raw

    current = normalize(initial_grid_sizes, "initial_grid_sizes")
    maximum = normalize(maximum_grid_sizes, "maximum_grid_sizes")
    if comparison_states < 2 or tolerance_MHz <= 0.0 or consecutive < 1:
        raise ValueError("grid convergence requires two states and positive controls")
    adjusted = []
    for size, order, limit in zip(current, orders, maximum, strict=True):
        periods = max(3, math.ceil(size / order))
        if periods % 2 == 0:
            periods += 1
        value = periods * order
        if value > limit:
            raise ValueError("maximum_grid_sizes is smaller than the normalized initial grid")
        adjusted.append(value)
    current = tuple(adjusted)
    if math.prod(current) < comparison_states:
        raise ValueError(
            "the normalized initial product grid has fewer points than comparison_states"
        )
    history: list[RAMGridConvergenceStep] = []
    previous: np.ndarray | None = None
    accepted = 0
    converged = False
    surface = ram_multi_top_surface(parameters, grid_sizes=current)
    while True:
        available = int(np.prod(current))
        count = min(available, max(comparison_states, comparison_states * 2))
        basis = solve_lam_basis(surface, state_count=count)
        relative = np.asarray(basis.energies_MHz[:comparison_states], dtype=float)
        relative -= relative[0]
        change = None if previous is None else float(np.max(np.abs(relative - previous)))
        history.append(RAMGridConvergenceStep(current, tuple(map(float, relative)), change))
        accepted = accepted + 1 if change is not None and change <= tolerance_MHz else 0
        if accepted >= consecutive:
            converged = True
            break
        following = tuple(
            size + 2 * order for size, order in zip(current, orders, strict=True)
        )
        if any(value > limit for value, limit in zip(following, maximum, strict=True)):
            break
        previous = relative
        current = following
        surface = ram_multi_top_surface(parameters, grid_sizes=current)
    return RAMGridConvergenceResult(
        surface=surface,
        steps=tuple(history),
        tolerance_MHz=float(tolerance_MHz),
        converged=converged,
    )


def plan_lam_contraction(
    basis,
    *,
    j_values: Sequence[int],
    policy: LAMContractionPolicy = LAMContractionPolicy(),
) -> tuple[LAMContractionBlock, ...]:
    """Select a stable LAM subspace separately for each physical J,K block.

    The cutoff grows with ``J(J+1)`` and ``K**2`` to admit LAM parents that can
    be pulled down by rotation.  A minimum number per product-symmetry sector
    prevents accidental symmetry loss at avoided crossings.
    """

    energies = np.asarray(basis.energies_MHz, dtype=float)
    if energies.ndim != 1 or not len(energies) or np.any(~np.isfinite(energies)):
        raise ValueError("LAM contraction requires a finite one-dimensional spectrum")
    sectors = np.asarray(
        basis.symmetry_sectors if basis.symmetry_sectors else (0,) * len(energies),
        dtype=int,
    )
    if sectors.shape != energies.shape:
        raise ValueError("LAM symmetry labels must match the contracted spectrum")
    order = np.argsort(energies, kind="stable")
    maximum = (
        len(energies)
        if policy.maximum_states is None
        else min(policy.maximum_states, len(energies))
    )
    blocks = []
    for J in sorted(set(int(value) for value in j_values)):
        if J < 0:
            raise ValueError("LAM contraction J values must be nonnegative")
        for K in range(-J, J + 1):
            cutoff = float(
                energies[order[0]]
                + policy.energy_window_MHz
                + policy.rotational_margin_MHz * (J * (J + 1) + K * K)
            )
            selected = set(int(index) for index in order[energies[order] <= cutoff])
            mandatory = set(int(index) for index in order[: policy.minimum_states])
            if policy.states_per_symmetry_sector:
                for sector in np.unique(sectors):
                    sector_order = order[sectors[order] == sector]
                    mandatory.update(
                        int(index) for index in sector_order[: policy.states_per_symmetry_sector]
                    )
            if len(mandatory) > maximum:
                raise ValueError(
                    "maximum_states is too small for the requested symmetry-sector floor"
                )
            selected.update(mandatory)
            optional = sorted(selected - mandatory, key=lambda index: (energies[index], index))[
                : maximum - len(mandatory)
            ]
            retained = tuple(
                sorted((*mandatory, *optional), key=lambda index: (energies[index], index))
            )
            blocks.append(LAMContractionBlock(J, K, retained, cutoff))
    return tuple(blocks)


def _fourier_product(
    mesh: tuple[np.ndarray, ...], harmonics: Sequence[int], functions: Sequence[str]
) -> np.ndarray:
    result = np.ones(tuple(array.shape[index] for index, array in enumerate(mesh)))
    for coordinate, harmonic, function in zip(mesh, harmonics, functions, strict=True):
        argument = int(harmonic) * coordinate
        if function == "cos":
            factor = np.cos(argument)
        elif function == "sin":
            factor = np.sin(argument)
        else:
            factor = 1.0 - np.cos(argument)
        result *= factor
    return result


def ram_multi_top_surface(
    parameters: MultiTopRAMParameters,
    *,
    grid_sizes: Sequence[int] | int = 12,
    dipole_au: Sequence[float] | None = None,
    spin_tensor_surfaces_MHz: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> LAMRotationSurface:
    """Build a general product-DVR surface for coupled internal rotors.

    Off-diagonal elements of ``kinetic_couplings_MHz`` represent LAM--LAM
    kinetic coupling.  ``potential_couplings`` provide torsion--torsion terms,
    while each rho vector and ``rotation_couplings`` provide LAM--rotation
    interactions.
    """

    dimension = len(parameters.tops)
    sizes = (
        (int(grid_sizes),) * dimension
        if isinstance(grid_sizes, int)
        else tuple(int(value) for value in grid_sizes)
    )
    if len(sizes) != dimension or any(size < 3 for size in sizes):
        raise ValueError("provide one grid size of at least three per LAM coordinate")
    orders = tuple(top.symmetry_order for top in parameters.tops)
    if any(size % order for size, order in zip(sizes, orders, strict=True)):
        raise ValueError("each periodic grid size must be divisible by its symmetry order")
    coordinates = tuple(np.arange(size, dtype=float) * (2.0 * np.pi / size) for size in sizes)
    mesh = tuple(np.meshgrid(*coordinates, indexing="ij", sparse=False))
    shape = tuple(sizes)
    potential = np.zeros(shape)
    for axis, top in enumerate(parameters.tops):
        potential += 0.5 * top.V3_MHz * (1.0 - np.cos(top.symmetry_order * mesh[axis]))
        potential += 0.5 * top.V6_MHz * (1.0 - np.cos(2 * top.symmetry_order * mesh[axis]))
    for term in parameters.potential_couplings:
        potential += term.coefficient_MHz * _fourier_product(mesh, term.harmonics, term.functions)

    kinetic = np.zeros((dimension, dimension))
    np.fill_diagonal(kinetic, [top.F_MHz for top in parameters.tops])
    if parameters.kinetic_couplings_MHz:
        coupling = np.asarray(parameters.kinetic_couplings_MHz, dtype=float)
        kinetic += coupling - np.diag(np.diag(coupling))
    if float(np.min(np.linalg.eigvalsh(kinetic))) <= 0.0:
        raise ValueError("the coupled multi-LAM kinetic metric must be positive definite")
    F = np.broadcast_to(kinetic, shape + kinetic.shape).copy()
    rho = np.zeros(shape + (3, dimension))
    for axis, top in enumerate(parameters.tops):
        rho[..., :, axis] = np.asarray(top.rho_abc)

    base_rotation = np.asarray(
        [
            [parameters.A_MHz, parameters.D_ab_MHz, 0.0],
            [parameters.D_ab_MHz, parameters.B_MHz, 0.0],
            [0.0, 0.0, parameters.C_MHz],
        ]
    )
    rotation = np.broadcast_to(base_rotation, shape + (3, 3)).copy()
    for axis, top in enumerate(parameters.tops):
        rotation += (1.0 - np.cos(top.symmetry_order * mesh[axis]))[..., None, None] * np.asarray(
            top.rotation_one_minus_cos_MHz
        )
    for term in parameters.rotation_couplings:
        rotation += _fourier_product(mesh, term.harmonics, term.functions)[
            ..., None, None
        ] * np.asarray(term.matrix_MHz)
    if float(np.min(np.linalg.eigvalsh(rotation))) <= 0.0:
        raise ValueError("the multi-LAM rotational metric must remain positive definite")

    quartic = np.zeros(shape + (dimension,))
    for axis, top in enumerate(parameters.tops):
        quartic[..., axis] = top.k4_MHz
    p2_rotation = np.zeros(shape + (dimension, 3, 3))
    for term in parameters.p2_rotation_couplings:
        p2_rotation[..., term.lam_axis, :, :] += _fourier_product(
            mesh, term.harmonics, term.functions
        )[..., None, None] * np.asarray(term.matrix_MHz)
    p1_rotation_cubic = np.zeros(shape + (dimension, 3))
    for term in parameters.p1_rotation_cubic_couplings:
        p1_rotation_cubic[..., term.lam_axis, :] += _fourier_product(
            mesh, term.harmonics, term.functions
        )[..., None] * np.asarray(term.vector_MHz)
    p1_rotation_anticommutator = np.zeros(shape + (dimension, 3, 3))
    for term in parameters.p1_rotation_anticommutator_couplings:
        p1_rotation_anticommutator[
            ..., term.lam_axis, term.squared_axis, term.linear_axis
        ] += term.coefficient_MHz * _fourier_product(
            mesh, term.harmonics, term.functions
        )
    quartic_distortion = np.broadcast_to(
        np.asarray(parameters.quartic_distortion_MHz), shape + (5,)
    ).copy()
    sextic_distortion = np.broadcast_to(
        np.asarray(parameters.sextic_distortion_MHz), shape + (7,)
    ).copy()
    dipole = None
    if dipole_au is not None:
        vector = np.asarray(tuple(dipole_au), dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("multi-LAM dipoles require three finite components")
        dipole = np.broadcast_to(vector, shape + (3,)).copy()
    return LAMRotationSurface(
        coordinates=coordinates,
        potential_MHz=potential,
        F_MHz=F,
        rho=rho,
        R_MHz=rotation,
        periodic=(True,) * dimension,
        labels=tuple(top.label for top in parameters.tops),
        dipole_au=dipole,
        lam_quartic_kinetic_MHz=quartic,
        p2_rotational_metric_MHz=p2_rotation,
        p_rotational_cubic_MHz=p1_rotation_cubic,
        p_rotational_anticommutator_MHz=p1_rotation_anticommutator,
        quartic_distortion_MHz=quartic_distortion,
        sextic_distortion_MHz=sextic_distortion,
        distortion_reduction=parameters.distortion_reduction,
        pi2_rotation_MHz=np.asarray(parameters.pi2_rotation_MHz),
        torsional_symmetry_order=orders[0],
        lam_symmetry_orders=orders,
        nuclear_spin_weights=parameters.nuclear_spin_weights,
        permutation_inversion_symmetry=parameters.permutation_inversion_symmetry,
        spin_tensor_surfaces_MHz=spin_tensor_surfaces_MHz,
        metadata={
            "representation": "multi-top rho-axis product DVR",
            "lam_symmetry_orders": list(orders),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "GeneralLAMCoordinate",
    "InternalRotorLAM",
    "LAMContractionBlock",
    "LAMContractionPolicy",
    "RAMGridConvergenceResult",
    "RAMGridConvergenceStep",
    "MultiLAMFourierTerm",
    "MultiLAMP1RotationCubicTerm",
    "MultiLAMP1RotationAnticommutatorTerm",
    "MultiLAMP2RotationTerm",
    "MultiLAMRotationTerm",
    "MultiTopRAMParameters",
    "ram_multi_top_surface",
    "converge_ram_multi_top_grid",
    "general_lam_surface",
    "plan_lam_contraction",
]
