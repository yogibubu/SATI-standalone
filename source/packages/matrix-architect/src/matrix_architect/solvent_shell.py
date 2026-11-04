"""Reproducible ice-lattice initialization of finite explicit-solvent shells."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Sequence, TypeVar

import numpy as np

from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.vdw_radii import descriptor_vdw_radius, uff_vdw_radius
from matrix_chem.geometry_alignment import rotation_vector_from_matrix
from matrix_zaff.tip3p_fb import tip3p_fb_parameters


ICE_SHELL_SCHEMA = "matrix.zaff.solvent_shell.v2"
WATER_NUMBER_DENSITY_ANGSTROM3 = 0.995 / 18.01528 * 6.02214076e23 / 1.0e24


@dataclass(frozen=True)
class IceShellSpecification:
    """Deterministic construction controls for a finite water shell."""

    padding_angstrom: float = 18.0
    target_number_density_angstrom3: float = WATER_NUMBER_DENSITY_ANGSTROM3
    shape_tolerance: float = 0.03
    solute_exclusion_scale: float = 0.92
    boundary_scale: float = 0.999
    loop_flips_per_water: float = 4.0
    density_tolerance_fraction: float = 0.01
    density_compensation_max_iterations: int = 6
    seed: int = 61453

    def __post_init__(self) -> None:
        positive = (
            self.padding_angstrom,
            self.target_number_density_angstrom3,
            self.shape_tolerance,
            self.solute_exclusion_scale,
            self.boundary_scale,
            self.loop_flips_per_water,
            self.density_tolerance_fraction,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("ice-shell construction parameters must be finite and positive")
        if not 0.0 < float(self.boundary_scale) < 1.0:
            raise ValueError("ice-shell boundary scale must be smaller than one")
        if not 0.0 < float(self.density_tolerance_fraction) < 1.0:
            raise ValueError("ice-shell density tolerance must lie between zero and one")
        if (
            int(self.density_compensation_max_iterations) < 1
            or int(self.density_compensation_max_iterations)
            != self.density_compensation_max_iterations
        ):
            raise ValueError("ice-shell density compensation needs at least one iteration")


@dataclass(frozen=True)
class IceShellBuild:
    """Canonical solute, cavity and rigid-water poses from one seeded build."""

    solute_elements: tuple[str, ...]
    solute_coordinates_angstrom: np.ndarray
    semiaxes_angstrom: np.ndarray
    shape: str
    water_poses_angstrom_radian: np.ndarray
    water_coordinates_angstrom: np.ndarray
    canonical_rotation: np.ndarray
    canonical_origin_angstrom: np.ndarray
    lattice_constant_angstrom: float
    parent_lattice_number_density_angstrom3: float
    nominal_number_density_angstrom3: float
    ellipsoid_volume_angstrom3: float
    fill_volume_angstrom3: float
    density_volume_basis: str
    target_molecule_count: int
    density_relative_error: float
    density_compensation_iterations: int
    seed: int
    schema: str = ICE_SHELL_SCHEMA


@dataclass(frozen=True)
class MolecularSolventLattice:
    """Documented parent lattice for a rigid molecular solvent."""

    name: str
    solvent: str
    lattice: str
    fractional_centers: np.ndarray
    cell_lengths_angstrom: tuple[float, float, float] | None
    orientation_policy: str
    provenance_doi: str


@dataclass(frozen=True)
class MolecularSolventShellBuild:
    """Canonical solute and rigid molecules cut from a molecular lattice."""

    template_name: str
    solute_elements: tuple[str, ...]
    solute_coordinates_angstrom: np.ndarray
    solvent_elements: tuple[str, ...]
    solvent_coordinates_angstrom: np.ndarray
    semiaxes_angstrom: np.ndarray
    shape: str
    molecular_centers_angstrom: np.ndarray
    molecular_rotations: np.ndarray
    canonical_rotation: np.ndarray
    canonical_origin_angstrom: np.ndarray
    parent_lattice_number_density_angstrom3: float
    nominal_number_density_angstrom3: float
    ellipsoid_volume_angstrom3: float
    fill_volume_angstrom3: float
    density_volume_basis: str
    target_molecule_count: int
    density_relative_error: float
    density_compensation_iterations: int
    seed: int
    schema: str = ICE_SHELL_SCHEMA


_FCC_CENTERS = np.asarray(
    ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
)
MOLECULAR_SOLVENT_LATTICES = {
    "generic-fcc-reference": MolecularSolventLattice(
        name="generic-fcc-reference", solvent="generic molecular solvent",
        lattice="FCC reference bath; molecular orientations supplied by seed",
        fractional_centers=_FCC_CENTERS, cell_lengths_angstrom=None,
        orientation_policy="independent seeded Haar orientations",
        provenance_doi="internal reference-lattice specification",
    ),
    "water-fcc-reference": MolecularSolventLattice(
        name="water-fcc-reference", solvent="water",
        lattice="FCC reference bath; molecular orientations supplied by seed",
        fractional_centers=_FCC_CENTERS, cell_lengths_angstrom=None,
        orientation_policy="independent seeded Haar orientations",
        provenance_doi="internal reference-lattice specification",
    ),
    "ammonia-fcc-reference": MolecularSolventLattice(
        name="ammonia-fcc-reference", solvent="ammonia",
        lattice="FCC reference bath; molecular orientations supplied by seed",
        fractional_centers=_FCC_CENTERS, cell_lengths_angstrom=None,
        orientation_policy="independent seeded Haar orientations",
        provenance_doi="internal reference-lattice specification",
    ),
    "formic-acid-fcc-reference": MolecularSolventLattice(
        name="formic-acid-fcc-reference", solvent="formic acid",
        lattice="FCC reference bath; molecular orientations supplied by seed",
        fractional_centers=_FCC_CENTERS, cell_lengths_angstrom=None,
        orientation_policy="independent seeded Haar orientations",
        provenance_doi="internal reference-lattice specification",
    ),
    "cyclohexane-fcc-plastic": MolecularSolventLattice(
        name="cyclohexane-fcc-plastic",
        solvent="cyclohexane",
        lattice="FCC (plastic phase I, Fm-3m)",
        fractional_centers=_FCC_CENTERS,
        cell_lengths_angstrom=None,
        orientation_policy="independent seeded Haar orientations",
        provenance_doi="10.1107/S2414314623001141",
    ),
    "carbon-tetrachloride-fcc-plastic": MolecularSolventLattice(
        name="carbon-tetrachloride-fcc-plastic",
        solvent="carbon tetrachloride",
        lattice="FCC plastic phase Ia",
        fractional_centers=_FCC_CENTERS,
        cell_lengths_angstrom=None,
        orientation_policy="independent seeded Haar orientations",
        provenance_doi="10.1063/1.440896",
    ),
    "benzene-pbca": MolecularSolventLattice(
        name="benzene-pbca",
        solvent="benzene",
        lattice="orthorhombic Pbca, pseudo-FCC molecular centers",
        fractional_centers=_FCC_CENTERS,
        cell_lengths_angstrom=(7.460, 9.666, 7.034),
        orientation_policy="alternating molecular-plane normals; anneal before production",
        provenance_doi="10.1038/173075a0",
    ),
}


_FillBuild = TypeVar("_FillBuild", IceShellBuild, MolecularSolventShellBuild)


def _compensated_lattice_fill(
    builder: Callable[[float], _FillBuild],
    count: Callable[[_FillBuild], int],
    *,
    target_count: int,
    initial_parent_density: float,
    tolerance_fraction: float,
    max_iterations: int,
) -> _FillBuild:
    """Solve the discrete ellipsoid-fill problem for any molecular lattice.

    Rebuilding from the same seed makes every density trial deterministic.
    The multiplicative update compensates jointly for the finite boundary,
    solute carving, and whole-molecule clipping.  A build is never returned
    silently outside its declared density tolerance.
    """

    parent_density = float(initial_parent_density)
    best: _FillBuild | None = None
    best_error = math.inf
    best_iteration = 0
    for iteration in range(1, int(max_iterations) + 1):
        if not math.isfinite(parent_density) or parent_density <= 0.0:
            raise RuntimeError("ellipsoid density compensation became nonphysical")
        candidate = builder(parent_density)
        achieved_count = count(candidate)
        if achieved_count < 0:
            raise RuntimeError("ellipsoid fill returned a negative molecule count")
        count_error = abs(achieved_count - target_count)
        relative_error = count_error / target_count
        if relative_error < best_error:
            best = candidate
            best_error = relative_error
            best_iteration = iteration
        # One whole molecule is the irreducible resolution of a finite fill.
        allowed_count_error = max(1, int(math.ceil(tolerance_fraction * target_count)))
        if count_error <= allowed_count_error:
            return replace(
                candidate,
                density_relative_error=relative_error,
                density_compensation_iterations=iteration,
            )
        parent_density *= 2.0 if achieved_count == 0 else target_count / achieved_count
    assert best is not None
    raise RuntimeError(
        "ellipsoid density compensation failed after "
        f"{max_iterations} iterations: best count {count(best)} versus target "
        f"{target_count} ({best_error:.3%} relative error at iteration "
        f"{best_iteration})"
    )


def available_molecular_solvent_lattices() -> tuple[MolecularSolventLattice, ...]:
    """Return immutable documented templates for nonaqueous shell builders."""

    return tuple(MOLECULAR_SOLVENT_LATTICES[name] for name in sorted(MOLECULAR_SOLVENT_LATTICES))


def molecular_solvent_lattice_centers(
    template_name: str,
    semiaxes_angstrom: Sequence[float],
    *,
    number_density_angstrom3: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate, rotate and clip a documented molecular-center lattice.

    The returned rotations are seeded Haar rotations for plastic crystals.
    Benzene centers use the experimental Pbca cell; its rotations are initial
    annealing guesses rather than a claim of fully refined atomic positions.
    """

    if template_name not in MOLECULAR_SOLVENT_LATTICES:
        raise ValueError(f"unknown molecular-solvent lattice: {template_name}")
    template = MOLECULAR_SOLVENT_LATTICES[template_name]
    axes = np.asarray(semiaxes_angstrom, dtype=float).reshape(3)
    density = float(number_density_angstrom3)
    if np.any(axes <= 0.0) or not math.isfinite(density) or density <= 0.0:
        raise ValueError("molecular lattice needs positive semiaxes and density")
    if template.cell_lengths_angstrom is None:
        cell = np.full(3, (len(template.fractional_centers) / density) ** (1.0 / 3.0))
    else:
        reference = np.asarray(template.cell_lengths_angstrom, dtype=float)
        reference_density = len(template.fractional_centers) / float(np.prod(reference))
        cell = reference * (reference_density / density) ** (1.0 / 3.0)
    extent = int(math.ceil(2.0 * (float(np.max(axes)) + np.max(cell)) / np.min(cell)))
    points = []
    offset = -0.5 * extent * cell
    for i in range(extent):
        for j in range(extent):
            for k in range(extent):
                cell_index = np.asarray((i, j, k), dtype=float)
                for center in template.fractional_centers:
                    points.append(offset + (cell_index + center) * cell)
    rng = np.random.default_rng(int(seed))
    global_rotation = _haar_rotation(rng)
    coordinates = np.asarray(points) @ global_rotation.T
    selected = np.linalg.norm(coordinates / axes, axis=1) < 0.999
    centers = coordinates[selected]
    rotations = np.asarray([_haar_rotation(rng) for _ in range(len(centers))])
    return centers, rotations


def build_molecular_solvent_shell(
    solute_elements: Sequence[str],
    solute_coordinates_angstrom: np.ndarray,
    solvent_elements: Sequence[str],
    solvent_coordinates_angstrom: np.ndarray,
    *,
    template_name: str,
    number_density_angstrom3: float,
    padding_angstrom: float = 18.0,
    shape_tolerance: float = 0.03,
    solute_exclusion_scale: float = 0.92,
    boundary_scale: float = 0.999,
    density_tolerance_fraction: float = 0.01,
    density_compensation_max_iterations: int = 6,
    seed: int = 61453,
) -> MolecularSolventShellBuild:
    """Cut a seeded rigid-molecule shell from a documented parent lattice.

    Solvent coordinates define one rigid molecule and are centered at their
    geometric centroid. Plastic-crystal orientations are deliberately seeded
    guesses that must be annealed before production sampling.
    """

    elements = tuple(str(symbol) for symbol in solute_elements)
    solute = np.asarray(solute_coordinates_angstrom, dtype=float)
    solvent_symbols = tuple(str(symbol) for symbol in solvent_elements)
    solvent = np.asarray(solvent_coordinates_angstrom, dtype=float)
    scalars = (
        number_density_angstrom3,
        padding_angstrom,
        shape_tolerance,
        solute_exclusion_scale,
        boundary_scale,
        density_tolerance_fraction,
    )
    if (
        not elements
        or solute.shape != (len(elements), 3)
        or not solvent_symbols
        or solvent.shape != (len(solvent_symbols), 3)
        or np.any(~np.isfinite(solute))
        or np.any(~np.isfinite(solvent))
        or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in scalars)
        or boundary_scale >= 1.0
        or density_tolerance_fraction >= 1.0
        or int(density_compensation_max_iterations) < 1
        or int(density_compensation_max_iterations)
        != density_compensation_max_iterations
    ):
        raise ValueError("solute, solvent and molecular-shell controls are inconsistent")
    canonical, extent, canonical_rotation, origin = _canonical_solute(elements, solute)
    semiaxes, shape = _classified_semiaxes(extent, padding_angstrom, shape_tolerance)
    centered_solvent = solvent - np.mean(solvent, axis=0)
    molecular_radius = float(np.max(np.linalg.norm(centered_solvent, axis=1)))
    center_axes = boundary_scale * semiaxes - molecular_radius
    if np.any(center_axes <= 0.0):
        raise ValueError("cavity is too small for the supplied solvent molecule")
    solute_radii = np.asarray([_element_radius(symbol) for symbol in elements])
    solvent_radii = np.asarray([_element_radius(symbol) for symbol in solvent_symbols])
    ellipsoid_volume = 4.0 * math.pi * float(np.prod(semiaxes)) / 3.0
    # Whole-molecule containment reduces the region accessible to molecular
    # centers.  Use that explicit geometric volume for the density contract;
    # subsequent compensation restores sites removed by solute carving.
    fill_volume = 4.0 * math.pi * float(np.prod(center_axes)) / 3.0
    target_count = max(5, int(round(float(number_density_angstrom3) * fill_volume)))

    def candidate(parent_density: float) -> MolecularSolventShellBuild:
        centers, rotations = molecular_solvent_lattice_centers(
            template_name,
            center_axes,
            number_density_angstrom3=parent_density,
            seed=seed,
        )
        accepted_coordinates = []
        accepted_centers = []
        accepted_rotations = []
        for center, rotation in zip(centers, rotations, strict=True):
            molecule = centered_solvent @ rotation.T + center
            if np.max(np.linalg.norm(molecule / semiaxes, axis=1)) >= boundary_scale:
                continue
            pair_distance = np.linalg.norm(
                molecule[:, None, :] - canonical[None, :, :],
                axis=2,
            )
            exclusion = solute_exclusion_scale * (
                solvent_radii[:, None] + solute_radii[None, :]
            )
            if np.any(pair_distance < exclusion):
                continue
            accepted_coordinates.append(molecule)
            accepted_centers.append(center)
            accepted_rotations.append(rotation)
        achieved = len(accepted_coordinates)
        return MolecularSolventShellBuild(
            template_name=template_name,
            solute_elements=elements,
            solute_coordinates_angstrom=canonical,
            solvent_elements=solvent_symbols,
            solvent_coordinates_angstrom=np.asarray(accepted_coordinates).reshape(
                achieved, len(solvent_symbols), 3
            ),
            semiaxes_angstrom=semiaxes,
            shape=shape,
            molecular_centers_angstrom=np.asarray(accepted_centers).reshape(achieved, 3),
            molecular_rotations=np.asarray(accepted_rotations).reshape(achieved, 3, 3),
            canonical_rotation=canonical_rotation,
            canonical_origin_angstrom=origin,
            parent_lattice_number_density_angstrom3=parent_density,
            nominal_number_density_angstrom3=achieved / fill_volume,
            ellipsoid_volume_angstrom3=ellipsoid_volume,
            fill_volume_angstrom3=fill_volume,
            density_volume_basis="conservative whole-molecule center ellipsoid",
            target_molecule_count=target_count,
            density_relative_error=abs(achieved - target_count) / target_count,
            density_compensation_iterations=1,
            seed=int(seed),
        )

    return _compensated_lattice_fill(
        candidate,
        lambda build: len(build.solvent_coordinates_angstrom),
        target_count=target_count,
        initial_parent_density=number_density_angstrom3,
        tolerance_fraction=density_tolerance_fraction,
        max_iterations=density_compensation_max_iterations,
    )


def build_composite_outer_cluster(
    composite_elements: Sequence[str],
    composite_coordinates_angstrom: np.ndarray,
    solvent_elements: Sequence[str],
    solvent_coordinates_angstrom: np.ndarray,
    *,
    template_name: str,
    shell_count: int = 3,
    solvent_number_density_angstrom3: float,
    delta_vdw_factor: float = 1.0,
    **kwargs,
) -> MolecularSolventShellBuild:
    """Carve one common outer cluster around a solute--first-shell composite.

    The cavity is defined by the composite extrema plus ``shell_count``
    solvent diameters and one configurable vdW-radius delta.  The returned
    build is suitable for every family when called with a common reference
    lattice and fixed controls.
    """
    if int(shell_count) < 1:
        raise ValueError("shell_count must be positive")
    radii = [_element_radius(symbol) for symbol in solvent_elements]
    solvent_radius = max(radii, default=0.0)
    if solvent_radius <= 0.0:
        raise ValueError("solvent vdW radii are unavailable")
    padding = (2.0 * int(shell_count) + float(delta_vdw_factor)) * solvent_radius
    return build_molecular_solvent_shell(
        composite_elements,
        composite_coordinates_angstrom,
        solvent_elements,
        solvent_coordinates_angstrom,
        template_name=template_name,
        number_density_angstrom3=solvent_number_density_angstrom3,
        padding_angstrom=padding,
        **kwargs,
    )


_DIAMOND_BASIS_QUARTERS = np.asarray(
    (
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 0),
        (3, 3, 1),
        (2, 0, 2),
        (3, 1, 3),
        (0, 2, 2),
        (1, 3, 3),
    ),
    dtype=int,
)
_A_BASIS = frozenset((0, 2, 4, 6))
_TETRAHEDRAL_QUARTER_VECTORS = np.asarray(
    ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
    dtype=int,
)


def _element_radius(symbol: str) -> float:
    number = int(atomic_number(symbol) or 0)
    value = descriptor_vdw_radius(number)
    if value is None:
        value = uff_vdw_radius(number)
    if value is None or value <= 0.0:
        raise ValueError(f"no van der Waals radius is available for {symbol}")
    return float(value)


def _canonical_solute(
    elements: tuple[str, ...],
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid = np.mean(coordinates, axis=0)
    centered = coordinates - centroid
    covariance = centered.T @ centered
    _values, vectors = np.linalg.eigh(covariance)
    vectors = vectors[:, ::-1]
    projected = centered @ vectors
    for axis in range(3):
        pivot = int(np.argmax(np.abs(projected[:, axis])))
        if projected[pivot, axis] < 0.0:
            vectors[:, axis] *= -1.0
            projected[:, axis] *= -1.0
    if np.linalg.det(vectors) < 0.0:
        vectors[:, -1] *= -1.0
        projected[:, -1] *= -1.0
    radii = np.asarray([_element_radius(symbol) for symbol in elements])
    lower = np.min(projected - radii[:, None], axis=0)
    upper = np.max(projected + radii[:, None], axis=0)
    midpoint = 0.5 * (lower + upper)
    canonical = projected - midpoint
    full_extent = upper - lower
    origin = centroid + midpoint @ vectors.T
    return canonical, full_extent, vectors, origin


def _classified_semiaxes(
    full_extent: np.ndarray,
    padding: float,
    tolerance: float,
) -> tuple[np.ndarray, str]:
    axes = 0.5 * np.asarray(full_extent, dtype=float) + float(padding)
    order = np.argsort(-axes)
    sorted_axes = axes[order].copy()
    if np.ptp(sorted_axes) / sorted_axes[0] <= 1.0e-10:
        shape = "SPHERE"
    elif abs(sorted_axes[0] - sorted_axes[1]) / sorted_axes[0] <= tolerance:
        shape = "OBLATE_SPHEROID"
    elif abs(sorted_axes[1] - sorted_axes[2]) / sorted_axes[0] <= tolerance:
        shape = "PROLATE_SPHEROID"
    else:
        shape = "TRIAXIAL_ELLIPSOID"
    return axes, shape


def _haar_rotation(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=4)
    raw /= np.linalg.norm(raw)
    w, x, y, z = raw
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def _periodic_ice_ic(
    half_width: float,
    number_density: float,
    rng: np.random.Generator,
    loop_flips_per_water: float,
) -> tuple[np.ndarray, list[list[int]], np.ndarray, float, int]:
    lattice = (8.0 / number_density) ** (1.0 / 3.0)
    cell_count = max(4, int(math.ceil(2.0 * (half_width + lattice) / lattice)))
    quarter_period = 4 * cell_count
    quarter_coordinates = []
    basis_labels = []
    for i in range(cell_count):
        for j in range(cell_count):
            for k in range(cell_count):
                shift = 4 * np.asarray((i, j, k), dtype=int)
                for basis_index, basis in enumerate(_DIAMOND_BASIS_QUARTERS):
                    quarter_coordinates.append(shift + basis)
                    basis_labels.append(basis_index)
    quarter = np.asarray(quarter_coordinates, dtype=int)
    lookup = {tuple(point): index for index, point in enumerate(quarter)}
    edges: list[tuple[int, int]] = []
    categories = []
    for node, (point, basis_index) in enumerate(zip(quarter, basis_labels, strict=True)):
        if basis_index not in _A_BASIS:
            continue
        for category, vector in enumerate(_TETRAHEDRAL_QUARTER_VECTORS):
            target = tuple(((point + vector) % quarter_period).tolist())
            edges.append((node, lookup[target]))
            categories.append(category)
    edge_array = np.asarray(edges, dtype=int)
    source_is_left = np.asarray([category < 2 for category in categories], dtype=bool)
    incident: list[list[int]] = [[] for _ in range(len(quarter))]
    for edge_index, (left, right) in enumerate(edge_array):
        incident[int(left)].append(edge_index)
        incident[int(right)].append(edge_index)

    def source(edge_index: int) -> int:
        left, right = edge_array[edge_index]
        return int(left if source_is_left[edge_index] else right)

    flip_count = int(round(loop_flips_per_water * len(quarter)))
    for _ in range(flip_count):
        current = int(rng.integers(len(quarter)))
        visited: dict[int, int] = {}
        path_edges: list[int] = []
        while current not in visited:
            visited[current] = len(path_edges)
            outgoing = [edge for edge in incident[current] if source(edge) == current]
            edge_index = outgoing[int(rng.integers(len(outgoing)))]
            path_edges.append(edge_index)
            left, right = edge_array[edge_index]
            current = int(right if current == left else left)
        for edge_index in path_edges[visited[current] :]:
            source_is_left[edge_index] = not source_is_left[edge_index]

    outgoing_neighbors: list[list[int]] = [[] for _ in range(len(quarter))]
    for edge_index, (left, right) in enumerate(edge_array):
        if source_is_left[edge_index]:
            outgoing_neighbors[int(left)].append(int(right))
        else:
            outgoing_neighbors[int(right)].append(int(left))
    if any(len(neighbors) != 2 for neighbors in outgoing_neighbors):
        raise RuntimeError("periodic ice-rule orientation lost two-donor conservation")
    coordinates = quarter.astype(float) * (lattice / 4.0)
    coordinates -= 0.5 * cell_count * lattice
    return coordinates, outgoing_neighbors, quarter, lattice, quarter_period


def build_tip3p_fb_ice_shell(
    solute_elements: Sequence[str],
    solute_coordinates_angstrom: np.ndarray,
    *,
    specification: IceShellSpecification | None = None,
) -> IceShellBuild:
    """Build a canonical TIP3P-FB shell from proton-disordered ice Ic.

    The periodic parent lattice obeys two-in/two-out ice rules.  Random
    directed-loop reversals decorrelate its proton order without changing
    those rules.  The lattice is then rotated, translated, clipped to the
    solute-derived cavity, and carved using element-resolved van der Waals
    exclusion radii.
    """

    policy = specification or IceShellSpecification()
    elements = tuple(str(symbol) for symbol in solute_elements)
    coordinates = np.asarray(solute_coordinates_angstrom, dtype=float)
    if (
        not elements
        or coordinates.shape != (len(elements), 3)
        or np.any(~np.isfinite(coordinates))
    ):
        raise ValueError("solute elements and coordinates are inconsistent")
    canonical, full_extent, canonical_rotation, origin = _canonical_solute(
        elements,
        coordinates,
    )
    semiaxes, shape = _classified_semiaxes(
        full_extent,
        policy.padding_angstrom,
        policy.shape_tolerance,
    )
    return _build_tip3p_fb_ice_in_ellipsoid(
        elements,
        canonical,
        semiaxes,
        shape,
        canonical_rotation,
        origin,
        policy,
    )


def build_tip3p_fb_ice_ellipsoid(
    semiaxes_angstrom: Sequence[float],
    *,
    specification: IceShellSpecification | None = None,
) -> IceShellBuild:
    """Build a pure-water Ice-Ic starting state in a fixed ellipsoid."""

    policy = specification or IceShellSpecification()
    semiaxes = np.asarray(semiaxes_angstrom, dtype=float).reshape(3)
    if np.any(~np.isfinite(semiaxes)) or np.any(semiaxes <= 0.0):
        raise ValueError("pure-water ellipsoid semiaxes must be finite and positive")
    sorted_axes = np.sort(semiaxes)[::-1]
    if np.ptp(sorted_axes) / sorted_axes[0] <= 1.0e-10:
        shape = "SPHERE"
    elif abs(sorted_axes[0] - sorted_axes[1]) / sorted_axes[0] <= policy.shape_tolerance:
        shape = "OBLATE_SPHEROID"
    elif abs(sorted_axes[1] - sorted_axes[2]) / sorted_axes[0] <= policy.shape_tolerance:
        shape = "PROLATE_SPHEROID"
    else:
        shape = "TRIAXIAL_ELLIPSOID"
    return _build_tip3p_fb_ice_in_ellipsoid(
        (),
        np.empty((0, 3)),
        semiaxes,
        shape,
        np.eye(3),
        np.zeros(3),
        policy,
    )


def _build_tip3p_fb_ice_in_ellipsoid(
    elements: tuple[str, ...],
    canonical: np.ndarray,
    semiaxes: np.ndarray,
    shape: str,
    canonical_rotation: np.ndarray,
    origin: np.ndarray,
    policy: IceShellSpecification,
) -> IceShellBuild:
    """Build at a compensated parent density and enforce the fill contract."""

    volume = 4.0 * math.pi * float(np.prod(semiaxes)) / 3.0
    target_count = max(5, int(round(policy.target_number_density_angstrom3 * volume)))
    return _compensated_lattice_fill(
        lambda parent_density: _build_tip3p_fb_ice_at_parent_density(
            elements,
            canonical,
            semiaxes,
            shape,
            canonical_rotation,
            origin,
            policy,
            parent_density,
            target_count,
            volume,
        ),
        lambda build: len(build.water_coordinates_angstrom),
        target_count=target_count,
        initial_parent_density=policy.target_number_density_angstrom3,
        tolerance_fraction=policy.density_tolerance_fraction,
        max_iterations=policy.density_compensation_max_iterations,
    )


def _build_tip3p_fb_ice_at_parent_density(
    elements: tuple[str, ...],
    canonical: np.ndarray,
    semiaxes: np.ndarray,
    shape: str,
    canonical_rotation: np.ndarray,
    origin: np.ndarray,
    policy: IceShellSpecification,
    parent_density: float,
    target_count: int,
    volume: float,
) -> IceShellBuild:
    """Make one deterministic Ice-Ic cut at a specified parent density."""

    rng = np.random.default_rng(int(policy.seed))
    lattice_xyz, outgoing, quarter, lattice, period = _periodic_ice_ic(
        float(np.max(semiaxes)),
        parent_density,
        rng,
        policy.loop_flips_per_water,
    )
    lattice_rotation = _haar_rotation(rng)
    translation = rng.uniform(-0.5 * lattice, 0.5 * lattice, size=3)
    rotated_oxygen = lattice_xyz @ lattice_rotation.T + translation
    solute_radii = np.asarray([_element_radius(symbol) for symbol in elements])
    oxygen_radius = _element_radius("O")
    water = tip3p_fb_parameters()
    half_angle = math.radians(water.hoh_angle_degree) / 2.0
    poses = []
    waters = []
    for node, oxygen in enumerate(rotated_oxygen):
        if np.linalg.norm(oxygen / semiaxes) >= policy.boundary_scale:
            continue
        if elements:
            distance = np.linalg.norm(canonical - oxygen[None, :], axis=1)
            exclusion = policy.solute_exclusion_scale * (solute_radii + oxygen_radius)
            if np.any(distance < exclusion):
                continue
        donor_vectors = []
        for neighbor in outgoing[node]:
            delta_quarter = quarter[neighbor] - quarter[node]
            delta_quarter = (
                (delta_quarter + period // 2) % period
            ) - period // 2
            vector = (delta_quarter.astype(float) * (lattice / 4.0)) @ lattice_rotation.T
            donor_vectors.append(vector / np.linalg.norm(vector))
        bisector = donor_vectors[0] + donor_vectors[1]
        bisector /= np.linalg.norm(bisector)
        transverse = donor_vectors[0] - donor_vectors[1]
        transverse /= np.linalg.norm(transverse)
        normal = np.cross(bisector, transverse)
        normal /= np.linalg.norm(normal)
        rotation = np.column_stack((bisector, transverse, normal))

        def water_from_frame(frame: np.ndarray) -> np.ndarray:
            local_bisector = frame[:, 0]
            local_transverse = frame[:, 1]
            hydrogen_1 = oxygen + water.oh_distance_angstrom * (
                math.cos(half_angle) * local_bisector
                + math.sin(half_angle) * local_transverse
            )
            hydrogen_2 = oxygen + water.oh_distance_angstrom * (
                math.cos(half_angle) * local_bisector
                - math.sin(half_angle) * local_transverse
            )
            return np.asarray((oxygen, hydrogen_1, hydrogen_2))

        molecule = water_from_frame(rotation)
        if np.max(np.linalg.norm(molecule / semiaxes, axis=1)) >= policy.boundary_scale:
            # A finite cut of an exact periodic ice network has dangling
            # surface bonds. Reorient only those boundary waters inward,
            # retaining the rigid TIP3P-FB geometry and every oxygen site.
            inward = -(oxygen / semiaxes**2)
            inward /= np.linalg.norm(inward)
            tangent = transverse - inward * float(np.dot(inward, transverse))
            if np.linalg.norm(tangent) <= 1.0e-10:
                tangent = normal - inward * float(np.dot(inward, normal))
            tangent /= np.linalg.norm(tangent)
            tangent_normal = np.cross(inward, tangent)
            best_radius = math.inf
            for angle in np.linspace(0.0, math.pi, 16, endpoint=False):
                trial_transverse = (
                    math.cos(angle) * tangent
                    + math.sin(angle) * tangent_normal
                )
                trial_normal = np.cross(inward, trial_transverse)
                trial_rotation = np.column_stack(
                    (inward, trial_transverse, trial_normal)
                )
                trial_molecule = water_from_frame(trial_rotation)
                trial_radius = float(
                    np.max(np.linalg.norm(trial_molecule / semiaxes, axis=1))
                )
                if trial_radius < best_radius:
                    best_radius = trial_radius
                    rotation = trial_rotation
                    molecule = trial_molecule
            if best_radius >= policy.boundary_scale:
                continue
        poses.append(
            np.concatenate((oxygen, rotation_vector_from_matrix(rotation)))
        )
        waters.append(molecule)
    water_count = len(waters)
    return IceShellBuild(
        solute_elements=elements,
        solute_coordinates_angstrom=canonical,
        semiaxes_angstrom=semiaxes,
        shape=shape,
        water_poses_angstrom_radian=np.asarray(poses).reshape(water_count, 6),
        water_coordinates_angstrom=np.asarray(waters).reshape(water_count, 3, 3),
        canonical_rotation=canonical_rotation,
        canonical_origin_angstrom=origin,
        lattice_constant_angstrom=lattice,
        parent_lattice_number_density_angstrom3=parent_density,
        nominal_number_density_angstrom3=water_count / volume,
        ellipsoid_volume_angstrom3=volume,
        fill_volume_angstrom3=volume,
        density_volume_basis="oxygen centers in the full confinement ellipsoid",
        target_molecule_count=target_count,
        density_relative_error=abs(water_count - target_count) / target_count,
        density_compensation_iterations=1,
        seed=int(policy.seed),
    )


__all__ = [
    "ICE_SHELL_SCHEMA",
    "IceShellBuild",
    "IceShellSpecification",
    "MOLECULAR_SOLVENT_LATTICES",
    "MolecularSolventLattice",
    "MolecularSolventShellBuild",
    "WATER_NUMBER_DENSITY_ANGSTROM3",
    "available_molecular_solvent_lattices",
    "build_molecular_solvent_shell",
    "build_tip3p_fb_ice_ellipsoid",
    "build_tip3p_fb_ice_shell",
    "molecular_solvent_lattice_centers",
]
