"""Symmetric fragment-ONIOM energy, derivative, and Hessian assembly."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .optimizer import OptimizerCoordinateModel, optimizer_hessian_from_cartesian


FRAGMENT_ONIOM_SCHEMA = "matrix.link.fragment_oniom.v1"
# Gaussian's standard ONIOM hydrogen-link placement: the cap is placed on the
# broken X--Y bond at this fraction of |XY| measured from the explicit atom X.
GAUSSIAN_ONIOM_LINK_SCALE = 0.723


@dataclass(frozen=True)
class CartesianEvaluation:
    """Energy derivatives for one model evaluated at a common geometry."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        hessian = np.asarray(self.hessian_hartree_per_bohr2, dtype=float)
        if hessian.shape != (gradient.size, gradient.size):
            raise ValueError("Cartesian Hessian shape must match the gradient dimension")
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
            raise ValueError("Cartesian evaluation contains non-finite derivatives")
        object.__setattr__(self, "energy_hartree", float(self.energy_hartree))
        object.__setattr__(self, "gradient_hartree_per_bohr", gradient)
        object.__setattr__(self, "hessian_hartree_per_bohr2", 0.5 * (hessian + hessian.T))


@dataclass(frozen=True)
class FragmentModelCorrection:
    """High-minus-low correction for one fragment."""

    identifier: str
    atom_indices: tuple[int, ...]
    high: CartesianEvaluation
    low: CartesianEvaluation
    link_atoms: tuple["HydrogenLinkAtom", ...] = ()


@dataclass(frozen=True)
class HydrogenLinkAtom:
    """One ONIOM hydrogen cap across a covalent fragment boundary.

    ``model_atom_index`` and ``environment_atom_index`` are full-system,
    zero-based indices.  The cap position is the fixed affine interpolation
    ``(1-scale) * r_model + scale * r_environment``.  Using the same mapping
    in both subtractive fragment terms makes analytic gradients and Hessians
    project back to the real atoms without an uncontrolled boundary force.
    """

    model_atom_index: int
    environment_atom_index: int
    scale: float

    def __post_init__(self) -> None:
        model = int(self.model_atom_index)
        environment = int(self.environment_atom_index)
        scale = float(self.scale)
        if model < 0 or environment < 0 or model == environment:
            raise ValueError("hydrogen link atom has invalid boundary indices")
        if not np.isfinite(scale) or not 0.0 < scale < 1.0:
            raise ValueError("hydrogen link atom scale must lie between zero and one")
        object.__setattr__(self, "model_atom_index", model)
        object.__setattr__(self, "environment_atom_index", environment)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True)
class CappedFragmentGeometry:
    """Physical model atoms followed by automatically generated H caps."""

    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    physical_atom_indices: tuple[int, ...]
    link_atoms: tuple[HydrogenLinkAtom, ...]


@dataclass(frozen=True)
class CappedSubtractiveEvaluation:
    """Resident low(real)+high(model)-low(model) energy and gradient."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    link_atom_count: int
    hessian_hartree_per_bohr2: np.ndarray | None = None
    component_results: tuple[object, ...] = ()
    component_labels: tuple[str, str, str] = (
        "low_real",
        "high_model",
        "low_model",
    )


class ResidentCappedSubtractiveSession:
    """Persistent internal ONIOM calculator with analytic H-cap projection."""

    def __init__(
        self,
        atoms: Sequence[str],
        coordinates_angstrom: np.ndarray,
        *,
        model_atom_indices: Sequence[int],
        covalent_bonds: Sequence[tuple[int, int]],
        low_real_calculator: object,
        high_model_calculator: object,
        low_model_calculator: object,
        link_atom_scale: float = GAUSSIAN_ONIOM_LINK_SCALE,
    ) -> None:
        self.link_atom_scale = float(link_atom_scale)
        self.atoms = tuple(str(atom) for atom in atoms)
        self.model_atom_indices = tuple(dict.fromkeys(int(atom) for atom in model_atom_indices))
        self.covalent_bonds = tuple(
            tuple(sorted((int(left), int(right)))) for left, right in covalent_bonds
        )
        initial = build_hydrogen_capped_fragment(
            self.atoms,
            coordinates_angstrom,
            model_atom_indices=self.model_atom_indices,
            covalent_bonds=self.covalent_bonds,
            link_atom_scale=self.link_atom_scale,
        )
        self.capped_atoms = initial.atoms
        self.link_atoms = initial.link_atoms
        self.low_real_calculator = low_real_calculator
        self.high_model_calculator = high_model_calculator
        self.low_model_calculator = low_model_calculator

    def evaluate(self, coordinates_angstrom: np.ndarray, *, compute_hessian: bool = False) -> CappedSubtractiveEvaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        capped = build_hydrogen_capped_fragment(
            self.atoms,
            coordinates,
            model_atom_indices=self.model_atom_indices,
            covalent_bonds=self.covalent_bonds,
            link_atom_scale=self.link_atom_scale,
        )
        if capped.atoms != self.capped_atoms or capped.link_atoms != self.link_atoms:
            raise RuntimeError("resident capped-subtractive boundary topology changed")
        options = {"compute_hessian": True} if compute_hessian else {}
        low_real = self.low_real_calculator.evaluate(coordinates, **options)
        high_model = self.high_model_calculator.evaluate(capped.coordinates_angstrom, **options)
        low_model = self.low_model_calculator.evaluate(capped.coordinates_angstrom, **options)
        projection = _fragment_cartesian_projection(
            self.model_atom_indices,
            self.link_atoms,
            natoms=len(self.atoms),
        )
        local_correction = (
            np.asarray(high_model.gradient_hartree_per_bohr, dtype=float).reshape(-1)
            - np.asarray(low_model.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        )
        if local_correction.size != projection.shape[0]:
            raise ValueError("capped model calculator gradient has the wrong dimension")
        gradient = np.asarray(
            low_real.gradient_hartree_per_bohr, dtype=float
        ).reshape(-1)
        if gradient.size != projection.shape[1]:
            raise ValueError("real-system calculator gradient has the wrong dimension")
        hessian = None
        if compute_hessian:
            matrices = [getattr(item, "hessian_hartree_per_bohr2", None)
                        for item in (low_real, high_model, low_model)]
            if any(item is None for item in matrices):
                raise RuntimeError("every QM/QM component must supply an analytic Hessian")
            real_h, high_h, low_h = (np.asarray(item, dtype=float) for item in matrices)
            if real_h.shape != (projection.shape[1],) * 2 or any(
                item.shape != (projection.shape[0],) * 2 for item in (high_h, low_h)
            ):
                raise ValueError("QM/QM component Hessian has wrong dimension")
            hessian = real_h + projection.T @ (high_h - low_h) @ projection
            if not np.all(np.isfinite(hessian)):
                raise ValueError("QM/QM Hessian contains non-finite values")
        if not np.all(np.isfinite(gradient + projection.T @ local_correction)):
            raise ValueError("QM/QM gradient contains non-finite values")
        return CappedSubtractiveEvaluation(
            hessian_hartree_per_bohr2=hessian,
            component_results=(low_real, high_model, low_model),
            energy_hartree=(
                float(low_real.energy_hartree)
                + float(high_model.energy_hartree)
                - float(low_model.energy_hartree)
            ),
            gradient_hartree_per_bohr=gradient + projection.T @ local_correction,
            link_atom_count=len(self.link_atoms),
        )

    def accept(self, component_results) -> None:
        """Advance all three references using the explicitly accepted evaluation."""
        if component_results is None or len(component_results) != 3:
            raise ValueError("accepted QM/QM point lacks its three component results")
        calculators = (self.low_real_calculator, self.high_model_calculator, self.low_model_calculator)
        for calculator, result in zip(calculators, component_results, strict=True):
            if hasattr(calculator, "validate_accept"):
                calculator.validate_accept(result)
        for calculator, result in zip(calculators, component_results, strict=True):
            if hasattr(calculator, "accept"):
                calculator.accept(result)


def build_hydrogen_capped_fragment(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    model_atom_indices: Sequence[int],
    covalent_bonds: Sequence[tuple[int, int]],
    link_atom_scale: float = GAUSSIAN_ONIOM_LINK_SCALE,
) -> CappedFragmentGeometry:
    """Close cut bonds with Gaussian-style hydrogen link atoms.

    ``link_atom_scale`` is the fraction of the broken X--Y bond measured from
    the retained explicit atom X.  The default 0.723 is Gaussian's standard
    ONIOM placement; exposing it keeps alternative calibrated caps possible
    without changing the boundary projection contract.
    """

    symbols = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(symbols), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("capped fragment coordinates must be finite natoms by 3")
    model = tuple(dict.fromkeys(int(atom) for atom in model_atom_indices))
    model_set = set(model)
    if not model or min(model) < 0 or max(model) >= len(symbols):
        raise ValueError("capped fragment model atom indices are invalid")
    scale = float(link_atom_scale)
    if not np.isfinite(scale) or not 0.0 < scale < 1.0:
        raise ValueError("link_atom_scale must lie strictly between zero and one")
    links: list[HydrogenLinkAtom] = []
    for raw_left, raw_right in covalent_bonds:
        left, right = int(raw_left), int(raw_right)
        if (left in model_set) == (right in model_set):
            continue
        center, environment = (left, right) if left in model_set else (right, left)
        if min(center, environment) < 0 or max(center, environment) >= len(symbols):
            raise ValueError("covalent boundary bond index is outside the system")
        links.append(HydrogenLinkAtom(center, environment, scale))
    links.sort(key=lambda item: (item.model_atom_index, item.environment_atom_index))
    local_coordinates = [coordinates[atom] for atom in model]
    for link in links:
        local_coordinates.append(
            (1.0 - link.scale) * coordinates[link.model_atom_index]
            + link.scale * coordinates[link.environment_atom_index]
        )
    return CappedFragmentGeometry(
        atoms=tuple(symbols[atom] for atom in model) + ("H",) * len(links),
        coordinates_angstrom=np.asarray(local_coordinates, dtype=float),
        physical_atom_indices=model,
        link_atoms=tuple(links),
    )


@dataclass(frozen=True)
class FragmentOniomEvaluation:
    """Composite full-system energy and Cartesian derivatives."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    fragment_identifiers: tuple[str, ...]
    schema: str = FRAGMENT_ONIOM_SCHEMA


def assemble_fragment_oniom_evaluation(
    low_full: CartesianEvaluation,
    corrections: Sequence[FragmentModelCorrection],
    *,
    natoms: int,
    require_partition: bool = True,
) -> FragmentOniomEvaluation:
    """Assemble E_L(full) + sum_i [E_H(F_i) - E_L(F_i)] and its derivatives.

    Atom indices are zero-based. Fragment high/low derivatives are local
    Cartesian arrays ordered as the fragment atom list. All terms must be
    evaluated at the same full-system geometry.
    """

    ncart = 3 * int(natoms)
    if natoms < 1 or low_full.gradient_hartree_per_bohr.size != ncart:
        raise ValueError("low-full evaluation does not match natoms")
    energy = low_full.energy_hartree
    gradient = low_full.gradient_hartree_per_bohr.copy()
    hessian = low_full.hessian_hartree_per_bohr2.copy()
    assigned: set[int] = set()
    identifiers: list[str] = []
    for correction in corrections:
        atoms = tuple(int(index) for index in correction.atom_indices)
        if not correction.identifier:
            raise ValueError("fragment correction needs an identifier")
        if not atoms or len(set(atoms)) != len(atoms):
            raise ValueError(f"fragment {correction.identifier} has invalid atom indices")
        if min(atoms) < 0 or max(atoms) >= natoms:
            raise ValueError(f"fragment {correction.identifier} atom index is outside the system")
        overlap = assigned.intersection(atoms)
        if overlap:
            raise ValueError("fragment corrections must not overlap")
        assigned.update(atoms)
        local_ncart = 3 * (len(atoms) + len(correction.link_atoms))
        if correction.high.gradient_hartree_per_bohr.size != local_ncart:
            raise ValueError(f"high evaluation for {correction.identifier} has wrong dimension")
        if correction.low.gradient_hartree_per_bohr.size != local_ncart:
            raise ValueError(f"low evaluation for {correction.identifier} has wrong dimension")
        projection = _fragment_cartesian_projection(
            atoms,
            correction.link_atoms,
            natoms=natoms,
        )
        energy += correction.high.energy_hartree - correction.low.energy_hartree
        local_gradient = (
            correction.high.gradient_hartree_per_bohr
            - correction.low.gradient_hartree_per_bohr
        )
        local_hessian = (
            correction.high.hessian_hartree_per_bohr2
            - correction.low.hessian_hartree_per_bohr2
        )
        gradient += projection.T @ local_gradient
        hessian += projection.T @ local_hessian @ projection
        identifiers.append(correction.identifier)
    if require_partition and assigned != set(range(natoms)):
        missing = sorted(set(range(natoms)) - assigned)
        raise ValueError(f"fragment corrections do not cover atoms: {missing}")
    hessian = 0.5 * (hessian + hessian.T)
    return FragmentOniomEvaluation(
        energy_hartree=float(energy),
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        fragment_identifiers=tuple(identifiers),
    )


def _fragment_cartesian_projection(
    physical_atoms: Sequence[int],
    link_atoms: Sequence[HydrogenLinkAtom],
    *,
    natoms: int,
) -> np.ndarray:
    """Return dx_local/dx_full for physical atoms plus affine H caps."""

    physical = tuple(int(atom) for atom in physical_atoms)
    local_atom_count = len(physical) + len(link_atoms)
    projection = np.zeros((3 * local_atom_count, 3 * int(natoms)), dtype=float)
    physical_set = set(physical)
    for local, atom in enumerate(physical):
        projection[3 * local : 3 * local + 3, 3 * atom : 3 * atom + 3] = np.eye(3)
    for offset, link in enumerate(link_atoms, start=len(physical)):
        if link.model_atom_index not in physical_set:
            raise ValueError("link-atom model center is absent from the fragment")
        if link.environment_atom_index in physical_set:
            raise ValueError("link-atom environment must lie outside the fragment")
        if link.environment_atom_index >= natoms:
            raise ValueError("link-atom environment index is outside the system")
        rows = slice(3 * offset, 3 * offset + 3)
        center = slice(3 * link.model_atom_index, 3 * link.model_atom_index + 3)
        environment = slice(
            3 * link.environment_atom_index,
            3 * link.environment_atom_index + 3,
        )
        projection[rows, center] = (1.0 - link.scale) * np.eye(3)
        projection[rows, environment] = link.scale * np.eye(3)
    return projection


def assemble_fragment_oniom_from_xyzin(
    xyzin_path: Path | str,
    low_full: CartesianEvaluation,
    *,
    high_by_fragment: dict[str, CartesianEvaluation],
    low_by_fragment: dict[str, CartesianEvaluation],
    require_electronic_states: bool = True,
) -> FragmentOniomEvaluation:
    """Assemble the composite derivatives using the shared #FRAGMENTS contract."""

    from matrix_chem import read_xyzin_geometry
    from matrix_fragments import read_fragment_records

    target = Path(xyzin_path)
    geometry = read_xyzin_geometry(target)
    fragments = read_fragment_records(target)
    if not fragments:
        raise ValueError("fragment ONIOM requires a built #FRAGMENTS section")
    identifiers = {fragment.identifier for fragment in fragments}
    if set(high_by_fragment) != identifiers or set(low_by_fragment) != identifiers:
        raise ValueError("high/low fragment evaluations must exactly match #FRAGMENTS")
    corrections: list[FragmentModelCorrection] = []
    for fragment in fragments:
        if require_electronic_states and (
            fragment.charge is None or fragment.multiplicity is None
        ):
            raise ValueError(
                f"fragment {fragment.identifier} needs explicit CHARGE and MULTIPLICITY"
            )
        corrections.append(
            FragmentModelCorrection(
                identifier=fragment.identifier,
                atom_indices=tuple(atom - 1 for atom in fragment.atoms),
                high=high_by_fragment[fragment.identifier],
                low=low_by_fragment[fragment.identifier],
            )
        )
    return assemble_fragment_oniom_evaluation(
        low_full,
        corrections,
        natoms=len(geometry.atoms),
        require_partition=True,
    )


def fragment_oniom_optimizer_hessian(
    evaluation: FragmentOniomEvaluation,
    model: OptimizerCoordinateModel,
) -> np.ndarray:
    """Transform the rigorously assembled Cartesian Hessian to active SONIC."""

    return optimizer_hessian_from_cartesian(evaluation.hessian_hartree_per_bohr2, model)


def assemble_fragment_optimizer_hessian_blocks(
    base_hessian: np.ndarray,
    full_labels: Sequence[str],
    fragment_blocks: Sequence[tuple[Sequence[str], np.ndarray]],
) -> np.ndarray:
    """Replace intrafragment blocks in a full SONIC Hessian by staged results."""

    labels = tuple(str(label) for label in full_labels)
    matrix = np.asarray(base_hessian, dtype=float).copy()
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("base Hessian shape does not match full SONIC labels")
    positions = {label: index for index, label in enumerate(labels)}
    assigned: set[int] = set()
    for block_labels, block_hessian in fragment_blocks:
        names = tuple(str(label) for label in block_labels)
        try:
            indices = [positions[label] for label in names]
        except KeyError as exc:
            raise ValueError(f"fragment Hessian label is absent from full SONIC: {exc}") from exc
        if assigned.intersection(indices):
            raise ValueError("fragment Hessian blocks overlap")
        assigned.update(indices)
        block = np.asarray(block_hessian, dtype=float)
        if block.shape != (len(indices), len(indices)):
            raise ValueError("fragment Hessian block shape does not match its labels")
        matrix[np.ix_(indices, indices)] = 0.5 * (block + block.T)
    return 0.5 * (matrix + matrix.T)


def symmetric_two_oniom_hessian_seed(
    fragment_one_oniom: np.ndarray,
    fragment_two_oniom: np.ndarray,
    *,
    fragment_one_atoms: Sequence[int],
    fragment_two_atoms: Sequence[int],
    natoms: int,
) -> np.ndarray:
    """Build the user's symmetric two-job seed without halving QM blocks.

    The F1--F1 block is taken from the job with F1 high, the F2--F2 block from
    the job with F2 high, and the intermolecular F1--F2 block is the arithmetic
    mean of the two jobs. This is a Hessian *seed*, not the derivative of the
    composite inclusion--exclusion energy.
    """

    shape = (3 * int(natoms), 3 * int(natoms))
    first = np.asarray(fragment_one_oniom, dtype=float)
    second = np.asarray(fragment_two_oniom, dtype=float)
    if first.shape != shape or second.shape != shape:
        raise ValueError(f"two-job Hessians must both have shape {shape}")
    atoms_one = tuple(int(index) for index in fragment_one_atoms)
    atoms_two = tuple(int(index) for index in fragment_two_atoms)
    if set(atoms_one).intersection(atoms_two):
        raise ValueError("two-job fragment atom sets overlap")
    if set(atoms_one).union(atoms_two) != set(range(natoms)):
        raise ValueError("two-job fragment atom sets must partition the system")
    one = _cartesian_indices(atoms_one)
    two = _cartesian_indices(atoms_two)
    seed = np.zeros(shape, dtype=float)
    seed[np.ix_(one, one)] = first[np.ix_(one, one)]
    seed[np.ix_(two, two)] = second[np.ix_(two, two)]
    cross = 0.5 * (
        first[np.ix_(one, two)]
        + second[np.ix_(one, two)]
    )
    seed[np.ix_(one, two)] = cross
    seed[np.ix_(two, one)] = cross.T
    return 0.5 * (seed + seed.T)


def align_cartesian_hessian_to_reference(
    hessian_hartree_per_bohr2: np.ndarray,
    moving_coordinates_angstrom: np.ndarray,
    reference_coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    """Rotate a full Cartesian Hessian into a reference geometry frame."""

    from matrix_chem import (
        kabsch_rotation,
        rotate_cartesian_derivatives,
    )

    moving = np.asarray(moving_coordinates_angstrom, dtype=float)
    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("moving and reference geometries must have equal natoms x 3 shape")
    rotation = kabsch_rotation(moving, reference)
    zero_gradient = np.zeros(moving.size, dtype=float)
    _gradient, rotated = rotate_cartesian_derivatives(
        zero_gradient,
        rotation,
        np.asarray(hessian_hartree_per_bohr2, dtype=float),
    )
    assert rotated is not None
    return rotated


def symmetric_two_oniom_geometry_seed(
    reference_coordinates_angstrom: np.ndarray,
    fragment_one_high_coordinates_angstrom: np.ndarray,
    fragment_two_high_coordinates_angstrom: np.ndarray,
    *,
    fragment_one_atoms: Sequence[int],
    fragment_two_atoms: Sequence[int],
) -> np.ndarray:
    """Combine two optimized ONIOM geometries in one common rigid frame.

    Both full geometries are first Kabsch-aligned to the reference. Internal
    shape F1 is taken from the F1-high job and shape F2 from the F2-high job.
    For each fragment, its center is the arithmetic midpoint of the two jobs
    and its rigid orientation is their SO(3) geodesic midpoint. Thus the
    intermolecular pose is averaged only after both jobs share one frame.
    """

    from matrix_chem import (
        kabsch_align,
        kabsch_rotation,
        rotation_matrix_from_vector,
        rotation_vector_from_matrix,
    )

    reference = np.asarray(reference_coordinates_angstrom, dtype=float)
    first_raw = np.asarray(fragment_one_high_coordinates_angstrom, dtype=float)
    second_raw = np.asarray(fragment_two_high_coordinates_angstrom, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference coordinates must have shape natoms x 3")
    if first_raw.shape != reference.shape or second_raw.shape != reference.shape:
        raise ValueError("two ONIOM geometries must match the reference shape")
    first = kabsch_align(first_raw, reference)
    second = kabsch_align(second_raw, reference)
    groups = (
        (tuple(int(index) for index in fragment_one_atoms), first),
        (tuple(int(index) for index in fragment_two_atoms), second),
    )
    if set(groups[0][0]).intersection(groups[1][0]):
        raise ValueError("two-job fragment atom sets overlap")
    if set(groups[0][0]).union(groups[1][0]) != set(range(reference.shape[0])):
        raise ValueError("two-job fragment atom sets must partition the system")
    seed = np.zeros_like(reference)
    for atoms, selected_geometry in groups:
        indices = np.asarray(atoms, dtype=int)
        ref_fragment = reference[indices]
        first_fragment = first[indices]
        second_fragment = second[indices]
        rotation_first = kabsch_rotation(ref_fragment, first_fragment)
        rotation_second = kabsch_rotation(ref_fragment, second_fragment)
        relative = rotation_first.T @ rotation_second
        midpoint_rotation = rotation_first @ rotation_matrix_from_vector(
            0.5 * rotation_vector_from_matrix(relative)
        )
        selected_rotation = kabsch_rotation(ref_fragment, selected_geometry[indices])
        intrinsic = (
            selected_geometry[indices] - np.mean(selected_geometry[indices], axis=0)
        ) @ selected_rotation.T
        midpoint_center = 0.5 * (
            np.mean(first_fragment, axis=0) + np.mean(second_fragment, axis=0)
        )
        seed[indices] = intrinsic @ midpoint_rotation + midpoint_center
    return seed


def _cartesian_indices(atom_indices: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [3 * atom + component for atom in atom_indices for component in range(3)],
        dtype=int,
    )
