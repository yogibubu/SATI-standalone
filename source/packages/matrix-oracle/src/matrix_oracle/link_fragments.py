"""ORACLE-owned capped fragment frames for local electronic preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from matrix_chem.topology.covalent_radii import covalent_radius

from .orbital_regions import OrbitalRegionPlan, QuasiNeutralFragmentProposal


@dataclass(frozen=True)
class LinkEmbeddedFragment:
    real_atoms: tuple[int, ...]
    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: np.ndarray
    link_bonds: tuple[tuple[int, int], ...]
    embedding_atoms: tuple[int, ...]
    embedding_coordinates_angstrom: np.ndarray
    embedding_charges_electron: np.ndarray
    estimated_fragment_charge: float


@dataclass(frozen=True)
class FragmentDerivativePullback:
    active_atoms: tuple[int, ...]
    gradient: np.ndarray
    hessian: np.ndarray | None


def pullback_link_embedded_derivatives(
    frame: LinkEmbeddedFragment,
    parent_coordinates_angstrom: np.ndarray,
    local_gradient: np.ndarray,
    local_hessian: np.ndarray | None = None,
) -> FragmentDerivativePullback:
    """Apply exact coordinate chain rules for real, link, and MM sites.

    Derivatives must include every local QM/link coordinate and every moving
    embedding point, in that order. The embedding charges are held fixed;
    charge-response terms and global interfragment response are separate.
    """
    parent = np.asarray(parent_coordinates_angstrom, dtype=float)
    active_atoms = tuple(sorted(
        set(frame.real_atoms)
        | {atom for bond in frame.link_bonds for atom in bond}
        | set(frame.embedding_atoms)
    ))
    atom_position = {atom: index for index, atom in enumerate(active_atoms)}
    nreal = len(frame.real_atoms)
    nlink = len(frame.link_bonds)
    nlocal = nreal + nlink + len(frame.embedding_atoms)
    gradient = np.asarray(local_gradient, dtype=float)
    hessian = None if local_hessian is None else np.asarray(local_hessian, dtype=float)
    if (
        parent.ndim != 2 or parent.shape[1] != 3
        or max(active_atoms, default=-1) >= len(parent)
        or gradient.shape != (nlocal, 3)
        or (hessian is not None and hessian.shape != (3 * nlocal, 3 * nlocal))
    ):
        raise ValueError("invalid link-fragment derivative dimensions")
    jacobian = np.zeros((3 * nlocal, 3 * len(active_atoms)))
    identity = np.eye(3)
    for local, atom in enumerate(frame.real_atoms):
        jacobian[3 * local:3 * local + 3,
                 3 * atom_position[atom]:3 * atom_position[atom] + 3] = identity
    for offset, atom in enumerate(frame.embedding_atoms):
        local = nreal + nlink + offset
        jacobian[3 * local:3 * local + 3,
                 3 * atom_position[atom]:3 * atom_position[atom] + 3] = identity
    nonlinear = np.zeros((3 * len(active_atoms), 3 * len(active_atoms))) if hessian is not None else None
    for offset, (inside, outside) in enumerate(frame.link_bonds):
        local = nreal + offset
        displacement = parent[outside] - parent[inside]
        distance = float(np.linalg.norm(displacement))
        if distance <= 1.0e-8:
            raise ValueError("zero-length link boundary")
        direction = displacement / distance
        link_length = float(np.linalg.norm(frame.coordinates_angstrom[local] - parent[inside]))
        projection = identity - np.outer(direction, direction)
        outside_jacobian = link_length * projection / distance
        for atom, block in (
            (inside, identity - outside_jacobian),
            (outside, outside_jacobian),
        ):
            jacobian[3 * local:3 * local + 3,
                     3 * atom_position[atom]:3 * atom_position[atom] + 3] = block
        if nonlinear is not None:
            second = np.empty((3, 3, 3))
            for component in range(3):
                for left in range(3):
                    for right in range(3):
                        second[component, left, right] = link_length / distance**2 * (
                            3.0 * direction[component] * direction[left] * direction[right]
                            - identity[component, left] * direction[right]
                            - identity[component, right] * direction[left]
                            - identity[left, right] * direction[component]
                        )
            correction = np.einsum("a,ajk->jk", gradient[local], second)
            for left_atom, left_sign in ((inside, -1.0), (outside, 1.0)):
                for right_atom, right_sign in ((inside, -1.0), (outside, 1.0)):
                    rows = slice(3 * atom_position[left_atom], 3 * atom_position[left_atom] + 3)
                    columns = slice(3 * atom_position[right_atom], 3 * atom_position[right_atom] + 3)
                    nonlinear[rows, columns] += left_sign * right_sign * correction
    pulled_gradient = (jacobian.T @ gradient.ravel()).reshape(-1, 3)
    pulled_hessian = None if hessian is None else jacobian.T @ hessian @ jacobian + nonlinear
    return FragmentDerivativePullback(active_atoms, pulled_gradient, pulled_hessian)


def prepare_link_embedded_fragments(
    plan: OrbitalRegionPlan,
    proposals: tuple[QuasiNeutralFragmentProposal, ...],
    coordinates_angstrom: np.ndarray,
    *,
    embedding_radius_angstrom: float = 12.0,
) -> tuple[LinkEmbeddedFragment, ...]:
    """Cap each cut with H and retain a bounded ORACLE charge environment.

    This prepares local-model inputs. It does not assert equivalence of a
    capped fragment's Hamiltonian to the uncut parent or certify a local SCF.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    natoms = len(plan.atomic_numbers)
    if (
        xyz.shape != (natoms, 3) or not np.all(np.isfinite(xyz))
        or embedding_radius_angstrom <= 0.0
        or len(plan.electronegativity_charges) != natoms
    ):
        raise ValueError("invalid ORACLE link-fragment geometry")
    if sorted(atom for proposal in proposals for atom in proposal.atoms) != list(range(natoms)):
        raise ValueError("ORACLE link-fragment proposals do not partition the atoms")
    charges = np.asarray(plan.electronegativity_charges, dtype=float)
    tree = cKDTree(xyz)
    result = []
    for proposal in proposals:
        atoms = proposal.atoms
        selected = set(atoms)
        link_bonds: list[tuple[int, int]] = []
        link_positions: list[np.ndarray] = []
        boundary_partners: set[int] = set()
        for left, right in proposal.boundary_bonds:
            if (left in selected) == (right in selected):
                raise ValueError("ORACLE fragment boundary does not cross the fragment")
            inside, outside = (left, right) if left in selected else (right, left)
            radius = covalent_radius(plan.atomic_numbers[inside])
            hydrogen_radius = covalent_radius(1)
            if radius is None or hydrogen_radius is None:
                raise ValueError("ORACLE link atom has no covalent-radius parameter")
            direction = xyz[outside] - xyz[inside]
            distance = float(np.linalg.norm(direction))
            if distance <= 1.0e-8:
                raise ValueError("ORACLE boundary bond has zero length")
            link_positions.append(
                xyz[inside] + (radius + hydrogen_radius) * direction / distance
            )
            link_bonds.append((inside, outside))
            boundary_partners.add(outside)
        local_xyz = np.concatenate(
            (xyz[list(atoms)], np.asarray(link_positions, dtype=float).reshape(-1, 3))
        )
        nearby = set().union(*tree.query_ball_point(xyz[list(atoms)], embedding_radius_angstrom))
        embedding_atoms = tuple(sorted(nearby - selected - boundary_partners))
        if link_positions:
            embedding_atoms = tuple(
                atom for atom in embedding_atoms
                if all(np.linalg.norm(xyz[atom] - link) > 1.2 for link in link_positions)
            )
        local_numbers = tuple(plan.atomic_numbers[atom] for atom in atoms) + (1,) * len(link_positions)
        result.append(LinkEmbeddedFragment(
            tuple(atoms), local_numbers, local_xyz,
            tuple(link_bonds), embedding_atoms,
            xyz[list(embedding_atoms)].copy(), charges[list(embedding_atoms)].copy(),
            proposal.estimated_charge,
        ))
    return tuple(result)
