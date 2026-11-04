"""AO-frame alignment and phase continuity for imported Dyson orbitals."""

from dataclasses import dataclass, replace

import numpy as np

from .dyson import DysonOrbital
from .population import pyscf_molecule_from_electronic_state


@dataclass(frozen=True)
class ElectronicFrameAlignment:
    rotation: np.ndarray
    translation_bohr: np.ndarray
    maximum_geometry_error_bohr: float


def reorient_electronic_state(state, target_coordinates_bohr, *, tolerance_bohr=2e-7):
    """Move a state between rigid frames, rotating its AO tensors as well.

    Ordered atoms must describe the same geometry. The target is the intended
    displaced geometry, not an equilibrium geometry fitted to a deformed one.
    Uses proper-SVD geometry alignment and PySCF AO rotation, qualified by
    overlap-metric preservation and real-space orbital covariance. The original state is preserved. Metadata of the transformation
    is returned separately from the transformed electronic state.
    """
    target = np.asarray(target_coordinates_bohr, float)
    if target.shape != state.coordinates_bohr.shape or not np.all(np.isfinite(target)):
        raise ValueError("Target must contain the same ordered Cartesian atoms")
    if not np.isfinite(tolerance_bohr) or tolerance_bohr <= 0:
        raise ValueError("Frame tolerance must be positive")
    origin = state.coordinates_bohr.mean(axis=0)
    destination = target.mean(axis=0)
    u, _, vt = np.linalg.svd((state.coordinates_bohr - origin).T @ (target - destination))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    translation = destination - origin @ rotation
    coordinates = state.coordinates_bohr @ rotation + translation
    error = float(np.max(np.linalg.norm(coordinates - target, axis=1)))
    if error > tolerance_bohr:
        raise ValueError(
            f"Electronic frame alignment would distort the geometry ({error:.3g} bohr)"
        )
    molecule = pyscf_molecule_from_electronic_state(state)
    ao_rotation = np.asarray(molecule.ao_rotation_matrix(rotation))
    moved = molecule.copy()
    moved.set_geom_(coordinates, unit="Bohr")
    overlap = moved.intor_symmetric("int1e_ovlp")
    if not np.allclose(ao_rotation.T @ overlap @ ao_rotation, state.overlap_ao, atol=2e-7, rtol=0):
        raise ValueError("AO rotation does not preserve the electronic overlap metric")

    def density(array):
        return ao_rotation @ array @ ao_rotation.T

    focks = ()
    if state.fock_matrices_ao:
        inverse = np.linalg.inv(ao_rotation)
        focks = tuple(inverse.T @ array @ inverse for array in state.fock_matrices_ao)
    excited = tuple(
        replace(
            item,
            transition_density_ao=density(item.transition_density_ao),
            difference_density_ao=None
            if item.difference_density_ao is None
            else density(item.difference_density_ao),
            transition_density_spin_ao=tuple(density(a) for a in item.transition_density_spin_ao),
        )
        for item in state.excited_states
    )
    transformed = replace(
        state,
        coordinates_bohr=coordinates,
        overlap_ao=overlap,
        mo_coefficients=tuple(ao_rotation @ c for c in state.mo_coefficients),
        density_matrices_ao=tuple(density(a) for a in state.density_matrices_ao),
        fock_matrices_ao=focks,
        excited_states=excited,
    )
    return transformed, ElectronicFrameAlignment(rotation, translation, error)


def cross_ao_overlap(reference, candidate):
    """Cross-geometry overlap of two APOC states in their current frames."""
    from pyscf import gto

    left = pyscf_molecule_from_electronic_state(reference)
    right = pyscf_molecule_from_electronic_state(candidate)
    return np.asarray(gto.intor_cross("int1e_ovlp", left, right))


@dataclass(frozen=True)
class DysonMatch:
    candidate_index: int
    normalized_overlap: float
    phase: int
    ambiguity_gap: float


def follow_dyson_orbital(
    reference, candidates, *, cross_overlaps=None, minimum_overlap=0.9, ambiguity_margin=0.05
):
    """Select an isolated Dyson channel by normalized AO overlap, retaining P.

    Frames must already agree. The optional cross matrices permit another AO
    backend to supply overlaps. Only the overall sign is changed; neither the
    reference nor candidate pole strength is normalised to unity.
    """
    candidates = tuple(candidates)
    if not candidates or not 0 < minimum_overlap <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError("Require candidates and valid overlap/ambiguity thresholds")
    matrices = (
        tuple(cross_ao_overlap(reference.state, c.state) for c in candidates)
        if cross_overlaps is None
        else tuple(cross_overlaps)
    )
    if len(matrices) != len(candidates):
        raise ValueError("One cross-AO overlap is required per candidate")
    signed = []
    for candidate, matrix in zip(candidates, matrices, strict=True):
        if candidate.spin_channel != reference.spin_channel or not np.array_equal(
            reference.state.atomic_numbers, candidate.state.atomic_numbers
        ):
            raise ValueError("Dyson candidates require the same spin and ordered atoms")
        raw = np.asarray(matrix)
        if (
            np.iscomplexobj(raw)
            or raw.shape != (reference.state.nao, candidate.state.nao)
            or not np.all(np.isfinite(raw))
        ):
            raise ValueError("Invalid cross-AO overlap")
        value = float(reference.coefficients_ao @ raw @ candidate.coefficients_ao) / np.sqrt(
            reference.norm_squared * candidate.norm_squared
        )
        if abs(value) > 1 + 1e-6:
            raise ValueError("Normalized Dyson overlap exceeds one; check AO conventions")
        signed.append(value)
    order = np.argsort(-np.abs(signed), kind="stable")
    index = int(order[0])
    best = abs(signed[index])
    gap = best - (abs(signed[order[1]]) if len(order) > 1 else 0.0)
    if best < minimum_overlap or gap < ambiguity_margin:
        raise ValueError(f"Dyson channel continuity failed (overlap={best:.6g}, gap={gap:.6g})")
    phase = -1 if signed[index] < 0 else 1
    chosen = replace(candidates[index], coefficients_ao=phase * candidates[index].coefficients_ao)
    return chosen, DysonMatch(index, best, phase, gap)


def reorient_dyson_orbital(dyson, target_coordinates_bohr, **kwargs):
    """Rotate a complete typed Dyson and its associated AO electronic state."""
    state, alignment = reorient_electronic_state(dyson.state, target_coordinates_bohr, **kwargs)
    # Reuse the state-frame AO rotation; avoid refitting or renormalising Dyson.
    molecule = pyscf_molecule_from_electronic_state(dyson.state)
    rotation = molecule.ao_rotation_matrix(alignment.rotation)
    moved = replace(dyson, state=state, coefficients_ao=rotation @ dyson.coefficients_ao)
    return moved, alignment
