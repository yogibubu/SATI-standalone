"""Conservative local pi-antibonding diagnostics for vertical postprocessing.

These empirical descriptors qualify candidate acceptors; they are not quantum
numbers or a general orbital classifier. No response matrix or derivative is
changed. Ambiguous states remain available in the returned diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Sequence

import numpy as np

from .electronic_state import GFN2OrbitalState
from .stda import ClosedShellSTDAResult
from .tda_angular_correction import (
    TDAAngularCorrectionParameters, TDAAngularEnergyCorrection, _array, _indices,
    correct_tda_vertical_energies, local_p_orbital_orientations,
)


@dataclass(frozen=True)
class TDAPiStarPolicy:
    """Explicit heuristic thresholds, fixed independently of energy residuals."""

    bond_radius_scale: float = 1.25
    local_plane_eigenvalue_ratio: float = 0.15
    minimum_p_weight: float = 0.05
    minimum_pi_fraction: float = 0.65
    minimum_antibonding_share: float = 0.55
    minimum_antibonding_strength: float = 0.01
    minimum_state_acceptor_weight: float = 0.80

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)) or not np.isscalar(raw) or not np.isfinite(raw):
                raise ValueError(f'{name} must be a finite scalar')
            value = float(raw)
            if value <= 0 or (name != 'bond_radius_scale' and value > 1):
                raise ValueError(f'{name} is outside its supported interval')
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TDAPiStarDiagnostics:
    orbital_indices: np.ndarray
    p_weights: np.ndarray
    pi_fractions: np.ndarray
    antibonding_shares: np.ndarray
    antibonding_strengths: np.ndarray
    accepted_orbitals: np.ndarray
    state_indices: np.ndarray
    state_acceptor_weights: np.ndarray
    eligible_states: np.ndarray
    bond_pairs: np.ndarray
    local_pi_projectors: np.ndarray
    policy: TDAPiStarPolicy

    def __post_init__(self):
        if not isinstance(self.policy, TDAPiStarPolicy):
            raise TypeError('diagnostics require an explicit pi-star policy')
        orbitals = _indices(self.orbital_indices, 'acceptor orbitals')
        states = _indices(self.state_indices, 'diagnostic states')
        object.__setattr__(self, 'orbital_indices', orbitals)
        object.__setattr__(self, 'state_indices', states)
        for name in ('p_weights', 'pi_fractions', 'antibonding_shares', 'antibonding_strengths',
                     'state_acceptor_weights', 'local_pi_projectors', 'bond_pairs',
                     'accepted_orbitals', 'eligible_states'):
            raw = np.asarray(getattr(self, name))
            if name in ('accepted_orbitals', 'eligible_states') and raw.dtype.kind != 'b':
                raise ValueError('acceptor masks must be boolean')
            if name == 'bond_pairs' and raw.dtype.kind not in 'iu':
                raise ValueError('bond pairs must be integers')
            dtype = bool if name in ('accepted_orbitals', 'eligible_states') else (np.int64 if name == 'bond_pairs' else float)
            object.__setattr__(self, name, _array(raw, dtype=dtype))
        for name in ('p_weights', 'pi_fractions', 'antibonding_shares', 'antibonding_strengths', 'accepted_orbitals'):
            if getattr(self, name).shape != orbitals.shape:
                raise ValueError('acceptor diagnostic dimensions are inconsistent')
        if self.state_acceptor_weights.shape != states.shape or self.eligible_states.shape != states.shape:
            raise ValueError('state diagnostic dimensions are inconsistent')
        for name in ('pi_fractions', 'antibonding_shares', 'state_acceptor_weights'):
            value = getattr(self, name)
            if np.any((value < 0) | (value > 1)):
                raise ValueError('diagnostic fractions must lie in [0, 1]')
        if self.bond_pairs.ndim != 2 or self.bond_pairs.shape[1:] != (2,):
            raise ValueError('bond pairs must have shape nbonds x 2')
        if self.local_pi_projectors.ndim != 3 or self.local_pi_projectors.shape[1:] != (3, 3):
            raise ValueError('local projectors must have shape natoms x 3 x 3')
        if np.any(self.p_weights < 0) or np.any(self.antibonding_strengths < 0):
            raise ValueError('descriptor weights and strengths must be nonnegative')
        projectors = self.local_pi_projectors
        if not np.allclose(projectors, projectors.swapaxes(-1, -2), atol=1e-12, rtol=0) or not np.allclose(projectors @ projectors, projectors, atol=1e-12, rtol=0):
            raise ValueError('local pi projectors must be symmetric and idempotent')
        if np.any(self.bond_pairs < 0) or np.any(self.bond_pairs >= len(projectors)):
            raise ValueError('bond atom lies outside the local projectors')


@dataclass(frozen=True)
class TDAPiStarVerticalCorrection:
    """Corrected eligible roots plus diagnostics of every requested root.

    ``correction`` is None when no root qualifies. Ineligible roots receive no
    corrected energy. Their original results remain entirely unchanged.
    """

    diagnostics: TDAPiStarDiagnostics
    correction: TDAAngularEnergyCorrection | None

    def __post_init__(self):
        if not isinstance(self.diagnostics, TDAPiStarDiagnostics):
            raise TypeError('automatic correction requires pi-star diagnostics')
        eligible = self.diagnostics.state_indices[self.diagnostics.eligible_states]
        if self.correction is None:
            if len(eligible):
                raise ValueError('eligible states require their energy correction')
        elif not isinstance(self.correction, TDAAngularEnergyCorrection) or not np.array_equal(self.correction.state_indices, eligible):
            raise ValueError('corrected roots must match eligible diagnostic roots')


def assess_tda_pi_star_acceptors(
    state: GFN2OrbitalState,
    excited: ClosedShellSTDAResult,
    *,
    policy: TDAPiStarPolicy | None = None,
    state_indices: Sequence[int] | None = None,
) -> TDAPiStarDiagnostics:
    """Combine local transverse p support and signed interatomic pi overlap.

    Neighbour directions define a local geometric Gram matrix. Its null/soft
    eigenspace defines the candidate pi subspace, including the two transverse
    directions of a terminal/linear site. Negative orbital overlap populations
    between linked heavy-atom pi subspaces provide antibonding evidence.
    """
    policy = TDAPiStarPolicy() if policy is None else policy
    if not isinstance(policy, TDAPiStarPolicy):
        raise TypeError('policy must be TDAPiStarPolicy')
    if (not isinstance(excited, ClosedShellSTDAResult) or not excited.converged
            or state.reference_kind != 'restricted-closed-shell' or excited.reference_model_id != state.model_id):
        raise ValueError('pi-star diagnostics require a matching converged closed-shell TDA reference')
    energies = np.asarray(excited.excitation_energies_hartree)
    if energies.ndim != 1 or not np.all(np.isfinite(energies)):
        raise ValueError('TDA energies must be a finite vector')
    states = _indices(np.arange(len(energies)) if state_indices is None else state_indices, 'diagnostic states')
    if np.any(states >= len(energies)):
        raise ValueError('requested state lies outside the TDA result')
    virtual = np.asarray(excited.virtual_orbitals)
    occupied = np.asarray(excited.occupied_orbitals)
    if virtual.ndim != 1 or virtual.dtype.kind not in 'iu' or not len(virtual) or occupied.shape != virtual.shape or occupied.dtype.kind not in 'iu':
        raise ValueError('TDA configuration indices must be integer vectors')
    if np.any(virtual < 0) or np.any(virtual >= state.nmo) or np.any(occupied < 0) or np.any(occupied >= state.nmo):
        raise ValueError('TDA orbital lies outside the reference')
    occupations = state.alpha_occupations + state.beta_occupations
    if np.any(occupations[virtual] > .01) or np.any(occupations[occupied] < 1.99):
        raise ValueError('TDA configurations must connect occupied to virtual orbitals')
    amplitudes = np.asarray(excited.amplitudes)
    if amplitudes.shape != (len(energies), len(virtual)) or not np.all(np.isfinite(amplitudes)):
        raise ValueError('TDA amplitudes have invalid dimensions or values')
    weights = np.abs(amplitudes[states])**2
    norms = weights.sum(axis=1)
    if not np.allclose(norms, 1., atol=1e-8, rtol=0):
        raise ValueError('TDA amplitudes must be normalized')
    weights /= norms[:, None]
    orbitals, inverse = np.unique(virtual, return_inverse=True)
    descriptors = local_p_orbital_orientations(state, orbitals)
    count = len(state.atomic_numbers)
    radii_table = np.asarray(json.loads(files('matrix_gfn2').joinpath('data/covalent-radii.json').read_text())['radii_bohr'])
    if np.any(state.atomic_numbers < 1) or np.any(state.atomic_numbers > len(radii_table)):
        raise ValueError('atomic number outside the resident covalent-radius table')
    radii = radii_table[state.atomic_numbers - 1]
    delta = state.coordinates_bohr[:, None] - state.coordinates_bohr[None, :]
    distances = np.linalg.norm(delta, axis=2)
    linked = (distances > 1e-8) & (distances <= policy.bond_radius_scale * (radii[:, None] + radii[None, :]))
    projectors = np.zeros((count, 3, 3))
    for atom in range(count):
        neighbours = np.flatnonzero(linked[atom])
        if not len(neighbours) or state.atomic_numbers[atom] == 1:
            continue
        directions = delta[atom, neighbours] / distances[atom, neighbours, None]
        values, axes = np.linalg.eigh(directions.T @ directions)
        selected = values <= policy.local_plane_eigenvalue_ratio * values[-1]
        projectors[atom] = axes[:, selected] @ axes[:, selected].T
    p_weights = descriptors.atom_weights.sum(axis=1)
    pi_weights = np.einsum('raij,aji->r', descriptors.tensors, projectors)
    pi_fraction = np.clip(np.divide(pi_weights, p_weights, out=np.zeros_like(pi_weights), where=p_weights > 1e-14), 0., 1.)
    basis = state.basis
    rows = np.flatnonzero(np.isin(basis.cartesian_type, [2, 3, 4]))
    transform = basis.spherical_to_cartesian[rows]
    cart_coefficients = transform @ state.coefficients[:, orbitals]
    metric = transform @ state.overlap @ transform.T
    groups = {}
    for local, row in enumerate(rows):
        start, stop = basis.primitive_offsets[row:row + 2]
        radial = (tuple(basis.primitive_exponents[start:stop]), tuple(basis.primitive_coefficients[start:stop]))
        groups.setdefault((int(basis.cartesian_atom[row]), radial), {})[int(basis.cartesian_type[row]) - 2] = local
    by_atom = [[] for _ in range(count)]
    projected = np.zeros_like(cart_coefficients)
    for (atom, _), group in groups.items():
        triplet = [group[axis] for axis in range(3)]
        projected[triplet] = projectors[atom] @ cart_coefficients[triplet]
        by_atom[atom].extend(triplet)
    pairs = np.array([(a, b) for a in range(count) for b in range(a + 1, count)
                      if linked[a, b] and by_atom[a] and by_atom[b]], dtype=np.int64).reshape(-1, 2)
    negative = np.zeros(len(orbitals))
    absolute = np.zeros(len(orbitals))
    for a, b in pairs:
        left, right = np.array(by_atom[a]), np.array(by_atom[b])
        population = 2 * np.einsum('ir,ij,jr->r', projected[left], metric[np.ix_(left, right)], projected[right])
        negative += np.maximum(-population, 0.)
        absolute += np.abs(population)
    share = np.divide(negative, absolute, out=np.zeros_like(negative), where=absolute > 1e-14)
    strength = np.divide(negative, p_weights, out=np.zeros_like(negative), where=p_weights > 1e-14)
    accepted = ((p_weights >= policy.minimum_p_weight) & (pi_fraction >= policy.minimum_pi_fraction)
                & (share >= policy.minimum_antibonding_share) & (strength >= policy.minimum_antibonding_strength))
    state_weight = np.clip(weights @ accepted[inverse].astype(float), 0., 1.)
    return TDAPiStarDiagnostics(orbitals, p_weights, pi_fraction, share, strength, accepted,
                               states, state_weight, state_weight >= policy.minimum_state_acceptor_weight,
                               pairs, projectors, policy)


def correct_pi_star_vertical_energies(
    state: GFN2OrbitalState,
    excited: ClosedShellSTDAResult,
    *,
    parameters: TDAAngularCorrectionParameters,
    policy: TDAPiStarPolicy | None = None,
    state_indices: Sequence[int] | None = None,
) -> TDAPiStarVerticalCorrection:
    """Correct eligible pi-star roots; retain diagnostics for excluded roots."""
    if not isinstance(parameters, TDAAngularCorrectionParameters):
        raise TypeError('explicit angular correction parameters are required')
    diagnostics = assess_tda_pi_star_acceptors(state, excited, policy=policy, state_indices=state_indices)
    selected = diagnostics.state_indices[diagnostics.eligible_states]
    correction = (correct_tda_vertical_energies(state, excited, parameters=parameters, state_indices=selected)
                  if len(selected) else None)
    return TDAPiStarVerticalCorrection(diagnostics, correction)


__all__ = ['TDAPiStarPolicy', 'TDAPiStarDiagnostics', 'TDAPiStarVerticalCorrection',
           'assess_tda_pi_star_acceptors', 'correct_pi_star_vertical_energies']
