"""Opt-in angular corrections to vertical closed-shell TDA energies only.

This module does not alter the reference, Hamiltonian, amplitudes, selection,
derivatives or root ordering. Parameters define a separate empirical model;
they are not supplied by the Grimme parametrization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .electronic_state import GFN2OrbitalState
from .sf_tda import HARTREE_TO_EV
from .stda import ClosedShellSTDAResult


TDA_ANGULAR_CORRECTION_MODEL = "TDA-LOCAL-P-ANGULAR-POSTHOC.v1"


def _array(value, *, dtype=float) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("angular-correction arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TDAAngularCorrectionParameters:
    """Two explicit energy endpoints and descriptor-resolution controls.

    ``same_atom`` compares local orientations at common atomic centres.
    ``all_atoms`` compares all pairs of centres, including spatially separated
    donor and acceptor regions. No automatic switch between the two is made.
    """

    sigma_to_pi_star_ev: float
    pi_to_pi_star_ev: float
    pairing: str = "same_atom"
    minimum_p_pair_weight: float = 1.0e-12
    maximum_unresolved_state_weight: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in ("sigma_to_pi_star_ev", "pi_to_pi_star_ev",
                     "minimum_p_pair_weight", "maximum_unresolved_state_weight"):
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)) or not np.isfinite(raw):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(raw))
        if self.pairing not in {"same_atom", "all_atoms"}:
            raise ValueError("angular pairing must be same_atom or all_atoms")
        if self.minimum_p_pair_weight <= 0:
            raise ValueError("minimum_p_pair_weight must be positive")
        if not 0 <= self.maximum_unresolved_state_weight < 1:
            raise ValueError("maximum_unresolved_state_weight must lie in [0, 1)")


@dataclass(frozen=True)
class LocalPOrbitalOrientations:
    """Positive local p-orientation tensors, indexed orbital/atom/x/y.

    Tensor traces are descriptor weights, not atomic charges or probabilities
    of the complete molecular orbital. Multiple radial p shells are summed
    after orthogonalizing the p sector in its own overlap metric.
    """

    orbital_indices: np.ndarray
    tensors: np.ndarray

    def __post_init__(self) -> None:
        indices = _indices(self.orbital_indices, "orientation orbitals")
        tensors = _array(self.tensors)
        if tensors.ndim != 4 or tensors.shape[0] != len(indices) or tensors.shape[2:] != (3, 3):
            raise ValueError("local p tensors must have shape norbitals x natoms x 3 x 3")
        if tensors.shape[1] < 1 or not np.allclose(tensors, tensors.swapaxes(-1, -2), atol=1e-12):
            raise ValueError("local p tensors must be symmetric and contain atoms")
        if np.min(np.linalg.eigvalsh(tensors), initial=0.0) < -1.0e-12:
            raise ValueError("local p orientation tensors must be positive semidefinite")
        object.__setattr__(self, "orbital_indices", indices)
        object.__setattr__(self, "tensors", tensors)

    @property
    def atom_weights(self) -> np.ndarray:
        return _array(np.trace(self.tensors, axis1=-2, axis2=-1))


@dataclass(frozen=True)
class TDAAngularEnergyCorrection:
    """Separate vertical-energy result; state_indices retain input root order.

    Configuration scores with a false ``configuration_resolved`` mask have
    zero placeholders, not an assigned sigma character. Their probability
    mass is reported separately for every returned state.
    """

    parameters: TDAAngularCorrectionParameters
    state_indices: np.ndarray
    original_excitation_energies_hartree: np.ndarray
    corrected_excitation_energies_hartree: np.ndarray
    corrections_ev: np.ndarray
    pi_pi_weights: np.ndarray
    unresolved_state_weights: np.ndarray
    configuration_pi_pi_weights: np.ndarray
    configuration_resolved: np.ndarray
    configuration_p_pair_weights: np.ndarray
    reference_model_id: str

    @property
    def model_id(self) -> str:
        return TDA_ANGULAR_CORRECTION_MODEL

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, TDAAngularCorrectionParameters):
            raise TypeError("explicit angular correction parameters are required")
        if not isinstance(self.reference_model_id, str) or not self.reference_model_id:
            raise ValueError("angular correction requires its reference model identity")
        states = _indices(self.state_indices, "state indices")
        object.__setattr__(self, "state_indices", states)
        for name in ("original_excitation_energies_hartree",
                     "corrected_excitation_energies_hartree", "corrections_ev",
                     "pi_pi_weights", "unresolved_state_weights"):
            values = _array(getattr(self, name))
            if values.shape != states.shape:
                raise ValueError(f"{name} must contain one value per requested state")
            object.__setattr__(self, name, values)
        raw_resolved = np.asarray(self.configuration_resolved)
        if raw_resolved.dtype.kind != "b" or raw_resolved.ndim != 1:
            raise ValueError("configuration resolution mask must be a vector")
        resolved = _array(raw_resolved, dtype=bool)
        object.__setattr__(self, "configuration_resolved", resolved)
        for name in ("configuration_pi_pi_weights", "configuration_p_pair_weights"):
            values = _array(getattr(self, name))
            if values.shape != resolved.shape:
                raise ValueError("configuration diagnostics have inconsistent shapes")
            object.__setattr__(self, name, values)
        if any(np.any((v < 0) | (v > 1)) for v in
               (self.pi_pi_weights, self.unresolved_state_weights, self.configuration_pi_pi_weights)):
            raise ValueError("angular weights must lie in [0, 1]")
        if np.any(self.configuration_p_pair_weights < 0):
            raise ValueError("p-pair descriptor weights must be nonnegative")
        expected = ((1 - self.pi_pi_weights) * self.parameters.sigma_to_pi_star_ev
                    + self.pi_pi_weights * self.parameters.pi_to_pi_star_ev)
        if not np.allclose(self.corrections_ev, expected, atol=1e-12, rtol=1e-12):
            raise ValueError("energy corrections do not match angular endpoint interpolation")
        if not np.allclose(
            self.corrected_excitation_energies_hartree,
            self.original_excitation_energies_hartree + self.corrections_ev / HARTREE_TO_EV,
            atol=1e-12, rtol=1e-12,
        ):
            raise ValueError("corrected vertical energies do not match their reported shifts")


def _indices(values, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw):
        raise ValueError(f"{label} must be a nonempty integer vector")
    if np.any(raw < 0) or len(np.unique(raw)) != len(raw):
        raise ValueError(f"{label} must contain unique nonnegative indices")
    if np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{label} exceeds supported integer indices")
    return _array(raw, dtype=np.int64)


def local_p_orbital_orientations(
    state: GFN2OrbitalState,
    orbital_indices: Sequence[int] | None = None,
) -> LocalPOrbitalOrientations:
    """Build local p-axis second moments without dipoles or a molecular plane.

    Complete px/py/pz radial triplets are identified from serialized primitive
    metadata. The p sector alone is symmetrically orthogonalized, so Cartesian
    d-shell conventions cannot introduce a spurious p-axis rotation. The
    supported spherical-to-Cartesian map must preserve individual p functions
    up to permutation and sign, as the resident GFN2 and wfn.xtb bases do.
    """

    indices = _indices(np.arange(state.nmo) if orbital_indices is None else orbital_indices,
                       "orientation orbitals")
    if np.any(indices >= state.nmo):
        raise ValueError("orientation orbital lies outside the reference")
    basis = state.basis
    rows = np.flatnonzero(np.isin(basis.cartesian_type, (2, 3, 4)))
    tensors = np.zeros((len(indices), len(state.atomic_numbers), 3, 3))
    if not len(rows):
        return LocalPOrbitalOrientations(indices, tensors)
    transform = basis.spherical_to_cartesian[rows]
    if not np.allclose(transform @ transform.T, np.eye(len(rows)), atol=1e-12, rtol=0):
        raise ValueError("p-component map must be orthogonal")
    if np.any(np.count_nonzero(np.abs(transform) > 1e-12, axis=1) != 1):
        raise ValueError("p-component map must preserve individual Cartesian p functions")
    groups: dict[tuple, dict[int, int]] = {}
    for local, row in enumerate(rows):
        start, stop = basis.primitive_offsets[row:row + 2]
        radial = (tuple(basis.primitive_exponents[start:stop]),
                  tuple(basis.primitive_coefficients[start:stop]))
        key = (int(basis.cartesian_atom[row]), radial)
        component = int(basis.cartesian_type[row]) - 2
        group = groups.setdefault(key, {})
        if component in group:
            raise ValueError("ambiguous duplicate radial p component")
        group[component] = local
    if any(set(group) != {0, 1, 2} for group in groups.values()):
        raise ValueError("local angular descriptors require complete radial p triplets")
    overlap = transform @ state.overlap @ transform.T
    values, vectors = np.linalg.eigh(overlap)
    if np.min(values) <= 1.0e-12:
        raise ValueError("p-sector overlap is not positive definite at the descriptor tolerance")
    coefficients = ((vectors * np.sqrt(values)) @ vectors.T) @ transform @ state.coefficients[:, indices]
    for (atom, _radial), group in groups.items():
        local = coefficients[[group[axis] for axis in range(3)]].T
        tensors[:, atom] += np.einsum("ri,rj->rij", local, local)
    return LocalPOrbitalOrientations(indices, tensors)


def correct_tda_vertical_energies(
    state: GFN2OrbitalState,
    excited: ClosedShellSTDAResult,
    *,
    parameters: TDAAngularCorrectionParameters,
    state_indices: Sequence[int] | None = None,
) -> TDAAngularEnergyCorrection:
    """Apply two angular energy endpoints after an unchanged closed-shell TDA.

    For rank-one tensors, the configuration score is the squared scalar
    product of the two unit axes. General tensors give the population-weighted
    average of these squared products. State scores use |X_ia|^2 in the
    supplied canonical configuration basis. Substantial unresolved descriptor
    weight raises an error; tiny excluded tails are reported and the resolved
    weights are renormalized. No gradient or Hessian correction is returned.
    """

    if not isinstance(parameters, TDAAngularCorrectionParameters):
        raise TypeError("explicit TDAAngularCorrectionParameters are required")
    if not isinstance(excited, ClosedShellSTDAResult) or state.reference_kind != "restricted-closed-shell":
        raise ValueError("angular vertical correction requires a restricted closed-shell TDA result")
    if not excited.converged or excited.reference_model_id != state.model_id:
        raise ValueError("TDA result must converge on the supplied reference model")
    energies = np.asarray(excited.excitation_energies_hartree, dtype=float)
    if energies.ndim != 1 or not np.all(np.isfinite(energies)):
        raise ValueError("TDA energies must be a finite vector")
    states = _indices(np.arange(len(energies)) if state_indices is None else state_indices, "state indices")
    if np.any(states >= len(energies)):
        raise ValueError("requested state lies outside the TDA result")
    occupied = np.asarray(excited.occupied_orbitals)
    virtual = np.asarray(excited.virtual_orbitals)
    if occupied.ndim != 1 or virtual.shape != occupied.shape or not len(occupied):
        raise ValueError("TDA configurations must be nonempty occupied/virtual vectors")
    if occupied.dtype.kind not in "iu" or virtual.dtype.kind not in "iu":
        raise ValueError("TDA configuration indices must be integers")
    if np.any(occupied < 0) or np.any(virtual < 0) or max(np.max(occupied), np.max(virtual)) >= state.nmo:
        raise ValueError("TDA configuration orbital lies outside the reference")
    if len(set(zip(occupied, virtual))) != len(occupied):
        raise ValueError("TDA configurations must be unique")
    occupations = state.alpha_occupations + state.beta_occupations
    if np.any(occupations[occupied] < 1.99) or np.any(occupations[virtual] > 0.01):
        raise ValueError("TDA configurations must connect occupied to virtual orbitals")
    amplitudes = np.asarray(excited.amplitudes)
    if amplitudes.shape != (len(energies), len(occupied)) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("TDA amplitude dimensions or values are invalid")
    weights = np.abs(amplitudes[states]) ** 2
    norms = np.sum(weights, axis=1)
    if not np.allclose(norms, 1.0, atol=1.0e-8, rtol=0):
        raise ValueError("TDA amplitudes must be normalized")
    weights = weights / norms[:, None]
    orbitals, mapping = np.unique(np.concatenate((occupied, virtual)), return_inverse=True)
    descriptors = local_p_orbital_orientations(state, orbitals)
    left = descriptors.tensors[mapping[:len(occupied)]]
    right = descriptors.tensors[mapping[len(occupied):]]
    if parameters.pairing == "all_atoms":
        left = np.sum(left, axis=1, keepdims=True)
        right = np.sum(right, axis=1, keepdims=True)
    left_weight = np.trace(left, axis1=-2, axis2=-1)
    right_weight = np.trace(right, axis1=-2, axis2=-1)
    denominator = np.sum(left_weight * right_weight, axis=1)
    numerator = np.einsum("caij,caji->c", left, right)
    resolved = denominator > parameters.minimum_p_pair_weight
    alignment = np.zeros(len(occupied))
    alignment[resolved] = numerator[resolved] / denominator[resolved]
    if np.any(alignment < -1e-10) or np.any(alignment > 1 + 1e-10):
        raise ValueError("local angular alignment lies outside its mathematical bounds")
    alignment = np.clip(alignment, 0.0, 1.0)
    unresolved = np.clip(np.sum(weights[:, ~resolved], axis=1), 0.0, 1.0)
    resolved_norm = np.sum(weights[:, resolved], axis=1)
    if np.any(unresolved > parameters.maximum_unresolved_state_weight) or np.any(resolved_norm <= 0):
        failed = states[(unresolved > parameters.maximum_unresolved_state_weight) | (resolved_norm <= 0)]
        raise ValueError(
            f"angular correction has unresolved p orientation for states {failed.tolist()}; "
            "inspect orbital support and the declared pairing convention"
        )
    pi_pi = np.clip((weights @ alignment) / resolved_norm, 0.0, 1.0)
    delta = (1 - pi_pi) * parameters.sigma_to_pi_star_ev + pi_pi * parameters.pi_to_pi_star_ev
    original = energies[states]
    return TDAAngularEnergyCorrection(
        parameters, states, original, original + delta / HARTREE_TO_EV, delta,
        pi_pi, unresolved, alignment, resolved, denominator, state.model_id,
    )


__all__ = [
    "TDA_ANGULAR_CORRECTION_MODEL", "TDAAngularCorrectionParameters",
    "LocalPOrbitalOrientations", "TDAAngularEnergyCorrection",
    "local_p_orbital_orientations", "correct_tda_vertical_energies",
]
