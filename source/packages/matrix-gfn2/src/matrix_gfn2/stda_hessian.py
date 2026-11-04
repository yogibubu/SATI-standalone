"""Molecular analytic Hessians of closed-shell resident sTDA-xTB roots.

The selected parent/primary/secondary configuration sets are held fixed,
including the analytic second derivative of rejected-space downfolding.
"""

from __future__ import annotations
import numpy as np
from ._mixed_jet import MixedJet as Jet, stack, contract
from .molecular_second_response import MolecularSecondResponse
from .sf_tda import HARTREE_TO_EV, SFTDAResult, grimme_stda_atomic_hardnesses
from .sf_tda_hessian import SFTDAHessianOperator
from .stda import grimme_xtb_stda, closed_shell_stda_matrix_derivative


def _square_root(S):
    values, vectors = np.linalg.eigh(S.v)
    if min(values) <= 0:
        raise ValueError("positive overlap required for analytic Lowdin response")
    roots = np.sqrt(values)
    denominator = roots[:, None] + roots[None, :]

    def solve(rhs):
        return vectors @ ((vectors.T @ rhs @ vectors) / denominator) @ vectors.T

    root = (vectors * roots) @ vectors.T
    first = np.stack([solve(x) for x in S.a])
    tangent = solve(S.b)
    mixed = np.stack(
        [solve(x - first[k] @ tangent - tangent @ first[k]) for k, x in enumerate(S.ab)]
    )
    return Jet(root, first, tangent, mixed)


def _matrix(state, result, response, spectral_range_ev, maximum_configuration_energy_ev):
    S, C, e = response.overlap, response.coefficients, response.energies
    D = S.n
    occupation = state.alpha_occupations + state.beta_occupations
    occupied_all = np.flatnonzero(occupation > 1.99)
    virtual_all = np.flatnonzero(occupation < 0.01)
    width = 2 * (1 + 0.8 * 0.5) * spectral_range_ev / HARTREE_TO_EV
    occupied = occupied_all[
        state.orbital_energies_hartree[occupied_all]
        > min(state.orbital_energies_hartree[virtual_all]) - width
    ]
    virtual = virtual_all[
        state.orbital_energies_hartree[virtual_all]
        < max(state.orbital_energies_hartree[occupied_all]) + width
    ]
    if result.selection is not None:
        occupied = np.asarray(result.selection.occupied, dtype=int)
        virtual = np.asarray(result.selection.virtual, dtype=int)
    ii = np.repeat(occupied, len(virtual))
    aa = np.tile(virtual, len(occupied))
    lookup = {(int(i), int(a)): k for k, (i, a) in enumerate(zip(ii, aa))}
    selected = np.array(
        [
            lookup[(int(i), int(a))]
            for i, a in zip(result.occupied_orbitals, result.virtual_orbitals)
        ]
    )
    lowdin = _square_root(S) @ C
    qia = stack(
        [
            contract(
                "mi,mi->i",
                lowdin[state.ao_atom == atom][:, ii],
                lowdin[state.ao_atom == atom][:, aa],
            )
            for atom in range(len(state.atomic_numbers))
        ],
        axis=1,
    )
    qij = stack(
        [
            contract(
                "mi,mj->ij",
                lowdin[state.ao_atom == atom][:, occupied],
                lowdin[state.ao_atom == atom][:, occupied],
            )
            for atom in range(len(state.atomic_numbers))
        ],
        axis=2,
    )
    qab = stack(
        [
            contract(
                "mi,mj->ij",
                lowdin[state.ao_atom == atom][:, virtual],
                lowdin[state.ao_atom == atom][:, virtual],
            )
            for atom in range(len(state.atomic_numbers))
        ],
        axis=2,
    )
    hardness = grimme_stda_atomic_hardnesses()[state.atomic_numbers - 1]
    average = 0.5 * (hardness[:, None] + hardness[None, :])
    # R derivatives are exact identities; the mixed coordinate derivative is zero.
    r = Jet(
        state.coordinates_bohr,
        np.eye(D).reshape(D, *state.coordinates_bohr.shape),
        response.coordinate_direction,
    )
    distance2 = ((r[:, None, :] - r[None, :, :]) ** 2).sum(axis=2)
    gamma_j = (distance2**2 + (0.5 * average) ** -4) ** -0.25
    gamma_k = (distance2 + average**-2) ** -0.5
    exchange = 2 * contract("xA,AB,yB->xy", qia, gamma_k, qia)
    coulomb = contract("ijA,AB,abB->iajb", qij, gamma_j, qab).reshape(len(ii), len(ii))
    response_matrix = exchange - coulomb
    ratio = contract("xA,AB,xB->x", qia, gamma_k, qia) / (0.1 / HARTREE_TO_EV)
    shift = (0.5 / HARTREE_TO_EV) / (1 + ratio**4)
    diagonal = e[aa] - e[ii] + response_matrix[np.arange(len(ii)), np.arange(len(ii))] + shift
    primary = selected[: result.primary_count]
    included = selected[result.primary_count :]
    upper = (
        np.inf
        if maximum_configuration_energy_ev is None
        else maximum_configuration_energy_ev / HARTREE_TO_EV
    )
    candidates = np.flatnonzero(
        (diagonal.v > spectral_range_ev / HARTREE_TO_EV) & (diagonal.v < upper)
    )
    rejected = candidates[~np.isin(candidates, included)]
    # Differentiate the same frozen downfolding space as energy and gradient.
    if result.selection is not None:
        rejected = np.asarray([lookup[pair] for pair in result.selection.rejected_pairs], dtype=int)
    diagonal_primary = diagonal[primary]
    if len(rejected):
        coupling = response_matrix[rejected[:, None], primary[None, :]]
        denominator = diagonal[rejected, None] - diagonal[primary][None, :] + 1e-10
        diagonal_primary = diagonal_primary - (coupling**2 / denominator).sum(axis=0)
    matrix = response_matrix[selected[:, None], selected[None, :]]
    for k in range(len(selected)):
        matrix[k, k] = diagonal_primary[k] if k < len(primary) else diagonal[selected[k]]
    return matrix


class _DenseOperator:
    def __init__(self, matrix):
        self.matrix = matrix
        self.size = len(matrix)
        self.diagonal_hartree = np.diag(matrix).copy()

    def apply(self, value):
        return self.matrix @ value


def _check_roots(matrix, result):
    residual = (
        matrix @ result.amplitudes.T - result.amplitudes.T * result.excitation_energies_hartree
    )
    if not np.all(np.isfinite(residual)) or np.max(np.linalg.norm(residual, axis=0)) > 2e-7:
        raise ValueError(
            "TDA roots do not match the differentiated Hamiltonian; posterior energy shifts are not derivative inputs"
        )


class MolecularSTDAHessian:
    """Complete total-state Hessian action plus inspectable analytic response."""

    derivative_route = "analytic molecular mixed CPSCF and TDA eigenvector response"

    def __init__(
        self,
        calculator,
        state,
        *,
        excited=None,
        root=0,
        roots=6,
        spectral_range_ev=7.0,
        maximum_configuration_energy_ev=None,
        response_tolerance=1e-10,
    ):
        occupation = state.alpha_occupations + state.beta_occupations
        if np.any((occupation > 1e-8) & (abs(occupation - 2) > 1e-8)):
            raise ValueError("closed-shell sTDA Hessian requires a closed-shell reference")
        self.state = state
        self.root = int(root)
        self.spectral_range_ev = float(spectral_range_ev)
        self.maximum_configuration_energy_ev = maximum_configuration_energy_ev
        self.excited = (
            excited
            if excited is not None
            else grimme_xtb_stda(
                state,
                roots=max(roots, root + 1),
                spectral_range_ev=spectral_range_ev,
                maximum_configuration_energy_ev=maximum_configuration_energy_ev,
            )
        )
        from dataclasses import replace

        frozen = {}
        for name in (
            "excitation_energies_hartree",
            "amplitudes",
            "occupied_orbitals",
            "virtual_orbitals",
            "residual_norms_hartree",
        ):
            value = np.array(getattr(self.excited, name), copy=True)
            value.setflags(write=False)
            frozen[name] = value
        self.excited = replace(self.excited, **frozen)
        if not self.excited.converged:
            raise ValueError("analytic Hessian requires converged TDA roots")
        self.molecular_response = MolecularSecondResponse(calculator, state)
        first = self.molecular_response.first
        ds, _ = calculator.ground_state_derivative_matrices(state).validated(state)
        self.first_matrix = closed_shell_stda_matrix_derivative(
            state,
            self.excited,
            ds,
            first.coefficient_derivative,
            first.orbital_energy_derivative,
            spectral_range_ev=spectral_range_ev,
            maximum_configuration_energy_ev=maximum_configuration_energy_ev,
        )
        _check_roots(self.first_matrix.matrix_hartree, self.excited)
        proxy = SFTDAResult(
            self.excited.excitation_energies_hartree,
            self.excited.amplitudes,
            self.excited.residual_norms_hartree,
            iterations=0,
            converged=self.excited.converged,
        )
        self.diagnostics = []
        self.operator = SFTDAHessianOperator(
            _DenseOperator(self.first_matrix.matrix_hartree),
            proxy,
            self.first_matrix.derivative_hartree_per_bohr,
            self.matrix_second_derivative_action,
            root=root,
            ground_state_hessian_operator=calculator.ground_state_hessian_operator(state),
            response_tolerance=response_tolerance,
        )

    @property
    def shape(self):
        return self.operator.shape

    @property
    def excitation_energy_hartree(self):
        return float(self.excited.excitation_energies_hartree[self.root])

    @property
    def excitation_gradient_hartree_per_bohr(self):
        root = self.operator.root_amplitude
        return np.einsum(
            "i,qij,j->q", root, self.operator.matrix_first_derivative, root, optimize=True
        ).reshape(self.state.coordinates_bohr.shape)

    def matrix_second_derivative_action(self, direction):
        v = np.asarray(direction, dtype=float).reshape(self.state.coordinates_bohr.shape)
        orbital = self.molecular_response.direction(v)
        # Keep the coordinate tangent explicit rather than reconstructing it
        # from an overlap derivative, which has a translation null space.
        from types import SimpleNamespace

        response = SimpleNamespace(
            overlap=orbital.overlap,
            coefficients=orbital.coefficients,
            energies=orbital.energies,
            coordinate_direction=v,
        )
        matrix = _matrix(
            self.state,
            self.excited,
            response,
            self.spectral_range_ev,
            self.maximum_configuration_energy_ev,
        )
        primal = float(np.max(abs(matrix.v - self.first_matrix.matrix_hartree)))
        first = float(
            np.max(
                abs(
                    matrix.a - self.first_matrix.derivative_hartree_per_bohr.reshape(matrix.a.shape)
                )
            )
        )
        if primal > 2e-9 or first > 2e-8:
            raise RuntimeError(
                f"mixed TDA matrix differs from its first-derivative reference: {primal:g}, {first:g}"
            )
        self.diagnostics.append(
            dict(
                fock_primal_error=orbital.fock_primal_error,
                fock_first_error=orbital.fock_first_error,
                mixed_eigen_residual=orbital.eigen_residual,
                mixed_normalization_residual=orbital.normalization_residual,
                tda_primal_error=primal,
                tda_first_error=first,
            )
        )
        return matrix.ab

    def matvec(self, direction):
        return self.operator.matvec(np.asarray(direction).reshape(-1))

    def excitation_matvec(self, direction):
        return self.operator.excitation_matvec(np.asarray(direction).reshape(-1))

    def materialize(self, *, symmetrize=True):
        return self.operator.materialize(symmetrize=symmetrize)

    def project(self, basis):
        return self.operator.project(basis)


def _spin_flip_matrix(state, selected, policy, orbital, direction):
    from .sf_tda import GRIMME_XTB_SF_STDA

    S, C, e = orbital.overlap, orbital.coefficients, orbital.energies
    D = S.n
    lowdin = _square_root(S) @ C
    occupied = selected.parent.occupied_alpha
    virtual = selected.parent.virtual_beta
    natoms = len(state.atomic_numbers)
    qij = stack(
        [
            contract(
                "mi,mj->ij",
                lowdin[state.ao_atom == atom][:, occupied],
                lowdin[state.ao_atom == atom][:, occupied],
            )
            for atom in range(natoms)
        ],
        axis=2,
    )
    qab = stack(
        [
            contract(
                "mi,mj->ij",
                lowdin[state.ao_atom == atom][:, virtual],
                lowdin[state.ao_atom == atom][:, virtual],
            )
            for atom in range(natoms)
        ],
        axis=2,
    )
    hardness = grimme_stda_atomic_hardnesses()[state.atomic_numbers - 1]
    screened = GRIMME_XTB_SF_STDA.hardness_scale * 0.5 * (hardness[:, None] + hardness[None, :])
    r = Jet(state.coordinates_bohr, np.eye(D).reshape(D, *state.coordinates_bohr.shape), direction)
    distance2 = ((r[:, None, :] - r[None, :, :]) ** 2).sum(axis=2)
    diagonal_atom = np.arange(natoms)
    distance2[diagonal_atom, diagonal_atom] = 1.0
    gamma = (distance2**1.5 + screened**-3) ** (-1 / 3)
    gamma[diagonal_atom, diagonal_atom] = np.diag(screened)
    response = -contract("ijA,AB,abB->iajb", qij, gamma, qab).reshape(
        selected.parent.size, selected.parent.size
    )
    ii, aa = selected.parent.configuration_orbitals()
    diagonal = e[aa] - e[ii] + response[np.arange(len(ii)), np.arange(len(ii))]
    indices = selected.parent_indices
    primary = indices[: selected.primary_count]
    included = indices[selected.primary_count :]
    candidates = np.flatnonzero(
        (diagonal.v > policy.spectral_range_hartree)
        & (diagonal.v < policy.maximum_configuration_energy_hartree)
    )
    if policy.frozen_selection is not None:
        candidates = np.asarray(policy.frozen_selection.candidates, dtype=int)
    rejected = candidates[~np.isin(candidates, included)]
    primary_diagonal = diagonal[primary]
    if len(rejected):
        coupling = response[rejected[:, None], primary[None, :]]
        denominator = diagonal[rejected, None] - diagonal[primary][None, :] + 1e-10
        primary_diagonal = primary_diagonal - (coupling**2 / denominator).sum(axis=0)
    matrix = response[indices[:, None], indices[None, :]]
    for k in range(selected.size):
        matrix[k, k] = primary_diagonal[k] if k < len(primary) else diagonal[indices[k]]
    return matrix


class MolecularSFTDAHessian(MolecularSTDAHessian):
    """Connect the analytic SF root Hessian to relaxed molecular derivatives."""

    def __init__(
        self,
        calculator,
        state,
        *,
        excited=None,
        root=0,
        roots=6,
        selection_policy=None,
        response_tolerance=1e-10,
    ):
        from .sf_tda import (
            SFTDASelectionPolicy,
            SpinFlipExcitationSpace,
            AtomicTransitionChargeResponse,
            select_grimme_spin_flip_configurations,
            grimme_xtb_sf_tda,
        )
        from .sf_tda_gradient import _sf_tda_gradient_with_coefficient_response

        self.state = state
        self.root = int(root)
        self.policy = selection_policy or SFTDASelectionPolicy()
        parent = SpinFlipExcitationSpace.from_grimme_window(state, self.policy)
        kernel = AtomicTransitionChargeResponse.grimme_xtb_spin_flip(state, parent)
        self.selected = select_grimme_spin_flip_configurations(parent, kernel, self.policy)
        self.excited = (
            excited
            if excited is not None
            else grimme_xtb_sf_tda(
                state,
                roots=min(max(roots, root + 1), self.selected.size),
                selection_policy=self.policy,
            )
        )
        if not self.excited.converged:
            raise ValueError("analytic Hessian requires converged SF-TDA roots")
        self.molecular_response = MolecularSecondResponse(calculator, state)
        first = self.molecular_response.first
        ds, _ = calculator.ground_state_derivative_matrices(state).validated(state)
        gradient = _sf_tda_gradient_with_coefficient_response(
            state,
            self.excited,
            ds,
            first.fock_derivative,
            first.coefficient_derivative,
            selection_policy=self.policy,
        )
        self.first_derivative = gradient.matrix_derivative_hartree_per_bohr
        zero = self.molecular_response.direction(np.zeros_like(state.coordinates_bohr))
        matrix = _spin_flip_matrix(
            state, self.selected, self.policy, zero, np.zeros_like(state.coordinates_bohr)
        )
        error = float(np.max(abs(matrix.a - self.first_derivative.reshape(matrix.a.shape))))
        if error > 2e-8:
            raise RuntimeError(f"mixed SF first-derivative reconstruction error: {error:g}")
        self.matrix = matrix.v
        _check_roots(self.matrix, self.excited)
        self.diagnostics = []
        self.operator = SFTDAHessianOperator(
            _DenseOperator(self.matrix),
            self.excited,
            self.first_derivative,
            self.matrix_second_derivative_action,
            root=root,
            ground_state_hessian_operator=calculator.ground_state_hessian_operator(state),
            response_tolerance=response_tolerance,
        )

    def matrix_second_derivative_action(self, direction):
        v = np.asarray(direction, dtype=float).reshape(self.state.coordinates_bohr.shape)
        orbital = self.molecular_response.direction(v)
        matrix = _spin_flip_matrix(self.state, self.selected, self.policy, orbital, v)
        first = float(np.max(abs(matrix.a - self.first_derivative.reshape(matrix.a.shape))))
        primal = float(np.max(abs(matrix.v - self.matrix)))
        if first > 2e-8 or primal > 2e-9:
            raise RuntimeError("mixed SF matrix reconstruction mismatch")
        self.diagnostics.append(
            dict(
                fock_primal_error=orbital.fock_primal_error,
                fock_first_error=orbital.fock_first_error,
                mixed_eigen_residual=orbital.eigen_residual,
                mixed_normalization_residual=orbital.normalization_residual,
                tda_primal_error=primal,
                tda_first_error=first,
            )
        )
        return matrix.ab
