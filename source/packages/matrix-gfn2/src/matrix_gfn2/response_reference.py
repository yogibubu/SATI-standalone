"""Stationary canonical references for consistent analytic nuclear responses.

An ordinary energy-converged SCC result can retain a DIIS Fock built from a
slightly different density. A Newton correction with the existing CPSCF Jacobian
closes that residual without changing the calculator's SCF stopping settings.
Independent points start from the same reference guess, never from trial history.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np


def _reset_history(calculator, *, cold):
    for name in vars(calculator):
        if name.startswith('_older_') or (cold and name.startswith('_previous_')):
            setattr(calculator, name, None)
    calculator._last_result_key = None
    calculator._last_result = None
    calculator._restart_fixed_point_contraction = 0.0


def stationary_response_reference(calculator, coordinates_angstrom, *, tolerance=1e-10, maximum_corrections=4):
    """Return a residual-certified state and its ordinary total energy/gradient.

    Only the dense integer-occupation baseline response contract is supported.
    No SCF tolerance, derivative guard or Hamiltonian parameter is modified.
    Every Newton update is recanonicalized and checked by the native evaluator.
    """
    from .ground_state_response import solve_ground_state_directional_response

    if not np.isfinite(tolerance) or tolerance <= 0 or maximum_corrections < 0:
        raise ValueError('invalid stationary-response reference controls')
    if getattr(calculator, 'nn_enabled', False) or len(calculator.point_charges_electron):
        raise NotImplementedError('stationary molecular response requires the baseline without fixed point charges')
    settings = (calculator.scf_energy_tolerance_hartree,
                calculator.scf_wavefunction_tolerance_electron, calculator.scf_mixer_tolerance)
    _reset_history(calculator, cold=True)
    residuals=[]
    canonical_residuals=[]
    canonical_tolerance=min(float(tolerance), 1e-12)
    scc_iterations=0
    for correction in range(maximum_corrections + 1):
        result=calculator.evaluate(coordinates_angstrom, retain_orbitals=True,
                                  compute_cm5=False, compute_bond_orders=False)
        scc_iterations += result.iterations
        if not result.converged or result.orbital_state is None:
            raise RuntimeError('stationary response requires a converged native SCC state')
        state=result.orbital_state
        kernel=calculator.density_response_kernel(state)
        core=calculator._model(state.coordinates_bohr, relocate_basis=True)[4]
        actual_fock=core + kernel.density_dependent_fock(np.zeros_like(state.fock))
        # A final native ISCF cycle returns the Fock of its updated density,
        # while C/epsilon diagonalize the INPUT Fock. Checking actual_fock
        # against the exported Fock alone therefore misses the canonical
        # residual, which is amplified in second-order orbital responses.
        sc=state.overlap @ state.coefficients
        canonical_fock=(sc * state.orbital_energies_hartree) @ sc.T
        residual=actual_fock - canonical_fock
        error=float(np.max(abs(residual)))
        canonical_error=float(np.max(abs(actual_fock @ state.coefficients
            - sc * state.orbital_energies_hartree)))
        residuals.append(error)
        canonical_residuals.append(canonical_error)
        if error <= tolerance and canonical_error <= canonical_tolerance:
            if settings != (calculator.scf_energy_tolerance_hartree,
                            calculator.scf_wavefunction_tolerance_electron, calculator.scf_mixer_tolerance):
                raise RuntimeError('stationary response changed native SCF settings')
            return result, dict(fock_closure_residuals_hartree=residuals,
                canonical_eigen_residuals_hartree=canonical_residuals,
                canonical_eigen_tolerance_hartree=canonical_tolerance,
                newton_corrections=correction,scc_evaluations=correction+1,
                scc_iterations=int(scc_iterations),reference_tolerance_hartree=tolerance,
                scf_settings_unchanged=True,initial_guess_policy='independent_reference')
        if correction == maximum_corrections:
            break
        response=solve_ground_state_directional_response(
            replace(state,fock=canonical_fock),np.zeros_like(residual),residual,kernel,
            tolerance=.1*canonical_tolerance)
        coefficients=state.coefficients + response.coefficient_derivative
        norm,u=np.linalg.eigh(coefficients.T @ state.overlap @ coefficients)
        if min(norm) <= 0:
            raise RuntimeError('stationary-reference Newton update lost the orbital metric')
        coefficients=coefficients @ ((u * norm**-.5) @ u.T)
        occupations=state.alpha_occupations + state.beta_occupations
        density=(coefficients * occupations) @ coefficients.T
        values,vectors=np.linalg.eigh(state.overlap)
        half=(vectors * np.sqrt(values)) @ vectors.T
        _reset_history(calculator,cold=False)
        calculator._previous_density=density
        calculator._previous_orthogonal_density=half @ density @ half
        calculator._previous_coordinates_bohr=state.coordinates_bohr.copy()
        calculator._previous_orbital_energies=state.orbital_energies_hartree + response.orbital_energy_derivative
        calculator._previous_occupations=occupations.copy()
    raise RuntimeError(f'stationary-reference Newton correction did not close the Fock residual: {residuals}')
