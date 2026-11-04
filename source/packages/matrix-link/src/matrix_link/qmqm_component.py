"""State-resolved components of LINK's capped subtractive QM/QM' surface.

Each component owns an accepted APOC reference. Trial evaluations never advance
it. No correspondence of root numbers across different Hamiltonians is assumed.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import RLock
import re

import numpy as np


def _matching_gaussian_response_grid(spec):
    """Match the response quadrature to the main grid; honor explicit choices."""
    route = spec.route
    pattern = r"\bcphf(?:\s*=\s*(\([^)]*\)|[^\s]+)|\s*(\([^)]*\)))?"
    match = re.search(pattern, route, flags=re.IGNORECASE)
    options = [] if match is None else (match.group(1) or match.group(2) or '').strip('()').split(',')
    if any(re.match(r"\s*grid\s*=", item, flags=re.IGNORECASE) for item in options):
        return route
    tier = str(spec.gaussian_accuracy_tier).lower()
    grid = {'coarse':'SG1Grid','initial':'FineGrid','intermediate':'FineGrid','final':'UltraFine'}
    if tier == 'protocol':
        integral = re.search(r"\b(?:int|integral)\s*(?:=\s*)?(\([^)]*\)|[^\s]+)",route,flags=re.IGNORECASE)
        if integral is None:
            # GDV and the supported G16 profile use an UltraFine main grid.
            value = 'UltraFine'
        else:
            value = integral.group(1)
            named = re.search(r"\b(SuperFineGrid|SuperFine|UltraFineGrid|UltraFine|FineGrid|SG1Grid|SG1)\b",value,flags=re.IGNORECASE)
            if named is None:
                raise ValueError('custom Gaussian main grid requires an explicit matching CPHF(Grid=...)')
            value = named.group(1)
    else:
        value = grid[tier]
    options = [x.strip() for x in options if x.strip()]
    options.append(f'Grid={value}')
    keyword = f"CPHF=({','.join(options)})"
    return (route + ' ' + keyword).strip() if match is None else route[:match.start()] + keyword + route[match.end():]


class _FingerprintTracker:
    """Reuse LINK's external-backend APOC policy without an optimizer chart."""

    def __init__(self, backend):
        self.backend = backend
        self._lock = RLock()
        self._state_reference = self._state_anchor = None
        self._state_reference_symmetry = None

    def _state_fingerprints(self, result):
        from .optimizer import GeometryEvaluationService
        return GeometryEvaluationService._state_fingerprints(self, result)

    def _state_symmetries(self, result):
        from .optimizer import GeometryEvaluationService
        return GeometryEvaluationService._state_symmetries(self, result)

    def follow(self, result):
        from .optimizer import GeometryEvaluationService
        return GeometryEvaluationService._follow_electronic_state(self, result)

    def accept(self, result):
        selected = int(result.execution['selected_electronic_state'])
        reference = next(x for x in self._state_fingerprints(result) if x.state_id == selected)
        if self._state_anchor is None:
            self._state_anchor = reference
        self._state_reference = reference
        self._state_reference_symmetry = result.execution.get('state_target_symmetry')


class CappedQMComponent:
    """Adapt a native or Gaussian/GDV calculator, including analytic Hessians."""

    def __init__(self, backend, calculator, run_dir):
        from .scan import _normalized_backend_name
        self.backend = backend
        self.calculator = calculator
        self.run_dir = Path(run_dir)
        self.native = _normalized_backend_name(backend.name) == 'matrix_gfn2'
        self.reference = None
        self.anchor = None
        self.selection_reference = None
        self.tracker = _FingerprintTracker(backend)
        if backend.electronic_state > 0 and backend.state_tracking != 'apoc':
            raise ValueError('excited QM/QM components require APOC state following')
        if self.native and backend.electronic_state > 0:
            spin = backend.state_spin.lower().replace('_', '-')
            if (backend.multiplicity == 1 and spin != 'singlet') or (
                backend.multiplicity != 1 and spin not in {'spin-flip', 'sf'}
            ):
                raise ValueError('native QM/QM response requires closed-shell singlet or explicit spin-flip')

    def evaluate(self, coordinates_angstrom, *, compute_hessian=False):
        if self.native:
            from matrix_gfn2.response_reference import stationary_response_reference
            from .scan import PointEvaluationResult
            calc = self.calculator._calculator
            ground, diagnostics = stationary_response_reference(calc, coordinates_angstrom)
            if self.backend.electronic_state > 0:
                return self._native_excited(coordinates_angstrom, compute_hessian, ground, diagnostics)
            hessian = (calc.ground_state_hessian(ground.orbital_state).hessian_hartree_per_bohr2
                       if compute_hessian else None)
            return PointEvaluationResult(0, 0., energy_hartree=ground.energy_hartree,
                gradient_hartree_per_bohr=ground.gradient_hartree_per_bohr.reshape(-1),
                hessian_hartree_per_bohr2=hessian,backend_coordinates_angstrom=np.asarray(coordinates_angstrom),
                source='native stationary QM/QM ground component',
                execution={'stationary_reference': diagnostics, 'hessian_source': 'analytic' if compute_hessian else None})
        if not hasattr(self.calculator, 'spec'):
            if self.backend.electronic_state > 0 or compute_hessian:
                raise NotImplementedError('this capped component does not supply excited states or Hessians')
            return self.calculator.evaluate(coordinates_angstrom)
        requested = ('energy', 'gradient', 'hessian') if compute_hessian else ('energy', 'gradient')
        spec = replace(self.backend, properties=requested)
        spec = replace(spec, route=_matching_gaussian_response_grid(spec))
        if spec.electronic_state > 0:
            from .scan import _gaussian_route_with_requested_excited_root
            roots = max(spec.excited_states or 0, spec.electronic_state + 3, 6)
            # The default TD vector residual (1e-6) contaminates differences
            # of analytic gradients. This is an electronic response setting,
            # independent of SCF/geometry stopping criteria. Explicit user
            # Conver choices are preserved by the route binder.
            spec = replace(spec, excited_states=roots,
                route=_gaussian_route_with_requested_excited_root(spec.route,
                    root=spec.electronic_state,nstates=roots,response_convergence_exponent=9))
        result = self.calculator.evaluate(coordinates_angstrom, spec=spec)
        if spec.electronic_state <= 0:
            return result
        followed = self.tracker.follow(result)
        selected = int(followed.execution['selected_electronic_state'])
        if selected != spec.electronic_state:
            result = self.calculator.evaluate(coordinates_angstrom, spec=replace(spec, electronic_state=selected))
            followed = self.tracker.follow(result)
            if int(followed.execution['selected_electronic_state']) != selected:
                raise RuntimeError('QM/QM component root changed during derivative reevaluation')
        # The manifold may use rounded printed excitation energies. The
        # derivative job owns the precise total energy of its selected root.
        return replace(followed, energy_hartree=result.energy_hartree, execution={**followed.execution,
            'state_energy_source': 'selected_root_derivative_job',
            'gradient_electronic_state': selected,
            'state_gradient_root_recalculated': selected != spec.electronic_state})

    def _native_excited(self, coordinates, compute_hessian, ground, diagnostics):
        from matrix_apoc import ExcitedState, follow_excited_state
        from matrix_gfn2.stda import grimme_xtb_stda, freeze_closed_shell_stda_selection
        from matrix_gfn2.sf_tda import grimme_xtb_sf_tda, SFTDASelectionPolicy
        from .scan import PointEvaluationResult

        calc = self.calculator._calculator
        state = ground.orbital_state
        roots = max(self.backend.excited_states or 0, self.backend.electronic_state + 3, 6)
        closed = self.backend.multiplicity == 1
        window = self.backend.tda_spectral_range_ev
        policy = SFTDASelectionPolicy(spectral_range_ev=window,
            frozen_selection=None if closed else self.selection_reference)
        excited = (grimme_xtb_stda(state, roots=roots, spectral_range_ev=window,
                   selection_reference=self.selection_reference) if closed
                   else grimme_xtb_sf_tda(state, roots=roots, selection_policy=policy, tolerance_hartree=1e-10))
        if closed:
            excited = freeze_closed_shell_stda_selection(state, excited, spectral_range_ev=window)
        if not closed:
            policy = replace(policy, frozen_selection=excited.selection_reference)
        if not excited.converged:
            raise RuntimeError('QM/QM component TDA manifold did not converge')
        if closed:
            densities = [np.einsum('pi,i,qi->pq', state.coefficients[:, excited.occupied_orbitals],
                         u, state.coefficients[:, excited.virtual_orbitals]) for u in excited.amplitudes]
        else:
            densities = [p.transition_density_ao for p in excited.transition_properties]
        candidates = tuple(ExcitedState(k + 1, d,
            energy_hartree=float(ground.energy_hartree + excited.excitation_energies_hartree[k]))
            for k, d in enumerate(densities))
        if len(candidates) < self.backend.electronic_state:
            raise RuntimeError('QM/QM component manifold lacks its requested root')
        selected = candidates[self.backend.electronic_state - 1]
        overlap = margin = anchor_overlap = 1.0
        if self.reference is not None:
            old_state, old_selected = self.reference[:2]
            selected, match = follow_excited_state(old_selected, candidates,
                self._cross_overlap(old_state, state), overlap_reference=old_state.overlap,
                overlap_candidate=state.overlap,
                minimum_overlap=self.backend.state_tracking_minimum_overlap,
                ambiguity_margin=self.backend.state_tracking_ambiguity_margin, strict=True)
            overlap, margin = match.overlap, match.margin
            anchor_state, anchor_selected = self.anchor[:2]
            _, anchor_match = follow_excited_state(anchor_selected, (selected,),
                self._cross_overlap(anchor_state, state), overlap_reference=anchor_state.overlap,
                overlap_candidate=state.overlap, minimum_overlap=0., ambiguity_margin=0., strict=False)
            anchor_overlap = anchor_match.overlap
        index = selected.state_id - 1
        operator = (calc.closed_shell_stda_hessian(state, excited=excited, root=index, spectral_range_ev=window) if closed
                    else calc.sf_tda_hessian(state, excited=excited, root=index, selection_policy=policy))
        hessian = operator.materialize(symmetrize=False) if compute_hessian else None
        return PointEvaluationResult(point_index=0, displacement=0.,
            energy_hartree=selected.energy_hartree,
            gradient_hartree_per_bohr=(ground.gradient_hartree_per_bohr +
                operator.excitation_gradient_hartree_per_bohr).reshape(-1),
            hessian_hartree_per_bohr2=hessian,
            backend_coordinates_angstrom=np.asarray(coordinates), source='native FAST_TDA QM/QM component',
            scf_state=(state, selected, excited if closed else excited.selection_reference), execution={
                'configuration_space_policy': 'frozen_initial_accepted',
                'configuration_count': int(excited.amplitudes.shape[1]),
                'stationary_reference': diagnostics,
                'state_ids': [item.state_id for item in candidates],
                'excitation_energies_hartree': excited.excitation_energies_hartree.tolist(),
                'selected_electronic_state': selected.state_id,
                'requested_electronic_state': self.backend.electronic_state,
                'gradient_electronic_state': selected.state_id,
                'state_gradient_root_recalculated': selected.state_id != self.backend.electronic_state,
                'state_assignment_overlap': float(overlap), 'state_assignment_margin': float(margin),
                'state_anchor_overlap': float(anchor_overlap),
                'state_following_contract': 'apoc_cross_ao_tdm_previous_accepted',
                'hessian_source': 'analytic' if compute_hessian else None,
                'response_model': 'closed-shell-stda' if closed else 'spin-flip-stda'})

    @staticmethod
    def _cross_overlap(left, right):
        from matrix_gfn2.tda_two_photon import _cartesian_basis_functions
        from matrix_gfn2.basis import contracted_overlap
        a, b = _cartesian_basis_functions(left), _cartesian_basis_functions(right)
        cart = np.array([[contracted_overlap(i, j) for j in b] for i in a])
        return left.basis.spherical_to_cartesian.T @ cart @ right.basis.spherical_to_cartesian

    def validate_accept(self, result):
        if self.backend.electronic_state <= 0:
            return
        selected = int(result.execution['selected_electronic_state'])
        if selected != int(result.execution['gradient_electronic_state']):
            raise ValueError('accepted QM/QM component has derivatives of another root')
        if self.native:
            if result.scf_state is None or result.scf_state[1].state_id != selected:
                raise ValueError('accepted native QM/QM component lacks its selected state')
        elif not any(x.state_id == selected for x in self.tracker._state_fingerprints(result)):
            raise ValueError('accepted external QM/QM component lacks its selected state')

    def accept(self, result):
        self.validate_accept(result)
        if self.backend.electronic_state <= 0:
            return
        if self.native:
            if self.anchor is None:
                self.anchor = result.scf_state
            self.reference = result.scf_state
            if self.selection_reference is None and len(result.scf_state) > 2:
                self.selection_reference = result.scf_state[2]
        else:
            self.tracker.accept(result)
