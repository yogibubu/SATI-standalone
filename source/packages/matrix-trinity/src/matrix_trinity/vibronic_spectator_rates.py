"""FGR sums over variational active levels and excited harmonic spectators."""
from __future__ import annotations

from collections import OrderedDict
from math import prod
import hashlib
import json
import sqlite3

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.csgraph import connected_components
from scipy.special import voigt_profile
from scipy.stats import t as student_t

from .vibronic_harmonic import harmonic_gaussian_generator, orient_harmonic_model
from .vibronic_rates import GOLDEN_RULE_FACTOR
from .vibronic_variational import _gaussian_overlap_grid
from .vibronic_vpt2 import _positive_integer


class _SpectatorOverlapKernel:
    """Reuse one Gaussian generator while spectator occupations change."""

    def __init__(self, model, initial_basis, final_basis, *, matrix_zero_tolerance,
                 maximum_grid_entries, maximum_harmonic_pairs, cache_components=False,
                 prepared_generator=None):
        if not np.isfinite(matrix_zero_tolerance) or matrix_zero_tolerance < 0:
            raise ValueError("nonnegative finite Gaussian matrix threshold required")
        _positive_integer(maximum_grid_entries, "maximum_grid_entries")
        _positive_integer(maximum_harmonic_pairs, "maximum_harmonic_pairs")
        self.n = len(model.initial_frequencies_cm)
        self.initial = np.asarray(initial_basis, np.int64)
        self.final = np.asarray(final_basis, np.int64)
        if len(self.initial)*len(self.final) > maximum_harmonic_pairs:
            raise MemoryError("spectator overlap harmonic-pair budget exceeded")
        generator = harmonic_gaussian_generator(model)
        prepared_deviation = 0.
        if prepared_generator is not None:
            # A portable definition retains its original floating-point bytes.
            # Only changes at ordinary arithmetic precision may bypass recomputation.
            candidate = dict(prepared_generator)
            for name,shape in (('matrix',(2*self.n,2*self.n)),('linear',(2*self.n,))):
                value = np.asarray(candidate[name],float)
                scale = max(1.,float(np.max(abs(generator[name]))))
                if (value.shape!=shape or not np.all(np.isfinite(value))
                        or np.max(abs(value-generator[name]))>64*np.finfo(float).eps*scale):
                    raise ValueError('prepared Gaussian generator differs from the supplied physical model')
                prepared_deviation = max(prepared_deviation,float(np.max(abs(value-generator[name]))))
                candidate[name] = value
            log_ground = float(candidate['log_ground'])
            ground = float(candidate['ground'])
            if (not np.isfinite(log_ground+ground) or ground<=0
                    or abs(log_ground-generator['log_ground'])>64*np.finfo(float).eps*max(1.,abs(generator['log_ground']))
                    or abs(ground-np.exp(log_ground))>64*np.finfo(float).eps*ground):
                raise ValueError('prepared Gaussian vacuum normalization differs')
            if not np.array_equal(candidate['matrix'],candidate['matrix'].T):
                raise ValueError('prepared Gaussian matrix must be exactly symmetric')
            generator = candidate
        if generator['log_ground'] < np.log(np.finfo(float).tiny)+4:
            raise FloatingPointError("spectator overlap vacuum amplitude underflows")
        original = generator['matrix']
        self.matrix = original.copy()
        off_diagonal = ~np.eye(2*self.n, dtype=bool)
        self.matrix[off_diagonal & (abs(original) <= matrix_zero_tolerance)] = 0.0
        self.linear = generator['linear']
        self.ground = (float(generator['ground']) if prepared_generator is not None
                       else np.exp(generator['log_ground']))
        self.maximum_grid_entries = maximum_grid_entries
        _, component_labels = connected_components(self.matrix != 0, directed=False)
        self.diagnostics = dict(
            gaussian_matrix_zero_tolerance=float(matrix_zero_tolerance),
            removed_matrix_frobenius_norm=float(np.linalg.norm(original-self.matrix)),
            removed_matrix_max_abs=float(np.max(abs(original-self.matrix), initial=0)),
            full_duschinsky_generator_used=True, gaussian_grid_evaluations=0,
            prepared_gaussian_component_sizes=np.bincount(component_labels).tolist(),
            maximum_component_grid_entries=0, coefficient_pruning_used=False,
            component_cache_hits=0, component_cache_bytes=0,
            maximum_component_cache_bytes=64*1024**2,
            maximum_component_cache_entries=4096,
            component_cache_enabled=bool(cache_components),
            prepared_gaussian_generator_used=prepared_generator is not None,
            prepared_generator_maximum_deviation=prepared_deviation,
        )
        self.components = []
        self.component_cache = OrderedDict()
        if cache_components and len(np.unique(component_labels)) > 1:
            for label in np.unique(component_labels):
                indices = np.flatnonzero(component_labels == label)
                left, right = indices[indices < self.n], indices[indices >= self.n]-self.n
                bi, bf = self.initial.copy(), self.final.copy()
                bi[:, np.setdiff1d(np.arange(self.n), left)] = 0
                bf[:, np.setdiff1d(np.arange(self.n), right)] = 0
                bi, inverse_i = np.unique(bi, axis=0, return_inverse=True)
                bf, inverse_f = np.unique(bf, axis=0, return_inverse=True)
                component = object.__new__(_SpectatorOverlapKernel)
                for name in ('n', 'matrix', 'linear', 'maximum_grid_entries', 'diagnostics'):
                    setattr(component, name, getattr(self, name))
                component.initial, component.final, component.ground = bi, bf, 1.0
                self.components.append((component, inverse_i, inverse_f, set(map(int, right))))

    def evaluate(self, spectator_modes, occupations):
        if not self.components:
            return self._evaluate_direct(spectator_modes, occupations)
        result = np.full((len(self.final), len(self.initial)), self.ground)
        for factor, inverse_i, inverse_f in self._component_factors(spectator_modes, occupations):
            result *= factor[inverse_f[:, None], inverse_i[None, :]]
        return result

    def _component_factors(self, spectator_modes, occupations):
        """Yield the small symmetry factors without expanding their row maps."""
        for index, (component, inverse_i, inverse_f, final_modes) in enumerate(self.components):
            assignments = tuple((int(mode), int(value)) for mode, value in zip(spectator_modes, occupations)
                                if int(mode) in final_modes)
            key = (index, assignments)
            factor = self.component_cache.pop(key, None)
            if factor is None:
                modes = np.array([m for m, _ in assignments], dtype=int)
                values = np.array([v for _, v in assignments], dtype=int)
                factor = component._evaluate_direct(modes, values)
                limit = self.diagnostics['maximum_component_cache_bytes']
                if factor.nbytes <= limit:
                    while self.component_cache and (
                            self.diagnostics['component_cache_bytes']+factor.nbytes > limit
                            or len(self.component_cache) >= self.diagnostics['maximum_component_cache_entries']):
                        _, removed = self.component_cache.popitem(last=False)
                        self.diagnostics['component_cache_bytes'] -= removed.nbytes
                    self.component_cache[key] = factor
                    self.diagnostics['component_cache_bytes'] += factor.nbytes
            else:
                self.component_cache[key] = factor
                self.diagnostics['component_cache_hits'] += 1
            yield factor, inverse_i, inverse_f

    def prepare_contraction(self, coefficients, vacuum_index, *, maximum_workspace_bytes=256*1024**2):
        """Contract the complete initial state before expanding final row maps.

        This is an exact tensor contraction of the existing Gaussian blocks.
        If their Cartesian product is too large, retain the original matrix
        contraction. No singular values or initial coefficients are discarded.
        """
        return _FactorizedSpectatorContraction(self, coefficients, vacuum_index,
                                               maximum_workspace_bytes)

    def _evaluate_direct(self, spectator_modes, occupations):
        final = self.final.copy()
        final[:, spectator_modes] = occupations
        maxima = np.r_[self.initial.max(axis=0), final.max(axis=0)]
        active = np.flatnonzero(maxima)
        graph = self.matrix[np.ix_(active, active)] != 0
        _, labels = connected_components(graph, directed=False)
        result = np.full((len(final), len(self.initial)), self.ground)
        for label in np.unique(labels):
            indices = active[labels == label]
            dimensions = maxima[indices]+1
            entries = prod(int(d) for d in dimensions)
            if entries > self.maximum_grid_entries:
                raise MemoryError(f"spectator Gaussian component needs {entries} entries; "
                                  f"limit {self.maximum_grid_entries}; no state was discarded")
            strides = np.array([prod(int(d) for d in dimensions[k+1:])
                                for k in range(len(indices))], np.int64)
            grid = _gaussian_overlap_grid(self.matrix[np.ix_(indices, indices)],
                                          self.linear[indices], dimensions, strides, 1.0)
            si, sf = np.zeros(self.n, np.int64), np.zeros(self.n, np.int64)
            for index, stride in zip(indices, strides):
                if index < self.n:
                    si[index] = stride
                else:
                    sf[index-self.n] = stride
            result *= grid[(final @ sf)[:, None]+(self.initial @ si)[None, :]]
            self.diagnostics['gaussian_grid_evaluations'] += 1
            self.diagnostics['maximum_component_grid_entries'] = max(
                self.diagnostics['maximum_component_grid_entries'], entries)
        return result


class _FactorizedSpectatorContraction:
    def __init__(self, kernel, coefficients, vacuum_index, maximum_workspace_bytes):
        coefficients = np.asarray(coefficients)
        if (coefficients.shape != (len(kernel.initial),)
                or not np.all(np.isfinite(coefficients))
                or not isinstance(vacuum_index, (int, np.integer))
                or not 0 <= vacuum_index < len(coefficients)
                or not np.all(kernel.initial[vacuum_index] == 0)
                or maximum_workspace_bytes <= 0):
            raise ValueError('finite initial coefficients, vacuum row and positive workspace required')
        self.kernel, self.coefficients = kernel, coefficients.copy()
        self.vacuum_index = int(vacuum_index)
        self.tensor = None
        self.order = []
        if not kernel.components:
            return
        ni = [len(c.initial) for c, *_ in kernel.components]
        nf = [len(c.final) for c, *_ in kernel.components]
        itemsize = np.dtype(np.result_type(coefficients.dtype, float)).itemsize
        size = prod(ni)
        peak = size
        pending = set(range(len(ni)))
        while pending:
            index = min(pending, key=lambda i: (size//ni[i]*nf[i], i))
            following = size//ni[index]*nf[index]
            peak = max(peak, size+following)
            self.order.append(index)
            pending.remove(index)
            size = following
        if peak*itemsize > maximum_workspace_bytes:
            return
        self.tensor = np.zeros(ni, dtype=np.result_type(coefficients.dtype, float))
        np.add.at(self.tensor, tuple(c[1] for c in kernel.components), coefficients)
        self.tensor.setflags(write=False)
        kernel.diagnostics.update(factorized_initial_contraction=True,
            contraction_initial_tensor_entries=prod(ni),
            contraction_maximum_live_tensor_bytes=peak*itemsize,
            contraction_coefficient_pruning_used=False)

    def __call__(self, spectator_modes, occupations):
        kernel = self.kernel
        if self.tensor is None:
            overlap = kernel.evaluate(spectator_modes, occupations)
            return overlap @ self.coefficients, overlap[:, self.vacuum_index].copy()
        factors = list(kernel._component_factors(spectator_modes, occupations))
        tensor = self.tensor
        labels = list(range(len(factors)))
        for index in self.order:
            axis = labels.index(index)
            tensor = np.tensordot(factors[index][0], tensor, axes=(1, axis))
            labels = [index] + labels[:axis] + labels[axis+1:]
        packet = kernel.ground*tensor[tuple(factors[index][2] for index in labels)]
        harmonic = np.full(len(kernel.final), kernel.ground)
        for factor, inverse_i, inverse_f in factors:
            harmonic *= factor[inverse_f, inverse_i[self.vacuum_index]]
        return packet, harmonic


class VariationalSpectatorObservable:
    """Sum active eigenlevels exactly for each supplied spectator occupation.

    Both inputs refer to the original model surfaces; emission reverses them
    together. The final active basis must have zero spectator occupations.
    The reference uses the same finite active HO basis, all its harmonic levels,
    its own harmonic ZPE, the same electronic gap, and the same sampled spectator
    domain. No variational eigenlevel is relabeled as a harmonic occupation.
    """

    def __init__(self, model, initial, final, coupling_hartree, *, kind="emission",
                 gaussian_sigma_cm=21.23, lorentz_hwhm_cm=0.0,
                 active_final_modes=None, matrix_zero_tolerance=0.0,
                 maximum_grid_entries=8_000_000, maximum_harmonic_pairs=2_000_000,
                 maximum_cached_spectators=2048, cache_gaussian_components=False,
                 persistent_cache_path=None):
        if (not np.array_equal(initial.frequencies_cm, model.initial_frequencies_cm)
                or not np.array_equal(final.frequencies_cm, model.final_frequencies_cm)):
            raise ValueError("variational harmonic bases must match the model exactly")
        oriented, _ = orient_harmonic_model(model, kind)
        if kind == 'emission':
            initial, final = final, initial
        coupling = np.asarray(coupling_hartree, complex)
        if coupling.ndim != 1 or not len(coupling) or not np.all(np.isfinite(coupling)):
            raise ValueError("finite constant coupling channels required")
        if (not np.isfinite(gaussian_sigma_cm+lorentz_hwhm_cm)
                or min(gaussian_sigma_cm, lorentz_hwhm_cm) < 0
                or gaussian_sigma_cm+lorentz_hwhm_cm <= 0):
            raise ValueError("finite nonzero Voigt linewidth required")
        if active_final_modes is None:
            active_final_modes = final.diagnostics.get('active_modes', np.flatnonzero(final.basis.max(axis=0)))
        active = np.asarray(active_final_modes)
        n = len(final.frequencies_cm)
        if (active.ndim != 1 or not len(active) or active.dtype.kind not in 'iu'
                or np.any(active < 0) or np.any(active >= n) or len(np.unique(active)) != len(active)):
            raise ValueError("unique valid final active mode indices required")
        spectator = np.setdiff1d(np.arange(n), active)
        if not len(spectator):
            raise ValueError("use variational_vibronic_transitions when there are no spectators")
        if np.any(final.basis[:, spectator]):
            raise ValueError("the embedded active basis must have harmonic-vacuum spectators")
        zero = np.flatnonzero(np.all(initial.basis == 0, axis=1))
        if len(zero) != 1:
            raise ValueError("initial active basis must contain its harmonic vacuum")
        _positive_integer(maximum_cached_spectators, 'maximum_cached_spectators')
        self.initial, self.final, self.oriented = initial, final, oriented
        self.final_conjugate_vectors = final.eigenvectors.conj()
        self.active_modes, self.spectator_modes = active, spectator
        self.frequencies_cm = final.frequencies_cm[spectator]
        self.coupling_strength = float(np.sum(abs(coupling)**2))
        self.sigma, self.gamma = float(gaussian_sigma_cm), float(lorentz_hwhm_cm)
        self.initial_harmonic_vacuum = int(zero[0])
        self.harmonic_final_energies = (final.basis+0.5) @ final.frequencies_cm
        self.harmonic_initial_energy = 0.5*initial.frequencies_cm.sum()
        self.vibronic_origin_cm = float(-oriented.electronic_gap_cm+initial.energies_cm[0]-final.energies_cm[0])
        self.harmonic_origin_cm = float(-oriented.electronic_gap_cm+self.harmonic_initial_energy
                                        -0.5*final.frequencies_cm.sum())
        self.kernel = _SpectatorOverlapKernel(
            oriented, initial.basis, final.basis, matrix_zero_tolerance=matrix_zero_tolerance,
            maximum_grid_entries=maximum_grid_entries, maximum_harmonic_pairs=maximum_harmonic_pairs,
            cache_components=cache_gaussian_components)
        self.cache = OrderedDict()
        self.maximum_cached_spectators = maximum_cached_spectators
        self.cache_hits = 0
        self.evaluated_spectators = 0
        self.persistent_cache = None
        self.persistent_cache_hits = 0
        self.persistent_cache_writes = 0
        self.persistent_model_key = None
        if persistent_cache_path is not None:
            digest = hashlib.sha256(b'matrix.sati.spectator_observable.v1')
            arrays = (self.kernel.matrix, self.kernel.linear, np.array([self.kernel.ground]),
                initial.basis, initial.energies_cm, initial.eigenvectors, initial.frequencies_cm,
                final.basis, final.energies_cm, final.eigenvectors, final.frequencies_cm,
                active, spectator, np.array([oriented.electronic_gap_cm, self.coupling_strength,
                                            self.sigma, self.gamma]))
            for array in arrays:
                array = np.ascontiguousarray(array)
                digest.update(json.dumps((array.dtype.str, array.shape)).encode())
                digest.update(array.tobytes())
            self.persistent_model_key = digest.hexdigest()
            connection = sqlite3.connect(str(persistent_cache_path), timeout=30, isolation_level=None)
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=NORMAL')
            connection.execute('CREATE TABLE IF NOT EXISTS spectator_values_v1 '
                '(model TEXT NOT NULL, occupations BLOB NOT NULL, anharmonic REAL NOT NULL, '
                'harmonic REAL NOT NULL, PRIMARY KEY(model,occupations)) WITHOUT ROWID')
            self.persistent_cache = connection

    def _remember(self, key, values):
        self.cache[key] = values
        if len(self.cache) > self.maximum_cached_spectators:
            self.cache.popitem(last=False)
        return values

    def _evaluate(self, occupations):
        key = tuple(int(value) for value in occupations)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        serialized = np.asarray(key, dtype='<i8').tobytes()
        if self.persistent_cache is not None:
            row = self.persistent_cache.execute(
                'SELECT anharmonic,harmonic FROM spectator_values_v1 WHERE model=? AND occupations=?',
                (self.persistent_model_key, serialized)).fetchone()
            if row is not None:
                if not np.all(np.isfinite(row)) or min(row) < 0:
                    raise FloatingPointError('invalid persistent spectator observable')
                self.persistent_cache_hits += 1
                return self._remember(key, tuple(row))
        overlap = self.kernel.evaluate(self.spectator_modes, occupations)
        spectator_energy = float(self.frequencies_cm @ occupations)
        mismatch = (self.oriented.electronic_gap_cm+self.final.energies_cm
                    -self.initial.energies_cm[0]+spectator_energy)
        profile = voigt_profile(mismatch, self.sigma, self.gamma)
        contributing = np.flatnonzero(profile > 0)
        projected_initial = np.einsum('fi,i->f', overlap, self.initial.eigenvectors[:, 0])
        amplitude = np.einsum('fk,f->k', self.final_conjugate_vectors[:, contributing], projected_initial)
        anharmonic = self.coupling_strength * float(np.sum(abs(amplitude)**2*profile[contributing]))
        harmonic_amplitude = overlap[:, self.initial_harmonic_vacuum]
        harmonic_mismatch = (self.oriented.electronic_gap_cm+self.harmonic_final_energies
                             -self.harmonic_initial_energy+spectator_energy)
        harmonic = self.coupling_strength * float(np.sum(abs(harmonic_amplitude)**2
            * voigt_profile(harmonic_mismatch, self.sigma, self.gamma)))
        self.evaluated_spectators += 1
        if self.persistent_cache is not None:
            if not np.all(np.isfinite((anharmonic, harmonic))) or min(anharmonic, harmonic) < 0:
                raise FloatingPointError('invalid spectator observable; cache write refused')
            inserted = self.persistent_cache.execute('INSERT OR IGNORE INTO spectator_values_v1 VALUES (?,?,?,?)',
                (self.persistent_model_key, serialized, anharmonic, harmonic))
            self.persistent_cache_writes += inserted.rowcount
        return self._remember(key, (anharmonic, harmonic))

    def close(self):
        """Close the optional persistent store; each scalar pair is already committed."""
        if self.persistent_cache is not None:
            self.persistent_cache.close()
            self.persistent_cache = None

    def _values(self, states, column):
        states = np.asarray(states)
        if (states.ndim != 2 or states.shape[1] != len(self.spectator_modes)
                or states.dtype.kind not in 'iu' or np.any(states < 0)
                or np.any(states > np.iinfo(np.int32).max)):
            raise ValueError("nonnegative integer spectator occupation rows required")
        return np.array([self._evaluate(row)[column] for row in states])

    def anharmonic(self, states):
        return self._values(states, 0)

    def harmonic(self, states):
        return self._values(states, 1)

    @property
    def diagnostics(self):
        return dict(
            **self.kernel.diagnostics, active_final_modes=self.active_modes,
            spectator_modes=self.spectator_modes, active_level_count=len(self.final.energies_cm),
            active_harmonic_basis_size=len(self.final.basis),
            active_excitation_max_cm=float(self.final.energies_cm[-1]-self.final.energies_cm[0]),
            active_energy_coverage_passed=bool(self.final.energies_cm[-1]-self.final.energies_cm[0]
                >= self.vibronic_origin_cm+8*max(self.sigma, self.gamma)),
            vibronic_origin_cm=self.vibronic_origin_cm, harmonic_origin_cm=self.harmonic_origin_cm,
            evaluated_spectators=self.evaluated_spectators, cache_hits=self.cache_hits,
            persistent_model_key=self.persistent_model_key,
            persistent_cache_hits=self.persistent_cache_hits,
            persistent_cache_writes=self.persistent_cache_writes,
            active_levels_summed_explicitly=True, initial_variational_level=0,
        )


class MarginalSpectatorObservable:
    """Exactly sum selected harmonic spectator modes before outer sampling.

    The inner sum obeys the reference provider's declared total-occupation and
    energy domain. No inner Monte Carlo or conditional normalization is used.
    This gives common outer axes when comparing to a larger final active space.
    """

    def __init__(self, provider, summed_modes, *, max_spectator_quanta,
                 energy_max_cm, maximum_inner_states=100000, cache_size=4096):
        from .vibronic_sparse import SparseSpectatorObservable
        if not isinstance(provider, (VariationalSpectatorObservable,SparseSpectatorObservable)):
            raise TypeError('a finite variational or sparse spectral spectator observable is required')
        modes = np.asarray(summed_modes)
        if (modes.ndim != 1 or not len(modes) or modes.dtype.kind not in 'iu'
                or len(np.unique(modes)) != len(modes)
                or not set(modes).issubset(provider.spectator_modes)):
            raise ValueError('unique spectator mode indices required for the exact inner sum')
        _positive_integer(max_spectator_quanta, 'max_spectator_quanta')
        _positive_integer(maximum_inner_states, 'maximum_inner_states')
        _positive_integer(cache_size, 'cache_size')
        if not np.isfinite(energy_max_cm) or energy_max_cm <= 0:
            raise ValueError('positive finite spectator energy edge required')
        self.provider = provider
        self.summed = np.array([list(provider.spectator_modes).index(m) for m in modes], np.int64)
        self.retained = np.array([i for i, m in enumerate(provider.spectator_modes) if m not in modes], np.int64)
        self.spectator_modes = provider.spectator_modes[self.retained]
        self.frequencies_cm = provider.frequencies_cm[self.retained]
        self.cap, self.edge = max_spectator_quanta, float(energy_max_cm)
        self.maximum_inner_states, self.cache_size = maximum_inner_states, cache_size
        self.cache = OrderedDict()
        self.outer_evaluations = self.inner_states = self.cache_hits = 0

    def _evaluate(self, outer):
        key = tuple(map(int, outer))
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        state = np.zeros(len(self.provider.spectator_modes), np.int64)
        state[self.retained] = outer
        rows = []
        margin = 64*np.finfo(float).eps*max(1., self.edge)*len(state)
        def visit(level, remaining, energy):
            if energy > self.edge+margin or remaining < 0:
                return
            if level == len(self.summed):
                if float(self.provider.frequencies_cm@state) >= self.edge:
                    return
                rows.append(state.copy())
                if len(rows) > self.maximum_inner_states:
                    raise MemoryError('exact spectator inner sum exceeds declared state budget')
                return
            axis = self.summed[level]
            frequency = self.provider.frequencies_cm[axis]
            limit = min(remaining, int(np.floor((self.edge-energy+margin)/frequency)))
            for occupation in range(limit+1):
                state[axis] = occupation
                visit(level+1, remaining-occupation, energy+occupation*frequency)
        visit(0, self.cap-int(np.sum(outer)), float(self.frequencies_cm@outer))
        if rows:
            states=np.asarray(rows,dtype=np.int64)
            # Public block callbacks reuse prepared spectra and Gaussian
            # components; scalar evaluation would discard that acceleration.
            result=(float(np.sum(self.provider.anharmonic(states))),
                    float(np.sum(self.provider.harmonic(states))))
        else:
            result=(0.,0.)
        self.outer_evaluations += 1
        self.inner_states += len(rows)
        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def _values(self, states, channel):
        states = np.asarray(states)
        if (states.ndim != 2 or states.shape[1] != len(self.spectator_modes)
                or states.dtype.kind not in 'iu' or np.any(states < 0)
                or np.any(states > np.iinfo(np.int32).max)):
            raise ValueError('nonnegative integer common-spectator rows required')
        return np.array([self._evaluate(row)[channel] for row in states])

    def anharmonic(self, states):
        return self._values(states, 0)

    def harmonic(self, states):
        return self._values(states, 1)

    @property
    def diagnostics(self):
        return dict(summed_modes=self.provider.spectator_modes[self.summed],
                    common_spectator_modes=self.spectator_modes,
                    outer_evaluations=self.outer_evaluations,
                    inner_states_summed=self.inner_states, outer_cache_hits=self.cache_hits,
                    exact_inner_sum=True, conditional_normalization_used=False,
                    max_spectator_quanta=self.cap, energy_max_cm=self.edge)


def variational_spectator_nonradiative_rate(
    model, initial, final, coupling_hartree, *, max_spectator_quanta,
    samples_per_replica=1000, replicas=8, pilot_rounds=1, pilot_samples_per_replica=64,
    seed=61453, energy_max_cm=None, proposal_scale=None, proposal_energy_tilt_inverse_cm=None,
    gaussian_sigma_cm=21.23, lorentz_hwhm_cm=0.0, kind='emission',
    active_final_modes=None, matrix_zero_tolerance=0.0,
    maximum_grid_entries=8_000_000, maximum_harmonic_pairs=2_000_000,
    cache_gaussian_components=False,
    persistent_cache_path=None, proposal_means=None, defensive_proposal_means=None,
    defensive_fraction=0.1, min_spectator_quanta=0,
    known_support_states=None, known_support_values=None, known_support_model_key=None,
):
    """Estimate the spectator sum with exact active energies and paired reference.

    The finite spectator domain is sum(n_s)<=max_spectator_quanta and
    E_s<energy_max_cm. This is distinct from a bound on total active+spectator
    harmonic occupations. Production excludes pilot samples. Its default 10%
    defensive component uses untilted harmonic number marginals; explicit
    independent-pilot means and a separate broad component are also accepted.
    """
    from matrix_sentinel.quantum_importance import (
        sample_integer_occupation_importance, sample_integer_occupation_residual_importance,
    )

    if isinstance(pilot_rounds, bool) or not isinstance(pilot_rounds, int) or pilot_rounds < 0:
        raise ValueError("nonnegative integer pilot_rounds required")
    provider = VariationalSpectatorObservable(
        model, initial, final, coupling_hartree, kind=kind,
        gaussian_sigma_cm=gaussian_sigma_cm, lorentz_hwhm_cm=lorentz_hwhm_cm,
        active_final_modes=active_final_modes, matrix_zero_tolerance=matrix_zero_tolerance,
        maximum_grid_entries=maximum_grid_entries, maximum_harmonic_pairs=maximum_harmonic_pairs,
        cache_gaussian_components=cache_gaussian_components, persistent_cache_path=persistent_cache_path)
    oriented = provider.oriented
    if (known_support_states is None) != (known_support_values is None):
        raise ValueError('both frozen known support states and values required')
    if known_support_model_key is not None and known_support_model_key != provider.persistent_model_key:
        raise ValueError('known support model key does not match the current observable')
    qmap = (np.sqrt(oriented.final_frequencies_cm)[:, None]*oriented.mode_map
            /np.sqrt(oriented.initial_frequencies_cm)[None, :])
    pmap = np.linalg.inv(qmap).T
    means = np.maximum(0.0, (0.5*np.sum(qmap*qmap, axis=1)+0.5*np.sum(pmap*pmap, axis=1)
                             +oriented.final_shift_dimensionless**2-1)/2)
    active_energy = float(oriented.final_frequencies_cm[provider.active_modes] @ means[provider.active_modes])
    means = means[provider.spectator_modes]
    nu = provider.frequencies_cm
    if energy_max_cm is None:
        energy_max_cm = max(provider.vibronic_origin_cm, provider.harmonic_origin_cm)+8*max(provider.sigma, provider.gamma)
    proposal_tilt = None
    if proposal_means is not None:
        if proposal_scale is not None or proposal_energy_tilt_inverse_cm is not None:
            raise ValueError('explicit proposal means are exclusive of scale and energy tilt')
        proposed = np.asarray(proposal_means, float)
        if proposed.shape != means.shape or not np.all(np.isfinite(proposed)) or np.any(proposed < 0):
            raise ValueError('finite nonnegative explicit spectator proposal means required')
    elif proposal_energy_tilt_inverse_cm is not None:
        if (proposal_scale is not None or not np.isfinite(proposal_energy_tilt_inverse_cm)
                or proposal_energy_tilt_inverse_cm < 0
                or proposal_energy_tilt_inverse_cm*max(nu) > 700):
            raise ValueError('finite nonnegative proposal energy tilt required, exclusive of proposal_scale')
        proposal_tilt = float(proposal_energy_tilt_inverse_cm)
        proposed = means*np.exp(proposal_tilt*nu)
    elif proposal_scale is None:
        target = max(float(nu @ means), max(provider.vibronic_origin_cm, provider.harmonic_origin_cm)-active_energy)
        high = 1.0/float(nu.max())
        expected = lambda tilt: float(nu @ np.maximum(means*np.exp(np.minimum(tilt*nu, 700)), 0.02))
        if np.any(means > 0) and target > expected(0):
            while expected(high) < target:
                high *= 2
            proposal_tilt = float(brentq(lambda tilt: expected(tilt)-target, 0.0, high))
        else:
            proposal_tilt = 0.0
        proposed = means*np.exp(proposal_tilt*nu)
    else:
        if not np.isfinite(proposal_scale) or proposal_scale <= 0:
            raise ValueError("positive finite spectator proposal scale required")
        proposed = means*proposal_scale
    defensive_means = means.copy() if defensive_proposal_means is None else np.asarray(defensive_proposal_means, float)
    if (defensive_means.shape != means.shape or not np.all(np.isfinite(defensive_means))
            or np.any(defensive_means < 0) or not np.isfinite(defensive_fraction)
            or not 0 < defensive_fraction < 1):
        raise ValueError('valid defensive spectator means and fraction strictly between zero and one required')
    common = dict(max_total_quanta=max_spectator_quanta, min_total_quanta=min_spectator_quanta, energy_max_cm=energy_max_cm,
                  batch_size=32)
    history = []
    for step in range(pilot_rounds):
        pilot = sample_integer_occupation_importance(
            nu, np.zeros((len(nu), len(nu))), proposal_means=proposed,
            observables={'anharmonic': provider.anharmonic},
            samples_per_replica=pilot_samples_per_replica, replicas=4,
            seed=seed+104729*(step+1), **common)
        learned = pilot['estimates']['anharmonic']['weighted_occupation_means']
        effective = pilot['estimates']['anharmonic']['effective_sample_size']
        weight = min(0.5, effective/20) if learned is not None else 0.0
        if learned is not None:
            proposed = (1-weight)*pilot['diagnostics']['proposal_means']+weight*learned
        history.append(dict(seed=seed+104729*(step+1),
                            effective_sample_size=effective, fitted_mean_weight=weight))
    estimator = sample_integer_occupation_importance
    known_controls = {}
    if known_support_states is not None:
        estimator = sample_integer_occupation_residual_importance
        known_controls = dict(known_states=known_support_states, known_values=known_support_values)
    sample = estimator(
        nu, np.zeros((len(nu), len(nu))), proposal_means=proposed,
        defensive_means=defensive_means,
        defensive_fraction=defensive_fraction,
        observables={'anharmonic': provider.anharmonic, 'harmonic': provider.harmonic},
        samples_per_replica=samples_per_replica, replicas=replicas, seed=seed, **common, **known_controls)
    estimates = sample['estimates']
    rates = {}
    for name, estimate in estimates.items():
        error = max(estimate['standard_error'], estimate['iid_standard_error'])
        relative = error/estimate['mean'] if estimate['mean'] > 0 else None
        rates[name] = dict(
            rate_s_1=float(GOLDEN_RULE_FACTOR*estimate['mean']),
            standard_error_s_1=float(GOLDEN_RULE_FACTOR*error), relative_standard_error=relative,
            replica_t_interval_95_s_1=[float(GOLDEN_RULE_FACTOR*max(0, estimate['mean']-student_t.ppf(.975, replicas-1)*error)),
                                       float(GOLDEN_RULE_FACTOR*(estimate['mean']+student_t.ppf(.975, replicas-1)*error))],
            effective_sample_size=estimate['effective_sample_size'],
            residual_effective_sample_size=estimate.get('residual_effective_sample_size'),
            known_fraction_of_estimate=estimate.get('known_fraction_of_estimate'),
            maximum_weight_fraction=estimate['maximum_weight_fraction'],
            finite_domain_precision_passed=bool(relative is not None and relative <= .1
                and estimate['effective_sample_size'] >= 100 and estimate['maximum_weight_fraction'] <= .01
                and estimate.get('residual_effective_sample_size', estimate['effective_sample_size']) >= 100))
    av, hv = estimates['anharmonic'], estimates['harmonic']
    ratio = av['mean']/hv['mean'] if hv['mean'] > 0 else None
    ratio_error = None
    if ratio is not None:
        paired = av['replica_estimates']-ratio*hv['replica_estimates']
        ratio_error = float(paired.std(ddof=1)/np.sqrt(replicas)/hv['mean'])
    provider.close()
    return dict(
        schema='matrix.sati.variational_spectator_fgr.v1', rates=rates,
        anharmonic_to_harmonic_ratio=ratio, paired_ratio_standard_error=ratio_error,
        diagnostics={**provider.diagnostics, 'transition_rate_converged': False,
                     'max_spectator_quanta': max_spectator_quanta,
                     'pilot_history': history, 'pilot_samples_used_in_rate': False,
                     'proposal_scale': proposal_scale,
                     'explicit_proposal_means_used': proposal_means is not None,
                     'proposal_energy_tilt_inverse_cm': proposal_tilt,
                     'paired_ratio_error_note': 'delta-method error across paired independent replicas; descriptive until precision gates pass',
                     'scope': 'finite active Hamiltonians and sampled finite spectator domain; active/spectator anharmonic couplings omitted'},
        importance=sample,
    )


__all__ = ['VariationalSpectatorObservable', 'MarginalSpectatorObservable', 'variational_spectator_nonradiative_rate']
