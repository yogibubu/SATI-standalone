"""Fixed mechanical mode blocks and selected, factorized GVPT2 properties.

All indices refer to one dimensionless harmonic modal frame. The partition is
part of the physical approximation and never changes during an evaluation.
Electric derivatives may connect mechanically independent blocks.
"""
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations, permutations
from math import factorial, prod
from threading import Lock

import numpy as np

from .curvilinear_vpt2 import (
    SonicGVPT2Result, SonicVPT2Input, _modalize_sonic_field, solve_sonic_gvpt2,
)
from .gvpt2_spectroscopy import SonicPropertyDerivatives
from matrix_gf.harmonic import CM_PER_HARTREE
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class VibrationalModePartition:
    """Disjoint exhaustive coupled blocks, diagonal modes and harmonic modes.

    Diagonal means a one-mode anharmonic Hamiltonian, including its supplied
    metric derivatives. Coupled blocks retain all intrablock tensor entries.
    Modes are zero based; no coefficient threshold selects or changes a block.
    """
    mode_count: int
    coupled_blocks: tuple = ()
    diagonal_modes: tuple = ()
    harmonic_modes: tuple = ()

    def __post_init__(self):
        n = _integer(self.mode_count, 'mode_count', 1)
        coupled = tuple(tuple(_integer(i, 'mode') for i in b) for b in self.coupled_blocks)
        diagonal = tuple(_integer(i, 'mode') for i in self.diagonal_modes)
        harmonic = tuple(_integer(i, 'mode') for i in self.harmonic_modes)
        modes = [i for b in coupled for i in b] + list(diagonal) + list(harmonic)
        if any(not b for b in coupled) or len(modes) != n or sorted(modes) != list(range(n)):
            raise ValueError('mode classes must partition every mode exactly once')
        object.__setattr__(self, 'mode_count', n)
        object.__setattr__(self, 'coupled_blocks', tuple(sorted(tuple(sorted(b)) for b in coupled)))
        object.__setattr__(self, 'diagonal_modes', tuple(sorted(diagonal)))
        object.__setattr__(self, 'harmonic_modes', tuple(sorted(harmonic)))

    @property
    def blocks(self):
        return tuple(sorted(self.coupled_blocks + tuple((i,) for i in self.diagonal_modes)
                            + tuple((i,) for i in self.harmonic_modes)))

    @property
    def tiers(self):
        return tuple('harmonic' if b[0] in self.harmonic_modes else
                     'diagonal' if b[0] in self.diagonal_modes else 'coupled' for b in self.blocks)

    def as_dict(self):
        return dict(mode_count=self.mode_count, coupled_blocks=self.coupled_blocks,
                    diagonal_modes=self.diagonal_modes, harmonic_modes=self.harmonic_modes)

    @classmethod
    def from_common_blocks(cls, blocks):
        """Reuse the existing common-coordinate vibronic block declaration."""
        from .vibronic_common_basis import _partition
        blocks = _partition(blocks)
        return cls(sum(len(b.modes) for b in blocks),
                   tuple(b.modes for b in blocks if b.tier == 'coupled'),
                   tuple(b.modes[0] for b in blocks if b.tier == 'diagonal'),
                   tuple(b.modes[0] for b in blocks if b.tier == 'harmonic'))

    def merge(self, interacting_sets):
        """Promote complete connected blocks, once, before evaluating a domain."""
        groups = [set(b) for b in self.blocks]
        promoted = set()
        for interaction in interacting_sets:
            modes = {_integer(i, 'interaction mode') for i in interaction}
            if not modes or any(i >= self.mode_count for i in modes):
                raise ValueError('nonempty valid interaction mode sets required')
            touched = [b for b in groups if b & modes]
            merged = set().union(*touched)
            groups = [b for b in groups if not b & merged] + [merged]
            promoted.update(merged)
        return type(self)(self.mode_count,
            tuple(tuple(sorted(b)) for b in groups if len(b) > 1 or b & promoted
                  or tuple(sorted(b)) in self.coupled_blocks),
            tuple(i for i in self.diagonal_modes if i not in promoted),
            tuple(i for i in self.harmonic_modes if i not in promoted))

    def validate_symmetry(self, actions, *, tolerance=1e-7):
        """Allow non-Abelian intrablock actions and permutations of whole blocks."""
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('positive finite symmetry tolerance required')
        blocks, tiers, n = self.blocks, self.tiers, self.mode_count
        harmonic = set(self.harmonic_modes)
        for action in actions:
            action = np.asarray(action, float)
            if (action.shape != (n, n) or not np.all(np.isfinite(action))
                    or not np.allclose(action.T @ action, np.eye(n), atol=tolerance, rtol=0)):
                raise ValueError('orthogonal actions in the common modal frame required')
            if harmonic:
                outside = sorted(set(range(n)) - harmonic)
                if np.any(abs(action[np.ix_(outside, sorted(harmonic))]) > tolerance):
                    raise ValueError('symmetry mixes mode classes; merge the affected blocks')
            for block, tier in zip(blocks, tiers):
                if tier == 'harmonic':
                    continue
                candidates = [b for b, t in zip(blocks, tiers) if t == tier and len(b) == len(block)]
                if not any(np.linalg.norm(action[np.ix_(sorted(set(range(n))-set(b)), block)])
                           <= tolerance for b in candidates):
                    raise ValueError('symmetry mixes mode blocks; merge the affected blocks')

    def mask(self, rank):
        rank = _integer(rank, 'tensor rank', 1)
        mask = np.zeros((self.mode_count,) * rank, bool)
        for block, tier in zip(self.blocks, self.tiers):
            if tier != 'harmonic':
                mask[np.ix_(*(block,) * rank)] = True
        return mask

    def project_mechanics(self, cubic, quartic, metric_first, metric_second):
        arrays, report = [], {}
        for name, value, rank in (('cubic', cubic, 3), ('quartic', quartic, 4),
                                  ('metric_first', metric_first, 3), ('metric_second', metric_second, 4)):
            value = np.asarray(value)
            if value.shape != (self.mode_count,) * rank or not np.all(np.isfinite(value)):
                raise ValueError('complete finite mechanical tensors in the common modal frame required')
            kept = self.mask(rank)
            arrays.append(np.where(kept, value, 0.))
            report[name] = dict(retained_nonzero=int(np.count_nonzero(value[kept])),
                                omitted_nonzero=int(np.count_nonzero(value[~kept])),
                                omitted_norm=float(np.linalg.norm(value[~kept])))
        return tuple(arrays), report

    def project_property(self, derivatives):
        """Keep D1 for every mode; select D2/D3 independently of mechanics."""
        arrays = _property_arrays(derivatives, self.mode_count)
        return SonicPropertyDerivatives(arrays[0].copy(),
            np.where(self.mask(2)[None], arrays[1], 0.),
            np.where(self.mask(3)[None], arrays[2], 0.))


def _property_arrays(derivatives, count):
    if not isinstance(derivatives, SonicPropertyDerivatives):
        raise TypeError('coordinate-property Taylor derivatives required')
    first = np.asarray(derivatives.first)
    components = first.shape[0] if first.ndim == 2 else 0
    if not components or derivatives.fourth is not None:
        raise ValueError('nonempty property components through cubic order required')
    arrays = tuple(np.zeros((components,) + (count,) * rank) if value is None else np.asarray(value)
                   for rank, value in enumerate((first, derivatives.second, derivatives.third), 1))
    if any(a.shape != (components,) + (count,) * r or not np.all(np.isfinite(a))
           for r, a in enumerate(arrays, 1)):
        raise ValueError('finite property tensors in the same modal frame required')
    return arrays


def _modal_field(omega, tensors, coverage):
    scale = HESSIAN_EIGENVALUE_TO_CM**2 / CM_PER_HARTREE
    f3, f4, g1, g2 = tensors
    return SonicVPT2Input(np.diag(omega / CM_PER_HARTREE), f3 / CM_PER_HARTREE,
        f4 / CM_PER_HARTREE, np.diag(omega / scale), g1 / scale, g2 / scale,
        0., quartic_coverage=coverage)


def _column(result, level):
    for block in result.blocks:
        ids = np.asarray(block['indices'], int)
        found = np.flatnonzero(ids == level)
        if len(found):
            values = np.asarray(block['eigenvectors'])[:, block['basis_to_root'][int(found[0])]]
            nonzero = values != 0
            return ids[nonzero], values[nonzero]
    return np.array([level]), np.array([1.])


@dataclass(frozen=True)
class BlockVibrationalModel:
    partition: VibrationalModePartition
    results: tuple
    fields: tuple
    modalized: tuple
    coordinate_id: str
    pseudopotential_cm: float
    diagnostics: dict
    symmetry_actions: tuple = ()
    symmetry_tolerance: float = 1e-7

    def __post_init__(self):
        # These arrays are produced by the local solves, not caller-owned inputs.
        # Read-only numerical data protect cached local moment contractions.
        for modal, field, result in zip(self.modalized, self.fields, self.results):
            for array in (*modal, field.harmonic_sonic, field.cubic_sonic, field.quartic_sonic,
                          field.metric_sonic, field.metric_first_sonic, field.metric_second_sonic,
                          result.frequencies_cm, result.energies_cm, result.deperturbed_cm,
                          result.fundamentals_cm):
                np.asarray(array).setflags(write=False)
            for block in result.blocks:
                for key in ('hamiltonian_cm', 'eigenvalues_cm', 'eigenvectors'):
                    if key in block:
                        np.asarray(block[key]).setflags(write=False)

    @property
    def model_id(self):
        """Identify the actual retained operators and local assigned states."""
        import hashlib
        import json
        digest = hashlib.sha256(b'mode-block-gvpt2-v1')
        digest.update(self.coordinate_id.encode())
        digest.update(json.dumps(self.partition.as_dict(), sort_keys=True).encode())
        digest.update(np.asarray(self.pseudopotential_cm, dtype='<f8').tobytes())
        for modal, result in zip(self.modalized, self.results):
            arrays = (*modal, np.asarray(result.states), result.energies_cm, result.deperturbed_cm)
            for array in arrays:
                a = np.asarray(array)
                dtype = '<i8' if a.dtype.kind in 'iu' else '<c16' if a.dtype.kind == 'c' else '<f8'
                digest.update(str(a.shape).encode()); digest.update(np.asarray(a, dtype=dtype).tobytes())
            for block in result.blocks:
                for key in ('indices', 'basis_to_root', 'eigenvectors', 'direct_resonance_states'):
                    a = np.asarray(block.get(key, ()))
                    digest.update(key.encode()); digest.update(str(a.shape).encode())
                    digest.update(np.asarray(a, dtype='<c16' if key == 'eigenvectors' else '<i8').tobytes())
        return digest.hexdigest()

    @property
    def zero_point_cm(self):
        return self.pseudopotential_cm + sum(r.zero_point_cm for r in self.results)

    def configurations(self, configurations):
        configurations = np.asarray(configurations)
        if (configurations.ndim != 2 or configurations.shape[1] != len(self.results)
                or configurations.dtype.kind not in 'iu' or np.any(configurations < 0)
                or any(np.any(configurations[:, b] >= len(r.states)) for b, r in enumerate(self.results))):
            raise ValueError('one valid local eigenlevel index per block required')
        return configurations

    def energies(self, configurations):
        configurations = self.configurations(configurations)
        return sum((r.energies_cm[configurations[:, b]] for b, r in enumerate(self.results)),
                   np.zeros(len(configurations)))

    def states_below(self, energy_cutoff_cm, *, maximum_states=30000, maximum_frontier=100000):
        """Enumerate only requested finite product roots in increasing energy.

        Local roots are energy sorted temporarily; returned labels retain the
        existing GVPT2 assignments. No product Hamiltonian or full Cartesian
        list is built. Exhausting a resource cap raises instead of truncating.
        """
        from heapq import heappop, heappush
        cutoff = float(energy_cutoff_cm)
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError('finite nonnegative energy cutoff required')
        maximum_states = _integer(maximum_states, 'maximum_states', 1)
        maximum_frontier = _integer(maximum_frontier, 'maximum_frontier', 1)
        ordering = [np.argsort(r.energies_cm, kind='stable') for r in self.results]
        energies = [r.energies_cm[ids] for r, ids in zip(self.results, ordering)]
        origin = (0,)*len(energies)
        heap = [(float(sum(e[0] for e in energies)), origin)]
        seen = {origin}; labels = []; values = []
        while heap:
            energy, state = heappop(heap)
            if energy > cutoff:
                break
            if len(labels) >= maximum_states:
                raise MemoryError('product-state output budget exceeded')
            labels.append(tuple(int(ids[j]) for ids, j in zip(ordering, state)))
            values.append(energy)
            for b, e in enumerate(energies):
                if state[b]+1 >= len(e):
                    continue
                following = list(state); following[b] += 1; following = tuple(following)
                if following in seen:
                    continue
                candidate = float(sum(e[j] for e, j in zip(energies, following)))
                if candidate > cutoff:
                    continue
                if len(seen) >= maximum_frontier:
                    raise MemoryError('product-state frontier budget exceeded')
                seen.add(following); heappush(heap, (candidate, following))
        return np.asarray(labels, int).reshape(-1, len(energies)), np.asarray(values)

    def properties(self, derivatives, *, coordinate_id, property_partition=None, **controls):
        if coordinate_id != self.coordinate_id:
            raise ValueError('property and mechanical model must share their modal coordinate_id')
        if property_partition is not None:
            if not isinstance(property_partition, VibrationalModePartition) or property_partition.mode_count != self.partition.mode_count:
                raise ValueError('compatible independent property partition required')
            property_partition.validate_symmetry(self.symmetry_actions, tolerance=self.symmetry_tolerance)
            derivatives = property_partition.project_property(derivatives)
        return BlockTransitionProperty(self, derivatives, **controls)


def solve_mode_blocks(field, partition, *, coordinate_id, max_quanta=3,
                      max_states_per_block=30000, solver_options=None,
                      candidate_states_per_block=None):
    """Solve independent GVPT2 blocks without constructing product states.

    All field and metric indices are partitioned AFTER common GF modalization.
    Harmonic members use analytic ladders. Electrical derivatives are supplied
    separately to the returned model and are never mechanically masked.
    Optional local candidate bases follow ``partition.blocks`` and are checked
    by the local solver. They permit comparison on an archived finite support
    without generating larger local bases. Harmonic blocks keep analytic ladders.
    """
    if not isinstance(partition, VibrationalModePartition):
        raise TypeError('VibrationalModePartition required')
    if not isinstance(coordinate_id, str) or not coordinate_id.strip():
        raise ValueError('identified common modal coordinates required')
    max_quanta = _integer(max_quanta, 'max_quanta', 1)
    max_states_per_block = _integer(max_states_per_block, 'max_states_per_block', 2)
    options = dict(solver_options or {})
    if set(options) & {'max_quanta', 'max_states', '_workspace', 'chart_connection', 'symmetry_actions',
                       'candidate_states', 'polyad_excluded_modes', 'targeted_polyad_options'}:
        raise ValueError('global basis/gauge/refinement options cannot be silently reused for local blocks')
    from inspect import signature
    signature(solve_sonic_gvpt2).bind_partial(field, **options)
    if options.get('resonance_policy', 'auto') not in {'auto', 'all', 'fermi', 'fermi_auto', 'gdv', 'intensity', 'none'}:
        raise ValueError('unsupported local resonance policy')
    modal = _modalize_sonic_field(field)
    omega, modes = modal[:2]
    if len(omega) != partition.mode_count:
        raise ValueError('partition and common modal field dimensions differ')
    actions = ()
    if field.symmetry_actions is not None:
        # s=modes/gamma*q; diagonal scaling is removed in the modal group action.
        gamma = np.sqrt(omega * CM_PER_HARTREE) / HESSIAN_EIGENVALUE_TO_CM
        displacement = modes / gamma[None]
        actions = tuple(np.linalg.solve(displacement, a @ displacement) for a in field.symmetry_actions)
        partition.validate_symmetry(actions, tolerance=field.symmetry_tolerance)
    projected, audit = partition.project_mechanics(*modal[2:])
    if candidate_states_per_block is not None:
        candidate_states_per_block = tuple(candidate_states_per_block)
        if len(candidate_states_per_block) != len(partition.blocks):
            raise ValueError('one candidate basis per mechanical block required')
    results, fields, local_modal = [], [], []
    for block, tier in zip(partition.blocks, partition.tiers):
        local_omega = omega[list(block)]
        tensors = tuple(a[np.ix_(*(block,) * a.ndim)] for a in projected)
        local_field = _modal_field(local_omega, tensors, field.quartic_coverage)
        local = (local_omega, np.eye(len(block)), *tensors)
        if tier == 'harmonic':
            q = min(max_quanta, options.get('max_mode_quanta', max_quanta))
            q = _integer(q, 'harmonic maximum quantum', 1)
            if q + 1 > max_states_per_block:
                raise MemoryError('harmonic block exceeds local state budget')
            energies = np.arange(q + 1) * local_omega[0]
            result = SonicGVPT2Result(local_omega, tuple((i,) for i in range(q + 1)),
                energies.copy(), energies, local_omega.copy(), (), (), float(local_omega[0] / 2),
                'Analytic harmonic block', diagnostics={'analytic_harmonic': True})
        else:
            candidates = (None if candidate_states_per_block is None else
                          candidate_states_per_block[len(results)])
            result = solve_sonic_gvpt2(local_field, max_quanta=max_quanta,
                max_states=max_states_per_block, candidate_states=candidates,
                _workspace={'modalized': local}, **options)
        results.append(result); fields.append(local_field); local_modal.append(local)
    diagnostics = dict(partition=partition.as_dict(), tiers=partition.tiers, mechanical_projection=audit,
        block_state_counts=tuple(len(r.states) for r in results),
        product_state_count=prod(len(r.states) for r in results), full_product_basis_constructed=False,
        property_interblock_terms_retained_by_default=True, fixed_partition=True)
    return BlockVibrationalModel(partition, tuple(results), tuple(fields), tuple(local_modal),
                                 coordinate_id, float(field.pseudopotential_cm), diagnostics, actions,
                                 field.symmetry_tolerance)


class BlockTransitionProperty:
    """Coherent PT2 Taylor-property moments between selected product roots.

    A rank-k electric monomial starts at perturbative order k-1. Mechanical
    orders from its local factors are convolved only up to total order two.
    Untouched blocks contribute exact orthogonality, including hot states.
    """
    def __init__(self, model, derivatives, *, maximum_terms=100000,
                 maximum_bare_pairs=100000, maximum_workspace_bytes=512*1024**2,
                 cached_local_pairs=256):
        from .native_resonant_property import NativeResonantProperty
        self.model = model
        arrays = _property_arrays(derivatives, model.partition.mode_count)
        import hashlib
        digest = hashlib.sha256(model.model_id.encode())
        for array in arrays:
            digest.update(np.asarray(array, dtype='<c16').tobytes())
        self.property_id = digest.hexdigest()
        self.components = len(arrays[0])
        self.maximum_bare_pairs = _integer(maximum_bare_pairs, 'maximum_bare_pairs', 1)
        self.cached_local_pairs = _integer(cached_local_pairs, 'cached_local_pairs', 1)
        maximum_terms = _integer(maximum_terms, 'maximum_terms', 1)
        maximum_workspace_bytes = _integer(maximum_workspace_bytes, 'maximum_workspace_bytes', 1)
        self._cache = OrderedDict()
        self._cache_lock = Lock()
        mapping = {mode: (b, j) for b, block in enumerate(model.partition.blocks) for j, mode in enumerate(block)}
        terms = {}
        for rank, array in enumerate(arrays, 1):
            for indices in np.argwhere(np.any(array != 0, axis=0)):
                key = tuple(sorted(map(int, indices)))
                if key not in terms and len(terms) >= maximum_terms:
                    raise MemoryError('property monomial budget exceeded')
                terms[key] = terms.get(key, 0) + array[(slice(None), *indices)] / factorial(rank)
        terms = {key: value for key, value in terms.items() if np.any(value)}
        factors = [set() for _ in model.results]
        term_factors = []
        for modes, coefficient in terms.items():
            local = {}
            for mode in modes:
                b, j = mapping[mode]; local.setdefault(b, []).append(j)
            local = tuple((b, tuple(ids)) for b, ids in sorted(local.items()))
            for b, ids in local:
                factors[b].add(ids)
            term_factors.append((len(modes)-1, coefficient, local))
        self.factor_ids = [{ids: j for j, ids in enumerate(sorted(fs))} for fs in factors]
        self.terms = tuple(term_factors)
        self._by_changed = {}
        for term in self.terms:
            active = tuple(b for b, _ in term[2])
            for count in range(len(active)+1):
                for changed in combinations(active, count):
                    self._by_changed.setdefault(changed, []).append(term)
        self.providers = []
        # The same local mechanical polynomials serve all local Q monomials.
        # Guard their conservative dense packed-storage upper bound beforehand.
        from math import comb
        estimate = sum(comb(2*len(block)+4, 4)*max(1, len(fs))*16
                       for block, fs in zip(model.partition.blocks, factors))
        if estimate > maximum_workspace_bytes:
            raise MemoryError('local property factors exceed packed-storage budget')
        for b, (block, ids) in enumerate(zip(model.partition.blocks, self.factor_ids)):
            n = len(block); count = len(ids)
            if not count:
                self.providers.append(None); continue
            local_arrays = [np.zeros((count,) + (n,)*rank) for rank in (1, 2, 3)]
            for word, component in ids.items():
                multiplicity = prod(factorial(word.count(i)) for i in set(word))
                for order in set(permutations(word)):
                    local_arrays[len(word)-1][(component, *order)] = multiplicity
            provider = NativeResonantProperty.from_gvpt2(model.results[b], model.fields[b],
                SonicPropertyDerivatives(*local_arrays), modalized=model.modalized[b])
            self.providers.append(provider)

    def _local_moments(self, block, initial, final):
        key = block, int(initial), int(final)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        result = self.model.results[block]
        ni, ci = _column(result, initial); nf, cf = _column(result, final)
        if len(ni)*len(nf) > self.maximum_bare_pairs:
            raise MemoryError('local coherent pair budget exceeded')
        left = np.repeat(ni, len(nf)); right = np.tile(nf, len(ni))
        n = np.asarray([result.states[i] for i in left]); f = np.asarray([result.states[i] for i in right])
        reverse = np.array([(sum(b), int(j)) < (sum(a), int(i))
                            for a, b, i, j in zip(n, f, left, right)])
        n0, f0 = n.copy(), f.copy(); n0[reverse], f0[reverse] = f[reverse], n[reverse]
        values = self.providers[block].moments(n0, f0)
        values[reverse] = values[reverse].conj()
        weights = np.repeat(ci, len(nf))*np.tile(cf.conj(), len(ni))
        value = np.einsum('p,prc->rc', weights, values)
        with self._cache_lock:
            self._cache[key] = value
            while len(self._cache) > self.cached_local_pairs:
                self._cache.popitem(last=False)
        return value

    def moments(self, initial, final):
        initial = self.model.configurations(initial); final = self.model.configurations(final)
        if initial.shape != final.shape:
            raise ValueError('paired product configurations required')
        values = np.zeros((len(initial), 3, self.components), complex)
        for row, (n, f) in enumerate(zip(initial, final)):
            local = {}
            changed = tuple(map(int, np.flatnonzero(n != f)))
            for order, coefficient, factors in self._by_changed.get(changed, ()):
                amplitude = np.zeros(3-order, complex); amplitude[0] = 1.
                for b, ids in factors:
                    if b not in local:
                        local[b] = self._local_moments(b, n[b], f[b])
                    start = len(ids)-1
                    series = local[b][start:start+3-order, self.factor_ids[b][ids]]
                    amplitude = np.convolve(amplitude, series)[:3-order]
                values[row, order:] += amplitude[:, None]*coefficient[None]
        return values

    def emission_rates(self, initial, final):
        from .quantum_dynamics import EINSTEIN_PER_CM3_AU2
        initial = self.model.configurations(initial); final = self.model.configurations(final)
        moments = self.moments(initial, final).sum(axis=1)
        gaps = self.model.energies(initial)-self.model.energies(final)
        return EINSTEIN_PER_CM3_AU2*np.maximum(gaps, 0.)**3*np.sum(abs(moments)**2, axis=1)
