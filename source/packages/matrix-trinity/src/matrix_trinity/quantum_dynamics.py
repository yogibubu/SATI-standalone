"""Quantum dynamics from operator actions, with no transition-pair inventory.

Hamiltonians use cm^-1, propagation times fs, dipoles atomic units, radiative
waiting times s. The owner of an effective Hamiltonian supplies its consistent
effective property and coordinate identity; this module does not change the
perturbative partition or infer operators from scalar fitted energies.
"""
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from math import ceil
from threading import Lock

import numpy as np
from numba import njit
from scipy import fft
from scipy.constants import c, epsilon_0, hbar, elementary_charge, physical_constants
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.special import gammaln, jv

PHASE_PER_CM_FS = 2*np.pi*c*100*1e-15
EINSTEIN_PER_CM3_AU2 = ((2*np.pi*c*100)**3
    * (elementary_charge*physical_constants['Bohr radius'][0])**2
    / (3*np.pi*epsilon_0*hbar*c**3))


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'positive integer {name} required')
    return int(value)


def _operator(value):
    operator = aslinearoperator(value)
    if len(operator.shape) != 2 or operator.shape[0] != operator.shape[1] or not operator.shape[0]:
        raise ValueError('nonempty square operator required')
    return operator


def _action_operator(shape, action, adjoint=None):
    def matvec(x):
        return action(np.asarray(x).reshape(shape[1], -1))[:, 0]
    def matmat(x):
        return action(np.asarray(x))
    kwargs = {}
    if adjoint is not None:
        kwargs['rmatvec'] = lambda x: adjoint(np.asarray(x).reshape(shape[0], -1))[:, 0]
    return LinearOperator(shape, matvec=matvec, matmat=matmat, dtype=complex, **kwargs)


def contact_dynamics_property(orders, generator_first, generator_second, *, embedding=None):
    """Apply e^-S O e^S through order two by nested operator actions.

    All factors use one common (possibly augmented) space. Projection onto the
    retained space is applied only after composing the factors: intermediate
    external states are preserved. Completeness of the supplied augmented
    space and the resonance convention belong to the generator provider.
    Returns the three separate perturbative orders, preserving interference.
    """
    p0, p1, p2 = map(_operator, orders)
    s1, s2 = map(_operator, (generator_first, generator_second))
    if any(o.shape != p0.shape for o in (p1, p2, s1, s2)):
        raise ValueError('all property and generator factors must share one space')
    # Algebra is valid for arbitrary representations. Check the physical
    # adjoint contract on fixed probes without forming a square matrix.
    rng = np.random.default_rng(729)
    probe = rng.normal(size=(p0.shape[0], 2)) + 1j*rng.normal(size=(p0.shape[0], 2))
    for op, sign in ((p0, 1), (p1, 1), (p2, 1), (s1, -1), (s2, -1)):
        value = op.matmat(probe)
        gram = probe.conj().T @ value
        if not np.allclose(gram, sign*gram.conj().T, atol=1e-11, rtol=1e-11):
            raise ValueError('Hermitian properties and anti-Hermitian generators required')
    def comm(a, b):
        return lambda x: a.matmat(b.matmat(x))-b.matmat(a.matmat(x))
    first_comm = _action_operator(p0.shape, comm(p0, s1))
    actions = (p0.matmat,
               lambda x: p1.matmat(x)+first_comm.matmat(x),
               lambda x: (p2.matmat(x)+comm(p1, s1)(x)+comm(p0, s2)(x)
                          +.5*comm(first_comm, s1)(x)))
    if embedding is not None:
        e = aslinearoperator(embedding)
        if e.shape[0] != p0.shape[0] or e.shape[1] < 1:
            raise ValueError('embedding must map the retained into the augmented space')
        actions = tuple((lambda x, action=action: e.H.matmat(action(e.matmat(x))))
                        for action in actions)
        shape = (e.shape[1], e.shape[1])
    else:
        shape = p0.shape
    return tuple(_action_operator(shape, action, action) for action in actions)


def sum_dynamics_operators(operators):
    operators = tuple(map(_operator, operators))
    if not operators or any(o.shape != operators[0].shape for o in operators):
        raise ValueError('nonempty compatible operators required')
    return _action_operator(operators[0].shape,
        lambda x: sum((o.matmat(x) for o in operators)),
        lambda x: sum((o.H.matmat(x) for o in operators)))


@njit(cache=True, nogil=True)
def _ladder_action(basis, sorted_rows, pointers, modes, creation, annihilation, coefficients, x):
    result = np.zeros((len(basis), x.shape[1]), dtype=np.complex128)
    target = np.empty(basis.shape[1], dtype=np.int64)
    for source in range(len(basis)):
        active = False
        for column in range(x.shape[1]):
            active = active or x[source, column] != 0
        if not active:
            continue
        for term in range(len(coefficients)):
            target[:] = basis[source]
            factor = 1.
            for k in range(pointers[term], pointers[term+1]):
                mode, r, s = modes[k], creation[k], annihilation[k]
                occupation = basis[source, mode]
                if occupation < s:
                    factor = 0.
                    break
                for j in range(s):
                    factor *= np.sqrt(float(occupation-j))
                for j in range(1, r+1):
                    factor *= np.sqrt(float(occupation-s+j))
                target[mode] += r-s
            if factor == 0:
                continue
            low, high = 0, len(basis)
            destination = -1
            while low < high:
                mid = (low+high)//2
                row = sorted_rows[mid]
                comparison = 0
                for mode in range(basis.shape[1]):
                    if basis[row, mode] < target[mode]:
                        comparison = -1
                        break
                    if basis[row, mode] > target[mode]:
                        comparison = 1
                        break
                if comparison == 0:
                    destination = row
                    break
                if comparison < 0:
                    low = mid+1
                else:
                    high = mid
            if destination >= 0:
                value = coefficients[term]*factor
                for column in range(x.shape[1]):
                    result[destination, column] += value*x[source, column]
    return result


class LadderDynamicsOperator(LinearOperator):
    """Compiled P O P action of an existing LadderPolynomial on a declared basis.

    No pair list or square matrix is built. Finite-basis projection is explicit;
    use an augmented basis before composing contact generators. Work limits
    raise instead of removing small coefficients or connected transitions.
    """
    def __init__(self, polynomial, basis, *, maximum_term_actions=500000000,
                 maximum_workspace_bytes=256*1024**2):
        from .occupation_operators import LadderPolynomial
        if not isinstance(polynomial, LadderPolynomial):
            raise TypeError('LadderPolynomial required')
        a = np.asarray(basis)
        if (a.ndim != 2 or a.dtype.kind not in 'iu' or not len(a)
                or a.shape[1] != polynomial.mode_count or np.any(a < 0)
                or len({tuple(row) for row in a}) != len(a)
                or np.max(a) > np.iinfo(np.int64).max//4):
            raise ValueError('unique nonnegative occupation basis required')
        self.basis = a.astype(np.int64, copy=True)
        self.basis.setflags(write=False)
        self.polynomial = polynomial
        self.coordinate_id = polynomial.coordinate_id
        self.maximum_term_actions = _positive_integer(maximum_term_actions, 'maximum_term_actions')
        self.maximum_workspace_bytes = _positive_integer(maximum_workspace_bytes, 'maximum_workspace_bytes')
        self.actions_per_source = sum(1+sum(r)+sum(s) for r, s in polynomial.terms)
        self.sorted_rows = np.asarray(sorted(range(len(a)), key=lambda i: tuple(a[i])), np.int64)
        def packed(terms):
            pointers = [0]; modes = []; creation = []; annihilation = []; coefficients = []
            for (r, s), coefficient in terms.items():
                for mode in range(polynomial.mode_count):
                    if r[mode] or s[mode]:
                        modes.append(mode); creation.append(r[mode]); annihilation.append(s[mode])
                pointers.append(len(modes)); coefficients.append(coefficient)
            return (*(np.asarray(v, np.int64) for v in (pointers, modes, creation, annihilation)),
                    np.asarray(coefficients, complex))
        self.forward = packed(polynomial.terms)
        self.backward = packed(polynomial.adjoint().terms)
        self.diagnostics = dict(pair_table_constructed=False, compiled=True,
                                projected_finite_basis=True, coordinate_id=self.coordinate_id)
        super().__init__(dtype=np.dtype(complex), shape=(len(a), len(a)))

    def _apply(self, x, terms):
        x = np.asarray(x, complex).reshape(self.shape[0], -1)
        if not np.all(np.isfinite(x)):
            raise ValueError('finite wavepacket required')
        active = int(np.count_nonzero(np.any(x != 0, axis=1)))
        if active*self.actions_per_source*x.shape[1] > self.maximum_term_actions:
            raise MemoryError('ladder operator action exceeds declared work budget')
        if (3*x.nbytes+self.basis.nbytes+self.sorted_rows.nbytes
                +sum(a.nbytes for a in self.forward+self.backward)) > self.maximum_workspace_bytes:
            raise MemoryError('ladder operator action exceeds workspace budget')
        result = _ladder_action(self.basis, self.sorted_rows, *terms, x)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('ladder action exceeds numerical range')
        return result

    def _matvec(self, x):
        return self._apply(x, self.forward)[:, 0]

    def _matmat(self, x):
        return self._apply(x, self.forward)

    def _rmatvec(self, x):
        return self._apply(x, self.backward)[:, 0]


def _chebyshev_degree(radius, tolerance, maximum_degree):
    if radius == 0:
        return 0, 0.
    degree = max(1, int(ceil(radius/2)))
    while degree <= maximum_degree:
        ratio = radius/(2*(degree+2))
        if ratio < 1:
            log_bound = (np.log(2)+(degree+1)*np.log(radius/2)-gammaln(degree+2)
                         -np.log1p(-ratio))
            if log_bound <= np.log(tolerance):
                return degree, float(np.exp(log_bound))
        degree += 1
    raise MemoryError('Chebyshev propagation exceeds degree budget')


class QuantumDynamics:
    """Hermitian wavepacket propagation and correlation-only Chebyshev sums."""
    def __init__(self, hamiltonian_cm, *, energy_bounds_cm, coordinate_id,
                 maximum_degree=32768, maximum_workspace_bytes=256*1024**2,
                 maximum_matvec_work=2000000):
        self.hamiltonian = _operator(hamiltonian_cm)
        bounds = np.asarray(energy_bounds_cm, float)
        if (bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]
                or not isinstance(coordinate_id, str) or not coordinate_id.strip()):
            raise ValueError('ordered finite spectral bounds and coordinate_id required')
        if not isinstance(hamiltonian_cm, LinearOperator):
            matrix = hamiltonian_cm if issparse(hamiltonian_cm) else np.asarray(hamiltonian_cm)
            data = matrix.data if issparse(matrix) else matrix
            if not np.all(np.isfinite(data)):
                raise ValueError('finite Hamiltonian required')
            difference = matrix-matrix.conj().T
            values = difference.data if issparse(difference) else difference
            scale = np.max(abs(data), initial=1.)
            if np.max(abs(values), initial=0.) > 1e-12*scale:
                raise ValueError('Hermitian Hamiltonian required')
        self.bounds = tuple(bounds)
        self.coordinate_id = coordinate_id
        self.maximum_degree = _positive_integer(maximum_degree, 'maximum_degree')
        self.maximum_workspace_bytes = _positive_integer(maximum_workspace_bytes, 'maximum_workspace_bytes')
        self.maximum_matvec_work = _positive_integer(maximum_matvec_work, 'maximum_matvec_work')
        self._diagonal_energies = None
        # Existing energy-basis adapters use scipy's DIA storage. Recognize
        # that representation without scanning/converting a general sparse H.
        if (issparse(hamiltonian_cm) and hamiltonian_cm.format == 'dia'
                and all(offset == 0 or not np.any(row)
                        for offset, row in zip(hamiltonian_cm.offsets, hamiltonian_cm.data))
                and np.all(np.isreal(hamiltonian_cm.diagonal()))):
            energies = np.asarray(hamiltonian_cm.diagonal()).real.copy()
            if np.any(energies < bounds[0]) or np.any(energies > bounds[1]):
                raise ValueError('spectral bounds must enclose every diagonal energy')
            energies.setflags(write=False)
            self._diagonal_energies = energies

    def _prepare(self, packets, times_fs, tolerance, store):
        x = np.asarray(packets, complex)
        single = x.ndim == 1
        if single:
            x = x[:, None]
        t = np.asarray(times_fs, float)
        if (x.ndim != 2 or x.shape[0] != self.hamiltonian.shape[0] or not x.shape[1]
                or not np.all(np.isfinite(x)) or t.ndim != 1 or not len(t)
                or not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(np.diff(t) <= 0)
                or not np.isfinite(tolerance) or not 0 < tolerance < 1):
            raise ValueError('finite packets, increasing nonnegative times and valid tolerance required')
        center, half = sum(self.bounds)/2, (self.bounds[1]-self.bounds[0])/2
        phase = PHASE_PER_CM_FS*t
        degree, bound = ((0, 0.) if self._diagonal_energies is not None else
                         _chebyshev_degree(half*phase[-1], tolerance, self.maximum_degree))
        if degree*x.shape[1] > self.maximum_matvec_work:
            raise MemoryError('Chebyshev operator-action budget exceeded')
        output_bytes = len(t)*x.shape[1]*16*(x.shape[0] if store else 1)
        if output_bytes+6*x.nbytes+64*len(t) > self.maximum_workspace_bytes:
            raise MemoryError('propagation exceeds workspace budget; request correlations or smaller batches')
        return x, t, phase, center, half, degree, bound, single

    def _sum(self, packets, times_fs, tolerance, store):
        x, t, phase, center, half, degree, bound, single = self._prepare(packets, times_fs, tolerance, store)
        output = np.zeros((len(t), *x.shape) if store else (len(t), x.shape[1]), complex)
        previous, current = x, None
        norm = np.sum(abs(x)**2, axis=0)
        if self._diagonal_energies is not None:
            # Exact spectral phases; no polynomial approximation is needed.
            # Bound phase-table memory independently of the time-grid length.
            available = self.maximum_workspace_bytes-output.nbytes-3*x.nbytes-64*len(t)
            per_time = 16*(3*len(x)+(len(x)*x.shape[1] if store else x.shape[1]))
            batch = min(len(t), available//per_time)
            if batch < 1:
                raise MemoryError('diagonal phase workspace exceeds declared budget')
            strengths = abs(x)**2
            for first in range(0, len(t), batch):
                last = min(first+batch, len(t))
                phases = np.exp(-1j*phase[first:last, None]*self._diagonal_energies[None, :])
                output[first:last] = (phases[:, :, None]*x[None, :, :] if store
                                      else phases@strengths)
            if single:
                output = output[..., 0]
            return dict(values=output, times_fs=t,
                diagnostics=dict(degree=0, operator_vector_products=0,
                    relative_propagator_error_bound=0., spectral_bounds_cm=self.bounds,
                    spectral_bounds_supplied_by_caller=True, coordinate_id=self.coordinate_id,
                    pair_table_constructed=False, stores_time_dependent_wavefunctions=store,
                    backend='analytic diagonal phases', phase_batch_size=int(batch),
                    propagation_bound_excludes_roundoff=True))
        if half == 0 and not np.allclose(self.hamiltonian.matmat(x), center*x, atol=1e-12, rtol=1e-12):
            raise ValueError('zero-width bounds do not enclose the wavepacket spectrum')
        for k in range(degree+1):
            if k == 0:
                value = previous
            elif k == 1:
                current = (self.hamiltonian.matmat(x)-center*x)/half
                value = current
            else:
                following = 2*(self.hamiltonian.matmat(current)-center*current)/half-previous
                previous, current = current, following
                value = current
            if not np.all(np.isfinite(value)) or np.any(np.sum(abs(value)**2, axis=0) > 1.00001*norm+1e-25):
                raise ArithmeticError('Chebyshev recurrence unstable; verify spectral enclosure and Hermiticity')
            coefficient = (1 if k == 0 else 2)*(-1j)**k*jv(k, half*phase)
            if store:
                output += coefficient[:, None, None]*value[None, :, :]
            else:
                moments = np.sum(x.conj()*value, axis=0)
                output += coefficient[:, None]*moments[None, :]
        output *= np.exp(-1j*center*phase).reshape((-1, 1, 1) if store else (-1, 1))
        if single:
            output = output[..., 0]
        return dict(values=output, times_fs=t,
                    diagnostics=dict(degree=degree, operator_vector_products=degree*x.shape[1],
                        relative_propagator_error_bound=bound, spectral_bounds_cm=self.bounds,
                        spectral_bounds_supplied_by_caller=True, coordinate_id=self.coordinate_id,
                        pair_table_constructed=False, stores_time_dependent_wavefunctions=store))

    def propagate(self, packets, times_fs, *, tolerance=1e-10):
        return self._sum(packets, times_fs, tolerance, True)

    def correlate(self, packets, times_fs, *, tolerance=1e-10, initial_energy_cm=0.):
        if not np.isfinite(initial_energy_cm):
            raise ValueError('finite initial energy required')
        result = self._sum(packets, times_fs, tolerance, False)
        phase = np.exp(1j*initial_energy_cm*PHASE_PER_CM_FS*result['times_fs'])
        result['values'] *= phase if result['values'].ndim == 1 else phase[:, None]
        result['initial_energy_cm'] = float(initial_energy_cm)
        return result


def correlation_emission_spectrum(correlation, *, gaussian_sigma_cm, fft_workers=1):
    """Positive-frequency emission strength from C(t)=sum S_if exp(+i gap*t).

    Returns strengths and their omega^3-weighted photon rates. Gaussian
    broadening precedes the rate weighting, so the integrated rate is a
    broadened diagnostic; resolved jump rates use exact gaps instead.
    Negative numerical densities and finite-window tails are reported intact.
    """
    times = np.asarray(correlation['times_fs'], float)
    signal = np.asarray(correlation['values'], complex)
    if signal.ndim == 1:
        signal = signal[:, None]
    if (len(times) < 3 or times[0] != 0 or signal.shape[0] != len(times)
            or not np.all(np.isfinite(signal)) or not np.isfinite(gaussian_sigma_cm)
            or gaussian_sigma_cm <= 0 or not np.allclose(np.diff(times), times[1], rtol=1e-10, atol=0)):
        raise ValueError('uniform time grid from zero and positive Gaussian width required')
    _positive_integer(fft_workers, 'fft_workers')
    step = PHASE_PER_CM_FS*times[1]
    damping = np.exp(-.5*(gaussian_sigma_cm*PHASE_PER_CM_FS*times)**2)
    density = fft.fftshift(fft.hfft((signal*damping[:, None]).T,
        n=2*(len(times)-1), axis=-1, workers=fft_workers), axes=-1).T*step/(2*np.pi)
    grid = fft.fftshift(2*np.pi*fft.fftfreq(2*(len(times)-1), d=step))
    rate = EINSTEIN_PER_CM3_AU2*np.maximum(grid, 0.)[:, None]**3*density
    spacing = grid[1]-grid[0]
    reference = np.maximum(abs(signal[0]), np.finfo(float).tiny)
    return dict(wavenumbers_cm=grid, strength_density_au2_per_cm=density,
                photon_rate_density_s_inverse_per_cm=rate,
                diagnostics=dict(frequency_step_cm=float(spacing), nyquist_cm=float(np.pi/step),
                    integrated_strength_au2=(density.sum(axis=0)*spacing).tolist(),
                    zero_time_strength_au2=signal[0].real.tolist(),
                    negative_strength_area_au2=(-np.minimum(density, 0).sum(axis=0)*spacing).tolist(),
                    window_tail_ratio=(abs(signal[-1]*damping[-1])/reference).tolist(),
                    gaussian_sigma_cm=float(gaussian_sigma_cm), fft_workers=fft_workers))


class SecularRadiativeDynamics:
    """Event-driven spontaneous emission in a supplied energy eigenbasis.

    Each visited source needs three dipole-vector actions, not a complete
    pair inventory. The secular population model projects each emission onto
    an energy-resolved final state. It assumes no instantaneous thermalization
    and does not evolve optical coherences. All retained final endpoints are
    considered, including hot bands, overtones and combinations.
    """
    def __init__(self, energies_cm, dipoles_au, *, coordinate_id, cached_sources=128,
                 maximum_source_evaluations=4096):
        e = np.asarray(energies_cm, float)
        operators = tuple(map(_operator, dipoles_au))
        if (e.ndim != 1 or not len(e) or not np.all(np.isfinite(e)) or len(operators) != 3
                or any(o.shape != (len(e), len(e)) for o in operators)
                or not isinstance(coordinate_id, str) or not coordinate_id.strip()):
            raise ValueError('finite eigenenergies, three compatible atomic-unit dipoles and coordinate_id required')
        self.energies = e.copy(); self.energies.setflags(write=False)
        self.dipoles = operators; self.coordinate_id = coordinate_id
        self.cached_sources = _positive_integer(cached_sources, 'cached_sources')
        self._cache, self._pending, self._lock = OrderedDict(), {}, Lock()
        self.source_evaluations = 0
        self.maximum_source_evaluations = _positive_integer(maximum_source_evaluations, 'maximum_source_evaluations')
        self._evaluation_attempts = 0
        self._rate_columns = None

    @classmethod
    def from_rate_columns(cls, energies_cm, columns, *, coordinate_id):
        """Resume from complete acquired downward columns in s^-1.

        A missing column is unknown, including a missing terminal state. During
        sampling its first visit stops that trajectory and is counted explicitly.
        Empty supplied columns describe known dark/terminal states. No dipole
        phases or coherent evolution can be inferred from these probabilities.
        """
        e = np.asarray(energies_cm, float)
        if e.ndim != 1 or not len(e) or not np.all(np.isfinite(e)):
            raise ValueError('finite eigenenergies required')
        zero = LinearOperator((len(e), len(e)), matvec=lambda x: np.zeros_like(x), dtype=float)
        obj = cls(e, (zero, zero, zero), coordinate_id=coordinate_id,
                  cached_sources=max(1, len(columns)), maximum_source_evaluations=max(1, len(columns)))
        obj.dipoles = ()
        saved = {}
        for source, (targets, rates) in columns.items():
            if (isinstance(source, (bool, np.bool_)) or not isinstance(source, (int, np.integer))
                    or not 0 <= source < len(e)):
                raise ValueError('valid source index required')
            targets = np.asarray(targets)
            rates = np.asarray(rates, float)
            if (targets.ndim != 1 or targets.dtype.kind not in 'iu' or rates.shape != targets.shape
                    or np.any(targets < 0) or np.any(targets >= len(e))
                    or len(np.unique(targets)) != len(targets)
                    or not np.all(np.isfinite(rates)) or np.any(rates < 0)
                    or np.any(e[targets] >= e[source])):
                raise ValueError('unique strictly downward targets and finite nonnegative rates required')
            selected = rates > 0
            targets, rates = targets[selected].copy(), rates[selected].copy()
            targets.setflags(write=False); rates.setflags(write=False)
            total = float(rates.sum())
            if not np.isfinite(total):
                raise ValueError('finite total source rate required')
            saved[int(source)] = targets, rates, total
        obj._rate_columns = saved
        return obj

    def outgoing(self, source):
        if isinstance(source, bool) or not isinstance(source, (int, np.integer)) or not 0 <= source < len(self.energies):
            raise ValueError('valid eigenstate index required')
        source = int(source)
        if self._rate_columns is not None:
            if source not in self._rate_columns:
                raise KeyError(f'unacquired radiative source {source}')
            return self._rate_columns[source]
        with self._lock:
            if source in self._cache:
                self._cache.move_to_end(source)
                return self._cache[source]
            owner = source not in self._pending
            if owner:
                if self._evaluation_attempts >= self.maximum_source_evaluations:
                    raise MemoryError('radiative source-evaluation budget exceeded; increase the explicit budget or cache')
                self._evaluation_attempts += 1
                self._pending[source] = Future()
            future = self._pending[source]
        if not owner:
            return future.result()
        try:
            initial = np.zeros(len(self.energies), complex); initial[source] = 1.
            packet = np.column_stack([op.matvec(initial) for op in self.dipoles])
            if not np.all(np.isfinite(packet)):
                raise FloatingPointError('nonfinite dipole action')
            gaps = self.energies[source]-self.energies
            rates = EINSTEIN_PER_CM3_AU2*np.maximum(gaps, 0.)**3*np.sum(abs(packet)**2, axis=1)
            selected = np.flatnonzero(rates > 0)
            values = rates[selected]
            total = float(values.sum())
            if not np.isfinite(total):
                raise FloatingPointError('nonfinite radiative rate')
            for array in (selected, values):
                array.setflags(write=False)
            result = selected, values, total
            with self._lock:
                self.source_evaluations += 1
                self._cache[source] = result
                while len(self._cache) > self.cached_sources:
                    self._cache.popitem(last=False)
                self._pending.pop(source)
                future.set_result(result)
            return result
        except BaseException as error:
            with self._lock:
                self._pending.pop(source, None)
                future.set_exception(error)
            raise

    def rate_reference(self, occupations, sources, *, sectors=None, branches=None,
                       maximum_pairs=100000):
        """Export selected complete source columns for the existing rate fitter.

        Missing sources stay unmeasured. Zero rates remain allowed zero-valued
        observations, not inferred symmetry-forbidden channels. No Monte Carlo
        population or time information is used as a training target.
        """
        from .gvpt2_rates import GVPT2RateReference
        occupations = np.asarray(occupations)
        sources = np.asarray(sources)
        _positive_integer(maximum_pairs, 'maximum_pairs')
        if (occupations.ndim != 2 or len(occupations) != len(self.energies)
                or occupations.dtype.kind not in 'iu' or np.any(occupations < 0)
                or sources.ndim != 1 or not len(sources) or sources.dtype.kind not in 'iu'
                or np.any(sources < 0) or np.any(sources >= len(self.energies))
                or len(np.unique(sources)) != len(sources)):
            raise ValueError('compatible occupation labels and unique valid source indices required')
        lower_sets = [np.flatnonzero(self.energies < self.energies[i]) for i in sources]
        if sum(map(len, lower_sets)) > maximum_pairs:
            raise MemoryError('selected reference exceeds explicit pair budget')
        pairs = []; rates = []
        for source, lower in zip(sources, lower_sets):
            targets, values, _ = self.outgoing(int(source))
            lookup = dict(zip(map(int, targets), values))
            pairs.extend((int(source), int(i)) for i in lower)
            rates.extend(lookup.get(int(i), 0.) for i in lower)
        if not pairs:
            raise ValueError('selected sources have no downward reference channels')
        return GVPT2RateReference(occupations, self.energies, np.asarray(pairs, int),
            np.asarray(rates), np.ones(len(pairs), bool), self.coordinate_id, sectors, branches,
            dict(provider=('reused acquired rate columns' if self._rate_columns is not None
                           else 'quantum-dynamics effective-dipole vector actions'),
                 source_indices=sources.tolist(), pairwise_moment_acquisition=False,
                 finite_eigenstate_support=True, symmetry_mask_inferred=False))

    def sample(self, initial_population, *, trajectories, frequency_edges_cm, seed=0, workers=1):
        population = np.asarray(initial_population, float)
        edges = np.asarray(frequency_edges_cm, float)
        trajectories = _positive_integer(trajectories, 'trajectories')
        workers = _positive_integer(workers, 'workers')
        if (population.shape != self.energies.shape or not np.all(np.isfinite(population))
                or np.any(population < 0) or not np.isclose(population.sum(), 1., rtol=0, atol=1e-12)
                or edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
                or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
            raise ValueError('normalized population and increasing nonnegative frequency edges required')
        seed_sequence = np.random.SeedSequence(seed)
        def trajectory(child_seed):
            rng = np.random.default_rng(child_seed)
            source = int(rng.choice(len(population), p=population))
            initial_energy = self.energies[source]
            counts = np.zeros(len(edges)-1)
            elapsed = energy = outside = photons = 0.
            unresolved = 0.
            for _ in range(len(self.energies)):
                if self._rate_columns is not None and source not in self._rate_columns:
                    unresolved = 1.
                    break
                targets, rates, total = self.outgoing(source)
                if total == 0:
                    break
                elapsed += rng.exponential(1./total)
                final = int(rng.choice(targets, p=rates/total))
                gap = self.energies[source]-self.energies[final]
                if gap <= 0:
                    raise ArithmeticError('radiative jump must strictly decrease energy')
                index = int(np.searchsorted(edges, gap, side='right')-1)
                if gap == edges[-1]:
                    index = len(edges)-2
                if 0 <= index < len(counts):
                    counts[index] += 1
                else:
                    outside += 1
                photons += 1; energy += gap; source = final
            else:
                raise ArithmeticError('downward cascade did not terminate within its state count')
            return counts, np.array((elapsed, photons, energy, self.energies[source], outside,
                                     initial_energy-energy-self.energies[source], unresolved))
        # Streaming aggregation bounds memory independently of trajectory count.
        spectrum = np.zeros(len(edges)-1); spectrum2 = spectrum.copy()
        means = np.zeros(7); squares = means.copy()
        executor = None if workers == 1 else ThreadPoolExecutor(max_workers=workers)
        def batches():
            for first in range(0, trajectories, 2*workers):
                children = seed_sequence.spawn(min(2*workers, trajectories-first))
                yield from (map(trajectory, children) if executor is None
                            else executor.map(trajectory, children))
        results = batches()
        try:
            for counts, values in results:
                spectrum += counts; spectrum2 += counts**2
                means += values; squares += values**2
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        def error(total, square):
            if trajectories < 2:
                return np.full_like(total, np.nan)
            return np.sqrt(np.maximum(square-total**2/trajectories, 0)/(trajectories*(trajectories-1)))
        return dict(frequency_edges_cm=edges.copy(), photon_yield_per_bin=spectrum/trajectories,
            photon_yield_standard_error=error(spectrum, spectrum2),
            mean_observables=means[:6]/trajectories, observable_standard_errors=error(means, squares)[:6],
            unresolved_first_hit_probability=float(means[6]/trajectories),
            unresolved_first_hit_standard_error=float(error(means, squares)[6]),
            observable_names=('terminal_arrival_s', 'photons', 'emitted_energy_cm',
                              'terminal_energy_cm', 'out_of_range_photons', 'energy_balance_error_cm'),
            diagnostics=dict(trajectories=trajectories, seed=seed, workers=workers,
                source_evaluations=self.source_evaluations, dipole_vector_actions=3*self.source_evaluations,
                acquired_rate_columns_reused=self._rate_columns is not None,
                incomplete_trajectories_stopped_at_first_unknown=True,
                pair_table_constructed=False, eigenbasis_required=True,
                dynamical_model='secular spontaneous-emission jumps; no bath rethermalization',
                coordinate_id=self.coordinate_id, state_support_supplied_by_caller=True,
                maximum_source_evaluations=self.maximum_source_evaluations))


def rotate_dynamics_operator(operator, basis_map):
    """U^dagger O U action; the supplied U maps eigenvectors to bare vectors."""
    op, u = _operator(operator), _operator(basis_map)
    if op.shape != u.shape:
        raise ValueError('property and basis map dimensions differ')
    rng = np.random.default_rng(930)
    probe = rng.normal(size=(u.shape[0], 2))+1j*rng.normal(size=(u.shape[0], 2))
    if not np.allclose(u.H.matmat(u.matmat(probe)), probe, atol=1e-10, rtol=1e-10):
        raise ValueError('unitary basis map required')
    return _action_operator(op.shape, lambda x: u.H.matmat(op.matmat(u.matmat(x))),
                            lambda x: u.H.matmat(op.H.matmat(u.matmat(x))))


def gvpt2_quantum_dynamics(result, effective_bare_dipoles_au=None, *, coordinate_id,
                          field=None, derivatives=None, modalized=None, native_controls=None, **controls):
    """Reuse assigned GVPT2 blocks with supplied consistent effective properties.

    No new diagonalization or transition-pair acquisition is performed. The
    Supplied effective properties must include mechanical/electrical corrections
    and the same resonance partition as ``result``. Alternatively, supplying
    ``field`` and complete ``derivatives`` constructs the native normalized PT2
    property with the result's direct exclusions and pole-first convention.
    """
    energies = np.asarray(result.energies_cm, float)
    n = len(energies)
    native = None
    if effective_bare_dipoles_au is None:
        if field is None or derivatives is None:
            raise ValueError('effective dipoles or a native field with complete dipole derivatives required')
        from .native_resonant_property import NativeResonantProperty
        options = {} if native_controls is None else dict(native_controls)
        operator_controls = options.pop('operators', {})
        native = NativeResonantProperty.from_gvpt2(result, field, derivatives, modalized=modalized, **options)
        effective_bare_dipoles_au = native.operators(np.asarray(result.states),
            coordinate_id=coordinate_id, **operator_controls)
    elif any(value is not None for value in (field, derivatives, modalized, native_controls)):
        raise ValueError('choose supplied effective properties or native construction')
    blocks = []; occupied = set()
    for block in result.blocks:
        indices = np.asarray(block['indices'], int)
        roots = np.asarray(block['basis_to_root'], int)
        vectors = np.asarray(block['eigenvectors'])
        if (indices.ndim != 1 or len(set(indices)) != len(indices) or np.any(indices < 0)
                or np.any(indices >= n) or any(int(i) in occupied for i in indices)
                or sorted(roots.tolist()) != list(range(len(indices)))
                or vectors.shape != (len(indices), len(indices)) or not np.all(np.isfinite(vectors))
                or not np.allclose(vectors.conj().T@vectors, np.eye(len(indices)), atol=1e-10, rtol=1e-10)):
            raise ValueError('disjoint GVPT2 blocks with unitary assigned eigenvectors required')
        occupied.update(map(int, indices))
        blocks.append((indices, vectors[:, roots].copy()))
    def transform(x, adjoint=False):
        y = np.array(x, dtype=complex, copy=True)
        for indices, vectors in blocks:
            y[indices] = (vectors.conj().T if adjoint else vectors)@x[indices]
        return y
    u = _action_operator((n, n), transform, lambda x: transform(x, True))
    operators = tuple(effective_bare_dipoles_au)
    if any(getattr(op, 'coordinate_id', coordinate_id) != coordinate_id for op in operators):
        raise ValueError('effective properties must share the declared coordinate_id')
    dipoles = tuple(rotate_dynamics_operator(op, u) for op in operators)
    from scipy.sparse import diags
    dynamics = QuantumDynamics(diags(energies), energy_bounds_cm=(float(energies.min()), float(energies.max())),
                               coordinate_id=coordinate_id, **controls)
    cascade = SecularRadiativeDynamics(energies, dipoles, coordinate_id=coordinate_id)
    return dict(dynamics=dynamics, radiative=cascade, dipoles_eigenbasis_au=dipoles,
                native_property_provider=native,
                diagnostics=dict(gvpt2_blocks_reused=len(blocks), new_diagonalization=False,
                    effective_property_supplied=True, pair_table_constructed=False,
                    native_resonant_property=native is not None,
                    resonance_partition_owned_by_property_provider=True))


def gvpt2_emission_packets(result, native_provider, sources, *, coordinate_id,
                           operator_controls=None):
    """Acquire several emission packets in one shared bare-basis action.

    Sources sharing an assigned resonance block share its expensive bare
    columns. Exact downward closure retains complete assigned endpoint blocks;
    the native provider continues to own the full model's exclusion masks.
    Consecutive triples of Cartesian components define separate dipole models.
    """
    from types import SimpleNamespace

    energies = np.asarray(result.energies_cm, float)
    sources = np.asarray(sources)
    if (energies.ndim != 1 or not len(energies) or not np.all(np.isfinite(energies))
            or sources.ndim != 1 or not len(sources) or sources.dtype.kind not in 'iu'
            or np.any(sources < 0) or np.any(sources >= len(energies))
            or len(np.unique(sources)) != len(sources)
            or native_provider.components < 3 or native_provider.components % 3):
        raise ValueError('finite energies, unique source indices and Cartesian dipole triples required')
    options = {} if operator_controls is None else dict(operator_controls)
    selected = set(map(int, np.flatnonzero(energies <= energies[sources].max())))
    for block in result.blocks:
        if selected.intersection(block['indices']):
            selected.update(map(int, block['indices']))
    kept = np.array(sorted(selected), int)
    positions = np.full(len(energies), -1, int)
    positions[kept] = np.arange(len(kept))
    components, count = native_provider.components, len(sources)
    workspace = _positive_integer(options.get('maximum_workspace_bytes', 256*1024**2),
                                  'maximum_workspace_bytes')
    estimated = len(kept)*count*(16*(3*components+4)+8*(4+components//3))
    if estimated > workspace:
        raise MemoryError('shared emission packets exceed declared workspace budget')
    known = options.pop('known_moments', None)
    if known is not None:
        if any(not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer))
               or not 0 <= i <= j < len(energies) for i, j in known):
            raise ValueError('canonical global reference-pair indices required')
        options['known_moments'] = {(int(positions[i]), int(positions[j])): value
            for (i, j), value in known.items() if positions[i] >= 0 and positions[j] >= 0}
    reduced = SimpleNamespace(states=tuple(result.states[i] for i in kept),
        energies_cm=energies[kept],
        blocks=tuple({**block, 'indices': positions[np.asarray(block['indices'], int)].tolist()}
                     for block in result.blocks if int(block['indices'][0]) in selected))
    operators = native_provider.operators(np.asarray(reduced.states), coordinate_id=coordinate_id, **options)
    rotated = []
    for first in range(0, components, 3):
        bridge = gvpt2_quantum_dynamics(reduced, operators[first:first+3], coordinate_id=coordinate_id)
        rotated.extend(bridge['dipoles_eigenbasis_au'])
    units = np.zeros((len(kept), count), complex)
    units[positions[sources], np.arange(count)] = 1.
    packets = np.stack([op.matmat(units) for op in rotated], axis=1)
    gaps = energies[sources][None, :]-energies[kept, None]
    packets *= (gaps > 0)[:, None, :]
    strengths = np.sum(abs(packets.reshape(len(kept), components//3, 3, count))**2, axis=2)
    rates = EINSTEIN_PER_CM3_AU2*np.maximum(gaps, 0.)[:, None, :]**3*strengths
    return dict(sources=sources.copy(), global_indices=kept, energies_cm=energies[kept],
                packets_au=packets, rates_s_inverse=rates,
                diagnostics=dict(shared_bare_action=True, source_count=count,
                    propagated_states=len(kept), original_states=len(energies),
                    exact_downward_block_closure=True, new_diagonalization=False,
                    coordinate_id=coordinate_id))


def stationary_emission_correlation(energies_cm, source, dipoles_au, times_fs, *,
                                   coordinate_id, tolerance=1e-10, **controls):
    """Emission-only correlation from three operator actions in an eigenbasis."""
    energies = np.asarray(energies_cm, float)
    if (energies.ndim != 1 or not len(energies) or not np.all(np.isfinite(energies))
            or isinstance(source, bool) or not isinstance(source, (int, np.integer))
            or not 0 <= source < len(energies)):
        raise ValueError('finite eigenenergies and a valid source index required')
    operators = tuple(map(_operator, dipoles_au))
    if len(operators) != 3 or any(op.shape != (len(energies),)*2 for op in operators):
        raise ValueError('three atomic-unit dipoles on the same eigenbasis required')
    initial = np.zeros(len(energies)); initial[source] = 1
    packets = np.column_stack([op.matvec(initial) for op in operators])
    packets[energies >= energies[source]] = 0.
    from scipy.sparse import diags
    model = QuantumDynamics(diags(energies),
        energy_bounds_cm=(float(energies.min()), float(energies.max())), coordinate_id=coordinate_id, **controls)
    result = model.correlate(packets, times_fs, tolerance=tolerance, initial_energy_cm=energies[source])
    result['values'] = result['values'].sum(axis=1)
    gaps = np.maximum(energies[source]-energies, 0)
    result['diagnostics'].update(dipole_vector_actions=3,
        exact_total_emission_rate_s_inverse=float(EINSTEIN_PER_CM3_AU2*np.sum(gaps**3*np.sum(abs(packets)**2, axis=1))),
        emission_only=True, initial_source=int(source))
    return result


def reduced_field_dynamics_properties(reduced_field, derivatives, augmented_basis, *,
                                     coordinate_id, retained_rows=None, momentum=False,
                                     include_electrical=True, maximum_terms=1000000,
                                     maximum_workspace_bytes=256*1024**2):
    """Native action from existing reduced contact generators and derivatives.

    Generators come from ``prepare_reduced_transition_field`` or its theory
    owner. Their derivation, resonance treatment and coordinate transport are
    preserved. The supplied augmented basis must cover the required external
    intermediates; its convergence is separate from time propagation.
    """
    from scipy.sparse import csr_matrix
    from .occupation_operators import LadderPolynomial
    from .reduced_transition_polynomials import ReducedTransitionField, _property_coefficients
    if not isinstance(reduced_field, ReducedTransitionField):
        raise TypeError('ReducedTransitionField with prepared contact generators required')
    n = len(reduced_field.frequencies_cm)
    basis = np.asarray(augmented_basis)
    arrays = tuple(np.asarray(v) for v in (derivatives.first, derivatives.second, derivatives.third))
    components = arrays[0].shape[0] if arrays[0].ndim == 2 else 0
    if not components or any(a.shape != (components,)+(n,)*rank or not np.all(np.isfinite(a))
                             for rank, a in enumerate(arrays, 1)):
        raise ValueError('complete compatible modal property derivatives required')
    _positive_integer(maximum_terms, 'maximum_terms')
    if any(len(p) > maximum_terms for p in (reduced_field.first_generator, reduced_field.second_generator)):
        raise MemoryError('prepared contact generator exceeds term budget')
    def operator(values):
        if len(values) > maximum_terms:
            raise MemoryError('property polynomial exceeds term budget')
        polynomial = LadderPolynomial(n, {(tuple(r), tuple(s)): value for (r, s), value in values.items()}, coordinate_id)
        return LadderDynamicsOperator(polynomial, basis, maximum_workspace_bytes=maximum_workspace_bytes)
    s1, s2 = (operator(p) for p in (reduced_field.first_generator, reduced_field.second_generator))
    if not include_electrical:
        arrays = arrays[0], np.zeros_like(arrays[1]), np.zeros_like(arrays[2])
    orders = _property_coefficients(*arrays, momentum)
    embedding = None
    if retained_rows is not None:
        rows = np.asarray(retained_rows)
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
                or np.any(rows < 0) or np.any(rows >= len(basis)) or len(np.unique(rows)) != len(rows)):
            raise ValueError('unique valid retained basis rows required')
        embedding = csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(basis), len(rows)))
    result = []
    for component in range(components):
        bare = tuple(operator({key: value[component] for key, value in p.items() if value[component] != 0})
                     for p in orders)
        result.append(contact_dynamics_property(bare, s1, s2, embedding=embedding))
    return dict(orders=tuple(result), properties=tuple(sum_dynamics_operators(p) for p in result),
                diagnostics=dict(native_contact_operator_actions=True, pair_table_constructed=False,
                    include_electrical=bool(include_electrical), augmented_states=len(basis),
                    retained_states=len(basis) if retained_rows is None else len(retained_rows),
                    generator_provider='prepared reduced transition field',
                    external_support_convergence_required=True, coordinate_id=coordinate_id))


def main(argv=None):
    """Run a prepared operator model; this command submits no QC calculation."""
    import argparse
    import json
    from pathlib import Path
    from scipy.sparse import csr_matrix
    from .occupation_operators import LadderPolynomial
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='prepared operator NPZ, allow_pickle=False')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-step-fs', type=float, default=1.)
    parser.add_argument('--time-points', type=int, default=4097)
    parser.add_argument('--gaussian-sigma-cm', type=float, default=10.)
    parser.add_argument('--trajectories', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError('output must differ from the prepared input')
    _positive_integer(args.time_points, 'time_points')
    if args.time_points < 3 or not np.isfinite(args.time_step_fs) or args.time_step_fs <= 0:
        raise ValueError('positive time step and at least three points required')
    with np.load(args.input, allow_pickle=False) as saved:
        energies = saved['energies_cm']; population = saved['initial_population']
        edges = saved['frequency_edges_cm']; coordinate_id = str(saved['coordinate_id'])
        representation = str(saved['basis_representation'])
        if representation not in ('energy_eigenbasis', 'harmonic_occupation'):
            raise ValueError('explicit energy_eigenbasis or harmonic_occupation representation required')
        if 'dipoles_au' in saved:
            dipoles = tuple(saved['dipoles_au'])
            supplied = 'explicit prepared operators'
        else:
            basis = saved['basis']; creation = saved['creation']; annihilation = saved['annihilation']
            coefficients = saved['dipole_coefficients_au']
            if (creation.shape != annihilation.shape or creation.ndim != 2
                    or coefficients.shape != (3, len(creation))):
                raise ValueError('compatible normal-ordered dipole arrays required')
            polynomials = []
            for component in coefficients:
                terms = {}
                for r, s, value in zip(creation, annihilation, component):
                    key = tuple(r), tuple(s)
                    terms[key] = terms.get(key, 0j)+value
                polynomials.append(LadderPolynomial(basis.shape[1], terms, coordinate_id))
            dipoles = tuple(LadderDynamicsOperator(polynomial, basis) for polynomial in polynomials)
            supplied = 'compiled normal-ordered operator actions'
        if representation == 'harmonic_occupation':
            u = csr_matrix((saved['basis_map_data'], saved['basis_map_indices'], saved['basis_map_indptr']),
                           shape=(len(energies), len(energies)))
            dipoles = tuple(rotate_dynamics_operator(op, u) for op in dipoles)
    source = int(np.argmax(population))
    times = np.arange(args.time_points)*args.time_step_fs
    correlation = stationary_emission_correlation(energies, source, dipoles, times, coordinate_id=coordinate_id)
    maximum_gap = float(np.max(abs(energies[source]-energies)))
    if np.pi/(PHASE_PER_CM_FS*args.time_step_fs) <= maximum_gap:
        raise ValueError('time step aliases the declared transition-energy range')
    spectrum = correlation_emission_spectrum(correlation, gaussian_sigma_cm=args.gaussian_sigma_cm,
                                             fft_workers=args.workers)
    engine = SecularRadiativeDynamics(energies, dipoles, coordinate_id=coordinate_id)
    cascade = engine.sample(population, trajectories=args.trajectories, frequency_edges_cm=edges,
                             seed=args.seed, workers=args.workers)
    diagnostics = dict(correlation=correlation['diagnostics'], spectrum=spectrum['diagnostics'],
                       cascade=cascade['diagnostics'], operator_input=supplied,
                       spectrum_source=source, spectrum_initial_population='one selected eigenstate',
                       cascade_initial_population='supplied incoherent population')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.pending.npz')
    np.savez_compressed(temporary, times_fs=times, correlation=correlation['values'],
        wavenumbers_cm=spectrum['wavenumbers_cm'],
        strength_density_au2_per_cm=spectrum['strength_density_au2_per_cm'],
        photon_rate_density_s_inverse_per_cm=spectrum['photon_rate_density_s_inverse_per_cm'],
        frequency_edges_cm=edges, photon_yield_per_bin=cascade['photon_yield_per_bin'],
        photon_yield_standard_error=cascade['photon_yield_standard_error'],
        mean_observables=cascade['mean_observables'], observable_standard_errors=cascade['observable_standard_errors'],
        observable_names=np.asarray(cascade['observable_names']), diagnostics_json=json.dumps(diagnostics))
    temporary.replace(args.output)
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
