"""Complete finite energy windows and bounded Gaussian contractions.

An eigenpair residual alone cannot establish that a spectral window is
complete. Every window therefore carries an independent count for the exact
operator identity. Numerical inertia is a numerical check, not interval proof.
"""
from dataclasses import dataclass
import numpy as np
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class SpectralWindowCount:
    hamiltonian_sha256: str
    lower_cm: float
    upper_cm: float
    below_lower: int
    below_upper: int
    method: str

    def __post_init__(self):
        if (not self.hamiltonian_sha256 or not self.method
                or not np.isfinite(self.lower_cm) or not np.isfinite(self.upper_cm)
                or self.lower_cm >= self.upper_cm
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                       for v in (self.below_lower, self.below_upper))
                or not 0 <= self.below_lower <= self.below_upper):
            raise ValueError('identified finite interval and monotone independent counts required')

    @property
    def count(self):
        return int(self.below_upper-self.below_lower)


def count_spectral_window(hamiltonian, lower_cm, upper_cm, *, backend):
    """Use a supplied checked-inertia backend; do not perturb interval edges."""
    # Validate the interval before performing a factorization.
    SpectralWindowCount(hamiltonian.fingerprint, lower_cm, upper_cm, 0, 0, 'validation')
    counts = []
    for shift in (lower_cm, upper_cm):
        with backend.shifted_factor(hamiltonian.matrix_cm, shift) as factor:
            if factor.shift_cm != shift:
                raise ArithmeticError('inertia backend changed a spectral boundary')
            counts.append(int(factor.negative_count))
    return SpectralWindowCount(hamiltonian.fingerprint, lower_cm, upper_cm,
                              *counts, 'unperturbed numerical inertia')


def qualify_operator_eigenpairs(operator, energies, vectors, *, workers=1,
        batch_size=64, maximum_workspace_bytes=128*1024**3):
    """Every residual and every entry of the Gram matrix, in float64/complex128."""
    from scipy.sparse.linalg import aslinearoperator
    if getattr(operator, 'approximate_products', False):
        raise ValueError('qualification requires an independent CPU operator')
    op = aslinearoperator(operator)
    e, u = np.asarray(energies), np.asarray(vectors)
    if (e.ndim != 1 or np.iscomplexobj(e) or u.shape != (op.shape[0], len(e))
            or op.shape[0] != op.shape[1] or not np.all(np.isfinite(e))
            or not np.all(np.isfinite(u)) or workers < 1 or batch_size < 1):
        raise ValueError('finite compatible operator/eigenpairs and positive resources required')
    dtype = np.result_type(u.dtype, np.float64)
    u = np.asarray(u, dtype=dtype)
    required = u.nbytes+4*op.shape[0]*min(batch_size, len(e))*np.dtype(dtype).itemsize
    required += len(e)*min(batch_size, len(e))*np.dtype(dtype).itemsize
    if required > maximum_workspace_bytes:
        raise MemoryError('eigenpair qualification exceeds workspace budget')
    residual = orthogonality = 0.
    with threadpool_limits(limits=workers, user_api='blas'):
        for first in range(0, len(e), batch_size):
            last = min(first+batch_size, len(e)); block = u[:, first:last]
            error = op.matmat(block)-block*e[first:last]
            residual = max(residual, float(np.linalg.norm(error, axis=0).max()))
            gram = u.conj().T@block
            gram[np.arange(first, last), np.arange(last-first)] -= 1
            orthogonality = max(orthogonality, float(abs(gram).max()))
    if residual > 1e-6 or orthogonality > 1e-8:
        raise ArithmeticError('eigenpairs failed full residual or orthogonality qualification')
    return dict(maximum_residual_cm=residual, maximum_orthogonality_error=orthogonality,
                qualification_arithmetic='CPU float64/complex128', eigenpairs_checked=len(e))


@dataclass(frozen=True)
class QualifiedEigenwindow:
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    count: SpectralWindowCount
    diagnostics: dict

    def __post_init__(self):
        e, u = np.asarray(self.energies_cm), np.asarray(self.eigenvectors)
        if (np.iscomplexobj(e) or e.ndim != 1 or u.ndim != 2
                or u.shape[1] != len(e) or len(e) != self.count.count
                or u.shape[0] < self.count.below_upper or not np.all(np.isfinite(e))
                or not np.all(np.isfinite(u)) or np.any(np.diff(e) < 0)
                or np.any(e <= self.count.lower_cm) or np.any(e >= self.count.upper_cm)):
            raise ValueError('all independently counted ordered interior eigenpairs required')
        for name, limit in (('maximum_residual_cm', 1e-6), ('maximum_orthogonality_error', 1e-8)):
            value = self.diagnostics.get(name, np.inf)
            if not np.isfinite(value) or not 0 <= value <= limit:
                raise ValueError('valid full eigenpair qualification required')
        if len(e) and min(e[0]-self.count.lower_cm, self.count.upper_cm-e[-1]) <= (
                10*self.diagnostics['maximum_residual_cm']):
            raise ArithmeticError('spectral boundary is unresolved at the measured residual')
        for name, value in (('energies_cm', e), ('eigenvectors', u)):
            value = value.copy(order='K'); value.setflags(write=False)
            object.__setattr__(self, name, value)


def acquire_dense_eigenwindow(matrix, lower_cm, upper_cm, *, hamiltonian_sha256,
        workers=1, maximum_workspace_bytes=128*1024**3):
    """Count the full dense spectrum, then solve only the requested interval.

    LAPACK's all-eigenvalues/no-vectors route provides a separate count from
    the interval eigenvector solve. This avoids sparse ordering/factorization
    overhead for dense Galerkin matrices. Counts are numerical, not interval
    arithmetic certificates. Full eigenpair checks use the original matrix.
    """
    from time import perf_counter
    from scipy.linalg import eigh
    a = np.asarray(matrix)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
            or not np.isfinite(lower_cm) or not np.isfinite(upper_cm) or lower_cm >= upper_cm
            or not hamiltonian_sha256 or not isinstance(workers, int) or workers < 1
            or not np.all(np.isfinite(a))):
        raise ValueError('finite identified dense Hermitian matrix and ordered interval required')
    if 6*a.size*np.dtype(np.result_type(a.dtype, np.float64)).itemsize > maximum_workspace_bytes:
        raise MemoryError('dense eigenwindow exceeds explicit workspace budget')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=1e-12):
        raise ValueError('Hermitian dense matrix required')
    started = perf_counter()
    with threadpool_limits(limits=workers, user_api='blas'):
        spectrum = eigh(a, eigvals_only=True, driver='evd', check_finite=False)
        below = np.searchsorted(spectrum, [lower_cm, upper_cm])
        count = SpectralWindowCount(hamiltonian_sha256, lower_cm, upper_cm,
            int(below[0]), int(below[1]), 'independent full-spectrum LAPACK eigenvalues-only evd')
        if count.count == 0: raise ValueError('requested spectral window is empty')
        if min(np.min(abs(spectrum-lower_cm)), np.min(abs(spectrum-upper_cm))) < 1e-6:
            raise ArithmeticError('spectral boundary too close to a counted eigenvalue')
        counted = perf_counter()
        e, u = eigh(a, subset_by_value=(lower_cm, upper_cm), driver='evr', check_finite=False)
        solved = perf_counter()
        checks = qualify_operator_eigenpairs(a, e, u, workers=workers,
            maximum_workspace_bytes=maximum_workspace_bytes)
    checks.update(solver='dense counted interval LAPACK evr', independently_counted_levels=count.count,
        count_seconds=counted-started, interval_solve_seconds=solved-counted,
        acquisition_seconds=perf_counter()-started)
    return QualifiedEigenwindow(e, u, count, checks)


class PreparedEnergyWindows:
    """Reusable disjoint energy intervals with full-norm omitted-band bounds."""
    def __init__(self, windows, *, operator, hamiltonian_sha256, workers=1,
            maximum_workspace_bytes=128*1024**3):
        self.windows = tuple(sorted(windows, key=lambda w: w.count.lower_cm))
        if not self.windows:
            raise ValueError('at least one independently counted window required')
        self.hamiltonian_sha256 = self.windows[0].count.hamiltonian_sha256
        if self.hamiltonian_sha256 != hamiltonian_sha256:
            raise ValueError('window counts belong to another operator identity')
        n = self.windows[0].eigenvectors.shape[0]
        previous = None
        for w in self.windows:
            if (w.count.hamiltonian_sha256 != self.hamiltonian_sha256
                    or w.eigenvectors.shape[0] != n
                    or (previous is not None and w.count.lower_cm <= previous)):
                raise ValueError('disjoint separated windows of one operator required')
            previous = w.count.upper_cm
        required = sum(w.eigenvectors.nbytes for w in self.windows)*3
        if required > maximum_workspace_bytes:
            raise MemoryError('energy windows exceed workspace budget')
        self.energies_cm = np.concatenate([w.energies_cm for w in self.windows])
        self.eigenvectors = np.concatenate([w.eigenvectors for w in self.windows], axis=1)
        # Includes cross-window orthogonality, not just each local Gram matrix.
        self.diagnostics = qualify_operator_eigenpairs(operator, self.energies_cm,
            self.eigenvectors, workers=workers, maximum_workspace_bytes=maximum_workspace_bytes)
        self.energies_cm.setflags(write=False); self.eigenvectors.setflags(write=False)

    def evaluate(self, wavepackets, targets_cm, *, sigma_cm,
            tolerance_per_strength, workers=1):
        from .vibronic_spectral_cache import _evaluate_spectral_blocks
        packets, targets = np.asarray(wavepackets), np.asarray(targets_cm, dtype=float)
        if (packets.ndim != 2 or packets.shape[0] != len(self.eigenvectors)
                or targets.shape != (packets.shape[1],) or not np.all(np.isfinite(packets))
                or not np.all(np.isfinite(targets)) or not np.isfinite(sigma_cm) or sigma_cm <= 0
                or not np.isfinite(tolerance_per_strength) or tolerance_per_strength <= 0):
            raise ValueError('finite packets/targets and positive Gaussian width/tolerance required')
        peak = 1/(np.sqrt(2*np.pi)*sigma_cm)
        distance = np.zeros(len(targets))
        for w in self.windows:
            inside = (targets > w.count.lower_cm)&(targets < w.count.upper_cm)
            distance[inside] = np.minimum(targets[inside]-w.count.lower_cm,
                                          w.count.upper_cm-targets[inside])
        omitted = peak*np.exp(-.5*(distance/sigma_cm)**2)
        accepted = omitted <= .5*tolerance_per_strength
        # The existing helper reserves half the tolerance for Gaussian pruning.
        result = _evaluate_spectral_blocks(
            ((slice(None), self.energies_cm, self.eigenvectors),), len(self.eigenvectors),
            None, len(self.energies_cm), packets, targets, sigma_cm=sigma_cm,
            tolerance_per_strength=tolerance_per_strength, workers=workers,
            backend='independently_counted_energy_windows')
        norms = np.sum(abs(packets)**2, axis=0)
        result['absolute_error_bounds_per_cm'] += norms*omitted
        result['accepted'] &= accepted
        result['diagnostics'].update(window_count=len(self.windows),
            completeness_source='independent finite-operator eigenvalue counts',
            omitted_bands_bounded=True, global_basis_convergence_certified=False)
        return result


def plan_gaussian_energy_windows(targets_cm, *, sigma_cm, tolerance_per_strength):
    """Merge only overlapping Gaussian supports; do not truncate their targets."""
    targets = np.asarray(targets_cm, dtype=float)
    if (targets.ndim != 1 or not len(targets) or not np.all(np.isfinite(targets))
            or not np.isfinite(sigma_cm) or sigma_cm <= 0
            or not np.isfinite(tolerance_per_strength) or tolerance_per_strength <= 0):
        raise ValueError('finite target vector and positive Gaussian width/tolerance required')
    peak = 1/(np.sqrt(2*np.pi)*sigma_cm)
    radius = sigma_cm*np.sqrt(2*max(0., np.log(2*peak/tolerance_per_strength)))
    # Positive edge margin avoids ambiguous touching intervals at zero radius.
    radius = max(radius, np.finfo(float).eps*max(1., float(abs(targets).max()))*8)
    intervals = []
    for target in np.sort(targets):
        lo, hi = float(target-radius), float(target+radius)
        if intervals and lo <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], hi))
        else:
            intervals.append((lo, hi))
    return tuple(intervals)


def save_prepared_energy_windows(spectrum, directory):
    """Store counted windows with immutable-array checksums and provenance."""
    import json
    from dataclasses import asdict
    from pathlib import Path
    from .vibronic_spectral_cache import _file_sha256
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    records = []
    for index, window in enumerate(spectrum.windows):
        arrays = {}
        for name, value in (('energies', window.energies_cm), ('vectors', window.eigenvectors)):
            filename = f'{name}-{index:04d}.npy'
            np.save(directory/filename, value, allow_pickle=False)
            arrays[name] = dict(file=filename, sha256=_file_sha256(directory/filename))
        records.append(dict(count=asdict(window.count), diagnostics=window.diagnostics, arrays=arrays))
    metadata = dict(schema='matrix.trinity.counted_energy_windows.v1', windows=records,
                    hamiltonian_sha256=spectrum.hamiltonian_sha256)
    temporary = directory/'metadata.tmp'; temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(directory/'metadata.json')


def load_prepared_energy_windows(directory, *, operator, hamiltonian_sha256, workers=1,
        maximum_workspace_bytes=128*1024**3):
    """Verify storage and recheck every residual and cross-window Gram entry."""
    import json
    from pathlib import Path
    from .vibronic_spectral_cache import _file_sha256
    directory = Path(directory); metadata = json.loads((directory/'metadata.json').read_text())
    if (metadata['schema'] != 'matrix.trinity.counted_energy_windows.v1'
            or metadata['hamiltonian_sha256'] != hamiltonian_sha256):
        raise ValueError('stored energy windows belong to another operator')
    windows = []
    for index, record in enumerate(metadata['windows']):
        arrays = {}
        for name in ('energies', 'vectors'):
            expected = f'{name}-{index:04d}.npy'; saved = record['arrays'][name]
            if saved['file'] != expected or _file_sha256(directory/expected) != saved['sha256']:
                raise ValueError('energy-window array checksum or filename mismatch')
            arrays[name] = np.load(directory/expected, mmap_mode='r', allow_pickle=False)
        windows.append(QualifiedEigenwindow(arrays['energies'], arrays['vectors'],
            SpectralWindowCount(**record['count']), record['diagnostics']))
    return PreparedEnergyWindows(windows, operator=operator, hamiltonian_sha256=hamiltonian_sha256,
        workers=workers, maximum_workspace_bytes=maximum_workspace_bytes)
