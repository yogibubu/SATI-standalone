"""Reusable low-energy spectral windows for repeated Gaussian contractions.

Acquisition is expensive once. Each subsequent contraction touches only
eigenvectors close to its target. The omitted Gaussian tails have an explicit
bound per unit wavepacket norm; roundoff and eigensolver errors are separate.
"""
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits


def _file_sha256(path):
    """Hash large spectral arrays without materializing their file bytes."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hamiltonian_fingerprint(hamiltonian):
    digest = hashlib.sha256(b'matrix.trinity.finite_sparse_hamiltonian.v1')
    matrix = hamiltonian.matrix_cm
    for array in (hamiltonian.basis, hamiltonian.frequencies_cm,
                  matrix.data, matrix.indices, matrix.indptr):
        array = np.ascontiguousarray(array)
        digest.update(json.dumps((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedGaussianSpectrum:
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    omitted_lower_bound_cm: float | None
    hamiltonian_sha256: str
    diagnostics: dict

    def __post_init__(self):
        if np.iscomplexobj(self.energies_cm):
            raise ValueError('real eigenenergies required')
        energies, vectors = np.asarray(self.energies_cm, float), np.asarray(self.eigenvectors)
        if (energies.ndim != 1 or not len(energies) or np.any(np.diff(energies) < 0)
                or vectors.ndim != 2 or vectors.shape[1] != len(energies)
                or vectors.shape[0] < len(energies)
                or (self.omitted_lower_bound_cm is None and vectors.shape[0] != len(energies))
                or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(vectors))
                or (self.omitted_lower_bound_cm is not None
                    and (not np.isfinite(self.omitted_lower_bound_cm)
                         or self.omitted_lower_bound_cm < energies[-1]))):
            raise ValueError('ordered finite complete-lowest eigenlevels required')
        for name, array in (('energies_cm', energies), ('eigenvectors', vectors)):
            if array.flags.writeable:
                array = array.copy(order='K')
                array.setflags(write=False)
            object.__setattr__(self, name, array)

    def evaluate(self, wavepackets, targets_cm, *, sigma_cm, tolerance_per_strength,
                 workers=1):
        """Return bounded contractions and a mask requiring a different backend."""
        return _evaluate_spectral_blocks(
            ((slice(None), self.energies_cm, self.eigenvectors),),
            len(self.eigenvectors), self.omitted_lower_bound_cm, len(self.energies_cm),
            wavepackets, targets_cm, sigma_cm=sigma_cm,
            tolerance_per_strength=tolerance_per_strength, workers=workers,
            backend='prepared_eigenwindow')


@dataclass(frozen=True)
class PreparedBlockGaussianSpectrum:
    """Lowest spectral window on exact disjoint row supports, without padding.

    Each part is (global basis rows, prepared local spectrum). Missing rows or
    omitted local levels must be covered by the common omitted-level bound.
    This representation is for exact conserved blocks, never weak-coupling
    truncation. Dense materialization is an explicit, memory-guarded operation.
    """
    basis_size: int
    parts: tuple
    omitted_lower_bound_cm: float | None
    hamiltonian_sha256: str
    diagnostics: dict

    def __post_init__(self):
        if (isinstance(self.basis_size, bool) or not isinstance(self.basis_size, int)
                or self.basis_size <= 0 or not self.parts):
            raise ValueError('positive basis size and nonempty exact spectral blocks required')
        used = np.zeros(self.basis_size, bool); parts = []; level_count = 0
        bound = self.omitted_lower_bound_cm
        if bound is not None and not np.isfinite(bound):
            raise ValueError('finite omitted-level bound required')
        for rows, spectrum in self.parts:
            rows = np.asarray(rows)
            if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
                    or np.any(rows < 0) or np.any(rows >= self.basis_size)
                    or len(np.unique(rows)) != len(rows)
                    or np.any(used[rows]) or not isinstance(spectrum, PreparedGaussianSpectrum)
                    or len(rows) != spectrum.eigenvectors.shape[0]):
                raise ValueError('disjoint valid rows and compatible local spectra required')
            if ((bound is None and spectrum.omitted_lower_bound_cm is not None)
                    or (bound is not None and
                        (bound < spectrum.energies_cm[-1] or
                         (spectrum.omitted_lower_bound_cm is not None
                          and bound > spectrum.omitted_lower_bound_cm)))):
                raise ValueError('common omitted bound must cover every local spectrum')
            used[rows] = True; level_count += len(spectrum.energies_cm)
            if rows.flags.writeable:
                rows = rows.copy(); rows.setflags(write=False)
            parts.append((rows, spectrum))
        if bound is None and (not np.all(used) or level_count != self.basis_size):
            raise ValueError('a complete spectrum must span every basis row')
        object.__setattr__(self, 'parts', tuple(parts))

    @cached_property
    def energies_cm(self):
        energies = np.sort(np.concatenate([s.energies_cm for _, s in self.parts]))
        energies.setflags(write=False)
        return energies

    def evaluate(self, wavepackets, targets_cm, *, sigma_cm, tolerance_per_strength,
                 workers=1):
        blocks = []
        for rows, spectrum in self.parts:
            index = (slice(int(rows[0]), int(rows[-1])+1)
                     if np.all(np.diff(rows) == 1) else rows)
            blocks.append((index, spectrum.energies_cm, spectrum.eigenvectors))
        return _evaluate_spectral_blocks(blocks, self.basis_size,
            self.omitted_lower_bound_cm, len(self.energies_cm), wavepackets, targets_cm,
            sigma_cm=sigma_cm, tolerance_per_strength=tolerance_per_strength,
            workers=workers, backend='prepared_block_eigenwindow')

    def to_dense(self, *, maximum_workspace_bytes=64*1024**3):
        """Explicit diagnostic conversion; production contractions stay blocked."""
        count = len(self.energies_cm)
        dtype = np.result_type(*[s.eigenvectors.dtype for _, s in self.parts])
        if 3*self.basis_size*count*dtype.itemsize > maximum_workspace_bytes:
            raise MemoryError('dense block-spectrum conversion exceeds workspace budget')
        vectors = np.zeros((self.basis_size, count), dtype=dtype, order='F')
        energies = np.concatenate([s.energies_cm for _, s in self.parts])
        offset = 0
        for rows, spectrum in self.parts:
            stop = offset+len(spectrum.energies_cm)
            vectors[np.ix_(rows, np.arange(offset, stop))] = spectrum.eigenvectors
            offset = stop
        order = np.argsort(energies)
        return PreparedGaussianSpectrum(energies[order], vectors[:, order],
            self.omitted_lower_bound_cm, self.hamiltonian_sha256, self.diagnostics)


def gaussian_omitted_tail_profile(omitted_lower_bound_cm, targets_cm, sigma_cm):
    """Existing Gaussian window bound per unit omitted nonnegative strength."""
    targets = np.asarray(targets_cm, float)
    if (not np.all(np.isfinite(targets)) or not np.isfinite(sigma_cm) or sigma_cm <= 0
            or (omitted_lower_bound_cm is not None and not np.isfinite(omitted_lower_bound_cm))):
        raise ValueError('finite targets, omitted bound and positive Gaussian width required')
    if omitted_lower_bound_cm is None:
        return np.zeros_like(targets)
    distance = np.maximum(omitted_lower_bound_cm-targets, 0.)
    peak = 1/(np.sqrt(2*np.pi)*sigma_cm)
    return peak*np.exp(-.5*(distance/sigma_cm)**2)


def _evaluate_spectral_blocks(blocks, basis_size, omitted_lower_bound_cm,
        level_count, wavepackets, targets_cm, *, sigma_cm,
        tolerance_per_strength, workers, backend):
    """Common Gaussian window and full-norm error contract for both layouts."""
    vectors, targets = np.asarray(wavepackets), np.asarray(targets_cm, float)
    if (vectors.ndim != 2 or vectors.shape[0] != basis_size
            or targets.shape != (vectors.shape[1],)
            or not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(targets))
            or not np.isfinite(sigma_cm) or sigma_cm <= 0
            or not np.isfinite(tolerance_per_strength) or tolerance_per_strength <= 0
            or not isinstance(workers, int) or workers <= 0):
        raise ValueError('finite compatible wavepackets, positive linewidth/tolerance/workers required')
    peak = 1/(np.sqrt(2*np.pi)*sigma_cm)
    omitted = gaussian_omitted_tail_profile(omitted_lower_bound_cm, targets, sigma_cm)
    accepted = omitted <= .5*tolerance_per_strength
    values, errors = np.zeros(len(targets)), np.zeros(len(targets))
    radius = sigma_cm*np.sqrt(2*max(0., np.log(2*peak/tolerance_per_strength)))
    projected_levels = 0
    with threadpool_limits(limits=workers, user_api='blas'):
        for column in np.flatnonzero(accepted):
            source = vectors[:, column]
            norm2 = float(np.vdot(source, source).real)
            if not np.isfinite(norm2):
                raise FloatingPointError('wavepacket norm overflow')
            for rows, levels, eigenvectors in blocks:
                lo, hi = np.searchsorted(levels,
                    (targets[column]-radius, targets[column]+radius), side='left')
                energies = levels[lo:hi]
                active = eigenvectors[:, lo:hi]
                local_source = source[rows]
                if np.isrealobj(active):
                    # Keep real spectral windows in real BLAS: conjugating them
                    # copies the window, and mixed real/complex GEMV casts it.
                    amplitudes = active.T @ local_source.real
                    if np.iscomplexobj(local_source):
                        amplitudes = amplitudes+1j*(active.T @ local_source.imag)
                else:
                    amplitudes = active.conj().T @ local_source
                profile = peak*np.exp(-.5*((energies-targets[column])/sigma_cm)**2)
                values[column] += float(abs(amplitudes)**2 @ profile)
                projected_levels += hi-lo
            errors[column] = norm2*(omitted[column]+min(peak, .5*tolerance_per_strength))
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors)):
        raise FloatingPointError('nonfinite prepared spectral contraction')
    return dict(densities_per_cm=values, absolute_error_bounds_per_cm=errors,
        accepted=accepted, diagnostics=dict(spectral_backend=backend,
            prepared_level_count=level_count, projected_level_count=projected_levels,
            gaussian_tail_bound_includes_roundoff=False))


def _qualify(hamiltonian, energies, vectors, *, workers,
             maximum_workspace_bytes=64*1024**3):
    """Check every residual and the full orthogonality Gram matrix in blocks."""
    maximum_residual, maximum_orthogonality = 0., 0.
    matrix = hamiltonian.matrix_cm
    n = matrix.shape[0]
    dense_bytes = n*n*matrix.dtype.itemsize
    # Native quadrature matrices can be effectively dense. Repeated CSR/dense
    # products then waste the dense BLAS available for the identical operator.
    # Keep the sparse path when conversion is not amortized or would exceed the
    # caller's budget, including live eigenvectors and block temporaries.
    block_width = min(64, len(energies))
    reserved = (matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
                +2*vectors.nbytes
                +4*n*block_width*np.result_type(matrix.dtype, vectors.dtype).itemsize)
    dense_residual = (matrix.nnz >= n*n//4 and len(energies) > 64
                      and dense_bytes+reserved <= maximum_workspace_bytes)
    shared_dense=False
    if dense_residual:
        from matrix_core.sparse_layout import dense_csr_view
        view=getattr(hamiltonian,'dense_matrix_view',None)
        if view is None:view=dense_csr_view(matrix)
        shared_dense=view is not None
        matrix = view if shared_dense else matrix.toarray(order='F')
    with threadpool_limits(limits=workers, user_api='blas'):
        for start in range(0, len(energies), 64):
            stop = min(start+64, len(energies))
            block = vectors[:, start:stop]
            residual = matrix@block-block*energies[start:stop]
            maximum_residual = max(maximum_residual, float(np.linalg.norm(residual, axis=0).max()))
            gram = vectors.conj().T@block
            gram[np.arange(start, stop), np.arange(stop-start)] -= 1.
            maximum_orthogonality = max(maximum_orthogonality, float(abs(gram).max()))
    if maximum_residual > 1e-6 or maximum_orthogonality > 1e-8:
        raise ArithmeticError('prepared spectrum failed residual or orthogonality qualification')
    return dict(maximum_residual_cm=maximum_residual,
                maximum_orthogonality_error=maximum_orthogonality,
                residual_product_backend='dense BLAS' if dense_residual else 'sparse CSR',
                residual_dense_values_shared=shared_dense)


def acquire_gaussian_spectrum(hamiltonian, *, target_max_cm, gaussian_sigma_cm,
        tolerance_per_strength=1e-16, workers=1, maximum_workspace_bytes=64*1024**3,
        method='dense', mkl_library=None, slice_processes=None, levels_per_slice=128,
        work_directory=None, progress=None, dense_driver='evr'):
    """Acquire all eigenlevels below a Gaussian-tail-controlled energy ceiling.

    ``evd`` is an explicit alternative for dense windows containing many levels:
    it solves the full problem before retaining the identical energy window.
    Benchmark the actual matrix; the default partial ``evr`` driver is unchanged.
    """
    from scipy.linalg import eigh
    if (not np.isfinite(target_max_cm) or not np.isfinite(gaussian_sigma_cm)
            or gaussian_sigma_cm <= 0 or not np.isfinite(tolerance_per_strength)
            or tolerance_per_strength <= 0 or not isinstance(workers, int) or workers <= 0):
        raise ValueError('finite target and positive linewidth/tolerance/workers required')
    matrix = hamiltonian.matrix_cm
    n = matrix.shape[0]
    required = 6*n*n*matrix.dtype.itemsize
    if method not in ('dense','mkl-dense','sliced'):
        raise ValueError('spectral acquisition method must be dense, mkl-dense or sliced')
    if dense_driver not in ('evr', 'evd') or (method != 'dense' and dense_driver != 'evr'):
        raise ValueError('dense_driver must be evr or evd and applies only to dense acquisition')
    if method!='dense' and mkl_library is None:
        raise ValueError('explicit MKL library path required for native acquisition')
    if method!='sliced' and required > maximum_workspace_bytes:
        raise MemoryError(f'prepared spectrum needs conservative workspace {required} bytes')
    peak = 1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    ceiling = target_max_cm+gaussian_sigma_cm*np.sqrt(2*max(0., np.log(2*peak/tolerance_per_strength)))
    started = time.monotonic()
    extra={}
    if method=='sliced':
        from .vibronic_spectrum_slicing import sliced_lowest_spectrum
        if work_directory is None:raise ValueError('checkpoint work directory required for slicing')
        processes=(min(16,max(1,workers//2)) if slice_processes is None else slice_processes)
        energies,vectors,ceiling,extra=sliced_lowest_spectrum(hamiltonian,ceiling,
            library_path=mkl_library,workers=workers,processes=processes,
            levels_per_slice=levels_per_slice,work_directory=work_directory,
            maximum_workspace_bytes=maximum_workspace_bytes,progress=progress)
        required=4*n*len(energies)*matrix.dtype.itemsize
    elif method=='mkl-dense':
        from .vibronic_mkl import MKLSpectralBackend
        energies,vectors=MKLSpectralBackend(mkl_library,workers=workers).dense_window(matrix,ceiling)
    else:
        with threadpool_limits(limits=workers, user_api='blas'):
            from matrix_core.sparse_layout import dense_csr_view
            view=getattr(hamiltonian,'dense_matrix_view',None)
            if view is None:view=dense_csr_view(matrix)
            dense = (np.array(view,order='F',copy=True) if view is not None
                     else matrix.toarray(order='F'))
            selection = ({'subset_by_value': (-np.inf, ceiling)}
                         if dense_driver == 'evr' else {})
            energies, vectors = eigh(dense, overwrite_a=True, check_finite=False,
                                     driver=dense_driver, **selection)
            if dense_driver == 'evd':
                stop = np.searchsorted(energies, ceiling, side='right')
                energies = energies[:stop].copy()
                vectors = vectors[:, :stop].copy(order='F')
            del dense
    if not len(energies):
        raise ValueError('requested spectral window contains no eigenlevels')
    diagnostics = dict(schema='matrix.trinity.prepared_gaussian_spectrum.v1',
        solver=(f'LAPACK {dense_driver}; all eigenvalues in (-inf, ceiling]' if method=='dense'
            else 'native MKL LAPACKE dsyevr' if method=='mkl-dense' else extra['solver']),
        acquisition_method=method,
        selected_energy_ceiling_cm=ceiling, target_max_cm=target_max_cm,
        gaussian_sigma_cm=gaussian_sigma_cm, tolerance_per_strength=tolerance_per_strength,
        basis_size=n, level_count=len(energies), conservative_workspace_bytes=required,
        **_qualify(hamiltonian, energies, vectors, workers=workers,
                   maximum_workspace_bytes=maximum_workspace_bytes))
    diagnostics.update({key:value for key,value in extra.items() if key not in ('solver','acquisition_seconds')})
    diagnostics['acquisition_seconds'] = time.monotonic()-started
    return PreparedGaussianSpectrum(energies, vectors, None if len(energies)==n else ceiling,
        hamiltonian_fingerprint(hamiltonian), diagnostics)


def prepare_lowest_spectrum(hamiltonian, states_path, *, workers=1):
    """Reuse a recorded complete-lowest subset from the general variational solver."""
    with np.load(states_path, allow_pickle=False) as data:
        diagnostics = json.loads(str(data['diagnostics_json']))
        if (diagnostics.get('schema') != 'matrix.sati.bare_quartic_variational.v1'
                or not np.array_equal(data['basis'], hamiltonian.basis)
                or not np.array_equal(data['frequencies_cm'], hamiltonian.frequencies_cm)):
            raise ValueError('complete-lowest variational solve in the identical basis required')
        energies, vectors = data['energies_cm'], data['eigenvectors']
    checks = _qualify(hamiltonian, energies, vectors, workers=workers)
    return PreparedGaussianSpectrum(energies, vectors,
        None if len(energies)==len(hamiltonian.basis) else float(energies[-1]),
        hamiltonian_fingerprint(hamiltonian), dict(
            schema='matrix.trinity.prepared_gaussian_spectrum.v1',
            solver='complete-lowest subset from general quartic solver',
            source_file=str(states_path), source_sha256=_file_sha256(states_path),
            basis_size=len(hamiltonian.basis), level_count=len(energies), **checks))


def save_gaussian_spectrum(spectrum, directory):
    """Save real/complex arrays without compression; allow read-only memory mapping."""
    directory = Path(directory); directory.mkdir(exist_ok=False, parents=True)
    if isinstance(spectrum, PreparedBlockGaussianSpectrum):
        blocks = []; checksums = {}
        for index, (rows, local) in enumerate(spectrum.parts):
            name = f'rows-{index:04d}.npy'; child = f'block-{index:04d}'
            np.save(directory/name, rows, allow_pickle=False)
            checksums[name] = _file_sha256(directory/name)
            save_gaussian_spectrum(local, directory/child)
            blocks.append(dict(rows=name, directory=child))
        metadata = dict(storage='disjoint_row_blocks.v1', basis_size=spectrum.basis_size,
            blocks=blocks, hamiltonian_sha256=spectrum.hamiltonian_sha256,
            omitted_lower_bound_cm=spectrum.omitted_lower_bound_cm,
            diagnostics=spectrum.diagnostics, sha256=checksums)
        temporary = directory/'metadata.tmp'
        temporary.write_text(json.dumps(metadata, indent=2)+'\n')
        temporary.replace(directory/'metadata.json')
        return
    np.save(directory/'energies_cm.npy', spectrum.energies_cm, allow_pickle=False)
    np.save(directory/'eigenvectors.npy', np.asfortranarray(spectrum.eigenvectors), allow_pickle=False)
    checksums = {name: _file_sha256(directory/name)
                 for name in ('energies_cm.npy', 'eigenvectors.npy')}
    metadata = dict(hamiltonian_sha256=spectrum.hamiltonian_sha256,
        omitted_lower_bound_cm=spectrum.omitted_lower_bound_cm,
        diagnostics=spectrum.diagnostics, sha256=checksums)
    temporary = directory/'metadata.tmp'
    temporary.write_text(json.dumps(metadata, indent=2)+'\n')
    temporary.replace(directory/'metadata.json')


def load_compatible_gaussian_spectrum(directory, hamiltonian, *, target_max_cm,
        gaussian_sigma_cm, tolerance_per_strength=1e-16):
    """Reuse an intact qualified checkpoint for this matrix and Gaussian window.

    Missing checkpoints return None. Incompatible or damaged checkpoints raise;
    existing scientific artifacts are never overwritten or silently accepted.
    """
    if (not np.isfinite(target_max_cm) or not np.isfinite(gaussian_sigma_cm)
            or gaussian_sigma_cm <= 0 or not np.isfinite(tolerance_per_strength)
            or tolerance_per_strength <= 0):
        raise ValueError('finite target and positive linewidth/tolerance required')
    directory = Path(directory)
    if not directory.exists():
        return None
    spectrum = load_gaussian_spectrum(directory)
    if spectrum.hamiltonian_sha256 != hamiltonian.fingerprint:
        raise ValueError('prepared spectrum belongs to another Hamiltonian')
    peak = 1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    ceiling = target_max_cm+gaussian_sigma_cm*np.sqrt(
        2*max(0., np.log(2*peak/tolerance_per_strength)))
    if (spectrum.omitted_lower_bound_cm is not None
            and spectrum.omitted_lower_bound_cm < ceiling):
        raise ValueError('prepared spectrum does not cover requested Gaussian window')
    parts = (tuple(local for _, local in spectrum.parts)
             if isinstance(spectrum, PreparedBlockGaussianSpectrum) else (spectrum,))
    for local in parts:
        residual = local.diagnostics.get('maximum_residual_cm', np.inf)
        orthogonality = local.diagnostics.get('maximum_orthogonality_error', np.inf)
        if (not np.isfinite(residual) or not 0 <= residual <= 1e-6
                or not np.isfinite(orthogonality) or not 0 <= orthogonality <= 1e-8):
            raise ValueError('prepared spectrum has no valid numerical qualification')
    return spectrum


def load_gaussian_spectrum(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    storage = metadata.get('storage', 'dense')
    if storage == 'disjoint_row_blocks.v1':
        parts = []; expected_files = set()
        for index, block in enumerate(metadata['blocks']):
            name = f'rows-{index:04d}.npy'; child = f'block-{index:04d}'
            if block != dict(rows=name, directory=child):
                raise ValueError('invalid spectral block filenames')
            expected_files.add(name)
            if _file_sha256(directory/name) != metadata['sha256'].get(name):
                raise ValueError('prepared spectral row checksum mismatch')
            local = load_gaussian_spectrum(directory/child)
            if not isinstance(local, PreparedGaussianSpectrum):
                raise ValueError('nested spectral blocks are not supported')
            parts.append((np.load(directory/name, mmap_mode='r'), local))
        if set(metadata['sha256']) != expected_files:
            raise ValueError('invalid spectral row checksum manifest')
        return PreparedBlockGaussianSpectrum(metadata['basis_size'], tuple(parts),
            metadata['omitted_lower_bound_cm'], metadata['hamiltonian_sha256'], metadata['diagnostics'])
    if storage != 'dense':
        raise ValueError('unknown prepared spectrum storage format')
    for name, expected in metadata['sha256'].items():
        if name not in ('energies_cm.npy', 'eigenvectors.npy'):
            raise ValueError('invalid spectral array filename')
        if _file_sha256(directory/name)!=expected:
            raise ValueError('prepared spectral array checksum mismatch')
    return PreparedGaussianSpectrum(np.load(directory/'energies_cm.npy', mmap_mode='r'),
        np.load(directory/'eigenvectors.npy', mmap_mode='r'), metadata['omitted_lower_bound_cm'],
        metadata['hamiltonian_sha256'], metadata['diagnostics'])


__all__ = ['PreparedGaussianSpectrum', 'PreparedBlockGaussianSpectrum', 'acquire_gaussian_spectrum',
           'prepare_lowest_spectrum', 'save_gaussian_spectrum', 'load_gaussian_spectrum',
           'load_compatible_gaussian_spectrum']
