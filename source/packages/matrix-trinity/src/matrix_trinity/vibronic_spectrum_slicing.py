"""Parallel sparse spectral windows with inertia-checked completeness.

Each independent interval uses a shifted PARDISO factor and ARPACK inverse
iteration. Inertia differences determine how many eigenpairs must be present;
an incomplete interval never becomes a prepared Gaussian spectrum.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import atexit
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import LinearOperator, eigsh
from threadpoolctl import threadpool_limits

_MATRIX = _BACKEND = _BLAS_LIMIT = _FACTOR = None


def _initialize(matrix_path, library_path, workers):
    global _MATRIX, _BACKEND, _BLAS_LIMIT
    from .vibronic_mkl import MKLSpectralBackend
    _BLAS_LIMIT = threadpool_limits(limits=1,user_api='blas')
    _MATRIX = load_npz(matrix_path)
    _BACKEND = MKLSpectralBackend(library_path,workers=workers)
    atexit.register(_close_factor)


def _close_factor():
    global _FACTOR
    if _FACTOR is not None:
        _FACTOR.close();_FACTOR=None


def _atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


@contextmanager
def _factor_near(requested, upper):
    global _FACTOR
    for fraction in (0.,1e-7,1e-5,1e-3):
        shift=requested+fraction*(upper-requested)
        try:
            if _FACTOR is not None:
                try:_FACTOR.refactor(shift)
                except ArithmeticError:
                    _close_factor()
                    _FACTOR=_BACKEND.shifted_factor(_MATRIX,shift)
            else:_FACTOR=_BACKEND.shifted_factor(_MATRIX,shift)
        except ArithmeticError as error:
            _close_factor()
            if 'inertia refused' not in str(error):raise
            continue
        yield _FACTOR
        return
    raise ArithmeticError('could not acquire an unperturbed shifted factor')


def _count(arguments):
    requested,upper,cache = arguments
    key = hashlib.sha256(float(requested).hex().encode()).hexdigest()[:24]
    path = Path(cache)/('count_'+key+'.json')
    if path.exists():return json.loads(path.read_text())
    with _factor_near(requested,upper) as factor:
        result=dict(factor.diagnostics,requested_shift_cm=requested)
    _atomic_json(path,result)
    return result


def _window(arguments):
    lower,upper,count,cache,tolerance = arguments
    key = hashlib.sha256(repr((float(lower).hex(),float(upper).hex(),count,tolerance)).encode()).hexdigest()[:24]
    directory = Path(cache)/('window_'+key)
    metadata = directory/'metadata.json'
    if metadata.exists():return dict(json.loads(metadata.read_text()),directory=str(directory))
    directory.mkdir(exist_ok=True)
    n = _MATRIX.shape[0]
    requested = min(n-1,count+max(8,min(32,count//4)))
    if requested<count:raise MemoryError('sparse interval contains too many levels for inverse iteration')
    midpoint = .5*(lower+upper)
    started = time.monotonic()
    with _factor_near(midpoint,upper) as factor:
        midpoint=factor.shift_cm
        inverse = LinearOperator(_MATRIX.shape,matvec=factor.solve,dtype=np.float64)
        initial = np.random.default_rng(int(key[:12],16)).normal(size=n)
        for attempt in range(2):
            ncv = min(n,max(3*requested,40))
            energies,vectors = eigsh(_MATRIX,k=requested,sigma=midpoint,which='LM',
                OPinv=inverse,ncv=ncv,tol=tolerance,maxiter=10000,v0=initial)
            keep = (energies>=lower)&(energies<upper)
            if int(keep.sum())==count:break
            requested = min(n-1,max(2*requested,count+64))
        else:raise ArithmeticError('inverse iteration did not recover the inertia-predicted count')
        energies,vectors = energies[keep],vectors[:,keep]
        order = np.argsort(energies);energies,vectors = energies[order],vectors[:,order]
        residual = np.linalg.norm(_MATRIX@vectors-vectors*energies,axis=0)
        maximum = float(residual.max())
        if maximum>1e-6:
            raise ArithmeticError('sliced eigenpair residual exceeds 1e-6 cm-1')
        separation = float(min(energies[0]-lower,upper-energies[-1]))
        if separation<=10*maximum:
            raise ArithmeticError('slice boundary is numerically unresolved; no completeness claim')
        result = dict(lower_cm=lower,upper_cm=upper,level_count=count,
            expected_inertia_difference=count,maximum_residual_cm=maximum,
            minimum_boundary_separation_cm=separation,seconds=time.monotonic()-started,
            factor=factor.diagnostics)
    np.save(directory/'energies.npy',energies,allow_pickle=False)
    np.save(directory/'vectors.npy',np.asfortranarray(vectors),allow_pickle=False)
    _atomic_json(metadata,result)
    return dict(result,directory=str(directory))


def sliced_lowest_spectrum(hamiltonian, ceiling_cm, *, library_path, workers,
        processes=4, levels_per_slice=128, work_directory,
        maximum_workspace_bytes=64*1024**3, tolerance=1e-12, progress=None):
    """Recover every eigenpair below a ceiling, retaining all matrix entries.

    Native MKL is optional and explicitly selected. Counts and completed
    windows are checkpointed against the exact finite-Hamiltonian fingerprint.
    Numerical inertia, residuals and orthogonality are not interval arithmetic.
    """
    for name,value in (('workers',workers),('processes',processes),('levels_per_slice',levels_per_slice)):
        if not isinstance(value,int) or value<=0:raise ValueError('positive '+name+' required')
    if workers<processes or not np.isfinite(ceiling_cm) or not 0<tolerance<1:
        raise ValueError('at least one worker per process and finite ceiling/tolerance required')
    if np.iscomplexobj(hamiltonian.matrix_cm.data):
        raise ValueError('inertia slicing currently requires a real symmetric Hamiltonian')
    directory = Path(work_directory);directory.mkdir(parents=True,exist_ok=True)
    identity = dict(hamiltonian_sha256=hamiltonian.fingerprint,
        requested_ceiling_cm=float(ceiling_cm),library_path=str(Path(library_path).resolve()),
        tolerance=float(tolerance),levels_per_slice=levels_per_slice)
    identity_path = directory/'problem.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=identity:raise ValueError('slicing checkpoint belongs to a different problem')
    else:
        save_npz(directory/'matrix.npz',hamiltonian.matrix_cm,compressed=False)
        _atomic_json(identity_path,identity)
    def emit(**data):
        if progress:progress(data)
    started = time.monotonic()
    lower = hamiltonian.spectral_interval_cm[0]-1.
    leaves=[]
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=processes,mp_context=context,
            initializer=_initialize,initargs=(str(directory/'matrix.npz'),
                str(Path(library_path).resolve()),workers//processes)) as pool:
        top = pool.submit(_count,(float(ceiling_cm),float(ceiling_cm)+1.,str(directory))).result()
        ceiling = top['shift_cm'];expected=top['negative_count']
        if not expected:raise ValueError('spectral window has no eigenlevels')
        required = 4*hamiltonian.matrix_cm.shape[0]*expected*8
        if required>maximum_workspace_bytes:raise MemoryError('assembled spectral window exceeds memory budget')
        emit(stage='inertia acquired',level_count=expected,ceiling_cm=ceiling)
        pending=[(lower,ceiling,0,expected)]
        count_records=[top]
        for depth in range(49):
            split=[]
            for lo,hi,nlo,nhi in pending:
                count=nhi-nlo
                if not count:continue
                if count<=levels_per_slice or hi-lo<1e-6 or depth==48:
                    if count>max(512,4*levels_per_slice):raise MemoryError('unresolved highly degenerate interval exceeds slice budget')
                    leaves.append((lo,hi,count,str(directory),tolerance))
                else:split.append((lo,hi,nlo,nhi))
            if not split:break
            records=list(pool.map(_count,[(.5*(lo+hi),hi,str(directory)) for lo,hi,_,_ in split]))
            pending=[]
            for (lo,hi,nlo,nhi),record in zip(split,records):
                middle,nmiddle=record['shift_cm'],record['negative_count']
                if not nlo<=nmiddle<=nhi:raise ArithmeticError('inertia counts are not monotone')
                pending.extend(((lo,middle,nlo,nmiddle),(middle,hi,nmiddle,nhi)))
            count_records.extend(records)
            emit(stage='partitioning',depth=depth+1,completed_intervals=len(leaves),pending_intervals=len(pending))
        leaves.sort(key=lambda row:row[0])
        if sum(row[2] for row in leaves)!=expected:raise ArithmeticError('partition does not conserve inertia count')
        results=[]
        for result in pool.map(_window,leaves):
            results.append(result)
            emit(stage='eigenwindows',completed=len(results),total=len(leaves),
                recovered_levels=sum(item['level_count'] for item in results))
    n = hamiltonian.matrix_cm.shape[0]
    energies=np.empty(expected);vectors=np.empty((n,expected),order='F');offset=0
    for result in results:
        count=result['level_count'];path=Path(result['directory'])
        energies[offset:offset+count]=np.load(path/'energies.npy',mmap_mode='r')
        vectors[:,offset:offset+count]=np.load(path/'vectors.npy',mmap_mode='r')
        offset+=count
    if offset!=expected or np.any(np.diff(energies)<0):raise ArithmeticError('assembled spectrum is incomplete or unordered')
    return energies,vectors,ceiling,dict(solver='PARDISO inertia and parallel ARPACK shifted inverse windows',
        acquisition_seconds=time.monotonic()-started,processes=processes,
        workers_per_process=workers//processes,expected_lowest_level_count=expected,
        recovered_lowest_level_count=offset,slice_count=len(results),
        reused_symbolic_factorizations=sum(bool(row.get('symbolic_analysis_reused'))
            for row in count_records)+sum(bool(row['factor'].get('symbolic_analysis_reused')) for row in results),
        inertia_counts=count_records,windows=results,
        completeness_checked_by_unperturbed_inertia=True,
        coefficient_pruning_used=False,dense_hamiltonian_allocated=False,
        roundoff_included_in_completeness_guarantee=False)


__all__=['sliced_lowest_spectrum']
