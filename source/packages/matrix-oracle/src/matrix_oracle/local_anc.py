"""Local-model conditioning with analytic rigid motions, no pair scoring.

This is an explicitly different rigid-space treatment from literal xTB ANC:
the exact reference still scores displaced all-pair distances. Both use the
same simple-primitive model, Hlow=0.01 and upper spectral cap=5 (atomic units).
"""
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from matrix_chem.structural_corrections import BOHR_TO_ANGSTROM


def rigid_motion_rows(xyz):
    """Assemble six atomwise fields and orthonormalize only a 6-by-3N block."""
    xyz = np.asarray(xyz, dtype=float)
    centered = xyz-xyz.mean(axis=0)
    fields = [np.tile(axis, len(xyz)) for axis in np.eye(3)]
    fields.extend(np.cross(axis, centered).ravel() for axis in np.eye(3))
    _, singular, rows = np.linalg.svd(np.asarray(fields), full_matrices=False)
    scale = max(float(singular[0]), 1.)
    threshold = max(1.e-10*scale, np.finfo(float).eps*max(6,xyz.size)*scale)
    rank = np.count_nonzero(singular > threshold)
    if rank not in (3,5,6):
        raise ValueError('ambiguous rigid-motion rank')
    return rows[:rank]


def condition_local_model(raw, atoms, xyz, *, spectral_budget=128):
    from .initial_hessian import OracleInitialHessian
    rows = rigid_motion_rows(xyz)
    n = xyz.size
    project = lambda v: np.asarray(v)-rows.T@(rows@np.asarray(v))
    action = lambda v: project(raw.cartesian_action(project(v)))*BOHR_TO_ANGSTROM**2
    if n <= rows.shape[0]:
        raise ValueError('initial Hessian has no non-rigid degrees of freedom')
    rng = np.random.default_rng(98231)
    calls = 0
    inverse_audits = []
    low_solver = 'BOUNDED_LANCZOS'
    def spectrum(function, k, which):
        nonlocal calls
        calls += 1
        operator = LinearOperator((n,n), matvec=function, dtype=float)
        try:
            values, vectors = eigsh(operator, k=k, which=which, tol=1.e-10,
                maxiter=128, ncv=min(n, max(96,2*k+1)), v0=rng.normal(size=n))
        except ArpackNoConvergence:
            # Highly repeated molecular blocks can make the strict bounded
            # Lanczos restart miss an otherwise elementary extremal root.
            # Retry the same sparse operator with a larger iteration budget;
            # no dense fallback or alternative Hessian is introduced.
            values, vectors = eigsh(operator, k=k, which=which, tol=1.e-9,
                maxiter=1024, ncv=min(n, max(96,2*k+1)), v0=rng.normal(size=n))
        residual = max(np.linalg.norm(function(vectors[:,i])-values[i]*vectors[:,i]) for i in range(k))
        if residual > 1.e-8:
            raise FloatingPointError('local SH spectral residual failed')
        return values, vectors
    # A fixed 0.02 external penalty is above Hlow. It cannot affect the
    # damping decision and avoids seeking ALL soft eigenvectors below 0.05.
    low_action = lambda v: action(v)+.02*(rows.T@(rows@v))
    if n > 96:
        from matrix_chem.sparse_krylov import SparseKrylovPreconditioner, checked_gmres
        # B.T K B is PSD. Gershgorin on the sparse D2 blocks therefore gives
        # a rigorous lower spectral bound without a spectral search.
        pair = raw.pair_hessian
        diagonal = pair.diagonal()
        lower = float(np.min(diagonal-np.asarray(abs(pair).sum(axis=1)).ravel()+np.abs(diagonal)))
        sigma = min(lower,0.)-.001
        sparse_raw = raw.b_matrix.T@raw.b_matrix.multiply(raw.curvatures[:,None])+pair
        inverse_preconditioner = SparseKrylovPreconditioner(sparse_raw, diagonal_shift=-sigma)
        external = lambda v: rows.T@(rows@v)
        def inverse(v):
            solution, audit = checked_gmres(lambda x:low_action(x)-sigma*x, v,
                lambda x:project(inverse_preconditioner.apply(project(x)))+external(x)/(.02-sigma),
                maximum_iterations=128, tolerance=1.e-10)
            inverse_audits.append(audit)
            return solution
        calls += 1
        try:
            low, vectors = eigsh(LinearOperator((n,n),matvec=low_action,dtype=float), k=1,
                sigma=sigma, which='LM', OPinv=LinearOperator((n,n),matvec=inverse,dtype=float),
                ncv=min(n,32), tol=1.e-10, maxiter=128, v0=rng.normal(size=n))
            if np.linalg.norm(low_action(vectors[:,0])-low[0]*vectors[:,0]) > 1.e-8:
                raise FloatingPointError('shift-invert extremal spectral residual failed')
            low_solver = 'SPARSE_PRECONDITIONED_SHIFT_INVERT'
        except (ArpackNoConvergence, FloatingPointError, RuntimeError):
            # Degenerate weakly coupled fragments are the difficult case for
            # shift-invert/GMRES.  A direct extremal Lanczos solve remains
            # sparse, bounded-memory and is markedly more robust there.
            low, vectors = spectrum(low_action, 1, 'SA')
            low_solver = 'DIRECT_SPARSE_LANCZOS_RETRY'
    else:
        low, _ = spectrum(low_action, 1, 'SA')
    shift = max(.01-float(low[0]), 0.)
    shifted = lambda v: action(v)+shift*project(v)
    high, vectors = spectrum(shifted, 1, 'LA')
    if high[-1] > 5.:
        k = min(12,n-1,spectral_budget)
        while True:
            high, vectors = spectrum(shifted, k, 'LA')
            if np.min(high) <= 5.:
                break
            if k >= spectral_budget or k >= n-1:
                raise RuntimeError('local SH upper-clipping spectral budget exceeded')
            k = min(2*k,n-1,spectral_budget)
        mask = high > 5.
        corrections = 5.-high[mask]
        vectors = vectors[:,mask]
    else:
        corrections = np.empty(0)
        vectors = np.empty((n,0))
    frozen = np.asarray(xyz).copy()
    frozen.setflags(write=False)
    audit = dict(raw.audit, owner='ORACLE', model='XTB_LOCAL_ANALYTIC_RIGID_CONDITIONED',
        formula='P_rigid (H_local + shift I) P_rigid with upper spectral cap',
        anc_conditioning=True, literal_xtb_anc=False, damping_shift=shift,
        rigid_mode_count=len(rows), rigid_mode_construction='ATOMWISE_TRANSLATIONS_AND_ROTATIONS',
        rigid_mode_pair_comparisons=0, rigid_mode_storage=int(rows.size),
        anc_candidate_scoring='NOT_USED_ANALYTIC_RIGID_PROJECTOR',
        spectral_correction_rank=len(corrections), spectral_budget=spectral_budget,
        spectral_calls=calls, bounded_spectral_iterations=128,
        low_spectrum_solver=low_solver,
        shift_invert_solve_count=len(inverse_audits),
        shift_invert_maximum_iterations=max((a['iterations'] for a in inverse_audits),default=0),
        shift_invert_maximum_relative_residual=max((a['relative_residual'] for a in inverse_audits),default=0.),
        dense_cartesian_hessian=False, conditioned_error_certified=False,
        raw_model_scaling=raw.audit['scaling'],
        conditioned_model_scaling='CONDITIONAL_ON_LOCAL_DENSITY_AND_FIXED_SPECTRAL_BUDGET_SUCCESS')
    return OracleInitialHessian(raw, shift, vectors, corrections, audit, tuple(atoms), frozen, rows)
