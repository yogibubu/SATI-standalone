"""Hessian-only calibration on SONIC; dense, bounded audit, never production.

Target: J.T H_xTB J. SH: A.T K A, A = B_primitive J.
No gradient or nonlinear-coordinate Hessian correction belongs to this
comparison of initial harmonic models at a common geometry.
"""
import numpy as np
from scipy.optimize import nnls


def sonic_target(hessian_cartesian_bohr, cartesian_from_sonic_angstrom):
    from matrix_chem.structural_corrections import BOHR_TO_ANGSTROM
    h=np.asarray(hessian_cartesian_bohr,dtype=float)
    j=np.asarray(cartesian_from_sonic_angstrom,dtype=float)/BOHR_TO_ANGSTROM
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or j.ndim!=2 or j.shape[0]!=h.shape[0]:
        raise ValueError('incompatible Cartesian Hessian and SONIC realization')
    if not np.isfinite(h).all() or not np.isfinite(j).all():
        raise ValueError('nonfinite SONIC Hessian target input')
    if not np.allclose(h,h.T,rtol=1.e-10,atol=1.e-12):
        raise ValueError('Cartesian Hessian must be symmetric')
    return j.T@h@j


def reproduction_error(actual,target):
    a=np.asarray(actual,dtype=float);t=np.asarray(target,dtype=float)
    scale=np.sqrt(np.maximum(np.abs(np.diag(t)),1.e-14))
    error=(a-t)/scale[:,None]/scale[None,:]
    return {'relative_frobenius':float(np.linalg.norm(a-t)/max(np.linalg.norm(t),1.e-14)),
        'max_scaled_element':float(np.max(np.abs(error))),
        'max_relative_diagonal':float(np.max(np.abs(np.diag(error)))),
        'scaled_frobenius':float(np.linalg.norm(error)/np.sqrt(len(t)))}


def fit_diagonal_primitive_sh(primitive_from_sonic,target,*,tolerance=1.e-6):
    """Test whether nonnegative diagonal primitive K can reproduce full Hq.

    Target-diagonal scaling makes the elementwise test independent of the
    units chosen for each SONIC. Columns correspond to individual elementary
    curvatures; this feasibility test is more permissive than a family atlas.
    """
    a=np.asarray(primitive_from_sonic,dtype=float);h=np.asarray(target,dtype=float)
    if a.ndim!=2 or h.shape!=(a.shape[1],a.shape[1]) or not a.size:
        raise ValueError('incompatible SH calibration dimensions')
    if not np.isfinite(a).all() or not np.isfinite(h).all():
        raise ValueError('nonfinite SH calibration input')
    design_bytes = a.shape[0]*a.shape[1]*(a.shape[1]+1)//2*8
    if a.shape[0]>4096 or a.shape[1]>384 or design_bytes>512*1024**2:
        raise ValueError('dense SH calibration exceeds the explicit audit budget')
    scales=np.sqrt(np.maximum(np.abs(np.diag(h)),1.e-14))
    z=a/scales
    i,j=np.triu_indices(len(h));w=np.where(i==j,1.,np.sqrt(2.))
    design=(z[:,i]*z[:,j]).T*w[:,None]
    rhs=(h/scales[:,None]/scales[None,:])[i,j]*w
    norms=np.linalg.norm(design,axis=0);active=norms>1.e-14
    constants=np.zeros(len(a))
    if np.any(active):
        fitted,_=nnls(design[:,active]/norms[active],rhs,maxiter=max(1000,10*len(a)))
        constants[active]=fitted/norms[active]
    reproduced=a.T@(constants[:,None]*a)
    error=reproduction_error(reproduced,h)
    return constants,reproduced,dict(error,
        reproduced=error['max_scaled_element']<=tolerance,tolerance=tolerance,
        target='J.T H_xTB J',model='A.T diag(k) A',
        zero_curvatures=int(np.count_nonzero(constants==0)),
        production_enabled=False,gradient_evaluations=0)


def coupled_primitive_reference(primitive_from_sonic,target,*,tolerance=1.e-8):
    """Diagnostic feasibility oracle allowing off-diagonal primitive K.

    The returned matrix is not a deployable local force field. It tests if
    the selected primitive span can reproduce the target when couplings
    are retained, as distinct from the diagonal-only restriction.
    """
    a=np.asarray(primitive_from_sonic,dtype=float);h=np.asarray(target,dtype=float)
    inverse=np.linalg.pinv(a,rcond=1.e-10)
    k=inverse.T@h@inverse
    reproduced=a.T@k@a
    error=reproduction_error(reproduced,h)
    return k,reproduced,dict(error,reproduced=error['max_scaled_element']<=tolerance,
        tolerance=tolerance,model='A.T K_full A',production_enabled=False,
        gradient_evaluations=0)


def fit_minimax_parameters(grams, linear, *, lower=1.e-4, upper=100., prevent_regression=False):
    """Fit shared curvature multipliers, minimizing worst molecular error.

    Each quadratic is ||sum_j c_j H_j - target||_F^2/||target||_F^2.
    Gram/linear data must already include this molecular normalization.
    Dense optimization belongs only to offline parameter calibration.
    """
    from scipy.optimize import minimize
    g = np.asarray(grams, dtype=float)
    b = np.asarray(linear, dtype=float)
    if g.ndim != 3 or g.shape[1] != g.shape[2] or b.shape != g.shape[:2]:
        raise ValueError('incompatible parameter calibration quadratics')
    if not np.isfinite(g).all() or not np.isfinite(b).all() or not len(g):
        raise ValueError('invalid parameter calibration quadratics')
    def errors(c):
        return np.einsum('i,mij,j->m', c, g, c)-2*b@c+1
    def derivatives(c):
        return 2*(np.einsum('mij,j->mi',g,c)-b)
    n = b.shape[1]
    def objective(x):
        return x[-1]+0.05*np.mean(errors(x[:-1]))+1.e-8*np.sum((x[:-1]-1)**2)
    def jacobian(x):
        return np.r_[0.05*np.mean(derivatives(x[:-1]),axis=0)+2.e-8*(x[:-1]-1),1.]
    initial = np.r_[np.ones(n),max(0.,max(errors(np.ones(n))))+1.e-8]
    constraints=[{
        'type':'ineq','fun':lambda x:x[-1]-errors(x[:-1]),
        'jac':lambda x:np.column_stack((-derivatives(x[:-1]),np.ones(len(g))))}]
    if prevent_regression:
        ceilings=errors(np.ones(n))+1.e-10
        constraints.append({'type':'ineq','fun':lambda x:ceilings-errors(x[:-1]),
            'jac':lambda x:np.column_stack((-derivatives(x[:-1]),np.zeros(len(g))))})
    result = minimize(objective, initial, jac=jacobian, method='SLSQP',
        bounds=[(lower,upper)]*n+[(0.,None)], constraints=constraints,
        options={'ftol':1.e-11,'maxiter':2000})
    violation = max(0.,float(max(errors(result.x[:-1])-result.x[-1])))
    if prevent_regression:
        violation=max(violation,float(max(errors(result.x[:-1])-ceilings)))
    if not result.success or violation>1.e-7:
        raise RuntimeError(f'SH minimax fit failed: {result.message}; violation={violation}')
    return result.x[:-1], dict(success=True,iterations=int(result.nit),
        maximum_training_error=float(np.sqrt(max(0.,max(errors(result.x[:-1]))))),
        constraint_violation=violation, objective='worst relative Frobenius squared + 0.05 mean',
        lower_bound=lower,upper_bound=upper,prevent_regression=prevent_regression)
