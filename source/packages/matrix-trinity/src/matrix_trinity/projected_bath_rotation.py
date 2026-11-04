"""Fixed fast-bath rotational resolvent in an arbitrary contracted LAM space.

No grid in the explicit oscillator coordinates is introduced. Projected
coordinate and Hermitian momentum actions on one normalized LAM state are
supplied by the caller. Full second coordinate moments retain closure beyond
the truncated LAM space. The shifted positive resolvent includes that entire
space, rather than a truncated list of its excited eigenstates.
"""
import numpy as np
from scipy.sparse.linalg import aslinearoperator,LinearOperator,cg
from .adiabatic_coriolis import _root


def dress_projected_fixed_bath_rotation(hamiltonian_cm,energy_cm,state,
        momentum_actions,coordinate_actions,coordinate_second_expectation,
        active_kinetic_cm,bath_kinetic_cm,bath_force_cm,coupling_a_cm,
        coupling_c_derivative_cm,*,relative_tolerance=1e-10,maximum_iterations=3000):
    H=aslinearoperator(hamiltonian_cm);psi=np.asarray(state,complex)
    p=np.asarray(momentum_actions,complex);q=np.asarray(coordinate_actions,complex)
    q2=np.asarray(coordinate_second_expectation,float);ga=np.asarray(active_kinetic_cm,float)
    gb=np.asarray(bath_kinetic_cm,float);hb=np.asarray(bath_force_cm,float)
    a=np.asarray(coupling_a_cm,float);dc=np.asarray(coupling_c_derivative_cm,float)
    if psi.ndim!=1 or p.ndim!=2:raise ValueError('State and projected momentum actions required')
    N,d=p.shape;nb=len(gb)
    if (H.shape!=(N,N) or psi.shape!=(N,) or q.shape!=p.shape or q2.shape!=(d,d)
            or ga.shape!=(d,d) or gb.shape!=(nb,nb) or hb.shape!=gb.shape
            or a.shape!=(d,nb,3) or dc.shape!=a.shape or not 0<relative_tolerance<1
            or maximum_iterations<1 or not np.isfinite(energy_cm)
            or not all(np.all(np.isfinite(x)) for x in (psi,p,q,q2,ga,gb,hb,a,dc))
            or abs(np.vdot(psi,psi)-1)>1e-8):raise ValueError('Compatible finite normalized projected fields required')
    for matrix in (ga,gb,hb):
        if not np.allclose(matrix,matrix.T) or np.linalg.eigvalsh(matrix).min()<=0:raise ValueError('Positive kinetic and fast force matrices required')
    if not np.allclose(q2,q2.T) or np.linalg.eigvalsh(q2).min()<-1e-10:raise ValueError('Positive symmetric unprojected coordinate moments required')
    residual=float(np.linalg.norm(H@psi-energy_cm*psi))
    if residual>1e-5:raise ValueError('The supplied state does not solve the LAM Hamiltonian')
    gh=_root(gb);w2,U=np.linalg.eigh(gh@hb@gh);w=np.sqrt(w2)
    qb=gh@(U/np.sqrt(w)[None,:]);pb=np.linalg.inv(gh)@(U*np.sqrt(w)[None,:])
    aa=np.einsum('sbr,bm->smr',a,qb)/np.sqrt(2)
    cc=np.einsum('sbr,bm->smr',dc,pb)/np.sqrt(2)
    diagonal=getattr(hamiltonian_cm,'diagonal_values',None)
    if diagonal is None and isinstance(hamiltonian_cm,np.ndarray):diagonal=np.diag(hamiltonian_cm)
    virtual=np.zeros((3,3));solver_residuals=[]
    for mode,gap in enumerate(w):
        W=p@aa[:,mode,:]+1j*q@cc[:,mode,:]
        shifted=LinearOperator(H.shape,matvec=lambda x,gap=gap:H@x+(gap-energy_cm)*x,dtype=complex)
        M=None
        if diagonal is not None:
            denom=np.asarray(diagonal).real+gap-energy_cm
            if denom.min()<=0:raise ValueError('Nonpositive shifted diagonal preconditioner')
            M=LinearOperator(H.shape,matvec=lambda x,denom=denom:x/denom,dtype=complex)
        X=np.zeros_like(W);errors=[]
        for axis in range(3):
            norm=np.linalg.norm(W[:,axis])
            if norm<1e-15:errors.append(0.);continue
            solution,info=cg(shifted,W[:,axis],M=M,rtol=relative_tolerance,atol=0.,maxiter=maximum_iterations)
            error=float(np.linalg.norm(shifted@solution-W[:,axis])/norm)
            if info or error>10*relative_tolerance:raise RuntimeError(f'Fast-bath shifted resolvent failed: {mode}, {axis}, {info}, {error}')
            X[:,axis]=solution;errors.append(error)
        virtual-=np.real(W.conj().T@X);solver_residuals.append(errors)
    direct=np.einsum('st,sbr,bc,tcd->rd',np.linalg.inv(ga),a,qb@qb.T/2,a,optimize=True)/2
    direct+=np.einsum('st,sbr,bc,tcd->rd',q2,dc,np.linalg.inv(gb),dc,optimize=True)/2
    virtual=.5*(virtual+virtual.T);direct=.5*(direct+direct.T)
    return dict(paired_tensor_cm=direct+virtual,direct_tensor_cm=direct,virtual_tensor_cm=virtual,
        response_relative_residuals=np.array(solver_residuals),ground_eigen_residual_cm=residual,
        minimum_fast_frequency_cm=float(w.min()),fast_frequencies_cm=w,contracted_LAM_dimension=N,
        model='fixed_quadratic_fast_bath_projected_paired_resolvent',
        limitations='Constant fast bath and leading mixed rotational vertex; no higher fast-bath anharmonicity or variable-fiber derivative metric.')
