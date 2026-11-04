"""Exact tensor-pair evaluation of the complete active weak rovibrational form."""
import numpy as np
from scipy.sparse import eye
from .native_tensor_projection import project_pruned_product_field
from .coupled_bath_galerkin import CoupledBathGalerkin,_constant_phase_real


def active_tensor_field_projection(core_basis, values_axes, momenta_axes, amplitude,
                                   log_envelope_gradient, coefficient_transform):
    """Exact field projector for weighted product values and weak derivatives.

    Returned callable accepts (left_axis,field,right_axis); None selects values
    and an integer selects the corresponding momentum action. Volume factors
    are weights, while the envelope ratio is differentiated explicitly.
    """
    B=tuple(np.asarray(x) for x in values_axes);P=tuple(np.asarray(x) for x in momenta_axes)
    a=np.asarray(amplitude);ell=np.asarray(log_envelope_gradient);T=np.asarray(coefficient_transform)
    d=len(B);n=int(np.prod([x.shape[1] for x in B]));size=len(core_basis)*len(B[-1])
    if (len(P)!=d or a.shape!=(n,) or ell.shape!=(n,d) or T.ndim!=2 or T.shape[0]!=size
            or not all(np.all(np.isfinite(x)) for x in (a,ell,T))):
        raise ValueError('Matching tensor trial space, physical weights and transform required')
    tables=[B]
    for s in range(d):
        t=list(B);t[s]=P[s];tables.append(tuple(t))
    def projection(left_axis, field, right_axis):
        left=0 if left_axis is None else left_axis+1
        right=0 if right_axis is None else right_axis+1
        f=np.asarray(field)*a*a
        result=project_pruned_product_field(core_basis,tables[left],tables[right],f)
        if left or right:result=result.astype(complex)
        if left:
            result+=1j*project_pruned_product_field(core_basis,B,tables[right],f*ell[:,left-1])
        if right:
            result-=1j*project_pruned_product_field(core_basis,tables[left],B,f*ell[:,right-1])
        if left and right:
            result+=project_pruned_product_field(core_basis,B,B,f*ell[:,left-1]*ell[:,right-1])
        if not np.iscomplexobj(T):
            phase,real=_constant_phase_real(result)
            if phase is not None:return phase*(T.T@real@T)
        return T.conj().T@result@T
    return projection


def contract_active_metric_tensor(core_basis,values_axes,momenta_axes,amplitude,
        log_envelope_gradient,kinetic_cm,potential_cm,*,progress=None):
    """Project the same value/momentum functions without tall basis matrices.

    ``amplitude`` is sqrt(volume density) times the envelope ratio. The
    envelope gradient differentiates that ratio only, exactly as the weak
    physical-measure owner; no volume derivative is inserted.
    """
    B=tuple(np.asarray(f) for f in values_axes);P=tuple(np.asarray(f) for f in momenta_axes)
    d=len(B);a=np.asarray(amplitude,float);ell=np.asarray(log_envelope_gradient,float)
    G=np.asarray(kinetic_cm,float);V=np.asarray(potential_cm,float);N=len(V)
    if (d<2 or len(P)!=d or a.shape!=(N,) or ell.shape!=(N,d) or G.shape!=(N,d+3,d+3)
            or np.any(a<=0) or not all(np.all(np.isfinite(x)) for x in (a,ell,G,V))
            or not np.allclose(G,G.swapaxes(-1,-2)) or np.linalg.eigvalsh(G).min()<=0):
        raise ValueError('Positive complete metric and compatible finite tensor fields required')
    weight=a*a;done=0
    def project(left,right,field):
        nonlocal done
        result=project_pruned_product_field(core_basis,left,right,field)
        done+=1
        if progress:progress(dict(stage='exact_tensor_projection',completed=done,total=2+d*(d+1)//2+4*d+6))
        return result
    overlap=project(B,B,weight)
    H=project(B,B,weight*(V+.5*np.einsum('pi,pij,pj->p',ell,G[:,:d,:d],ell)))
    derivative=[]
    for i in range(d):
        tables=list(B);tables[i]=P[i];derivative.append(tuple(tables))
    for i in range(d):
        for j in range(i,d):
            term=project(derivative[i],derivative[j],weight*G[:,i,j])
            H+=.5*term if i==j else .5*(term+term.conj().T)
        term=project(derivative[i],B,weight*np.einsum('pi,pi->p',G[:,i,:d],ell))
        H=H-.5j*term+.5j*term.conj().T
    linear=[];quadratic=np.empty((3,3)+H.shape,complex)
    for axis in range(3):
        L=np.zeros_like(H,complex)
        for i in range(d):
            term=project(derivative[i],B,weight*G[:,i,d+axis])
            L+=.5*(term+term.conj().T)
        linear.append(L)
        for other in range(axis,3):
            term=.5*project(B,B,weight*G[:,d+axis,d+other])
            quadratic[axis,other]=quadratic[other,axis]=term
    error=float(abs(H-H.conj().T).max())
    if error>1e-7:raise ValueError('Tensor weak form is not Hermitian')
    L=np.linalg.cholesky(overlap);R=np.linalg.solve(L.conj().T,np.eye(len(L)))
    def normalize(M):
        if not np.iscomplexobj(R):
            phase,real=_constant_phase_real(M)
            if phase is not None:return phase*(R.T@real@R)
        return R.conj().T@M@R
    H=normalize(.5*(H+H.conj().T))
    identity=eye(1,format='csr')
    lterms=tuple(((normalize(M),identity),) for M in linear)
    qterms=tuple(tuple(((normalize(quadratic[i,j]),identity),) for j in range(3)) for i in range(3))
    operator=CoupledBathGalerkin(H,lterms,qterms,len(H),1,((0,),),error,
        'exact finite tensor pair contraction; no basis or coupling truncation')
    return operator,R
