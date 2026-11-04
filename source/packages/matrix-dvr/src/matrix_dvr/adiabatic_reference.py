"""Full stationary Cartesian F2/F3/F4 jet of a physical constrained minimum.

Fields use cm^-1 and angstrom powers in the declared Cartesian frame.
Supply jet_transport when they belong to a reference Eckart section. All transverse cubic couplings are retained when computing the second
mean response. No electronic field mode is selected or deleted. Source
stationarity is the same reference assumption made by VPT2.
"""
from itertools import permutations,combinations
import numpy as np
from .mean_tree_bath import mean_tree_bath_jets


def stationary_coordinate_field(force_second,force_third,force_fourth,first,second,third):
    """Transform every F2/F3/F4 entry with a stationary Cartesian geometry jet.

    Derivatives use ordinary Taylor conventions, not factorial-divided
    coefficients. Geometry jets and force tensors must share the same section
    and units. No normal-mode, resonance or observable selection is performed.
    """
    F2,F3,F4,D,DD,D3=(np.asarray(a,float) for a in
        (force_second,force_third,force_fourth,first,second,third))
    nc,n=D.shape if D.ndim==2 else (0,0)
    if (not nc or not n or F2.shape!=(nc,nc) or F3.shape!=(nc,)*3
            or F4.shape!=(nc,)*4 or DD.shape!=(nc,n,n) or D3.shape!=(nc,n,n,n)
            or not all(np.all(np.isfinite(a)) for a in (F2,F3,F4,D,DD,D3))):
        raise ValueError('Matching complete stationary field and Cartesian jets required')
    H=D.T@F2@D
    C=np.einsum('ijk,ia,jb,kc->abc',F3,D,D,D,optimize=True)
    connection=np.einsum('ij,iab,jc->abc',F2,DD,D,optimize=True)
    C+=connection+connection.transpose(0,2,1)+connection.transpose(2,0,1)
    Q=np.einsum('ijkl,ia,jb,kc,ld->abcd',F4,D,D,D,D,optimize=True)
    mixed=np.einsum('ijk,iab,jc,kd->abcd',F3,DD,D,D,optimize=True)
    for pair in combinations(range(4),2):
        axes=list(pair)+[i for i in range(4) if i not in pair];Q+=mixed.transpose(*np.argsort(axes))
    paired=np.einsum('ij,iab,jcd->abcd',F2,DD,DD,optimize=True)
    Q+=paired+paired.transpose(0,2,1,3)+paired.transpose(0,2,3,1)
    value=np.einsum('ij,iabc,jd->abcd',F2,D3,D,optimize=True)
    for singled in range(4):
        axes=[i for i in range(4) if i!=singled]+[singled];Q+=value.transpose(*np.argsort(axes))
    Q=sum(Q.transpose(*p) for p in permutations(range(4)))/24
    return H,C,Q


def minimum_reference_quartic(path,bath_order,force_second,force_third,force_fourth,*,difference_steps=(4e-5,2e-5),jet_transport=None):
    order=np.asarray(bath_order,int);d=len(path.active_indices);n=len(order);nc=3*len(path.tree.order)
    F2=np.asarray(force_second,float);F3=np.asarray(force_third,float);F4=np.asarray(force_fourth,float)
    if (F2.shape!=(nc,nc) or F3.shape!=(nc,)*3 or F4.shape!=(nc,)*4
            or not all(np.all(np.isfinite(t)) for t in (F2,F3,F4))
            or sorted(order.tolist())!=sorted(path.bath_indices.tolist())):
        raise ValueError('Complete stationary Cartesian field and physical bath permutation required')
    raw=path.cartesian_jet(np.zeros(d));permutation=np.array([np.flatnonzero(path.bath_indices==k)[0] for k in order]);ids=np.r_[np.arange(d),d+permutation]
    D0=raw['tangent'][:,ids];DD0=raw['second'][:,ids[:,None],ids[None,:]]
    if jet_transport is not None:
        transported=jet_transport(raw['geometry_A'],D0,DD0);D0=transported['first'];DD0=transported['second']
    H=D0.T@F2@D0
    if np.linalg.eigvalsh(H[d:,d:]).min()<=0:raise ValueError('Stable transverse reference required')
    C=np.einsum('ijk,ia,jb,kc->abc',F3,D0,D0,D0,optimize=True)
    connection=np.einsum('ij,iab,jc->abc',F2,DD0,D0,optimize=True);C+=connection+connection.transpose(0,2,1)+connection.transpose(2,0,1)
    b1=-np.linalg.solve(H[d:,d:],H[d:,:d]);linear=np.eye(d+n);linear[d:,:d]=b1
    shifted=np.einsum('ijk,ia,jb,kc->abc',C,linear,linear,linear,optimize=True)
    b2=-np.linalg.solve(H[d:,d:],shifted[d:,:d,:d].reshape(n,d*d)).reshape(n,d,d)
    partition=dict(bath_order=order,explicit_indices=np.empty(0,int),fast_indices=np.arange(n),conditional_response=np.zeros((1,n,0)),conditional_response_first=np.zeros((1,n,0,d)))
    def mean_jet(q):
        b=b1@q+.5*np.einsum('ist,s,t->i',b2,q,q);first=b1+np.einsum('ist,t->is',b2,q)
        result=mean_tree_bath_jets(path,q[None],b[None],first[None],partition,response_second=b2[None])
        if jet_transport is not None:
            transported=jet_transport(result['geometry_A'][0],result['first'][0],result['second'][0])
            result['geometry_A']=np.asarray(transported['geometry_A'])[None];result['first']=np.asarray(transported['first'])[None];result['second']=np.asarray(transported['second'])[None]
        return result
    origin=mean_jet(np.zeros(d));D=origin['first'][0,:,:d];DD=origin['second'][0,:,:d,:d];fourths=[]
    for step in difference_steps:
        if not np.isfinite(step) or step<=0:raise ValueError('Positive finite geometric differentiation step required')
        D3=np.stack([(mean_jet(np.eye(d)[i]*step)['second'][0,:,:d,:d]-mean_jet(-np.eye(d)[i]*step)['second'][0,:,:d,:d])/(2*step) for i in range(d)],axis=-1)
        D3=sum(D3.transpose(0,*[i+1 for i in p]) for p in permutations(range(3)))/6
        fourth=np.einsum('ijkl,ia,jb,kc,ld->abcd',F4,D,D,D,D,optimize=True)
        mixed=np.einsum('ijk,iab,jc,kd->abcd',F3,DD,D,D,optimize=True)
        for pair in combinations(range(4),2):
            axes=list(pair)+[i for i in range(4) if i not in pair];fourth+=mixed.transpose(*np.argsort(axes))
        paired=np.einsum('ij,iab,jcd->abcd',F2,DD,DD,optimize=True)
        fourth+=paired+paired.transpose(0,2,1,3)+paired.transpose(0,2,3,1)
        third=np.einsum('ij,iabc,jd->abcd',F2,D3,D,optimize=True)
        for singled in range(4):
            axes=[i for i in range(4) if i!=singled]+[singled];fourth+=third.transpose(*np.argsort(axes))
        fourth=sum(fourth.transpose(*p) for p in permutations(range(4)))/24;fourths.append(fourth)
    if not fourths:raise ValueError('At least one geometric derivative step required')
    # Local curvature along the same minimum, using the full Cartesian
    # quartic jet. At a stationary origin no fourth geometric jet is needed.
    def bath_hessian(q):
        j=mean_jet(q);dx=(j['geometry_A'][0]-origin['geometry_A'][0]).reshape(-1)
        gradient=F2@dx+.5*np.einsum('ijk,j,k->i',F3,dx,dx,optimize=True)+np.einsum('ijkl,j,k,l->i',F4,dx,dx,dx,optimize=True)/6
        curvature=F2+np.einsum('ijk,k->ij',F3,dx,optimize=True)+.5*np.einsum('ijkl,k,l->ij',F4,dx,dx,optimize=True)
        Db=j['first'][0,:,d:];Dbb=j['second'][0,:,d:,d:]
        return Db.T@curvature@Db+np.einsum('i,ijk->jk',gradient,Dbb)
    h=float(difference_steps[-1]);zero_Hb=bath_hessian(np.zeros(d));Hb1=np.zeros((n,n,d));Hb2=np.zeros((n,n,d,d))
    for i in range(d):
        plus=bath_hessian(np.eye(d)[i]*h);minus=bath_hessian(-np.eye(d)[i]*h)
        Hb1[:,:,i]=(plus-minus)/(2*h);Hb2[:,:,i,i]=(plus-2*zero_Hb+minus)/h**2
        for k in range(i):
            v=[bath_hessian(h*(si*np.eye(d)[i]+sk*np.eye(d)[k])) for si in (-1,1) for sk in (-1,1)]
            Hb2[:,:,i,k]=Hb2[:,:,k,i]=(v[0]-v[1]-v[2]+v[3])/(4*h**2)
    expected_Hb1=np.einsum('ijk,ka->ija',C[d:,d:,:],linear[:,:d])
    minimum_H=H[:d,:d]-H[:d,d:]@np.linalg.solve(H[d:,d:],H[d:,:d])
    cross=D.T@F2@origin['first'][0,:,d:]
    return dict(reference_H_full=H,reference_C_full=C,reference_H=minimum_H,reference_response_first=b1,reference_response_second=b2,
        reference_Hbb_first=expected_Hb1,reference_Hbb_second=Hb2,reference_Hbb_first_check=float(abs(Hb1-expected_Hb1).max()),minimum_fourth_cm=fourths[-1],
        cartesian_D=D,cartesian_DD=DD,eliminated_cross_residual=float(abs(cross).max(initial=0)),
        reference_Hessian_residual=float(abs(D.T@F2@D-minimum_H).max(initial=0)),
        fourth_step_change=None if len(fourths)<2 else float(abs(fourths[-1]-fourths[-2]).max()),field_modes_removed=False)
