"""Conditional internal normal modes after fixing geometric local LAMs."""
from dataclasses import dataclass
import numpy as np
from matrix_chem import BOHR_TO_ANGSTROM
from .harmonic import HESSIAN_EIGENVALUE_TO_CM,CM_PER_HARTREE


@dataclass(frozen=True)
class ConditionalInternalModes:
    frequencies_cm: np.ndarray
    coordinate_displacements: np.ndarray
    cartesian_displacements_angstrom: np.ndarray
    mass_weighted_modes: np.ndarray
    force_constants_hartree: np.ndarray
    conditional_metric: np.ndarray

    def realization(self,geometry):
        """Restrict one nonlinear internal inverse to these conditional modes.

        Inactive coordinates remain fixed at finite displacements. Both
        Cartesian derivative orders are transformed with the same matrix.
        """
        k=self.coordinate_displacements
        def point(q):
            q=np.asarray(q,float)
            if q.shape!=(k.shape[1],) or not np.all(np.isfinite(q)):
                raise ValueError('finite conditional mode displacement required')
            xyz,a,b=geometry(k@q)
            return xyz,a@k,np.einsum('iab,ac,bd->icd',b,k,k,optimize=True)
        return point


def conditional_internal_modes(gradient_hartree_per_angstrom,hessian_hartree_per_angstrom2,
                               first_angstrom,second_angstrom,masses,active_indices):
    """Include nonstationary coordinate curvature and the conditional metric.

    The caller's nonlinear realization holds all inactive local coordinates
    fixed. The metric is inverse(J_active.T M J_active), not a subblock of the
    unconstrained inverse metric. Signed eigenvalues remain signed.
    """
    g=np.asarray(gradient_hartree_per_angstrom,float);h=np.asarray(hessian_hartree_per_angstrom2,float)
    a=np.asarray(first_angstrom,float);b=np.asarray(second_angstrom,float);m=np.repeat(np.asarray(masses,float),3)
    ids=np.asarray(active_indices,int)
    if (a.ndim!=2 or a.shape[0]!=len(m) or g.shape!=m.shape or h.shape!=(len(m),len(m))
            or b.shape!=(len(m),a.shape[1],a.shape[1]) or ids.ndim!=1 or not len(ids)
            or len(set(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=a.shape[1])
            or np.any(m<=0) or not all(np.all(np.isfinite(x)) for x in (g,h,a,b,m))):
        raise ValueError('finite consistent local geometry jets, masses and distinct active coordinates required')
    j=a[:,ids];curvature=b[:,ids][:,:,ids]
    f=j.T@h@j+np.einsum('i,iab->ab',g,curvature)
    metric=np.linalg.inv(j.T@(m[:,None]*j))
    w,u=np.linalg.eigh(metric);root=(u*np.sqrt(w))@u.T
    eigen,v=np.linalg.eigh(root@((f+f.T)/2)@root)
    freq=np.sign(eigen)*np.sqrt(abs(eigen))*HESSIAN_EIGENVALUE_TO_CM*BOHR_TO_ANGSTROM
    if np.any(abs(freq)<1e-8):raise ValueError('zero conditional frequency; a larger local LAM sector is required')
    q=root@v*np.sqrt(HESSIAN_EIGENVALUE_TO_CM**2*BOHR_TO_ANGSTROM**2/(CM_PER_HARTREE*abs(freq)))[None,:]
    full=np.zeros((a.shape[1],len(ids)));full[ids]=q
    xyz=a@full;weighted=np.sqrt(m)[:,None]*xyz
    weighted/=np.linalg.norm(weighted,axis=0)
    return ConditionalInternalModes(freq,full,xyz,weighted,(f+f.T)/2,metric)


def conditional_metric_jet(jacobian,masses,dimension,*,step=.015,coordinate_dependent=True):
    """Shared intrinsic SAM G/G1/G2 and ordered Podolsky scalar.

    Geometry-only finite differences use the same established symmetric
    stencil as the existing molecular kinetic acquisitions.
    """
    from functools import lru_cache
    masses=np.asarray(masses,float);n=int(dimension)
    if n!=dimension or n<1 or not np.isfinite(step) or step<=0 or np.any(masses<=0):
        raise ValueError('positive metric dimension, masses and finite step required')
    roots=np.repeat(np.sqrt(masses),3)
    @lru_cache(None)
    def value(point):
        a=np.asarray(jacobian(np.asarray(point,float)),float)
        if a.shape!=(len(roots),n) or not np.all(np.isfinite(a)):
            raise ValueError('finite constrained Jacobian matching the SAM dimension required')
        j=roots[:,None]*a;g=np.linalg.inv(j.T@j);return (g+g.T)/2
    zero=np.zeros(n);g0=value(tuple(zero));g1=np.zeros((n,)*3);g2=np.zeros((n,)*4)
    if not coordinate_dependent:
        return g0*BOHR_TO_ANGSTROM**2,g1,g2,0.
    for i,ei in enumerate(np.eye(n)*step):
        plus,minus=value(tuple(ei)),value(tuple(-ei))
        g1[:,:,i]=(plus-minus)/(2*step);g2[:,:,i,i]=(plus-2*g0+minus)/step**2
        for j in range(i):
            ej=np.eye(n)[j]*step
            g2[:,:,i,j]=g2[:,:,j,i]=(value(tuple(ei+ej))-value(tuple(ei-ej))-value(tuple(-ei+ej))+value(tuple(-ei-ej)))/(4*step**2)
    cov=np.linalg.inv(g0);dl=-np.einsum('ab,bai->i',cov,g1);ddl=np.empty((n,n))
    for i in range(n):
        for j in range(n):ddl[i,j]=np.trace(cov@g1[:,:,j]@cov@g1[:,:,i])-np.trace(cov@g2[:,:,i,j])
    divergence=sum(g1[i,j,i]*dl[j]+g0[i,j]*ddl[i,j] for i in range(n) for j in range(n))
    scale=HESSIAN_EIGENVALUE_TO_CM**2*BOHR_TO_ANGSTROM**2/CM_PER_HARTREE
    scalar=scale*(divergence/8+dl@g0@dl/32)
    return g0*BOHR_TO_ANGSTROM**2,g1*BOHR_TO_ANGSTROM**2,g2*BOHR_TO_ANGSTROM**2,float(scalar)


def acquire_conditional_internal_field(cartesian_jet,geometry,dimension,masses,*,steps=(.04,.06),metric_step=.015,metric_jet=None):
    """Acquire one local SAM field on a caller's fixed-LAM realization.

    Cartesian callbacks return (E, gradient, Hessian) in hartree and angstrom.
    The existing tiered owner supplies full cubics and paired quartics; missing
    four-distinct quartics remain an explicit model closure. No normal-mode
    selection or electronic sampling policy is introduced here.
    An optional metric_jet may be reused only when the caller binds its full
    nonlinear coordinate definition, geometry and signed frame to the source.
    The zero-displacement metric is checked again independently here.
    """
    from .tiered_anharmonic import acquire_tiered_anharmonic
    n=int(dimension);h1,h2=map(float,steps)
    if n<1 or not 0<h1<h2:raise ValueError('positive dimension and two ordered derivative steps required')
    packets={}
    def packet(q):
        key=tuple(q)
        if key not in packets:
            x,a,b=geometry(np.asarray(q,float));e,g,h=cartesian_jet(x)
            f=a.T@h@a+np.einsum('i,iab->ab',g,b)
            packets[key]=(float(e)*CM_PER_HARTREE,a.T@g*CM_PER_HARTREE,(f+f.T)/2*CM_PER_HARTREE)
        return packets[key]
    scans=[acquire_tiered_anharmonic(lambda q:packet(q)[0],lambda q:packet(q)[1],lambda q:packet(q)[2],
        np.zeros(n),reaction_mode=0,primary_modes=range(1,n),steps=h) for h in (h1,h2)]
    c,q=[(h2*h2*scans[0][key]-h1*h1*scans[1][key])/(h2*h2-h1*h1)/CM_PER_HARTREE
        for key in ('cubic','quartic')]
    indices=np.indices((n,)*4)
    four_distinct=np.ones((n,)*4,dtype=bool)
    for i in range(4):
        for j in range(i):four_distinct&=indices[i]!=indices[j]
    if (not np.all(np.isfinite(c)) or not np.all(np.isfinite(q[~four_distinct]))
            or np.any(np.isinf(q))):
        raise ValueError('nonfinite observed conditional cubic or paired-quartic derivative')
    q=np.where(four_distinct & np.isnan(q),0.,q)
    if metric_jet is None:
        metric=conditional_metric_jet(lambda z:geometry(z)[1],masses,n,step=metric_step)
    else:
        if len(metric_jet)!=4:raise ValueError('four source-bound kinetic metric entries required')
        metric=tuple(np.asarray(x,float) for x in metric_jet)
        if any(x.shape!=(n,)*degree for x,degree in zip(metric[:3],(2,3,4))) or metric[3].shape!=():
            raise ValueError('cached kinetic metric dimensions differ')
        if any(not np.all(np.isfinite(x)) for x in metric):raise ValueError('finite cached kinetic metric required')
        first=geometry(np.zeros(n))[1];weights=np.repeat(np.asarray(masses,float),3)
        actual=np.linalg.inv(first.T@(weights[:,None]*first))*BOHR_TO_ANGSTROM**2
        if not np.allclose(metric[0],actual,rtol=1e-10,atol=1e-12):
            raise ValueError('cached kinetic metric has a different coordinate frame')
        if not np.allclose(metric[0],metric[0].T,rtol=1e-12,atol=1e-12) or np.linalg.eigvalsh(metric[0]).min()<=0:
            raise ValueError('positive symmetric cached kinetic metric required')
    return dict(hessian=scans[0]['reference_hessian']/CM_PER_HARTREE,cubic=c,quartic=q,
        **dict(zip(('G','G1','G2','scalar'),metric)),electronic_hartree=packet(np.zeros(n))[0]/CM_PER_HARTREE,
        derivative_steps=np.array([h1,h2]),four_distinct_quartic_model_zero=True)
