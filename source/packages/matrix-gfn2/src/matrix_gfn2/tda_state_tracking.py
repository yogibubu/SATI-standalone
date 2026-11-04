"""Fixed-geometry TDA state/subspace continuity, including cross-basis metrics.

No Hamiltonian, eigenvector, state energy or derivative is modified. Density
and NTO diagnostics refine correspondences but never certify a unique root
inside an unresolved near-degenerate block.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .electronic_state import _readonly_array
from .imported_tda import ImportedTDAReference, ImportedTDAResult
from .tda_virtual_space import _reference_fingerprint
from .sf_tda import HARTREE_TO_EV


@dataclass(frozen=True)
class TDAStateTrackingPolicy:
    near_degeneracy_ev: float = .05
    minimum_overlap: float = .9
    minimum_margin: float = .1
    minimum_subspace_overlap: float = .9
    minimum_subspace_margin: float = .1

    def __post_init__(self):
        if not np.isfinite(self.near_degeneracy_ev) or self.near_degeneracy_ev < 0:
            raise ValueError('near_degeneracy_ev must be finite and nonnegative')
        for name in ('minimum_overlap','minimum_subspace_overlap'):
            if not np.isfinite(getattr(self,name)) or not 0 < getattr(self,name) <= 1:
                raise ValueError(f'{name} must lie in (0,1]')
        for name in ('minimum_margin','minimum_subspace_margin'):
            if not np.isfinite(getattr(self,name)) or not 0 <= getattr(self,name) <= 1:
                raise ValueError(f'{name} must lie in [0,1]')


@dataclass(frozen=True)
class TDAStateMatch:
    source_index: int
    candidate_index: int
    amplitude_overlap: float
    margin: float
    hole_density_overlap: float
    particle_density_overlap: float
    nto_spectrum_overlap: float
    individually_resolved: bool
    subspace_index: int


@dataclass(frozen=True)
class TDASubspaceMatch:
    source_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    principal_overlaps: np.ndarray
    margin: float
    closed: bool
    resolved: bool

    def __post_init__(self):
        object.__setattr__(self,'principal_overlaps',_readonly_array(self.principal_overlaps))


@dataclass(frozen=True)
class TDAStateTrackingResult:
    """All indices are zero-based. Candidate eigenvectors remain unrotated."""
    matches: tuple[TDAStateMatch, ...]
    subspaces: tuple[TDASubspaceMatch, ...]
    policy: TDAStateTrackingPolicy
    cross_basis: bool


def _rectangle(ref, result):
    if result.reference_model_id != ref.model_id or not result.converged:
        raise ValueError('tracking needs converged results on their supplied references')
    occ=np.flatnonzero(ref.occupations>1)
    vir=np.asarray(result.retained_virtual_orbitals)
    if result.virtual_space is None:
        if vir.ndim!=1 or not len(vir) or np.any(vir<0) or np.any(vir>=len(ref.occupations)) or len(np.unique(vir))!=len(vir) or np.any(ref.occupations[vir]>=1):
            raise ValueError('invalid canonical virtual labels')
        cv=ref.coefficients[:,vir]
    else:
        result.virtual_space.validate_reference(ref)
        if not np.array_equal(vir,np.arange(result.virtual_space.dimension)):
            raise ValueError('invalid projected virtual labels')
        cv=ref.coefficients[:,result.virtual_space.parent_virtual_orbitals]@result.virtual_space.rotation
    if not np.array_equal(result.occupied_orbitals,np.repeat(occ,len(vir))) or not np.array_equal(result.virtual_orbitals,np.tile(vir,len(occ))):
        raise ValueError('tracking requires complete occupied/virtual rectangles')
    x=np.asarray(result.amplitudes)
    if x.shape!=(len(result.excitation_energies_hartree),len(occ)*len(vir)) or not np.all(np.isfinite(x)) or not np.allclose(x@x.T,np.eye(len(x)),atol=1e-7,rtol=0):
        raise ValueError('tracking requires orthonormal finite state vectors')
    return x.reshape(len(x),len(occ),len(vir)),ref.coefficients[:,occ],cv


def _energy_clusters(energies,threshold):
    order=np.argsort(energies,kind='stable');groups=[]
    for i in order:
        if not groups or energies[i]-energies[groups[-1][0]]>threshold:
            groups.append([int(i)])
        else:groups[-1].append(int(i))
    return groups


def track_imported_tda_states(
    source_reference: ImportedTDAReference, source: ImportedTDAResult,
    candidate_reference: ImportedTDAReference, candidate: ImportedTDAResult, *,
    source_state_indices=None, cross_overlap_ao: np.ndarray | None = None,
    policy: TDAStateTrackingPolicy | None = None,
) -> TDAStateTrackingResult:
    """Compare energy clusters first, then diagnose individual correspondences.

    An explicit <source AO|candidate AO> metric is required across references.
    The amplitude assignment seeds cluster correspondence; principal angles
    certify a whole cluster before density/NTO scores refine its root mapping.
    Extra nearby candidate roots make a block non-closed, hence unresolved.
    Root diagnostics use normalized Hilbert--Schmidt hole/particle density
    overlaps and a Bhattacharyya overlap of sorted NTO weights. These do not
    override a failed amplitude/margin criterion for individual root identity.
    """
    policy=policy or TDAStateTrackingPolicy()
    if not isinstance(policy,TDAStateTrackingPolicy):raise TypeError('invalid tracking policy')
    x,co,cv=_rectangle(source_reference,source)
    y,do,dv=_rectangle(candidate_reference,candidate)
    same=_reference_fingerprint(source_reference)==_reference_fingerprint(candidate_reference)
    if cross_overlap_ao is None:
        if not same:raise ValueError('different references require an explicit cross_overlap_ao')
        metric=source_reference.overlap
    else:
        if np.iscomplexobj(cross_overlap_ao):raise ValueError('cross overlap must be real')
        metric=np.asarray(cross_overlap_ao,dtype=float)
        if metric.shape!=(co.shape[0],do.shape[0]) or not np.all(np.isfinite(metric)):
            raise ValueError('cross overlap must have source/candidate AO dimensions')
    oo=co.T@metric@do;vv=cv.T@metric@dv
    if max(np.linalg.norm(oo,2),np.linalg.norm(vv,2))>1+2e-6:
        raise ValueError('cross overlap is inconsistent with normalized orbital spaces')
    requested=np.arange(len(x)) if source_state_indices is None else np.asarray(source_state_indices)
    if requested.ndim!=1 or (requested.size and (requested.dtype.kind not in 'iu' or np.any(requested<0) or np.any(requested>=len(x)))) or len(np.unique(requested))!=len(requested):
        raise ValueError('source_state_indices must be unique zero-based integers')
    requested=requested.astype(int)
    mapped=np.einsum('ij,sjb,ab->sia',oo,y,vv,optimize=True)
    signed=x.reshape(len(x),-1)@mapped.reshape(len(y),-1).T
    scores=np.abs(signed)
    if np.max(scores,initial=0)>1+2e-6:raise ValueError('unphysical transition-density overlaps')
    ri,ci=linear_sum_assignment(-scores)
    assignment=np.full(len(x),-1,dtype=int);assignment[ri]=ci
    se=source.excitation_energies_hartree*HARTREE_TO_EV
    ce=candidate.excitation_energies_hartree*HARTREE_TO_EV
    # Bound the whole cluster width: chains of small adjacent gaps must not
    # turn a dense but nondegenerate spectral region into one giant manifold.
    # Merge source blocks when their corresponding candidate block joins,
    # provided the combined span remains near-degenerate in either spectrum.
    parent=list(range(len(x)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(group):
        if group:
            r=root(group[0])
            for i in group[1:]:parent[root(i)]=r
    for group in _energy_clusters(se,policy.near_degeneracy_ev):join(group)
    candidate_clusters=_energy_clusters(ce,policy.near_degeneracy_ev)
    for group in candidate_clusters:
        members=[int(i) for i in np.flatnonzero(np.isin(assignment,group))]
        representatives={root(i) for i in members}
        merged=[i for i in range(len(x)) if root(i) in representatives]
        mapped_indices=[int(assignment[i]) for i in merged if assignment[i]>=0]
        if merged and (np.ptp(se[merged])<=policy.near_degeneracy_ev+1e-12
                       or (mapped_indices and np.ptp(ce[mapped_indices])<=policy.near_degeneracy_ev+1e-12)):
            join(merged)
    groups={}
    for i in range(len(x)):groups.setdefault(root(i),[]).append(i)
    subspaces=[];matches={}
    # Density/NTO scores are evaluated only within corresponding blocks.
    hx=np.einsum('ria,rja->rij',x,x);hy=np.einsum('ria,rja->rij',y,y)
    xnorm=np.sqrt(np.einsum('rij,rji->r',hx,hx));ynorm=np.sqrt(np.einsum('rij,rji->r',hy,hy))
    mapped_h=np.einsum('ij,sjk,lk->sil',oo,hy,oo,optimize=True)
    nx=[np.linalg.svd(z,compute_uv=False)**2 for z in x]
    ny=[np.linalg.svd(z,compute_uv=False)**2 for z in y]
    def diagnostics(i,j):
        hole=float(np.sum(hx[i]*mapped_h[j].T)/(xnorm[i]*ynorm[j]))
        particle=float(np.sum((x[i]@vv@y[j].T)**2)/(xnorm[i]*ynorm[j]))
        n=min(len(nx[i]),len(ny[j]));nto=float(np.sqrt(nx[i][:n]*ny[j][:n]).sum())
        return tuple(float(np.clip(v,0,1)) for v in (hole,particle,nto))
    for group in groups.values():
        if not np.any(np.isin(requested,group)):continue
        candidates=sorted(int(assignment[i]) for i in group if assignment[i]>=0)
        closed=len(candidates)==len(group)
        for cluster in candidate_clusters:
            if set(cluster)&set(candidates) and not set(cluster)<=set(candidates):closed=False
        block=signed[np.ix_(group,candidates)]
        principal=np.linalg.svd(block,compute_uv=False) if candidates else np.array([])
        outside_c=np.setdiff1d(np.arange(len(y)),candidates);outside_s=np.setdiff1d(np.arange(len(x)),group)
        competitor=max(float(np.linalg.norm(signed[np.ix_(group,outside_c)],2)) if len(outside_c) else 0.,float(np.linalg.norm(signed[np.ix_(outside_s,candidates)],2)) if len(outside_s) and candidates else 0.)
        margin=(float(principal[-1]) if len(principal) else 0.)-competitor
        resolved=bool(closed and len(principal)==len(group) and principal[-1]>=policy.minimum_subspace_overlap and margin>=policy.minimum_subspace_margin)
        subindex=len(subspaces)
        subspaces.append(TDASubspaceMatch(tuple(group),tuple(candidates),principal,margin,closed,resolved))
        if resolved and len(group)>1:
            combined=np.zeros((len(group),len(candidates)))
            for ii,i in enumerate(group):
                for jj,j in enumerate(candidates):
                    h,p,n=diagnostics(i,j)
                    combined[ii,jj]=.5*scores[i,j]**2+.2*h+.2*p+.1*n
            # Preserve already unique amplitude matches; density diagnostics
            # only resolve the remaining internal correspondence choices.
            locked=[]
            for ii,i in enumerate(group):
                j=int(assignment[i]);jj=candidates.index(j)
                competitor=max(float(np.max(np.delete(scores[i],j),initial=0)),
                               float(np.max(np.delete(scores[:,j],i),initial=0)))
                if scores[i,j]>=policy.minimum_overlap and scores[i,j]-competitor>=policy.minimum_margin:
                    locked.append((ii,jj,float(combined[ii,jj])))
            for ii,jj,value in locked:
                combined[ii,:]=-100.;combined[:,jj]=-100.;combined[ii,jj]=value
            ar,ac=linear_sum_assignment(-combined)
            for i,j in zip(ar,ac):assignment[group[i]]=candidates[j]
        for i in group:
            j=int(assignment[i]);overlap=float(scores[i,j]) if j>=0 else 0.
            margin_i=overlap-max(float(np.max(np.delete(scores[i],j),initial=0)) if j>=0 else 0.,float(np.max(np.delete(scores[:,j],i),initial=0)) if j>=0 else 0.)
            h,p,n=diagnostics(i,j) if j>=0 else (0.,0.,0.)
            matches[i]=TDAStateMatch(i,j,overlap,margin_i,h,p,n,bool(j>=0 and overlap>=policy.minimum_overlap and margin_i>=policy.minimum_margin),subindex)
    return TDAStateTrackingResult(tuple(matches[int(i)] for i in requested),tuple(subspaces),policy,not same)
