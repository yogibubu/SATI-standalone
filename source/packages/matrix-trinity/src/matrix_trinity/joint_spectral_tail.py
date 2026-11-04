"""Joint variational spectra in the resident disjoint occupation cones.

Finite occupations retain their HDC energies. Joint tables replace forward
spectral gaps only. Positive constant edge spacings complete the table;
partition sums and occupation energies use exactly that same completion.
"""
from dataclasses import dataclass
import numpy as np
from .spectral_tail import spectral_matched_tail_factors,spectral_matched_energies


def _table(table):
    e=np.asarray(table,float)
    if (e.ndim!=2 or min(e.shape)<4 or not np.all(np.isfinite(e))
            or abs(e[0,0])>1e-8 or np.any(np.diff(e,axis=0)<=0) or np.any(np.diff(e,axis=1)<=0)):
        raise ValueError('finite joint spectrum with positive forward gaps and zero origin required')
    return e,np.array([np.diff(e,axis=0)[-1].min(),np.diff(e,axis=1)[:,-1].min()])


def joint_spectral_energies(table,occupations):
    e,gap=_table(table);n=np.asarray(occupations)
    if n.ndim!=2 or n.shape[1]!=2 or n.dtype.kind not in 'iu' or np.any(n<0):raise ValueError('two nonnegative integer occupations required')
    last=np.array(e.shape)-1;inside=np.minimum(n,last)
    return e[inside[:,0],inside[:,1]]+np.maximum(n-last,0)@gap


def pair_spectral_residual(raw, uncoupled):
    """Remove the local quadratic contribution shared with an acquired field.

    The six n_i+n_j<=2 anchors fix a unique occupation polynomial. Its
    subtraction retains only the mixed-spectrum contribution beyond that
    local model. Labels must already have passed spectral continuation.
    """
    raw=np.asarray(raw,float);one=np.asarray(uncoupled,float)
    if (raw.ndim!=2 or raw.shape[0]!=raw.shape[1] or len(raw)<4
            or one.shape!=(2,len(raw)) or not np.all(np.isfinite(raw))
            or not np.all(np.isfinite(one))):
        raise ValueError('finite square pair spectrum and two matching diagonal spectra required')
    a,b=one-one[:,:1];raw=raw-raw[0,0];delta=raw-a[:,None]-b[None,:]
    i,j=np.indices(raw.shape);c10=delta[1,0];c01=delta[0,1]
    c20=delta[2,0]-2*c10;c02=delta[0,2]-2*c01;c11=delta[1,1]-c10-c01
    local=c10*i+c01*j+c20*i*(i-1)/2+c02*j*(j-1)/2+c11*i*j
    return dict(residual_cm=delta-local,local_coupling_coefficients_cm=np.array([c10,c01,c20,c02,c11]))


def anchor_pair_spectrum(raw,uncoupled,anchored_diagonals,cross_cm):
    """Match qualified pair levels without counting the local coupling twice."""
    result=pair_spectral_residual(raw,uncoupled);L=np.asarray(anchored_diagonals,float)
    residual=result['residual_cm'];i,j=np.indices(residual.shape)
    if L.shape!=(2,len(residual)) or not np.all(np.isfinite(L)) or not np.isfinite(cross_cm):
        raise ValueError('matching finite diagonal anchors and mixed constant required')
    e=L[0,:,None]+L[1,None,:]+float(cross_cm)*i*j+residual
    _table(e)
    return dict(levels_cm=e,local_coupling_coefficients_cm=result['local_coupling_coefficients_cm'],common_local_quadratic_removed=True)


@dataclass(frozen=True)
class BoundedPairSpectralCorrection:
    """Bounded higher-order pair correction for any full occupation model.

    Add ``evaluate(states)`` to the existing coercive full-field energy;
    no diagonal or mixed coefficient is removed. Beyond a table edge the
    residual holds its edge value, not its slope. Consequently the correction
    is bounded globally and preserves convergence of a coercive base model.
    Table-size sensitivity must be qualified on the resulting observable.
    """
    modes: tuple[int,int]
    residual_cm: np.ndarray

    def __post_init__(self):
        modes=tuple(self.modes);r=np.asarray(self.residual_cm,float)
        if (len(modes)!=2 or len(set(modes))!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i<0 for i in modes)
                or r.ndim!=2 or min(r.shape)<4 or not np.all(np.isfinite(r))):
            raise ValueError('distinct nonnegative modes and a finite pair residual table required')
        i,j=np.indices(r.shape)
        if np.any(abs(r[i+j<=2])>1e-8):raise ValueError('local quadratic anchors must remain unchanged')
        r=r.copy();r.setflags(write=False)
        object.__setattr__(self,'modes',modes);object.__setattr__(self,'residual_cm',r)

    @classmethod
    def from_spectrum(cls,modes,raw,uncoupled):
        return cls(tuple(modes),pair_spectral_residual(raw,uncoupled)['residual_cm'])

    def evaluate(self,states):
        n=np.asarray(states)
        if (n.ndim!=2 or n.shape[1]<=max(self.modes) or n.dtype.kind not in 'iu' or np.any(n<0)):
            raise ValueError('nonnegative integer occupations containing both pair modes required')
        q=np.minimum(n[:,self.modes],np.array(self.residual_cm.shape)-1)
        return self.residual_cm[q[:,0],q[:,1]]

    def minimum_linear_bound(self,positive_linear_coefficients):
        a=np.asarray(positive_linear_coefficients,float)
        if a.ndim!=1 or len(a)<=max(self.modes) or not np.all(np.isfinite(a)) or np.any(a<=0):
            raise ValueError('positive full-model linear minorant required')
        i,j=np.indices(self.residual_cm.shape)
        return float(np.min(a[self.modes[0]]*i+a[self.modes[1]]*j+self.residual_cm))


def _setup(model,matched,pair,table,replacement_modes):
    pair=tuple(pair);e,gap=_table(table);m=len(model.fundamentals_cm)
    if (len(pair)!=2 or len(set(pair))!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<m for i in pair)
            or (replacement_modes is not None and any(i not in replacement_modes for i in pair))
            or matched['maximum_explicit_quanta']>=min(e.shape)-1):
        raise ValueError('two replaced modes and a joint table beyond the finite domain required')
    return pair,e,gap


def joint_spectral_tail_factors(model,matched,diagonal_levels,*,inverse_temperatures,pair,pair_levels_cm,replacement_modes=None):
    pair,e,gap=_setup(model,matched,pair,pair_levels_cm,replacement_modes)
    beta=np.asarray(inverse_temperatures,float);L=np.asarray(diagonal_levels,float)
    result=spectral_matched_tail_factors(model,matched,L,inverse_temperatures=beta,replacement_modes=replacement_modes)
    off=model.x_matrix_cm-np.diag(np.diag(model.x_matrix_cm));products=[];fractions=[]
    for anchor,old in zip(matched['tail_anchor_states'],result['residual_products']):
        active=int(np.max(np.flatnonzero(anchor))) if np.any(anchor) else 0
        free=np.array([i>=active for i in pair]);a=anchor[list(pair)].astype(int);cross=(off@anchor)[list(pair)]
        other=cross-np.array([off[pair[0],pair[1]]*a[1],off[pair[1],pair[0]]*a[0]])
        oldz=np.ones(len(beta))
        for k in np.flatnonzero(free):
            mode=pair[k];n=np.arange(a[k],L.shape[1]);v=L[mode,a[k]:]-L[mode,a[k]]+cross[k]*(n-a[k]);dg=v[-1]-v[-2]
            oldz*=np.exp(-beta[:,None]*v).sum(axis=1)+np.exp(-beta*(v[-1]+dg))/(-np.expm1(-beta*dg))
        if not free.any():products.append(old);fractions.append(np.zeros(len(beta)));continue
        if np.any((gap+other)[free]<=0):raise ValueError('conditional joint completion gap is nonpositive')
        axes=[np.arange(a[k],e.shape[k]) if free[k] else np.array([a[k]]) for k in range(2)]
        ii,jj=np.meshgrid(*axes,indexing='ij');v=e[ii,jj]-e[tuple(a)]+other[0]*(ii-a[0])+other[1]*(jj-a[1])
        for k in np.flatnonzero(free):
            if np.any(np.diff(v,axis=k)<=0):raise ValueError('conditional joint table has a nonpositive gap')
        z=np.exp(-beta[:,None]*v.ravel()).sum(axis=1);tail=np.zeros(len(beta))
        g=np.zeros((2,len(beta)))
        for k in np.flatnonzero(free):g[k]=1/np.expm1(beta*(gap[k]+other[k]))
        if free[0]:tail+=np.exp(-beta[:,None]*v[-1,:]).sum(axis=1)*g[0]
        if free[1]:tail+=np.exp(-beta[:,None]*v[:,-1]).sum(axis=1)*g[1]
        if free.all():tail+=np.exp(-beta*v[-1,-1])*g[0]*g[1]
        products.append((old+1)*(z+tail)/oldz-1);fractions.append(tail/(z+tail))
    return dict(result,residual_products=np.asarray(products),joint_pair=pair,joint_table_completion_fraction=np.asarray(fractions),
                joint_boundary_spacings_cm=gap,joint_completion='positive constant minimum boundary gaps; same energy and partition definition')


def joint_matched_energies(model,occupations,matched,diagonal_levels,*,pair,pair_levels_cm,replacement_modes=None):
    pair,e,gap=_setup(model,matched,pair,pair_levels_cm,replacement_modes)
    n=np.asarray(occupations);L=np.asarray(diagonal_levels,float)
    values=spectral_matched_energies(model,n,matched,L,replacement_modes=replacement_modes)
    off=model.x_matrix_cm-np.diag(np.diag(model.x_matrix_cm));cutoff=matched['maximum_explicit_quanta'];last=L.shape[1]-1
    checked=set()
    for row in np.flatnonzero(n.sum(axis=1)>cutoff):
        anchor=np.zeros(n.shape[1],int);remaining=cutoff
        for i in range(n.shape[1]):anchor[i]=min(int(n[row,i]),remaining);remaining-=anchor[i]
        delta=n[row]-anchor;cross=off@anchor;p=np.array(pair);a=anchor[p];extra=delta[p]
        old=(L[p,np.minimum(n[row,p],last)]+np.maximum(n[row,p]-last,0)*(L[p,-1]-L[p,-2])-L[p,a]+cross[p]*extra).sum()
        other=cross[p]-np.array([off[p[0],p[1]]*a[1],off[p[1],p[0]]*a[0]])
        if tuple(anchor) not in checked:
            active=int(np.max(np.flatnonzero(anchor))) if np.any(anchor) else 0
            axes=[slice(int(a[k]),None) if pair[k]>=active else slice(int(a[k]),int(a[k])+1) for k in range(2)]
            conditional=e[tuple(axes)]
            for k in range(2):
                if pair[k]>=active and (gap[k]+other[k]<=0 or np.any(np.diff(conditional,axis=k)+other[k]<=0)):
                    raise ValueError('conditional joint table has a nonpositive gap')
            checked.add(tuple(anchor))
        new=joint_spectral_energies(e,n[row,p][None])[0]-e[tuple(a)]+other@extra
        values[row]+=new-old
    return values
