"""Generic Taylor properties and bilinear responses in the HDC unitary model."""
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb, factorial, prod
import numpy as np
from scipy.sparse import coo_matrix

from .curvilinear_vpt2 import _canonical_operator_key, _compiled_operator_couplings
from .hdc_hamiltonian import sam_polynomial_operator, hdc_transition_spectrum


def _positive(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
        raise ValueError(name+' must be a positive integer')


@dataclass(frozen=True)
class HDCPropertyExpansion:
    """Consecutive raw derivatives D0...Dp, component axes before modal axes.

    Dn has shape component_shape + (mode_count,)*n. Derivatives refer to real
    dimensionless initial-surface normal coordinates. Missing orders must be
    supplied explicitly as zeros; symmetry in derivative axes is required.
    Units describe the observable in these dimensionless coordinates. Complex
    transition-property derivatives are permitted.
    """
    derivatives: tuple
    mode_count: int
    coordinate_id: str
    units: str
    name: str = 'property'

    def __post_init__(self):
        _positive(self.mode_count,'mode_count')
        for name in ('coordinate_id','units','name'):
            if not isinstance(getattr(self,name),str) or not getattr(self,name).strip():
                raise ValueError(name+' must be a nonempty string')
        arrays=tuple(np.array(a,complex,copy=True) for a in self.derivatives)
        if not arrays:raise ValueError('D0 through the declared maximum Taylor order required')
        shape=arrays[0].shape
        if any(v==0 for v in shape):raise ValueError('empty property component axes')
        for order,array in enumerate(arrays):
            if array.shape!=shape+(self.mode_count,)*order or not np.all(np.isfinite(array)):
                raise ValueError('finite derivatives with common component axes and one modal axis per order required')
            for axis in range(order-1):
                if not np.allclose(array,np.swapaxes(array,len(shape)+axis,len(shape)+axis+1),rtol=1e-10,atol=1e-12):
                    raise ValueError('Taylor derivative axes must be symmetric')
            array.setflags(write=False)
        object.__setattr__(self,'derivatives',arrays)

    @property
    def component_shape(self):return self.derivatives[0].shape

    def terms(self,*,maximum_terms=1000000):
        """Unique symmetric monomials with 1/(k1!...kn!), not 1/n! again."""
        _positive(maximum_terms,'maximum_terms')
        if sum(comb(self.mode_count+d-1,d) for d in range(len(self.derivatives)))>maximum_terms:
            raise MemoryError('Taylor monomial budget exceeded before enumeration')
        for order,derivative in enumerate(self.derivatives):
            for ids in combinations_with_replacement(range(self.mode_count),order):
                denominator=prod(factorial(ids.count(i)) for i in set(ids))
                coefficient=derivative[(Ellipsis,)+ids]/denominator
                if np.any(coefficient):
                    yield coefficient,tuple(('q',i) for i in ids)


def hdc_property_matrix(initial, expansion, *, final=None, overlap=None,
                        maximum_terms=1000000, maximum_dense_bytes=512*1024**2):
    """Build P_f O(Q_i) P_i before any premature intermediate projection.

    For different coordinate bases supply overlap(final_occupations,
    extended_initial_occupations). It returns the harmonic overlaps to ALL
    polynomial-generated intermediate states; the callback may use the native
    Duschinsky Gaussian/Hermite evaluator. The final projection is performed
    only after complete monomial action. Only dense output and a bounded
    overlap tile are allocated, with sparse cached ladder actions between.
    """
    if not isinstance(expansion,HDCPropertyExpansion):raise TypeError('HDCPropertyExpansion required')
    _positive(maximum_terms,'maximum_terms');_positive(maximum_dense_bytes,'maximum_dense_bytes')
    final=initial if final is None else final
    if len(initial.frequencies_cm)!=expansion.mode_count:
        raise ValueError('property mode count differs from initial coordinate basis')
    identity=getattr(initial,'basis_id','')
    if identity and identity!=expansion.coordinate_id:
        raise ValueError('property coordinate_id differs from initial HDC basis_id')
    ni,nf=len(initial.basis),len(final.basis)
    shape=expansion.component_shape;components=int(np.prod(shape)) if shape else 1
    output_bytes=16*ni*nf*components
    if output_bytes>maximum_dense_bytes//3:
        raise MemoryError('generic property output exceeds working-set budget')
    monomials=sum(comb(expansion.mode_count+d-1,d) for d in range(len(expansion.derivatives)))
    if 16*components*monomials>maximum_dense_bytes//3:
        raise MemoryError('Taylor coefficient working-set budget exceeded')
    terms=list(expansion.terms(maximum_terms=maximum_terms))
    if not terms:terms=[(np.zeros(shape,complex),())]
    shared=final is initial or (identity and identity==getattr(final,'basis_id',''))
    metadata=dict(schema='matrix.trinity.hdc_property_matrix.v1',name=expansion.name,
        units=expansion.units,coordinate_id=expansion.coordinate_id,
        coordinate_binding_checked=bool(identity),maximum_taylor_order=len(expansion.derivatives)-1,
        component_shape=list(shape),missing_orders_inferred=False,
        supplied_derivatives_only=True,full_monomial_action_before_projection=True,
        property_evaluation='full_finite_taylor_operator_with_unitary_hdc_states',
        strict_perturbative_intensity_truncation=False)
    if shared and overlap is None:
        # The existing kernel directly applies each complete monomial.
        if final is initial:
            matrix=sam_polynomial_operator(initial,terms)
            return dict(matrix=matrix,diagnostics=metadata)
        lookup={tuple(s):i for i,s in enumerate(final.basis)}
    elif not callable(overlap):
        raise ValueError('different coordinate bases require an extended-state overlap callback')
    else:lookup=None
    targets={};rows=[];cols=[];values=[]
    for col,state in enumerate(initial.basis):
        for coefficient,operators in terms:
            key=_canonical_operator_key(operators)
            for target,value in _compiled_operator_couplings(tuple(state),((key,1.),)).items():
                if value==0 or not np.any(coefficient):continue
                if len(rows)>=maximum_terms or (len(rows)+1)*(320+64*components)>maximum_dense_bytes//3:
                    raise MemoryError('property sparse-action budget exceeded')
                row=targets.setdefault(target,len(targets));rows.append(row);cols.append(col)
                values.append(np.asarray(coefficient).reshape(components)*value)
    if not rows:
        return dict(matrix=np.zeros((nf,ni)+shape,complex),diagnostics=metadata)
    # dict insertion order is also the sparse row order.
    states=np.array(list(targets),int);values=np.asarray(values)
    actions=[coo_matrix((values[:,c],(rows,cols)),shape=(len(states),ni)).tocsr() for c in range(components)]
    output=np.zeros((nf,ni,components),complex)
    if lookup is not None:
        for target,row in targets.items():
            final_row=lookup.get(target)
            if final_row is not None:
                for c,action in enumerate(actions):output[final_row,:,c]=action.getrow(row).toarray()[0]
    else:
        # Reserve room for sparse arrays, output and two dense overlap tiles.
        tile=max(1,(maximum_dense_bytes//6)//max(16*nf,1))
        if 16*nf>maximum_dense_bytes//6:raise MemoryError('one harmonic-overlap column exceeds memory budget')
        for start in range(0,len(states),tile):
            end=min(len(states),start+tile)
            block=np.asarray(overlap(final.basis,states[start:end]),complex)
            if block.shape!=(nf,end-start) or not np.all(np.isfinite(block)):
                raise ValueError('overlap callback returned a nonfinite or incorrectly shaped block')
            for c,action in enumerate(actions):output[:,:,c]+=block@action[start:end]
    metadata['extended_initial_support']=len(states)
    return dict(matrix=output.reshape((nf,ni)+shape),diagnostics=metadata)


def hdc_property_spectrum(initial, expansion, *, final=None, overlap=None,
                          initial_levels=(0,),final_levels=None,initial_populations=None,
                          maximum_terms=1000000,maximum_dense_bytes=512*1024**2):
    """Generic component amplitudes using the same HDC states and energies.

    Default strengths are sums of squared components, not universal optical
    intensities. Observable-specific metrics, interference, frequency factors
    and polarization/population conventions remain explicit inputs.
    """
    final=initial if final is None else final
    built=hdc_property_matrix(initial,expansion,final=final,overlap=overlap,
                             maximum_terms=maximum_terms,maximum_dense_bytes=maximum_dense_bytes)
    result=hdc_transition_spectrum(initial,final,built['matrix'],initial_levels=initial_levels,
        final_levels=final_levels,initial_populations=initial_populations)
    result['property']=built['diagnostics']
    return result


def hdc_property_bilinear(left, right=None, *, contraction=None):
    """Per-line A* W B; scalar/vector/tensor, complex and signed interference.

    Component axes are flattened only for the specified contraction. With
    right=None and W=identity this returns ordinary component-summed squared
    strengths. No taking an absolute square of an interference response.
    """
    a=np.asarray(left,complex);b=a if right is None else np.asarray(right,complex)
    if a.ndim<1 or b.ndim<1 or a.shape[0]!=b.shape[0] or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('finite amplitudes with matching line axes required')
    na=int(np.prod(a.shape[1:]));nb=int(np.prod(b.shape[1:]))
    if not na or not nb:raise ValueError('empty property component space')
    a=a.reshape(len(a),na);b=b.reshape(len(b),nb)
    if contraction is None:
        if na!=nb:raise ValueError('unequal component spaces require an explicit contraction')
        return np.sum(a.conj()*b,axis=1)
    else:metric=np.asarray(contraction,complex)
    if metric.shape!=(na,nb) or not np.all(np.isfinite(metric)):
        raise ValueError('finite contraction with left/right component axes required')
    return np.einsum('la,ab,lb->l',a.conj(),metric,b,optimize=True)


def hdc_harmonic_overlap(model, *, maximum_quanta, maximum_integrals=1000000,
                         batch_size=256):
    """Native compiled Duschinsky overlap callback for generic HDC properties.

    The packing bound must include property-generated occupations, not only
    the initial retained states. This evaluates overlaps without inserting the
    model's dipole/HT operator, which is instead supplied by the Taylor field.
    """
    from .vibronic_harmonic import harmonic_gaussian_generator
    from .vibronic_selection import PackedHermiteCache
    for value,name in ((maximum_quanta,'maximum_quanta'),
                       (maximum_integrals,'maximum_integrals'),(batch_size,'batch_size')):
        _positive(value,name)
    gaussian=harmonic_gaussian_generator(model)
    if gaussian['log_ground']<np.log(np.finfo(float).tiny)+4:
        raise FloatingPointError('harmonic overlap underflows; log-scaled recurrence required')
    n=len(model.initial_frequencies_cm)
    cache=PackedHermiteCache(gaussian,np.zeros((1,2*n)),maximum_integrals,maximum_quanta)
    def overlap(final_states,source_states):
        final_states=np.asarray(final_states);source_states=np.asarray(source_states)
        for states in (final_states,source_states):
            if states.ndim!=2 or states.shape[1]!=n or states.dtype.kind not in 'iu':
                raise ValueError('integer overlap occupations in the harmonic model dimension required')
        count=len(final_states)*len(source_states)
        output=np.empty(count)
        for start in range(0,count,batch_size):
            ids=np.arange(start,min(count,start+batch_size))
            # Native generator ordering is INITIAL followed by FINAL.
            states=np.concatenate((source_states[ids%len(source_states)],
                                   final_states[ids//len(source_states)]),axis=1)
            where,exhausted=cache.evaluate(states)
            if exhausted:raise MemoryError('HDC property harmonic overlap cache exhausted')
            output[start:start+len(ids)]=cache.values[where[:,0]]
        return output.reshape(len(final_states),len(source_states))
    return overlap
