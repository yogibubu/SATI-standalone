"""Native high-order response and conditioned minimum-norm observable fits."""
from itertools import combinations_with_replacement
from pathlib import Path
import json
import numpy as np
from scipy.linalg.lapack import get_lapack_funcs
from .asymptotic_pic import AsymptoticPicFactor


def extend_native_nonlocal_support(primitives, independent_indices, existing, *, degrees=(5, 6), excited_indices=None):
    """Add native axis powers decorated by a linear/quadratic response.

    Every added product starts at degree >=5. Thus its measured cubic/quartic
    reference jets vanish algebraically, without adding derivative observations.
    Products use existing inverse-distance/cosine/sine factors and the resident
    evaluator's complete defining-bond switches. No new radial function is used.
    """
    axes=tuple(map(int,independent_indices));degrees=tuple(degrees)
    if not axes or len(set(axes))!=len(axes) or any(not 0<=i<len(primitives) for i in axes):raise ValueError('invalid independent native axes')
    if not degrees or any(isinstance(i,bool) or not isinstance(i,int) or not 5<=i<=10 for i in degrees):raise ValueError('nonlocal native degrees must be integers between five and ten')
    def factor(index,power):
        kind=primitives[index].kind
        return AsymptoticPicFactor(index,'inverse_distance' if kind=='bond' else 'native' if kind=='angle' else 'sin',power)
    terms=list(existing)
    decorations=[()]+[(i,) for i in axes]+list(combinations_with_replacement(axes,2))
    excited=axes if excited_indices is None else tuple(int(i) for i in excited_indices)
    if not set(excited)<=set(axes):raise ValueError("excited axes must belong to the independent native view")
    for axis in excited:
        for degree in degrees:
            for decoration in decorations:
                powers={axis:degree}
                for index in decoration:powers[index]=powers.get(index,0)+1
                terms.append(tuple(factor(i,p) for i,p in sorted(powers.items())))
    terms=tuple(dict.fromkeys(terms))
    return terms,dict(excited_primitive_indices=list(excited),degrees=list(degrees),added_terms=len(terms)-len(existing),maximum_distinct_primitives_added=3,local_derivative_observations_added=0,reference_four_jet_preserved=True)


def equilibrated_minimum_norm_fit(design,target,*,relative_rank_cutoff=1e-11):
    """Use the complete orthogonal-factorization solution, not pivot columns.

    Scaling equilibrates observable column norms. Minimum norm is explicitly in
    these normalized contribution coordinates; no uniqueness of an unobserved
    PES is claimed. Zero-response columns are excluded and reported, not called
    physically negligible. LAPACK's complete minimum-norm answer is retained.
    """
    a=np.asarray(design,float);y=np.asarray(target,float)
    if a.ndim!=2 or y.shape!=(len(a),) or not np.all(np.isfinite(y)):raise ValueError('invalid observable system')
    scales=np.sqrt(np.einsum('ij,ij->j',a,a,optimize=True))
    if not np.all(np.isfinite(scales)):raise ValueError('nonfinite observable design')
    active=np.flatnonzero(scales>max(np.finfo(float).tiny,scales.max()*1e-13))
    if not len(active):raise ValueError('empty observable native support')
    matrix=np.empty((len(a),len(active)),order='F')
    for start in range(0,len(active),1024):
        columns=active[start:start+1024];matrix[:,start:start+len(columns)]=a[:,columns]/scales[columns]
    rows,columns=matrix.shape;rhs=np.zeros((max(rows,columns),1));rhs[:rows,0]=y
    gelsy,lwork=get_lapack_funcs(('gelsy','gelsy_lwork'),(matrix,))
    work,info=lwork(rows,columns,1,relative_rank_cutoff)
    if info:raise np.linalg.LinAlgError('native minimum-norm workspace query failed')
    _,answer,_,rank,info=gelsy(matrix,rhs,np.zeros(columns,np.int32),relative_rank_cutoff,int(work),overwrite_a=True,overwrite_b=True)
    if info:raise np.linalg.LinAlgError('equilibrated native minimum-norm fit failed')
    coefficients=np.zeros(a.shape[1]);coefficients[active]=answer[:columns,0]/scales[active]
    residual=a@coefficients-y
    return coefficients,dict(solver='column_equilibrated_complete_minimum_norm',relative_rank_cutoff=relative_rank_cutoff,resolved_rank=int(rank),observable_columns=int(len(active)),unresolved_response_columns=int(len(scales)-len(active)),relative_column_observability_floor=1e-13,maximum_normalized_residual=float(np.max(abs(residual))),normalized_coefficient_norm=float(np.linalg.norm(answer[:columns,0])),physical_field_qualified=False)


def save_native_observable_cache(directory,design,target,factors,parent,metadata):
    """Persist the expensive design so solver/pruning audits do not rebuild it."""
    from .native_pic_field import save_native_pic_field
    path=Path(directory);path.mkdir(parents=True,exist_ok=False)
    np.save(path/'design.npy',design,allow_pickle=False);np.save(path/'target.npy',target,allow_pickle=False)
    rows=[[[int(t.primitive_index),t.form,int(t.power),int(t.harmonic)] for t in term] for term in factors]
    (path/'factors.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    save_native_pic_field(path/'parent-native.json',parent)
    (path/'metadata.json').write_text(json.dumps(dict(schema='matrix.architect.native_observable_cache.v1',**metadata),indent=2,allow_nan=False)+'\n')
