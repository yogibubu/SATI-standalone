"""Observable-based removal of complete native coupling terms.

Columns must be complete native terms or symmetry-linked groups, including
all their dissociation switches. Pure diagonal/fixed terms can be protected.
Tolerances define significance in energy, force, Hessian or derivative units;
coefficient magnitude is never an acceptance criterion.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import lstsq
from .esalci_reconstruction import _independent_scaled_fit


@dataclass(frozen=True)
class NativeCouplingPruning:
    coefficients: np.ndarray
    retained_indices: tuple[int,...]
    removed_indices: tuple[int,...]
    report: dict


def prune_native_couplings(design,target,coefficients,*,row_tolerances,eligible=None,validation_design=None,validation_target=None,validation_tolerances=None,relative_rank_cutoff=1e-9,significance_threshold=.01,total_removal_budget=.25,rank_reduce=True):
    """Refit a reduced support and accept only if every supplied gate passes.

    A fixed-coefficient triangle bound controls the cumulative proposed cut.
    The subsequent refit is independently rechecked, including all held-out
    rows. Rank redundancy is removed only if those same controls survive.
    A rejected reduction returns the original coefficients. This is a finite
    observable-set certificate, not a uniform global PES or rate-error bound.
    """
    a=np.asarray(design,float);y=np.asarray(target,float);original=np.asarray(coefficients,float)
    if a.ndim!=2 or y.shape!=(a.shape[0],) or original.shape!=(a.shape[1],):raise ValueError('native pruning shapes differ')
    tolerances=np.broadcast_to(np.asarray(row_tolerances,float),y.shape)
    if not all(np.all(np.isfinite(v)) for v in (a,y,original,tolerances)) or np.any(tolerances<=0):raise ValueError('finite observations and positive row tolerances required')
    if not 0<relative_rank_cutoff<1 or not np.isfinite(significance_threshold) or significance_threshold<0 or not 0<=total_removal_budget<=1:raise ValueError('invalid pruning significance policy')
    eligible=np.ones(len(original),bool) if eligible is None else np.asarray(eligible,bool)
    if eligible.shape!=original.shape:raise ValueError('coupling eligibility mask differs')
    a=a/tolerances[:,None];y=y/tolerances
    validation_provided=validation_design is not None
    if validation_provided:
        v=np.asarray(validation_design,float);z=np.asarray(validation_target,float)
        vt=np.broadcast_to(np.asarray(validation_tolerances,float),z.shape)
        if v.ndim!=2 or v.shape!=(len(z),len(original)) or np.any(vt<=0) or not all(np.all(np.isfinite(x)) for x in (v,z,vt)):raise ValueError('incomplete independent pruning observations')
        v=v/vt[:,None];z=z/vt
    else:v=np.empty((0,len(original)));z=np.empty(0)
    def errors(c):return float(np.max(abs(a@c-y))),float(np.max(abs(v@c-z))) if len(z) else 0.
    observed=np.max(abs(a),axis=0)>1e-14
    if len(z):observed|=np.max(abs(v),axis=0)>1e-14
    unobserved=int(np.count_nonzero(eligible & ~observed))
    eligible=eligible & observed
    initial_errors=errors(original);current=original.copy();rank_reduction_accepted=False
    candidates=np.flatnonzero(eligible)
    fixed=np.flatnonzero(~eligible)
    rank=0
    if rank_reduce and len(candidates):
        values,rank,_=_independent_scaled_fit(a[:,candidates],y-a[:,fixed]@original[fixed],relative_rank_cutoff,'gelsy')
        reduced=original.copy();reduced[candidates]=values
        if max(errors(reduced))<=1:current=reduced;rank_reduction_accepted=True
    contribution=np.max(abs(a),axis=0)*abs(current)
    if len(z):contribution=np.maximum(contribution,np.max(abs(v),axis=0)*abs(current))
    proposed=[];bound=0.
    for index in np.argsort(contribution,kind='stable'):
        if not eligible[index] or current[index]==0:continue
        value=float(contribution[index])
        if value>significance_threshold:break
        if bound+value>total_removal_budget:break
        proposed.append(int(index));bound+=value
    cut=current.copy();cut[proposed]=0.
    active=np.flatnonzero(eligible & (cut!=0))
    if len(active):
        values,_,rank,_=lstsq(a[:,active],y-a[:,fixed]@original[fixed],cond=relative_rank_cutoff,lapack_driver='gelsy')
        cut[active]=values
    final_errors=errors(cut);accepted=max(initial_errors)<=1 and max(final_errors)<=1
    if not accepted:cut=original.copy()
    retained=tuple(int(i) for i in np.flatnonzero(cut!=0));removed=tuple(int(i) for i in np.flatnonzero(eligible & (original!=0) & (cut==0)))
    report=dict(candidate_terms=len(original),unobserved_couplings_protected=unobserved,eligible_couplings=int(eligible.sum()),retained_terms=len(retained),removed_couplings=len(removed),rank=int(rank),rank_reduction_accepted=rank_reduction_accepted and accepted,significance_threshold_tolerance_units=float(significance_threshold),fixed_coefficient_removal_bound_tolerance_units=bound,refitted_after_removal=True,initial_training_max_normalized_error=initial_errors[0],initial_validation_max_normalized_error=initial_errors[1],proposed_training_max_normalized_error=final_errors[0],proposed_validation_max_normalized_error=final_errors[1],reduction_accepted=accepted,independent_validation_provided=validation_provided,physical_field_qualified=False,scope='supplied finite observable set; no global or rate convergence claim')
    return NativeCouplingPruning(cut,retained,removed,report)


def run_native_coupling_pruning(request_path):
    """Apply observable designs to all mapped mixed terms of a native field."""
    from pathlib import Path
    from dataclasses import replace
    import json,hashlib,socket,resource
    from .native_pic_field import NativePicField,load_native_pic_field,save_native_pic_field
    from .asymptotic_pic import AsymptoticPicCorrection
    source=Path(request_path).resolve();request=json.loads(source.read_text());base=source.parent
    from matrix_core.exchange_contracts import validate_request_budget
    validate_request_budget(request)
    required={'schema','required_host','memory_gib','native_field','training_observables','output_directory'}
    allowed=required|{'validation_observables','significance_threshold','total_removal_budget','workers'}
    if not required<=set(request) or set(request)-allowed or request['schema']!='matrix.architect.native_coupling_pruning_request.v1':raise ValueError('invalid native coupling pruning request')
    if request['required_host'].split('.')[0]!=socket.gethostname().split('.')[0]:raise RuntimeError('pruning requires the declared remote host')
    if request['memory_gib']!=128:raise ValueError('128 GiB required for native coupling pruning')
    resource.setrlimit(resource.RLIMIT_AS,(128*1024**3,)*2)
    parent_path=(base/request['native_field']).resolve();parent=load_native_pic_field(parent_path)
    parent_hash=hashlib.sha256(parent_path.read_bytes()).hexdigest()
    with np.load(base/request['training_observables'],allow_pickle=False) as data:
        a=data['design'];y=data['target'];tolerances=data['row_tolerances'];eligible=data['eligible'];families=data['term_family'];indices=data['term_index'];expected_hash=str(data['native_field_sha256']);reference=data['reference_angstrom']
    if expected_hash!=parent_hash:raise ValueError('observable columns belong to another native field')
    mapping=list(zip(map(str,families),map(int,indices)))
    if len(mapping)!=len(set(mapping)) or len(mapping)!=a.shape[1]:raise ValueError('invalid native term map')
    current=[]
    for family,index in mapping:
        if family=='coupling':current.append(parent.field.retained_couplings[index].force_constant)
        elif family=='correction':current.append(parent.corrections[index].coefficient)
        else:raise ValueError('unsupported native term family')
    mandatory={('coupling',i) for i in range(len(parent.field.retained_couplings))}
    mandatory|={('correction',i) for i,c in enumerate(parent.corrections) if len({t.primitive_index for t in c.factors})>1}
    if not mandatory<=set(mapping):raise ValueError('the complete mixed-coupling catalog must be covered')
    if any(not eligible[i] for i,key in enumerate(mapping) if key in mandatory):raise ValueError('every mixed coupling must be eligible; observable gates retain necessary terms')
    validation={}
    if request.get('validation_observables'):
        with np.load(base/request['validation_observables'],allow_pickle=False) as data:
            if str(data['native_field_sha256'])!=parent_hash:raise ValueError('validation field differs')
            validation=dict(validation_design=data['design'],validation_target=data['target'],validation_tolerances=data['row_tolerances'])
    from threadpoolctl import threadpool_limits
    workers=request.get('workers',4)
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=128:raise ValueError('invalid worker budget')
    with threadpool_limits(limits=workers):reduction=prune_native_couplings(a,y,np.array(current),row_tolerances=tolerances,eligible=eligible,significance_threshold=request.get('significance_threshold',.01),total_removal_budget=request.get('total_removal_budget',.25),**validation)
    coefficients=dict(zip(mapping,reduction.coefficients));couplings=[]
    sparse=parent.field.sparse_typed_hessian.copy()
    for i,c in enumerate(parent.field.retained_couplings):
        value=coefficients.get(('coupling',i),c.force_constant);left,right=c.primitive_indices
        sparse[left,right]=sparse[right,left]=value
        if value!=0:couplings.append(replace(c,force_constant=float(value),normalized_magnitude=float(abs(value)/np.sqrt(parent.field.diagonal_force_constants[left]*parent.field.diagonal_force_constants[right]))))
    corrections=tuple(AsymptoticPicCorrection(float(coefficients.get(('correction',i),c.coefficient)),c.factors) for i,c in enumerate(parent.corrections) if coefficients.get(('correction',i),c.coefficient)!=0)
    field=replace(parent.field,retained_couplings=tuple(couplings),sparse_typed_hessian=sparse)
    metadata=dict(getattr(parent,'metadata',{}));metadata.pop('zaff4_certificate',None);metadata['coupling_pruning']=reduction.report
    child=NativePicField(parent.primitives,field,parent.nonbonded_evaluator,corrections=corrections,energy_offset_hartree=parent.energy_offset_hartree,metadata=metadata)
    before=parent(reference);after=child(reference)
    jet_error=float(max(abs(after.energy_hartree-before.energy_hartree),np.max(abs(after.gradient_hartree_per_bohr-before.gradient_hartree_per_bohr)),np.max(abs(after.hessian_hartree_per_bohr2-before.hessian_hartree_per_bohr2))))
    reduction.report['reference_jet_change_max_atomic_units']=jet_error
    reduction.report['reference_jet_preserved']=jet_error<1e-9
    reduction.report['reduction_accepted']=bool(reduction.report['reduction_accepted'] and jet_error<1e-9)
    target=(base/request['output_directory']).resolve();target.mkdir(parents=True,exist_ok=True)
    if any(target.iterdir()):raise FileExistsError('pruning output must be a fresh directory')
    if reduction.report['reduction_accepted']:save_native_pic_field(target/'native-pruned-candidate.json',child)
    from matrix_core import resolve_keymaker_protocol
    protocol=resolve_keymaker_protocol('zaff4-native-coupling-prune.v1')
    report=dict(schema='matrix.architect.native_coupling_pruning_result.v1',protocol_id=protocol.protocol_id,protocol_sha256=protocol.sha256,**reduction.report,native_field_sha256=parent_hash,removed_term_map=[mapping[i] for i in reduction.removed_indices],request_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    from matrix_core.exchange_contracts import write_exchange_json
    write_exchange_json(target/'result.json', report)
    np.savez_compressed(target/'coefficients.npz',coefficients=reduction.coefficients)
    return report
