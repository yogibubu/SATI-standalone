"""ZAFF4: general native third/fourth response with four-primitive products.

The reference two-jet and full nonbonded field are frozen. Local construction
never implies a physical/global certificate or a measured complete quartic.
"""
from pathlib import Path
import hashlib,json,socket,resource
import numpy as np
from matrix_chem import BOHR_TO_ANGSTROM
from matrix_gf.harmonic import CM_PER_HARTREE
from .native_pic_field import NativePicField,load_native_pic_field,save_native_pic_field
from .native_modal_derivatives import native_factor_modal_two_jet,complete_native_quartic_catalog,fit_native_modal_derivatives
from .native_refinement import native_refinement_catalog


def baseline_modal_derivatives(runtime,reference,displacements,cubic_keys,quartic_keys,*,hessian_cache=None):
    """Differentiate actual native Hessians; never evaluate an off-reference QFF."""
    xyz=np.asarray(reference,float);dx=np.asarray(displacements,float);flat=dx.reshape(len(dx),-1)
    if hessian_cache is None:
        central=flat@(runtime(xyz).hessian_hartree_per_bohr2/BOHR_TO_ANGSTROM**2)@flat.T*CM_PER_HARTREE
        points=np.empty((len(dx),2,2,len(dx),len(dx)))
        for axis in range(len(dx)):
            for j,h in enumerate((.01,.005)):
                for k,sign in enumerate((1,-1)):
                    hessian=runtime(xyz+sign*h*dx[axis]).hessian_hartree_per_bohr2
                    points[axis,j,k]=flat@(hessian/BOHR_TO_ANGSTROM**2)@flat.T*CM_PER_HARTREE
    else:central,points=hessian_cache
    if central.shape!=(len(dx),)*2 or points.shape!=(len(dx),2,2,len(dx),len(dx)) or not np.all(np.isfinite(points)):raise ValueError('invalid native Hessian cache')
    result=[];errors=[];mixed={}
    for degree,keys in [(3,cubic_keys),(4,quartic_keys)]:
        values={}
        for key in keys:
            if degree==3:axis=key[-1];left,right=key[:2]
            else:
                repeated=[i for i in key if key.count(i)>=2]
                if not repeated:
                    estimates=[]
                    for h in (.01,.005):
                        value=0.
                        for s in (1,-1):
                            for t in (1,-1):
                                cache_key=(key[0],key[1],h,s,t)
                                if cache_key not in mixed:
                                    hessian=runtime(xyz+h*s*dx[key[0]]+h*t*dx[key[1]]).hessian_hartree_per_bohr2
                                    mixed[cache_key]=flat@(hessian/BOHR_TO_ANGSTROM**2)@flat.T*CM_PER_HARTREE
                                value+=s*t*mixed[cache_key][key[2],key[3]]/(4*h*h)
                        estimates.append(value)
                    values[key]=(4*estimates[1]-estimates[0])/3
                    errors.append(abs(estimates[1]-estimates[0])/3)
                    continue
                axis=repeated[0];rest=list(key);rest.remove(axis);rest.remove(axis);left,right=rest
            estimates=[]
            for j,h in enumerate((.01,.005)):
                plus,minus=points[axis,j,:,left,right]
                estimates.append((plus-minus)/(2*h) if degree==3 else (plus-2*central[left,right]+minus)/h**2)
            values[key]=(4*estimates[1]-estimates[0])/3;errors.append(abs(estimates[1]-estimates[0])/3)
        result.append(values)
    return *result,dict(baseline_extrapolation_change_cm=float(max(errors)),hessian_cache_used=hessian_cache is not None,mixed_native_hessians_evaluated=len(mixed))


def build_native_zaff4_field(parent,reference,displacements,observed_cubic,observed_quartic,*,baseline=None,cubic_tolerance_cm=.01,quartic_tolerance_cm=.05,maximum_matrix_bytes=16*1024**3):
    """Construct a ZAFF4 candidate using measured sparse derivative maps.

    Four-primitive support completes observable response; it does not invent
    unavailable four-distinct modal derivatives. Rank resolution and a native
    minimum-norm coefficient convention are explicitly reported.
    """
    charge_source=str(getattr(parent.nonbonded_evaluator,'charge_source',''))
    if 'APOC' not in charge_source or 'CM5' not in charge_source:raise ValueError('ZAFF4 construction requires complete APOC CM5 charges')
    if min(cubic_tolerance_cm,quartic_tolerance_cm)<=0:raise ValueError('positive derivative tolerances required')
    if baseline is None:baseline3,baseline4,numeric=baseline_modal_derivatives(parent,reference,displacements,observed_cubic,observed_quartic)
    else:baseline3,baseline4,numeric=baseline
    first,second=native_factor_modal_two_jet(parent.primitives,reference,displacements)
    factors=native_refinement_catalog(parent.primitives,maximum_active_primitives=3)
    factors,completion=complete_native_quartic_catalog(parent.primitives,first,factors,complete_cubic=True)
    corrections,report=fit_native_modal_derivatives(first,second,factors,observed_cubic,observed_quartic,baseline3,baseline4,maximum_matrix_bytes=maximum_matrix_bytes)
    passed=report['3']['maximum_residual_cm']<=cubic_tolerance_cm and report['4']['maximum_residual_cm']<=quartic_tolerance_cm
    report.update(schema='matrix.architect.zaff4_local_fit.v1',model_level='ZAFF4',extends_support='ZAFF3 plus rank-complete native cubic/quartic products',completion=completion,baseline_numerics=numeric,local_derivatives_passed=bool(passed),global_physical_validation='pending independent validation',full_quartic_tensor_measured=len(observed_quartic)==__import__('math').comb(len(displacements)+3,4))
    metadata=dict(getattr(parent,'metadata',{}));metadata.update(reference_coordinates_angstrom=np.asarray(reference).tolist(),model_level='ZAFF4',construction_status='LOCAL_DERIVATIVES_MATCHED' if passed else 'REJECTED_LOCAL_FIT',local_fit=report)
    candidate=NativePicField(parent.primitives,parent.field,parent.nonbonded_evaluator,corrections=parent.corrections+corrections,energy_offset_hartree=parent.energy_offset_hartree,metadata=metadata)
    before=parent(reference);after=candidate(reference)
    change=max(abs(after.energy_hartree-before.energy_hartree),np.max(abs(after.gradient_hartree_per_bohr-before.gradient_hartree_per_bohr)),np.max(abs(after.hessian_hartree_per_bohr2-before.hessian_hartree_per_bohr2)))
    report['reference_jet_change_max_atomic_units']=float(change)
    report['local_derivatives_passed']=bool(passed and change<1e-10)
    if change>=1e-10:metadata['construction_status']='REJECTED_REFERENCE_JET'
    candidate.metadata.update(metadata)
    return candidate,report


def run_zaff4_native_protocol(request_path):
    """KEYMAKER/ARCHITECT owner action for a portable, remote ZAFF4 request."""
    path=Path(request_path).resolve();request=json.loads(path.read_text());base=path.parent
    from matrix_core.exchange_contracts import validate_request_budget
    validate_request_budget(request)
    required={'schema','required_host','memory_gib','parent_field','observations','output_directory'}
    if set(request)-required-{'workers','baseline_cache'} or not required<=set(request) or request['schema']!='matrix.architect.zaff4_native_request.v1':raise ValueError('unsupported or incomplete ZAFF4 request')
    if socket.gethostname().split('.')[0]!=request['required_host'].split('.')[0]:raise RuntimeError('ZAFF4 request requires its declared remote host')
    if request['memory_gib']!=128:raise ValueError('ZAFF4 requires 128 GiB per process')
    resource.setrlimit(resource.RLIMIT_AS,(128*1024**3,)*2)
    field_path=(base/request['parent_field']).resolve();parent=load_native_pic_field(field_path)
    charge_source=str(getattr(parent.nonbonded_evaluator,'charge_source',''))
    if 'APOC' not in charge_source or 'CM5' not in charge_source:raise ValueError('ZAFF requires the complete KEYMAKER/APOC CM5 baseline; estimated charges are not admissible')
    field_hash=hashlib.sha256(field_path.read_bytes()).hexdigest()
    observations_path=(base/request['observations']).resolve()
    with np.load(observations_path,allow_pickle=False) as data:
        reference=data['reference_angstrom'];dx=data['displacements_angstrom'];meta=json.loads(str(data['metadata_json']))
        observations=[]
        for degree in (3,4):
            keys=data[f'indices_{degree}'];values=data[f'values_{degree}_cm']
            if keys.shape!=(len(values),degree) or keys.dtype.kind not in 'iu' or np.any(keys<0) or np.any(keys>=len(dx)) or not np.all(np.isfinite(values)):raise ValueError('invalid measured derivative entries')
            table={tuple(int(i) for i in key):float(v) for key,v in zip(keys,values)}
            if len(table)!=len(values):raise ValueError('duplicate derivative observation')
            observations.append(table)
    if meta.get('schema')!='matrix.architect.sparse_modal_derivative_observations.v1' or not meta.get('source_sha256') or meta.get('unprinted_as_zero') is not False:raise ValueError('measured derivative provenance is incomplete')
    target=(base/request['output_directory']).resolve();target.mkdir(parents=True,exist_ok=True)
    if any(target.iterdir()):raise FileExistsError('ZAFF4 output must be a fresh directory')
    baseline=None
    if request.get('baseline_cache'):
        with np.load(base/request['baseline_cache'],allow_pickle=False) as cache:
            if str(cache['parent_sha256'])!=field_hash or not np.array_equal(cache['displacements_angstrom'],dx):raise ValueError('native baseline cache belongs to a different field/frame')
            baseline=baseline_modal_derivatives(parent,reference,dx,*observations,hessian_cache=(cache['central'],cache['points']))
    from matrix_core import resolve_keymaker_protocol,ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID
    protocol=resolve_keymaker_protocol('zaff4-native-coupling-fit.v1')
    from threadpoolctl import threadpool_limits
    workers=request.get('workers',4)
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=128:raise ValueError('invalid worker budget')
    with threadpool_limits(limits=workers):candidate,report=build_native_zaff4_field(parent,reference,dx,*observations,baseline=baseline)
    report.update(protocol_id=protocol.protocol_id,protocol_sha256=protocol.sha256,hierarchy=ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID,request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent_sha256=field_hash,observations_sha256=hashlib.sha256(observations_path.read_bytes()).hexdigest(),execution_host=socket.gethostname(),memory_gib=128)
    if report['local_derivatives_passed']:save_native_pic_field(target/'zaff4-local-candidate.json',candidate)
    from matrix_core.exchange_contracts import write_exchange_json
    write_exchange_json(target/'result.json', report)
    return report
