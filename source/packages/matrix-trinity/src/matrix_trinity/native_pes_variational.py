"""Direct native-PES grids and qualified local states for vibronic overlaps.

ARCHITECT owns field construction. This module consumes actual native energies;
it never reconstructs a quartic and never adds a quadratic tail to the PES.
The kinetic model is an explicitly frozen linear Cartesian-modal subspace.
"""
from __future__ import annotations
import hashlib,json,socket
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from matrix_gf.harmonic import CM_PER_HARTREE,HESSIAN_EIGENVALUE_TO_CM
from matrix_chem import BOHR_TO_ANGSTROM
from .vibronic_local_states import solve_local_potential_ground_state

REQUEST_SCHEMA='matrix.trinity.native_pes_variational_request.v1'
GRID_SCHEMA='matrix.trinity.native_modal_grid.v1'
RESULT_SCHEMA='matrix.trinity.native_pes_variational_result.v1'


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    return value


def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_modal_displacements_angstrom(frequencies_cm,mass_weighted_modes,masses_amu):
    """Cartesian displacement per dimensionless q in an orthonormal MW frame."""
    w=np.asarray(frequencies_cm,float);m=np.asarray(masses_amu,float)
    modes=np.asarray(mass_weighted_modes,float)
    if (w.ndim!=1 or m.ndim!=1 or not len(w) or np.any(w<=0) or np.any(m<=0)
            or modes.shape!=(len(w),len(m),3)
            or not all(np.all(np.isfinite(a)) for a in (w,m,modes))):
        raise ValueError('positive frequencies/masses and modal rows (mode,atom,3) required')
    rows=modes.reshape(len(w),-1)
    if not np.allclose(rows@rows.T,np.eye(len(w)),atol=1e-8,rtol=1e-8):
        raise ValueError('orthonormal mass-weighted Cartesian modal rows required')
    return modes/np.sqrt(m)[None,:,None]*(BOHR_TO_ANGSTROM*HESSIAN_EIGENVALUE_TO_CM/np.sqrt(CM_PER_HARTREE*w))[:,None,None]


@dataclass(frozen=True)
class NativeCartesianModalPotential:
    reference_angstrom: np.ndarray
    displacements_angstrom: np.ndarray
    energy_evaluator: object
    reference_energy_hartree: float
    field_identifier: str
    batch_size: int=256

    def __post_init__(self):
        x=np.asarray(self.reference_angstrom,float);d=np.asarray(self.displacements_angstrom,float)
        if (x.ndim!=2 or x.shape[1]!=3 or d.ndim!=3 or d.shape[1:]!=x.shape
                or not len(d) or not all(np.all(np.isfinite(a)) for a in (x,d))
                or not callable(self.energy_evaluator) or not np.isfinite(self.reference_energy_hartree)
                or not self.field_identifier or isinstance(self.batch_size,bool)
                or not isinstance(self.batch_size,int) or self.batch_size<1):
            raise ValueError('finite Cartesian frame and identified native energy evaluator required')
        for name,a in [('reference_angstrom',x),('displacements_angstrom',d)]:
            a=a.copy();a.setflags(write=False);object.__setattr__(self,name,a)

    def __call__(self,coordinates):
        q=np.asarray(coordinates,float)
        if q.ndim!=2 or q.shape[1]!=len(self.displacements_angstrom) or not np.all(np.isfinite(q)):
            raise ValueError('finite (point,mode) dimensionless coordinates required')
        energies=np.empty(len(q))
        for start in range(0,len(q),self.batch_size):
            xyz=self.reference_angstrom+np.einsum('pm,mac->pac',q[start:start+self.batch_size],self.displacements_angstrom)
            for j,geometry in enumerate(xyz):
                energy=float(self.energy_evaluator(geometry))
                if not np.isfinite(energy):raise FloatingPointError('native PES returned nonfinite energy')
                energies[start+j]=(energy-self.reference_energy_hartree)*CM_PER_HARTREE
        return energies


def export_native_modal_grid(path, potential, frequencies_cm, *, half_width,
        grid_points, provenance, maximum_grid_points=2_000_000):
    """Sample a resident ARCHITECT/ZAFF callback once into a restartable grid.

    The native field certificate and electronic state must be supplied by the
    caller; this function does not promote an uncertified field. Off-reference
    energies come from the callback, including all native mixed terms.
    """
    destination=Path(path)
    if destination.exists():raise FileExistsError(destination)
    w=np.asarray(frequencies_cm,float)
    if (w.ndim!=1 or not len(w) or np.any(w<=0) or not np.all(np.isfinite(w))
            or not np.isfinite(half_width) or half_width<=0 or isinstance(grid_points,bool)
            or not isinstance(grid_points,int) or grid_points<3):raise ValueError('positive modal frequencies and box/grid required')
    n=len(w);size=grid_points**n
    if size>maximum_grid_points:raise MemoryError('native grid exceeds declared point budget')
    required={'field_identifier','electronic_state','native_functional_form','reference_geometry_sha256','coordinate_frame_sha256','field_certificate_sha256','full_native_potential','bare_quadratic_tail','polynomial_reexpansion'}
    if not required<=set(provenance):raise ValueError('incomplete native field provenance')
    if provenance['full_native_potential'] is not True or provenance['bare_quadratic_tail'] is not False or provenance['polynomial_reexpansion'] is not False:raise ValueError('a full native potential without a bare quadratic tail is required')
    axis=-half_width+2*half_width/(grid_points+1)*np.arange(1,grid_points+1)
    q=np.array(np.meshgrid(*([axis]*n),indexing='ij')).reshape(n,-1).T
    values=np.asarray(potential(q))
    if np.iscomplexobj(values) or values.shape!=(size,) or not np.all(np.isfinite(values)):raise ValueError('real finite native potential energies required')
    metadata=dict(schema=GRID_SCHEMA,coordinate_system='LINEAR_CARTESIAN_MODAL',kinetic_metric='CONSTANT_DIAGONAL',energy_unit='cm-1',half_width=float(half_width),grid_points=grid_points,provenance=dict(provenance))
    with destination.open('xb') as stream:
        np.savez_compressed(stream,frequencies_cm=w,potential_cm=values,metadata_json=np.array(json.dumps(_json(metadata),sort_keys=True)))
    return dict(path=str(destination),sha256=_hash(destination),point_count=size,**metadata)


def run_native_pes_variational_protocol(request_path):
    """Execute a strict, resumable KEYMAKER request on precomputed native grids.

    Every grid is an immutable direct evaluation of the same native field.
    Independent width, grid and HO controls are mandatory for acceptance.
    Input paths are resolved relative to the request, never the process cwd.
    """
    from matrix_core.exchange_contracts import validate_exchange_tree
    source=Path(request_path).resolve();request=json.loads(source.read_text());base=source.parent
    allowed={'schema','required_host','memory_gib','workers','cases','tolerances','output_directory','field_identifier','electronic_state','maximum_grid_points','maximum_ho_states','scope','native_field_certificate'}
    if set(request)-allowed or request.get('schema')!=REQUEST_SCHEMA:raise ValueError('unsupported native-PES request')
    required={'schema','required_host','memory_gib','cases','output_directory','field_identifier','electronic_state','scope','native_field_certificate'}
    if not required<=set(request):raise ValueError('incomplete native-PES request')
    host=str(request.get('required_host','')).strip()
    if not host or socket.gethostname().split('.')[0]!=host.split('.')[0]:raise RuntimeError('execution host differs from the explicitly required host')
    if request.get('scope')!='LOCAL_INITIAL_STATE':raise ValueError('v1 qualifies local initial states; final accepting-state spectra and rates are separate stages')
    if request.get('memory_gib')!=128:raise ValueError('v1 requires a 128 GiB process memory limit')
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(128*1024**3,)*2)
    certificate_path=(base/request['native_field_certificate']).resolve()
    certificate_hash=_hash(certificate_path)
    from matrix_core import resolve_keymaker_protocol
    protocol=resolve_keymaker_protocol('trinity-native-pes-variational.v1')
    if protocol.status!='approved':raise RuntimeError('native-PES protocol is not approved')
    workers=request.get('workers',4)
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=128:raise ValueError('explicit positive worker budget required')
    cases=request.get('cases',[])
    if len(cases)<4:raise ValueError('at least four independent domain/grid/projection controls required')
    tolerance=dict(request.get('tolerances',{}))
    defaults=dict(energy_cm=1e-3,wavefunction_loss=1e-8,outer_box_probability=1e-7,projection_loss=1e-6,grid_residual_cm=1e-5)
    if set(tolerance)-set(defaults):raise ValueError('unknown native-PES tolerance')
    defaults.update(tolerance);tolerance=defaults
    if any(not np.isfinite(v) or v<=0 for v in tolerance.values()):raise ValueError('positive finite acceptance tolerances required')
    target=(base/request['output_directory']).resolve();target.mkdir(parents=True,exist_ok=True)
    output=target/'result.json'
    if output.exists():
        saved=validate_exchange_tree(json.loads(output.read_text()), expected_schema=RESULT_SCHEMA)
        if saved.get('request_sha256')!=_hash(source):raise FileExistsError('output belongs to a different immutable request')
        if saved.get('field_certificate_sha256')!=certificate_hash or saved.get('protocol',{}).get('sha256')!=protocol.sha256:raise ValueError('completed certificate or protocol changed')
        if len(saved['cases'])!=len(cases):raise ValueError('incomplete saved case inventory')
        for c,case in zip(saved['cases'],request['cases']):
            if _hash(target/c['file'])!=c['sha256'] or _hash(base/case['grid'])!=c['grid_sha256']:raise ValueError('completed native-PES evidence changed')
        return saved
    marker=target/'request.sha256'
    if marker.exists() and marker.read_text().strip()!=_hash(source):raise FileExistsError('partial output belongs to another request')
    if not marker.exists() and any(target.iterdir()):raise FileExistsError('unidentified partial output directory')
    marker.write_text(_hash(source)+'\n')
    records=[];states=[];frames=[]
    for case in cases:
        if set(case)!={'grid','maximum_total_quanta'}:raise ValueError('each case requires only grid and maximum_total_quanta')
        path=(base/case['grid']).resolve()
        with np.load(path,allow_pickle=False) as d:
            w=d['frequencies_cm'];values=d['potential_cm'];meta=json.loads(str(d['metadata_json']))
        p=meta.get('provenance',{})
        if (meta.get('schema')!=GRID_SCHEMA or meta.get('coordinate_system')!='LINEAR_CARTESIAN_MODAL'
                or meta.get('kinetic_metric')!='CONSTANT_DIAGONAL' or meta.get('energy_unit')!='cm-1'
                or p.get('field_identifier')!=request['field_identifier'] or p.get('electronic_state')!=request['electronic_state']
                or p.get('full_native_potential') is not True or p.get('bare_quadratic_tail') is not False or p.get('polynomial_reexpansion') is not False
                or not p.get('native_functional_form') or p.get('field_certificate_sha256')!=certificate_hash):raise ValueError('native grid/state contract mismatch')
        frames.append((p.get('coordinate_frame_sha256'),p.get('reference_geometry_sha256'),p.get('field_certificate_sha256'),tuple(w)))
        if any(not value for value in frames[-1][:3]) or frames[-1]!=frames[0]:raise ValueError('controls do not share an immutable field and coordinate frame')
        width=meta['half_width'];points=meta['grid_points'];n=len(w)
        axis=-width+2*width/(points+1)*np.arange(1,points+1)
        def evaluate(q):
            indices=np.rint((q+width)*(points+1)/(2*width)-1).astype(int)
            if np.any(indices<0) or np.any(indices>=points) or not np.allclose(q,axis[indices],rtol=0,atol=1e-12):raise ValueError('solver requested an unacquired native grid point')
            return values.reshape((points,)*n)[tuple(indices.T)]
        state=solve_local_potential_ground_state(w,evaluate,half_width=width,grid_points=points,maximum_total_quanta=case['maximum_total_quanta'],maximum_grid_points=request.get('maximum_grid_points',2_000_000),maximum_ho_states=request.get('maximum_ho_states',8192),maximum_projection_loss=tolerance['projection_loss'],workers=workers,potential_identifier=request['field_identifier'])
        states.append(state);name=f'state-{len(records):03d}.npz';dest=target/name
        with dest.with_suffix('.partial').open('wb') as stream:np.savez_compressed(stream,basis=state.basis,energies_cm=state.energies_cm,eigenvectors=state.eigenvectors,frequencies_cm=state.frequencies_cm,diagnostics_json=np.array(json.dumps(_json(state.diagnostics))))
        dest.with_suffix('.partial').replace(dest)
        records.append(dict(file=name,sha256=_hash(dest),grid_sha256=_hash(path),half_width=width,grid_points=points,maximum_total_quanta=case['maximum_total_quanta'],diagnostics=_json(state.diagnostics)))
    # Compare normalized projections on their common HO basis. Each unnormalized
    # retained norm has already passed the independent projection-loss gate.
    reference=states[-1];lookup={tuple(row):i for i,row in enumerate(reference.basis)}
    comparisons=[];variation={'width':False,'grid':False,'projection':False}
    for i,(a,b) in enumerate(zip(records[:-1],records[1:])):
        changed=[name for name,key in [('width','half_width'),('grid','grid_points'),('projection','maximum_total_quanta')] if a[key]!=b[key]]
        if len(changed)==1:variation[changed[0]]=True
    for i,state in enumerate(states[:-1]):
        overlap=sum(np.conjugate(state.eigenvectors[j,0])*reference.eigenvectors[lookup[tuple(row)],0] for j,row in enumerate(state.basis) if tuple(row) in lookup)
        comparisons.append(dict(case=i,energy_change_cm=float(state.energies_cm[0]-reference.energies_cm[0]),squared_wavefunction_overlap=float(abs(overlap)**2)))
    passed=all(variation.values()) and all(abs(c['energy_change_cm'])<=tolerance['energy_cm'] and 1-c['squared_wavefunction_overlap']<=tolerance['wavefunction_loss'] for c in comparisons) and all(r['diagnostics']['outer_box_probability']<=tolerance['outer_box_probability'] and r['diagnostics']['grid_residual_norm_cm']<=tolerance['grid_residual_cm'] for r in records)
    report=dict(schema=RESULT_SCHEMA,status='qualified local state' if passed else 'controls incomplete or failed; no local state accepted',local_state_numerical_controls_passed=passed,accepted_local_state_files=[records[-1]['file']] if passed else [],cases=records,comparisons=comparisons,independent_variations=variation,tolerances=tolerance,request_sha256=_hash(source),execution_host=socket.gethostname(),memory_limit_bytes=128*1024**3,field_identifier=request['field_identifier'],electronic_state=request['electronic_state'],polynomial_reexpansion_used=False,potential_modified=False,global_bound_state_claimed=False,full_molecular_rate_qualified=False,field_certificate_sha256=certificate_hash,field_certificate_semantics='Upstream producer evidence; this tool certifies numerical local-state controls only',protocol=dict(id=protocol.protocol_id,sha256=protocol.sha256,manifest_version=protocol.manifest_version))
    report=validate_exchange_tree(_json(report), expected_schema=RESULT_SCHEMA)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request',type=Path);a=p.parse_args(argv)
    report=run_native_pes_variational_protocol(a.request);print(json.dumps(report,indent=2));return 0 if report['local_state_numerical_controls_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
