"""Executable KEYMAKER gate for a fixed-budget positive native SAM response.

This stage fits already transported common-chart kernels. It acquires no
source points and does not promote a complete scalar potential or rate model.
"""
from pathlib import Path
import hashlib
import json
import socket
import numpy as np
from .reaction_path_hessian_fit import fit_positive_native_hessian_response
from .reactive_bond_order import ReactiveBondOrderFunctions

SCHEMA='matrix.architect.positive_transverse_request.v1'
HOSTS=frozenset(('oracle.sns.it','Mac.homenet.telecomitalia.it'))
POSITIONS=np.array([-.5,-.15,-.05,0.,.15,.5])
CM_PER_HARTREE=219474.6313632


def _verified_file(root,record):
    if not isinstance(record,dict) or set(record)!={'path','sha256'}:
        raise ValueError('file path and exact SHA256 required')
    path=(root/record['path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('input SHA256 mismatch')
    return path


def run_positive_transverse_protocol(request_path):
    request_path=Path(request_path).resolve();raw=request_path.read_bytes();r=json.loads(raw)
    keys={'schema','compute_host','observations','coordinate_definition','breaking_bond',
        'forming_bond','output_directory','kernel_coordinate_convention'}
    if set(r)!=keys or r['schema']!=SCHEMA:
        raise ValueError('closed positive-transverse request schema required')
    host=socket.gethostname()
    if host not in HOSTS or r['compute_host']!=host:
        raise RuntimeError('declared and verified remote compute host required')
    if r['kernel_coordinate_convention']!='common_TS_oscillator_scaled_native_SAM':
        raise ValueError('common native SAM kernel convention required')
    archive=_verified_file(request_path.parent,r['observations'])
    definition=_verified_file(request_path.parent,r['coordinate_definition'])
    with np.load(archive,allow_pickle=False) as d:
        if set(d.files)!={'positions','geometries','kernels','masses','symbols','sam_modes','lam_modes'}:
            raise ValueError('closed native-kernel archive schema required')
        xyz=d['geometries'];h=d['kernels'];mass=d['masses'];symbols=d['symbols']
        sam=d['sam_modes'];lam=d['lam_modes'];positions=d['positions']
        n=len(mass);vib=3*n-6
        if (positions.shape!=(6,) or not np.allclose(positions,POSITIONS,rtol=0,atol=1e-10)
            or xyz.shape!=(6,n,3) or symbols.shape!=(n,)
            or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(mass)) or np.any(mass<=0)
            or sam.ndim!=1 or lam.ndim!=1 or sam.dtype.kind not in 'iu' or lam.dtype.kind not in 'iu'
            or sorted(np.r_[sam,lam].tolist())!=list(range(1,vib))
            or h.shape!=(6,len(sam),len(sam)) or not len(sam)):
            raise ValueError('five off-TS observations and disjoint exhaustive SAM/LAM mode partition required')
    bonds=[]
    for key in ('breaking_bond','forming_bond'):
        pair=r[key]
        if (not isinstance(pair,list) or len(pair)!=2 or any(type(i)!=int or not 0<=i<n for i in pair)
                or pair[0]==pair[1]):raise ValueError('zero-based distinct atom pair required')
        bonds.append(tuple(pair))
    if set(bonds[0])==set(bonds[1]):raise ValueError('different breaking and forming bonds required')
    lengths=[float(np.linalg.norm(xyz[3,i]-xyz[3,j])) for i,j in bonds]
    if min(lengths)<=0:raise ValueError('positive TS bond lengths required')
    support=ReactiveBondOrderFunctions(n,*bonds,lengths,(4.,4.))
    u0=support.progress(xyz[3]).value
    design=np.array([[(support.progress(x).value-u0)**k for k in range(1,6)] for x in xyz])
    fit=fit_positive_native_hessian_response(design,h,reference_index=3,
        tolerance=10/CM_PER_HARTREE,basis_labels=[f'bounded_inverse_fourth_exchange^{k}' for k in range(1,6)],
        hessian_unit='hartree per transverse TS oscillator coordinate product')
    out=(request_path.parent/r['output_directory']).resolve();out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(raw).hexdigest();report=out/'result.json'
    if report.exists() and json.loads(report.read_text()).get('request_sha256')!=digest:
        raise ValueError('output belongs to another request; preserve its checkpoint')
    model_path=out/'positive_native_response.npz'
    np.savez_compressed(model_path,reference_half=fit.reference_half,coefficients=fit.coefficients,
        reference_bond_lengths_angstrom=lengths,breaking_bond=bonds[0],forming_bond=bonds[1],
        sam_modes=sam,lam_modes=lam,reference_progress=u0)
    result=dict(schema='matrix.architect.positive_transverse_result.v1',request_sha256=digest,
        protocol_id='zaff-positive-transverse-response.v1',kernel_fit_qualified=True,
        scalar_potential_qualified=False,molecular_rates_qualified=False,
        compute_host=host,coordinate_definition_sha256=r['coordinate_definition']['sha256'],
        observations_sha256=r['observations']['sha256'],model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        fit=fit.report,source_off_TS_Hessians=5,new_source_observations=0,
        pending_gates=['geometric SAM/LAM separation along the complete path',
            'stationary four-jets bound to source eigenvectors and frozen coordinates',
            'protected source E/G/H and TS fourth-order jet',
            'rigid invariance and scalar Cartesian derivative checks',
            'positive HDCPT2 fields on all original kinetic nodes',
            'finite-difference step convergence of complete rates below one percent',
            'unchanged regularized Wang-Landau, DVR and rotational sectors',
            'matched original-PES comparison where a reference exists'])
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
