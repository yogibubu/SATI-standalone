"""Recoverable discrepancy-controlled regularization of native observable fits.

The penalty is a prior in normalized native response coordinates. It does not
create zero-valued unobserved force constants or electronic observations.
Independent electronic targets must not be used to choose its strength.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def regularize_native_observable_cache(cache_directory,output_directory,*,relative_strengths=(1e-3,1e-5,1e-7,1e-9,1e-11,1e-13),column_floor=1e-13,block_columns=2048):
    """Reuse the finite calibration design and choose its strongest passing ridge.

    The row-space Gram system is cached independently of the ridge strength.
    Every proposed residual is checked against the ORIGINAL observable design,
    not inferred solely from a potentially ill-conditioned normal equation.
    No candidate is accepted unless all original calibration rows pass.
    """
    cache=Path(cache_directory);output=Path(output_directory)
    if output.exists():raise FileExistsError('regularized output must be fresh')
    output.mkdir(parents=True)
    a=np.load(cache/'design.npy',mmap_mode='r',allow_pickle=False);target=np.load(cache/'target.npy',allow_pickle=False)
    if a.ndim!=2 or target.shape!=(len(a),) or not np.all(np.isfinite(target)):raise ValueError('invalid cached observable system')
    if not 0<column_floor<1 or block_columns<1:raise ValueError('invalid column observability policy')
    strengths=tuple(float(v) for v in relative_strengths)
    if not strengths or any(not np.isfinite(v) or not 0<v<1 for v in strengths):raise ValueError('positive relative ridge strengths required')
    scales=np.sqrt(np.einsum('ij,ij->j',a,a,optimize=True));active=np.flatnonzero(scales>max(np.finfo(float).tiny,scales.max()*column_floor))
    gram=np.zeros((len(a),len(a)))
    for start in range(0,len(active),block_columns):
        indices=active[start:start+block_columns];block=np.array(a[:,indices]/scales[indices],order='F')
        gram+=block@block.T
        if start//block_columns%10==0:print('native regularization Gram columns',start+len(indices),'of',len(active),flush=True)
    gram=(gram+gram.T)/2
    np.save(output/'gram.npy',gram,allow_pickle=False);np.save(output/'column-scales.npy',scales,allow_pickle=False)
    scale=float(np.max(np.diag(gram)));history=[];selected=None;best=None;best_error=float('inf')
    for relative in sorted(set(strengths),reverse=True):
        strength=relative*scale;system=gram.copy();system.flat[::len(system)+1]+=strength
        try:dual=cho_solve(cho_factor(system,lower=True,check_finite=False),target,check_finite=False)
        except np.linalg.LinAlgError:
            history.append(dict(relative_strength=relative,status='NON_POSITIVE_DEFINITE'));continue
        coefficients=np.zeros(a.shape[1]);predicted=np.zeros(len(a))
        for start in range(0,len(active),block_columns):
            indices=active[start:start+block_columns];block=a[:,indices]
            coefficients[indices]=(block.T@dual)/scales[indices]**2
            predicted+=block@coefficients[indices]
        residual=predicted-target;error=float(np.max(abs(residual)))
        record=dict(relative_strength=relative,absolute_strength=strength,maximum_normalized_calibration_residual=error,normalized_coefficient_norm=float(np.linalg.norm(coefficients[active]*scales[active])),calibration_passed=error<=1.)
        history.append(record);print('native regularization',record,flush=True)
        if error<best_error:best=coefficients;best_error=error
        if error<=1.:selected=coefficients;break
    if best is None:raise np.linalg.LinAlgError('no numerically usable regularized system')
    final=best if selected is None else selected
    report=dict(schema='matrix.architect.native_regularized_fit.v1',calibration_passed=selected is not None,selection='strongest tested ridge passing every original calibration row; independent targets excluded',history=history,physical_field_qualified=False,independent_validation_required=True,source_cache=str(cache.resolve()),unprinted_derivatives_used_as_zero=False)
    np.save(output/'coefficients.npy',final,allow_pickle=False)
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return final,report


def native_field_from_observable_cache(cache_directory,coefficients,*,metadata=None):
    """Reconstruct the exact native field, retaining its frozen CM5 nonbonded."""
    from matrix_gf.harmonic import CM_PER_HARTREE
    from .native_pic_field import load_native_pic_field,NativePicField
    from .asymptotic_pic import AsymptoticPicFactor,AsymptoticPicCorrection
    cache=Path(cache_directory);parent=load_native_pic_field(cache/'parent-native.json');rows=json.loads((cache/'factors.json').read_text());c=np.asarray(coefficients,float)
    if c.shape!=(len(rows),) or not np.all(np.isfinite(c)):raise ValueError('finite coefficient per cached native term required')
    corrections=tuple(AsymptoticPicCorrection(float(value)/CM_PER_HARTREE,tuple(AsymptoticPicFactor(int(i),form,int(power),int(harmonic)) for i,form,power,harmonic in row)) for value,row in zip(c,rows) if value!=0)
    details=dict(parent.metadata);details.update(metadata or {});details.update(model_level='ZAFF4',physical_field_qualified=False)
    return NativePicField(parent.primitives,parent.field,parent.nonbonded_evaluator,corrections=parent.corrections+corrections,energy_offset_hartree=parent.energy_offset_hartree,metadata=details)


def validate_native_egh(field,coordinates,energies_hartree,gradients_bohr,hessians_bohr,displacements_angstrom,*,energy_tolerance_cm=1.,gradient_tolerance_Eh_bohr=1e-4,hessian_tolerance_Eh_bohr2=1e-4):
    """Check finite held-out E/G/H in exactly the calibration projection gauge.

    This function makes no coefficient or hyperparameter choices. Callers must
    supply genuinely held-out observations and keep their provenance separate.
    """
    from matrix_chem import BOHR_TO_ANGSTROM
    from matrix_gf.harmonic import CM_PER_HARTREE
    xyz=np.asarray(coordinates,float);e=np.asarray(energies_hartree,float);g=np.asarray(gradients_bohr,float);h=np.asarray(hessians_bohr,float)
    dx=np.asarray(displacements_angstrom,float).reshape(len(displacements_angstrom),-1);norm=np.linalg.norm(dx,axis=1)
    if xyz.ndim!=3 or xyz.shape[-1]!=3 or e.shape!=(len(xyz),) or g.shape!=(len(xyz),xyz.shape[1]*3) or h.shape!=(len(xyz),xyz.shape[1]*3,xyz.shape[1]*3) or dx.shape[1]!=xyz.shape[1]*3 or np.any(norm<=0):raise ValueError('independent E/G/H dimensions differ')
    tolerances=(energy_tolerance_cm,gradient_tolerance_Eh_bohr,hessian_tolerance_Eh_bohr2)
    if any(not np.isfinite(v) or v<=0 for v in tolerances) or not all(np.all(np.isfinite(v)) for v in (xyz,e,g,h,dx)):raise ValueError('finite observations and positive physical tolerances required')
    unit=dx/norm[:,None];points=[]
    for i,geometry in enumerate(xyz):
        model=field(geometry)
        errors=(float((model.energy_hartree-e[i])*CM_PER_HARTREE),float(np.max(abs(unit@(model.gradient_hartree_per_bohr-g[i])))),float(np.max(abs(unit@(model.hessian_hartree_per_bohr2-h[i])@unit.T))))
        normalized=max(abs(v)/t for v,t in zip(errors,tolerances))
        points.append(dict(index=i,energy_error_cm=errors[0],projected_gradient_error_Eh_bohr=errors[1],projected_hessian_error_Eh_bohr2=errors[2],maximum_normalized_error=normalized,passed=normalized<=1.))
        print('native independent E/G/H',i+1,'of',len(xyz),'normalized error',normalized,flush=True)
    return dict(schema='matrix.architect.native_independent_egh.v1',points=points,independent_gates_passed=bool(points) and all(p['passed'] for p in points),scope='supplied finite geometries and frozen calibration projection; no global or rate certificate',physical_field_qualified=False)
