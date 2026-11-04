"""Joint fit of available observations to one resident native ZAFF surface.

Observations may come from several stationary points and IRC Hessians. Their
coordinate frames may differ; the native coefficient basis must be identical.
Unavailable entries are omitted, and the fitted surface has no source-PES
callback. Existing native functions, switches and nonbonded terms are retained.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np
from matrix_chem import BOHR_TO_ANGSTROM
from .native_pic_field import NativePicField,FrozenNativeNonbonded,save_native_pic_field
from .asymptotic_pic import AsymptoticPicCorrection
from .asymptotic_pic_hessian import NativeCorrectionHessianDesign
from .native_nonlocal_support import equilibrated_minimum_norm_fit


def native_fit_basis_sha256(parent,factors):
    record=dict(primitives=[repr(p) for p in parent.primitives],
        reference=np.asarray(parent.field.equilibrium_values).tolist(),
        functions=repr(parent.field.functional_parameters),
        factors=[[repr(f) for f in term] for term in factors])
    if hasattr(parent,'radial_representation'):
        record['radial_representation']=parent.radial_representation
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class NativeObservationBlock:
    name: str
    design: np.ndarray
    target: np.ndarray
    baseline: np.ndarray
    tolerance: np.ndarray
    basis_sha256: str
    source_sha256: str
    coordinate_id: str
    source_kind: str = 'measured'


def _coordinate_columns(parent,kernel,geometry,first,second):
    xyz=np.asarray(geometry,float);a=np.asarray(first,float);b=np.asarray(second,float)
    de,dg,dh=kernel.value(xyz);point=parent(xyz)
    gx=point.gradient_hartree_per_bohr/BOHR_TO_ANGSTROM
    hx=point.hessian_hartree_per_bohr2/BOHR_TO_ANGSTROM**2
    designs=[np.asarray(de)[None,:],a.T@dg,
        np.einsum('ia,ijt,jb->abt',a,dh,a,optimize=True)+np.einsum('it,iab->abt',dg,b)]
    bases=[np.array([point.energy_hartree]),a.T@gx,a.T@hx@a+np.einsum('i,iab->ab',gx,b)]
    return designs,bases


def native_egh_observation_block(parent,factors,geometry,*,name,source_sha256,coordinate_id,
        target_energy=None,target_gradient=None,target_hessian=None,
        energy_tolerance=1e-5,gradient_tolerance=1e-5,hessian_tolerance=1e-5,
        first=None,second=None,energy_scale=1.,source_kind='measured',column_kernel=None):
    """Build rows from actual native E/G/H, with the nonstationary chain rule.

    Input geometries and Cartesian realization derivatives use angstrom.
    Targets are derivatives in the caller's declared q coordinates, multiplied
    by energy_scale (e.g. hartree to cm-1). NaN entries mean unavailable.
    Only the provided observations generate rows; no target is inferred.
    """
    xyz=np.asarray(geometry,float);d=xyz.size
    a=np.eye(d) if first is None else np.asarray(first,float)
    if a.ndim!=2 or a.shape[0]!=d or not np.all(np.isfinite(a)):
        raise ValueError('finite Cartesian realization Jacobian required')
    n=a.shape[1];b=np.zeros((d,n,n)) if second is None else np.asarray(second,float)
    if b.shape!=(d,n,n) or not np.all(np.isfinite(b)) or not np.isfinite(energy_scale) or energy_scale<=0:
        raise ValueError('aligned geometry curvature and positive energy scale required')
    factors=tuple(tuple(term) for term in factors)
    if column_kernel is not None:
        kernel=column_kernel
    elif hasattr(parent,'radial_representation'):
        from .native_valence_observations import NativeValenceCorrectionDesign
        kernel=NativeValenceCorrectionDesign(parent,factors)
    else:
        kernel=NativeCorrectionHessianDesign(parent.primitives,parent.field,
            [tuple((f.primitive_index,f.form,f.power,f.harmonic,f.window_start,f.window_end) for f in term) for term in factors])
    designs,bases=_coordinate_columns(parent,kernel,xyz,a,b)
    rows=[];targets=[];baselines=[];tolerances=[]
    for order,(target,tolerance) in enumerate(zip((target_energy,target_gradient,target_hessian),
            (energy_tolerance,gradient_tolerance,hessian_tolerance))):
        if target is None:continue
        target=np.atleast_1d(np.asarray(target,float))
        expected=(1,) if order==0 else (n,)*order
        if target.shape!=expected or np.any(np.isinf(target)) or not np.isfinite(tolerance) or tolerance<=0:
            raise ValueError('aligned available observations and positive tolerances required')
        mask=np.isfinite(target)
        if order==2:mask &= np.triu(np.ones((n,n),bool))
        if not mask.any():continue
        rows.append(designs[order][mask]*energy_scale);targets.append(target[mask])
        baselines.append(bases[order][mask]*energy_scale);tolerances.append(np.full(mask.sum(),tolerance))
    if not rows:raise ValueError('at least one available observation required')
    return NativeObservationBlock(name,np.vstack(rows),np.concatenate(targets),np.concatenate(baselines),
        np.concatenate(tolerances),native_fit_basis_sha256(parent,factors),source_sha256,coordinate_id,source_kind)


def fit_native_observation_blocks(parent,factors,blocks,*,cache_directory,relative_rank_cutoff=1e-11,tie_permutations=()):
    """Fit all anchors simultaneously, then return the same native PES interface.

    A rejected calibration returns no usable field. Passing the finite fit does
    not certify withheld derivatives or rates; those are independent checks.
    """
    from .native_valence_energy_fit import NativeValenceEnergySurface
    if type(parent) not in (NativePicField,NativeValenceEnergySurface):
        raise TypeError('joint fitting requires a resident NativePicField or native valence surface')
    factors=tuple(tuple(t) for t in factors);blocks=tuple(blocks)
    if not factors or not blocks:raise ValueError('native basis and available observation blocks required')
    basis=native_fit_basis_sha256(parent,factors);names=set();design=[];target=[];records=[];offset=0
    for block in blocks:
        if block.name in names or not block.name or block.basis_sha256!=basis:
            raise ValueError('unique observation blocks in the exact same native basis required')
        if (block.source_kind not in ('measured','declared_model') or not block.coordinate_id
                or len(block.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in block.source_sha256)):
            raise ValueError('source identity, coordinate frame and observation provenance required')
        a=np.asarray(block.design,float);t=np.asarray(block.target,float);base=np.asarray(block.baseline,float)
        tol=np.broadcast_to(np.asarray(block.tolerance,float),t.shape)
        if (a.ndim!=2 or a.shape[1]!=len(factors) or t.shape!=(len(a),) or base.shape!=t.shape or not len(a)
                or not all(np.all(np.isfinite(v)) for v in (a,t,base,tol)) or np.any(tol<=0)):
            raise ValueError('finite observed rows and positive physical tolerances required')
        names.add(block.name);design.append(a/tol[:,None]);target.append((t-base)/tol)
        records.append(dict(name=block.name,start=offset,stop=offset+len(t),source_sha256=block.source_sha256,
            coordinate_id=block.coordinate_id,source_kind=block.source_kind));offset+=len(t)
    directory=Path(cache_directory);directory.mkdir(parents=True,exist_ok=False)
    a=np.vstack(design);y=np.concatenate(target)
    np.save(directory/'design.npy',a,allow_pickle=False);np.save(directory/'residual_target.npy',y,allow_pickle=False)
    metadata=dict(schema='matrix.architect.native_multi_anchor_fit.v1',basis_sha256=basis,blocks=records,
        native_functions_unchanged=True,nonbonded_unchanged=True,new_reference_observations=0)
    (directory/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    nonbonded=getattr(parent,'nonbonded_evaluator',None)
    if isinstance(nonbonded,FrozenNativeNonbonded):
        save_native_pic_field(directory/'parent-native.json',parent)
    n=len(parent.primitives);permutations=[tuple(range(n)),*[tuple(p) for p in tie_permutations]]
    if any(sorted(p)!=list(range(n)) for p in permutations):raise ValueError('true coefficient permutation required')
    groups={}
    for i,term in enumerate(factors):
        key=min(tuple(sorted((p[f.primitive_index],f.form,f.power,f.harmonic,f.window_start,f.window_end) for f in term)) for p in permutations)
        groups.setdefault(key,[]).append(i)
    groups=list(groups.values()) if tie_permutations else [[i] for i in range(len(factors))];grouped=np.column_stack([a[:,group].sum(axis=1) for group in groups])
    fitted,solver=equilibrated_minimum_norm_fit(grouped,y,relative_rank_cutoff=relative_rank_cutoff)
    coefficients=np.zeros(len(factors))
    for value,group in zip(fitted,groups):coefficients[group]=value
    residual=a@coefficients-y;checks=[]
    for row in records:
        error=float(np.max(abs(residual[row['start']:row['stop']])))
        checks.append(dict(name=row['name'],maximum_normalized_error=error,passed=error<=1.))
    passed=all(c['passed'] for c in checks)
    report=dict(metadata,solver=solver,coefficient_groups=groups,calibration_checks=checks,calibration_passed=passed,
        withheld_validation_passed=False,physical_field_qualified=False,
        source_PES_required_after_fit=False,nonbonded_charge_source=str(getattr(nonbonded,'charge_source','unspecified')))
    np.save(directory/'coefficients.npy',coefficients,allow_pickle=False)
    (directory/'fit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not passed:return None,report
    corrections=tuple(AsymptoticPicCorrection(float(c),f) for c,f in zip(coefficients,factors) if c!=0.)
    if type(parent) is NativePicField:
        field=NativePicField(parent.primitives,parent.field,parent.nonbonded_evaluator,
            corrections=parent.corrections+corrections,energy_offset_hartree=parent.energy_offset_hartree,
            metadata=dict(parent.metadata,joint_native_fit=report))
        if isinstance(nonbonded,FrozenNativeNonbonded):save_native_pic_field(directory/'fitted-native.json',field)
    else:
        field=NativeValenceEnergySurface(parent.primitives,parent.reference,parent.gradient,parent.hessian,
            parent.radial_exponent,parent.corrections+corrections,metadata=dict(parent.metadata,joint_native_fit=report),
            radial_representation=parent.radial_representation)
        field.save(directory/'fitted-native.json')
    return field,report


def native_derivative_observation_block(parent,factors,coordinate_map,dimension,*,name,
        source_sha256,coordinate_id,cubic=None,quartic=None,steps=(.04,.02),
        cubic_tolerance=10.,quartic_tolerance=10.,energy_scale=1.,source_kind='measured'):
    """Fit supplied F3/F4 entries by differentiating native model columns only.

    coordinate_map(q) returns (geometry_angstrom, dx/dq, d2x/dq2). It performs
    geometry realization only. Finite differences below never call the source
    electronic/reference potential. Four-distinct entries are supported when
    actually supplied; missing entries do not become observation rows.
    """
    from functools import lru_cache
    n=int(dimension);h,half=map(float,steps)
    if n<1 or h<=0 or half<=0 or not np.isclose(h,2*half) or not np.isfinite(energy_scale) or energy_scale<=0:
        raise ValueError('positive dimension and factor-two finite model steps required')
    factors=tuple(tuple(term) for term in factors)
    kernel=NativeCorrectionHessianDesign(parent.primitives,parent.field,
        [tuple((f.primitive_index,f.form,f.power,f.harmonic,f.window_start,f.window_end) for f in term) for term in factors])
    @lru_cache(None)
    def value(key):
        xyz,a,b=coordinate_map(np.asarray(key,float))
        a=np.asarray(a,float);b=np.asarray(b,float)
        if a.shape!=(np.size(xyz),n) or b.shape!=(np.size(xyz),n,n) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError('finite consistent geometry realization derivatives required')
        designs,bases=_coordinate_columns(parent,kernel,xyz,a,b)
        return designs[2],bases[2]
    zero=np.zeros(n);d0,b0=value(tuple(zero));rows=[];bases=[];targets=[];tols=[]
    for degree,observations,tolerance in ((3,cubic,cubic_tolerance),(4,quartic,quartic_tolerance)):
        if observations is None:continue
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('positive derivative tolerance required')
        for key,target in sorted(observations.items()):
            if len(key)!=degree or any(not isinstance(i,(int,np.integer)) or not 0<=i<n for i in key) or not np.isfinite(target):
                raise ValueError('finite explicitly supplied derivative entries required')
            estimates=[]
            for step in (h,half):
                if degree==3:
                    i,j,k=key;e=np.eye(n)[k]*step
                    dp,bp=value(tuple(e));dm,bm=value(tuple(-e))
                    estimates.append(((dp[i,j]-dm[i,j])/(2*step),(bp[i,j]-bm[i,j])/(2*step)))
                else:
                    repeated=[i for i in key if key.count(i)>=2]
                    if repeated:
                        k=repeated[0];rest=list(key);rest.remove(k);rest.remove(k);i,j=rest;e=np.eye(n)[k]*step
                        dp,bp=value(tuple(e));dm,bm=value(tuple(-e))
                        estimates.append(((dp[i,j]-2*d0[i,j]+dm[i,j])/step**2,(bp[i,j]-2*b0[i,j]+bm[i,j])/step**2))
                    else:
                        i,j,k,l=key;derivative=np.zeros(len(factors));base=0.
                        for sign in (-1,1):
                            for other in (-1,1):
                                e=step*(sign*np.eye(n)[i]+other*np.eye(n)[j]);d,b=value(tuple(e))
                                derivative+=sign*other*d[k,l]/(4*step**2);base+=sign*other*b[k,l]/(4*step**2)
                        estimates.append((derivative,base))
            rows.append((4*estimates[1][0]-estimates[0][0])/3*energy_scale)
            bases.append((4*estimates[1][1]-estimates[0][1])/3*energy_scale)
            targets.append(float(target));tols.append(float(tolerance))
    if not rows:raise ValueError('available cubic or quartic entries required')
    return NativeObservationBlock(name,np.array(rows),np.array(targets),np.array(bases),np.array(tols),
        native_fit_basis_sha256(parent,factors),source_sha256,coordinate_id,source_kind)


@dataclass(frozen=True)
class NativeReferencePoint:
    """Available data at a minimum, saddle, or ordinary path geometry.

    coordinate_map realizes displacements and geometry derivatives only.
    Energies from different points must share the caller's energy zero.
    Cubic/quartic maps contain supplied entries, not implicitly filled tensors.
    """
    name: str
    role: str
    source_sha256: str
    coordinate_id: str
    dimension: int
    coordinate_map: object
    energy: object = None
    gradient: object = None
    hessian: object = None
    cubic: object = None
    quartic: object = None
    source_kind: str = 'measured'


def fit_native_reference_data(parent,factors,reference_points,*,cache_directory,
        energy_scale=1.,energy_tolerance=10.,gradient_tolerance=10.,hessian_tolerance=10.,
        cubic_tolerance=10.,quartic_tolerance=10.,model_derivative_steps=(.04,.02),relative_rank_cutoff=1e-11):
    """Generalize the single-minimum fit to any finite available reference set.

    One minimum takes exactly the same route as minima, saddles and path
    Hessians together. The result is one NativePicField using the original
    ZAFF functions. There is no source-PES dependency in the fitted field.
    Tolerances refer to the declared coordinates and energy scale.
    """
    factors=tuple(tuple(f) for f in factors);points=tuple(reference_points);blocks=[]
    if not points:raise ValueError('at least one reference point required')
    names=set()
    for point in points:
        if (not isinstance(point,NativeReferencePoint) or point.name in names
                or point.role not in ('minimum','transition_state','path')
                or not isinstance(point.dimension,(int,np.integer)) or isinstance(point.dimension,bool)
                or point.dimension<1 or not callable(point.coordinate_map)):
            raise ValueError('distinct reference points with declared roles and coordinate maps required')
        names.add(point.name)
        options=dict(source_sha256=point.source_sha256,coordinate_id=point.coordinate_id,
            source_kind=point.source_kind,energy_scale=energy_scale)
        if any(value is not None for value in (point.energy,point.gradient,point.hessian)):
            xyz,first,second=point.coordinate_map(np.zeros(point.dimension))
            blocks.append(native_egh_observation_block(parent,factors,xyz,name=point.name+':E_G_H',
                target_energy=point.energy,target_gradient=point.gradient,target_hessian=point.hessian,
                energy_tolerance=energy_tolerance,gradient_tolerance=gradient_tolerance,hessian_tolerance=hessian_tolerance,
                first=first,second=second,**options))
        if point.cubic or point.quartic:
            blocks.append(native_derivative_observation_block(parent,factors,point.coordinate_map,point.dimension,
                name=point.name+':F3_F4',cubic=point.cubic,quartic=point.quartic,steps=model_derivative_steps,
                cubic_tolerance=cubic_tolerance,quartic_tolerance=quartic_tolerance,**options))
        if all(value is None for value in (point.energy,point.gradient,point.hessian)) and not point.cubic and not point.quartic:
            raise ValueError('each reference point must supply available observations')
    field,report=fit_native_observation_blocks(parent,factors,blocks,
        cache_directory=cache_directory,relative_rank_cutoff=relative_rank_cutoff)
    references=[dict(name=p.name,role=p.role,coordinate_id=p.coordinate_id,source_sha256=p.source_sha256) for p in points]
    report['reference_points']=references
    report['single_minimum_special_case']=len(points)==1 and points[0].role=='minimum'
    Path(cache_directory,'fit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if field is not None:
        field.metadata['joint_native_fit']=report
        destination=Path(cache_directory,'fitted-native.json')
        if destination.exists():
            # Replace only the artifact just created by this same fit invocation.
            destination.unlink()
            save_native_pic_field(destination,field)
    return field,report
