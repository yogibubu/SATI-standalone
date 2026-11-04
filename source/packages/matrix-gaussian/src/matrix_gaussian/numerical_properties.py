"""Generic property Taylor tensors from Gaussian axis numerical derivatives.

ND12=-2 records first/second derivatives of both a property and its Cartesian
Jacobian along mass-weighted normal directions. This yields full D2 and
repeated-index D3. Unacquired three-distinct derivatives are an explicit model.
"""
from dataclasses import dataclass
from itertools import permutations
import hashlib
from pathlib import Path
import numpy as np
from .fchk import _read_fchk_blocks,read_gaussian_fchk
from .modes import gaussian_cartesian_modes_to_mass_weighted


@dataclass(frozen=True)
class GaussianNumericalProperty:
    derivatives: tuple
    coordinate_id: str
    diagnostics: dict


def read_gaussian_axis_property(path, *, property_key, jacobian_key,
                                component_shape, cubic_coverage,
                                numerical_directions_qmw=None,
                                relative_tolerance=1e-3, absolute_tolerance=1e-8,
                                symmetrize_second_derivatives=False):
    """Return D0,D1,full D2 and declared repeated-index D3 in real modal q.

    Component axes precede derivative axes. Properties and Jacobians must be
    in the same Cartesian frame as the fchk normal modes. Source directions
    can be supplied explicitly, including rotations in degenerate subspaces.
    Otherwise signs are identified from independent first-derivative and
    diagonal-second-derivative records and linked by D2 symmetry; unresolved
    gauges fail rather than being guessed. No molecule-specific rule is used.
    Explicit symmetrization projects D2 onto its symmetric part even when its
    antisymmetric residual exceeds the consistency tolerance; the correction
    is reported. Independent frame and diagonal checks remain active.
    """
    from matrix_gf.harmonic import HESSIAN_EIGENVALUE_TO_CM, CM_PER_HARTREE

    if cubic_coverage!='repeated_index_model':
        raise ValueError('explicit repeated_index_model cubic property coverage required')
    if not isinstance(symmetrize_second_derivatives,(bool,np.bool_)):
        raise ValueError('symmetrize_second_derivatives must be boolean')
    if (not np.isfinite(relative_tolerance) or relative_tolerance<=0
            or not np.isfinite(absolute_tolerance) or absolute_tolerance<=0):
        raise ValueError('positive finite numerical consistency tolerances required')
    shape=tuple(component_shape)
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1 for v in shape):
        raise ValueError('positive integer component dimensions required')
    source=Path(path);raw=_read_fchk_blocks(source);mol=read_gaussian_fchk(source)
    n=len(mol.harmonic_frequencies_cm);nc=3*len(mol.masses_amu);components=int(np.prod(shape)) if shape else 1
    if (int(raw.get('NDeriv for static properties',-1))!=n
            or int(raw.get('ND12 for static properties',0))!=-2):
        raise ValueError('Gaussian ND12=-2 full normal-direction record required')
    def array(key,size):
        value=np.asarray(raw.get(key,[]),float)
        if value.size!=size or not np.all(np.isfinite(value)):
            raise ValueError('missing or malformed numerical property block: '+key)
        return value
    d0=array(property_key,components)
    jac=array(jacobian_key,nc*components).reshape(nc,components).T
    moments=array(property_key+' num derivs',2*n*components).reshape(2,n,components)
    jacnum=array(jacobian_key+' num derivs',2*n*nc*components).reshape(2,n,nc,components)
    masses=np.repeat(mol.masses_amu,3)
    target=gaussian_cartesian_modes_to_mass_weighted(mol.normal_modes,mol.masses_amu).T/np.sqrt(masses)[:,None]
    analytic=jac@target
    mixed=np.einsum('jxa,xi->aij',jacnum[0],target)
    scale=max(1e-300,float(np.max(abs(moments[1]))),float(np.max(abs(mixed))))
    # Absolute consistency threshold is declared in source physical units.
    tolerance=absolute_tolerance+relative_tolerance*scale
    if numerical_directions_qmw is None:
        signs=np.zeros(n)
        for j in range(n):
            for left,right in ((analytic[:,j],moments[0,j]),(mixed[:,j,j],moments[1,j])):
                if max(np.linalg.norm(left),np.linalg.norm(right))<=absolute_tolerance:continue
                plus=np.linalg.norm(left-right);minus=np.linalg.norm(left+right)
                if min(plus,minus)>absolute_tolerance+relative_tolerance*max(np.linalg.norm(left),np.linalg.norm(right)):
                    raise ValueError('source numerical direction is not a resolved sign of the target mode; supply numerical_directions_qmw')
                sign=1. if plus<minus else -1.
                if signs[j] and signs[j]!=sign:raise ValueError('inconsistent Gaussian numerical-coordinate gauge anchors')
                signs[j]=sign
        # Symmetry propagates an identified gauge through nonzero cross terms.
        changed=True
        while changed:
            changed=False
            for i in range(n):
                for j in range(i):
                    a,b=mixed[:,i,j],mixed[:,j,i]
                    if min(np.linalg.norm(a),np.linalg.norm(b))<=tolerance:continue
                    relative=1. if np.linalg.norm(a-b)<np.linalg.norm(a+b) else -1.
                    if signs[i] and not signs[j]:signs[j]=relative*signs[i];changed=True
                    elif signs[j] and not signs[i]:signs[i]=relative*signs[j];changed=True
        if np.any(signs==0):raise ValueError('unresolved numerical-coordinate signs; supply numerical_directions_qmw')
        directions=target*signs
        gauge='identified from redundant physical derivative records'
    else:
        directions=np.asarray(numerical_directions_qmw,float)
        if directions.shape!=target.shape or not np.all(np.isfinite(directions)):
            raise ValueError('finite mass-weighted displacement directions in the fchk Cartesian frame required')
        gauge='explicit source numerical directions'
    transform=(np.diag(signs) if numerical_directions_qmw is None
               else directions.T@(masses[:,None]*target))
    if (not np.allclose(transform@transform.T,np.eye(n),atol=1e-6,rtol=1e-6)
            or not np.allclose(directions.T@(masses[:,None]*directions),np.eye(n),atol=1e-6,rtol=1e-6)):
        raise ValueError('source numerical directions do not span an orthonormal vibrational basis')
    if not np.allclose(jac@directions,moments[0].T,atol=absolute_tolerance,rtol=relative_tolerance):
        raise ValueError('first derivative gauge/frame consistency failed')
    second=np.einsum('jxa,xi->aij',jacnum[0],directions)
    symmetry_error=float(np.max(abs(second-second.transpose(0,2,1))))
    if symmetry_error>tolerance and not symmetrize_second_derivatives:
        raise ValueError('numerical second derivative symmetry fails the declared tolerance')
    diagonal_error=float(np.max(abs(np.einsum('aii->ia',second)-moments[1])))
    if diagonal_error>tolerance:raise ValueError('independent property/Jacobian second derivatives disagree')
    symmetric_second=(second+second.transpose(0,2,1))/2
    correction=symmetric_second-second
    correction_max=float(np.max(abs(correction)))
    correction_relative=float(np.linalg.norm(correction)/max(np.linalg.norm(second),1e-300))
    second=symmetric_second
    repeated=np.einsum('jxa,xi->aij',jacnum[1],directions)
    third=np.zeros((components,n,n,n))
    for i in range(n):
        for j in range(n):
            for key in set(permutations((i,j,j))):third[(slice(None),)+key]=repeated[:,i,j]
    # Close only the SOURCE-basis property model, then transform all of it.
    second=np.einsum('aij,ik,jl->akl',second,transform,transform,optimize=True)
    third=np.einsum('aijk,il,jm,kn->almn',third,transform,transform,transform,optimize=True)
    frequencies=mol.harmonic_frequencies_cm
    if np.any(frequencies==0) or not np.all(np.isfinite(frequencies)):
        raise ValueError('finite nonzero signed mode frequencies required')
    sizes=HESSIAN_EIGENVALUE_TO_CM/np.sqrt(CM_PER_HARTREE*np.abs(frequencies))
    first=analytic*sizes
    second*=sizes[None,:,None]*sizes[None,None,:]
    third*=sizes[None,:,None,None]*sizes[None,None,:,None]*sizes[None,None,None,:]
    derivatives=tuple(a.reshape(shape+(n,)*order) for order,a in enumerate((d0,first,second,third)))
    coordinate_id=hashlib.sha256(target.astype('<f8').tobytes()+mol.cartesian_coordinates_bohr.astype('<f8').tobytes()+frequencies.astype('<f8').tobytes()).hexdigest()
    return GaussianNumericalProperty(derivatives,coordinate_id,dict(
        schema='matrix.gaussian.axis_property.v1',property_key=property_key,jacobian_key=jacobian_key,
        component_shape=list(shape),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_coordinate_gauge=gauge,source_to_target_qmw=transform.tolist(),
        derivative_orders=[0,1,2,3],second_derivative_coverage='complete',
        third_derivative_coverage=cubic_coverage,source_three_distinct_third_derivatives_acquired=False,
        second_derivative_symmetry_error_source_units=symmetry_error,
        second_derivative_symmetrization_authorized=bool(symmetrize_second_derivatives),
        second_derivative_symmetry_tolerance_source_units=tolerance,
        second_derivative_symmetry_within_tolerance=bool(symmetry_error<=tolerance),
        second_derivative_symmetrization_maximum_correction_source_units=correction_max,
        second_derivative_symmetrization_relative_frobenius_correction=correction_relative,
        independent_diagonal_second_derivative_error_source_units=diagonal_error,
        relative_tolerance=relative_tolerance,absolute_tolerance=absolute_tolerance,
        convention='real dimensionless target normal coordinates; source repeated-index cubic property closed once before transport',
    ))
