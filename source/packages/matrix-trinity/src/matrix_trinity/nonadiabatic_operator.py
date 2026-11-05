"""Frozen-reference Born--Huang operators with explicit scalar evidence."""
from collections.abc import Mapping
import numpy as np
from scipy.constants import physical_constants


def symmetry_forbidden_scalar(*, initial_character, final_character,
        scalar_character=1, operation, reference_invariant, evidence):
    """Certify a zero scalar under one explicit 1D symmetry operation.

    For example inversion forbids an even nuclear scalar between g/u states.
    Electronic assignments and reference invariance are caller-supplied evidence;
    they are never inferred from a small printed derivative-coupling component.
    """
    chars = (initial_character, final_character, scalar_character)
    if (any(x not in (-1, 1) for x in chars) or not operation
            or reference_invariant is not True or not isinstance(evidence, Mapping)
            or not evidence):
        raise ValueError('Explicit real one-dimensional characters and reference evidence required')
    if initial_character*final_character*scalar_character != -1:
        raise ValueError('Supplied symmetry does not forbid this scalar')
    return dict(value_hartree=0., kind='symmetry_forbidden_reference_scalar',
        operation=operation, initial_character=initial_character,
        final_character=final_character, scalar_character=scalar_character,
        reference_invariant=True, evidence=dict(evidence))


def frozen_born_huang_operator(nac_cartesian_per_bohr, masses_amu,
        coordinate_jacobian_per_bohr, *, scalar_coupling_hartree,
        scalar_evidence, reference_identity):
    """Transport a complete supplied frozen-reference kinetic coupling.

    q is dimensionless, B=dq/dR, and p_q=-i*d/dq.  The source/target
    convention is <target|d/dR|source>.  The scalar already includes
    -sum_A tau_A/(2 M_A) in Hartree.  It is mandatory, including when zero.
    Coefficients are frozen at the specified reference in a flat local measure;
    no coordinate-dependent NAC field or missing divergence is generated.
    """
    d=np.asarray(nac_cartesian_per_bohr, complex)
    mass=np.asarray(masses_amu, float)
    b=np.asarray(coordinate_jacobian_per_bohr, float)
    if (d.shape != (len(mass),3) or mass.ndim != 1
            or b.ndim != 2 or b.shape[1] != d.size or not len(b)
            or np.any(mass <= 0) or not all(np.all(np.isfinite(x)) for x in (d,mass,b))
            or np.linalg.matrix_rank(b) != len(b)):
        raise ValueError('Finite full-rank coordinate Jacobian, Cartesian NAC and positive isotope masses required')
    scalar=complex(scalar_coupling_hartree)
    if (not np.isfinite(scalar) or not isinstance(scalar_evidence, Mapping)
            or not scalar_evidence or not reference_identity):
        raise ValueError('Explicit scalar kinetic coupling, evidence and reference identity required')
    if scalar == 0 and not scalar_evidence.get('kind'):
        raise ValueError('A zero scalar needs an explicit symmetry or model justification')
    if scalar_evidence.get('kind') == 'symmetry_forbidden_reference_scalar' and scalar != 0:
        raise ValueError('A symmetry-forbidden scalar must be zero')
    electron_mass_per_amu=(physical_constants['atomic mass constant'][0]
                           /physical_constants['electron mass'][0])
    momentum=-1j*(b@(d.ravel()/np.repeat(mass*electron_mass_per_amu,3)))
    return dict(schema='matrix.trinity.frozen_born_huang_operator.v1',
        constant_hartree=scalar, momentum_hartree=momentum,
        adjoint_constant_hartree=scalar.conjugate(),
        adjoint_momentum_hartree=momentum.conjugate(),
        scalar_evidence=dict(scalar_evidence),reference_identity=reference_identity,
        scope='Frozen Cartesian NAC and scalar coefficients at one reference; flat local nuclear measure',
        missing_scalar_inferred=False,coordinate_dependent_field_inferred=False)


def symmetry_project_cartesian_nac(coupling, coordinates_bohr, atomic_numbers,
        masses_amu, operations, transition_characters, *, tolerance_bohr=1e-6):
    """Average a NAC covector over supplied symmetry operations and characters.

    Operations act on column vectors. Atom permutations are recovered from the
    reference and preserve both element and isotope. No electronic character
    or point group is guessed by this routine.
    """
    from scipy.optimize import linear_sum_assignment
    d=np.asarray(coupling,complex);x=np.asarray(coordinates_bohr,float)
    z=np.asarray(atomic_numbers,int);mass=np.asarray(masses_amu,float)
    ops=np.asarray(operations,float);chars=np.asarray(transition_characters,float)
    if (x.ndim!=2 or x.shape[1]!=3 or d.shape!=x.shape or z.shape!=(len(x),)
            or mass.shape!=(len(x),) or ops.ndim!=3 or ops.shape[1:]!=(3,3)
            or chars.shape!=(len(ops),) or not len(ops) or np.any(abs(chars)!=1)
            or tolerance_bohr<=0 or not all(np.all(np.isfinite(v)) for v in (d,x,mass,ops,chars))):
        raise ValueError('Consistent finite reference, NAC, isotope masses and supplied symmetry operations required')
    x=x-np.average(x,axis=0,weights=mass)
    result=np.zeros_like(d);maximum_error=0.
    for op,char in zip(ops,chars):
        if not np.allclose(op.T@op,np.eye(3),rtol=0,atol=1e-10):
            raise ValueError('Orthogonal Cartesian symmetry operations required')
        cost=np.linalg.norm((x@op.T)[:,None,:]-x[None,:,:],axis=2)
        incompatible=(z[:,None]!=z[None,:])|(mass[:,None]!=mass[None,:])
        cost[incompatible]=np.inf
        rows,perm=linear_sum_assignment(cost)
        error=float(cost[rows,perm].max())
        if error>tolerance_bohr:
            raise ValueError('Supplied operation does not preserve the isotope-resolved reference')
        maximum_error=max(maximum_error,error)
        result+=char*(d[perm]@op)
    result/=len(ops)
    change=float(np.linalg.norm(result-d))
    return dict(coupling_per_bohr=result,reference_symmetry_error_bohr=maximum_error,
        projection_change_per_bohr=change,
        projection_relative_change=change/max(float(np.linalg.norm(d)),np.finfo(float).tiny),
        operation_count=len(ops),electronic_characters_inferred=False)
