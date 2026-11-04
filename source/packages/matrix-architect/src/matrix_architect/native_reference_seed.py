"""Physical native seed independent of the signature of a reference Hessian."""
from dataclasses import fields, MISSING
import numpy as np
from matrix_chem.primitive_coordinates import eval_primitives, primitive_b_matrix
from matrix_chem.primitive_force_field import (PrimitiveMMForceField,
    PrimitiveFunctionalParameter, TorsionFourierParameter)
from .native_pic_field import NativePicField


def build_native_reference_seed(primitives, coordinates_angstrom, radial_priors,
                                nonbonded_evaluator, *, metadata=None):
    """Initialize original ZAFF functions at a minimum, saddle or path point.

    ``radial_priors`` maps bond atom pairs to physical Morse catalogue priors.
    Their existing ZAFF inverse-power conversion fixes dissociation energies
    and radial scales. Angular and periodic seed coefficients start at zero;
    they are parameters to fit, never observations set to zero. No Hessian
    eigenvalue is changed, and no assumption of positive curvature is made.
    The caller freezes the reactive primitive set before fitting.
    """
    from matrix_zaff.radial import inverse_power_stretch_from_morse_prior
    primitives=tuple(primitives);xyz=np.asarray(coordinates_angstrom,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.all(np.isfinite(xyz)) or not primitives:
        raise ValueError('finite reference geometry and frozen native primitives required')
    if any(p.kind not in ('bond','angle','dihedral','out_of_plane') for p in primitives):
        raise ValueError('original native bond, angle, dihedral or out-of-plane functions required')
    q=eval_primitives(primitives,xyz);parameters=[];torsions=[];ids=[]
    priors={frozenset(key):value for key,value in radial_priors.items()}
    for i,(primitive,reference) in enumerate(zip(primitives,q)):
        identifier=f'native_reference_{i}';ids.append(identifier)
        if primitive.kind=='bond':
            prior=priors.get(frozenset(primitive.atoms))
            if prior is None:raise ValueError('a physical catalogue prior is required for every support bond')
            inverse_power_stretch_from_morse_prior(prior.d_e_hartree,
                prior.r0_angstrom,prior.alpha_per_angstrom)
            form='inverse_power_stretch'
            values=(('D',prior.d_e_hartree),('R0',prior.r0_angstrom),
                    ('P',prior.alpha_per_angstrom*prior.r0_angstrom))
            source=prior.source
        elif primitive.kind=='dihedral':
            torsions.append(TorsionFourierParameter(identifier,(i,),reference,0.,0.,0.,(),
                'unfitted periodic seed',0.,0.,False,0.,0.));continue
        elif primitive.kind=='angle':
            if abs(np.sin(reference))<1e-6:
                raise ValueError('linear angles require a nonsingular frozen primitive set')
            form='cosine';values=(('THETA0',reference),('A',0.));source='unfitted cosine seed'
        else:
            form='harmonic';values=(('Q0',reference),('K',0.));source='unfitted switched out-of-plane seed'
        parameters.append(PrimitiveFunctionalParameter(identifier,(i,),primitive.kind,form,
            reference,0.,0.,values,source,None))
    count=len(primitives);zero=np.zeros((count,count))
    values={f.name:0 for f in fields(PrimitiveMMForceField)
            if f.default is MISSING and f.default_factory is MISSING}
    values.update(equilibrium_values=q,coupling_reference_values=q,primitive_hessian=zero,
        diagonal_force_constants=np.zeros(count),typed_force_constants=np.zeros(count),
        typed_equilibrium_values=q,atomic_types=(),primitive_types=(),primitive_type_ids=tuple(ids),
        sparse_typed_hessian=zero,retained_couplings=(),coupling_alerts=(),
        torsion_fourier_parameters=tuple(torsions),functional_parameters=tuple(parameters),
        equilibrium_fit_status='UNFITTED_PHYSICAL_SEED',hessian_input_unit='NOT_USED')
    record=dict(metadata or {});record.update(reference_coordinates_angstrom=xyz.tolist(),
        seed_uses_reference_hessian=False,primitive_tangent_rank=int(np.linalg.matrix_rank(primitive_b_matrix(primitives,xyz))))
    return NativePicField(primitives,PrimitiveMMForceField(**values),nonbonded_evaluator,metadata=record)
