"""Existing Gaussian normal QFFs in explicitly selected harmonic/modal frames.

No new electronic derivatives are acquired. An excluded source coordinate
removes every cubic/quartic monomial containing it before tensor transport;
all supplied target harmonic modes remain in the quadratic model.
"""
from itertools import permutations
import numpy as np
from .vibronic_vpt2 import SAMVPT2Field
from .vibronic_variational import project_anharmonic_potential


def gaussian_normal_quartic_field(qff, source_modes):
    """Build the declared printed Gaussian polynomial for positive source modes."""
    modes=np.asarray(source_modes)
    frequencies=np.asarray(qff.harmonic_frequencies_cm,float)
    if (modes.ndim!=1 or modes.dtype.kind not in 'iu' or not len(modes)
            or len(np.unique(modes))!=len(modes) or np.any(modes<0)
            or np.any(modes>=len(frequencies)) or np.any(frequencies[modes]<=0)):
        raise ValueError('unique positive-frequency Gaussian source modes required')
    n=len(modes);position={int(mode):i for i,mode in enumerate(modes)}
    tensors=[]
    for degree,entries in ((3,qff.cubic_cm),(4,qff.quartic_cm)):
        tensor=np.zeros((n,)*degree)
        for key,value in entries.items():
            if all(mode in position for mode in key):
                for row in set(permutations(tuple(position[mode] for mode in key))):tensor[row]=value
        tensors.append(tensor)
    return SAMVPT2Field(frequencies[modes],*tensors,quartic_coverage='semidiagonal_model')


def gaussian_field_in_harmonic_frame(source, target, qff, *, source_modes,
                                      target_modes, allow_harmonic_only_directions=False):
    """Transport the supplied source polynomial; use target quadratic frequencies.

    The stationary target geometry identifies the harmonic frame. Higher
    derivatives are tangent-transported from the original source geometry,
    without a Taylor translation or numerical reevaluation. Rank-deficient
    anharmonic maps require explicit permission; this means some target
    collective coordinates have harmonic quadratic terms only.
    """
    from matrix_gaussian.modes import gaussian_cartesian_modes_to_mass_weighted
    modes=np.asarray(source_modes);active=np.asarray(target_modes)
    masses=source.masses_amu
    if not np.array_equal(masses,target.masses_amu):raise ValueError('atom/mass ordering differs')
    if (active.ndim!=1 or active.dtype.kind not in 'iu' or not len(active)
            or len(np.unique(active))!=len(active) or np.any(active<0)
            or np.any(active>=len(target.harmonic_frequencies_cm))):
        raise ValueError('unique valid target mode indices required')
    field=gaussian_normal_quartic_field(qff,modes)
    bx=source.cartesian_coordinates_bohr-np.average(source.cartesian_coordinates_bohr,axis=0,weights=masses)
    rx=target.cartesian_coordinates_bohr-np.average(target.cartesian_coordinates_bohr,axis=0,weights=masses)
    u,_,vt=np.linalg.svd((rx*masses[:,None]).T@bx)
    rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    bm=gaussian_cartesian_modes_to_mass_weighted(source.normal_modes,masses).reshape(-1,len(masses),3)
    rm=gaussian_cartesian_modes_to_mass_weighted(target.normal_modes,masses).reshape(-1,len(masses),3)
    rm=np.einsum('mac,cd->mad',rm,rotation)
    overlap=bm[modes].reshape(len(modes),-1)@rm[active].reshape(len(active),-1).T
    frequencies=target.harmonic_frequencies_cm[active]
    transform=np.sqrt(field.frequencies_cm)[:,None]*overlap/np.sqrt(frequencies)[None,:]
    result=project_anharmonic_potential(field,frequencies,transform,
        allow_rank_deficient=allow_harmonic_only_directions)
    excluded=sorted(set(range(len(qff.harmonic_frequencies_cm)))-set(map(int,modes)))
    return result,dict(source_modes_1based=(modes+1).tolist(),target_modes_1based=(active+1).tolist(),
        excluded_source_modes_1based=[i+1 for i in excluded],source_from_target_dimensionless=transform,
        mode_subspace_singular_values=np.linalg.svd(overlap,compute_uv=False),
        anharmonic_map_rank=int(np.linalg.matrix_rank(transform)),target_quadratic_mode_count=len(active),
        harmonic_only_collective_directions=len(active)-int(np.linalg.matrix_rank(transform)),
        target_harmonic_frequencies_used=True,new_electronic_derivatives_computed=False,
        derivative_geometry='original Gaussian B3LYP geometry; no Taylor translation',
        coordinate_transport='mass-weighted Eckart alignment and dimensionless tangent projection',
        source_quartic_coverage='printed semidiagonal polynomial; unprinted source terms zero')


def normal_field_as_sonic(field):
    """Represent a reduced normal QFF as the existing potential-only GVPT2 input."""
    from matrix_gf.harmonic import CM_PER_HARTREE,HESSIAN_EIGENVALUE_TO_CM
    from .curvilinear_vpt2 import SonicVPT2Input
    if not isinstance(field,SAMVPT2Field):raise TypeError('SAMVPT2Field required')
    if np.any(field.metric_first_cm) or np.any(field.metric_second_cm):raise ValueError('potential-only field required')
    w=field.frequencies_cm;n=len(w)
    scale=np.sqrt(w*CM_PER_HARTREE)/HESSIAN_EIGENVALUE_TO_CM
    f3=field.cubic_cm/CM_PER_HARTREE*scale[:,None,None]*scale[None,:,None]*scale[None,None,:]
    f4=field.quartic_cm/CM_PER_HARTREE*scale[:,None,None,None]*scale[None,:,None,None]*scale[None,None,:,None]*scale[None,None,None,:]
    return SonicVPT2Input(np.diag((w/HESSIAN_EIGENVALUE_TO_CM)**2),f3,f4,np.eye(n),
        np.zeros_like(f3),np.zeros_like(f4),field.constant_shift_cm,
        quartic_coverage='semidiagonal' if field.quartic_coverage=='semidiagonal_model' else 'complete')


def gaussian_saddle_field_in_harmonic_frame(source, target, saddle, *, target_modes):
    """Transport verified real saddle F3/F4 into a positive target harmonic field.

    abs(source frequency) sets only the unit of each real source coordinate.
    The signed B3LYP quadratic is not used in the returned Hamiltonian. Source
    FI phases/signs must already have been resolved by read_gaussian_saddle_qff.
    No unstable source direction or its mixed derivatives are omitted here.
    """
    from matrix_gaussian.modes import gaussian_cartesian_modes_to_mass_weighted
    from .vibronic_variational import project_anharmonic_potential
    signed=np.asarray(saddle.signed_frequencies_cm,float)
    if (not np.allclose(source.harmonic_frequencies_cm,signed,rtol=0,atol=.05)
            or not np.array_equal(source.masses_amu,target.masses_amu)):
        raise ValueError('saddle source frequencies or atom/mass gauge differs')
    if saddle.diagnostics.get('coordinate_convention')!='real dimensionless q scaled with absolute harmonic frequency; signed quadratic retained':
        raise ValueError('verified real signed-mode derivative convention required')
    active=np.asarray(target_modes)
    if (active.ndim!=1 or active.dtype.kind not in 'iu' or not len(active)
            or len(np.unique(active))!=len(active) or np.any(active<0)
            or np.any(active>=len(target.harmonic_frequencies_cm))):
        raise ValueError('unique target mode indices required')
    masses=source.masses_amu
    bx=source.cartesian_coordinates_bohr-np.average(source.cartesian_coordinates_bohr,axis=0,weights=masses)
    rx=target.cartesian_coordinates_bohr-np.average(target.cartesian_coordinates_bohr,axis=0,weights=masses)
    u,_,vt=np.linalg.svd((rx*masses[:,None]).T@bx)
    rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    bm=gaussian_cartesian_modes_to_mass_weighted(source.normal_modes,masses)
    rm=gaussian_cartesian_modes_to_mass_weighted(target.normal_modes,masses).reshape(-1,len(masses),3)
    rm=np.einsum('mac,cd->mad',rm,rotation).reshape(len(signed),-1)
    overlap=bm@rm[active].T
    w=target.harmonic_frequencies_cm[active]
    transform=np.sqrt(abs(signed))[:,None]*overlap/np.sqrt(w)[None,:]
    # This container only supplies real derivative units to the existing
    # tensor transporter; its temporary quadratic is never evaluated.
    derivatives=SAMVPT2Field(abs(signed),saddle.cubic_cm,saddle.quartic_cm,quartic_coverage='semidiagonal_model')
    field=project_anharmonic_potential(derivatives,w,transform)
    return field,dict(source_signed_frequencies_cm=signed,target_modes_1based=(active+1).tolist(),
        source_from_target_dimensionless=transform,mass_weighted_mode_overlap=overlap,
        anharmonic_map_rank=int(np.linalg.matrix_rank(transform)),
        quadratic_frequencies_cm=w,quadratic_source='revDSD harmonic',
        absolute_source_frequencies_used_only_for_real_coordinate_units=True,
        negative_source_derivatives_included=True,source_diagnostics=saddle.diagnostics,
        new_electronic_calculations=False,
        derivative_geometry='existing B3LYP geometry, mass-weighted Eckart tangent transport')
