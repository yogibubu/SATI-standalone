"""Gaussian ingestion for native two-surface ESALC/GF spectroscopy.

Electronic data ingestion is isolated here; spectra consume backend-neutral
HarmonicVibronicModel objects. No automatic absolute-value repair of Hessians.
"""
from pathlib import Path
import hashlib
import numpy as np
from matrix_gaussian.fchk import _read_fchk_blocks, lower_to_symmetric, BOHR_TO_ANGSTROM
from matrix_smith import (build_gic_b_matrix, evaluate_gic_values,
                          sonic_definition_identity_sha256)
from .vibronic_harmonic import ESALCHarmonicState, harmonic_vibronic_from_esalc
from matrix_gf.harmonic import CM_PER_HARTREE


def read_gaussian_vibronic_data(path):
    """Read geometry, force field and length-gauge ETran E1/M1 properties.

    Accept the explicitly zero-padded mass arrays written by reform_fchk.
    ETran derivatives follow the 16*nstate + 48 header, then 16 per Cartesian
    coordinate. Derivatives of non-target states are never fabricated.
    """
    path = Path(path); blocks = _read_fchk_blocks(path)
    numbers = np.asarray(blocks['Atomic numbers'], int); n = len(numbers)
    masses = np.asarray(blocks['Real atomic weights'], float)
    if len(masses) != n:
        if len(masses) != 3*n or np.any(masses[n:] != 0):
            raise ValueError('unsupported atomic mass array; padding must be explicit zeros')
        masses = masses[:n]
    if masses.shape != (n,) or np.any(masses <= 0) or not np.all(np.isfinite(masses)):
        raise ValueError('positive finite isotope masses required')
    coordinates = np.asarray(blocks['Current cartesian coordinates'], float).reshape(n, 3)
    gradient = np.asarray(blocks['Cartesian Gradient'], float)
    hessian = lower_to_symmetric(blocks['Cartesian Force Constants'])
    if gradient.shape != (3*n,) or hessian.shape != (3*n, 3*n):
        raise ValueError('incomplete Cartesian force field')
    result = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        atomic_numbers=numbers, masses_amu=masses, coordinates_bohr=coordinates,
        gradient_hartree_per_bohr=gradient, hessian_hartree_per_bohr2=hessian,
        energy_hartree=float(blocks['Total Energy']), electronic=None)
    if 'ETran state values' in blocks:
        scalar = np.asarray(blocks['ETran scalars'], int); count, target = int(scalar[0]), int(scalar[4])
        values = np.asarray(blocks['ETran state values'], float)
        if not 1 <= target <= count or len(values) < 16*count:
            raise ValueError('incomplete ETran state table')
        states = values[:16*count].reshape(count, 16)
        derivatives = magnetic_derivatives = None
        if len(values) == 16*count+48+16*3*n:
            cartesian = values[16*count+48:].reshape(3*n, 16)
            derivatives = cartesian[:, 1:4].T.copy()
            magnetic_derivatives = cartesian[:, 7:10].T.copy()
        # Gaussian prints the real angular-generator matrix elements.  In the
        # <i|operator|f> convention used by the optical bank, M1/c is +i alpha/2
        # times this record (Bohr-magneton units). Thus R=-alpha/2 mu.M.
        from scipy.constants import fine_structure
        magnetic_factor = .5j*fine_structure
        result['electronic'] = dict(target_state=target, states=states,
            transition_dipole_au=states[target-1, 1:4].copy(),
            magnetic_dipole_over_c_au=magnetic_factor*states[target-1, 7:10],
            electric_gradient_cartesian=derivatives,
            magnetic_gradient_cartesian=None if magnetic_derivatives is None else magnetic_factor*magnetic_derivatives,
            derivative_zero_cartesian_columns=None if derivatives is None else int(np.count_nonzero(np.all(derivatives == 0, axis=0))),
            derivative_maximum_absolute=None if derivatives is None else float(np.max(abs(derivatives))))
    for value in (coordinates, gradient, hessian):
        if not np.all(np.isfinite(value)): raise ValueError('nonfinite Gaussian force-field data')
    return result


def align_gaussian_vibronic_data(data, reference):
    """Proper mass-weighted Eckart/Kabsch alignment, transforming all tensors."""
    if (not np.array_equal(data['atomic_numbers'], reference['atomic_numbers'])
            or not np.array_equal(data['masses_amu'], reference['masses_amu'])):
        raise ValueError('identical atom ordering and isotope masses required')
    masses = data['masses_amu']; x = data['coordinates_bohr']; y = reference['coordinates_bohr']
    cx = np.average(x, axis=0, weights=masses); cy = np.average(y, axis=0, weights=masses)
    u, _, vh = np.linalg.svd(((x-cx)*masses[:, None]).T @ (y-cy))
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vh)]) @ vh
    transform = np.kron(np.eye(len(x)), rotation.T)
    out = dict(data, coordinates_bohr=(x-cx) @ rotation+cy,
        gradient_hartree_per_bohr=transform @ data['gradient_hartree_per_bohr'],
        hessian_hartree_per_bohr2=transform @ data['hessian_hartree_per_bohr2'] @ transform.T)
    if data['electronic'] is not None:
        electronic = dict(data['electronic']); derivative = electronic['electric_gradient_cartesian']
        electronic['transition_dipole_au'] = rotation.T @ electronic['transition_dipole_au']
        if derivative is not None:
            electronic['electric_gradient_cartesian'] = rotation.T @ derivative @ transform.T
        if 'magnetic_dipole_over_c_au' in electronic:
            electronic['magnetic_dipole_over_c_au'] = rotation.T @ electronic['magnetic_dipole_over_c_au']
            magnetic_derivative = electronic['magnetic_gradient_cartesian']
            if magnetic_derivative is not None:
                electronic['magnetic_gradient_cartesian'] = rotation.T @ magnetic_derivative @ transform.T
        out['electronic'] = electronic
    out['alignment'] = dict(rotation=rotation, mass_weighted_rmsd_bohr=float(np.sqrt(np.average(np.sum((out['coordinates_bohr']-y)**2, axis=1), weights=masses))))
    return out


def gaussian_force_field_in_esalc(data, definition, *, curvature_step_bohr=1e-4):
    """Transform through a frozen SMITH chart, including gradient*curvature.

    B-prime is obtained by a central difference of existing analytic B rows.
    This explicit step is recorded for refinement; it is not an analytic
    B-prime claim. The Wilson mass-metric right inverse fixes the tangent gauge.
    """
    if not np.isfinite(curvature_step_bohr) or curvature_step_bohr <= 0:
        raise ValueError('positive finite Cartesian curvature step required')
    xyz = data['coordinates_bohr']*BOHR_TO_ANGSTROM
    def b_matrix(coords):
        return np.asarray(build_gic_b_matrix(definition, coordinates_angstrom=coords).rows)*BOHR_TO_ANGSTROM
    b = b_matrix(xyz); masses = np.repeat(data['masses_amu'], 3)
    metric = (b/masses[None, :]) @ b.T
    if len(b) != xyz.size-6 or np.linalg.matrix_rank(b) != len(b):
        raise ValueError('a complete nonredundant nonlinear-molecule SMITH chart is required')
    back = (b.T/masses[:, None]) @ np.linalg.solve(metric, np.eye(len(b)))
    gradient = back.T @ data['gradient_hartree_per_bohr']
    curvature = np.empty_like(data['hessian_hartree_per_bohr2'])
    for column in range(xyz.size):
        plus = xyz.copy(); minus = xyz.copy()
        plus.flat[column] += curvature_step_bohr*BOHR_TO_ANGSTROM
        minus.flat[column] -= curvature_step_bohr*BOHR_TO_ANGSTROM
        curvature[:, column] = gradient @ ((b_matrix(plus)-b_matrix(minus))/(2*curvature_step_bohr))
    asymmetry = float(np.max(abs(curvature-curvature.T)))
    curvature = (curvature+curvature.T)/2
    force = back.T @ (data['hessian_hartree_per_bohr2']-curvature) @ back
    force = (force+force.T)/2
    position = evaluate_gic_values(definition, coordinates_angstrom=xyz)
    state = ESALCHarmonicState(force, metric, position, sonic_definition_identity_sha256(definition),
                               gradient_hartree=gradient, label=Path(data['path']).stem)
    centered = xyz-np.average(xyz, axis=0, weights=data['masses_amu'])
    rotations = np.column_stack([np.cross(np.tile(axis, (len(xyz), 1)), centered).ravel() for axis in np.eye(3)])
    rotations = np.linalg.qr(rotations)[0]
    rotation_leak = float(np.linalg.norm(b @ rotations)/np.linalg.norm(b))
    frozen = any(p.function in {'SOOP', 'SYCART'} for p in definition.primitives)
    return state, back, dict(curvature_step_bohr=curvature_step_bohr,
        curvature_cartesian_asymmetry=asymmetry,
        frozen_cartesian_primitives=frozen, rigid_rotation_relative_leakage=rotation_leak,
        reference_displacement_angstrom=float(np.linalg.norm(xyz-np.asarray(definition.reference_coordinates_angstrom))),
        gradient_projection_residual=float(np.linalg.norm(data['gradient_hartree_per_bohr']-b.T @ gradient)),
        right_inverse_error=float(np.max(abs(b @ back-np.eye(len(b))))),
        wilson_metric_condition=float(np.linalg.cond(metric)), gradient_norm_hartree_per_bohr=float(np.linalg.norm(data['gradient_hartree_per_bohr'])))


def gaussian_esalc_vibronic_model(initial_data, final_data, definition, *, herzberg_teller=False,
                                  property_data=None, curvature_step_bohr=1e-4,
                                  allow_frozen_cartesian_chart=False):
    """Construct the native relaxed local harmonic model with an audited chart.

    AH-like local fields retain and relax residual optimization gradients.
    No unstable mode is dropped or reflected. HT must be present and complete;
    exact-zero Cartesian derivative columns fail closed (symmetry-reduced data).
    """
    chart_reference = dict(initial_data, coordinates_bohr=np.asarray(definition.reference_coordinates_angstrom)/BOHR_TO_ANGSTROM)
    initial_data = align_gaussian_vibronic_data(initial_data, chart_reference)
    final_data = align_gaussian_vibronic_data(final_data, initial_data)
    frozen = any(p.function in {'SOOP', 'SYCART'} for p in definition.primitives)
    if (frozen and not allow_frozen_cartesian_chart
            and final_data['alignment']['mass_weighted_rmsd_bohr'] > 1e-8):
        raise ValueError('finite vibronic displacements require a curvilinear chart; use SMITH build_vibronic_esalc_chart, or explicitly allow the frozen Cartesian approximation')
    property_data = final_data if property_data is None else align_gaussian_vibronic_data(property_data, initial_data)
    left, _, dl = gaussian_force_field_in_esalc(initial_data, definition, curvature_step_bohr=curvature_step_bohr)
    right, _, dr = gaussian_force_field_in_esalc(final_data, definition, curvature_step_bohr=curvature_step_bohr)
    electronic = property_data['electronic']
    if electronic is None: raise ValueError('ETran transition data required')
    derivative = None
    reference = evaluate_gic_values(definition, coordinates_angstrom=property_data['coordinates_bohr']*BOHR_TO_ANGSTROM)
    if herzberg_teller:
        derivative = electronic['electric_gradient_cartesian']
        if derivative is None or np.any(np.all(derivative == 0, axis=0)):
            raise ValueError('HT requires complete Cartesian transition derivatives; symmetry-reduced/absent columns are unsupported')
        _, back, _ = gaussian_force_field_in_esalc(property_data, definition, curvature_step_bohr=curvature_step_bohr)
        derivative = derivative @ back
    model = harmonic_vibronic_from_esalc(left, right,
        electronic_gap_cm=(final_data['energy_hartree']-initial_data['energy_hartree'])*CM_PER_HARTREE,
        transition_dipole_au=electronic['transition_dipole_au'], transition_gradient_esalc_au=derivative,
        transition_reference_esalc=reference, pes_model='local')
    model.provenance.update(gaussian_source_sha256=[initial_data['sha256'], final_data['sha256'], property_data['sha256']],
        chart_backend=definition.backend, local_force_field_audit=[dl, dr], alignment=final_data['alignment'],
        potential_model='local quadratic fields at supplied geometries, including gradient relaxation',
        property_source_sha256=property_data['sha256'], herzberg_teller=bool(herzberg_teller),
        frozen_cartesian_chart_explicitly_allowed=bool(allow_frozen_cartesian_chart))
    return model, (left, right)


def gaussian_esalc_circular_operators(initial_state, property_data, definition,
                                     *, herzberg_teller=True, curvature_step_bohr=1e-4):
    """Map Gaussian E1/M1 operators into the SAME initial normal coordinates.

    Both operators retain their common electronic phase and origin. Only proper
    rotations align the supplied property frame; no gauge/origin correction is
    inferred. Derivatives, when requested, must be present for both operators.
    """
    reference = dict(property_data, coordinates_bohr=
        np.asarray(definition.reference_coordinates_angstrom)/BOHR_TO_ANGSTROM)
    data = align_gaussian_vibronic_data(property_data, reference)
    electronic = data['electronic']
    if electronic is None or 'magnetic_dipole_over_c_au' not in electronic:
        raise ValueError('Gaussian ETran electric AND magnetic transition records required')
    frequencies, normal_map, minimum, _ = initial_state.modes()
    position = evaluate_gic_values(definition,
        coordinates_angstrom=data['coordinates_bohr']*BOHR_TO_ANGSTROM)
    derivatives = []
    if herzberg_teller:
        _, back, _ = gaussian_force_field_in_esalc(data, definition,
            curvature_step_bohr=curvature_step_bohr)
        for name in ('electric_gradient_cartesian', 'magnetic_gradient_cartesian'):
            derivative = electronic[name]
            if derivative is None or np.asarray(derivative).shape != (3, data['coordinates_bohr'].size):
                raise ValueError('ECD HT requires complete electric AND magnetic derivatives')
            derivatives.append(derivative @ back)
    else:
        derivatives = [np.zeros((3,len(frequencies)), complex) for _ in range(2)]
    result = {}
    for label, name, derivative in zip(('electric', 'magnetic'),
            ('transition_dipole_au', 'magnetic_dipole_over_c_au'), derivatives):
        result[label] = np.asarray(electronic[name])+derivative@(minimum-position)
        result[label+'_gradient'] = derivative @ normal_map / np.sqrt(frequencies)[None,:]
    return result


__all__ = ['read_gaussian_vibronic_data', 'align_gaussian_vibronic_data',
           'gaussian_force_field_in_esalc', 'gaussian_esalc_vibronic_model',
           'gaussian_esalc_circular_operators']


def gaussian_cartesian_vibronic_model(initial_data, final_data, *, property_data,
                                      herzberg_teller=False):
    """Build Cartesian AH fields from the same imported electronic checkpoints.

    Properties must refer to the initial geometry. Both Hessians are
    independently projected onto their vibration-only Cartesian spaces;
    the existing proper mass-weighted alignment transports the final field.
    No orthogonalization or absolute-value replacement changes the mode map.
    """
    from matrix_qm import cartesian_normal_modes_from_hessian
    from matrix_gf.harmonic import HESSIAN_EIGENVALUE_TO_CM
    from .vibronic_harmonic import HarmonicVibronicModel

    final = align_gaussian_vibronic_data(final_data, initial_data)
    properties = align_gaussian_vibronic_data(property_data, initial_data)
    if not np.allclose(properties['coordinates_bohr'], initial_data['coordinates_bohr'],
                       rtol=0, atol=1e-6):
        # Independent ASCII checkpoints round Cartesian coordinates.
        raise ValueError('Optical properties must refer to the initial geometry')
    fields = (initial_data, final)
    modes = [cartesian_normal_modes_from_hessian(d['hessian_hartree_per_bohr2'],
             d['masses_amu'], d['coordinates_bohr']) for d in fields]
    if any(np.any(m.frequencies_cm <= 0) for m in modes):
        raise ValueError('Stable positive Cartesian AH modes required')
    wi, wf = (m.frequencies_cm for m in modes)
    if len(wi) != len(wf):
        raise ValueError('The two vibrational spaces must have equal dimension')
    ui, uf = (m.mass_weighted_modes.reshape(len(m.frequencies_cm), -1) for m in modes)
    masses = initial_data['masses_amu']
    scale = HESSIAN_EIGENVALUE_TO_CM / np.sqrt(CM_PER_HARTREE)
    mode_map = uf @ ui.T
    displacement = (np.sqrt(masses)[:, None] *
                    (initial_data['coordinates_bohr'] - final['coordinates_bohr'])).ravel()
    shift = np.sqrt(wf) * (uf @ displacement) / scale
    electronic = properties['electronic']
    if electronic is None:
        raise ValueError('An explicit initial-geometry optical operator is required')
    gradient = np.zeros((3, len(wi)))
    if herzberg_teller:
        derivative = electronic['electric_gradient_cartesian']
        if derivative is None:
            raise ValueError('HT requires explicit Cartesian transition derivatives')
        gradient = (derivative / np.sqrt(np.repeat(masses, 3))[None, :]) @ ui.T
        gradient *= scale / np.sqrt(wi)[None, :]
    return HarmonicVibronicModel(wi, wf, mode_map, shift,
        (final['energy_hartree'] - initial_data['energy_hartree']) * CM_PER_HARTREE,
        electronic['transition_dipole_au'], gradient,
        dict(coordinates='Cartesian AH', initial_sha256=initial_data['sha256'],
             final_sha256=final_data['sha256'], property_sha256=property_data['sha256']))
