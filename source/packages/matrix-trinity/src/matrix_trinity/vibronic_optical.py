"""ECD, CPL, weak-field MCD and degenerate TPA/TPCD in native SATI.

Conventions are defined by explicit helicities and multipolar amplitudes, not
by software-dependent names of rotatory-strength or Faraday invariants.
All property gradients use dimensionless initial-state normal coordinates.
"""
from functools import lru_cache
import numpy as np
from .vibronic_observables import harmonic_operator_spectrum
from .vibronic_harmonic import HCK_K_CM, PHASE_PER_CM_FS


def _property(value, gradient, shape, n):
    value = np.asarray(value, complex)
    gradient = np.zeros(shape+(n,), complex) if gradient is None else np.asarray(gradient, complex)
    if value.shape != shape or gradient.shape != shape+(n,) or not all(np.all(np.isfinite(x)) for x in (value, gradient)):
        raise ValueError(f'finite property of shape {shape} and gradient of shape {shape+(n,)} required')
    return value.reshape(-1), gradient.reshape(-1, n)


def _outer(left, right=None):
    return np.outer(left, (left if right is None else right).conj())


@lru_cache(maxsize=1)
def _orientations():
    """Normalized SO(3) quadrature exact for the rank <= 6 polynomials used here."""
    x, weights = np.polynomial.legendre.leggauss(4)
    rotations = []; quadrature = []
    def rz(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    for cosine, weight in zip(x, weights, strict=True):
        sine = np.sqrt(1-cosine*cosine)
        ry = np.array([[cosine, 0, sine], [0, 1., 0], [-sine, 0, cosine]])
        for alpha in np.arange(8)*np.pi/4:
            for gamma in np.arange(8)*np.pi/4:
                rotations.append(rz(alpha) @ ry @ rz(gamma))
                quadrature.append(weight/128)
    rotations = np.array(rotations)
    # R maps molecular to laboratory coordinates; photon propagates along +z.
    n = rotations[:, 2, :]
    linear = rotations[:, 0, :].astype(complex)
    plus = (rotations[:, 0, :]+1j*rotations[:, 1, :])/np.sqrt(2)
    return np.array(quadrature), n, linear, plus, plus.conj()


def _run(model, properties, forms, labels, **controls):
    assignment_path=controls.pop('selection_assignment_report_path',None)
    constants = np.concatenate([p[0] for p in properties])
    gradients = np.concatenate([p[1] for p in properties])
    result = harmonic_operator_spectrum(model, constants, np.asarray(forms), gradients=gradients, **controls)
    result['observable_names'] = labels
    if result.get('assignment_report') is not None:
        result['assignment_report']['observable_names']=labels
        if assignment_path is not None:
            from .vibronic_assignment_report import save_targeted_assignment_report
            save_targeted_assignment_report(result['assignment_report'],assignment_path)
    elif assignment_path is not None:
        raise ValueError('assignment report requires targeted TI selection')
    result['conventions'] = dict(helicity='e_plus=(x+i*y)/sqrt(2), propagation +z; differences plus-minus',
        strengths='quadratic forms in the supplied electric-equivalent atomic-unit amplitudes; not molar ellipticities or GM cross sections',
        electronic_response='supplied electronic tensors and derivatives; nuclear FC/HT dressing is calculated here')
    return result


def harmonic_circular_spectrum(model, electric_dipole_au, magnetic_dipole_over_c_au, *,
        electric_gradient=None, magnetic_gradient=None, spectroscopy='ECD', **controls):
    """ECD absorption or CPL emission, retaining E1/M1 HT interference.

    Magnetic input is m_SI/c in electric-dipole atomic units (e a0).
    For magnetic atomic units e*hbar/m_e multiply by alpha, or alpha/2 for
    Bohr magnetons. R=Im[<i|mu|f>.<i|m/c|f>*]. Emission reverses/conjugates
    BOTH operators before contraction. With the stated helicities g=4R/(D+M)
    for absorption and -4R/(D+M) for emission (opposite photon vertex).
    """
    if spectroscopy not in ('ECD', 'CPL'):
        raise ValueError('spectroscopy must be ECD or CPL')
    if 'kind' in controls:
        raise ValueError('ECD/CPL determine absorption/emission; do not override kind')
    n = len(model.initial_frequencies_cm)
    electric = _property(electric_dipole_au, electric_gradient, (3,), n)
    magnetic = _property(magnetic_dipole_over_c_au, magnetic_gradient, (3,), n)
    d = np.diag([1.]*3+[0.]*3); m = np.eye(6)-d
    r = np.zeros((6, 6), complex)
    r[:3, 3:] = -.5j*np.eye(3); r[3:, :3] = .5j*np.eye(3)
    result = _run(model, [electric, magnetic], [d, m, r], ['electric_strength', 'magnetic_strength', 'rotatory_strength'],
                  kind='emission' if spectroscopy == 'CPL' else 'absorption', **controls)
    key = 'strengths' if result['method'] == 'TI' else 'densities_per_cm'
    values = result[key]; denominator = values[0]+values[1]
    # Ratios at unresolved/zero intensity carry an explicit validity mask.
    mask = denominator > 1e-12*max(float(np.max(abs(denominator), initial=0.)), np.finfo(float).tiny)
    result['dissymmetry_valid'] = mask
    numerator = (4 if spectroscopy == 'ECD' else -4)*values[2]
    violation = mask & (abs(numerator) > 2*denominator*(1+1e-7))
    result['diagnostics']['dissymmetry_bound_violations'] = int(np.count_nonzero(violation))
    mask &= ~violation
    result['dissymmetry_valid'] = mask
    result['dissymmetry'] = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=mask)
    result['spectroscopy'] = spectroscopy
    result['conventions']['dissymmetry'] = '2(I_plus-I_minus)/(I_plus+I_minus): +4R/(D+M) absorption, -4R/(D+M) emission, R in the oriented <i|operator|f> order'
    result['conventions']['magnetic_gauge'] = 'electric and magnetic moments must have the same electronic phase, frame, gauge and origin; no electronic gauge correction is inferred'
    result['conventions']['CPL'] = 'strength representation; multiply emission components by positive photon wavenumber cubed for photon rates; g is unchanged'
    return result


def harmonic_two_photon_spectrum(model, electric_tensor, *, electric_gradient=None,
        magnetic_tensor_over_c=None, magnetic_gradient=None, quadrupole_tensor=None,
        quadrupole_gradient=None, photon_wavenumber_cm=None, tensor_origin_bohr=None,
        gauge=None, spectroscopy='TPA', **controls):
    """Degenerate co-propagating TPA/TPCD, exact isotropic orientation average.

    Tensor conventions are defined by the *total symmetrized* amplitude
    A=S_ij e_i e_j + U_ij e_i (n cross e)_j
      + i*k*Q_ijk e_i e_j n_k/6.
    U includes m/c, Q uses traceless electric quadrupoles; both include ALL
    photon placements/time orderings. k is in bohr^-1. TPCD retains only the
    first-order E1E1--E1M1 and E1E1--E1E2 interference. Raw single-placement
    response tensors must be assembled before calling. No pure-M1/E2 squares.
    Tensors are evaluated at the supplied fixed photon frequency; broad vibronic
    bands do not implicitly recompute electronic denominators at each line.
    """
    if spectroscopy not in ('TPA', 'TPCD'):
        raise ValueError('spectroscopy must be TPA or TPCD')
    if controls.get('kind', 'absorption') != 'absorption':
        raise ValueError('TPA/TPCD require absorption')
    nmode = len(model.initial_frequencies_cm)
    electric = _property(electric_tensor, electric_gradient, (3, 3), nmode)
    s = electric[0].reshape(3, 3); ds = electric[1].reshape(3, 3, nmode)
    if not np.allclose(s, s.T, atol=1e-12, rtol=1e-12) or not np.allclose(ds, ds.swapaxes(0, 1), atol=1e-12, rtol=1e-12):
        raise ValueError('degenerate electric two-photon tensor and gradients must be symmetric')
    weights, directions, linear, plus, minus = _orientations()
    count = 9 if spectroscopy == 'TPA' else 45
    forms = []
    amplitudes = []
    for polarization in (linear, plus, minus):
        v = np.zeros((len(weights), count), complex)
        v[:, :9] = np.einsum('ti,tj->tij', polarization, polarization).reshape(-1, 9)
        amplitudes.append(v)
        forms.append(np.einsum('t,ta,tb->ab', weights, v, v.conj()))
    properties = [electric]
    labels = ['linear_TPA', 'plus_TPA', 'minus_TPA']
    if spectroscopy == 'TPCD':
        if magnetic_tensor_over_c is None or quadrupole_tensor is None:
            raise ValueError('TPCD requires BOTH mixed magnetic and quadrupole response tensors, including explicit zeros')
        if (photon_wavenumber_cm is None or not np.isfinite(photon_wavenumber_cm) or photon_wavenumber_cm <= 0
                or gauge != 'length' or tensor_origin_bohr is None
                or np.asarray(tensor_origin_bohr).shape != (3,) or not np.all(np.isfinite(tensor_origin_bohr))):
            raise ValueError('TPCD requires positive photon frequency, gauge="length" and a common finite origin in bohr')
        magnetic = _property(magnetic_tensor_over_c, magnetic_gradient, (3, 3), nmode)
        quadrupole = _property(quadrupole_tensor, quadrupole_gradient, (3, 3, 3), nmode)
        q = quadrupole[0].reshape(3, 3, 3); dq = quadrupole[1].reshape(3, 3, 3, nmode)
        if (not np.allclose(q, q.swapaxes(1, 2)) or not np.allclose(dq, dq.swapaxes(1, 2))
                or not np.allclose(np.trace(q, axis1=1, axis2=2), 0, atol=1e-12)
                or not np.allclose(np.trace(dq, axis1=1, axis2=2), 0, atol=1e-12)):
            raise ValueError('Q_ijk requires symmetric traceless quadrupole indices j,k, including gradients')
        from scipy.constants import physical_constants
        wavevector = 2*np.pi*photon_wavenumber_cm*100*physical_constants['Bohr radius'][0]
        mixed_forms = []
        for polarization, electric_amplitude in zip((plus, minus), amplitudes[1:], strict=True):
            mixed = np.zeros_like(electric_amplitude)
            mixed[:, 9:18] = np.einsum('ti,tj->tij', polarization, np.cross(directions, polarization)).reshape(-1, 9)
            mixed[:, 18:] = (1j*wavevector/6*np.einsum('ti,tj,tk->tijk', polarization, polarization, directions)).reshape(-1, 27)
            cross = np.einsum('t,ta,tb->ab', weights, electric_amplitude, mixed.conj())
            mixed_forms.append(cross+cross.conj().T)
        difference = mixed_forms[0]-mixed_forms[1]
        magnetic_form = np.zeros_like(difference)
        magnetic_form[:9, 9:18] = difference[:9, 9:18]
        magnetic_form[9:18, :9] = difference[9:18, :9]
        quadrupole_form = difference-magnetic_form
        forms.extend([mixed_forms[0], mixed_forms[1], magnetic_form, quadrupole_form, difference])
        labels.extend(['plus_multipole_interference', 'minus_multipole_interference',
                       'TPCD_magnetic', 'TPCD_quadrupole', 'TPCD_plus_minus'])
        properties.extend([magnetic, quadrupole])
    elif any(x is not None for x in (magnetic_tensor_over_c, magnetic_gradient, quadrupole_tensor, quadrupole_gradient,
                                     photon_wavenumber_cm, tensor_origin_bohr, gauge)):
        raise ValueError('mixed multipole controls require spectroscopy="TPCD"')
    result = _run(model, properties, forms, labels, **controls)
    result['spectroscopy'] = spectroscopy
    result['conventions']['tensor_order'] = 'effective absorption amplitudes <final|response|initial>, defined by A=S e e + U e(n cross e) + i*k*Q e e n/6'
    if spectroscopy == 'TPCD':
        result['conventions'].update(gauge=gauge, tensor_origin_bohr=list(tensor_origin_bohr),
            photon_wavenumber_cm=photon_wavenumber_cm, origin_independence='not repaired or guaranteed for inconsistent/incomplete electronic tensors',
            multipole_order='interference through first order in magnetic/quadrupole amplitudes')
    return result


def _mcd_profile(result, temperature_K, controls):
    if result['method'] == 'TD':
        # Analytic frequency derivative in Fourier space, not finite differences.
        times = result['times_fs']; tau = PHASE_PER_CM_FS*times
        origin = result['origin_cm']
        signal = result['correlations'][0]*np.exp(1j*origin*tau)
        sigma = controls.get('gaussian_sigma_cm', 50.); gamma = controls.get('lorentz_hwhm_cm', 0.)
        signal *= 1j*tau*np.exp(-gamma*tau-.5*(sigma*tau)**2)
        from scipy import fft
        step = PHASE_PER_CM_FS*controls.get('time_step_fs', .25)
        derivative = fft.fftshift(fft.hfft(signal.conj(), n=2*(len(times)-1)))*step/(2*np.pi)
        density = result['densities_per_cm']
        result['MCD_density_per_T_per_cm'] = derivative+density[1]+(density[2]/temperature_K if temperature_K else 0.)
        result['A_derivative_density'] = derivative
    result['spectroscopy'] = 'MCD'
    result['conventions']['MCD'] = 'linear field, plus-minus; A multiplies frequency derivative, B ordinary profile, C/T thermal profile'
    result['conventions']['tensor_order'] = 'absorption amplitudes <final|mu|initial>; response derivative uses the same phase and index order'
    return result


def harmonic_mcd_spectrum(model, electric_dipole_au, electric_field_response_per_T, *,
        electric_gradient=None, response_gradient=None, transition_zeeman_cm_per_T=(0., 0., 0.),
        lower_zeeman_cm_per_T=(0., 0., 0.), electronic_population=1., temperature_K=298.15, **controls):
    """One resolved Zeeman branch of isotropic, longitudinal weak-field MCD.

    dmu_i/dB_j is the electronic response per tesla. Supply all branches of
    degenerate manifolds in their first-order Zeeman eigenbasis and sum their
    population-weighted spectra. No divisions by near-degenerate level gaps.
    Delta I/B = A*dL/dnu + B*L + C/T*L, for nu=observation-transition.
    C uses the lower-state Zeeman slope relative to the ensemble mean; that
    subtraction enforces normalization of electronic populations.
    Vibronic FC/HT is general; field derivatives of force fields are not included.
    """
    if 'kind' in controls:
        raise ValueError('MCD is absorption')
    slopes = np.asarray([transition_zeeman_cm_per_T, lower_zeeman_cm_per_T], float)
    if (slopes.shape != (2, 3) or not np.all(np.isfinite(slopes))
            or not np.isfinite(electronic_population) or not 0 <= electronic_population <= 1
            or not np.isfinite(temperature_K) or temperature_K < 0
            or (temperature_K == 0 and np.any(slopes[1] != 0))):
        raise ValueError('finite Zeeman slopes, population in [0,1] and valid weak-field temperature required')
    n = len(model.initial_frequencies_cm)
    electric = _property(electric_dipole_au, electric_gradient, (3,), n)
    response = _property(electric_field_response_per_T, response_gradient, (3, 3), n)
    weights, directions, _, plus, minus = _orientations()
    a = np.zeros((12, 12), complex); b = a.copy(); c = a.copy()
    for polarization, sign in ((plus, 1), (minus, -1)):
        v = np.zeros((len(weights), 12), complex); v[:, :3] = polarization
        derivative = np.zeros_like(v)
        derivative[:, 3:] = np.einsum('ti,tj->tij', polarization, directions).reshape(-1, 9)
        norm = np.einsum('ta,tb->tab', v, v.conj())
        a -= sign*np.einsum('t,t,tab->ab', weights, directions @ slopes[0], norm)
        c -= sign*HCK_K_CM*np.einsum('t,t,tab->ab', weights, directions @ slopes[1], norm)
        cross = np.einsum('t,ta,tb->ab', weights, v, derivative.conj())
        b += sign*(cross+cross.conj().T)
    result = _run(model, [electric, response], electronic_population*np.array([a, b, c]),
                  ['A_derivative_coefficient', 'B_strength_per_T', 'C_strength_K_per_T'],
                  temperature_K=temperature_K, **controls)
    result['conventions']['electronic_branches'] = 'resolved branch; sum all electronic branches with their normalized populations'
    return _mcd_profile(result, temperature_K, controls)


def harmonic_mcd_manifold_spectrum(model, electric_dipole_au, electric_field_response_per_T, *,
        lower_zeeman_matrices_cm_per_T, upper_zeeman_matrices_cm_per_T,
        electric_gradient=None, response_gradient=None, electronic_population=1.,
        lower_ensemble_mean_zeeman_cm_per_T=None, temperature_K=298.15, **controls):
    """Basis-invariant MCD for complete degenerate lower/upper manifolds.

    Dipoles have shape (n_lower,n_upper,3), field response adds a final magnetic
    Cartesian index. Zeeman matrices (3,n,n) are Hermitian, including spin/SOC.
    Stored dipoles [g,e,i]=<upper_e|mu_i|lower_g>. If D_i has the usual
    (upper,lower) matrix order, off-diagonal terms enter as
    A=-Tr(D† Z_u D-Z_l D† D), C=-hc/k Tr((Z_l-<Z>) D† D),
    with helicity/orientation contraction.
    No arbitrary choice of eigenvectors or small electronic denominators.
    Each manifold shares one harmonic surface; distinct vibronic surfaces must
    be separate transitions. Electronic B-response is supplied from outside the
    included degenerate subspaces, without their singular perturbation terms.
    """
    if 'kind' in controls: raise ValueError('MCD is absorption')
    value = np.asarray(electric_dipole_au)
    if value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1:
        raise ValueError('manifold dipoles require shape (n_lower,n_upper,3)')
    ng, ne = value.shape[:2]; n = len(model.initial_frequencies_cm)
    electric = _property(value, electric_gradient, (ng, ne, 3), n)
    response = _property(electric_field_response_per_T, response_gradient, (ng, ne, 3, 3), n)
    zl = np.asarray(lower_zeeman_matrices_cm_per_T, complex)
    zu = np.asarray(upper_zeeman_matrices_cm_per_T, complex)
    if (zl.shape != (3, ng, ng) or zu.shape != (3, ne, ne)
            or not all(np.all(np.isfinite(z)) and np.allclose(z, z.conj().swapaxes(1, 2), atol=1e-12, rtol=1e-12) for z in (zl, zu))):
        raise ValueError('finite Hermitian Cartesian Zeeman matrices required')
    mean = np.trace(zl, axis1=1, axis2=2).real/ng if lower_ensemble_mean_zeeman_cm_per_T is None else np.asarray(lower_ensemble_mean_zeeman_cm_per_T, float)
    if mean.shape != (3,) or not np.all(np.isfinite(mean)):
        raise ValueError('finite Cartesian ensemble mean Zeeman slope required')
    centered = zl-mean[:, None, None]*np.eye(ng)
    if (not np.isfinite(temperature_K) or temperature_K < 0
            or (temperature_K == 0 and np.any(abs(centered) > 1e-14))
            or not np.isfinite(electronic_population) or not 0 <= electronic_population <= 1):
        raise ValueError('valid population and positive temperature for a paramagnetic C response required')
    weights, directions, _, plus, minus = _orientations()
    helicity = np.einsum('ti,tj->tij', plus, plus.conj())-np.einsum('ti,tj->tij', minus, minus.conj())
    k = np.einsum('t,tij,ta->ija', weights, helicity, directions)
    size = 3*ng*ne
    if 3*(4*size)**2*16 > controls.get('memory_budget_bytes', 256*1024**2):
        raise MemoryError('MCD manifold observable forms exceed declared memory budget')
    forms = np.zeros((3, 4*size, 4*size), complex)
    # Axes before flattening: lower,upper,electric,lower*,upper*,electric*.
    a = (-np.einsum('gh,ija,afe->geihfj', np.eye(ng), k, zu)
         +np.einsum('ef,ija,agh->geihfj', np.eye(ne), k, zl))
    c = -HCK_K_CM*np.einsum('ef,ija,agh->geihfj', np.eye(ne), k, centered)
    forms[0, :size, :size] = a.reshape(size, size)
    forms[2, :size, :size] = c.reshape(size, size)
    cross = np.einsum('gh,ef,ija->geihfja', np.eye(ng), np.eye(ne), k).reshape(size, 3*size)
    forms[1, :size, size:] = cross
    forms[1, size:, :size] = cross.conj().T
    result = _run(model, [electric, response], forms*electronic_population/ng,
                  ['A_derivative_coefficient', 'B_strength_per_T', 'C_strength_K_per_T'],
                  temperature_K=temperature_K, **controls)
    result['conventions']['electronic_manifolds'] = dict(lower_dimension=ng, upper_dimension=ne,
        population=electronic_population, treatment='degenerate-subspace traces; no Zeeman eigenvector ordering',
        ensemble_mean_zeeman_cm_per_T=mean.tolist())
    return _mcd_profile(result, temperature_K, controls)

__all__ = ['harmonic_circular_spectrum', 'harmonic_two_photon_spectrum', 'harmonic_mcd_spectrum', 'harmonic_mcd_manifold_spectrum']
