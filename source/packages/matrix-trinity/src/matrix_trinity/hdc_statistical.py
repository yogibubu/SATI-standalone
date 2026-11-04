"""Fixed-action saddle sampling and separable rotational DOS convolution."""
import numpy as np

from matrix_chem import Phy, get_physical_constants


def sample_transition_state_wang_landau(normal_form, reactive_action, **controls):
    """Sample only stable transverse occupations at fixed real reactive action.

    This is a conditional channel partition function, not an integration over
    reactive action, a transmission coefficient, or a complete reaction rate.
    The QNF scalar barrier is included in the energy origin exactly once.
    """
    from .hdc_polynomial import transition_state_hdc_polynomial_energy
    from .quantum_wang_landau import sample_hdcpt2_quantum_wang_landau
    alpha = controls.pop("alpha", 1.)
    beta_cm2 = controls.pop("beta_cm2", 5e5)
    model = transition_state_hdc_polynomial_energy(normal_form, reactive_action,
                                                  alpha=alpha, beta_cm2=beta_cm2)
    sampled = sample_hdcpt2_quantum_wang_landau(model, **controls)
    return dict(energy_model=model, sampling=sampled, reactive_action=float(reactive_action),
                reactive_coordinate_is_bound=False, stable_mode_count=len(model.fundamentals_cm),
                conditional_channel_partition=True, reaction_rate_claimed=False)


def convolve_hdc_rotational_levels(vibrational, rotational_energies_cm, rotational_weights):
    """Compatibility call to the shared LOCK discrete-spectrum convolution."""
    from matrix_rovib.statistical_convolution import convolve_discrete_levels
    result=convolve_discrete_levels(vibrational, rotational_energies_cm, rotational_weights, sector="rotation")
    result["schema"]="matrix.trinity.hdc_rotational_convolution.v1"
    return result


def converge_hdc_thermal_domain(model, *, temperatures_K=(100.,200.,300.), **controls):
    """Production X/Y thermal domain, independent errors for Q, U and Cv.

    Uses SENTINEL's compiled bounded canonical reference. The returned status
    must be inspected; sampling and deterministic truncation are distinct.
    A fixed-action saddle polynomial enters the identical interface.
    """
    from matrix_sentinel.quantum_thermal import sample_integer_canonical_thermal
    temperatures = np.asarray(temperatures_K,float)
    if (temperatures.ndim != 1 or not len(temperatures) or not np.all(np.isfinite(temperatures))
            or np.any(temperatures <= 0) or not np.isfinite(model.zero_point_cm)):
        raise ValueError('finite zero point and positive finite temperatures required')
    c = get_physical_constants(); hc = c[Phy.PLANCK]*c[Phy.C_LIGHT]
    beta = hc/(c[Phy.BOLTZMANN]*temperatures)
    result = sample_integer_canonical_thermal(model.fundamentals_cm,model.x_matrix_cm,
        y_tensor_cm=getattr(model,'y_tensor_cm',None),inverse_temperatures=beta,**controls)
    gas = c[Phy.AVOGADRO]*c[Phy.BOLTZMANN]
    q,u,cv = result.partition_sums,result.mean_energy_cm,result.heat_capacity_over_k
    return dict(schema='matrix.trinity.hdc_thermal_convergence.v1',canonical=result,
        converged=result.diagnostics['converged'],temperatures_K=temperatures,
        excitation_partition=q,excitation_partition_standard_error=result.partition_standard_error,
        log_partition_vibrational=-beta*model.zero_point_cm+np.log(q),
        internal_energy_kj_mol=hc*c[Phy.AVOGADRO]*(model.zero_point_cm+u)/1000,
        entropy_j_mol_K=gas*(np.log(q)+beta*u),heat_capacity_j_mol_K=gas*cv,
        zero_point_counted_once=True,physical_domain_is_dissociation_limit=False)


def sample_hdc_windowed_wang_landau(model, *, temperatures_K=(100.,200.,300.), **controls):
    """Replica-exchange windows; frozen-only thermal/microcanonical measures."""
    from matrix_sentinel.quantum_rewl import sample_integer_windowed_wang_landau
    t = np.asarray(temperatures_K,float)
    if (t.ndim!=1 or not len(t) or not np.all(np.isfinite(t)) or np.any(t<=0)
            or not np.isfinite(model.zero_point_cm)):
        raise ValueError('finite zero point and positive finite temperatures required')
    c = get_physical_constants();hc=c[Phy.PLANCK]*c[Phy.C_LIGHT]
    beta=hc/(c[Phy.BOLTZMANN]*t)
    result=sample_integer_windowed_wang_landau(model.fundamentals_cm,model.x_matrix_cm,
        inverse_temperatures=beta,y_tensor_cm=getattr(model,'y_tensor_cm',None),**controls)
    q=result.partition_sums;u=result.first_energy_sums/q
    cv=beta**2*np.maximum(0.,result.second_energy_sums/q-u*u)
    gas=c[Phy.AVOGADRO]*c[Phy.BOLTZMANN]
    return dict(schema='matrix.trinity.hdc_windowed_wang_landau.v1',sampling=result,
        converged=result.diagnostics['converged'],temperatures_K=t,
        excitation_partition=q,excitation_partition_standard_error=result.partition_standard_error,
        log_partition_vibrational=-beta*model.zero_point_cm+np.log(q),
        internal_energy_kj_mol=hc*c[Phy.AVOGADRO]*(model.zero_point_cm+u)/1000,
        entropy_j_mol_K=gas*(np.log(q)+beta*u),heat_capacity_j_mol_K=gas*cv,
        microcanonical_properties=result.microcanonical_properties,zero_point_counted_once=True,
        adaptive_observations_used=False)
