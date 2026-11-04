"""Phase and property-sensitivity diagnostics for complex FC/HT amplitudes."""
import numpy as np


def optical_interference_audit(fc, ht, *, fractional_steps=(.01, .05)):
    """Keep absolute strengths; vary the HT scale, without fitting to spectra.

    Scaling probes conditioning with respect to the property model. It is not
    an estimated electronic-structure uncertainty. Simultaneous phase changes
    of FC and HT are gauge changes; flipping only HT changes the operator.
    """
    fc, ht = np.asarray(fc, complex), np.asarray(ht, complex)
    if fc.shape != ht.shape or fc.ndim != 2 or fc.shape[1] != 3 or not all(
            np.all(np.isfinite(a)) for a in (fc, ht)):
        raise ValueError('finite matching (line, three Cartesian components) amplitudes required')
    norm = lambda a: np.sum(abs(a)**2, axis=1)
    base = norm(fc+ht)
    if np.any(base <= 0):
        raise ValueError('nonzero physical line strengths required for relative sensitivities')
    phase = np.exp(.731j)
    rows = []
    for step in fractional_steps:
        if not np.isfinite(step) or not 0 < step < 1:
            raise ValueError('fractional sensitivity step must lie in (0,1)')
        rows.append(dict(fractional_step=float(step),
            HT_scale_minus_relative_change=(norm(fc+(1-step)*ht)/base-1).tolist(),
            HT_scale_plus_relative_change=(norm(fc+(1+step)*ht)/base-1).tolist()))
    incoherent = norm(fc)+norm(ht)
    return dict(FC=norm(fc).tolist(), HT=norm(ht).tolist(), FCHT=base.tolist(),
        interference=(base-incoherent).tolist(),
        FCHT_over_incoherent=(base/incoherent).tolist(),
        simultaneous_phase_max_relative_error=float(np.max(abs(norm(phase*fc+phase*ht)/base-1))),
        HT_only_sign_flip_strength=norm(fc-ht).tolist(),
        HT_scale_relative_condition_number=(2*np.sum(((fc+ht).conj()*ht).real, axis=1)/base).tolist(),
        sensitivity=rows,
        sensitivity_is_uncertainty_estimate=False)
