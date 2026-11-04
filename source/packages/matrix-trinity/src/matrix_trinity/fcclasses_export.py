"""Export an existing harmonic operator to FCclasses3's IMPLICIT interface.

No geometry, mode, property or energy is recomputed. FCclasses uses
Q1 = J Q2 + K with mass-weighted atomic coordinates; TRINITY uses the
inverse map and dimensionless initial-coordinate dipole derivatives.
"""
from pathlib import Path
import json
import numpy as np
from matrix_gf.harmonic import CM_PER_HARTREE


def export_fcclasses_harmonic(model, directory):
    """Write real electric-dipole AH data, preserving the nonorthogonal map.

    Complex/momentum operators have no representation in this interface and
    are rejected. Input files alone do not choose numerical spectral settings.
    """
    mu=model.transition_dipole_au
    derivative=model.transition_gradient_dimensionless_au
    if (np.any(mu.imag) or np.any(derivative.imag)
            or np.any(model.transition_momentum_dimensionless_au)):
        raise ValueError('FCclasses interface requires a real position-only electric dipole')
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    j=np.linalg.inv(model.mode_map)
    k=-np.sqrt(CM_PER_HARTREE)*(j@(model.final_shift_dimensionless/
        np.sqrt(model.final_frequencies_cm)))
    grad=derivative.real*np.sqrt(model.initial_frequencies_cm/CM_PER_HARTREE)[None,:]
    arrays={'freq1.dat':model.initial_frequencies_cm,'freq2.dat':model.final_frequencies_cm,
        'dusch.dat':j.ravel(),'disp.dat':k,
        'eldip.dat':np.vstack([mu.real,mu.real,grad.T])}
    for name,value in arrays.items():np.savetxt(out/name,value,fmt='%.17e')
    metadata=dict(schema='matrix.trinity.fcclasses_implicit.v1',modes=len(k),
        map_convention='Q1 = J Q2 + K; Q in bohr sqrt(electron mass)',
        dipole_derivatives='initial mass-weighted atomic normal coordinates',
        electronic_gap_eV=float(model.electronic_gap_cm/8065.544005),
        origin_cm=model.origin_cm,determinant_native_map=float(np.linalg.det(model.mode_map)),
        unused_final_geometry_dipole='copied initial value; not a linearization validation',
        altered_frequencies=False,orthogonalized_map=False)
    (out/'export.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata
