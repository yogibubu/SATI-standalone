"""A fixed-parent-mode composite reference with rigid-motion completion."""
import numpy as np


def replace_fixed_mode_harmonics(coordinates_angstrom, masses_amu, modes_mw,
                                 target_frequencies_cm, normal_derivatives):
    """Replace F2 and preserve full physical normal F3/F4/... in the old basis.

    Rebuild the Cartesian rigid-rotation completion with the *new* Hessian
    before any isotope transformation. Keeping old Cartesian rotational
    derivatives after changing F2 would violate the rotation Ward identities.
    Normal derivatives use hartree/(sqrt(amu)*bohr)**rank.
    """
    from matrix_qm import HESSIAN_EIGENVALUE_TO_CM
    from .isotope_field import cartesian_derivatives_from_normal_field
    w = np.asarray(target_frequencies_cm, float)
    m = np.asarray(masses_amu, float)
    modes = np.asarray(modes_mw, float)
    if (w.ndim != 1 or modes.shape != (len(w),len(m),3)
            or np.any(w <= 0) or np.any(m <= 0)
            or not all(np.all(np.isfinite(v)) for v in (w,m,modes))):
        raise ValueError('Finite positive frequencies/masses and matching complete parent modes required')
    dual = modes.reshape(len(w),-1)*np.sqrt(np.repeat(m,3))[None,:]
    hessian = dual.T@np.diag((w/HESSIAN_EIGENVALUE_TO_CM)**2)@dual
    return cartesian_derivatives_from_normal_field(normal_derivatives, modes, m,
        coordinates_angstrom=coordinates_angstrom, hessian_cartesian=hessian)
