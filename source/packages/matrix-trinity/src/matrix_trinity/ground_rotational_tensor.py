"""Full body-frame rotational VPT2 correction from the modal metric."""
from __future__ import annotations
import numpy as np


def contract_modal_ground_rotational_tensor(frequencies_cm, cubic_cm,
                                            metric_first_cm,
                                            metric_second_diagonal_cm):
    """Contract Eq. 50 without discarding off-diagonal rotational entries.

    T=P.G.P/2, dimensionless harmonic normal coordinates, all coefficients
    in cm^-1. G' has shape (n+3,n+3,n), G'' the same shape with its final
    index denoting the repeated derivative direction. The result is the
    symmetric body-frame coefficient of J_a J_b for B0-Be. Its diagonal
    in the equilibrium principal frame reproduces sonic_rovibrational's
    ground correction. No resonant individual alpha rows are selected.
    """
    w, f = np.asarray(frequencies_cm,float), np.asarray(cubic_cm,float)
    g1, g2 = np.asarray(metric_first_cm,float), np.asarray(metric_second_diagonal_cm,float)
    n = w.size
    if (w.ndim != 1 or np.any(w<=0) or f.shape!=(n,n,n)
            or g1.shape!=(n+3,n+3,n) or g2.shape!=g1.shape
            or not all(np.all(np.isfinite(v)) for v in (w,f,g1,g2))
            or not np.allclose(g1,g1.swapaxes(0,1),atol=1e-8)
            or not np.allclose(g2,g2.swapaxes(0,1),atol=1e-8)):
        raise ValueError('finite stable complete modal metric and cubic field required')
    curvature = g2[n:,n:,:].sum(axis=-1)/8
    linear = g1[:n,n:,:].transpose(0,2,1)
    vertex = linear+linear.swapaxes(0,1)
    virtual = -np.einsum('ija,ijb,ij->ab',vertex,vertex,1/(w[:,None]+w[None,:]))/8
    rho = np.einsum('jii->j',f)+np.einsum('iij->j',g1[:n,:n,:])
    mean = -np.einsum('j,abj,j->ab',rho,g1[n:,n:,:],1/w)/8
    return dict(delta_tensor_cm=curvature+virtual+mean,
                curvature_tensor_cm=curvature,virtual_tensor_cm=virtual,
                mean_geometry_tensor_cm=mean)


__all__=['contract_modal_ground_rotational_tensor']
