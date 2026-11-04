"""Diagonal periodic LAM channels from previously fitted Fourier potentials.

No potential evaluator is accepted. Dense angular values are generated solely
from supplied analytic coefficients. Off-diagonal reactive/LAM couplings are
neglected; all eigenvalues, including degeneracies, remain in the trace.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from matrix_dvr.msr import periodic_rotor_levels


class DiagonalFourierLAMPath:
    """Interpolate Fourier coefficients and log inertia on a finite path.

    Coefficients are [constant, cos(phi), sin(phi), cos(2phi), ...].
    PCHIP is local and shape preserving for each coefficient. Extension beyond
    measured path anchors is forbidden unless the caller explicitly requests
    the constant-boundary model; such positions are returned in diagnostics.
    """

    def __init__(self, positions, coefficients_cm, moments_amu_A2):
        s=np.asarray(positions,float); c=np.asarray(coefficients_cm,float)
        m=np.asarray(moments_amu_A2,float)
        if (s.ndim!=1 or len(s)<2 or np.any(np.diff(s)<=0)
                or c.ndim!=2 or c.shape[0]!=len(s) or c.shape[1]%2!=1
                or m.shape!=s.shape or np.any(m<=0)
                or not all(np.all(np.isfinite(a)) for a in (s,c,m))):
            raise ValueError('ordered finite anchors, odd Fourier coefficient count and positive moments required')
        self.positions=s.copy()
        self._coefficients=PchipInterpolator(s,c,axis=0,extrapolate=False)
        self._log_moment=PchipInterpolator(s,np.log(m),extrapolate=False)
        self.degree=(c.shape[1]-1)//2

    def spectrum(self, position, *, basis_size=121, angular_grid_size=2048,
                 boundary_model=None):
        if not np.isfinite(position):raise ValueError('finite path position required')
        outside=not self.positions[0]<=position<=self.positions[-1]
        if boundary_model not in (None,'constant'):
            raise ValueError('unknown boundary model')
        if outside and boundary_model is None:
            raise ValueError('path extrapolation needs an explicit boundary model')
        if angular_grid_size<4*basis_size:
            raise ValueError('angular grid must resolve the supplied Fourier basis')
        s=float(np.clip(position,self.positions[0],self.positions[-1]))
        c=self._coefficients(s); moment=float(np.exp(self._log_moment(s)))
        phi=np.linspace(0.,2*np.pi,angular_grid_size,endpoint=False)
        potential=np.full(len(phi),c[0])
        for n in range(1,self.degree+1):
            potential+=c[2*n-1]*np.cos(n*phi)+c[2*n]*np.sin(n*phi)
        levels=periodic_rotor_levels(moment,1,float(np.ptp(potential)),
            phi_rad=phi,potential_cm=potential,basis_size=basis_size)+float(np.min(potential))
        return dict(levels_cm=levels,moment_amu_A2=moment,
            constant_boundary_model_used=outside,coefficient_position=s,
            channel_order='instantaneous ascending eigenvalues including degeneracies',
            off_diagonal_LAM_couplings=False)
