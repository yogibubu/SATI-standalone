"""Compact C4 correction to all already available IRC energies.

Acquired gradients/Hessians and stationary four-jets are protected explicitly.
The correction depends only on path progress, so it is constant on the
progress function's internal transverse sections. Native asymptotes remain.
"""
import numpy as np
from scipy.interpolate import CubicSpline,BPoly
from .native_pic_field import NativePicPoint


class IRCEnergyCompletion:
    def __init__(self,base,progress,positions,residual_hartree,*,protected_orders):
        self.base=base;self.progress=progress;self.atoms=tuple(base.atoms)
        self.positions=np.asarray(positions,float);e=np.asarray(residual_hartree,float)
        s=self.positions
        if (s.ndim!=1 or len(s)<3 or e.shape!=s.shape or np.any(np.diff(s)<=0)
                or not np.all(np.isfinite(s)) or not np.all(np.isfinite(e))):
            raise ValueError('ordered finite IRC positions and energy residuals required')
        reference=CubicSpline(s,e)
        jets=[[e[i],reference(t,1),reference(t,2),reference(t,3),0.] for i,t in enumerate(s)]
        self.protected_orders=dict(protected_orders)
        for position,order in self.protected_orders.items():
            ids=np.flatnonzero(np.isclose(s,float(position),rtol=0,atol=1e-12))
            if len(ids)!=1 or order not in (1,2,3,4):raise ValueError('derivative order 1--4 at an existing IRC knot required')
            jets[int(ids[0])][1:order+1]=[0.]*order
        for i in (0,-1):
            if abs(e[i])>1e-10:raise ValueError('zero correction at separated native endpoints required')
            jets[i]=[0.]*5
        self.correction=BPoly.from_derivatives(s,jets)

    def __call__(self,xyz):
        base=self.base(xyz);s=self.progress.value(xyz)
        if s<=self.positions[0] or s>=self.positions[-1]:return base
        p=self.progress(xyz);e=float(self.correction(s));d=float(self.correction(s,1));dd=float(self.correction(s,2))
        return NativePicPoint(base.energy_hartree+e,base.gradient_hartree_per_bohr+d*p.gradient_per_bohr,
            base.hessian_hartree_per_bohr2+d*p.hessian_per_bohr2+dd*np.outer(p.gradient_per_bohr,p.gradient_per_bohr))
