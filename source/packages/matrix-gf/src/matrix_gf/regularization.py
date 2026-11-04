"""Shared potential regularizers preserving a declared local quartic jet.

These complete a local PES; they do not establish its physical remote tail.
Widths and wall strengths are explicit model inputs, never inferred from a
convergence failure. Potential regularization is separate from HDC resonance
regularization and from finite occupation-domain certification.
"""
from dataclasses import dataclass
import numpy as np


def handy_gaussian_derivative_potential(q,f2,f4,beta):
    """Resident Burcl--Carter--Handy even coordinate, preserving F2/F4."""
    if not np.all(np.isfinite([f2,f4,beta])) or f2<=0 or beta<=0:
        raise ValueError('positive finite F2/beta and finite F4 required')
    q=np.asarray(q,float)
    if not np.all(np.isfinite(q)):raise ValueError('finite coordinates required')
    z2=-np.expm1(-beta*q*q)
    c2=.5*f2/beta;c4=.25*f2/beta+f4/(24*beta*beta)
    return c2*z2+c4*z2*z2,dict(handy_gaussian_beta=float(beta),
        **{'handy_gaussian_z2_coeff_cm-1':float(c2),'handy_gaussian_z4_coeff_cm-1':float(c4)},
        handy_gaussian_reference='Burcl-Carter-Handy CPL 373, Eqs. 19-22')


def handy_morse_derivative_potential(q,f2,f3,f4):
    """Resident Morse-like coordinate, preserving F2/F3/F4.

    This formula alone is not always confining; inspect the leading coefficient
    or apply a separately declared protected completion before high-level use.
    """
    if not np.all(np.isfinite([f2,f3,f4])) or f2<=0 or abs(f3)<1e-14:
        raise ValueError('positive F2 and nonzero finite F3 required')
    q=np.asarray(q,float)
    if not np.all(np.isfinite(q)):raise ValueError('finite coordinates required')
    alpha=-f3/(3*f2)
    # Retain the resident overflow guard; its presence is recorded explicitly.
    y=1-np.exp(np.clip(-alpha*q,-100.,100.))
    c2=.5*f2/alpha**2
    c3=.5*f2/alpha**2+f3/(6*alpha**3)
    c4=11*f2/(24*alpha**2)+f3/(4*alpha**3)+f4/(24*alpha**4)
    return c2*y*y+c3*y**3+c4*y**4,dict(handy_morse_alpha=float(alpha),
        **{'handy_morse_y2_coeff_cm-1':float(c2),'handy_morse_y3_coeff_cm-1':float(c3),
           'handy_morse_y4_coeff_cm-1':float(c4)},
        handy_morse_reference='Burcl-Carter-Handy CPL 373, Eqs. 6 and 10-13')


@dataclass(frozen=True)
class RegularizedQuarticPotential:
    """V0 + F1.q + H[q,q]/2 + exp[-(q.M.q/R^2)^p]*(C[q^3]/6+Q[q^4]/24).

    An optional positive wall W*(q.M.q/R^2)^(d/2), even d>=6, also preserves
    the four-jet. Coordinate units and energy units are caller-defined and
    must be consistent. M is a positive definite coordinate scaling metric.
    p=1 is Gaussian; p=3 is the protected ARCHITECT super-Gaussian rule.
    """
    hessian: np.ndarray
    cubic: np.ndarray
    quartic: np.ndarray
    radius: float
    metric: np.ndarray | None = None
    damping_power: int = 3
    wall_strength: float = 0.
    wall_degree: int = 8
    constant: float = 0.
    linear: np.ndarray | None = None

    def __post_init__(self):
        H=np.asarray(self.hessian,float);n=len(H) if H.ndim==2 else 0
        arrays=[H,np.asarray(self.cubic,float),np.asarray(self.quartic,float)]
        if not n or [a.shape for a in arrays]!=[(n,n),(n,)*3,(n,)*4]:
            raise ValueError('complete square H and cubic/quartic tensors required')
        for a in arrays:
            if not np.all(np.isfinite(a)):raise ValueError('finite tensors required')
            if any(not np.allclose(a,np.swapaxes(a,0,i),atol=1e-12,rtol=1e-10) for i in range(1,a.ndim)):
                raise ValueError('symmetric potential derivative tensors required')
        metric=np.eye(n) if self.metric is None else np.asarray(self.metric,float)
        if (metric.shape!=(n,n) or not np.all(np.isfinite(metric)) or not np.allclose(metric,metric.T)
                or min(np.linalg.eigvalsh(metric))<=0):raise ValueError('positive definite scaling metric required')
        if (not np.all(np.isfinite([self.radius,self.wall_strength,self.constant])) or self.radius<=0 or self.wall_strength<0
                or isinstance(self.damping_power,bool) or not isinstance(self.damping_power,int) or self.damping_power<1
                or isinstance(self.wall_degree,bool) or not isinstance(self.wall_degree,int)
                or self.wall_degree<6 or self.wall_degree%2):raise ValueError('positive radius/power and nonnegative even high-order wall required')
        linear=np.zeros(n) if self.linear is None else np.asarray(self.linear,float)
        if linear.shape!=(n,) or not np.all(np.isfinite(linear)):raise ValueError('matching finite linear derivative required')
        linear=linear.copy();linear.setflags(write=False);object.__setattr__(self,'linear',linear)
        for name,a in zip(['hessian','cubic','quartic','metric'],[*arrays,metric]):
            a=a.copy();a.setflags(write=False);object.__setattr__(self,name,a)

    def energy_gradient_hessian(self,q):
        q=np.asarray(q,float)
        if q.shape!=(len(self.hessian),) or not np.all(np.isfinite(q)):
            raise ValueError('matching finite coordinate vector required')
        M=self.metric/self.radius**2;a=float(q@M@q)
        da=2*M@q;dda=2*M;p=self.damping_power
        # Avoid evaluating an enormous polynomial times an underflowed switch.
        if a>745.**(1/p):w=wa=waa=0.
        else:
            w=np.exp(-a**p)
            wa=-p*a**(p-1)*w
            waa=(p*p*a**(2*p-2)-(p*(p-1)*a**(p-2) if p>=2 else 0.))*w
        dw=wa*da;ddw=waa*np.outer(da,da)+wa*dda
        delta=np.einsum('ijk,i,j,k',self.cubic,q,q,q)/6+np.einsum('ijkl,i,j,k,l',self.quartic,q,q,q,q)/24
        dg=np.einsum('ijk,j,k->i',self.cubic,q,q)/2+np.einsum('ijkl,j,k,l->i',self.quartic,q,q,q)/6
        dh=np.einsum('ijk,k->ij',self.cubic,q)+np.einsum('ijkl,k,l->ij',self.quartic,q,q)/2
        energy=self.constant+self.linear@q+.5*q@self.hessian@q+w*delta
        grad=self.linear+self.hessian@q+w*dg+delta*dw
        hess=self.hessian+w*dh+np.outer(dw,dg)+np.outer(dg,dw)+delta*ddw
        if self.wall_strength:
            order=self.wall_degree//2;wall=self.wall_strength*a**order
            wall_a=self.wall_strength*order*a**(order-1)
            wall_aa=self.wall_strength*order*(order-1)*a**(order-2)
            energy+=wall;grad+=wall_a*da;hess+=wall_aa*np.outer(da,da)+wall_a*dda
        if not np.isfinite(energy) or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
            raise FloatingPointError('regularized potential exceeds numerical range')
        return float(energy),grad,hess

    def energy(self,q):return self.energy_gradient_hessian(q)[0]

    def metadata(self):
        return dict(model='protected_quartic_completion',damping_power=self.damping_power,radius=self.radius,
                    metric=self.metric.tolist(),wall_strength=self.wall_strength,wall_degree=self.wall_degree,
                    local_derivatives_preserved_through=4,leading_added_cubic_order=3+2*self.damping_power,
                    mathematically_confining=bool(self.wall_strength>0 or min(np.linalg.eigvalsh(self.hessian))>0),
                    global_absence_of_additional_minima_proved=False,physical_tail_validated=False)
