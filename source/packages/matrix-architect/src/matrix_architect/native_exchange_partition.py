"""Compact C4 partition of resident native valence-exchange potentials.

Local scalar fields, rather than matrices of modal force constants, are
combined. Analytic differentiation therefore retains every weight derivative
and the moving native primitive directions. No reference PES is called.
"""
import numpy as np
from types import SimpleNamespace
from .native_pic_field import NativePicPoint


class NativeFieldPartition:
    """Blend neighboring scalar exchange fields with protected anchor four-jets.

    Positive ninth-degree windows combine only the two neighboring fields,
    with weight derivatives zero through fourth order at every anchor.
    Consequently each local energy/gradient/Hessian/cubic/quartic jet is
    preserved. A common physical progress two-jet is supplied by the caller.
    Each field must carry its own separated-channel support.
    """
    def __init__(self,fields,positions,progress,*,continuity_order=4):
        self.fields=tuple(fields);self.positions=np.asarray(positions,float)
        if (len(self.fields)<2 or self.positions.shape!=(len(self.fields),)
                or np.any(np.diff(self.positions)<=0) or not np.all(np.isfinite(self.positions))
                or not callable(progress) or continuity_order!=4 or any(not callable(f) for f in self.fields)):
            raise ValueError('ordered resident fields, progress function and C4 continuity required')
        self.progress=progress
        self.atoms=tuple(getattr(self.fields[0],'atoms',()))
        if not self.atoms or any(tuple(getattr(f,'atoms',()))!=self.atoms for f in self.fields):
            raise ValueError('all local native fields must share the same atom order')
        self.smooth=np.polynomial.Polynomial([1,0,0,0,0,-126,420,-540,315,-70])
        self.rise=np.polynomial.Polynomial([0,0,0,0,0,126,-420,540,-315,70])
        self.schema='matrix.architect.native_exchange_partition.v1'

    def weights(self,s):
        if not np.isfinite(s):raise ValueError('finite native reaction progress required')
        n=len(self.positions);values=np.zeros(n);first=np.zeros(n);second=np.zeros(n)
        if s<=self.positions[0] or s>=self.positions[-1]:
            values[0 if s<=self.positions[0] else -1]=1.
            return values,first,second
        i=int(np.searchsorted(self.positions,s,side='right')-1)
        width=self.positions[i+1]-self.positions[i];t=(s-self.positions[i])/width
        if t<=.5:
            right=float(self.rise(t));value=1-right
            gradient=-float(self.rise.deriv(1)(t))/width;hessian=-float(self.rise.deriv(2)(t))/width**2
        else:
            value=float(self.rise(1-t))
            gradient=-float(self.rise.deriv(1)(1-t))/width;hessian=float(self.rise.deriv(2)(1-t))/width**2
        values[i],values[i+1]=value,1-value
        first[i],first[i+1]=gradient,-gradient;second[i],second[i+1]=hessian,-hessian
        return values,first,second

    def __call__(self,xyz):
        # A separated endpoint needs only its own coordinate definition. The
        # reactive chart may lose rank on disconnected/linear fragments.
        value_only=getattr(self.progress,'value',None)
        if callable(value_only):
            s=value_only(xyz)
            if s<=self.positions[0]:return self.fields[0](xyz)
            if s>=self.positions[-1]:return self.fields[-1](xyz)
        p=self.progress(xyz);w,w1,w2=self.weights(float(p.value));d=np.size(xyz)
        g=np.asarray(p.gradient_per_bohr,float);h=np.asarray(p.hessian_per_bohr2,float)
        if g.shape!=(d,) or h.shape!=(d,d) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
            raise ValueError('finite Cartesian reaction-progress two-jet required')
        e=0.;gradient=np.zeros(d);hessian=np.zeros((d,d))
        for field,a,a1,a2 in zip(self.fields,w,w1,w2):
            if a==0 and a1==0 and a2==0:continue
            value=field(xyz);wg=a1*g;wh=a1*h+a2*np.outer(g,g)
            e+=a*value.energy_hartree
            gradient+=a*value.gradient_hartree_per_bohr+value.energy_hartree*wg
            hessian+=(a*value.hessian_hartree_per_bohr2+value.energy_hartree*wh
                +np.outer(wg,value.gradient_hartree_per_bohr)+np.outer(value.gradient_hartree_per_bohr,wg))
        return NativePicPoint(float(e),gradient,hessian)

    def energy(self,xyz):
        return self(xyz).energy_hartree

    def energy_gradient(self,xyz):
        point=self(xyz)
        return point.energy_hartree,point.gradient_hartree_per_bohr

    def coupling_jet(self,xyz):
        p=self(xyz);d=np.size(xyz);m=np.zeros((2,2));g=np.zeros((2,2,d));h=np.zeros((2,2,d,d))
        m[0,1]=m[1,0]=p.energy_hartree;g[0,1]=g[1,0]=p.gradient_hartree_per_bohr
        h[0,1]=h[1,0]=p.hessian_hartree_per_bohr2
        return SimpleNamespace(matrix=m,gradient=g,hessian=h)
