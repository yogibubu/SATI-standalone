"""Implicit path progress whose transverse sections are internal hyperplanes.

The supplied covector annihilates transverse displacements in the same frozen
internal definition. Unlike a bond-order ratio, progress is therefore constant
on each local transverse section. Units of the callback's Cartesian jet must
match the consumer (native potentials use bohr).
"""
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BPoly,CubicSpline
from scipy.optimize import brentq


class InternalPathProgress:
    def __init__(self,positions,coordinates,tangents,covectors,coordinate_jet,*,coordinate_values=None):
        s=np.asarray(positions,float);q=np.asarray(coordinates,float)
        v=np.asarray(tangents,float);a=np.asarray(covectors,float)
        if (s.ndim!=1 or len(s)<2 or np.any(np.diff(s)<=0) or q.ndim!=2
                or q.shape[0]!=len(s) or v.shape!=q.shape or a.shape!=q.shape
                or not all(np.all(np.isfinite(x)) for x in (s,q,v,a))
                or np.any(np.einsum('ij,ij->i',v,a)<=0) or not callable(coordinate_jet)):
            raise ValueError('ordered finite path with positively oriented covectors required')
        self.positions=s;self.coordinates=q;self.coordinate_jet=coordinate_jet
        self.coordinate_values=coordinate_values
        # Complete the geometry jets from the same fixed knots, then use C4
        # Hermite pieces. A merely cubic C1 centerline would introduce jumps
        # in higher potential derivatives at geometric (non-Hessian) knots.
        qs=CubicSpline(s,q);ns=CubicSpline(s,a)
        self.center=BPoly.from_derivatives(s,
            [[q[i],v[i],qs(t,2),qs(t,3),np.zeros(q.shape[1])] for i,t in enumerate(s)])
        self.normal=BPoly.from_derivatives(s,
            [[a[i],ns(t,1),ns(t,2),ns(t,3),np.zeros(q.shape[1])] for i,t in enumerate(s)])

    def _section(self,q):
        if q.shape!=(self.coordinates.shape[1],) or not np.all(np.isfinite(q)):
            raise ValueError('finite aligned internal-coordinate values required')
        def equation(s):return float(self.normal(s)@(q-self.center(s)))
        # Select a positively oriented normal section. Multiple sections can
        # occur on a curved path; the nearest internal point selects the tube.
        values=[equation(s) for s in self.positions];roots=[]
        # A curved section may cross twice inside one interval even when the
        # endpoint signs agree. Start local solves at the nearest recorded
        # centers so that roundoff at an existing center cannot hide its root.
        # These are evaluations of the same interpolant, not extra path data.
        distances=np.linalg.norm(self.coordinates-q,axis=1)
        for index in np.argsort(distances)[:min(3,len(distances))]:
            s=float(self.positions[index])
            for _ in range(20):
                a=self.normal(s);c=self.center(s);r=float(a@(q-c))
                derivative=float(self.normal(s,1)@(q-c)-a@self.center(s,1))
                if derivative>=-1e-12:break
                tolerance=32*np.finfo(float).eps*np.linalg.norm(a)*max(1.,np.linalg.norm(q),np.linalg.norm(c))
                if abs(r)<=tolerance:
                    roots.append(s);break
                candidate=s-r/derivative
                if not self.positions[0]<=candidate<=self.positions[-1]:break
                s=float(candidate)
        for i in range(len(values)-1):
            if values[i]>=0 and values[i+1]<=0:
                roots.append(brentq(equation,self.positions[i],self.positions[i+1],xtol=2e-14))
        if roots:
            s=min(roots,key=lambda t:np.linalg.norm(q-self.center(t)))
        elif values[0]<0 or values[-1]>0:
            return float(self.positions[0] if values[0]<0 else self.positions[-1]),True
        else:raise ValueError('geometry lies outside the oriented internal path tube')
        return s,False

    def value(self,xyz):
        q=self.coordinate_values(xyz) if self.coordinate_values is not None else self.coordinate_jet(xyz)[0]
        return self._section(np.asarray(q))[0]

    def __call__(self,xyz):
        q,j,h=map(np.asarray,self.coordinate_jet(xyz))
        if (j.shape!=(len(q),np.size(xyz)) or h.shape!=(len(q),np.size(xyz),np.size(xyz))
                or not all(np.all(np.isfinite(x)) for x in (j,h))):
            raise ValueError('aligned internal-coordinate Cartesian two-jet required')
        s,clamped=self._section(q)
        if clamped:return SimpleNamespace(value=s,gradient_per_bohr=np.zeros(np.size(xyz)),hessian_per_bohr2=np.zeros((np.size(xyz),np.size(xyz))))
        a=self.normal(s);ap=self.normal(s,1);app=self.normal(s,2)
        c=self.center(s);cp=self.center(s,1);cpp=self.center(s,2)
        fs=float(ap@(q-c)-a@cp)
        if fs>=-1e-12:raise ValueError('singular or reversed internal path section')
        fss=float(app@(q-c)-2*ap@cp-a@cpp)
        first=-a/fs
        second=-(fss*np.outer(first,first)+np.outer(ap,first)+np.outer(first,ap))/fs
        return SimpleNamespace(value=float(s),gradient_per_bohr=first@j,
            hessian_per_bohr2=j.T@second@j+np.einsum('a,aij->ij',first,h))
