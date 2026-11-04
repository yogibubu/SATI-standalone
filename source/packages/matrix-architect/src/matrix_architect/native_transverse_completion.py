"""Higher-order native corrections that vanish through E/G/H on the IRC."""
import numpy as np
from scipy.interpolate import BPoly,CubicSpline
from matrix_chem import BOHR_TO_ANGSTROM


class C4CartesianPath:
    def __init__(self,positions,coordinates,velocities):
        self.s=np.asarray(positions,float);self.x=np.asarray(coordinates,float)
        self.v=np.asarray(velocities,float)
        if (np.any(np.diff(self.s)<=0) or self.x.shape!=self.v.shape or self.x.shape[0]!=len(self.s)
                or not all(np.all(np.isfinite(z)) for z in (self.s,self.x,self.v))):
            raise ValueError('ordered finite path geometries and velocities required')
        approximate=CubicSpline(self.s,self.x)
        jets=[[self.x[i],self.v[i],approximate(t,2),approximate(t,3),np.zeros_like(self.x[i])]
              for i,t in enumerate(self.s)]
        for i in (0,-1):jets[i][2:]=[np.zeros_like(self.x[i]) for _ in range(3)]
        self.curve=BPoly.from_derivatives(self.s,jets)
    def __call__(self,s,order=0):
        if self.s[0]<=s<=self.s[-1]:return self.curve(s,order)
        i=0 if s<self.s[0] else -1
        if order==0:return self.x[i]+(s-self.s[i])*self.v[i]
        return self.v[i] if order==1 else np.zeros_like(self.x[i])


class TransverseNativeDisplacement:
    def __init__(self,kernel,progress,cartesian_path,projection):
        self.kernel=kernel;self.progress=progress;self.path=cartesian_path
        self.projection=np.asarray(projection,float)
    def value(self,xyz):
        p=self.progress(xyz);s=p.value;center=self.path(s)
        u,j,h=self.kernel.value(xyz);uc,jc,hc=self.kernel.value(center)
        velocity=self.path(s,1).ravel();acceleration=self.path(s,2).ravel()
        derivative=jc.T@velocity
        curvature=np.einsum('i,ijt,j->t',velocity,hc,velocity)+jc.T@acceleration
        ds=p.gradient_per_bohr/BOHR_TO_ANGSTROM;d2s=p.hessian_per_bohr2/BOHR_TO_ANGSTROM**2
        first=j-np.outer(ds,derivative)
        second=h-np.einsum('ij,t->ijt',d2s,derivative)-np.einsum('i,j,t->ijt',ds,ds,curvature)
        a=self.projection
        return a@(u-uc),first@a.T,np.einsum('ijt,at->ija',second,a)
