"""Periodic valley profiles with positive transverse quadratic curvature.

The valley energy, centre and positive matrix-log curvature are independent
analytic profiles of one angle. Nonnegative even transverse monomials add
anharmonicity without changing the prescribed valley energy or Hessian.
There are no electronic callbacks or molecule-specific constants.
"""
from dataclasses import dataclass
import numpy as np
from .periodic_fields import PeriodicQuadraticBath,PeriodicVectorField
from .periodic_shepard import PeriodicShepardHermiteField
from .periodic_hermite import PeriodicHermiteSurface


def _profile(data):
    if data.get('schema')=='matrix.dvr.periodic_shepard_hermite.v1':
        return PeriodicShepardHermiteField.from_dict(data)
    return PeriodicVectorField.from_dict(data)


@dataclass(frozen=True)
class PeriodicValleySurface:
    reference_hessian: np.ndarray
    valley_axis: int
    energy_profile: object
    center_profile: object
    curvature_profile: PeriodicQuadraticBath
    residual_powers: np.ndarray
    residual_coefficients: np.ndarray
    softening_rates: np.ndarray | None = None
    residual_gram: np.ndarray | None = None
    octic_restoration: bool = True

    def __post_init__(self):
        H=np.array(self.reference_hessian,float,copy=True)
        powers=np.array(self.residual_powers,int,copy=True);c=np.array(self.residual_coefficients,float,copy=True)
        d=len(H);n=d-1
        if (H.shape!=(d,d) or not np.allclose(H,H.T) or not 0<=self.valley_axis<d
                or powers.ndim!=2 or powers.shape[1]!=n or np.any(powers<0) or np.any(powers%2)
                or np.any(powers.sum(axis=1)<4) or c.shape!=(len(powers),) or np.any(c<0)
                or not all(np.all(np.isfinite(a)) for a in (H,powers,c))):
            raise ValueError('Symmetric reference, one valley axis and nonnegative even residual required')
        for a in (H,powers,c):a.flags.writeable=False
        transverse=[i for i in range(d) if i!=self.valley_axis]
        if np.linalg.eigvalsh(H[np.ix_(transverse,transverse)]).min()<=0:
            raise ValueError('The reference transverse block must be positive')
        object.__setattr__(self,'reference_hessian',H);object.__setattr__(self,'residual_powers',powers);object.__setattr__(self,'residual_coefficients',c)
        rates=np.zeros(n) if self.softening_rates is None else np.array(self.softening_rates,float,copy=True)
        if rates.shape!=(n,) or np.any(rates<0) or not np.all(np.isfinite(rates)):raise ValueError('Finite nonnegative transverse softening rates required')
        rates.flags.writeable=False;object.__setattr__(self,'softening_rates',rates)
        size=n*(n+1)//2;gram=np.zeros((size,size)) if self.residual_gram is None else np.array(self.residual_gram,float,copy=True)
        if gram.shape!=(size,size) or not np.all(np.isfinite(gram)) or not np.allclose(gram,gram.T) or np.linalg.eigvalsh(gram).min()<-1e-8:
            raise ValueError('Positive semidefinite transverse quartic Gram matrix required')
        gram.flags.writeable=False;object.__setattr__(self,'residual_gram',gram)

    @property
    def baseline(self):
        d=len(self.reference_hessian)
        return PeriodicHermiteSurface(self.reference_hessian,np.empty((0,d),int),np.empty(0),np.ones(d))

    def _jets(self,points,axes):
        q=np.asarray(points,float);d=len(self.reference_hessian)
        if q.ndim!=2 or q.shape[1]!=d or not np.all(np.isfinite(q)):raise ValueError('Finite matching angular points required')
        transverse=np.array([i for i in range(d) if i!=self.valley_axis]);w=q[:,self.valley_axis,None]
        center=self.center_profile.evaluate(w);y=np.sin(q[:,transverse])-center
        first={};second={}
        for i in set(axes):
            a=np.zeros_like(y)
            if i==self.valley_axis:a-=self.center_profile.evaluate(w,[1])
            else:a[:,np.flatnonzero(transverse==i)[0]]=np.cos(q[:,i])
            first[i]=a
        if len(axes)==2:
            i,j=axes;a=np.zeros_like(y)
            if i==j==self.valley_axis:a-=self.center_profile.evaluate(w,[2])
            elif i==j:a[:,np.flatnonzero(transverse==i)[0]]=-np.sin(q[:,i])
            second[i,j]=a
        return w,y,first,second

    def residual_features(self,points,derivative=None,*,powers=None):
        d=len(self.reference_hessian);order=np.zeros(d,int) if derivative is None else np.asarray(derivative)
        if order.shape!=(d,) or np.any(order<0) or np.any(order!=order.astype(int)) or order.sum()>2:raise ValueError('Angular derivatives through order two required')
        axes=list(np.repeat(np.arange(d),order.astype(int)));w,y,y1,y2=self._jets(points,axes);n=y.shape[1]
        def monomial(power):return np.prod(y**np.asarray(power),axis=1)
        result=[]
        for p in (self.residual_powers if powers is None else powers):
            if not axes:value=monomial(p)
            else:
                gradient=np.zeros((len(y),n));hessian=np.zeros((len(y),n,n))
                for a in range(n):
                    if p[a]:gradient[:,a]=p[a]*monomial(p-np.eye(n,dtype=int)[a])
                    for b in range(n):
                        factor=p[a]*(p[b]-(a==b))
                        if factor:hessian[:,a,b]=factor*monomial(p-np.eye(n,dtype=int)[a]-np.eye(n,dtype=int)[b])
                if len(axes)==1:value=np.einsum('ga,ga->g',gradient,y1[axes[0]])
                else:
                    i,j=axes;value=np.einsum('ga,ga->g',gradient,y2[i,j])+np.einsum('ga,gab,gb->g',y1[i],hessian,y1[j])
            result.append(value)
        return np.stack(result,axis=1) if result else np.empty((len(y),0))

    def gram_features(self,points,derivative=None):
        n=len(self.reference_hessian)-1;eye=np.eye(n,dtype=int)
        p=np.array([2*e for e in eye]+[eye[i]+eye[j] for i in range(n) for j in range(i)])
        tri=np.tril_indices(len(p));features=self.residual_features(points,derivative,powers=p[tri[0]]+p[tri[1]])
        return features*np.where(tri[0]==tri[1],1.,2.)

    def evaluate(self,points,derivative=None,*,allow_extrapolation=False):
        d=len(self.reference_hessian);order=np.zeros(d,int) if derivative is None else np.asarray(derivative)
        if order.shape!=(d,) or np.any(order<0) or np.any(order!=order.astype(int)) or order.sum()>2:raise ValueError('Angular derivatives through order two required')
        axes=list(np.repeat(np.arange(d),order.astype(int)));w,y,y1,y2=self._jets(points,axes)
        denominator=1+self.softening_rates*y*y
        h1=denominator**(-1.5);h2=-3*self.softening_rates*y*denominator**(-2.5)
        if len(axes)==2:
            i,j=axes;y2[i,j]=h1*y2[i,j]+h2*y1[i]*y1[j]
        y1={i:h1*a for i,a in y1.items()};y=y/np.sqrt(denominator)
        fields=self.curvature_profile.evaluate(w,derivatives=bool(axes),second=len(axes)==2);K=fields['hessian']
        profile_order=int(order[self.valley_axis]);energy=self.energy_profile.evaluate(w,[profile_order])[:,0]
        if int(order.sum())!=profile_order:energy=np.zeros(len(y))
        if not axes:value=.5*np.einsum('ga,gab,gb->g',y,K,y)
        elif len(axes)==1:
            i=axes[0];value=np.einsum('ga,gab,gb->g',y1[i],K,y)
            if i==self.valley_axis:value+=.5*np.einsum('ga,gab,gb->g',y,fields['hessian_first'][...,0],y)
        else:
            i,j=axes;value=np.einsum('ga,gab,gb->g',y2[i,j],K,y)+np.einsum('ga,gab,gb->g',y1[i],K,y1[j])
            if j==self.valley_axis:value+=np.einsum('ga,gab,gb->g',y1[i],fields['hessian_first'][...,0],y)
            if i==self.valley_axis:value+=np.einsum('ga,gab,gb->g',y1[j],fields['hessian_first'][...,0],y)
            if i==j==self.valley_axis:value+=.5*np.einsum('ga,gab,gb->g',y,fields['hessian_second'][...,0,0],y)
        tail=0.
        if self.octic_restoration:
            transverse=[i for i in range(d) if i!=self.valley_axis]
            coefficients=.5*np.diag(self.reference_hessian)[transverse]*self.softening_rates/(1+self.softening_rates)
            tail=self.residual_features(points,order,powers=8*np.eye(d-1,dtype=int))@coefficients
        return (energy+value+self.residual_features(points,order)@self.residual_coefficients
                +self.gram_features(points,order)@self.residual_gram[np.tril_indices(len(self.residual_gram))]+tail)

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_valley_surface.v1',reference_hessian=self.reference_hessian.tolist(),
            valley_axis=self.valley_axis,energy_profile=self.energy_profile.to_dict(),center_profile=self.center_profile.to_dict(),
            curvature_profile=self.curvature_profile.to_dict(),residual_powers=self.residual_powers.tolist(),
            residual_coefficients=self.residual_coefficients.tolist(),softening_rates=self.softening_rates.tolist(),
            residual_gram=self.residual_gram.tolist(),octic_restoration=bool(self.octic_restoration),positive_transverse_quadratic_form=True)

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.dvr.periodic_valley_surface.v1':raise ValueError('Unsupported periodic valley')
        return cls(np.array(data['reference_hessian']),int(data['valley_axis']),_profile(data['energy_profile']),
            _profile(data['center_profile']),PeriodicQuadraticBath.from_dict(data['curvature_profile']),
            np.array(data['residual_powers']),np.array(data['residual_coefficients']),np.array(data.get('softening_rates',np.zeros(len(data['reference_hessian'])-1))),
            None if data.get('residual_gram') is None else np.array(data['residual_gram']),bool(data.get('octic_restoration',False)))
