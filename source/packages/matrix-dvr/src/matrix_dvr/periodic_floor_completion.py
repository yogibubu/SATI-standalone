"""Compact smooth periodic floor completion, inactive on protected data.

A positive radial periodic floor can lift unsupported outer wells. The
completion is exactly zero whenever base-floor exceeds its transition width,
so every derivative at protected observations is preserved. Absence of
additional stationary minima must still be audited on the completed field.
"""
from dataclasses import dataclass
import numpy as np


def compact_positive_part(delta,width):
    if not np.isfinite(width) or width<=0:raise ValueError('Positive finite transition width required')
    x=np.asarray(delta,float)
    if not np.all(np.isfinite(x)):raise ValueError('Finite potential differences required')
    t=np.clip((x+width)/(2*width),0,1)
    value=2*width*t**6*(21+t*(-60+t*(67.5+t*(-35+7*t))))
    first=t**5*(126+t*(-420+t*(540+t*(-315+70*t))))
    second=630*t**4*(1-t)**4/(2*width)
    return np.where(x<=-width,0,np.where(x>=width,x,value)),np.where(x<=-width,0,np.where(x>=width,1,first)),np.where(abs(x)>=width,0,second)


@dataclass(frozen=True)
class PeriodicRadialFloor:
    center: np.ndarray
    coefficient: float
    offset: float

    def __post_init__(self):
        c=np.array(self.center,float,copy=True)
        if c.ndim!=1 or not len(c) or not np.all(np.isfinite(c)) or not np.isfinite(self.coefficient) or self.coefficient<=0 or not np.isfinite(self.offset):raise ValueError('Finite center, positive coefficient and finite offset required')
        c.flags.writeable=False;object.__setattr__(self,'center',c)

    def evaluate(self,points):
        q=np.asarray(points,float);delta=q-self.center;r=np.sin(delta)**2;s=r.sum(axis=-1);ds=np.sin(2*delta)
        e=self.coefficient*s*s+self.offset;g=2*self.coefficient*s[...,None]*ds
        h=2*self.coefficient*(ds[..., :,None]*ds[...,None,:]+s[...,None,None]*np.eye(len(self.center))*2*np.cos(2*delta)[...,None,:])
        return e,g,h

    def to_dict(self):return dict(schema='matrix.dvr.periodic_radial_floor.v1',center=self.center.tolist(),coefficient=float(self.coefficient),offset=float(self.offset))

    @classmethod
    def from_dict(cls,r):
        if r.get('schema')!='matrix.dvr.periodic_radial_floor.v1':raise ValueError('Unsupported periodic floor')
        return cls(r['center'],r['coefficient'],r['offset'])


@dataclass(frozen=True)
class SmoothFloorCompletion:
    base: object
    floor: PeriodicRadialFloor
    transition_width: float

    def __post_init__(self):
        if not np.isfinite(self.transition_width) or self.transition_width<=0:raise ValueError('Positive finite transition width required')

    def evaluate(self,points):
        e,g,h=self.base.evaluate(points);w,wg,wh=self.floor.evaluate(points);value,first,second=compact_positive_part(w-e,self.transition_width);dg=wg-g
        return e+value,g+first[...,None]*dg,h+first[...,None,None]*(wh-h)+second[...,None,None]*dg[..., :,None]*dg[...,None,:]

    def to_dict(self):return dict(schema='matrix.dvr.smooth_floor_completion.v1',base=self.base.to_dict(),floor=self.floor.to_dict(),transition_width=float(self.transition_width))

    @classmethod
    def from_dict(cls,r,base_loader):
        if r.get('schema')!='matrix.dvr.smooth_floor_completion.v1':raise ValueError('Unsupported smooth floor completion')
        return cls(base_loader(r['base']),PeriodicRadialFloor.from_dict(r['floor']),r['transition_width'])
