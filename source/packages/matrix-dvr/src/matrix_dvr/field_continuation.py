"""Explicit bounded continuation of a finite-domain quadratic bath field.

The fitted field is unchanged throughout its sampled componentwise envelope.
Beyond it, a C6 rational map approaches a finite auxiliary argument. This is
an explicit continuation assumption, not new electronic information or an
assertion of a molecular dissociation asymptote. Numerical diagnostics must
request it explicitly and assess its sensitivity before claiming observables.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SmoothEnvelopeMap:
    extent: np.ndarray
    margin_fraction: float = .1
    power: int = 6

    def __post_init__(self):
        extent=np.array(self.extent,float,copy=True)
        if (extent.ndim!=1 or np.any(extent<=0) or not np.all(np.isfinite(extent))
                or not np.isfinite(self.margin_fraction) or self.margin_fraction<=0
                or isinstance(self.power,bool) or int(self.power)!=self.power or self.power<3):
            raise ValueError('Positive finite field extents, margin and integer smoothness power required')
        extent.flags.writeable=False;object.__setattr__(self,'extent',extent)

    def evaluate(self,points):
        q=np.asarray(points,float)
        if q.ndim!=2 or q.shape[1]!=len(self.extent) or not np.all(np.isfinite(q)):
            raise ValueError('Compatible finite field arguments required')
        t=np.maximum(abs(q)/self.extent-1.,0.)
        r=t/self.margin_fraction;den=1+r**self.power
        outer=t/den**(1/self.power)
        inside=abs(q)<=self.extent
        mapped=np.where(inside,q,np.sign(q)*self.extent*(1+outer))
        first=np.where(inside,1.,den**(-1-1/self.power))
        second=np.where(inside,0.,-np.sign(q)*(self.power+1)*r**(self.power-1)
                        /self.margin_fraction/den**(2+1/self.power)/self.extent)
        return mapped,first,second

    def to_dict(self):
        return dict(schema='matrix.dvr.smooth_envelope_map.v1',extent=self.extent.tolist(),
                    margin_fraction=float(self.margin_fraction),power=int(self.power),
                    identity_inside_sampled_envelope=True,
                    outside_scope='Explicit bounded continuation assumption; physical accuracy not inferred')


@dataclass(frozen=True)
class SmoothEnvelopeBath:
    base: object
    mapping: SmoothEnvelopeMap

    @property
    def scale(self):return self.base.scale

    def evaluate(self,points,*,second=False,allow_extrapolation=False):
        q=np.asarray(points,float)
        if not allow_extrapolation and np.any(abs(q)>self.mapping.extent+1e-12):
            raise ValueError('Outside-source bath continuation requires an explicit diagnostic request')
        mapped,J,K=self.mapping.evaluate(q)
        fields=self.base.evaluate(mapped,second=second,allow_extrapolation=True)
        first={key:fields[key].copy() for key in ('force_first','hessian_first','response_first','energy_lowering_first')}
        for key,value in first.items():
            fields[key]=value*J.reshape((len(q),)+(1,)*(value.ndim-2)+(J.shape[1],))
        if second:
            for key in ('force','hessian','response'):
                value=fields[key+'_second'];d=J.shape[1];leading=(len(q),)+(1,)*(value.ndim-3)
                value=value*J.reshape(leading+(d,1))*J.reshape(leading+(1,d))
                for i in range(d):
                    value[...,i,i]+=first[key+'_first'][...,i]*K[:,i].reshape((len(q),)+(1,)*(value.ndim-3))
                fields[key+'_second']=value
        return fields
