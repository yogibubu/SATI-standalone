"""C6 periodic continuation of auxiliary fields on an unrestricted angle.

The physical integration angle is unchanged. Only the fitted auxiliary field
argument is continued: identity on the complete sampled box, bounded outside,
and smoothly zero at the periodic seam. This is a declared continuation
assumption and is not a molecular asymptote inferred from electronic data.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import betainc, betaln
from .field_continuation import SmoothEnvelopeMap
from .periodic_fields import PeriodicVectorField


@dataclass(frozen=True)
class PeriodicEnvelopeMap:
    extent: np.ndarray
    margin_fraction: float = .1

    def __post_init__(self):
        base = SmoothEnvelopeMap(self.extent, self.margin_fraction)
        if np.any(base.extent >= np.pi): raise ValueError('Sampled angular extents must be below the periodic seam')
        object.__setattr__(self, 'extent', base.extent)

    def evaluate(self, points):
        q = np.asarray(points, float)
        wrapped = np.arctan2(np.sin(q),np.cos(q))
        base, first, second = SmoothEnvelopeMap(self.extent,self.margin_fraction).evaluate(wrapped)
        width = np.pi-self.extent
        t = np.clip((abs(wrapped)-self.extent)/width,0.,1.)
        w = 1.-betainc(7,7,t)
        normalization = np.exp(-betaln(7,7))
        density = normalization*t**6*(1-t)**6
        density_first = 6*normalization*(t**5*(1-t)**6-t**6*(1-t)**5)
        w1 = -np.sign(wrapped)*density/width
        w2 = -density_first/width**2
        return base*w, first*w+base*w1, second*w+2*first*w1+base*w2

    def to_dict(self):
        return dict(schema='matrix.dvr.periodic_envelope_map.v1',extent=self.extent.tolist(),
            margin_fraction=self.margin_fraction,period_radian=2*np.pi,smoothness='C6',
            identity_inside_sampled_envelope=True,physical_angle_unchanged=True,
            physical_tail_accuracy_qualified=False)


@dataclass(frozen=True)
class ContinuedPeriodicVectorField:
    base: object
    mapping: PeriodicEnvelopeMap

    @property
    def coefficients(self): return self.base.coefficients

    def evaluate(self, points, derivative=None):
        q=np.asarray(points,float);d=len(self.mapping.extent)
        order=np.zeros(d,int) if derivative is None else np.asarray(derivative)
        if order.shape != (d,) or np.any(order < 0) or np.any(order != order.astype(int)) or order.sum()>2:
            raise ValueError('Integer derivative orders through second order required')
        mapped,J,K=self.mapping.evaluate(q)
        axes=np.repeat(np.arange(d),order.astype(int))
        if not len(axes): return self.base.evaluate(mapped)
        value=self.base.evaluate(mapped,tuple(order))
        for i in axes: value=value*J[:,i,None]
        if len(axes)==2 and axes[0]==axes[1]:
            e=np.eye(d,dtype=int)[axes[0]]
            value=value+self.base.evaluate(mapped,e)*K[:,axes[0],None]
        return value

    def to_dict(self):
        return dict(schema='matrix.dvr.continued_periodic_vector_field.v1',base=self.base.to_dict(),mapping=self.mapping.to_dict())


def periodic_vector_field_from_dict(data):
    if data.get('schema')=='matrix.dvr.continued_periodic_vector_field.v1':
        m=data['mapping']
        if m.get('schema')!='matrix.dvr.periodic_envelope_map.v1': raise ValueError('Unsupported periodic envelope map')
        return ContinuedPeriodicVectorField(periodic_vector_field_from_dict(data['base']),PeriodicEnvelopeMap(np.array(m['extent']),m['margin_fraction']))
    return PeriodicVectorField.from_dict(data)
