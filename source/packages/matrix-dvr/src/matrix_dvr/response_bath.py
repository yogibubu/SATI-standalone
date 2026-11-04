"""Bounded analytic mean-response parameterization of a quadratic bath.

The complete quadratic scalar is E(q)+g(q).b+1/2 b.H(q).b, with
 g(q)=-H(q)b_star(q). A smooth bounded response is fitted directly instead
of dividing an independently extrapolated force by a soft Hessian. Derivatives
include every product-rule term. Bounds are model parameters, not clipping.
"""
from dataclasses import dataclass
import numpy as np
from .periodic_fields import PeriodicVectorField,PeriodicQuadraticBath


@dataclass(frozen=True)
class AdditivePeriodicResponseField:
    """Sum bounded periodic fields without refitting their archived jets."""
    terms: tuple

    def __post_init__(self):
        terms=tuple(self.terms)
        if not terms or len({term.coefficients.shape[1] for term in terms}) != 1:
            raise ValueError('Compatible nonempty periodic response fields required')
        object.__setattr__(self,'terms',terms)

    @property
    def coefficients(self):
        return self.terms[0].coefficients

    def evaluate(self, points, derivative=None):
        return sum(term.evaluate(points, derivative) for term in self.terms)

    def to_dict(self):
        return dict(schema='matrix.dvr.additive_periodic_response.v1',
                    terms=[term.to_dict() for term in self.terms])


def response_field_from_dict(data):
    schema = data.get('schema')
    if schema == 'matrix.dvr.additive_periodic_response.v1':
        return AdditivePeriodicResponseField(tuple(response_field_from_dict(term) for term in data['terms']))
    if schema == 'matrix.dvr.periodic_shepard_hermite.v1':
        from .periodic_shepard import PeriodicShepardHermiteField
        return PeriodicShepardHermiteField.from_dict(data)
    if schema in ('matrix.dvr.periodic_vector_field.v1', 'matrix.dvr.periodic_vector_field.v2'):
        return PeriodicVectorField.from_dict(data)
    return SaturatingPeriodicVectorField.from_dict(data)


@dataclass(frozen=True)
class SaturatingPeriodicVectorField:
    base: PeriodicVectorField
    amplitudes: np.ndarray

    def __post_init__(self):
        a=np.array(self.amplitudes,float,copy=True)
        if a.shape!=(self.base.coefficients.shape[1],) or np.any(a<=0) or not np.all(np.isfinite(a)):
            raise ValueError('Positive finite response amplitudes required')
        a.flags.writeable=False;object.__setattr__(self,'amplitudes',a)

    @property
    def coefficients(self):return self.base.coefficients

    def evaluate(self,points,derivative=None):
        d=len(self.base.scales);order=np.zeros(d,int) if derivative is None else np.asarray(derivative)
        if order.shape!=(d,) or np.any(order<0) or np.any(order!=np.rint(order)) or order.sum()>2:
            raise ValueError('Integer response derivative through order two required')
        axes=np.repeat(np.arange(d),order.astype(int));t=np.tanh(self.base.evaluate(points)/self.amplitudes)
        if not len(axes):return self.amplitudes*t
        eye=np.eye(d,dtype=int);first=self.base.evaluate(points,eye[axes[0]])
        if len(axes)==1:return (1-t*t)*first
        other=self.base.evaluate(points,eye[axes[1]])
        return (1-t*t)*(self.base.evaluate(points,order)-2*t*first*other/self.amplitudes)

    def to_dict(self):return dict(schema='matrix.dvr.saturating_periodic_vector_field.v1',base=self.base.to_dict(),amplitudes=self.amplitudes.tolist())

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.dvr.saturating_periodic_vector_field.v1':raise ValueError('Unsupported bounded response field')
        return cls(response_field_from_dict(data['base']),np.array(data['amplitudes']))


@dataclass(frozen=True)
class PeriodicResponseQuadraticBath:
    reference_cholesky: np.ndarray
    response_field: SaturatingPeriodicVectorField
    matrix_log: object

    def evaluate(self,points,*,second=False,derivatives=True):
        q=np.asarray(points,float)
        if q.ndim!=2 or not np.all(np.isfinite(q)):raise ValueError('Finite angular points required')
        if second and not derivatives:raise ValueError('Second derivatives require first derivatives')
        n=len(self.reference_cholesky);d=q.shape[1];eye=np.eye(d,dtype=int)
        zero=PeriodicVectorField(np.zeros((1,d),int),np.zeros((1,n)),np.ones(d),np.ones(d),1)
        provider=PeriodicQuadraticBath(self.reference_cholesky,zero,self.matrix_log)
        fields=provider.evaluate(q,second=second,derivatives=derivatives);H=fields['hessian'];H1=fields['hessian_first']
        b=self.response_field.evaluate(q);g=-np.einsum('pij,pj->pi',H,b)
        b1=np.stack([self.response_field.evaluate(q,e) for e in eye],axis=-1) if derivatives else None
        g1=(-np.einsum('pijs,pj->pis',H1,b)-np.einsum('pij,pjs->pis',H,b1)) if derivatives else None
        fields.update(force=g,force_first=g1,response=b,response_first=b1,energy_lowering=.5*np.einsum('pi,pi->p',g,b))
        if derivatives:fields['energy_lowering_first']=np.einsum('pis,pi->ps',b1,g)-.5*np.einsum('pi,pijs,pj->ps',b,H1,b)
        if second:
            H2=fields['hessian_second'];b2=np.stack([np.stack([self.response_field.evaluate(q,e+f) for f in eye],axis=-1) for e in eye],axis=-2)
            mix=np.einsum('pijs,pjt->pist',H1,b1)
            fields.update(response_second=b2,force_second=-np.einsum('pijst,pj->pist',H2,b)-mix-mix.swapaxes(-1,-2)-np.einsum('pij,pjst->pist',H,b2))
            lowering2=(-np.einsum('pist,pij,pj->pst',b2,H,b)
                -np.einsum('pis,pij,pjt->pst',b1,H,b1)
                -np.einsum('pis,pijt,pj->pst',b1,H1,b)
                -np.einsum('pit,pijs,pj->pst',b1,H1,b)
                -.5*np.einsum('pi,pijst,pj->pst',b,H2,b))
            fields['energy_lowering_second']=lowering2
        return fields

    def to_dict(self):return dict(schema='matrix.dvr.periodic_response_quadratic_bath.v1',reference_cholesky=np.asarray(self.reference_cholesky).tolist(),response_field=self.response_field.to_dict(),matrix_log=self.matrix_log.to_dict(),physical_tail_accuracy_qualified=False)

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.dvr.periodic_response_quadratic_bath.v1':raise ValueError('Unsupported response bath')
        matrix=response_field_from_dict(data['matrix_log'])
        field = response_field_from_dict(data['response_field'])
        return cls(np.array(data['reference_cholesky']),field,matrix)
