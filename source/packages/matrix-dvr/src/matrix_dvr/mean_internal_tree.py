"""Analytic mean-response path on a retained transverse internal-tree frame."""
from dataclasses import dataclass
import numpy as np
from .field_continuation import SmoothEnvelopeMap


@dataclass(frozen=True)
class MeanInternalTreePath:
    chart: object
    response: object
    valence_join_radian: float = .15
    valence_limit_radian: float = .05

    def __post_init__(self):
        if not 0<self.valence_limit_radian<self.valence_join_radian<np.pi/2:
            raise ValueError('Ordered physical valence-angle continuation limits required')

    def internal_first_jets(self, points):
        q=np.asarray(points,float);N,d=q.shape;path=self.chart.path;n=len(path.bath_indices);nv=n+d
        jet=self.chart.internal_jets(q);v=self.response.evaluate(q,allow_extrapolation=True)
        b,db=v['response'],v['response_first'];A=jet['first'][:,:,d:];B=jet['second'][:,:,d:,d:]
        A1=jet['second'][:,:,d:,:d];tri=np.tril_indices(n);eye=np.eye(d,dtype=int)
        B1bb=np.zeros((N,nv,d));pairs=b[:,tri[0]]*b[:,tri[1]]*(1+(tri[0]!=tri[1]))[None,:]
        for i,e in enumerate(eye):
            field=self.chart.bath_curvature_field.evaluate(q,e).reshape(N,nv,len(tri[0]))
            B1bb[:,:,i]=np.einsum('pij,pj->pi',field,pairs)
        values=np.tile(path.reference_values,(N,1));values[:,path.active_indices]+=q
        values[:,path.bath_indices]+=path.path_field.evaluate(q)
        values+=np.einsum('pij,pj->pi',A,b)+.5*np.einsum('pijk,pj,pk->pi',B,b,b)
        first=(jet['first'][:,:,:d]+np.einsum('pijk,pj->pik',A1,b)+np.einsum('pij,pjk->pik',A,db)
            +.5*B1bb+np.einsum('pijk,pj,pka->pia',B,b,db))
        bath_first=A+np.einsum('pijk,pk->pij',B,b)
        indices=np.array([i for i,p in enumerate(path.tree.primitives) if p.kind=='angle'])
        extent=np.full(len(indices),np.pi/2-self.valence_join_radian)
        mapping=SmoothEnvelopeMap(extent,(self.valence_join_radian-self.valence_limit_radian)/extent[0])
        prior=values[:,indices].copy();mapped,J,_=mapping.evaluate(prior-np.pi/2)
        values[:,indices]=np.pi/2+mapped
        first[:,indices]*=J[:,:,None];bath_first[:,indices]*=J[:,:,None]
        return dict(values=values,first=first,bath_first=bath_first,response_fields=v,
            valence_continued=np.any(abs(prior-values[:,indices])>1e-12,axis=1))

    def reference_internal_jet(self):
        d=len(self.chart.path.active_indices);n=len(self.chart.path.bath_indices)
        source=self.chart.internal_jet(np.zeros(d));v=self.response.evaluate(np.zeros((1,d)),second=True,allow_extrapolation=True)
        if np.max(abs(v['response']))>1e-10:raise ValueError('Reference response must vanish for the anchored mean jet')
        J=np.eye(d+n);J[d:,:d]=v['response_first'][0]
        Q=np.zeros((d+n,d+n,d+n));Q[d:,:d,:d]=v['response_second'][0]
        first=source['first']@J
        second=np.einsum('aij,ik,jl->akl',source['second'],J,J,optimize=True)+np.einsum('ai,ijk->ajk',source['first'],Q)
        return dict(values=self.chart.path.reference_values.copy(),first=first,second=second,response_fields=v)
