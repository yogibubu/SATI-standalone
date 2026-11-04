"""One common Cartesian mean-response embedding for force and metric jets."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CartesianResponsePath:
    baseline: object
    response: object

    def __post_init__(self):
        if self.response.values.shape[1]!=3*len(self.baseline.tree.order):
            raise ValueError('Cartesian response outputs must match every atom and Cartesian component')

    def evaluate(self,points,*,second=False):
        q=np.asarray(points,float);result=self.baseline.evaluate(q,second=second)
        x=result['geometry_A'];d=q.shape[1];eye=np.eye(d,dtype=int)
        result['geometry_A']=x+self.response.evaluate(q).reshape(x.shape)
        for i in range(d):
            result['tangents'][:,:,:,i]+=self.response.evaluate(q,eye[i]).reshape(x.shape)
            if second:
                for j in range(i,d):
                    term=self.response.evaluate(q,eye[i]+eye[j]).reshape(x.shape)
                    result['second_derivatives'][:,:,:,i,j]+=term
                    if i!=j:result['second_derivatives'][:,:,:,j,i]+=term
        return result


def dihedral_batch(coordinates,atoms):
    """The ORACLE primitive dihedral convention evaluated on many geometries."""
    x=np.asarray(coordinates,float);a,b,c,d=atoms
    v0=x[:,a]-x[:,b];v1=x[:,c]-x[:,b];v2=x[:,d]-x[:,c]
    n0=np.cross(v0,v1);n1=np.cross(v1,v2)
    norm=np.linalg.norm(v1,axis=1)
    if np.any(norm<1e-12) or np.any(np.linalg.norm(n0,axis=1)<1e-12) or np.any(np.linalg.norm(n1,axis=1)<1e-12):
        raise ValueError('Nonsingular dihedral planes required')
    return np.arctan2(np.einsum('pi,pi->p',np.cross(n0,n1),v1)/norm,np.einsum('pi,pi->p',n0,n1))
