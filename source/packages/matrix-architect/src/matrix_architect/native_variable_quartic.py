"""Local quartic constraints represented in separable native variables.

This is a conditional local field, not a complete dissociative molecular ZAFF.
A caller must supply independent physical coordinates and the molecular support
functions when embedding it in a global potential.
"""
from itertools import combinations_with_replacement, permutations
from math import factorial
import numpy as np


def _contract(tensor, matrix):
    degree=np.ndim(tensor)
    old='ijkl'[:degree];new='abcd'[:degree]
    return np.einsum(old+','+','.join(i+j for i,j in zip(old,new))+'->'+new,
                     tensor,*([matrix]*degree),optimize=True)


class NativeVariableQuartic:
    """Preserve a reference four-jet in inverse-distance/linear variables.

    The inverse-distance coordinate is exactly the one used in native ZAFF:
    u=R0**2*(1/R0-1/R). All mixed products of supplied derivatives are retained.
    Finite radial limits do not certify the shape away from the local domain.
    """
    def __init__(self, reference, hessian, cubic, quartic, *, radial_indices):
        self.reference=np.asarray(reference,float)
        if self.reference.ndim!=1 or not np.all(np.isfinite(self.reference)):
            raise ValueError('finite one-dimensional physical reference required')
        n=len(self.reference);self.radial=tuple(radial_indices)
        if (any(not isinstance(i,(int,np.integer)) or not 0<=i<n or self.reference[i]<=0 for i in self.radial)
                or len(set(self.radial))!=len(self.radial)):
            raise ValueError('distinct positive physical radial coordinates required')
        tensors=[np.asarray(t,float) for t in (hessian,cubic,quartic)]
        if any(t.shape!=(n,)*k or not np.all(np.isfinite(t)) for k,t in enumerate(tensors,2)):
            raise ValueError('finite complete reference tensors required')
        radial=set(self.radial);poly={}
        # Compose x(u)=u+u²/R0+u³/R0²+u⁴/R0³ and truncate total degree at four.
        for degree,tensor in enumerate(tensors,2):
            for key in combinations_with_replacement(range(n),degree):
                counts={i:key.count(i) for i in set(key)}
                value=float(tensor[key])/np.prod([factorial(v) for v in counts.values()])
                if value==0.:continue
                terms={():value}
                for i in key:
                    expanded={}
                    for indices,coefficient in terms.items():
                        for power in range(1,5-len(indices)) if i in radial else ((1,) if len(indices)<4 else ()):
                            new=tuple(sorted(indices+(i,)*power))
                            expanded[new]=expanded.get(new,0.)+coefficient*(self.reference[i]**(1-power) if i in radial else 1.)
                    terms=expanded
                for indices,coefficient in terms.items():
                    poly[indices]=poly.get(indices,0.)+coefficient
        self.tensors=[np.zeros((n,)*k) for k in range(1,5)]
        for key,value in poly.items():
            coefficient=value*np.prod([factorial(key.count(i)) for i in set(key)])
            for indices in set(permutations(key)):self.tensors[len(key)-1][indices]=coefficient

    def variable_jet(self, coordinates):
        x=np.asarray(coordinates,float)
        if x.shape!=self.reference.shape or not np.all(np.isfinite(x)):
            raise ValueError('finite aligned independent physical coordinates required')
        u=x-self.reference;a=np.ones(len(x));b=np.zeros(len(x));c=b.copy();d=b.copy()
        for i in self.radial:
            if x[i]<=0:raise ValueError('positive bond length required')
            r=self.reference[i]
            u[i]=r*r*(1/r-1/x[i]);a[i]=r*r/x[i]**2
            b[i]=-2*r*r/x[i]**3;c[i]=6*r*r/x[i]**4;d[i]=-24*r*r/x[i]**5
        return u,a,b,c,d

    @staticmethod
    def _chain(p1,p2,p3,p4,a,b,c,d):
        n=len(a)
        h=p2*a[:,None]*a[None,:]+np.diag(p1*b)
        third=p3*np.einsum('i,j,k->ijk',a,a,a)
        fourth=p4*np.einsum('i,j,k,l->ijkl',a,a,a,a)
        for i in range(n):
            third[i,i,:]+=p2[i,:]*b[i]*a
            third[i,:,i]+=p2[i,:]*b[i]*a
            third[:,i,i]+=p2[:,i]*b[i]*a
            third[i,i,i]+=p1[i]*c[i]
            for pair in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                index=[slice(None)]*4;index[pair[0]]=index[pair[1]]=i
                fourth[tuple(index)]+=p3[i,:,:]*b[i]*np.outer(a,a)
            for j in range(n):
                for indices in {(i,i,j,j),(i,j,i,j),(i,j,j,i)}:
                    # Ordered i,j and its reverse supply the six placements;
                    # the i==j case has three pair partitions, counted below.
                    if i!=j:fourth[indices]+=p2[i,j]*b[i]*b[j]
                if i==j:fourth[i,i,i,i]+=3*p2[i,i]*b[i]**2
                for single in range(4):
                    indices=[i]*4;indices[single]=j
                    fourth[tuple(indices)]+=p2[i,j]*c[i]*a[j]
            fourth[i,i,i,i]+=p1[i]*d[i]
        return h,third,fourth

    def derivatives(self, coordinates, *, native_quadratic_correction=None):
        u,a,b,c,d=self.variable_jet(coordinates)
        t1,t2,t3,t4=self.tensors
        p1=t1+t2@u+.5*np.einsum('ijk,j,k->i',t3,u,u)+np.einsum('ijkl,j,k,l->i',t4,u,u,u)/6
        p2=t2+np.einsum('ijk,k->ij',t3,u)+.5*np.einsum('ijkl,k,l->ij',t4,u,u)
        p3=t3+np.einsum('ijkl,l->ijk',t4,u)
        h,third,fourth=self._chain(p1,p2,p3,t4,a,b,c,d)
        if native_quadratic_correction is not None:
            k=np.asarray(native_quadratic_correction,float)
            if k.shape!=h.shape or not np.allclose(k,k.T):raise ValueError('symmetric native quadratic correction required')
            corrections=self._chain(np.zeros(len(a)),k,np.zeros_like(third),np.zeros_like(fourth),a,b,c,d)
            h,third,fourth=[x+y for x,y in zip((h,third,fourth),corrections)]
        return h,third,fourth
