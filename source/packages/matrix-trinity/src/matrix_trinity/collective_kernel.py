"""Exact common Feshbach kernel and property vertices on a fixed partition.

This is an established projection construction, not a new HDCPT2 closure.
It provides the compatibility baseline for a future regularized self-energy.
No physical eigenvectors or projectors are computed here. Evaluation uses
linear solves; obtaining spectral poles would still be a spectral calculation.
"""
import numpy as np

from .collective_gram import _hermitian


class CollectiveResolventKernel:
    """H=[[A,B†],[B,D]] in one fixed orthonormal coordinate representation.

    Energy blocks and B share units. coordinate_id identifies the ambient basis.
    Property matrices may have arbitrary leading component axes. Complex contour
    energies avoid poles of the eliminated block. No pole snapping, resonance
    threshold or empirical width is introduced.
    """
    def __init__(self, a, d, b, *, coordinate_id, maximum_workspace_bytes=536870912):
        a,d=_hermitian(a,'a'),_hermitian(d,'d')
        b=np.asarray(b,dtype=complex)
        p,q=len(a),len(d)
        if b.shape!=(q,p) or not np.all(np.isfinite(b)):
            raise ValueError('finite b with shape (q,p) required')
        if not isinstance(coordinate_id,str) or not coordinate_id.strip():
            raise ValueError('nonempty coordinate_id required')
        if (isinstance(maximum_workspace_bytes,(bool,np.bool_))
                or not isinstance(maximum_workspace_bytes,(int,np.integer))
                or maximum_workspace_bytes<=0):
            raise ValueError('positive integer maximum_workspace_bytes required')
        estimate=24*16*(p+q)**2
        if estimate>maximum_workspace_bytes:
            raise MemoryError('collective kernel workspace budget exceeded')
        self.a,self.d,self.b=(np.array(x,copy=True) for x in (a,d,b))
        for value in (self.a,self.d,self.b): value.setflags(write=False)
        self.p,self.q=p,q
        self.coordinate_id=coordinate_id
        self.maximum_workspace_bytes=int(maximum_workspace_bytes)

    @staticmethod
    def _energy(z):
        z=complex(z)
        if not np.isfinite(z):
            raise ValueError('finite complex energy required')
        return z

    def partner_resolvent(self,z):
        """Q(z)=(zI-D)^−1. A real partner pole requires another chart/contour."""
        z=self._energy(z)
        return np.linalg.solve(z*np.eye(self.q)-self.d,np.eye(self.q))

    def self_energy(self,z):
        q=self.partner_resolvent(z)
        return self.b.conj().T@q@self.b

    def self_energy_derivative(self,z):
        q=self.partner_resolvent(z)
        return -self.b.conj().T@q@q@self.b

    def reduced_resolvent(self,z):
        z=self._energy(z)
        return np.linalg.solve(z*np.eye(self.p)-self.a-self.self_energy(z),np.eye(self.p))

    def lift(self,z):
        return np.vstack((np.eye(self.p),self.partner_resolvent(z)@self.b))

    def metric(self,z,w=None):
        """Analytic two-energy metric, using products to avoid subtractive loss.

        For real E, metric(E,E)=I-Sigma'(E) is positive. For complex energies
        the formula is analytic and is not a Hilbert-space positive norm.
        """
        if w is None: w=z
        qz,qw=self.partner_resolvent(z),self.partner_resolvent(w)
        return np.eye(self.p)+self.b.conj().T@qz@qw@self.b

    def property_vertex(self,z,w,operator):
        """Gamma_O(z,w)=Omega(z*)† O Omega(w), with analytic left leg.

        At equal real energies this includes mechanical dressing of every
        supplied electrical derivative; it does not generate missing derivatives.
        No electric, magnetic or tensor-component specialization is assumed.
        """
        value=np.asarray(operator,dtype=complex)
        size=self.p+self.q
        if value.ndim<2 or value.shape[-2:]!=(size,size) or not np.all(np.isfinite(value)):
            raise ValueError('finite property with full ambient matrix dimensions required')
        output_size=int(np.prod(value.shape[:-2],dtype=object))*self.p**2
        if 4*16*output_size>self.maximum_workspace_bytes:
            raise MemoryError('property vertex output budget exceeded')
        left=np.hstack((np.eye(self.p),self.b.conj().T@self.partner_resolvent(z)))
        right=self.lift(w)
        return left@value@right

    def projected_double_resolvent(self,z,w,operator):
        return self.reduced_resolvent(z)@self.property_vertex(z,w,operator)@self.reduced_resolvent(w)

    def resolvent(self,z):
        """Reconstruct (zI-H)^−1; no eigenvalues/eigenvectors are returned."""
        q=self.partner_resolvent(z)
        g=self.reduced_resolvent(z)
        qb=q@self.b
        bq=self.b.conj().T@q
        return np.block([[g,g@bq],[qb@g,q+qb@g@bq]])
