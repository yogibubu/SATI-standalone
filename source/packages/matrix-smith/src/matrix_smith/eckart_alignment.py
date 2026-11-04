"""Cartesian two-jet of the mass Eckart alignment in a regular rotation domain."""
import numpy as np
from .curvilinear_tunneling import mass_eckart_align
from .eckart_chart import eckart_rigid_constraint_rows


class EckartAlignment:
    """Remove rigid pose while differentiating the alignment itself.

    The implicit rotational stationarity equations supply first and second
    derivatives. Singular/nonunique rotational fits are rejected. No finite
    differences of rotations or Cartesian field values are used.
    """
    def __init__(self,reference,masses):
        self.reference=np.array(reference,float,copy=True);self.masses=np.array(masses,float,copy=True)
        self.rigid=eckart_rigid_constraint_rows(self.reference,self.masses)
        self.rotation_constraints=self.rigid[3:]

    def value(self,xyz):
        return mass_eckart_align(xyz,self.reference,self.masses)

    def differential(self,xyz):
        """Return aligned geometry, first derivative and a curvature contraction.

        The callable contracts the second derivative with a Cartesian covector
        in O(ncart**2) storage, avoiding a Cartesian third-rank array.
        """
        xyz=np.asarray(xyz,float)
        if xyz.shape!=self.reference.shape or not np.all(np.isfinite(xyz)):
            raise ValueError('finite matching Cartesian geometry required')
        aligned,rotation=mass_eckart_align(xyz,self.reference,self.masses,return_rotation=True)
        n=len(xyz);d=xyz.size
        centered=aligned-np.average(self.reference,axis=0,weights=self.masses)
        # t is the derivative with the current proper rotation held fixed.
        eye=np.eye(d).reshape(n,3,d)
        eye-=np.einsum('i,iaj->aj',self.masses,eye)[None]/self.masses.sum()
        t=np.einsum('iaj,ab->ibj',eye,rotation).reshape(d,d)
        axes=np.eye(3)
        rotations=np.cross(axes[:,None,:],centered[None,:,:]).transpose(1,2,0).reshape(d,3)
        c=self.rotation_constraints;system=c@rotations
        singular=np.linalg.svd(system,compute_uv=False)
        if singular[-1]<=1e-12*singular[0]:
            raise ValueError('Eckart alignment has a singular rotational derivative')
        omega=np.linalg.solve(system,-c@t)
        first=t+rotations@omega
        trows=t.reshape(n,3,d).transpose(0,2,1)
        def curvature_contraction(covector):
            g=np.asarray(covector,float)
            if g.shape!=(d,) or not np.all(np.isfinite(g)):
                raise ValueError('finite aligned Cartesian covector required')
            g=(g-c.T@np.linalg.solve(system.T,rotations.T@g)).reshape(n,3)
            cross=np.cross(trows,g[:,None,:]).sum(axis=0).T
            mixed=omega.T@cross
            shape=centered.T@g-np.sum(centered*g)*np.eye(3)
            nested=omega.T@shape@omega
            return mixed+mixed.T+.5*(nested+nested.T)
        return aligned,first,curvature_contraction

    def jet(self,xyz):
        """Explicit second geometry derivative for small diagnostic problems."""
        aligned,first,contract=self.differential(xyz)
        second=np.array([contract(e) for e in np.eye(np.size(xyz))])
        return aligned,first,second
