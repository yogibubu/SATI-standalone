"""Reference E/G/H embedded in caller-supplied nonredundant native variables."""
import numpy as np
from matrix_chem import BOHR_TO_ANGSTROM
from .native_pic_field import NativePicPoint


class NativeLocalField:
    def __init__(self,atoms,kernel,energy,linear,quadratic,quartic_model=None):
        self.atoms=tuple(atoms);self.kernel=kernel;self.energy=float(energy)
        self.linear=np.asarray(linear,float);self.quadratic=np.asarray(quadratic,float)
        self.quartic_model=quartic_model
        n=len(self.linear)
        if (not self.atoms or self.linear.shape!=(n,) or self.quadratic.shape!=(n,n)
                or not np.isfinite(self.energy) or not np.all(np.isfinite(self.linear))
                or not np.all(np.isfinite(self.quadratic))
                or not np.allclose(self.quadratic,self.quadratic.T)):
            raise ValueError('finite aligned native field coefficients required')

    @classmethod
    def from_cartesian_jet(cls,atoms,kernel,xyz,energy,gradient,hessian,*,rigid_constraint_rows=None):
        """Cartesian input units are hartree and angstrom; output uses bohr.

        With frozen frame coordinates the inverse must use the same rigid
        constraints as the vibrational geometry. The resulting field matches
        the intrinsic E/G/H jet; it need not reproduce unphysical rigid torques
        of frozen coordinate functions away from their reference geometry.
        """
        u,j,h=kernel.value(xyz);gradient=np.asarray(gradient,float);hessian=np.asarray(hessian,float)
        if (gradient.shape!=(np.size(xyz),) or hessian.shape!=(np.size(xyz),np.size(xyz))
                or not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian))
                or not np.allclose(hessian,hessian.T) or not np.allclose(u,0.,atol=1e-10)
                or np.linalg.matrix_rank(j)!=len(u)):
            raise ValueError('centered independent native variables and finite Cartesian jet required')
        if rigid_constraint_rows is None:
            inverse=np.linalg.pinv(j,rcond=1e-10)
        else:
            rigid=np.asarray(rigid_constraint_rows,float)
            if (rigid.ndim!=2 or rigid.shape[1]!=np.size(xyz)
                    or rigid.shape[0]+len(u)!=np.size(xyz)
                    or not np.all(np.isfinite(rigid))):
                raise ValueError('complete finite native plus rigid constraints required')
            first=np.linalg.solve(np.vstack((j.T,rigid)),
                np.vstack((np.eye(len(u)),np.zeros((len(rigid),len(u))))))
            inverse=first.T
        linear=inverse@gradient
        residual=hessian-np.einsum('t,ijt->ij',linear,h)
        quadratic=inverse@residual@inverse.T
        return cls(atoms,kernel,energy,linear,(quadratic+quadratic.T)/2)

    def __call__(self,xyz):
        u,g,h=self.kernel.value(xyz)
        if self.quartic_model is None:
            e=self.energy+self.linear@u+.5*u@self.quadratic@u
            gu=self.linear+self.quadratic@u;hu=self.quadratic
        else:
            de,gu,hu=self.quartic_model.potential_jet(u);e=self.energy+de
        return NativePicPoint(float(e),(g@gu)*BOHR_TO_ANGSTROM,
            (g@hu@g.T+np.einsum('t,ijt->ij',gu,h))*BOHR_TO_ANGSTROM**2)

    def energy_gradient(self,xyz):
        p=self(xyz)
        return p.energy_hartree,p.gradient_hartree_per_bohr
