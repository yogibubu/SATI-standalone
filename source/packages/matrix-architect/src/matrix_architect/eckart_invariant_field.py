"""Rigid-invariant scalar fields and reaction progress with complete two-jets."""
import numpy as np
from types import SimpleNamespace
from matrix_chem import BOHR_TO_ANGSTROM
from matrix_smith.eckart_alignment import EckartAlignment
from .native_pic_field import NativePicPoint


class EckartInvariantField:
    def __init__(self,base,reference,masses):
        self.base=base;self.atoms=base.atoms
        self.alignment=EckartAlignment(reference,masses)

    def __call__(self,xyz):
        x,a,curvature=self.alignment.differential(xyz);p=self.base(x)
        return NativePicPoint(p.energy_hartree,a.T@p.gradient_hartree_per_bohr,
            a.T@p.hessian_hartree_per_bohr2@a+BOHR_TO_ANGSTROM*curvature(p.gradient_hartree_per_bohr))

    def energy_gradient(self,xyz):
        p=self(xyz);return p.energy_hartree,p.gradient_hartree_per_bohr


class EckartInvariantProgress:
    def __init__(self,base,reference,masses):
        self.base=base;self.alignment=EckartAlignment(reference,masses)

    def value(self,xyz):return self.base.value(self.alignment.value(xyz))

    def __call__(self,xyz):
        x,a,curvature=self.alignment.differential(xyz);p=self.base(x)
        return SimpleNamespace(value=p.value,gradient_per_bohr=a.T@p.gradient_per_bohr,
            hessian_per_bohr2=a.T@p.hessian_per_bohr2@a+BOHR_TO_ANGSTROM*curvature(p.gradient_per_bohr))


class EckartInvariantKernel:
    """Native coordinate values and derivatives per angstrom in a fixed gauge."""
    def __init__(self,base,reference,masses):
        self.base=base;self.alignment=EckartAlignment(reference,masses)

    def value(self,xyz):
        x,a,curvature=self.alignment.differential(xyz);u,j,h=self.base.value(x)
        second=np.array([a.T@h[:,:,i]@a+curvature(j[:,i]) for i in range(len(u))])
        return u,a.T@j,second.transpose(1,2,0)
