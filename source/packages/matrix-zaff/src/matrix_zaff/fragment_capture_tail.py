"""Neutral-fragment capture asymptotes using resident ZAFF dipole tensors.

Permanent dipole coupling, leading dipole-induced-dipole induction with
isotropic fragment polarizabilities, and supplied dispersion C6 are retained.
Higher multipoles and short-range exchange belong to the original ZAFF prior.
No coefficient is inferred from a reference reaction rate.
"""
from dataclasses import dataclass
import numpy as np
from .polarization import _dipole_tensor

BOHR_A=.529177210903
HARTREE_CM=219474.6313705


@dataclass(frozen=True)
class NeutralFragmentCaptureTail:
    dipole_a_e_bohr: np.ndarray
    dipole_b_e_bohr: np.ndarray
    polarizability_a_bohr3: float
    polarizability_b_bohr3: float
    dispersion_c6_hartree_bohr6: float
    provenance: str

    def __post_init__(self):
        for name in ('dipole_a_e_bohr','dipole_b_e_bohr'):
            value=np.array(getattr(self,name),float,copy=True)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError('finite space-fixed dipoles in e Bohr required')
            value.setflags(write=False);object.__setattr__(self,name,value)
        if (any(not np.isfinite(value) or value < 0 for value in (
                self.polarizability_a_bohr3,self.polarizability_b_bohr3,self.dispersion_c6_hartree_bohr6))
                or not self.provenance):
            raise ValueError('nonnegative polarizabilities/C6 and explicit provenance required')

    def radial_jets(self,radius_angstrom,direction):
        direction=np.asarray(direction,float)
        if (direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction)==0
                or not np.isfinite(radius_angstrom) or radius_angstrom <= BOHR_A):
            raise ValueError('finite nonzero separation direction and a long-range radius above one Bohr required')
        unit=direction/np.linalg.norm(direction)
        # The point limit of the existing Thole tensor is exact at this scale.
        tensor=_dipole_tensor(unit,1e-12,1e-12,.39)
        a,b=self.dipole_a_e_bohr,self.dipole_b_e_bohr
        c3=-float(a@tensor@b)
        induction=.5*(self.polarizability_a_bohr3*np.sum((tensor@b)**2)
                       +self.polarizability_b_bohr3*np.sum((tensor@a)**2))
        c6=self.dispersion_c6_hartree_bohr6+induction
        radius=radius_angstrom/BOHR_A
        value=c3/radius**3-c6/radius**6
        first=-3*c3/radius**4+6*c6/radius**7
        second=12*c3/radius**5-42*c6/radius**8
        return np.array([value,first/BOHR_A,second/BOHR_A**2])*HARTREE_CM
