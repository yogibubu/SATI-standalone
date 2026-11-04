"""Bounded reactive ZAFF functions involving both exchanging bonds.

The ratio of two native inverse-power bond orders supplies reaction progress;
it is not a polynomial correction in the original stretching displacement.
Explicit powers select the radial family; no universal fitted power is assumed.
"""
import numpy as np
from scipy.special import expit
from dataclasses import dataclass
from matrix_chem import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class BondOrderJet:
    value: float
    gradient_per_bohr: np.ndarray
    hessian_per_bohr2: np.ndarray


class ReactiveBondOrderFunctions:
    """xi=(b_form-b_break)/(b_form+b_break), b=(Rc/R)**p.

    Progress is bounded and reverses sign on exchange. The product of bounded
    occupancies b/(1+b) is an overlap window vanishing in either separated
    channel, also when both changing bonds separate. Original electrostatic
    and dispersion terms are supplied separately by the resident ZAFF fields.
    Powers/reference distances are explicit model inputs, not accuracy claims.
    """
    def __init__(self,natoms,breaking_bond,forming_bond,reference_distances,powers):
        self.natoms=int(natoms)
        self.bonds=tuple(tuple(sorted(map(int,p))) for p in (breaking_bond,forming_bond))
        self.references=np.asarray(reference_distances,float);self.powers=np.asarray(powers,float)
        if (self.natoms<2 or self.bonds[0]==self.bonds[1]
                or any(len(p)!=2 or p[0]==p[1] or min(p)<0 or max(p)>=self.natoms for p in self.bonds)
                or self.references.shape!=(2,) or self.powers.shape!=(2,)
                or not np.all(np.isfinite(self.references)) or not np.all(np.isfinite(self.powers))
                or np.any(self.references<=0) or np.any(self.powers<=0)):
            raise ValueError('two distinct exchanging bonds and positive native reference distances/powers required')

    def _log_orders(self,xyz):
        xyz=np.asarray(xyz,float)
        if xyz.shape!=(self.natoms,3) or not np.all(np.isfinite(xyz)):
            raise ValueError('finite geometry matching the reactive atom order required')
        result=[];d=xyz.size
        for (i,j),rc,p in zip(self.bonds,self.references,self.powers):
            delta=xyz[i]-xyz[j];r=float(np.linalg.norm(delta))
            if r<=0:raise ValueError('coincident atoms in exchanging bond')
            unit=delta/r;g=np.zeros(d);g[3*i:3*i+3]=-p/r*unit;g[3*j:3*j+3]=p/r*unit
            local=p/r**2*(2*np.outer(unit,unit)-np.eye(3));h=np.zeros((d,d))
            for a,sa in ((i,1),(j,-1)):
                for b,sb in ((i,1),(j,-1)):h[3*a:3*a+3,3*b:3*b+3]=sa*sb*local
            result.append((float(p*np.log(rc/r)),g*BOHR_TO_ANGSTROM,h*BOHR_TO_ANGSTROM**2))
        return result

    def progress(self,xyz):
        a,b=self._log_orders(xyz);logratio=(b[0]-a[0])/2
        g=(b[1]-a[1])/2;h=(b[2]-a[2])/2
        value=float(np.tanh(logratio));first=1-value*value
        return BondOrderJet(value,first*g,first*(h-2*value*np.outer(g,g)))

    def occupancy(self,xyz,bond_index):
        if bond_index not in (0,1):raise ValueError('breaking (0) or forming (1) bond required')
        v,g,h=self._log_orders(xyz)[bond_index];value=float(expit(v));first=value*(1-value)
        return BondOrderJet(value,first*g,first*(h+(1-2*value)*np.outer(g,g)))

    def overlap(self,xyz):
        jets=self._log_orders(xyz);d=np.size(xyz);logw=0.;g=np.zeros(d);h=np.zeros((d,d))
        for value,first,second in jets:
            occupation=expit(value);logw-=np.logaddexp(0.,-value)
            g+=(1-occupation)*first
            h+=(1-occupation)*second-occupation*(1-occupation)*np.outer(first,first)
        value=float(np.exp(logw))
        return BondOrderJet(value,value*g,value*(h+np.outer(g,g)))
