"""Valence exchange includes the bonds that form as well as those that break.

State-specific native ZAFF fields supply bonded/angular/torsional energies.
This module supplies smooth, reversible EVB couplings; it acquires no PES data.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.special import expit
from matrix_chem import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class ValenceConnectivity:
    natoms: int
    bonds: tuple

    def __post_init__(self):
        if not isinstance(self.natoms,int) or isinstance(self.natoms,bool) or self.natoms<2:
            raise ValueError('positive molecular atom count required')
        canonical=[]
        for pair in self.bonds:
            if (len(pair)!=2 or any(not isinstance(i,(int,np.integer)) or isinstance(i,bool)
                    or not 0<=i<self.natoms for i in pair) or pair[0]==pair[1]):
                raise ValueError('distinct zero-based atom indices required for every bond')
            canonical.append(tuple(sorted(map(int,pair))))
        if len(set(canonical))!=len(canonical):raise ValueError('duplicate valence bond')
        object.__setattr__(self,'bonds',tuple(sorted(canonical)))

    def native_primitives(self,coordinates_angstrom,atomic_numbers):
        """Build every state's own bonded terms, including newly formed angles.

        The shared primitive builder retains its linear-bend handling. A
        runtime must support the resulting primitives or explicitly reject
        them; no linear angle is silently converted to an ordinary bend.
        """
        from matrix_chem.primitive_coordinates import build_primitives
        xyz=np.asarray(coordinates_angstrom,float)
        if xyz.shape!=(self.natoms,3) or len(atomic_numbers)!=self.natoms or not np.all(np.isfinite(xyz)):
            raise ValueError('geometry and elements must match the valence state')
        adjacency=[set() for _ in range(self.natoms)]
        for i,j in self.bonds:adjacency[i].add(j);adjacency[j].add(i)
        graph=SimpleNamespace(natoms=self.natoms,Z=atomic_numbers,bonds=self.bonds,adjacency=adjacency)
        return tuple(build_primitives(graph,xyz,include_pseudo_bonds=False))


class ValenceExchangeCoupling:
    """H_ab = A_ab product[b in B_a symmetric-difference B_b] w_b(R_b).

    w = expit[-alpha (R-Rc)] is a continuous bond occupancy. All changing
    bonds are included, so forming bonds activate the exchange and cleaving
    bonds deactivate it. Compression never turns an occupied bond off.
    Both single-channel separation limits, and separation of every changing
    bond, have zero coupling. Swapping state labels leaves EVB eigenvalues
    unchanged. Rc and alpha must be explicitly supplied from the frozen
    reference/catalogue or fitted; no universal numerical values are assumed.

    This is a coupling model, not a qualification of the diagonal ZAFF fields
    or of molecular rates. E/G/H follow the resident EVB bohr convention.
    """
    def __init__(self,states,amplitudes_hartree,reference_distances_angstrom,
                 inverse_widths_per_angstrom):
        self.states=tuple(states)
        if (len(self.states)<2 or any(not isinstance(s,ValenceConnectivity) for s in self.states)
                or len({s.natoms for s in self.states})!=1
                or len({s.bonds for s in self.states})!=len(self.states)):
            raise ValueError('distinct valence connectivities in the same atom order required')
        self.natoms=self.states[0].natoms;n=len(self.states)
        amp=np.asarray(amplitudes_hartree,float)
        if amp.shape!=(n,n) or not np.all(np.isfinite(amp)) or not np.allclose(amp,amp.T,atol=0.,rtol=0.) or np.any(np.diag(amp)!=0):
            raise ValueError('finite symmetric off-diagonal coupling amplitudes required')
        self.amplitudes=amp.copy();self.exchanges={}
        for a in range(n):
            for b in range(a+1,n):self.exchanges[a,b]=tuple(sorted(set(self.states[a].bonds)^set(self.states[b].bonds)))
        self.reactive_bonds=tuple(sorted(set().union(*map(set,self.exchanges.values()))))
        def parameters(values):
            result={tuple(sorted(k)):float(v) for k,v in values.items()}
            if set(result)!=set(self.reactive_bonds) or any(not np.isfinite(v) or v<=0 for v in result.values()):
                raise ValueError('positive explicit parameters for every forming and breaking bond required')
            return result
        self.reference_distances=parameters(reference_distances_angstrom)
        self.inverse_widths=parameters(inverse_widths_per_angstrom)
        self.schema='matrix.architect.valence_exchange_coupling.v1'

    def __call__(self,coordinates_angstrom):
        xyz=np.asarray(coordinates_angstrom,float);d=3*self.natoms;n=len(self.states)
        if xyz.shape!=(self.natoms,3) or not np.all(np.isfinite(xyz)):
            raise ValueError('finite Cartesian geometry in the declared atom order required')
        jets={}
        for pair in self.reactive_bonds:
            i,j=pair;delta=xyz[i]-xyz[j];r=float(np.linalg.norm(delta))
            if r<=0:raise ValueError('coincident atoms in reactive bond')
            alpha=self.inverse_widths[pair];t=alpha*(r-self.reference_distances[pair]);w=expit(-t)
            logw=-np.logaddexp(0.,t);first=-alpha*expit(t);second=-alpha*alpha*w*expit(t)
            u=delta/r;g=np.zeros(d);g[3*i:3*i+3]=first*u;g[3*j:3*j+3]=-first*u
            local=second*np.outer(u,u)+first/r*(np.eye(3)-np.outer(u,u));h=np.zeros((d,d))
            for a,sa in ((i,1),(j,-1)):
                for b,sb in ((i,1),(j,-1)):h[3*a:3*a+3,3*b:3*b+3]=sa*sb*local
            jets[pair]=(logw,g,h)
        matrix=np.zeros((n,n));gradient=np.zeros((n,n,d));hessian=np.zeros((n,n,d,d))
        for (a,b),pairs in self.exchanges.items():
            logw=sum(jets[p][0] for p in pairs);g=sum((jets[p][1] for p in pairs),np.zeros(d))
            h=sum((jets[p][2] for p in pairs),np.zeros((d,d)))
            value=self.amplitudes[a,b]*np.exp(logw)
            matrix[a,b]=matrix[b,a]=value
            gradient[a,b]=gradient[b,a]=value*g*BOHR_TO_ANGSTROM
            hessian[a,b]=hessian[b,a]=value*(h+np.outer(g,g))*BOHR_TO_ANGSTROM**2
        return SimpleNamespace(matrix=matrix,gradient=gradient,hessian=hessian)
