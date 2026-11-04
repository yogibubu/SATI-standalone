"""Adiabatic elimination of a few SAMs coupled to a periodic local rotor.

The model is V(phi,u)=V_Fourier(phi)+u.K.u/2+g.u*sin(m*phi)/m,
with mass-weighted SAM coordinates u in Angstrom sqrt(amu). K and g are
in cm^-1/u^2 and cm^-1/u/radian. Positive K is held fixed with angle.
Its harmonic SAM elimination is analytic; no multidimensional DVR is used.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PeriodicSAMRelaxation:
    periodicity: int
    energy_amplitude_cm: float
    inertia_amplitude_amu_A2: float
    mixed_inertia_amu_A2: float
    selected_indices: tuple
    selected_groups: tuple
    omitted_energy_bound_cm: float
    omitted_inertia_bound_amu_A2: float
    omitted_mixed_inertia_bound_amu_A2: float
    displacement_amplitudes: np.ndarray

    def potential_correction_cm(self, angles):
        return -self.energy_amplitude_cm*np.sin(self.periodicity*np.asarray(angles))**2

    def inertia_correction_amu_A2(self, angles):
        c=np.cos(self.periodicity*np.asarray(angles))
        return self.inertia_amplitude_amu_A2*c*c-2*self.mixed_inertia_amu_A2*c


def build_periodic_sam_relaxation(stiffness_cm, mixed_force_cm, *, periodicity,
        reference_inertia_amu_A2, mixed_mass_metric=None,
        energy_tolerance_cm=10., relative_inertia_tolerance=.01,
        degeneracy_relative_tolerance=1e-8):
    """Greedy selection of degenerate mode groups meeting BOTH budgets.

    Groups are ranked by their largest normalized energy or inertia error.
    Whole degenerate subspaces are retained, making the relaxed potential
    and metric independent of their arbitrary eigenvector orientation.
    Returned omission bounds hold uniformly for all torsional angles.
    """
    k=np.asarray(stiffness_cm,float);g=np.asarray(mixed_force_cm,float)
    n=len(g) if g.ndim==1 else 0
    c=np.zeros(n) if mixed_mass_metric is None else np.asarray(mixed_mass_metric,float)
    moment=float(reference_inertia_amu_A2)
    if (n<1 or k.shape!=(n,n) or c.shape!=(n,)
        or isinstance(periodicity,bool) or int(periodicity)!=periodicity or periodicity<1
        or moment<=0 or energy_tolerance_cm<=0 or relative_inertia_tolerance<=0
        or degeneracy_relative_tolerance<0
        or not all(np.all(np.isfinite(a)) for a in (k,g,c,moment,
            energy_tolerance_cm,relative_inertia_tolerance,degeneracy_relative_tolerance))
        or not np.allclose(k,k.T,rtol=1e-12,atol=1e-10)):
        raise ValueError('finite symmetric SAM stiffness, matching couplings and positive tolerances/inertia required')
    eigen,v=np.linalg.eigh(k)
    if np.any(eigen<=0):raise ValueError('positive conditional SAM stiffness required')
    if moment<=float(c@c):raise ValueError('full local LAM/SAM mass metric must be positive')
    force=v.T@g;metric=v.T@c;amplitudes=force/eigen
    energy=force*amplitudes/(2*periodicity**2)
    inertia=amplitudes**2
    groups=[]
    for i in range(n):
        if not groups or abs(eigen[i]-eigen[groups[-1][0]])>degeneracy_relative_tolerance*max(eigen[i],eigen[groups[-1][0]]):
            groups.append([i])
        else:groups[-1].append(i)
    # The mixed-metric bound uses a norm on each complete subspace so it is
    # also invariant under rotations of a degenerate eigenbasis.
    group_mixed=[float(np.linalg.norm(metric[ids])*np.linalg.norm(amplitudes[ids])) for ids in groups]
    mass_budget=relative_inertia_tolerance*moment
    order=sorted(range(len(groups)),key=lambda j:max(
        float(energy[groups[j]].sum())/energy_tolerance_cm,
        (float(inertia[groups[j]].sum())+2*group_mixed[j])/mass_budget),reverse=True)
    selected=[];remaining=set(range(len(groups)))
    def errors():
        ids=[i for j in remaining for i in groups[j]]
        return float(energy[ids].sum()),float(inertia[ids].sum()),sum(group_mixed[j] for j in remaining)
    for j in order:
        e,b,m=errors()
        if e<=energy_tolerance_cm and b+2*m<=mass_budget:break
        selected.append(j);remaining.remove(j)
    e,b,m=errors();indices=tuple(sorted(i for j in selected for i in groups[j]))
    ids=list(indices);response=np.zeros(n)
    if ids:response=v[:,ids]@amplitudes[ids]
    return PeriodicSAMRelaxation(int(periodicity),float(energy[ids].sum()),
        float(response@response),float(c@response),indices,
        tuple(tuple(groups[j]) for j in selected),e,b,float(m),response)


def coupled_periodic_reference(cartesian_hessian,cartesian_gradient,back_transform,
        second_back_transform,masses_amu,*,phase_indices,sam_indices):
    """Coupled local phases after analytic elimination of a shared SAM bath.

    Coordinate first/second back transforms come from SMITH. The complete
    curvilinear Hessian, including gradient times coordinate curvature, is
    transformed by its existing owner. Physical phases are in radians; the
    remaining coordinates may use any consistent normalization. Fixed reaction
    coordinates are omitted from both index sets, not relaxed. The returned
    full 2D curvature and mass matrix are harmonic reference data, not a global
    periodic potential or evidence of high-amplitude accuracy.
    """
    from matrix_smith.curvilinear_hessian import curvilinear_hessian
    A=np.asarray(back_transform,float);m=np.asarray(masses_amu,float)
    p=tuple(phase_indices);s=tuple(sam_indices)
    if (A.ndim!=2 or A.shape[0]!=3*len(m) or len(p)!=2 or len(set(p+s))!=len(p+s)
            or not len(s) or any(isinstance(i,(bool,np.bool_))or not isinstance(i,(int,np.integer))or not 0<=i<A.shape[1]for i in p+s)
            or not np.all(np.isfinite(m))or np.any(m<=0)):
        raise ValueError('two physical phase indices, disjoint SAM indices and matching positive masses required')
    H=curvilinear_hessian(cartesian_hessian,A,cartesian_gradient,second_back_transform=second_back_transform)
    K=H[np.ix_(s,s)];g=H[np.ix_(s,p)]
    if np.linalg.eigvalsh(K).min()<=0:raise ValueError('shared conditional SAM stiffness must be positive')
    response=-np.linalg.solve(K,g)
    relaxed=H[np.ix_(p,p)]+g.T@response
    tangents=A[:,p]+A[:,s]@response
    metric=tangents.T@(np.repeat(m,3)[:,None]*tangents)
    if np.linalg.eigvalsh(metric).min()<=0:raise ValueError('adiabatic phase metric must remain positive')
    return dict(rigid_phase_hessian=H[np.ix_(p,p)],relaxed_phase_hessian=relaxed,
        shared_sam_hessian=K,phase_sam_hessian=g,sam_response=response,
        phase_tangents=tangents,inertia_amu_A2=metric,
        minimum_relaxed_phase_curvature=float(np.linalg.eigvalsh(relaxed).min()),
        minimum_phase_inertia_amu_A2=float(np.linalg.eigvalsh(metric).min()),
        maximum_potential_correlation=float(abs(relaxed[0,1])/np.sqrt(abs(relaxed[0,0]*relaxed[1,1]))),
        kinetic_correlation=float(metric[0,1]/np.sqrt(metric[0,0]*metric[1,1])),
        frozen_reaction_coordinates_preserved=True)
