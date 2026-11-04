"""Finite-support VSCF from sampled n-mode cuts, with explicit domain checks."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping,Sequence
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator,RegularGridInterpolator
from .vscf import DiagonalPairVSCFResult,solve_diagonal_pair_vscf


@lru_cache(maxsize=32)
def _unit_sine_kinetic(points):
    k=np.arange(1,points+1,dtype=float)
    transform=np.sqrt(2/(points+1))*np.sin(np.pi*np.outer(k,k)/(points+1))
    matrix=(transform*(np.pi*k)**2)@transform.T
    matrix=.5*(matrix+matrix.T);matrix.setflags(write=False)
    return matrix


def _grid(values):
    if not np.isrealobj(values):raise ValueError('sample coordinates must be real')
    q=np.asarray(values,float)
    if q.ndim!=1 or len(q)<4 or not np.all(np.isfinite(q)) or np.any(np.diff(q)<=0):
        raise ValueError('at least four increasing finite sample coordinates required')
    return q


@dataclass(frozen=True)
class TabulatedVSCFResult:
    coordinates: tuple[np.ndarray,...]
    domains: tuple[tuple[float,float],...]
    solution: DiagonalPairVSCFResult
    energy: float
    boundary_probabilities: tuple[float,...]
    fixed_point_residual: float
    interpolation: str

    def moments(self,powers:Sequence[Sequence[int]]):
        """Coordinate-multiplication moments in the converged product state."""
        words=np.asarray(powers)
        if (words.ndim!=2 or words.shape[1]!=len(self.coordinates)
            or not np.all(np.isfinite(words)) or np.any(words<0) or np.any(words!=np.floor(words))):
            raise ValueError('moment words must be nonnegative integer powers over all modes')
        values=[]
        for word in words:
            value=1.
            for q,modal,degree in zip(self.coordinates,self.solution.modal_coefficients,word,strict=True):
                value*=float(abs(modal[:,0])**2 @ q**int(degree))
            values.append(value)
        return np.asarray(values)


    def contract_field(self,mode_indices,coordinates,values):
        """Average a sampled coordinate property without leaving its support."""
        modes=tuple(mode_indices);axes=tuple(_grid(q) for q in coordinates)
        if not modes or len(modes)!=len(axes) or len(set(modes))!=len(modes) or any(
                isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<len(self.coordinates) for i in modes):
            raise ValueError('unique physical mode indices and matching property axes required')
        if not np.isrealobj(values):raise ValueError('sampled fields must be real')
        field=np.asarray(values,float)
        if field.shape[:len(axes)]!=tuple(len(q) for q in axes) or not np.all(np.isfinite(field)):
            raise ValueError('sampled property must cover every declared coordinate')
        for mode,q in zip(modes,axes,strict=True):
            if self.domains[mode][0]<q[0] or self.domains[mode][1]>q[-1]:
                raise ValueError('property support cannot be extrapolated')
        if len(modes)==1:
            field=CubicSpline(axes[0],field,axis=0,extrapolate=False)(self.coordinates[modes[0]])
        else:
            mesh=np.stack(np.meshgrid(*(self.coordinates[i] for i in modes),indexing='ij'),axis=-1)
            field=RegularGridInterpolator(axes,field,bounds_error=True)(mesh)
        for mode in modes:
            field=np.tensordot(abs(self.solution.modal_coefficients[mode][:,0])**2,field,axes=(0,0))
        return field


def solve_tabulated_pair_vscf(coordinates,potentials,kinetic_coefficients,*,
        pair_potentials:Mapping|None=None,domains=None,points=63,retained_modals=1,
        interpolation='cubic',boundary_fraction=.1,**vscf_controls):
    """Build local sine DVRs and reuse the existing diagonal-pair VSCF solver.

    Every potential is a complete one-mode cut in one consistent energy unit;
    the supplied kinetic coefficient multiplies -d²/dq² in that unit. Pair
    records supply their own coordinates and pure increment values. Evaluation
    outside measured support is forbidden. Dirichlet walls are numerical
    boundaries and must pass independent domain tests before qualification.
    """
    grids=tuple(_grid(q) for q in coordinates);count=len(grids)
    if count<1 or len(potentials)!=count:raise ValueError('one sampled potential per coordinate required')
    if not np.isrealobj(kinetic_coefficients):raise ValueError('kinetic coefficients must be real')
    if isinstance(retained_modals,bool) or not isinstance(retained_modals,(int,np.integer)) or retained_modals<1:
        raise ValueError('positive integer retained modal count required')
    kinetic=np.asarray(kinetic_coefficients,float)
    if kinetic.shape!=(count,) or not np.all(np.isfinite(kinetic)) or np.any(kinetic<=0):
        raise ValueError('positive finite kinetic coefficients required')
    sizes=np.broadcast_to(np.asarray(points),(count,))
    if np.any(sizes!=np.floor(sizes)) or np.any(sizes<3) or np.any(sizes<retained_modals):
        raise ValueError('integer DVR sizes must accommodate the retained modals')
    if interpolation not in {'cubic','pchip'} or not np.isfinite(boundary_fraction) or not 0<boundary_fraction<.5:
        raise ValueError('invalid interpolation or physical boundary-strip fraction')
    if domains is None:domains=tuple((q[0],q[-1]) for q in grids)
    if len(domains)!=count:raise ValueError('one physical domain per coordinate required')
    local_grids=[];hamiltonians=[];bounded=[]
    for q,raw,coefficient,size,domain in zip(grids,potentials,kinetic,sizes,domains,strict=True):
        if not np.isrealobj(raw):raise ValueError('sampled potentials must be real')
        y=np.asarray(raw,float);domain=tuple(map(float,domain))
        if y.shape!=q.shape or not np.all(np.isfinite(y)):raise ValueError('finite potential values must cover every sample coordinate')
        if len(domain)!=2 or not np.all(np.isfinite(domain)) or domain[0]<q[0] or domain[1]>q[-1] or domain[0]>=domain[1]:
            raise ValueError('domain cannot extrapolate beyond supplied samples')
        x=np.linspace(*domain,int(size)+2)[1:-1]
        interpolator=CubicSpline(q,y,extrapolate=False) if interpolation=='cubic' else PchipInterpolator(q,y,extrapolate=False)
        h=coefficient*_unit_sine_kinetic(int(size))/(domain[1]-domain[0])**2+np.diag(interpolator(x))
        local_grids.append(x);hamiltonians.append(h);bounded.append(domain)
    pairs={}
    for key,record in (pair_potentials or {}).items():
        if len(key)!=2 or not all(isinstance(i,(int,np.integer)) and not isinstance(i,bool) for i in key):
            raise ValueError('pair keys require two integer mode indices')
        i,j=key
        if not 0<=i<j<count:raise ValueError('pair keys must be unique and ordered')
        qi,qj=tuple(_grid(q) for q in record['coordinates'])
        value=np.asarray(record['values'],float)
        if value.shape!=(len(qi),len(qj)) or not np.all(np.isfinite(value)) or not np.isrealobj(record['values']):
            raise ValueError('pair increment must cover its complete finite sample grid')
        for mode,q in ((i,qi),(j,qj)):
            if bounded[mode][0]<q[0] or bounded[mode][1]>q[-1]:
                raise ValueError('pair support cannot cover the requested domain without extrapolation')
        if not np.any(qi==0) or not np.any(qj==0) or max(np.max(abs(value[qi==0])),np.max(abs(value[:,qj==0])))>1e-10:
            raise ValueError('pure pair increments must vanish on both origin cuts')
        mesh=np.stack(np.meshgrid(local_grids[i],local_grids[j],indexing='ij'),axis=-1)
        pairs[key]=RegularGridInterpolator((qi,qj),value,method='linear',bounds_error=True)(mesh)
    solution=solve_diagonal_pair_vscf(hamiltonians,pairs,retained_modals=retained_modals,**vscf_controls)
    densities=[abs(v[:,0])**2 for v in solution.modal_coefficients]
    residual=0.
    for i,(h,vector) in enumerate(zip(hamiltonians,solution.modal_coefficients,strict=True)):
        field=np.zeros(len(h))
        for (a,b),value in pairs.items():
            if a==i:field+=value@densities[b]
            if b==i:field+=value.T@densities[a]
        ground=vector[:,0];applied=h@ground+field*ground
        residual=max(residual,float(np.linalg.norm(applied-ground*float(ground@applied))))
    edge=[]
    for x,rho,(left,right) in zip(local_grids,densities,bounded,strict=True):
        strip=boundary_fraction*(right-left)
        edge.append(float(np.sum(rho[(x<left+strip)|(x>right-strip)])))
    return TabulatedVSCFResult(tuple(local_grids),tuple(bounded),solution,
        float(sum(v[0] for v in solution.modal_energies)-solution.double_counting_energy),
        tuple(edge),residual,interpolation)


def qualify_tabulated_pair_vscf(coordinates,potentials,kinetic_coefficients,*,domains,
        grid_sizes=(31,63,127),energy_tolerance=1e-3,boundary_tolerance=1e-6,
        moment_powers=(),moment_tolerance=1e-5,observable_evaluator=None,observable_tolerance=1e-4,**kwargs):
    """Require grid, nested-domain, boundary and fixed-point checks separately.

    Domains must be supplied in increasing order for every mode. Two successive
    domain comparisons are required; changes cannot cancel one another.
    Uncertainty of the interpolant and electronic model is not certified here.
    """
    count=len(coordinates);domain_list=np.asarray(domains,float);sizes=np.asarray(grid_sizes)
    if (domain_list.ndim!=3 or domain_list.shape[1:]!=(count,2) or len(domain_list)<3
        or not np.all(np.isfinite(domain_list)) or np.any(domain_list[:,:,0]>=domain_list[:,:,1])
        or np.any(np.diff(domain_list[:,:,0],axis=0)>0) or np.any(np.diff(domain_list[:,:,1],axis=0)<0)
        or any(np.array_equal(a,b) for a,b in zip(domain_list[:-1],domain_list[1:],strict=True))):
        raise ValueError('at least three distinct nested physical domains required')
    if sizes.ndim!=1 or len(sizes)<2 or np.any(sizes!=np.floor(sizes)) or np.any(np.diff(sizes)<=0):
        raise ValueError('increasing integer DVR sizes required')
    if not np.all(np.isfinite([energy_tolerance,boundary_tolerance,moment_tolerance])) or min(energy_tolerance,boundary_tolerance,moment_tolerance)<=0:
        raise ValueError('positive finite qualification tolerances required')
    words=tuple(tuple(p) for p in moment_powers);history=[];results=[]
    for domain in domain_list:
        records=[];previous=None
        for size in sizes:
            result=solve_tabulated_pair_vscf(coordinates,potentials,kinetic_coefficients,
                domains=domain,points=int(size),**kwargs)
            moments=result.moments(words) if words else np.empty(0)
            observables=np.asarray(observable_evaluator(result),float).ravel() if observable_evaluator else np.empty(0)
            tolerances=np.broadcast_to(np.asarray(observable_tolerance,float),observables.shape)
            if not np.all(np.isfinite(observables)) or not np.all(np.isfinite(tolerances)) or np.any(tolerances<=0):
                raise ValueError('finite observables and positive observable tolerances required')
            record=dict(points=int(size),energy=result.energy,moments=moments.tolist(),observables=observables.tolist(),
                boundary_probabilities=list(result.boundary_probabilities),fixed_point_residual=result.fixed_point_residual,
                energy_change=None if previous is None else abs(result.energy-previous[0].energy),
                maximum_moment_change=None if previous is None else float(np.max(abs(moments-previous[1]),initial=0.)),
                observable_change=None if previous is None else abs(observables-previous[2]).tolist())
            records.append(record);previous=result,moments,observables
        results.append((result,moments,observables));history.append(dict(domains=domain.tolist(),mesh_checks=records))
    grid_passed=all(h['mesh_checks'][-1]['energy_change']<=energy_tolerance and
        h['mesh_checks'][-1]['maximum_moment_change']<=moment_tolerance and
        np.all(np.asarray(h['mesh_checks'][-1]['observable_change'])<=tolerances) for h in history)
    changes=[dict(energy=abs(b[0].energy-a[0].energy),moments=float(np.max(abs(b[1]-a[1]),initial=0.)),observables=abs(b[2]-a[2]).tolist())
        for a,b in zip(results[:-1],results[1:],strict=True)]
    domain_passed=all(c['energy']<=energy_tolerance and c['moments']<=moment_tolerance and np.all(np.asarray(c['observables'])<=tolerances) for c in changes[-2:])
    independent=[]
    for stage in range(max(0,len(domain_list)-3),len(domain_list)-1):
        baseline,before_moments,before_observables=results[stage]
        energy_sum=0.;moment_sum=np.zeros_like(before_moments);observable_sum=np.zeros_like(before_observables)
        for mode in range(count):
            proposed=domain_list[stage].copy();proposed[mode]=domain_list[stage+1,mode]
            probe=solve_tabulated_pair_vscf(coordinates,potentials,kinetic_coefficients,
                domains=proposed,points=int(sizes[-1]),**kwargs)
            probe_moments=probe.moments(words) if words else np.empty(0)
            probe_observables=np.asarray(observable_evaluator(probe),float).ravel() if observable_evaluator else np.empty(0)
            energy_sum+=abs(probe.energy-baseline.energy)
            moment_sum+=abs(probe_moments-before_moments)
            observable_sum+=abs(probe_observables-before_observables)
        independent.append(dict(stage=stage,energy_L1=energy_sum,moment_L1=moment_sum.tolist(),observable_L1=observable_sum.tolist()))
    independent_passed=all(c['energy_L1']<=energy_tolerance and np.all(np.asarray(c['moment_L1'])<=moment_tolerance)
        and np.all(np.asarray(c['observable_L1'])<=tolerances) for c in independent)
    final=results[-1][0]
    boundary_passed=max(final.boundary_probabilities)<=boundary_tolerance
    residual_passed=all(h['mesh_checks'][-1]['fixed_point_residual']<=energy_tolerance for h in history)
    report=dict(schema='matrix.gf.tabulated_vscf_qualification.v1',
        qualified=bool(grid_passed and domain_passed and boundary_passed and residual_passed and independent_passed),
        grid_passed=bool(grid_passed),domain_passed=bool(domain_passed),boundary_passed=bool(boundary_passed),
        fixed_point_passed=bool(residual_passed),independent_domain_passed=bool(independent_passed),
        independent_domain_checks=independent,history=history,domain_changes=changes,
        energy_tolerance=energy_tolerance,boundary_tolerance=boundary_tolerance,moment_tolerance=moment_tolerance,
        observable_tolerances=tolerances.tolist(),
        scope='finite-support VSCF numerical convergence only; no extrapolation, interpolation-data or electronic-model accuracy claim')
    return final,report


__all__=['TabulatedVSCFResult','solve_tabulated_pair_vscf','qualify_tabulated_pair_vscf']
