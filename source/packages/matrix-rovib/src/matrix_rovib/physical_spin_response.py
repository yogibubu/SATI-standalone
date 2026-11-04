"""SMITH/QC surface response through moving LAM spans into native coupled F."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from .lam_surface_response import FIELDS, native_surface_response, lam_span_response
from .lam_rotation import solve_lam_basis, primitive_momentum_operators
from .lam_response import apply_factors
from .native_spin_effective import build_native_lam_spin_operator, native_spin_n_values
from .spin_effective_observables import native_spin_dipole_operators
from .spin_hyperfine import (
    rank1_body_block, rank2_body_block, dipole_transition_matrix,
    spin_hyperfine_reduced_dipole_matrices,
)


def align_lam_span(reference, vectors, tangent):
    """Polar transport and its Sylvester derivative, invariant to eigenvector gauge.

    All columns spanning the retained manifold are transported together.
    Loss of overlap rank rejects the chart, rather than clipping singular values.
    """
    a=np.asarray(reference,complex);u=np.asarray(vectors,complex);du=np.asarray(tangent,complex)
    if a.shape!=u.shape or du.shape!=u.shape or not all(np.all(np.isfinite(v)) for v in (a,u,du)):
        raise ValueError('matching finite LAM spans and tangent required')
    overlap=a.conj().T@u
    l,sv,rh=np.linalg.svd(overlap)
    if min(sv,default=0.)<=np.sqrt(np.finfo(float).eps):
        raise ValueError('LAM transport chart lost overlap rank; rebuild the outer chart')
    t=rh.conj().T@l.conj().T
    c=overlap@t;dc=a.conj().T@du@t
    e,v=np.linalg.eigh(.5*(c+c.conj().T))
    omega=v@((v.conj().T@(dc.conj().T-dc)@v)/(e[:,None]+e[None,:]))@v.conj().T
    return u@t,du@t+u@t@omega,float(min(sv))


def _field(values,delta,u,du):
    """Exact derivative of U^dagger field U, also for zero fields."""
    f=np.asarray(values).reshape(-1,1);df=np.asarray(delta).reshape(-1,1)
    return du.conj().T@(f*u)+u.conj().T@(df*u+f*du)


class _Contractions:
    def __init__(self,surface,lam,du,direction):
        self.surface=surface;self.lam=lam;self.u=lam.basis.vectors;self.du=du
        self.delta=direction.get('spin_tensor_surfaces_MHz',{})
        self.momenta=None;self.cache={}

    def values(self,constant,key,suffix):
        n=len(self.u);shape=(n,)+suffix
        base=(self.surface.spin_tensor_surfaces_MHz or {}).get(key)
        value=np.broadcast_to(np.asarray(constant),shape) if base is None else np.asarray(base).reshape(shape)
        delta=np.asarray(self.delta.get(key,np.zeros(shape))).reshape(shape)
        if not np.all(np.isfinite(delta)):raise ValueError('nonfinite spin field tangent')
        return value,delta

    def scalar(self,surface,lam,N,constant,key):
        if key not in self.cache:
            value,delta=self.values(constant,key,())
            self.cache[key]=_field(value,delta,self.u,self.du)
        return np.kron(self.cache[key],np.eye(2*N+1))

    def rank2(self,surface,lam,N,Np,constant,key):
        if key not in self.cache:
            value,delta=self.values(constant,key,(6,))
            self.cache[key]=[_field(value[:,k],delta[:,k],self.u,self.du) for k in range(6)]
        out=np.zeros((self.u.shape[1]*(2*N+1),self.u.shape[1]*(2*Np+1)),complex)
        for k in range(6):
            unit=np.eye(6)[k]
            out+=np.kron(self.cache[key][k],
                         rank2_body_block(N,Np,tuple(unit),lam.representation))
        return out

    def momentum(self,surface,lam,N,Np,constant,key):
        value,delta=self.values(constant,key,(surface.dimension,3))
        if self.momenta is None:
            self.momenta=lam.basis.momentum_operators or primitive_momentum_operators(surface)
        out=np.zeros((self.u.shape[1]*(2*N+1),self.u.shape[1]*(2*Np+1)),complex)
        for mu,p in enumerate(self.momenta):
            for axis in range(3):
                f=value[:,mu,axis];df=delta[:,mu,axis]
                def act(coefficient,v):
                    return .5*(coefficient[:,None]*(p@v)+p@(coefficient[:,None]*v))
                d=self.du.conj().T@act(f,self.u)+self.u.conj().T@(act(df,self.u)+act(f,self.du))
                out+=np.kron(d,rank1_body_block(N,Np,tuple(np.eye(3)[axis]),lam.representation))
        return out


def physical_spin_matrices(surface,basis,directions,model,*,two_f_values,reference_span=None,
                           representation='Ir',response_states=48,memory_budget_bytes=512*1024**2,
                           operator_manifest=None,operator_parameters_MHz=None,operator_directions=None):
    """Assemble H,dH,mu,dmu automatically from physical surface directions.

    Grid, primitive basis, masses and body-frame convention stay fixed.
    Geometry/PES/metric directions may come from PhysicalLAMProducer/SMITH.
    Native spin fields and dipoles obey the full contraction product rule.
    Rotational product coefficients are fixed; no spin-free eigenvector
    derivatives, molecular corrections or displaced spectra are needed.
    """
    fs=tuple(sorted(set(two_f_values)));directions=tuple(directions)
    if not fs or not directions:raise ValueError('F blocks and physical parameter directions required')
    manifest_parameters=None
    if operator_manifest is not None:
        from .general_lam_operator import normalize_general_lam_operator_manifest
        operator_manifest=normalize_general_lam_operator_manifest(operator_manifest)
        manifest_parameters=operator_manifest.parameters_with(operator_parameters_MHz)
    elif operator_parameters_MHz or (operator_directions is not None and any(operator_directions)):
        raise ValueError('operator parameter directions require a manifest')
    operator_directions=tuple({} for _ in directions) if operator_directions is None else tuple(operator_directions)
    if len(operator_directions)!=len(directions):raise ValueError('one manifest tangent per physical direction required')
    if any(set(d)-(set(manifest_parameters) if manifest_parameters is not None else set()) for d in operator_directions):
        raise ValueError('unknown manifest parameter tangent')
    from .body_frame import frame_rotations
    if frame_rotations(surface) is not None:
        raise ValueError("physical matrix response requires coefficients in one fixed body frame")
    ns=sorted(set(n for f in fs for n in native_spin_n_values(model,f)))
    k=basis.vectors.shape[1]
    # Conservative guard covers coupled spin blocks plus all derivative arrays.
    from .spin_hyperfine import _lam_spin_basis_registry
    dummy=SimpleNamespace(basis=basis,representation=representation,blocks={
        n:SimpleNamespace(energies_MHz=np.zeros(k*(2*n+1))) for n in ns})
    _,registry,_=_lam_spin_basis_registry(dummy,model)
    if 16*(len(directions)+1)*sum(len(registry[f])**2 for f in fs)*16>memory_budget_bytes:
        raise MemoryError('physical spin matrix response exceeds workspace budget')
    for n in ns:dummy.blocks[n].vectors=np.eye(k*(2*n+1))
    reference=basis.vectors if reference_span is None else np.asarray(reference_span)
    transported=[];minimum=1.;workspace={}
    for direction in directions:
        if set(direction.get('spin_tensor_surfaces_MHz',{}))-set(surface.spin_tensor_surfaces_MHz or {}):
            raise ValueError('spin field derivatives require an explicit base field; override semantics must stay fixed')
        scalar={key:value for key,value in direction.items() if key in FIELDS}
        du=lam_span_response(surface,dummy,scalar,response_states=response_states,workspace=workspace)
        u,du,overlap=align_lam_span(reference,basis.vectors,du)
        transported.append((du,scalar,direction));minimum=min(minimum,overlap)
    aligned=replace(basis,vectors=u,spectral_coefficients=None)
    lam=SimpleNamespace(basis=aligned,representation=representation,blocks=dummy.blocks)
    zero=np.zeros_like(u)
    # At fixed rho, all other native coefficients enter H linearly. The
    # homogeneous tangent therefore builds the primal factors exactly.
    scale={name:getattr(surface,name) for name in FIELDS if name!='rho'}
    aligned_workspace={}
    factors,_=native_surface_response(surface,lam,scale,basis_tangent=zero,workspace=aligned_workspace,
        manifest=operator_manifest,parameters=manifest_parameters,operator_direction=manifest_parameters)
    spin_free={n:apply_factors(f,np.eye(k*(2*n+1)),k,2*n+1) for n,f in factors.items()}
    blocks={f:build_native_lam_spin_operator(surface,lam,model,two_F=f,
        _spin_free_blocks=spin_free,memory_budget_bytes=memory_budget_bytes) for f in fs}
    h={f:b.hamiltonian_MHz for f,b in blocks.items()}
    mu=native_spin_dipole_operators(surface,lam,blocks.values(),memory_budget_bytes=memory_budget_bytes)
    dh={f:[] for f in fs};dmu={pair:[] for pair in mu}
    if surface.dipole_au is None:raise ValueError('physical dipole surface required')
    dipole=np.asarray(surface.dipole_au).reshape(-1,3)
    for (du,scalar,direction),operator_direction in zip(transported,operator_directions,strict=True):
        factors,_=native_surface_response(surface,lam,scalar,basis_tangent=du,workspace=aligned_workspace,
            manifest=operator_manifest,parameters=manifest_parameters,operator_direction=operator_direction)
        derivative_free={n:apply_factors(f,np.eye(k*(2*n+1)),k,2*n+1) for n,f in factors.items()}
        contracts=_Contractions(surface,lam,du,direction)
        hooks={key:getattr(contracts,key) for key in ('scalar','rank2','momentum')}
        for f in fs:
            dh[f].append(build_native_lam_spin_operator(surface,lam,model,two_F=f,
                _spin_free_blocks=derivative_free,_contractions=hooks,
                memory_budget_bytes=memory_budget_bytes).hamiltonian_MHz)
        dd=np.asarray(direction.get('dipole_au',np.zeros_like(surface.dipole_au))).reshape(-1,3)
        dm=[_field(dipole[:,axis],dd[:,axis],u,du) for axis in range(3)]
        transition={(n,m):sum((np.kron(dm[a],dipole_transition_matrix(n,m,name,representation))
            for a,name in enumerate(('a','b','c'))),start=np.zeros((k*(2*n+1),k*(2*m+1)),complex))
            for n in ns for m in ns if abs(n-m)<=1 and n+m>0}
        view=SimpleNamespace(model=model,blocks=tuple(blocks.values()),n_values=ns,representation=representation)
        derivative_mu=spin_hyperfine_reduced_dipole_matrices(view,(0.,0.,0.),
            _transition_blocks=transition,_raw_operators=True)
        for pair in mu:dmu[pair].append(derivative_mu[pair])
    report=dict(schema='matrix.lock.physical_spin_response.v1',minimum_lam_overlap=minimum,
        lam_states=k,response_states=response_states,parameter_count=len(directions),
        scope='analytic fixed-grid/body-frame response; discarded LAM response requires independent convergence')
    return (h,{f:np.stack(v) for f,v in dh.items()},mu,{pair:np.stack(v) for pair,v in dmu.items()}),blocks,report


class PhysicalSpinLAMProducer:
    """Cache and transport an existing data-configured physical surface producer.

    The wrapped producer is evaluated once per parameter point. It supplies
    PhysicalSurfaceEvaluation, including analytic SMITH/QC directions.
    """
    def __init__(self,producer,*,system,parameter_names,model,two_f_values,lam_states,
                 primitive_basis='auto',fourier_cutoffs=None,representation='Ir',
                 response_states=48,memory_budget_bytes=512*1024**2):
        self.producer=producer;self.system=system;self.names=tuple(parameter_names);self.model=model
        self.fs=tuple(two_f_values);self.count=lam_states;self.primitive=primitive_basis
        self.cutoffs=fourier_cutoffs;self.representation=representation;self.response_states=response_states
        self.memory_budget=memory_budget_bytes;self.reference=None;self.cache={};self.blocks={};self.report={}

    def initial_models(self,values,*,active_lam_states=(0,),maximum_order=4,include_gap_MHz=0.):
        """Build native reductions for whole anchor LAM parents automatically."""
        from .native_spin_effective import reduce_native_spin_operator
        states=tuple(active_lam_states)
        if not states or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<self.count for i in states):
            raise ValueError('valid nonempty active LAM parent indices required')
        self(values)
        return {f:reduce_native_spin_operator(block,[i for i,b in enumerate(block.basis)
                    if b.k_index//(2*b.N+1) in states],maximum_order=maximum_order,
                    include_gap_MHz=include_gap_MHz,memory_budget_bytes=self.memory_budget)
                for f,block in self.blocks.items()}

    def __call__(self,values):
        x=np.asarray(values,float)
        if x.shape!=(len(self.names),) or not np.all(np.isfinite(x)):raise ValueError('finite physical parameter vector required')
        key=x.tobytes()
        if key not in self.cache:
            evaluation=self.producer.evaluate(dict(zip(self.names,x,strict=True)),derivatives=True)
            surface=evaluation.surfaces[self.system]
            if self.reference is not None:
                if surface.periodic!=self.signature[0] or any(not np.array_equal(a,b) for a,b in zip(surface.coordinates,self.signature[1],strict=True)):
                    raise ValueError('physical response requires an unchanged coordinate grid')
            directions=evaluation.directions[self.system]
            if isinstance(directions,dict):directions=[directions[name] for name in self.names]
            basis=solve_lam_basis(surface,state_count=self.count,primitive_basis=self.primitive,fourier_cutoffs=self.cutoffs)
            if self.reference is None:
                self.reference=np.array(basis.vectors,copy=True)
                self.signature=(surface.periodic,tuple(np.array(q,copy=True) for q in surface.coordinates))
            data,self.blocks,self.report=physical_spin_matrices(surface,basis,directions,self.model,
                two_f_values=self.fs,reference_span=self.reference,representation=self.representation,
                response_states=self.response_states,memory_budget_bytes=self.memory_budget)
            self.cache={key:data}
        return self.cache[key]
