"""Canonical Hermitian downfolding with noncommuting rotational words.

The Bloch wave operator is solved order by order and orthonormalized with
(I+chi^dagger chi)^(-1/2). Degree means the number of angular-momentum
factors, not the number of perturbation vertices. Sources are commuting
bookkeeping labels; angular words are never sorted or commuted.
Reference: Bravyi, DiVincenzo, Loss, Ann. Phys. 326 (2011), arXiv:1105.0675.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .lam_rotation import _rotational_operators


@dataclass(frozen=True)
class OrderedEffectiveHamiltonian:
    coefficients: dict
    active_states: tuple
    external_states: tuple
    maximum_order: int
    minimum_external_gap_MHz: float | None
    wave_operator: dict
    invariance_residual_by_order: dict
    external_labels: tuple = ()
    second_order_external_contributions: tuple = ()

    def hamiltonian(self, J, *, scale=1.0, order=None, representation="Ir", sources=None):
        if (isinstance(J,bool) or int(J)!=J or J<0 or not np.isfinite(scale)
            or (order is not None and (isinstance(order,bool) or int(order)!=order))):
            raise ValueError("integer J/order and finite rotational scale required")
        J=int(J)
        degree=self.maximum_order if order is None else int(order)
        if not 0 <= degree <= self.maximum_order:
            raise ValueError("requested order outside the computed expansion")
        angular=_rotational_operators(J,representation)
        identity=np.eye(2*J+1,dtype=complex)
        n=len(self.active_states)
        out=np.zeros((n*(2*J+1),)*2,complex)
        words={():identity}
        for (word,labels),matrix in sorted(self.coefficients.items()):
            if len(word)>degree or (sources is not None and tuple(labels)!=tuple(sorted(sources))):
                continue
            if word not in words:
                op=identity
                for axis in word:
                    op=op@angular[axis]
                words[word]=op
            out+=scale**len(word)*np.kron(matrix,words[word])
        return out

    def to_dict(self):
        def encode(poly):
            return [dict(rotation_word=list(w),sources=list(s),real=m.real.tolist(),imag=m.imag.tolist())
                    for (w,s),m in sorted(poly.items())]
        return dict(schema="matrix.lock.ordered_effective_hamiltonian.v1",
            active_states=list(self.active_states),external_states=list(self.external_states),
            external_labels=list(self.external_labels),maximum_rotation_order=self.maximum_order,
            second_order_external_contributions=list(self.second_order_external_contributions),
            minimum_external_gap_MHz=self.minimum_external_gap_MHz,
            coefficients=encode(self.coefficients),wave_operator=encode(self.wave_operator),
            invariance_residual_by_order=self.invariance_residual_by_order,
            convention="canonical Hermitian Bloch-wave-operator orthonormalization; ordered rotation words",
            scope="formal low-rotation expansion; external-space and order convergence must be qualified")


def downfold_ordered_operators(h0, terms, active_states, *, maximum_order=4,
                               include_gap_MHz=0.0, external_labels=None,
                               maximum_terms=100000, memory_budget_bytes=512*1024**2):
    """Eliminate Q while retaining a complete quasi-degenerate active P space.

    terms are {rotation_word:[0,1,...], matrix_MHz:complex matrix, source:str}.
    The complete polynomial must be Hermitian word by reversed word.
    h0 contains all rotation-independent terms. Source labels carry the
    operator-family products at every order; they do not assert independent
    physical additivity of interacting higher-order pathways.
    """
    h0=np.asarray(h0,complex)
    if (h0.ndim!=2 or h0.shape[0]!=h0.shape[1] or not len(h0)
        or not np.all(np.isfinite(h0)) or not np.allclose(h0,h0.conj().T,rtol=1.e-12,atol=1.e-12)
        or isinstance(maximum_order,bool) or int(maximum_order)!=maximum_order or maximum_order<1
        or maximum_terms<1 or memory_budget_bytes<1 or not np.isfinite(include_gap_MHz) or include_gap_MHz<0):
        raise ValueError("invalid Hermitian ordered reduction")
    maximum_order=int(maximum_order);n=len(h0)
    energy,u=np.linalg.eigh(h0)
    raw=list(active_states)
    if not raw or any(isinstance(i,bool) or int(i)!=i or not 0<=i<n for i in raw) or len(set(raw))!=len(raw):
        raise ValueError("unique active eigenstate indices required")
    active=set(map(int,raw));gap_tol=max(float(include_gap_MHz),128*np.finfo(float).eps*max(1.,max(abs(energy))))
    while True:
        expanded={j for i in active for j in range(n) if abs(energy[i]-energy[j])<=gap_tol}
        if expanded==active:break
        active=expanded
    p=tuple(sorted(active));q=tuple(i for i in range(n) if i not in active)
    origin=float(energy[p[0]])
    centered=energy-origin
    ep,eq=centered[list(p)],centered[list(q)]
    gap=ep[None,:]-eq[:,None]
    dimension=max(n,len(p),len(q),1)
    # Bound persistent coefficients and transient products conservatively.
    capacity=min(int(maximum_terms),int(memory_budget_bytes)//max(16*dimension**2*16,1))
    if capacity<1:raise MemoryError("downfolding coefficient workspace exceeds memory budget")

    def add(*polys):
        result={}
        for poly in polys:
            for key,value in poly.items():
                if len(key[0])<=maximum_order:
                    result[key]=result[key]+value if key in result else value.copy()
            if len(result)>capacity:raise MemoryError("ordered expansion exceeds its term/memory budget")
        return {k:v for k,v in result.items() if np.any(v)}
    def mul(a,b):
        result={}
        for (wa,sa),ma in a.items():
            for (wb,sb),mb in b.items():
                w=wa+wb
                if len(w)>maximum_order:continue
                key=(w,tuple(sorted(sa+sb)));value=ma@mb
                if not np.any(value):continue
                result[key]=result[key]+value if key in result else value
                if len(result)>capacity:raise MemoryError("ordered expansion exceeds its term/memory budget")
        return result
    def dagger(a):return {(tuple(reversed(w)),s):m.conj().T for (w,s),m in a.items()}
    def scale(a,c):return {k:c*v for k,v in a.items()}
    def block(a,rows,cols):return {k:v[np.ix_(rows,cols)] for k,v in a.items()}

    perturb={}
    for record in terms:
        w=tuple(record["rotation_word"]);source=str(record["source"])
        m=np.asarray(record["matrix_MHz"],complex)
        if (not w or any(isinstance(a,bool) or a not in (0,1,2) for a in w)
            or not source or m.shape!=h0.shape or not np.all(np.isfinite(m))):
            raise ValueError("invalid ordered perturbation word, source or matrix")
        if len(w)<=maximum_order:
            perturb=add(perturb,{(w,(source,)):u.conj().T@m@u})
    adjoint=dagger(perturb)
    for key in perturb.keys()|adjoint.keys():
        left=perturb.get(key,np.zeros_like(h0));right=adjoint.get(key,np.zeros_like(h0))
        # Unitary projection leaves roundoff in nominally zero entries. An
        # elementwise relative check would reject a Hermitian MHz-scale
        # operator merely because one zero entry is O(eps*||A||).
        bound=128*np.finfo(float).eps*n*max(1.,np.linalg.norm(left),np.linalg.norm(right))
        if np.linalg.norm(left-right)>bound:
            raise ValueError("ordered perturbation is not Hermitian word by reversed word")
    perturb=scale(add(perturb,adjoint),.5)
    pp,pq,qp,qq=(block(perturb,a,b) for a,b in ((p,p),(p,q),(q,p),(q,q)))
    chi={}
    for degree in range(1,maximum_order+1):
        rhs=add(qp,mul(qq,chi),scale(mul(chi,pp),-1),scale(mul(mul(chi,pq),chi),-1))
        chi=add(chi,{k:v/gap for k,v in rhs.items() if len(k[0])==degree})
    identity={((),()):np.eye(len(p),dtype=complex)}
    metric_correction=mul(dagger(chi),chi)
    inverse_root=identity.copy();power=identity.copy();coefficient=1.
    for k in range(1,maximum_order//2+1):
        power=mul(power,metric_correction)
        coefficient*=(-.5-(k-1))/k
        inverse_root=add(inverse_root,scale(power,coefficient))
    hpp=add({((),()):np.diag(ep)},pp)
    hqq=add({((),()):np.diag(eq)},qq)
    numerator=add(hpp,mul(pq,chi),mul(dagger(chi),qp),mul(mul(dagger(chi),hqq),chi))
    effective=mul(mul(inverse_root,numerator),inverse_root)
    # Roundoff symmetrization only; every ordered adjoint is retained.
    effective=scale(add(effective,dagger(effective)),.5)
    effective=add(effective,{((),()):origin*np.eye(len(p))})
    if any(not np.all(np.isfinite(v)) for v in effective.values()):
        raise FloatingPointError("nonfinite coefficient in ordered reduction")
    residual=add(qp,mul(hqq,chi),scale(mul(chi,hpp),-1),scale(mul(mul(chi,pq),chi),-1))
    residuals={str(k):float(max((np.linalg.norm(v) for (w,_),v in residual.items() if len(w)==k),default=0.))
               for k in range(1,maximum_order+1)}
    if external_labels is not None and len(external_labels)!=n:
        raise ValueError("external labels must span the full H0 eigenstate registry")
    labels=tuple(str(i) for i in q) if external_labels is None else tuple(str(external_labels[i]) for i in q)
    resolved=[]
    if maximum_order>=2:
        for (wa,sa),a in pq.items():
            if len(wa)!=1:continue
            for (wb,sb),b in qp.items():
                if len(wb)!=1:continue
                for r,state in enumerate(q):
                    matrix=.5*a[:,r,None]*b[r,None,:]*(1/gap[r,:,None]+1/gap[r,None,:])
                    if np.any(matrix):
                        resolved.append(dict(external_state=state,external_label=labels[r],
                            rotation_word=list(wa+wb),sources=list(sorted(sa+sb)),
                            real=matrix.real.tolist(),imag=matrix.imag.tolist()))
                    if len(resolved)>capacity:
                        raise MemoryError("external-path report exceeds ordered reduction budget")
    return OrderedEffectiveHamiltonian(effective,p,q,maximum_order,
        None if not len(q) else float(np.min(abs(gap))),chi,residuals,labels,tuple(resolved))


class _RotationPolynomial:
    """Small exact algebra adapter consumed by the existing native assembler."""
    __array_priority__=10000
    def __init__(self,terms,maximum_order=8):
        self.maximum_order=maximum_order
        self.terms={k:complex(v) for k,v in terms.items() if v!=0 and len(k)<=maximum_order}
    def __add__(self,other):
        if np.isscalar(other):other=_RotationPolynomial({():other},self.maximum_order)
        result=self.terms.copy()
        for k,v in other.terms.items():result[k]=result.get(k,0)+v
        return _RotationPolynomial(result,min(self.maximum_order,other.maximum_order))
    __radd__=__add__
    def __neg__(self):return self*(-1)
    def __sub__(self,other):return self+(-other)
    def __mul__(self,scalar):return _RotationPolynomial({k:v*scalar for k,v in self.terms.items()},self.maximum_order)
    __rmul__=__mul__
    def __truediv__(self,scalar):return self*(1/scalar)
    def __matmul__(self,other):
        out={}
        for a,ca in self.terms.items():
            for b,cb in other.terms.items():
                if len(a+b)<=min(self.maximum_order,other.maximum_order):
                    out[a+b]=out.get(a+b,0)+ca*cb
        return _RotationPolynomial(out,min(self.maximum_order,other.maximum_order))
    def conj(self):return _RotationPolynomial({k:v.conjugate() for k,v in self.terms.items()},self.maximum_order)
    @property
    def T(self):return _RotationPolynomial({tuple(reversed(k)):v for k,v in self.terms.items()},self.maximum_order)
    def power(self,n):
        result=_RotationPolynomial({():1},self.maximum_order)
        for _ in range(n):result=result@self
        return result


def lam_ordered_effective_hamiltonian(surface, request, values=None, *, active_states=(0,),
                                      maximum_order=4, include_gap_MHz=0.0,
                                      maximum_terms=100000,memory_budget_bytes=512*1024**2,
                                      external_labels=None, spin_operator_builder=None,
                                      spin_model=None, two_F=None):
    """Downfold the production native Fourier/DVR operator expansion.

    Native Watson quartic/sextic/octic and pi² operators use the same algebra
    as numerical assembly. Declarative momentum/rotation words retain their
    source labels. No fit to sampled J energies or Watson constants is used.
    ``spin_operator_builder(surface, basis)`` may return a production
    (CompiledOperatorModel, rotation_words) pair on LAM plus explicit spin
    spaces. Its first space must be named LAM and use this exact basis. It
    supplies spin corrections only; native spin-free terms are lifted onto
    the spin identities here. Active indices then refer to the complete H0.
    Alternatively spin_model and two_F select the automatic native Wigner
    route: all angularly allowed N are built, active_states denotes LAM
    parent indices, and the returned expansion is in intermanifold coupling
    at fixed F (not powers of J).
    """
    from .lam_fit import _lam_variable,_apply_lam_variables,_operator_parameter_values,_projected_coupled_factors
    from .lam_rotation import solve_lam_basis
    from .general_lam_operator import normalize_general_lam_operator_manifest,contract_general_lam_operators
    if spin_model is not None and spin_operator_builder is not None:
        raise ValueError('choose native spin_model or an explicit tensor builder')
    if surface.spin_tensor_surfaces_MHz and spin_operator_builder is None and spin_model is None:
        raise ValueError("ordered LAM downfolding requires explicit spin tensor spaces")
    manifest=None if request.get("operator_manifest") is None else normalize_general_lam_operator_manifest(request["operator_manifest"])
    variables=tuple(_lam_variable(v,surface,k,{} if manifest is None else manifest.parameters_MHz)
                    for k,v in enumerate(request.get("variables",[])))
    values=np.asarray([v["initial"] for v in variables] if values is None else values,float)
    fitted=_apply_lam_variables(surface,variables,values)
    parameters=_operator_parameter_values(manifest,variables,values)
    opts=request.get("options",{})
    basis=solve_lam_basis(fitted,state_count=int(request.get("lam_states",12)),
        primitive_basis=opts.get("primitive_basis","auto"),fourier_cutoffs=opts.get("fourier_cutoffs"))
    if spin_model is not None:
        from .native_spin_effective import native_spin_n_values, native_spin_effective_hamiltonian
        from .lam_rotation import solve_lam_rotation
        if external_labels is not None:
            raise ValueError('native spin reduction uses its coupled-F basis registry')
        ns = native_spin_n_values(spin_model, two_F)
        nlam = len(basis.energies_MHz)
        if sum(16*(nlam*(2*N+1))**2*4 for N in ns) > memory_budget_bytes:
            raise MemoryError('automatic native N closure exceeds the LAM workspace budget')
        lam = solve_lam_rotation(fitted, j_values=ns, lam_state_count=nlam, lam_basis=basis,
            operator_manifest=manifest, operator_parameters_MHz=parameters,
            representation='Ir', primitive_basis=opts.get('primitive_basis','auto'),
            fourier_cutoffs=opts.get('fourier_cutoffs'))
        parents = set(active_states)
        if not parents or any(isinstance(i,bool) or int(i)!=i or not 0<=i<nlam for i in parents):
            raise ValueError('active native LAM parent indices required')
        active_levels = {N:[i for i,l in enumerate(block.levels) if l.torsional_state in parents]
                         for N,block in lam.blocks.items()}
        return native_spin_effective_hamiltonian(fitted, lam, spin_model, two_F=two_F,
            active_levels=active_levels, maximum_order=maximum_order,
            include_gap_MHz=include_gap_MHz, maximum_terms=maximum_terms,
            memory_budget_bytes=memory_budget_bytes)
    symbols=tuple(_RotationPolynomial({(a,):1},maximum_order) for a in range(3))
    identity=_RotationPolynomial({():1},maximum_order)
    factors=_projected_coupled_factors(fitted,basis.vectors,1,basis,
        representation="Ir",rotation_algebra=(symbols,identity),maximum_rotation_order=maximum_order)
    labels=["H0",*[f"linear:{a}" for a in range(3)],
            *[f"quadratic:{a}:{b}" for a in range(3) for b in range(3)]]
    if maximum_order>=4 and (fitted.quartic_distortion_MHz is not None or fitted.sextic_distortion_MHz is not None):
        labels.extend(f"quartic:{i}" for i in range(5))
        labels.extend(f"sextic:{i}" for i in range(7))
    if fitted.p2_rotational_metric_MHz is not None:
        labels.extend(f"p2:{mu}:{a}:{b}" for mu in range(fitted.dimension) for a in range(3) for b in range(3))
    if maximum_order>=3 and fitted.p_rotational_cubic_MHz is not None:
        labels.extend(f"pJ3:{mu}:{a}" for mu in range(fitted.dimension) for a in range(3))
    if maximum_order>=3 and fitted.p_rotational_anticommutator_MHz is not None:
        labels.extend(f"pJanticommutator:{mu}:{a}:{b}" for mu in range(fitted.dimension) for a in range(3) for b in range(3))
    if fitted.pi2_rotation_MHz is not None and np.any(fitted.pi2_rotation_MHz):
        count=13 if maximum_order>=4 else (4 if maximum_order>=3 else 1)
        labels.extend(f"pi2:{i}:{j}" for i in range(3) for j in range(count))
    if len(labels)!=len(factors):
        raise RuntimeError("native factor provenance differs from the production assembler")
    h0=np.zeros((len(basis.energies_MHz),)*2,complex);terms=[]
    for k,(matrix,polynomial) in enumerate(factors):
        for word,value in polynomial.terms.items():
            coefficient=value*matrix
            if not word:h0+=coefficient
            else:terms.append(dict(rotation_word=word,matrix_MHz=coefficient,source=f"native:{labels[k]}"))
    if manifest is not None:
        contracted=contract_general_lam_operators(fitted,basis,manifest)
        for term,matrix in zip(manifest.terms,contracted.lam_factors,strict=True):
            amplitude=.5*term.prefactor*parameters[term.parameter]
            for word,coefficient in ((term.rotation_word,amplitude*matrix),
                                     (tuple(reversed(term.rotation_word)),amplitude*matrix.conj().T)):
                if not word:h0+=coefficient
                else:terms.append(dict(rotation_word=word,matrix_MHz=coefficient,source=f"manifest:{term.label}"))
    # Numerical native factors are Hermitian in the physical angular algebra.
    # Reverse every native word explicitly: J² commutes with angular momenta
    # but the free-word algebra does not assume those Lie-algebra identities.
    native=[]
    for t in terms:
        if t["source"].startswith("native:"):
            native.extend([dict(t,matrix_MHz=.5*t["matrix_MHz"]),
                           dict(t,rotation_word=tuple(reversed(t["rotation_word"])),matrix_MHz=.5*t["matrix_MHz"].conj().T)])
        else:native.append(t)
    if spin_operator_builder is not None:
        from .operator_engine import CompiledOperatorModel, FactorizedOperatorTerm
        from .tensor_downfold import downfold_tensor_operator_model
        model, words = spin_operator_builder(fitted, basis)
        if (not isinstance(model, CompiledOperatorModel) or model.spaces[0].name != 'LAM'
                or model.spaces[0].dimension != len(h0)):
            raise ValueError('spin builder must use the supplied production LAM basis as its first space')
        additions = [FactorizedOperatorTerm('native:reference', (('LAM', h0),), adjoint_symmetrized=False)]
        words = dict(words)
        for k, term in enumerate(native):
            label = f'{term["source"]}:word:{k}'
            words[label] = term['rotation_word']
            additions.append(FactorizedOperatorTerm(label, (('LAM', term['matrix_MHz']),),
                                                    adjoint_symmetrized=False))
        if set(t.label for t in additions) & set(t.label for t in model.terms):
            raise ValueError('spin source labels collide with native operator provenance')
        combined = CompiledOperatorModel(model.spaces, model.terms+tuple(additions), model.parameter_defaults)
        return downfold_tensor_operator_model(combined, words, active_states,
            maximum_order=maximum_order, include_gap_MHz=include_gap_MHz,
            maximum_terms=maximum_terms, memory_budget_bytes=memory_budget_bytes,
            external_labels=external_labels)
    return downfold_ordered_operators(h0,native,active_states,maximum_order=maximum_order,
        include_gap_MHz=include_gap_MHz,maximum_terms=maximum_terms,memory_budget_bytes=memory_budget_bytes,
        external_labels=external_labels)
