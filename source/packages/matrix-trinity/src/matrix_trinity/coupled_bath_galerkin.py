"""Variational active-space times total-quanta harmonic-bath contraction.

Coefficient fields are supplied in a fixed flat measure. A supplied volume
connection makes the weak kinetic form the corresponding Laplace--Beltrami
operator. No fitted energy or rotational offset is inserted by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import eigsh, gmres, lobpcg, LinearOperator
from .kinetic_bath_response import _normal


def _occupations(count, total):
    states = [tuple(0 for _ in range(count))]
    for n in range(1,total+1):
        def fill(prefix, remaining, axes):
            if axes == 1:
                states.append(tuple(prefix+[remaining])); return
            for k in range(remaining+1): fill(prefix+[k],remaining-k,axes-1)
        fill([],n,count)
    return tuple(states)


def harmonic_bath_matrices(kinetic_cm, force_cm, maximum_total_quanta, *, occupations=None):
    """Physical coordinate/momentum moments with exact oscillator closure.

    Products use an extended one-quantum space before restriction. Thus the
    retained Fock boundary does not discard the n+1 intermediate in p*p or
    q*q. The default cutoff is on total occupation. An explicit downward-closed
    occupation set may instead specify a weighted space; every mode and all
    exterior one-quantum intermediates remain available to moment products.
    """
    if (isinstance(maximum_total_quanta,bool)
            or not isinstance(maximum_total_quanta,(int,np.integer))
            or maximum_total_quanta < 0):
        raise ValueError('nonnegative total oscillator cutoff required')
    w,q,p = _normal(kinetic_cm,force_cm)
    nb=len(w)
    if occupations is None:
        states=_occupations(nb,maximum_total_quanta)
        extended=_occupations(nb,maximum_total_quanta+1)
    else:
        from .harmonic_occupations import oscillator_space_with_closure
        states,extended=oscillator_space_with_closure(occupations,nb)
        if max(map(sum,states))>maximum_total_quanta:
            raise ValueError('Explicit occupations exceed the declared total upper bound')
    lookup={s:i for i,s in enumerate(extended)}
    ns=len(states);rq=[[] for _ in range(nb)];cq=[[] for _ in range(nb)];vq=[[] for _ in range(nb)];vp=[[] for _ in range(nb)]
    for col,state in enumerate(states):
        for mode,n in enumerate(state):
            raised=list(state);raised[mode]+=1;row=lookup[tuple(raised)]
            rq[mode].append(row);cq[mode].append(col);vq[mode].append(np.sqrt((n+1)/2));vp[mode].append(1j*np.sqrt((n+1)/2))
            if n:
                lowered=list(state);lowered[mode]-=1;row=lookup[tuple(lowered)]
                rq[mode].append(row);cq[mode].append(col);vq[mode].append(np.sqrt(n/2));vp[mode].append(-1j*np.sqrt(n/2))
    normal_q=[csr_matrix((vq[i],(rq[i],cq[i])),shape=(len(extended),ns)) for i in range(nb)]
    normal_p=[csr_matrix((vp[i],(rq[i],cq[i])),shape=(len(extended),ns)) for i in range(nb)]
    Q=[sum((q[b,i]*normal_q[i] for i in range(nb)),start=csr_matrix((len(extended),ns))) for b in range(nb)]
    P=[sum((p[b,i]*normal_p[i] for i in range(nb)),start=csr_matrix((len(extended),ns))) for b in range(nb)]
    return dict(occupations=states,frequencies_cm=w,
                coordinate=[v[:ns] for v in Q],momentum=[v[:ns] for v in P],
                coordinate_second=_SecondMoments(Q),momentum_second=_SecondMoments(P),
                normal_coordinate=[v[:ns] for v in normal_q],normal_momentum=[v[:ns] for v in normal_p],
                normal_coordinate_second=_SecondMoments(normal_q),normal_momentum_second=_SecondMoments(normal_p),
                coordinate_map=q,momentum_map=p,
                normal_momentum_coordinate=_CrossMoments(normal_p,normal_q))


class _SecondMoments:
    def __init__(self,operators):self.operators=operators;self.cache={}
    def __getitem__(self,pair):
        if pair not in self.cache:
            b,c=pair;self.cache[pair]=(self.operators[b].conj().T@self.operators[c]).tocsr()
        return self.cache[pair]


class _CrossMoments:
    def __init__(self,left,right):self.left=left;self.right=right;self.cache={}
    def __getitem__(self,pair):
        if pair not in self.cache:
            b,c=pair;self.cache[pair]=(self.left[b].conj().T@self.right[c]).tocsr()
        return self.cache[pair]


def rotational_metric_linear_completion(kinetic_vibrational_cm, vertex_bath_gradient_cm):
    """Second bath derivative of D(b)^T G_vv^-1 D(b) for linear D(b)."""
    g=np.asarray(kinetic_vibrational_cm,float);d=np.asarray(vertex_bath_gradient_cm,float)
    if (g.ndim!=3 or g.shape[1]!=g.shape[2] or d.ndim!=4
            or d.shape[:2]!=g.shape[:2] or d.shape[-1]!=3
            or not np.all(np.isfinite(g)) or not np.all(np.isfinite(d))
            or not np.allclose(g,g.swapaxes(-1,-2))
            or np.min(np.linalg.eigvalsh(g))<=0):
        raise ValueError('compatible vibrational metric and full rotational vertex jet required')
    value=np.einsum('guba,guv,gvcd->gbcad',d,np.linalg.inv(g),d,optimize=True)
    return value+value.swapaxes(1,2)


class _TensorSumOperator(LinearOperator):
    def __init__(self,terms,active,bath):
        converted=[]
        for a,b in terms:
            pa,ar=_constant_phase_real(a);pb,br=_constant_phase_real(b.data)
            if pa is not None and pb is not None and np.isreal(pa*pb):
                bb=b.copy();bb.data=br;converted.append((float(np.real(pa*pb))*ar,bb))
            else:converted.append((a,b))
        # Many contributions share exactly the same oscillator operator
        # (notably every active kinetic term shares its identity). Sum their
        # angular factors once instead of repeating dense products per matvec.
        grouped={}
        for a,b in converted:
            key=(b.shape,b.dtype.str,b.indptr.tobytes(),b.indices.tobytes(),b.data.tobytes())
            if key in grouped:
                if np.can_cast(a.dtype,grouped[key][0].dtype,casting='same_kind'):grouped[key][0]+=a
                else:grouped[key][0]=grouped[key][0]+a
            else:grouped[key]=[np.array(a,copy=True),b]
        self.terms=tuple((a,b) for a,b in grouped.values());self.active=active;self.bath=bath
        dtype=np.result_type(*[x.dtype for a,b in self.terms for x in (a,b)])
        self.diagonal_values=sum((np.outer(np.diag(a),b.diagonal()).ravel() for a,b in self.terms),start=np.zeros(active*bath,dtype))
        super().__init__(dtype=dtype,shape=(active*bath,active*bath))
    def _matvec(self,vector):
        x=np.asarray(vector).reshape(self.active,self.bath)
        if self.dtype.kind!='c' and np.iscomplexobj(x):
            real=self._matvec(x.real.ravel()) if np.any(x.real) else np.zeros(x.size)
            imag=self._matvec(x.imag.ravel()) if np.any(x.imag) else np.zeros(x.size)
            return real+1j*imag
        y=np.zeros_like(x,dtype=np.result_type(self.dtype,x.dtype))
        for a,b in self.terms:y+=a@(b@x.T).T
        return y.ravel()
    def _matmat(self,vectors):return np.column_stack([self._matvec(v) for v in np.asarray(vectors).T])


def _constant_phase_real(value):
    """Exact real/pure-imaginary factorization; never discard a small part."""
    a=np.asarray(value)
    if not np.iscomplexobj(a):return 1.,a
    if not np.any(a.imag):return 1.,a.real
    if not np.any(a.real):return 1j,a.imag
    return None,a


def compress_coupled_tensor_operator(operator, *, maximum_error_cm=1e-5):
    """Compress separable terms with a conservative Frobenius error bound.

    Only tensor factors are recombined. No individual matrix element, mode or
    potential term is screened. The discarded left-factor singular norm times
    the complete right-bank Frobenius norm bounds the full operator error.
    Symmetrization preserves Hermiticity and cannot increase this bound for a
    Hermitian input. Callers still check eigenvectors against the original H.
    """
    from scipy.sparse import vstack
    from scipy.linalg import svd
    if (not isinstance(operator,_TensorSumOperator) or not np.isfinite(maximum_error_cm)
            or maximum_error_cm<=0):
        raise ValueError('Resident tensor operator and positive absolute error budget required')
    terms=operator.terms
    left=np.column_stack([a.ravel() for a,b in terms])
    right=vstack([b.reshape((1,operator.bath**2)) for a,b in terms],format='csr')
    right.sum_duplicates()
    norm=float(np.linalg.norm(right.data))
    u,s,vh=svd(left,full_matrices=False,check_finite=False)
    tail=np.r_[np.sqrt(np.cumsum(s[::-1]**2))[::-1],0.]*norm
    rank=next(i for i in range(1,len(s)+1) if tail[i]<=maximum_error_cm)
    diagnostics=dict(original_tensor_terms=len(terms),retained_left_rank=rank,
        maximum_requested_operator_error_cm=float(maximum_error_cm),
        operator_frobenius_error_bound_cm=float(tail[rank]),
        individual_matrix_elements_pruned=False,potential_modified=False)
    if 2*rank>=len(terms):
        return dict(operator=operator,diagnostics=dict(diagnostics,compression_used=False,
            operator_frobenius_error_bound_cm=0.,compressed_tensor_terms=len(terms)))
    factors=csr_matrix(s[:rank,None]*vh[:rank])@right
    compressed=[]
    for k in range(rank):
        a=u[:,k].reshape(operator.active,operator.active)
        b=factors[k].reshape((operator.bath,operator.bath)).tocsr()
        compressed.extend([(.5*a,b),(.5*a.conj().T,b.conj().T.tocsr())])
    result=_TensorSumOperator(compressed,operator.active,operator.bath)
    return dict(operator=result,diagnostics=dict(diagnostics,compression_used=True,
        compressed_tensor_terms=len(result.terms)))



@dataclass
class CoupledBathGalerkin:
    """Hermitian zero-J matrix and unfactored J and sym(JJ) coefficients."""
    hamiltonian_cm: object
    linear_terms: tuple
    quadratic_terms: tuple
    active_dimension: int
    bath_dimension: int
    occupations: tuple
    hermiticity_error_cm: float
    hermiticity_check: str

    def apply_terms(self, terms, vector):
        x=np.asarray(vector).reshape(self.active_dimension,self.bath_dimension)
        y=np.zeros_like(x,dtype=complex)
        for a,b in terms: y+=a@(b@x.T).T
        return y.ravel()

    def solve(self, state_count=6, *, initial_vectors=None):
        h=self.hamiltonian_cm;n=h.shape[0]
        if not 1 <= state_count < n: raise ValueError('invalid coupled state count')
        if initial_vectors is not None:
            x=np.asarray(initial_vectors)
            if (x.shape!=(n,state_count) or not np.all(np.isfinite(x))
                    or np.linalg.matrix_rank(x)!=state_count):
                raise ValueError('Independent finite checkpoint columns matching all requested states required')
            x=np.linalg.qr(x,mode='reduced')[0]
            rayleigh=np.real(np.einsum('ij,ij->j',x.conj(),h@x))
            diagonal=h.diagonal_values if isinstance(h,_TensorSumOperator) else np.diag(h)
            safe=np.maximum(np.real(diagonal)-float(min(rayleigh))+1.,1.)
            pre=LinearOperator(h.shape,matvec=lambda v:np.asarray(v).reshape(n)/safe,
                matmat=lambda v:v/safe[:,None],dtype=h.dtype)
            e,v=lobpcg(h,x,M=pre,largest=False,tol=1e-6,maxiter=500)
            order=np.argsort(e);e,v=e[order],v[:,order]
        elif n<=256:
            dense=h if isinstance(h,np.ndarray) else h@np.eye(n)
            e,v=eigh(dense,subset_by_index=(0,state_count-1))
        else:
            e,v=eigsh(h,k=state_count,which='SA',tol=1e-10,
                      v0=np.linspace(1,2,n))
            order=np.argsort(e);e,v=e[order],v[:,order]
        residual=np.linalg.norm(h@v-v*e,axis=0)
        return dict(energies_cm=e,vectors=v,eigen_residual_norm_cm=residual)

    def rotational_tensor(self, energy_cm, vector, *, tolerance=1e-9):
        """Isolated-band coefficient in the complete contracted space.

        The virtual response solves the projected full resolvent, rather
        than summing only over the few reported coupled eigenstates.
        """
        h=self.hamiltonian_cm;v=np.asarray(vector,complex)
        diagonal=h.diagonal_values if isinstance(h,_TensorSumOperator) else np.diag(h)
        if v.shape!=(h.shape[0],) or abs(np.vdot(v,v)-1)>1e-7:
            raise ValueError('normalized isolated coupled eigenvector required')
        residual=float(np.linalg.norm(h@v-energy_cm*v))
        if residual>1e-5:raise ValueError('coupled band is not a qualified eigenvector')
        lift=max(1.,float(np.max(abs(diagonal-energy_cm))))
        def action(x):return h@x-energy_cm*x+lift*v*np.vdot(v,x)
        op=LinearOperator(h.shape,matvec=action,dtype=complex)
        diag=diagonal-energy_cm+lift*abs(v)**2
        safe=np.where(abs(diag)>1.,diag,1.)
        pre=LinearOperator(h.shape,matvec=lambda x:x/safe,dtype=complex)
        columns=[];responses=[];errors=[]
        for terms in self.linear_terms:
            rhs=self.apply_terms(terms,v);rhs-=v*np.vdot(v,rhs)
            if np.linalg.norm(rhs)<1e-14:
                z=np.zeros_like(rhs);error=0.
            else:
                z,info=gmres(op,rhs,M=pre,rtol=tolerance,atol=1e-12,
                             restart=min(100,h.shape[0]),maxiter=100)
                z-=v*np.vdot(v,z);error=float(np.linalg.norm(action(z)-rhs))
                if info or error>max(1e-8,tolerance*10*np.linalg.norm(rhs)):
                    raise RuntimeError('coupled rotational resolvent did not converge')
            columns.append(rhs);responses.append(z);errors.append(error)
        direct=np.empty((3,3));virtual=np.empty((3,3))
        for a in range(3):
            for b in range(3):
                direct[a,b]=np.real(np.vdot(v,self.apply_terms(self.quadratic_terms[a][b],v)))
                virtual[a,b]=-np.real(np.vdot(columns[a],responses[b]))
        tensor=direct+(virtual+virtual.T)/2
        return dict(tensor_cm=tensor,direct_tensor_cm=direct,
                    virtual_tensor_cm=(virtual+virtual.T)/2,
                    response_residual_norm_cm=np.array(errors),
                    eigen_residual_norm_cm=residual)


def contract_active_quadratic_bath(
    grid_vectors, grid_momenta, kinetic_cm, potential_cm, bath_force_cm,
    bath_kinetic_reference_cm, bath_force_reference_cm, *,
    maximum_total_quanta=2, log_volume_gradient=None,
    bath_linear_force_cm=None, active_rotation_bath_gradient_cm=None,
    rotation_bath_gradient_cm=None, rotation_bath_second_cm=None,
    bath_rotation_bath_gradient_cm=None, complete_rotational_metric=False,
    maximum_bytes=1024**3, assembly='auto', measure='flat', basis_overlap=None,
    field_projection=None, bath_occupations=None,
):
    """Contract a full metric, quadratic local bath, and rotational vertices.

    Basis U(g,n) is orthonormal in the flat active-grid measure, pU(g,s,n)
    is the unprojected momentum action. G(g,s+t+3,s+t+3) includes all mixed
    kinetic blocks in the order active, bath, rotation. V(g) excludes the
    bath zero-point energy: it is generated variationally here. H_b(g,t,t)
    defines V_b=b.H_b.b/2, optionally plus f_b(g,t).b. G is evaluated at the
    bath origin and frozen in bath displacement in the zero-J block. This
    is a defined quadratic-bath approximation, not a complete metric jet.

    log_volume_gradient(g,s)=d_s log(rho), where rho is the physical volume
    density. It enters D_s=p_s+i*d_s log(rho)/2. A zero connection specifies
    an explicitly flat-measure model. Bath derivatives of the volume are
    absent in this frozen-in-bath metric. Supplied extra rotational metric
    derivatives specify G_active,R;b, G_RR;b and G_RR;bc (true derivatives,
    with the Taylor factors 1/2 included here). bath_rotation_bath_gradient_cm
    supplies G_bath,R;bath and is contracted as a Hermitian p-b vertex.
    Sparse tensor factors are used when the dense contraction is too large.
    No harmonic subtraction or
    retained spectator VPT2 is performed implicitly. When only a linear
    rotational vertex jet is known, complete_rotational_metric adds its
    matched positive Schur completion D(b)^T G_vv^-1 D(b). It is an explicit
    model choice, not a claim to know the full second native metric jet.

    With measure='weak', U and pU already carry the same physical volume
    weight and pU is the weighted derivative of the physical wavefunction.
    No additional volume connection may then be supplied. The bath volume
    remains frozen in its displacement, as in the default model.
    A supplied basis_overlap is the matching precomputed quadrature Gram,
    for example the coefficient-space Gram of physical-measure normalization.
    The caller owns its correspondence to U, as for the supplied metric jets.
    An optional exact field_projection(left_axis, field, right_axis) replaces
    the tall grid contraction. None denotes U; integer s denotes the same
    derivative[:,s] used here, including any supplied volume connection.
    It must preserve the complete supplied quadrature and retained basis.
    """
    if assembly not in ('auto','dense','tensor'):raise ValueError('assembly must be auto, dense or tensor')
    if measure not in ('flat','weak'):raise ValueError('measure must be flat or weak')
    if measure=='weak' and log_volume_gradient is not None:
        raise ValueError('A physical weak-measure basis must not receive a second volume connection')
    u=np.asarray(grid_vectors);pu=np.asarray(grid_momenta,complex)
    g=np.asarray(kinetic_cm,float);pot=np.asarray(potential_cm,float)
    hb=np.asarray(bath_force_cm,float)
    if u.ndim!=2 or pu.ndim!=3:raise ValueError('active basis and momenta required')
    ng,na=u.shape;dim=pu.shape[1]
    if hb.ndim!=3:raise ValueError('local bath force matrices required')
    nb=hb.shape[1];nv=dim+nb
    gram=u.conj().T@u if basis_overlap is None else np.asarray(basis_overlap)
    if nb<1 or dim<1:raise ValueError('active and bath spaces must both be nonempty')
    if (pu.shape!=(ng,dim,na) or g.shape!=(ng,nv+3,nv+3)
            or pot.shape!=(ng,) or hb.shape!=(ng,nb,nb)
            or not all(np.all(np.isfinite(x)) for x in (u,pu,g,pot,hb))
            or gram.shape!=(na,na) or not np.all(np.isfinite(gram))
            or not np.allclose(gram,np.eye(na),atol=1e-8)
            or not np.allclose(g,g.swapaxes(-1,-2),atol=1e-8)
            or not np.allclose(hb,hb.swapaxes(-1,-2),atol=1e-8)
            or np.min(np.linalg.eigvalsh(g))<=0
            or np.min(np.linalg.eigvalsh(hb))<=0):
        raise ValueError('compatible stable complete kinetic and force fields required')
    gradient=np.zeros((ng,dim)) if log_volume_gradient is None else np.asarray(log_volume_gradient,float)
    if gradient.shape!=(ng,dim) or not np.all(np.isfinite(gradient)):
        raise ValueError('finite volume gradient required')
    derivative=pu+.5j*gradient[:,:,None]*u[:,None,:]
    bath=harmonic_bath_matrices(bath_kinetic_reference_cm,bath_force_reference_cm,
                              maximum_total_quanta,occupations=bath_occupations)
    if len(bath['frequencies_cm'])!=nb:raise ValueError('bath oscillator dimension differs')
    ns=len(bath['occupations']);size=na*ns
    identity=sparse_eye(ns,format='csr');Q=bath['normal_coordinate'];P=bath['normal_momentum']
    qq=bath['normal_coordinate_second'];pp=bath['normal_momentum_second']
    qmap=bath['coordinate_map'];pmap=bath['momentum_map']
    gbb=np.einsum('bi,gbc,cj->gij',pmap,g[:,dim:nv,dim:nv],pmap,optimize=True)
    hbb=np.einsum('bi,gbc,cj->gij',qmap,hb,qmap,optimize=True)
    gab=np.einsum('gsb,bi->gsi',g[:,:dim,dim:nv],pmap,optimize=True)
    gbr=np.einsum('bi,gba->gia',pmap,g[:,dim:nv,nv:],optimize=True)
    def signature(a):return (a.__array_interface__['data'][0],a.shape,a.strides)
    phase_data={signature(a):_constant_phase_real(a) for a in [u,*[derivative[:,s] for s in range(dim)]]}
    axis_by_signature={signature(u):None, **{signature(derivative[:,s]):s for s in range(dim)}}
    def project(left,field,right):
        if field_projection is not None:
            value=np.asarray(field_projection(axis_by_signature[signature(left)],field,
                                              axis_by_signature[signature(right)]))
            if value.shape!=(na,na) or not np.all(np.isfinite(value)):
                raise ValueError('Exact field projection returned incompatible matrix')
            return value
        pl,l=phase_data[signature(left)];pr,r=phase_data[signature(right)]
        if pl is None or pr is None:return left.conj().T@(field[:,None]*right)
        phase=np.conjugate(pl)*pr
        if np.isreal(phase):phase=float(np.real(phase))
        return phase*(l.T@(field[:,None]*r))
    terms=[]
    def add(a,b):
        if np.max(abs(a))>1e-14 and b.nnz and np.max(abs(b.data))>1e-14:terms.append((a,b))
    add(project(u,pot,u),identity)
    for s in range(dim):
        for t in range(dim):add(.5*project(derivative[:,s],g[:,s,t],derivative[:,t]),identity)
    for b in range(nb):
        for c in range(nb):
            add(.5*project(u,gbb[:,b,c],u),pp[b,c])
            add(.5*project(u,hbb[:,b,c],u),qq[b,c])
        for s in range(dim):
            a=.5*project(derivative[:,s],gab[:,s,b],u)
            add(a+a.conj().T,P[b])
    if bath_linear_force_cm is not None:
        force=np.asarray(bath_linear_force_cm,float)
        if force.shape!=(ng,nb) or not np.all(np.isfinite(force)):raise ValueError('finite bath force field required')
        force=force@qmap
        for b in range(nb):add(project(u,force[:,b],u),Q[b])
    linear=[[] for _ in range(3)];quad=[[[] for _ in range(3)] for _ in range(3)]
    for a in range(3):
        for s in range(dim):
            x=.5*project(derivative[:,s],g[:,s,nv+a],u)
            linear[a].append((x+x.conj().T,identity))
        for b in range(nb):linear[a].append((project(u,gbr[:,b,a],u),P[b]))
        for c in range(3):quad[a][c].append((.5*project(u,g[:,nv+a,nv+c],u),identity))
    vertex_gradient=np.zeros((ng,nv,nb,3))
    if active_rotation_bath_gradient_cm is not None:
        ar=np.asarray(active_rotation_bath_gradient_cm,float)
        if ar.shape!=(ng,dim,nb,3) or not np.all(np.isfinite(ar)):raise ValueError('invalid active rotational metric derivative')
        vertex_gradient[:,:dim]=ar
        ar=np.einsum('gsba,bi->gsia',ar,qmap,optimize=True)
        for a,s,b in product(range(3),range(dim),range(nb)):
            x=.5*project(derivative[:,s],ar[:,s,b,a],u)
            linear[a].append((x+x.conj().T,Q[b]))
    if bath_rotation_bath_gradient_cm is not None:
        br=np.asarray(bath_rotation_bath_gradient_cm,float)
        if br.shape!=(ng,nb,nb,3) or not np.all(np.isfinite(br)):
            raise ValueError('invalid bath rotational metric derivative')
        vertex_gradient[:,dim:]=br
        br=np.einsum('gbka,bi,kj->gija',br,pmap,qmap,optimize=True)
        pq=bath['normal_momentum_coordinate']
        for a,b,k in product(range(3),range(nb),range(nb)):
            moment=(pq[b,k]+pq[b,k].conj().T)/2
            linear[a].append((project(u,br[:,b,k,a],u),moment))
    if complete_rotational_metric:
        if rotation_bath_second_cm is not None:
            raise ValueError('supply either an explicit second rotational metric or its linear-jet completion')
        rotation_bath_second_cm=rotational_metric_linear_completion(g[:,:nv,:nv],vertex_gradient)
    if rotation_bath_gradient_cm is not None:
        rr=np.asarray(rotation_bath_gradient_cm,float)
        if rr.shape!=(ng,nb,3,3) or not np.all(np.isfinite(rr)) or not np.allclose(rr,rr.swapaxes(-1,-2)):
            raise ValueError('invalid rotational metric bath derivative')
        rr=np.einsum('gbac,bi->giac',rr,qmap,optimize=True)
        for a,c,b in product(range(3),range(3),range(nb)):
            quad[a][c].append((.5*project(u,rr[:,b,a,c],u),Q[b]))
    if rotation_bath_second_cm is not None:
        rr=np.asarray(rotation_bath_second_cm,float)
        if rr.shape!=(ng,nb,nb,3,3) or not np.all(np.isfinite(rr)) or not np.allclose(rr,rr.swapaxes(-1,-2)) or not np.allclose(rr,rr.swapaxes(1,2)):
            raise ValueError('invalid rotational metric second bath derivative')
        rr=np.einsum('gbkac,bi,kj->gijac',rr,qmap,qmap,optimize=True)
        for a,c,b,k in product(range(3),range(3),range(nb),range(nb)):
            quad[a][c].append((.25*project(u,rr[:,b,k,a,c],u),qq[b,k]))
    storage=sum(a.nbytes+b.data.nbytes+b.indices.nbytes+b.indptr.nbytes for a,b in terms)
    if storage>maximum_bytes:raise MemoryError('coupled tensor factors exceed memory budget')
    if assembly=='dense' and 3*size*size*np.dtype(complex).itemsize>maximum_bytes:
        raise MemoryError('dense coupled matrix exceeds memory budget')
    if assembly=='dense' or (assembly=='auto' and size<=2000 and 3*size*size*np.dtype(complex).itemsize<=maximum_bytes):
        h=sum((np.kron(a,b.toarray()) for a,b in terms),start=np.zeros((size,size),complex))
        error=float(np.max(abs(h-h.conj().T)))
        if error>1e-7:raise ValueError('coupled kinetic contraction is not Hermitian')
        h=(h+h.conj().T)/2
        if np.max(abs(h.imag))<1e-10:h=h.real.copy()
    else:
        h=_TensorSumOperator(terms,na,ns);rng=np.random.default_rng(0);error=0.
        for _ in range(3):
            x=rng.normal(size=size)+1j*rng.normal(size=size);x/=np.linalg.norm(x)
            y=rng.normal(size=size)+1j*rng.normal(size=size);y/=np.linalg.norm(y)
            error=max(error,float(abs(np.vdot(x,h@y)-np.vdot(h@x,y))))
        if error>1e-7:raise ValueError('coupled tensor factors are not Hermitian')
    return CoupledBathGalerkin(h,tuple(tuple(v) for v in linear),
                              tuple(tuple(tuple(v) for v in row) for row in quad),
                              na,ns,bath['occupations'],error,
                              'dense maximum element' if isinstance(h,np.ndarray) else 'three normalized sesquilinear probes')


__all__=['harmonic_bath_matrices','contract_active_quadratic_bath','CoupledBathGalerkin','rotational_metric_linear_completion',
         'compress_coupled_tensor_operator']
