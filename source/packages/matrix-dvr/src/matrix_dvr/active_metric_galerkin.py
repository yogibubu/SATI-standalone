"""Physical-measure active Galerkin contraction with full rotational tensors."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class ActiveMetricGalerkin:
    hamiltonian_cm: np.ndarray
    linear_cm: np.ndarray
    quadratic_cm: np.ndarray
    transform: np.ndarray
    sectors: np.ndarray
    orthogonality_error: float

    def lowest_sector_branches(self,*,retained_per_sector=16,maximum_J=5):
        from matrix_rovib.native_rotor import angular_momentum_operators
        if len(set(self.sectors.tolist()))!=2:raise ValueError('Two exchange sectors required for this branch comparison')
        branches={};gaps={}
        for sector in sorted(set(self.sectors.tolist())):
            ids=np.flatnonzero(self.sectors==sector);n=min(retained_per_sector,len(ids))
            h=self.hamiltonian_cm[np.ix_(ids,ids)];E,C=eigh(h,subset_by_index=(0,n-1))
            linear=np.array([C.T.conj()@v[np.ix_(ids,ids)]@C for v in self.linear_cm]);quad=np.array([[C.T.conj()@v[np.ix_(ids,ids)]@C for v in row] for row in self.quadratic_cm]);gaps[int(sector)]=float(E[0])
            for J in range(maximum_J+1):
                angular=angular_momentum_operators(J,'Ir');nr=2*J+1;H=np.kron(np.diag(E),np.eye(nr)).astype(complex)
                for a in range(3):
                    H+=np.kron(linear[a],angular[a])
                    for b in range(3):H+=np.kron(quad[a,b],.5*(angular[a]@angular[b]+angular[b]@angular[a]))
                branches[J,int(sector)]=float(eigh(H,subset_by_index=(0,0),eigvals_only=True)[0])
        sectors=sorted(gaps);rows=[]
        for J in range(1,maximum_J+1):
            nu=[(branches[J,s]-branches[J-1,s])*29979.2458 for s in sectors]
            rows.append(dict(J_upper=J,sector_frequencies_MHz=nu,sector_difference_MHz=nu[0]-nu[1]))
        return dict(zero_J_ground_energies_cm=gaps,zero_J_sector_gap_cm=gaps[sectors[1]]-gaps[sectors[0]],transitions=rows)


    def restrict_primitive_space(self,indices):
        """Reuse a completed normalized contraction in a nested primitive span."""
        ids=np.asarray(indices,int)
        if ids.ndim!=1 or not len(ids) or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(self.sectors)):raise ValueError('Distinct primitive subspace indices required')
        inverse=np.linalg.solve(self.transform,np.eye(len(self.sectors)));A=inverse[:,ids];S=A.T.conj()@A;labels=self.sectors[ids];T=np.zeros_like(S)
        for sector in set(labels.tolist()):
            ii=np.flatnonzero(labels==sector);ev,U=eigh(S[np.ix_(ii,ii)])
            if min(ev)<=1e-10*max(ev):raise ValueError('Restricted primitive overlap is singular')
            T[np.ix_(ii,ii)]=(U/np.sqrt(ev))@U.T.conj()
        Z=A@T
        return ActiveMetricGalerkin(Z.T.conj()@self.hamiltonian_cm@Z,np.array([Z.T.conj()@v@Z for v in self.linear_cm]),np.array([[Z.T.conj()@v@Z for v in row] for row in self.quadratic_cm]),T,labels,float(np.max(abs(Z.T.conj()@Z-np.eye(len(ids))))))

    def isolated_band_rotational_tensors(self):
        from matrix_trinity.coupled_bath_galerkin import CoupledBathGalerkin
        rows=[]
        for sector in sorted(set(self.sectors.tolist())):
            ids=np.flatnonzero(self.sectors==sector);h=self.hamiltonian_cm[np.ix_(ids,ids)];E,C=eigh(h,subset_by_index=(0,0));one=np.ones((1,1))
            linear=tuple(((v[np.ix_(ids,ids)],one),) for v in self.linear_cm)
            quadratic=tuple(tuple(((v[np.ix_(ids,ids)],one),) for v in row) for row in self.quadratic_cm)
            native=CoupledBathGalerkin(h,linear,quadratic,len(ids),1,((),),float(np.max(abs(h-h.T.conj()))),'dense active weak-form matrix')
            result=native.rotational_tensor(float(E[0]),C[:,0],tolerance=1e-10)
            tensor=result['tensor_cm']*29979.2458;constants=np.linalg.eigvalsh(tensor)[::-1]
            rows.append(dict(sector=int(sector),energy_cm=float(E[0]),rotational_constants_MHz=constants.tolist(),tensor_MHz=tensor.tolist(),direct_tensor_MHz=(result['direct_tensor_cm']*29979.2458).tolist(),virtual_tensor_MHz=(result['virtual_tensor_cm']*29979.2458).tolist(),response_residual_cm=result['response_residual_norm_cm'].tolist()))
        return dict(sectors=rows,sector_difference_B_plus_C_MHz=float(sum(rows[0]['rotational_constants_MHz'][1:])-sum(rows[1]['rotational_constants_MHz'][1:])),scope='Complete same-sector vibrational resolvent in the supplied active basis; quadratic rotational coefficient, not a finite-J spectrum')


def contract_active_metric_galerkin(values,momenta,kinetic_cm,potential_cm,*,volume_density,sectors):
    U=np.asarray(values);P=np.asarray(momenta);G=np.asarray(kinetic_cm);V=np.asarray(potential_cm);rho=np.asarray(volume_density);labels=np.asarray(sectors,int)
    ng,nb=U.shape;d=P.shape[1]
    if P.shape!=(ng,d,nb) or G.shape!=(ng,d+3,d+3) or V.shape!=(ng,) or rho.shape!=(ng,) or labels.shape!=(nb,) or np.any(rho<=0):raise ValueError('Compatible analytic weak-basis values, derivatives and full metric required')
    root=np.sqrt(rho);u=U*root[:,None];p=P*root[:,None,None];S=u.T.conj()@u;T=np.zeros_like(S)
    cross=labels[:,None]!=labels[None,:]
    if np.max(abs(S[cross]),initial=0)>1e-9*max(1.,np.max(abs(S))):raise ValueError('Physical quadrature breaks declared sectors')
    for sector in set(labels.tolist()):
        ids=np.flatnonzero(labels==sector);s,w=eigh(S[np.ix_(ids,ids)])
        if min(s)<=1e-10*max(s):raise ValueError('Quadrature does not resolve the active basis')
        T[np.ix_(ids,ids)]=(w/np.sqrt(s))@w.T.conj()
    u=u@T;p=p@T;H=(u.T.conj()@(V[:,None]*u)).astype(complex);linear=np.zeros((3,nb,nb),complex);quad=np.empty((3,3,nb,nb),complex)
    for i in range(d):
        for j in range(d):H+=.5*p[:,i].T.conj()@(G[:,i,j,None]*p[:,j])
    for a in range(3):
        for i in range(d):
            term=.5*p[:,i].T.conj()@(G[:,i,d+a,None]*u);linear[a]+=term+term.T.conj()
        for b in range(3):quad[a,b]=.5*u.T.conj()@(G[:,d+a,d+b,None]*u)
    for operator in [H,*linear,*quad.reshape(9,nb,nb)]:
        if np.max(abs(operator[cross]),initial=0)>1e-8*max(1.,np.max(abs(operator))):raise ValueError('Contracted physical field breaks declared sectors')
    H=.5*(H+H.T.conj())
    return ActiveMetricGalerkin(H,linear,quad,T,labels,float(np.max(abs(u.T.conj()@u-np.eye(nb)))))
