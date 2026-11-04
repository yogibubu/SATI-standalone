"""Exact rotation-reduced Cartesian mass metric for disjoint rigid rotors.

Angles follow right-handed rotations about the supplied oriented bonds.
Coordinates are in bohr and masses in electron masses. The returned inverse
metric and full rotational/internal volume connection feed CurvilinearSineDVR.
No potential values or electronic evaluations are produced here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RigidRotorMetric:
    reference_bohr: np.ndarray
    masses_electron: np.ndarray
    bonds_zero_based: tuple
    fragments_zero_based: tuple

    def __post_init__(self):
        r=np.asarray(self.reference_bohr,float);m=np.asarray(self.masses_electron,float)
        if r.shape!=(len(m),3) or np.any(m<=0) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(m)) or len(self.bonds_zero_based)!=len(self.fragments_zero_based):raise ValueError('Finite geometry, masses and one fragment per oriented bond required')
        seen=set()
        for (left,right),fragment in zip(self.bonds_zero_based,self.fragments_zero_based):
            ids=set(fragment)
            if not ids or seen&ids or not ids<=set(range(len(m))) or left==right or min(left,right)<0 or max(left,right)>=len(m) or np.linalg.norm(r[right]-r[left])<1e-10:raise ValueError('Disjoint rotor fragments and nonzero oriented axes required')
            seen|=ids
            # Another moving fragment must not move this axis.
            for pair in self.bonds_zero_based:
                for atom in pair:
                    if atom in ids and np.linalg.norm(np.cross(r[right]-r[left],r[atom]-r[left]))>1e-8:raise ValueError('Every rotor axis must remain fixed under other fragment rotations')

    def geometry_and_tangents(self,points):
        q=np.asarray(points,float);r0=np.asarray(self.reference_bohr);mass=np.asarray(self.masses_electron);d=len(self.bonds_zero_based)
        if q.ndim!=2 or q.shape[1]!=d or not np.all(np.isfinite(q)):raise ValueError('Finite rotor coordinate matrix required')
        r=np.broadcast_to(r0,(len(q),)+r0.shape).copy();v=np.zeros(r.shape+(d,));acc=np.zeros_like(v)
        for j,((left,right),fragment) in enumerate(zip(self.bonds_zero_based,self.fragments_zero_based)):
            ids=np.asarray(fragment,int);axis=r0[right]-r0[left];axis/=np.linalg.norm(axis);offset=r0[ids]-r0[left]
            angle=q[:,j,None,None];r[:,ids]=r0[left]+offset*np.cos(angle)+np.cross(axis,offset)*np.sin(angle)+(offset@axis)[None,:,None]*axis*(1-np.cos(angle))
            v[...,j][:,ids]=np.cross(axis,r[:,ids]-r0[left]);acc[...,j][:,ids]=np.cross(axis,v[...,j][:,ids])
        return r,v,acc

    def fields(self,points):
        r,v,acc=self.geometry_and_tangents(points);mass=np.asarray(self.masses_electron);d=len(self.bonds_zero_based)
        r-=np.einsum('a,nac->nc',mass,r)[:,None,:]/mass.sum();v-=np.einsum('a,nacd->ncd',mass,v)[:,None,:,:]/mass.sum();acc-=np.einsum('a,nacd->ncd',mass,acc)[:,None,:,:]/mass.sum()
        rotations=np.stack([np.cross(axis,r) for axis in np.eye(3)],axis=-1);columns=np.concatenate([v,rotations],axis=-1)
        metric=np.einsum('naci,a,nacj->nij',columns,mass,columns,optimize=True);inverse=np.linalg.inv(metric)
        if np.any(np.linalg.eigvalsh(metric)<=0):raise ValueError('Positive internal/rotational metric required')
        gradient=[]
        for j in range(d):
            dc=np.zeros_like(columns);dc[:,:,:,j]=acc[:,:,:,j];dc[:,:,:,d:]=np.stack([np.cross(axis,v[:,:,:,j]) for axis in np.eye(3)],axis=-1)
            derivative=np.einsum('naci,a,nacj->nij',dc,mass,columns,optimize=True);derivative+=derivative.swapaxes(1,2)
            gradient.append(.5*np.einsum('nij,nji->n',inverse,derivative))
        return inverse[:,:d,:d]*219474.6313705,np.column_stack(gradient)
