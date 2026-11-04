"""Periodic Cartesian path with exact independent terminal dihedral offsets.

Terminal rotations follow the instantaneous bond axes, centers and transverse
geometry. A fixed-reference Rodrigues axis would change the requested internal
angles whenever the supplied path deforms the molecular frame.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TerminalAnglePath:
    tree: object
    reference_values: np.ndarray
    active_indices: np.ndarray
    origin: float
    frequencies: np.ndarray
    coefficients: np.ndarray
    atoms: tuple
    centers_coefficients: np.ndarray
    radial_coefficients: np.ndarray
    slope_coefficients: np.ndarray

    @classmethod
    def from_samples(cls,tree,reference_values,active_indices,angles,geometries,*,preserve_terminal_profile=False):
        q0=np.asarray(reference_values,float);active=np.asarray(active_indices,int)
        phi=np.asarray(angles,float);x=np.asarray(geometries,float)
        if (active.shape!=(4,) or phi.ndim!=1 or len(phi)<17 or x.shape!=(len(phi),len(tree.order),3)
                or not np.allclose(phi,np.linspace(phi[0],phi[0]+2*np.pi,len(phi)),atol=1e-12)
                or np.max(abs(x[-1]-x[0]))>1e-8):
            raise ValueError('Three terminal dihedrals and one uniform closed periodic path required')
        reference=tree.geometry(q0);jet=tree.cartesian_jet(reference,second=False)
        nodes=[];centers=[];radials=[];slopes=[];atoms=[];signs=[]
        for index in active[:3]:
            primitive=tree.primitives[index]
            if primitive.kind!='dihedral':raise ValueError('Terminal dihedral coordinates required')
            _,b,c,h=primitive.atoms;axis=reference[b]-reference[c];axis/=np.linalg.norm(axis)
            radial=reference[h]-reference[c]-axis*np.dot(reference[h]-reference[c],axis)
            tangent=jet['tangent'][:,index].reshape(len(tree.order),3)
            if np.max(abs(np.delete(tangent,h,axis=0)))>1e-8:raise ValueError('Selected coordinate is not a terminal rotation')
            signs.append(np.sign(np.dot(np.cross(axis,radial),tangent[h])));atoms.append(h)
        for original in x[:-1]:
            values=tree.values(original)
            if not preserve_terminal_profile:values[active[:3]]=q0[active[:3]]
            current=tree.geometry(values);nodes.append(current);cc=[];rr=[];ss=[]
            for index,sign in zip(active[:3],signs):
                _,b,c,h=tree.primitives[index].atoms;axis=current[b]-current[c];axis/=np.linalg.norm(axis)
                center=current[c]+axis*np.dot(current[h]-current[c],axis);radial=current[h]-center
                cc.append(center);rr.append(radial);ss.append(sign*np.cross(axis,radial))
            centers.append(cc);radials.append(rr);slopes.append(ss)
        n=len(nodes);transform=lambda a:np.fft.fft(np.asarray(a),axis=0)/n
        return cls(tree,q0,active,float(phi[0]),np.fft.fftfreq(n,d=1/n),transform(nodes),tuple(atoms),
            transform(centers),transform(radials),transform(slopes))

    def evaluate(self,points,*,second=False):
        q=np.asarray(points,float)
        if q.ndim!=2 or q.shape[1]!=4 or not np.all(np.isfinite(q)):raise ValueError('Finite four-coordinate points required')
        theta,ids=np.unique(q[:,3],return_inverse=True)
        phase=np.exp(1j*np.outer(theta-self.origin,self.frequencies))
        def periodic(coeff,order):
            return np.einsum('pk,k...->p...',phase*(1j*self.frequencies)**order,coeff).real[ids]
        x=periodic(self.coefficients,0);D=np.zeros(x.shape+(4,));D[:,:,:,3]=periodic(self.coefficients,1)
        DD=np.zeros(x.shape+(4,4)) if second else None
        if second:DD[:,:,:,3,3]=periodic(self.coefficients,2)
        c=periodic(self.centers_coefficients,0);r=periodic(self.radial_coefficients,0);s=periodic(self.slope_coefficients,0)
        cp=periodic(self.centers_coefficients,1);rp=periodic(self.radial_coefficients,1);sp=periodic(self.slope_coefficients,1)
        if second:
            cpp=periodic(self.centers_coefficients,2);rpp=periodic(self.radial_coefficients,2);spp=periodic(self.slope_coefficients,2)
        for i,h in enumerate(self.atoms):
            co=np.cos(q[:,i])[:,None];si=np.sin(q[:,i])[:,None]
            x[:,h]=c[:,i]+co*r[:,i]+si*s[:,i]
            D[:,h,:,i]=-si*r[:,i]+co*s[:,i]
            D[:,h,:,3]=cp[:,i]+co*rp[:,i]+si*sp[:,i]
            if second:
                DD[:,h,:,i,i]=-co*r[:,i]-si*s[:,i]
                DD[:,h,:,i,3]=DD[:,h,:,3,i]=-si*rp[:,i]+co*sp[:,i]
                DD[:,h,:,3,3]=cpp[:,i]+co*rpp[:,i]+si*spp[:,i]
        result=dict(geometry_A=x,tangents=D)
        if second:result['second_derivatives']=DD
        return result
