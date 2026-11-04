"""Resident stationary fields completed by acquired local E/G/H constraints.

The donor weight uses only the acquired transverse Hessian in a mass-whitened
native basis. All higher derivatives remain derivatives of one scalar field.
No withheld anharmonic tensor or rate enters the interpolation weight.
This is an experimental closure: its algebraic constraints do not qualify
the molecular anharmonic spectrum, which must be checked independently.
"""
import numpy as np
from scipy.linalg import null_space
from matrix_chem import BOHR_TO_ANGSTROM
from .native_local_field import NativeLocalField
from .native_pic_field import NativePicPoint


class NativeFieldSum:
    def __init__(self,fields,weights):
        self.fields=tuple(fields);self.weights=np.asarray(weights,float)
        self.atoms=tuple(self.fields[0].atoms) if self.fields else ()
        if (not self.atoms or self.weights.shape!=(len(self.fields),)
                or not np.all(np.isfinite(self.weights))
                or any(tuple(f.atoms)!=self.atoms for f in self.fields)):
            raise ValueError('aligned resident fields and finite weights required')
    def __call__(self,xyz):
        e=0.;g=np.zeros(np.size(xyz));h=np.zeros((np.size(xyz),np.size(xyz)))
        for field,weight in zip(self.fields,self.weights):
            if weight==0:continue
            p=field(xyz);e+=weight*p.energy_hartree
            g+=weight*p.gradient_hartree_per_bohr;h+=weight*p.hessian_hartree_per_bohr2
        return NativePicPoint(float(e),g,h)


def complete_native_anchor(target,xyz,masses,tangent,ts_donor,minimum_donor,*,rigid_constraint_rows=None):
    """Retain acquired E/G/H and interpolate stationary higher derivatives.

    The normalized tangent is mass weighted. Geometry, masses, the target
    Hessian and both resident stationary fields completely determine the rule.
    """
    xyz=np.asarray(xyz,float);mass=np.asarray(masses,float);t=np.asarray(tangent,float)
    if (mass.shape!=(len(xyz),) or np.any(mass<=0) or t.shape!=(xyz.size,)
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(mass))):
        raise ValueError('consistent geometry, masses and mass-weighted tangent required')
    _,j,h=target.kernel.value(xyz);n=j.shape[1]
    if rigid_constraint_rows is None:first=np.linalg.pinv(j.T,rcond=1e-10)
    else:
        constraints=np.asarray(rigid_constraint_rows,float)
        system=np.vstack((j.T,constraints))
        if system.shape!=(xyz.size,xyz.size):raise ValueError('complete native plus rigid constraints required')
        first=np.linalg.solve(system,np.vstack((np.eye(n),np.zeros((len(constraints),n)))))
    # Exact inverse-coordinate curvature in the declared frozen rigid gauge.
    second=-np.einsum('ia,ajk,jb,kc->ibc',first,h.transpose(2,0,1),first,first,optimize=True)
    metric=np.linalg.inv(first.T@(np.repeat(mass,3)[:,None]*first))
    ev,v=np.linalg.eigh((metric+metric.T)/2)
    if ev.min()<=0:raise ValueError('positive nonredundant native geometry metric required')
    half=(v*np.sqrt(ev))@v.T
    direction=np.linalg.solve(half,j.T@(t/np.repeat(np.sqrt(mass),3)))
    if np.linalg.norm(direction)<1e-10:raise ValueError('represented IRC tangent required')
    bath=null_space(direction[None,:]);a=half@bath
    def curvature(field):
        p=field(xyz);g=p.gradient_hartree_per_bohr/BOHR_TO_ANGSTROM
        hh=p.hessian_hartree_per_bohr2/BOHR_TO_ANGSTROM**2
        native=first.T@hh@first+np.einsum('i,iab->ab',g,second)
        return a.T@native@a
    acquired=curvature(target);stationary=curvature(ts_donor);minimum=curvature(minimum_donor)
    difference=minimum-stationary;den=float(np.sum(difference*difference))
    alpha=float(np.clip(np.sum((acquired-stationary)*difference)/den,0.,1.)) if den>0 else 0.
    baseline=NativeFieldSum([ts_donor,minimum_donor],[1-alpha,alpha])
    reference=target(xyz);donor=baseline(xyz)
    delta_g=(reference.gradient_hartree_per_bohr-donor.gradient_hartree_per_bohr)/BOHR_TO_ANGSTROM
    delta_h=(reference.hessian_hartree_per_bohr2-donor.hessian_hartree_per_bohr2)/BOHR_TO_ANGSTROM**2
    native_h=first.T@delta_h@first+np.einsum('i,iab->ab',delta_g,second)
    correction=NativeLocalField(target.atoms,target.kernel,reference.energy_hartree-donor.energy_hartree,
        first.T@delta_g,(native_h+native_h.T)/2)
    result=NativeFieldSum([baseline,correction],[1.,1.])
    check=result(xyz)
    audit=dict(minimum_weight=alpha,
        donor_relative_transverse_Hessian_residual=float(np.linalg.norm((1-alpha)*stationary+alpha*minimum-acquired)/max(np.linalg.norm(acquired),1e-30)),
        energy_residual_hartree=float(check.energy_hartree-reference.energy_hartree),
        gradient_residual_hartree_per_bohr=float(np.max(abs(check.gradient_hartree_per_bohr-reference.gradient_hartree_per_bohr))),
        Hessian_residual_hartree_per_bohr2=float(np.max(abs(check.hessian_hartree_per_bohr2-reference.hessian_hartree_per_bohr2))),
        internal_gradient_residual_hartree=float(np.max(abs(first.T@(check.gradient_hartree_per_bohr-reference.gradient_hartree_per_bohr)/BOHR_TO_ANGSTROM))),
        internal_Hessian_residual_hartree=float(np.max(abs(first.T@(check.hessian_hartree_per_bohr2-reference.hessian_hartree_per_bohr2)@first/BOHR_TO_ANGSTROM**2
            +np.einsum('i,iab->ab',(check.gradient_hartree_per_bohr-reference.gradient_hartree_per_bohr)/BOHR_TO_ANGSTROM,second)))))
    return result,audit
