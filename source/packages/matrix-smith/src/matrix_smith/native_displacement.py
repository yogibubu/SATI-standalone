"""Nonredundant native displacements from one frozen SMITH definition.

Bond lengths use the ZAFF inverse-distance coordinate. Bends retain SMITH's
regular angular/linear-bend functions; torsional phases are periodic. This
avoids introducing cot(theta0) into a nearly linear bending coordinate.
The returned Cartesian derivatives are per angstrom.
"""
import numpy as np
from .evaluation import evaluate_primitive_values,build_primitive_b_matrix
from .second_derivatives import analytic_primitive_hessian
from matrix_zaff.primitive_second_derivative import analytic_primitive_second_derivative


class NativeInternalDisplacement:
    def __init__(self,definition,reference):
        self.definition=definition;self.reference=np.asarray(reference,float)
        self.primitives=tuple(definition.primitives)
        ids={p.identifier:i for i,p in enumerate(self.primitives)}
        self.combinations=np.zeros((len(definition.gics),len(ids)))
        for i,g in enumerate(definition.gics):
            for key,value in g.coefficients:self.combinations[i,ids[key]]=value
        values=evaluate_primitive_values(definition,coordinates_angstrom=self.reference)
        self.values=np.array([values[p.identifier] for p in self.primitives])

    def value(self,xyz):
        xyz=np.asarray(xyz,float)
        values=evaluate_primitive_values(self.definition,coordinates_angstrom=xyz)
        q=np.array([values[p.identifier] for p in self.primitives]);u=q-self.values
        first=np.ones(len(q));second=np.zeros(len(q))
        for i,p in enumerate(self.primitives):
            if p.function=='R':
                r0=self.values[i]
                if r0<=0 or q[i]<=0:raise ValueError('positive native bond lengths required')
                u[i]=r0*r0*(1/r0-1/q[i]);first[i]=r0*r0/q[i]**2;second[i]=-2*r0*r0/q[i]**3
            elif p.function in ('D','IMPD'):
                angle=u[i];u[i]=np.sin(angle);first[i]=np.cos(angle);second[i]=-np.sin(angle)
        b=np.asarray(build_primitive_b_matrix(self.definition,coordinates_angstrom=xyz).rows)
        h=[]
        for p in self.primitives:
            if p.function in ('SOOP','SYCART'):hh=analytic_primitive_hessian(p,xyz)
            else:
                hh=np.zeros((xyz.size,xyz.size))
                for i,j,v in analytic_primitive_second_derivative(p,xyz,
                        reference_coordinates=self.definition.reference_coordinates_angstrom).entries:hh[i,j]+=v
            h.append(hh)
        j=first[:,None]*b
        h=first[:,None,None]*np.asarray(h)+second[:,None,None]*np.einsum('pi,pj->pij',b,b)
        c=self.combinations
        return c@u,(c@j).T,np.einsum('ap,pij->ija',c,h)
