# SPDX-License-Identifier: LGPL-3.0-or-later
# xTB ddvopt reference: Copyright (C) 2017-2020 Stefan Grimme.
"""Factorized simple-primitive Lindh model, with separate D2 pair blocks.

Equations/helpers follow the existing independently translated xTB reference
in lindh_hessian. No dense Cartesian Hessian or electronic gradient is built.
All-pair parity retains the O(N^2) pair terms of ddvopt. No additional pair
cutoff, out-of-plane term, improper term, or refitted coefficient is applied.
Source: https://github.com/grimme-lab/xtb/blob/v6.7.1/src/model_hessian.f90
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial import cKDTree
from . import lindh_hessian as ref
from .primitive_coordinates import (
    Primitive,
    compiled_local_primitive_values_and_csr,
    grad_primitive,
)
from .structural_corrections import BOHR_TO_ANGSTROM
from .topology.elements import atomic_number


@dataclass(frozen=True)
class SimpleLindhOperator:
    b_matrix: object  # primitive derivatives with respect to Cartesian bohr
    curvatures: np.ndarray
    pair_hessian: object  # D2 contribution in hartree/bohr^2
    families: tuple[str, ...]
    audit: dict
    primitive_supports: tuple[tuple[int, ...], ...] = ()

    def cartesian_action(self, displacement_angstrom):
        vector=np.asarray(displacement_angstrom,dtype=float)
        if vector.shape!=(self.b_matrix.shape[1],) or not np.isfinite(vector).all():
            raise ValueError('invalid simple SH displacement')
        return (self.b_matrix.T@(self.curvatures*(self.b_matrix@vector))+
                self.pair_hessian@vector)/BOHR_TO_ANGSTROM**2

    def realize(self, forward, transpose):
        return lambda v:np.asarray(transpose(self.cartesian_action(forward(v))))

    def dense(self, cartesian_from_active):
        """Bounded audit/explicit dense route; never called by realize."""
        j=np.asarray(cartesian_from_active,dtype=float)
        a=self.b_matrix@j
        return (a.T@(self.curvatures[:,None]*a)+j.T@(self.pair_hessian@j))/BOHR_TO_ANGSTROM**2


def simple_lindh_operator(atoms, coordinates_angstrom, *, s6=20., pair_error_tolerance=None):
    # Match the Cartesian input supplied by LINK to the installed xTB audit.
    xyz=np.asarray(coordinates_angstrom,dtype=float)*(1.0/0.52917721092)
    n=len(atoms)
    if n<1 or xyz.shape!=(n,3) or not np.isfinite(xyz).all():
        raise ValueError('invalid simple Lindh geometry')
    numbers=tuple(int(atomic_number(a) or 0) for a in atoms)
    if any(z<1 or z>100 for z in numbers):raise ValueError('Lindh requires elements Z=1..100')
    if not math.isfinite(s6) or s6<0:raise ValueError('invalid D2 scale')
    locality_audit = {}
    if pair_error_tolerance is None:
        tree=cKDTree(xyz)
        if n>1 and np.min(tree.query(xyz,k=2)[0][:,1])<=np.finfo(float).eps:
            raise ValueError('coincident atoms are invalid for the Lindh Hessian')
        neighbors=[set(v)-{i} for i,v in enumerate(tree.query_ball_point(xyz,math.sqrt(ref._PAIR_CUTOFF2_BOHR)))]
        pair_neighbors = None
    else:
        from .lindh_locality import certified_pair_radius, linked_cell_neighbors
        radius, locality_audit = certified_pair_radius(xyz, numbers, pair_error_tolerance, s6=s6)
        pair_neighbors = linked_cell_neighbors(xyz, radius)
        neighbors = [{j for j in row if float((xyz[i]-xyz[j])@(xyz[i]-xyz[j])) <= ref._PAIR_CUTOFF2_BOHR}
                     for i,row in enumerate(pair_neighbors)]
    periods=tuple(ref._period_row(z) for z in numbers)
    def rho(i,j):
        d=xyz[i]-xyz[j];a=ref._A_AV[periods[i],periods[j]];r=ref._R_AV[periods[i],periods[j]]
        return math.exp(a*(r*r-float(d@d)))
    primitives=[];curvatures=[];families=[];supports=[]
    def add(ids,k,family,*,mode=0):
        if k==0:return
        kind={'R':'bond','A':'angle','L':'linear_bend','D':'dihedral'}[family]
        primitives.append(Primitive(kind,tuple(ids),mode=mode))
        curvatures.append(float(k));families.append(family);supports.append(tuple(ids))
    pr=[];pc=[];pv=[]
    pairs=(((i,j) for i in range(n) for j in range(i)) if pair_neighbors is None else
           ((i,j) for i,row in enumerate(pair_neighbors) for j in sorted(row) if j<i))
    pair_count=0
    for i,j in pairs:
        pair_count+=1;d=xyz[i]-xyz[j]
        add((i,j),ref._STRETCH_FORCE*rho(i,j),'R')
        if s6:
            block=-ref._d2_pair_hessian(d,numbers[i],numbers[j],s6=s6)
            for a,sa in ((i,1),(j,-1)):
                for b,sb in ((i,1),(j,-1)):
                    for u in range(3):
                        for v in range(3):
                            pr.append(3*a+u);pc.append(3*b+v);pv.append(sa*sb*block[u,v])
    for center in range(n):
        for first in sorted(neighbors[center]):
            for second in sorted(neighbors[center]&neighbors[first]):
                if second>=first:continue
                left=xyz[first]-xyz[center];right=xyz[second]-xyz[center]
                cosine=float(left@right/(np.linalg.norm(left)*np.linalg.norm(right)))
                # Preserve the ddvopt exclusion exactly.  Nearly coincident
                # rays are not angular coordinates: retaining one here makes
                # the factorized primitive pool disagree with the dense xTB
                # reference and eventually evaluates an undefined acos(1).
                if cosine >= 1.0 - 1.0e-14:
                    continue
                ids=(first,center,second)
                cross=np.cross(left,right);cross2=float(cross@cross)
                sine=(math.sqrt(cross2) if cross2>=1.e-14 else 0.)/(np.linalg.norm(left)*np.linalg.norm(right))
                if sine>1.e-10:
                    family='A'; modes=(0,)
                else:
                    family='L'; modes=(-1,-2)
                for mode in modes:
                    add(ids,ref._BEND_FORCE*rho(center,first)*rho(center,second),family,
                        mode=mode)
    for second in range(n):
        for third in sorted(neighbors[second]):
            common=neighbors[second]&neighbors[third]
            for first in sorted(common):
                for fourth in sorted(common&neighbors[first]):
                    if n*second+first<=n*third+fourth:continue
                    r12=xyz[first]-xyz[second];r23=xyz[second]-xyz[third];r34=xyz[third]-xyz[fourth]
                    norms=[np.linalg.norm(v) for v in (r12,r23,r34)]
                    if abs(r12@r23/(norms[0]*norms[1]))>ref._TORSION_COSINE_LIMIT or abs(r34@r23/(norms[2]*norms[1]))>ref._TORSION_COSINE_LIMIT:continue
                    ids=(first,second,third,fourth)
                    add(ids,ref._TORSION_FORCE*rho(first,second)*rho(second,third)*rho(third,fourth),'D')
    compiled=compiled_local_primitive_values_and_csr(primitives,xyz)
    if compiled is None:
        # Source-tree/debug fallback.  Production installations compile this
        # complete batch and may evaluate its independent rows in parallel.
        data=[];columns=[];pointers=[0]
        for primitive in primitives:
            gradient=np.asarray(grad_primitive(primitive,xyz),dtype=float)
            if not np.isfinite(gradient).all():
                raise FloatingPointError('nonfinite Lindh primitive row')
            for atom in primitive.atoms:
                for axis,value in enumerate(gradient[atom]):
                    if value!=0:columns.append(3*atom+axis);data.append(float(value))
            pointers.append(len(data))
        b=csr_matrix((data,columns,pointers),shape=(len(curvatures),3*n))
    else:
        _values,data,columns,pointers=compiled
        b=csr_matrix((data,columns,pointers),shape=(len(curvatures),3*n))
    pair=coo_matrix((pv,(pr,pc)),shape=(3*n,3*n)).tocsr()
    pair.eliminate_zeros()
    return SimpleLindhOperator(b,np.asarray(curvatures),pair,tuple(families),{
        'model':'SIMPLE_LINDH_XTB_RAW','s6':float(s6),'pair_cutoff_bohr':locality_audit.get('pair_radius_bohr'),
        'exact_all_pair_reference':pair_error_tolerance is None,'pair_count':pair_count,
        'primitive_count':len(curvatures),'primitive_nnz':b.nnz,'pair_nnz':pair.nnz,
        'primitive_derivative_kernel':('CPP_BATCH' if compiled is not None else 'PYTHON_FALLBACK'),
        'gradient_evaluations':0,'dense_cartesian_hessian':False,'anc_conditioning':False,
        'scaling':('quadratic all-pair reference' if pair_error_tolerance is None else
                   'linear local construction at bounded density and fixed tolerance'),
        **locality_audit}, tuple(supports))
