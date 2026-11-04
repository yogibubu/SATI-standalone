"""Local periodic torsions and exact fixed-torsion internal sections.

Rotor phases are the mass-metric duals of rigid relative fragment rotations,
expressed in geometric internal coordinates. Their selection never uses
normal-mode overlaps. Existing SMITH primitive derivatives and joint
GIC/Eckart inverses provide the nonlinear constrained realizations.
"""
from dataclasses import dataclass,replace
import numpy as np
from scipy.linalg import null_space
from .models import FrozenGIC
from .evaluation import build_gic_b_matrix,build_primitive_b_matrix,evaluate_gic_values
from .numerics import _primitive_value
from .eckart_chart import build_eckart_constrained_gic_chart,eckart_projected_b_matrix
from .frozen_gic_realization import frozen_gic_realization


@dataclass(frozen=True)
class LocalTorsionSection:
    definition: object
    reference: np.ndarray
    masses: np.ndarray
    central_bonds: tuple
    moving_atoms: tuple
    torsion_indices: tuple
    fixed_indices: tuple
    sam_indices: tuple
    torsion_coefficients: np.ndarray

    def realization(self,cache_directory):
        inverse=build_eckart_constrained_gic_chart(self.definition,self.reference,self.masses,tolerance=2e-12)
        b=np.asarray(build_gic_b_matrix(self.definition,coordinates_angstrom=self.reference).rows)
        a=np.linalg.solve(np.vstack((b,inverse.rigid_constraint_rows)),
            np.vstack((np.eye(len(b)),np.zeros((6,len(b))))))
        return frozen_gic_realization(self.definition,self.reference,self.masses,a,cache_directory)

    def torsion_displacement(self,xyz):
        q=evaluate_gic_values(self.definition,coordinates_angstrom=xyz)
        q0=evaluate_gic_values(self.definition,coordinates_angstrom=self.reference)
        return np.asarray(q)[list(self.torsion_indices)]-np.asarray(q0)[list(self.torsion_indices)]

    def rotor_velocity(self,xyz,index=0):
        xyz=np.asarray(xyz,float);left,right=self.central_bonds[index]
        axis=xyz[right-1]-xyz[left-1];axis/=np.linalg.norm(axis)
        velocity=np.zeros_like(xyz);moving=np.array(self.moving_atoms[index],int)-1
        velocity[moving]=np.cross(axis,xyz[moving]-xyz[right-1])
        return velocity.ravel()

    def displacement_kernel(self):
        """Exact local phase and Cartesian two-jet for scalar-field consumers.

        This local phase is used to constrain the SAM section. The global
        periodic rotor potential continues to use the physical rigid-top
        angle and its existing periodic DVR representation.
        """
        return LocalTorsionDisplacement(self)


class LocalTorsionDisplacement:
    """Actual geometric local phases, without sine or inverse-bond mapping.

    The value/Jacobian/Hessian convention matches native scalar kernels:
    (k,), (3N,k), (3N,3N,k), with Cartesian derivatives per angstrom.
    """

    def __init__(self,section):
        self.section=section
        self.definition=section.definition
        self.reference=section.reference

    def value(self,xyz):
        from .second_derivatives import analytic_primitive_hessian
        from matrix_zaff.primitive_second_derivative import analytic_primitive_second_derivative
        xyz=np.asarray(xyz,float)
        if xyz.shape!=self.reference.shape or not np.all(np.isfinite(xyz)):
            raise ValueError('finite geometry matching the local torsion section required')
        bp=np.asarray(build_primitive_b_matrix(self.definition,coordinates_angstrom=xyz).rows)
        primitive=[]
        for p in self.definition.primitives:
            if p.function in ('SOOP','SYCART'):
                h=analytic_primitive_hessian(p,xyz)
            else:
                h=np.zeros((xyz.size,xyz.size))
                for i,j,v in analytic_primitive_second_derivative(p,xyz,
                        reference_coordinates=self.reference).entries:
                    h[i,j]+=v
            primitive.append(h)
        c=self.section.torsion_coefficients
        return (self.section.torsion_displacement(xyz),(c@bp).T,
                np.einsum('kp,pij->ijk',c,np.asarray(primitive)))


def build_local_torsion_section(definition,reference,masses,*,central_bonds=None,fixed_cartesian_covectors=None,rank_tolerance=1e-10):
    """Keep each local torsion exactly fixed during SAM displacements.

    Atom indices in the frozen definition and optional central bonds are one
    based. Automatic bridge rotors use ordinary covalent stretches and proper
    dihedrals; protected reactive contacts never define a rotor terminal bond.
    Rings need explicit constrained local coordinates and are rejected here.
    Additional covectors (e.g. reaction progress) are frozen combinations of
    the existing GICs. SAM normal modes must be solved AFTER this construction.
    """
    x=np.asarray(reference,float);m=np.asarray(masses,float);n=x.size-6
    if (x.ndim!=2 or x.shape[1]!=3 or m.shape!=(len(x),) or np.any(m<=0)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(m))
            or definition.rank!=n or not 0<rank_tolerance<1):
        raise ValueError('finite nonlinear full-rank internal definition and matching masses required')
    ordinary={tuple(sorted(p.atoms)) for p in definition.primitives
        if p.function=='R' and p.family=='STRETCH' and len(p.atoms)==2}
    def usable(p):
        return (p.function=='D' and tuple(sorted(p.atoms[1:3])) in ordinary
            and tuple(sorted(p.atoms[:2])) in ordinary and tuple(sorted(p.atoms[2:])) in ordinary)
    available=sorted({tuple(sorted(p.atoms[1:3])) for p in definition.primitives if usable(p)})
    bonds=available if central_bonds is None else [tuple(sorted(pair)) for pair in central_bonds]
    if not bonds or len(set(bonds))!=len(bonds) or any(b not in available for b in bonds):
        raise ValueError('distinct local torsion bonds with retained covalent dihedrals required')
    primitives=tuple(replace(p,chart='PERIODIC_CONTINUATION',chart_reference_radian=float(_primitive_value(p,x)))
        if p.function=='D' else p for p in definition.primitives)
    base=replace(definition,primitives=primitives,reference_coordinates_angstrom=tuple(map(tuple,x)))
    bp=np.asarray(build_primitive_b_matrix(base,coordinates_angstrom=x).rows)
    ids={p.identifier:i for i,p in enumerate(primitives)}
    coefficients=np.zeros((n,len(primitives)))
    for i,g in enumerate(base.gics):
        for key,value in (g.coefficients or ((g.primitive_id,1.),)):coefficients[i,ids[key]]=value
    tops=[];velocities=[]
    for left,right in bonds:
        adjacency={i:set() for i in range(1,len(x)+1)}
        for a,b in ordinary-{(left,right)}:adjacency[a].add(b);adjacency[b].add(a)
        component={right};front=[right]
        while front:
            atom=front.pop()
            for other in adjacency[atom]-component:component.add(other);front.append(other)
        if left in component:raise ValueError('cyclic torsion requires a constrained ring coordinate')
        axis=x[right-1]-x[left-1];axis/=np.linalg.norm(axis)
        velocity=np.zeros_like(x);moving=np.array(sorted(component))-1
        velocity[moving]=np.cross(axis,x[moving]-x[right-1])
        selected=[i for i,p in enumerate(primitives) if usable(p) and tuple(sorted(p.atoms[1:3]))==(left,right)]
        response=bp[selected]@velocity.ravel()
        if np.any(abs(abs(response)-1)>1e-7):raise ValueError('retained dihedral is not a regular local rotor angle')
        tops.append(tuple(sorted(component)));velocities.append(velocity.ravel())
    inverse=build_eckart_constrained_gic_chart(base,x,m,tolerance=2e-12)
    b=np.asarray(build_gic_b_matrix(base,coordinates_angstrom=x).rows)
    a=np.linalg.solve(np.vstack((b,inverse.rigid_constraint_rows)),np.vstack((np.eye(n),np.zeros((6,n)))))
    # A local rotational mode is the rigid-top rotation in the mass metric.
    # Its conjugate phase is geometric; harmonic eigenvectors never enter.
    # The full inertia matrix retains coupling between multiple local rotors.
    mass=np.repeat(m,3);rigid=inverse.rigid_constraint_rows
    tops_velocity=np.column_stack(velocities)
    mr=rigid.T/mass[:,None]
    tops_velocity-=mr@np.linalg.solve(rigid@mr,rigid@tops_velocity)
    inertia=tops_velocity.T@(mass[:,None]*tops_velocity)
    if np.linalg.eigvalsh(inertia).min()<=rank_tolerance*np.linalg.norm(inertia,2):
        raise ValueError('geometric rotor motions are dependent in the mass metric')
    phase_covectors=np.linalg.solve(inertia,tops_velocity.T*mass[None,:])
    phi=(phase_covectors@a)@coefficients
    extra=np.empty((0,x.size)) if fixed_cartesian_covectors is None else np.asarray(fixed_cartesian_covectors,float)
    if extra.ndim!=2 or extra.shape[1]!=x.size or not np.all(np.isfinite(extra)):
        raise ValueError('additional fixed covectors must have one Cartesian row each')
    rx=(extra@a)@coefficients
    initial_rx=rx@bp
    rx-=(rx@bp@tops_velocity)@phi
    if len(rx) and np.any(np.linalg.norm(rx@bp,axis=1)<=rank_tolerance*np.linalg.norm(initial_rx,axis=1)):
        raise ValueError('local torsion/reaction constraints are dependent')
    protected=np.vstack((rx,phi));c=protected@bp
    # Gradient normals determine the tangent section; no harmonic eigensystem
    # or frequency cutoff participates in this geometric partition.
    projected=eckart_projected_b_matrix(b,m,inverse.rigid_constraint_rows)
    cp=eckart_projected_b_matrix(c,m,inverse.rigid_constraint_rows)
    normals=(projected/np.repeat(m,3)[None,:])@cp.T
    if np.linalg.matrix_rank(normals,tol=rank_tolerance*np.linalg.norm(normals,2))!=len(c):
        raise ValueError('local torsion/reaction constraints are dependent')
    sam=null_space(normals.T,rcond=rank_tolerance).T@coefficients
    all_coeff=np.vstack((rx,phi,sam))
    def gic(i,row):
        r=len(rx);k=len(phi)
        family='LOCAL_FIXED' if i<r else ('LOCAL_TORSION' if i<r+k else 'LOCAL_SAM')
        return FrozenGIC(f'LOCAL{i+1:04d}',f'{family}_{i+1}',family,'',primitives[0].identifier,'NONE',
            tuple((p.identifier,float(v)) for p,v in zip(primitives,row) if v!=0.))
    result=replace(base,backend='matrix-smith-local-torsion-section.v1',symmetrize=False,
        gics=tuple(gic(i,row) for i,row in enumerate(all_coeff)),symmetry_diagnostics=None,reduction_diagnostics=None,
        periodic_coordinate_estimates=(),semantic_diagnostics=(*base.semantic_diagnostics,
            'LOCAL_TORSIONS_RIGID_TOP_KINETIC_DUAL_PHASE',
            'SAM_NORMAL_MODES_REQUIRE_CONDITIONAL_GF'))
    rb=np.asarray(build_gic_b_matrix(result,coordinates_angstrom=x).rows)
    if np.linalg.matrix_rank(np.vstack((rb,inverse.rigid_constraint_rows)))!=x.size:
        raise ValueError('local torsion section did not retain the full vibrational rank')
    section=LocalTorsionSection(result,x.copy(),m.copy(),tuple(bonds),tuple(tops),
        tuple(range(len(rx),len(rx)+len(phi))),tuple(range(len(rx))),tuple(range(len(rx)+len(phi),n)),phi)
    # Check the finite geometric rotor, not just its linearized tangent.
    q0=np.asarray(evaluate_gic_values(result,coordinates_angstrom=x))
    for k,(left,right) in enumerate(bonds):
        axis=x[right-1]-x[left-1];axis/=np.linalg.norm(axis)
        moving=np.array(tops[k])-1;delta=x[moving]-x[right-1]
        for angle in (-.1,.1):
            y=x.copy();y[moving]=x[right-1]+delta*np.cos(angle)+np.cross(axis,delta)*np.sin(angle)+(delta@axis)[:,None]*axis*(1-np.cos(angle))
            change=np.asarray(evaluate_gic_values(result,coordinates_angstrom=y))-q0
            expected=np.zeros(n);expected[section.torsion_indices[k]]=angle
            if np.max(abs(change-expected))>1e-8:
                raise ValueError('frozen primitives do not support a pure finite local rotor; use a valence internal chart')
    return section
