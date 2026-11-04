"""Physical R/A/D integration charts using the ORACLE primitive convention.

This is a coordinate map for nuclear integration, not a replacement for a
SMITH selection, symmetry adaptation, or MORPHEUS fitting coordinate basis.
Distances use angstrom and all angles use radians. Torsions are periodic.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InternalTreeChart:
    """Place atoms in order; each row lists earlier bond/angle/dihedral refs.

    Atom indices are zero based. The first three atoms define a fixed frame:
    first at zero, second on +z, third in the positive-x half of the xz plane.
    Collinear angle/reference singularities fail explicitly.
    """
    order: tuple
    references: tuple

    def __post_init__(self):
        n = len(self.order)
        if n < 3 or sorted(self.order) != list(range(n)) or len(self.references) != n:
            raise ValueError('A complete nonlinear atom permutation is required')
        for k, refs in enumerate(self.references):
            if len(refs) != min(k, 3) or len(set(refs)) != len(refs) or any(a not in self.order[:k] for a in refs):
                raise ValueError('Distinct, previously placed reference atoms required')

    @property
    def primitives(self):
        from matrix_chem.primitive_coordinates import Primitive
        result = []
        for atom, refs in zip(self.order, self.references):
            if len(refs) >= 1:
                result.append(Primitive('bond', (refs[0], atom)))
            if len(refs) >= 2:
                result.append(Primitive('angle', (refs[1], refs[0], atom)))
            if len(refs) >= 3:
                result.append(Primitive('dihedral', (refs[2], refs[1], refs[0], atom)))
        return tuple(result)

    def values(self, coordinates):
        from matrix_chem.primitive_coordinates import angle, dihedral
        x = np.asarray(coordinates, float)
        if x.shape != (len(self.order), 3) or not np.all(np.isfinite(x)):
            raise ValueError('Compatible finite Cartesian geometry required')
        result = []
        for p in self.primitives:
            if p.kind == 'bond':
                result.append(np.linalg.norm(x[p.atoms[1]]-x[p.atoms[0]]))
            elif p.kind == 'angle':
                result.append(angle(*p.atoms, x))
            else:
                result.append(dihedral(*p.atoms, x))
        return np.array(result)

    def geometry(self, values):
        q = np.asarray(values, float)
        if q.shape != (3*len(self.order)-6,) or not np.all(np.isfinite(q)):
            raise ValueError('A complete finite internal coordinate vector is required')
        x = np.zeros((len(self.order), 3)); cursor = 0
        for k, (atom, refs) in enumerate(zip(self.order, self.references)):
            if k == 0:
                continue
            r = q[cursor]; cursor += 1
            if r <= 0:
                raise ValueError('Positive distances required')
            if k == 1:
                x[atom] = x[refs[0]] + np.array([0., 0., r])
                continue
            theta = q[cursor]; cursor += 1
            if not 0 < theta < np.pi:
                raise ValueError('Valence angles must lie strictly between zero and pi')
            axis = x[refs[1]]-x[refs[0]]
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                raise ValueError('Degenerate bond reference')
            axis /= norm
            if k == 2:
                trial = np.array([1., 0., 0.])
                perpendicular = trial - axis*np.dot(trial, axis)
                if np.linalg.norm(perpendicular) < 1e-10:
                    raise ValueError('Degenerate initial frame')
                perpendicular /= np.linalg.norm(perpendicular)
            else:
                phi = q[cursor]; cursor += 1
                u = x[refs[2]]-x[refs[1]]
                u -= axis*np.dot(u, axis)
                if np.linalg.norm(u) < 1e-10:
                    raise ValueError('Collinear dihedral references')
                u /= np.linalg.norm(u)
                perpendicular = -np.cos(phi)*u + np.sin(phi)*np.cross(axis, u)
            x[atom] = x[refs[0]] + r*(np.cos(theta)*axis + np.sin(theta)*perpendicular)
        return x

    def coordinate_jet(self, coordinates):
        """Invariant internal-coordinate jets in any Cartesian embedding."""
        from matrix_chem.primitive_coordinates import grad_primitive
        from matrix_smith.models import GICPrimitive
        from matrix_zaff.primitive_second_derivative import analytic_primitive_second_derivative
        x = np.asarray(coordinates, float)
        primitives = self.primitives
        B = np.array([grad_primitive(p, x).reshape(-1) for p in primitives])
        nv, nc = B.shape
        curvature = np.zeros((nv, nc, nc))
        for k, p in enumerate(primitives):
            primitive = GICPrimitive(str(k), str(k), p.kind, p.function, tuple(a+1 for a in p.atoms))
            for a, b, value in analytic_primitive_second_derivative(primitive, x).entries:
                curvature[k, a, b] = value
        return dict(first=B, second=curvature)

    def cartesian_jet(self, coordinates, *, second=True):
        """Exact first/second Cartesian jets from analytic primitive B/B'."""
        x = np.asarray(coordinates, float)
        expected = self.geometry(self.values(x))
        if np.max(abs(expected-x)) > 1e-8:
            raise ValueError('Cartesian jets require this chart fixed embedding')
        if second:
            internal = self.coordinate_jet(x); B = internal['first']; curvature = internal['second']
        else:
            from matrix_chem.primitive_coordinates import grad_primitive
            B=np.array([grad_primitive(p,x).reshape(-1) for p in self.primitives])
        nv, nc = B.shape
        rigid = np.zeros((6, nc))
        indices = [3*self.order[0]+k for k in range(3)] + [3*self.order[1], 3*self.order[1]+1, 3*self.order[2]+1]
        rigid[np.arange(6), indices] = 1.
        augmented = np.vstack([B, rigid])
        D = np.linalg.solve(augmented, np.vstack([np.eye(nv), np.zeros((6, nv))]))
        if not second:return dict(tangent=D,wilson_condition=float(np.linalg.cond(augmented)))
        rhs = -np.einsum('iab,aj,bk->ijk', curvature, D, D, optimize=True)
        DD = np.linalg.solve(augmented, np.vstack([rhs.reshape(nv, -1), np.zeros((6, nv*nv))])).reshape(nc, nv, nv)
        return dict(tangent=D, second=DD, wilson_condition=float(np.linalg.cond(augmented)))

    def to_dict(self):
        return dict(schema='matrix.dvr.internal_tree_chart.v1', atom_indexing='zero_based',
                    order=list(self.order), references=[list(r) for r in self.references],
                    units='angstrom,radian', primitive_owner='ORACLE', curvature_owner='ZAFF',
                    purpose='Physical integration chart; no SMITH selection or MORPHEUS coordinate replacement')


@dataclass(frozen=True)
class MovingInternalTreeChart:
    """A periodic active path plus independent physical transverse offsets."""
    tree: InternalTreeChart
    reference_values: np.ndarray
    active_indices: np.ndarray
    path_field: object

    def __post_init__(self):
        q = np.array(self.reference_values, float, copy=True); active = np.array(self.active_indices, int, copy=True)
        if q.shape != (3*len(self.tree.order)-6,) or active.ndim != 1 or len(set(active)) != len(active) or np.any(active < 0) or np.any(active >= len(q)):
            raise ValueError('Complete physical reference and distinct active indices required')
        if any(self.tree.primitives[int(k)].kind != 'dihedral' for k in active):
            raise ValueError('Periodic active directions must be physical dihedrals')
        if self.path_field.coefficients.shape[1] != len(q)-len(active): raise ValueError('Path field must span the complement')
        q.flags.writeable = False; active.flags.writeable = False
        object.__setattr__(self, 'reference_values', q); object.__setattr__(self, 'active_indices', active)

    @property
    def bath_indices(self):
        return np.array([k for k in range(len(self.reference_values)) if k not in self.active_indices])

    def geometry(self, active_displacement, bath_displacement=None):
        theta = np.asarray(active_displacement, float); d = len(self.active_indices); b = self.bath_indices
        if theta.shape != (d,): raise ValueError('One active displacement required')
        offset = np.zeros(len(b)) if bath_displacement is None else np.asarray(bath_displacement, float)
        if offset.shape != b.shape: raise ValueError('Matching bath displacement required')
        q = self.reference_values.copy(); q[self.active_indices] += theta
        q[b] += self.path_field.evaluate(theta[None])[0]+offset
        return self.tree.geometry(q)

    def cartesian_jet(self, active_displacement, bath_displacement=None):
        theta = np.asarray(active_displacement, float); x = self.geometry(theta, bath_displacement)
        native = self.tree.cartesian_jet(x); d = len(self.active_indices); b = self.bath_indices; nv = d+len(b)
        eye = np.eye(d, dtype=int); first = np.stack([self.path_field.evaluate(theta[None], e)[0] for e in eye], axis=-1)
        second = np.stack([np.stack([self.path_field.evaluate(theta[None], e+f)[0] for f in eye], axis=-1) for e in eye], axis=-2)
        T = np.zeros((nv, nv)); T[self.active_indices, np.arange(d)] = 1.
        T[b, d+np.arange(len(b))] = 1.; T[b, :d] = first
        Q = np.zeros((nv, nv, nv)); Q[b, :d, :d] = second
        D = native['tangent']@T
        DD = np.einsum('aij,ik,jl->akl', native['second'], T, T, optimize=True)+np.einsum('ai,ijk->ajk', native['tangent'], Q)
        return dict(geometry_A=x, tangent=D, second=DD, path_first=first, path_second=second)


@dataclass(frozen=True)
class EckartInternalTreeEmbedding:
    """Realize invariant tree values with supplied frozen Eckart constraints.

    The constraint rows come from the SMITH reference chart. Alignment gives
    the finite geometry; tangent and curvature are solved jointly with those
    same constraints. Rotating a fixed-tree tangent afterwards is insufficient.
    """
    tree: InternalTreeChart
    reference_coordinates: np.ndarray
    masses_amu: np.ndarray
    rigid_constraint_rows: np.ndarray

    def __post_init__(self):
        for name in ('reference_coordinates','masses_amu','rigid_constraint_rows'):
            value=np.array(getattr(self,name),float,copy=True)
            if not np.all(np.isfinite(value)):raise ValueError('Finite Eckart embedding required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        n=len(self.tree.order)
        if (self.reference_coordinates.shape!=(n,3) or self.masses_amu.shape!=(n,)
                or np.any(self.masses_amu<=0) or self.rigid_constraint_rows.shape!=(6,3*n)
                or np.linalg.matrix_rank(self.rigid_constraint_rows)!=6):
            raise ValueError('Complete nonlinear reference and six supplied rigid rows required')

    def geometry(self, values):
        m=self.masses_amu;ref=self.reference_coordinates;center=np.average(ref,axis=0,weights=m)
        fixed=ref-center;x=self.tree.geometry(values);x-=np.average(x,axis=0,weights=m)
        u,s,vt=np.linalg.svd((x*m[:,None]).T@fixed)
        if s[1]<1e-12:raise ValueError('Degenerate mass-weighted Eckart alignment')
        sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vt)
        x=x@(u@sign@vt)+center
        if np.max(abs(self.rigid_constraint_rows@(x-ref).reshape(-1)))>1e-8:
            raise ValueError('Alignment and supplied Eckart constraints disagree')
        return x

    def cartesian_jet(self, values, internal_first=None, internal_second=None):
        x=self.geometry(values);native=self.tree.coordinate_jet(x);B=native['first'];nv,nc=B.shape
        T=np.eye(nv) if internal_first is None else np.asarray(internal_first,float)
        if T.ndim!=2 or T.shape[0]!=nv or not np.all(np.isfinite(T)):raise ValueError('Compatible internal tangent map required')
        d=T.shape[1];Q=np.zeros((nv,d,d)) if internal_second is None else np.asarray(internal_second,float)
        if Q.shape!=(nv,d,d) or not np.all(np.isfinite(Q)):raise ValueError('Compatible internal curvature map required')
        augmented=np.vstack([B,self.rigid_constraint_rows])
        D=np.linalg.solve(augmented,np.vstack([T,np.zeros((6,d))]))
        rhs=Q-np.einsum('iab,aj,bk->ijk',native['second'],D,D,optimize=True)
        DD=np.linalg.solve(augmented,np.vstack([rhs.reshape(nv,-1),np.zeros((6,d*d))])).reshape(nc,d,d)
        return dict(geometry_A=x,tangent=D,second=DD,
            constraint_first_error=float(np.max(abs(self.rigid_constraint_rows@D))),
            constraint_second_error=float(np.max(abs(self.rigid_constraint_rows@DD.reshape(nc,-1)))))
