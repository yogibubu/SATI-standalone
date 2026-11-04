"""Internal-tree integration jets retaining an existing transverse basis.

The active path is angular; the transverse coordinates retain their original
definition. A matrix exponential continues the transverse frame without a
rank-changing interpolation. This is a geometric continuation, not an
electronic potential or a new SMITH coordinate selection.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, expm_frechet
from .internal_tree import InternalTreeChart, MovingInternalTreeChart
from .periodic_fields import PeriodicVectorField
from .periodic_continuation import periodic_vector_field_from_dict


@dataclass(frozen=True)
class RetainedBathInternalTreeChart:
    path: MovingInternalTreeChart
    reference_effective_frame: np.ndarray
    active_bath_field: PeriodicVectorField
    frame_log_field: PeriodicVectorField
    bath_curvature_field: PeriodicVectorField

    def __post_init__(self):
        d = len(self.path.active_indices); n = len(self.path.bath_indices)
        frame = np.array(self.reference_effective_frame, float, copy=True)
        if (frame.shape != (n, n) or not np.all(np.isfinite(frame))
                or np.linalg.matrix_rank(frame) != n
                or self.active_bath_field.coefficients.shape[1] != d*n
                or self.frame_log_field.coefficients.shape[1] != n*n
                or self.bath_curvature_field.coefficients.shape[1] != (d+n)*n*(n+1)//2):
            raise ValueError('Complete nonsingular retained transverse frame required')
        frame.flags.writeable = False
        object.__setattr__(self, 'reference_effective_frame', frame)

    def geometry(self, active_displacement, bath_displacement=None):
        if bath_displacement is None: return self.path.geometry(active_displacement)
        theta=np.asarray(active_displacement,float);offset=np.asarray(bath_displacement,float)
        n=len(self.path.bath_indices);d=len(theta)
        if offset.shape != (n,) or not np.all(np.isfinite(offset)): raise ValueError('Finite retained bath offset required')
        jet=self.internal_jet(theta)
        natural=self.path.reference_values.copy();natural[self.path.active_indices]+=theta
        natural[self.path.bath_indices]+=self.path.path_field.evaluate(theta[None])[0]
        natural+=jet['first'][:,d:]@offset+.5*np.einsum('ijk,j,k->i',jet['second'][:,d:,d:],offset,offset)
        return self.path.tree.geometry(natural)

    def internal_jet(self, active_displacement):
        theta = np.asarray(active_displacement, float)
        if theta.shape != (len(self.path.active_indices),): raise ValueError('One active coordinate vector required')
        return {key:value[0] for key,value in self.internal_jets(theta[None]).items()}

    def internal_jets(self, active_displacements):
        q = np.asarray(active_displacements, float)
        a = self.path.active_indices; b = self.path.bath_indices
        d, n = len(a), len(b); nv = d+n; eye = np.eye(d, dtype=int)
        if q.ndim != 2 or q.shape[1] != d or not len(q) or not np.all(np.isfinite(q)): raise ValueError('Nonempty finite active coordinate matrix required')
        N = len(q)
        c1 = np.stack([self.path.path_field.evaluate(q, e) for e in eye], axis=-1)
        c2 = np.stack([np.stack([self.path.path_field.evaluate(q, e+f) for f in eye], axis=-1) for e in eye], axis=-2)
        U = self.active_bath_field.evaluate(q).reshape(N, d, n)
        U1 = np.stack([self.active_bath_field.evaluate(q, e).reshape(N, d, n) for e in eye], axis=-1)
        S = self.frame_log_field.evaluate(q).reshape(N, n, n)
        if np.max(np.linalg.norm(S, 2, axis=(-2,-1)), initial=0.) > 15:
            raise ValueError('Geometric frame continuation exceeds its declared exponential norm bound')
        base = self.reference_effective_frame
        effective = np.stack([base@expm(s) for s in S])
        S1 = np.stack([self.frame_log_field.evaluate(q, e).reshape(N, n, n) for e in eye], axis=-1)
        effective1 = np.stack([np.stack([base@expm_frechet(S[k], S1[k,:,:,j], compute_expm=False) for j in range(d)],axis=-1) for k in range(N)])
        A = np.zeros((N, nv, n)); A[:,a] = U; A[:,b] = c1@U+effective
        A1 = np.zeros((N, nv, n, d)); A1[:,a] = U1
        A1[:,b] = np.einsum('pijc,pjk->pikc', c2, U)+np.einsum('pij,pjkc->pikc', c1, U1)+effective1
        T = np.zeros((N, nv, nv)); T[:,a, np.arange(d)] = 1.; T[:,b, :d] = c1; T[:,:,d:] = A
        Q = np.zeros((N, nv, nv, nv)); Q[:,b, :d, :d] = c2
        Q[:,:,d:, :d] = A1; Q[:,:,:d, d:] = A1.transpose(0, 1, 3, 2)
        tri = np.tril_indices(n)
        curvature = self.bath_curvature_field.evaluate(q).reshape(N, nv, len(tri[0]))
        Q[:,:,d+tri[0], d+tri[1]] = curvature
        Q[:,:,d+tri[1], d+tri[0]] = curvature
        return dict(first=T, second=Q, effective_frame=effective)

    def cartesian_jet(self, active_displacement):
        x = self.geometry(active_displacement); tree = self.path.tree.cartesian_jet(x)
        jet = self.internal_jet(active_displacement); T, Q = jet['first'], jet['second']
        D = tree['tangent']@T
        DD = np.einsum('aij,ik,jl->akl', tree['second'], T, T, optimize=True)+np.einsum('ai,ijk->ajk', tree['tangent'], Q)
        return dict(geometry_A=x, tangent=D, second=DD,
                    internal_first=T, internal_second=Q,
                    effective_frame=jet['effective_frame'])

    def to_dict(self):
        p = self.path
        return dict(schema='matrix.dvr.retained_bath_internal_tree.v1',
            tree=p.tree.to_dict(), reference_values=p.reference_values.tolist(), active_indices=p.active_indices.tolist(),
            path_field=p.path_field.to_dict(), reference_effective_frame=self.reference_effective_frame.tolist(),
            active_bath_field=self.active_bath_field.to_dict(), frame_log_field=self.frame_log_field.to_dict(),
            bath_curvature_field=self.bath_curvature_field.to_dict(),
            scope='Jets at zero retained bath displacement; bath metric dependence remains frozen',
            geometric_continuation_is_not_electronic_validation=True)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.retained_bath_internal_tree.v1': raise ValueError('Unsupported retained-bath chart')
        t = data['tree']; tree = InternalTreeChart(tuple(t['order']), tuple(tuple(r) for r in t['references']))
        field = periodic_vector_field_from_dict
        path = MovingInternalTreeChart(tree, np.array(data['reference_values']), np.array(data['active_indices']), field(data['path_field']))
        return cls(path, np.array(data['reference_effective_frame']), *[field(data[k]) for k in
            ('active_bath_field', 'frame_log_field', 'bath_curvature_field')])
