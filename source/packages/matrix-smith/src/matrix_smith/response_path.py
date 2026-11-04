"""Compose an analytic modal transverse response with an unchanged native chart."""
import numpy as np
from . import build_gic_b_matrix


class AdiabaticResponseChart:
    """Exact first/second Cartesian jets of an analytic response path.

    The response object owns only b(q), db/dq and optionally d2b/dq2.
    Every geometry is solved by the original ORACLE/SMITH chart; no Cartesian
    interpolation or molecular specialization is introduced here.
    """
    def __init__(self,chart,definition,modal_displacement,active_modes,bath_modes,response):
        self.chart=chart;self.definition=definition
        self.S=np.array(modal_displacement,float,copy=True)
        self.active=np.asarray(active_modes,int);self.bath=np.asarray(bath_modes,int)
        self.response=response;self.dimension=len(self.active)
        indices=np.concatenate([self.active,self.bath])
        if (self.S.ndim!=2 or self.S.shape[0]!=definition.rank
                or sorted(indices.tolist())!=list(range(self.S.shape[1]))
                or not 1<=self.dimension<self.S.shape[1] or not np.all(np.isfinite(self.S))):
            raise ValueError('An exhaustive active/bath partition and finite native modal map required')
        self.S.flags.writeable=False

    def modal_jet(self,q,*,second=False):
        point=np.asarray(q,float)
        if point.shape!=(self.dimension,) or not np.all(np.isfinite(point)):
            raise ValueError('A finite active modal point required')
        response=self.response.evaluate(point,second=second)
        n=self.S.shape[1];full=np.zeros(n);J=np.zeros((n,self.dimension))
        full[self.active]=point;full[self.bath]=response['response']
        J[self.active]=np.eye(self.dimension);J[self.bath]=response['response_first']
        H=None
        if second:
            H=np.zeros((n,self.dimension,self.dimension));H[self.bath]=response['response_second']
        return full,J,H

    def cartesian_jet(self,q,*,include_transverse=False,second=False):
        full,J,H=self.modal_jet(q,second=second)
        x=self.chart.geometry_at_displacement_angstrom(self.S@full)
        B=np.asarray(build_gic_b_matrix(self.definition,coordinates_angstrom=x).rows)
        aug=np.vstack([B,self.chart.rigid_constraint_rows])
        M=np.column_stack([J,np.eye(len(full))[:,self.bath]]) if include_transverse else J
        D=np.linalg.solve(aug,np.vstack([self.S@M,np.zeros((6,M.shape[1]))]))
        dd=None
        if second:
            from matrix_zaff.bmatrix_derivative import build_sparse_gic_b_matrix_derivative
            bp=build_sparse_gic_b_matrix_derivative(self.definition,coordinates_angstrom=x)
            if bp.numerical_fallback_primitives:
                raise ValueError('Analytic native chart curvature required')
            n=M.shape[1];rhs=-np.einsum('iac,aj,ck->ijk',bp.to_dense(),D,D,optimize=True)
            rhs[:,:self.dimension,:self.dimension]+=np.einsum('ab,bij->aij',self.S,H)
            dd=np.linalg.solve(aug,np.vstack([rhs.reshape(len(B),n*n),np.zeros((6,n*n))])).reshape(x.shape+(n,n))
        return dict(coordinates_angstrom=x,tangents=D.reshape(x.shape+(M.shape[1],)),second_derivatives=dd)
