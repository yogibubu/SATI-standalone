"""ORACLE-owned xTB initial-Hessian construction and ANC conditioning.

The raw xTB pair model is not linear scaling. ANC spectral corrections are
represented explicitly as a scalar shift plus low-rank updates; the Cartesian
Hessian is never assembled densely. A spectral budget failure is explicit.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from matrix_chem import read_enriched_xyz
from matrix_chem.structural_corrections import BOHR_TO_ANGSTROM
from matrix_chem.lindh_hessian import _condition_anc_eigenvalues
from matrix_chem.vibrational import is_linear_geometry
from matrix_chem.lindh_operator import simple_lindh_operator


@dataclass(frozen=True)
class OracleInitialHessian:
    raw: object
    shift: float
    correction_vectors: np.ndarray
    correction_values: np.ndarray
    audit: dict
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    external_rows: np.ndarray | None = None

    @property
    def b_matrix(self):return self.raw.b_matrix

    @property
    def curvatures(self):return self.raw.curvatures

    def simple_primitives(self):
        """Frozen measurements selected by ddvopt, including all pair distances.

        These describe the coordinate pool, not additional Hessian terms.
        D2 and ANC corrections remain in the Cartesian model operator.
        """
        from matrix_chem.primitive_coordinates import Primitive
        kinds = {'R': 'bond', 'A': 'angle', 'L': 'linear_bend', 'D': 'dihedral'}
        linear_modes = {}
        result = []
        for family, atoms in zip(self.raw.families, self.raw.primitive_supports, strict=True):
            mode = 0
            if family == 'L':
                component = linear_modes.get(atoms, 0) + 1
                linear_modes[atoms] = component
                mode = -component  # ORACLE local linear-bend convention: -1, -2
            result.append(Primitive(kinds[family], atoms, mode=mode))
        return tuple(result)

    def cartesian_action(self, vector):
        v=np.asarray(vector,dtype=float)
        if self.external_rows is not None:
            v=v-self.external_rows.T@(self.external_rows@v)
        correction=self.shift*v+self.correction_vectors@(self.correction_values*(self.correction_vectors.T@v))
        result=self.raw.cartesian_action(v)+correction/BOHR_TO_ANGSTROM**2
        if self.external_rows is not None:
            result=result-self.external_rows.T@(self.external_rows@result)
        return result


def build_initial_hessian(path, *, coordinates_angstrom=None, spectral_budget=128, pair_error_tolerance=None):
    geometry=read_enriched_xyz(path)
    xyz=np.asarray(geometry.coordinates_angstrom if coordinates_angstrom is None else coordinates_angstrom,dtype=float)
    raw=simple_lindh_operator(geometry.atoms,xyz, pair_error_tolerance=pair_error_tolerance)
    n=xyz.size
    if n<6:raise ValueError('xTB ANC SH requires at least two atoms')
    if spectral_budget<6:raise ValueError('SH spectral budget must be at least six')
    if pair_error_tolerance is not None:
        from .local_anc import condition_local_model
        return condition_local_model(raw, geometry.atoms, xyz, spectral_budget=spectral_budget)
    action=lambda v:raw.cartesian_action(v)*BOHR_TO_ANGSTROM**2
    translations=np.tile(np.eye(3),(len(xyz),1))/np.sqrt(len(xyz))
    # A single-vector Lanczos iteration can miss multiplicity in the exactly
    # degenerate translational null space. Include it analytically and solve
    # only the orthogonal part; never infer that three translations were found.
    absolute_b=abs(raw.b_matrix)
    bound=float(np.max(absolute_b.T@(raw.curvatures*(absolute_b@np.ones(n)))+abs(raw.pair_hessian)@np.ones(n)))+1.
    low_action=lambda v:action(v)+bound*(translations@(translations.T@v))
    rng=np.random.default_rng(98231)
    def spectrum(which,edge):
        selected_action=low_action if which=='SA' else action
        operator=LinearOperator((n,n),matvec=selected_action,dtype=float)
        k=min(12,n-1,spectral_budget)
        while True:
            w,v=eigsh(operator,k=k,which=which,tol=1.e-12,
                v0=rng.normal(size=n),maxiter=(128 if pair_error_tolerance is not None else max(1000,50*n)),
                ncv=min(n,max(2*k+1,96 if pair_error_tolerance is not None else 24)))
            order=np.argsort(w);w=w[order];v=v[:,order]
            residual=max(np.linalg.norm(selected_action(v[:,i])-w[i]*v[:,i]) for i in range(k))
            if residual>1.e-8:raise FloatingPointError('SH spectral residual failed')
            if k==n-1:
                index=int(np.argmax(1-np.sum(v*v,axis=1)))
                q=np.eye(1,n,index).ravel()-v@v[index,:]
                q/=np.linalg.norm(q)
                value=float(q@selected_action(q))
                if np.linalg.norm(selected_action(q)-value*q)>1.e-8:
                    raise FloatingPointError('SH spectral complement failed')
                w=np.r_[w,value];v=np.column_stack((v,q));order=np.argsort(w)
                return w[order],v[:,order]
            if (which=='SA' and w[-1]>edge) or (which=='LA' and w[0]<edge):
                return w,v
            if k>=spectral_budget:
                raise RuntimeError('SH ANC spectral budget exceeded; no different Hessian substituted')
            k=min(2*k,n-1,spectral_budget)
    low,lv=spectrum('SA',.05)
    physical=np.linalg.norm(translations.T@lv,axis=0)<.5
    low=np.r_[np.zeros(3),low[physical]]
    lv=np.column_stack((translations,lv[:,physical]))
    order=np.argsort(low);low=low[order];lv=lv[:,order]
    conditioned,nulled,shift=_condition_anc_eigenvalues(low,lv,xyz/BOHR_TO_ANGSTROM,
        linear=is_linear_geometry(xyz),hlow=.01)
    # xTB shifts only eigenvalues retained by its numerical zero threshold.
    excluded=np.flatnonzero(np.abs(conditioned)<=1.e-10)
    vectors=[lv[:,i] for i in excluded]
    values=[-low[i]-shift for i in excluded]
    if len(low)==n:
        high,hv=low,lv
    else:
        high,hv=spectrum('LA',5.-shift)
    for i,value in enumerate(high):
        if value+shift>5.:
            vectors.append(hv[:,i]);values.append(5.-value-shift)
    u=np.column_stack(vectors) if vectors else np.empty((n,0))
    frozen_xyz=xyz.copy()
    frozen_xyz.setflags(write=False)
    return OracleInitialHessian(raw,float(shift),u,np.asarray(values),dict(raw.audit,
        owner='ORACLE',model='XTB_ANC_INITIAL_HESSIAN',formula='H_xTB_ANC',anc_conditioning=True,
        damping_shift=float(shift),spectral_correction_rank=len(values),
        spectral_budget=spectral_budget,dense_cartesian_hessian=False,
        raw_model_scaling=raw.audit['scaling'],
        conditioned_model_scaling='NOT_CERTIFIED_LINEAR_SCALING',
        anc_candidate_scoring='EXACT_ALL_PAIR_DISTANCE_CHANGE',
        anc_candidate_scoring_scaling='QUADRATIC_PER_SPECTRAL_CANDIDATE',
        conditioned_error_certified=False if pair_error_tolerance is not None else None,
        bounded_spectral_iterations=128 if pair_error_tolerance is not None else None),tuple(geometry.atoms),frozen_xyz)


def build_simple_hessian(path, *, coordinates_angstrom=None, s6=20.):
    """Build the unconditioned xTB simple-primitive model, before any SONIC."""
    geometry=read_enriched_xyz(path)
    xyz=geometry.coordinates_angstrom if coordinates_angstrom is None else coordinates_angstrom
    return simple_lindh_operator(geometry.atoms,xyz,s6=s6)
