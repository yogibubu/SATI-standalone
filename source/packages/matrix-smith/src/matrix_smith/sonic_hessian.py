"""SMITH owns only the realization SH = J.T H_ORACLE J on SONIC.

ORACLE supplies the complete xTB initial-Hessian operator and ANC treatment;
SMITH neither selects nor estimates its primitive terms or force constants.
"""
from dataclasses import dataclass
import numpy as np
from matrix_chem.structural_corrections import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class SonicHessian:
    oracle_hessian: object

    @property
    def raw(self):return self.oracle_hessian.raw

    @property
    def b_matrix(self):return self.raw.b_matrix

    @property
    def curvatures(self):return self.raw.curvatures

    @property
    def audit(self):
        return dict(self.oracle_hessian.audit,owner='SMITH',source_owner='ORACLE',
            model=('SH_XTB_LOCAL_ANALYTIC_RIGID_ON_SONIC' if self.oracle_hessian.external_rows is not None
                   else 'SH_XTB_ANC_ON_SONIC'),formula='J.T H_ORACLE J')

    def cartesian_action(self, vector):
        return self.oracle_hessian.cartesian_action(vector)

    def realize(self, forward, transpose):
        """Apply the congruence using J and J.T, without materializing either."""
        return lambda v:np.asarray(transpose(self.cartesian_action(forward(v))))

    def dense(self, cartesian_from_active):
        """Explicit active-space realization for the full route and audit."""
        j=np.asarray(cartesian_from_active,dtype=float)
        source=self.oracle_hessian
        if source.external_rows is not None:
            j=j-source.external_rows.T@(source.external_rows@j)
        projected=source.correction_vectors.T@j
        return self.raw.dense(j)+(source.shift*(j.T@j)+projected.T@(source.correction_values[:,None]*projected))/BOHR_TO_ANGSTROM**2


def build_sonic_hessian(path, *, coordinates_angstrom=None, spectral_budget=128, oracle_hessian=None, pair_error_tolerance=None):
    """Consume an existing ORACLE operator, or request one through its API."""
    if oracle_hessian is None:
        from matrix_oracle import build_initial_hessian
        oracle_hessian=build_initial_hessian(path,coordinates_angstrom=coordinates_angstrom,
            spectral_budget=spectral_budget, pair_error_tolerance=pair_error_tolerance)
    else:
        from matrix_oracle import OracleInitialHessian
        if not isinstance(oracle_hessian,OracleInitialHessian):
            raise TypeError('SMITH requires an ORACLE initial-Hessian operator')
        expected=coordinates_angstrom
        if path is not None:
            from matrix_chem import read_enriched_xyz
            geometry=read_enriched_xyz(path)
            if tuple(geometry.atoms)!=oracle_hessian.atoms:
                raise ValueError('ORACLE Hessian atom identities differ from the SONIC input')
            if expected is None:expected=geometry.coordinates_angstrom
        if expected is not None and not np.array_equal(np.asarray(expected),oracle_hessian.coordinates_angstrom):
            raise ValueError('ORACLE Hessian geometry differs from the SONIC input')
    return SonicHessian(oracle_hessian)
