"""Analytic molecular NN-xTB Hessian on the resident GFN2 stationary state.

All electronic responses are solved with the existing CPSCF kernel.  The
geometry/parameter and parameter/parameter blocks differentiate the stationary
head adjoints, including moving-basis Pulay terms and CN radius response.
"""

from dataclasses import dataclass

import numpy as np

from .gradients import coordination_number_directional_derivative_sparse
from .ground_state_response import solve_ground_state_directional_response
from .multipole import (
    AtomicMultipoles, dense_scc_observables,
    cumulative_atomic_multipole_directional_derivative_sparse,
)
from .nn_xtb import _MATRIX_SHIFT_FACTORS
from .nn_xtb_parameters import (
    build_nn_parameter_state, analytic_h0_head_adjoints,
    analytic_h0_head_adjoint_directional, analytic_aes_head_adjoints,
    analytic_aes_head_adjoint_directional, analytic_scc_head_adjoint_directional,
    analytic_repulsion_head_adjoint_directional,
    analytic_rcov_coordination_jacobian, analytic_rcov_head_adjoint_directional,
)
from .dispersion import d4_global_head_adjoint_direction_dense


_REP = {"atom.repa", "atom.repb", "global.kexp", "global.klight"}
_H0 = {"shell.lev", "shell.kcn", "shell.poly", "shell.exp", "shell.h0exp", "atom.arad",
       "atom.en", "global.ks", "global.kp", "global.kd", "global.ksd",
       "global.kpd", "global.enscale", "global.wexp"}
_SCC = {"atom.gam", "atom.gam3", "shell.lpar", "shell.third", "global.alphaj"}
_AES = {"atom.dpol", "atom.qpol", "atom.mprad", "atom.mpvcn", "global.aesdmp3",
        "global.aesdmp5", "global.aesexp", "global.aesshift", "global.aesrmax"}
_D4 = {"global.a1", "global.a2", "global.s6", "global.s8", "global.s9", "global.s10"}


@dataclass(frozen=True)
class NNXTBAnalyticHessian:
    hessian_hartree_per_bohr2: np.ndarray
    fixed_parameter_hartree_per_bohr2: np.ndarray
    mixed_parameter_hartree_per_bohr2: np.ndarray
    parameter_response_hartree_per_bohr2: np.ndarray
    network_curvature_hartree_per_bohr2: np.ndarray
    reciprocity_error_hartree_per_bohr2: float
    translational_residual_hartree_per_bohr2: float
    maximum_cpscf_residual: float
    fixed_parameter_reciprocity_error_hartree_per_bohr2: float

    @property
    def parameter_geometry_hartree_per_bohr2(self):
        """Symmetrized E_Rp J, for the legacy four-block result interface."""
        return 0.5 * self.mixed_parameter_hartree_per_bohr2

    @property
    def head_adjoint_response_hartree_per_bohr2(self):
        return (0.5 * self.mixed_parameter_hartree_per_bohr2
                + self.parameter_response_hartree_per_bohr2)

    @property
    def chain_rule_error_hartree_per_bohr2(self):
        """Block-sum rounding residual; numerical audit error is not implied."""
        assembled = (self.fixed_parameter_hartree_per_bohr2
                     + self.mixed_parameter_hartree_per_bohr2
                     + self.parameter_response_hartree_per_bohr2
                     + self.network_curvature_hartree_per_bohr2)
        return float(np.max(np.abs(self.hessian_hartree_per_bohr2 - assembled)))


class NNAnalyticResponse:
    """One retained stationary reference and its analytic head responses."""

    def __init__(self, calculator, coordinates_angstrom):
        from .calculator import ANGSTROM_TO_BOHR, _covalent_radii

        self.calculator = calculator
        self.coordinates = np.asarray(coordinates_angstrom, dtype=float)
        self.positions = self.coordinates * ANGSTROM_TO_BOHR
        self.n = len(calculator.atomic_numbers)
        if (self.coordinates.shape != (self.n, 3)
                or not np.all(np.isfinite(self.coordinates))):
            raise ValueError("NN Hessian coordinates must be finite natoms x 3")
        self.shifts = calculator.nn_overlay.matrix_parameter_shifts(
            calculator.atomic_numbers, self.positions,
            charge=float(calculator.charge), unpaired=float(calculator.multiplicity - 1),
        )
        self.fixed, self.worker = calculator._nn_fixed_parameter_point(
            self.positions, self.shifts, retain_orbitals=True, return_worker=True,
        )
        self.orbitals = self.fixed.orbital_state
        if not self.fixed.converged or self.orbitals is None:
            raise RuntimeError("analytic NN Hessian requires a converged retained state")
        self.parameters = build_nn_parameter_state(calculator.atomic_numbers, calculator.parameters, self.shifts)
        self.context = self.worker._prepare_density_response_gradient_context(self.orbitals)
        self.kernel = self.context["kernel"]
        self.basis = self.context["basis"]
        self.partition = self.context["partition"]
        self.derivatives = self.worker.ground_state_derivative_matrices(self.orbitals)
        self.shell_labels = [None] * self.partition.nshells
        for orbital, shell in enumerate(self.partition.orbital_shell):
            self.shell_labels[shell] = self.basis.spherical_shell[orbital]
        self.moments = self.kernel.reference_multipoles
        self.cn = self.worker._active_coordination
        self.h0 = analytic_h0_head_adjoints(
            calculator.atomic_numbers, self.positions, self.parameters,
            self.orbitals.overlap, self.fixed.density,
            self.orbitals.ao_atom, self.orbitals.ao_shell,
        )
        self.aes = analytic_aes_head_adjoints(self.positions, self.cn, self.parameters, self.moments)
        self.radius_jacobian, _ = analytic_rcov_coordination_jacobian(self.positions, self.parameters)
        self.d4_radii = _covalent_radii()[np.asarray(calculator.atomic_numbers) - 1]
        self.pair = self.kernel._d4(self.moments.charges)
        self.maximum_residual = 0.0
        self.cutoff = max(24.0, float(np.max(np.linalg.norm(
            self.positions[:, None] - self.positions[None, :], axis=2,
        ))) + 1.0)

    def head_response(self, *, coordinate_direction=None, parameter_direction=None):
        """Return E_pR dR or E_pp dp, including full first-order CPSCF."""
        calc = self.calculator
        geometry = coordinate_direction is not None
        dx = np.asarray(coordinate_direction) if geometry else np.zeros_like(self.positions)
        heads = {} if parameter_direction is None else parameter_direction
        if geometry and heads:
            raise ValueError("request one response block at a time")
        if geometry:
            ds = np.einsum("ab,abij->ij", dx, self.derivatives.overlap_per_bohr)
            df = np.einsum("ab,abij->ij", dx, self.derivatives.fock_hartree_per_bohr)
            response = solve_ground_state_directional_response(self.orbitals, ds, df, self.kernel)
            operator_moments = cumulative_atomic_multipole_directional_derivative_sparse(
                self.context["reference_density"], self.context["integrals"],
                self.positions, dx, self.basis.spherical_atom,
            )
            cn_dot = coordination_number_directional_derivative_sparse(
                self.positions, dx,
                np.asarray([a.covalent_radius_bohr for a in self.parameters.atoms]),
                cutoff_bohr=self.cutoff,
            )
            d4_cn_dot = coordination_number_directional_derivative_sparse(
                self.positions, dx, self.d4_radii,
                electronegativities=self.worker._electronegativities,
                cutoff_bohr=self.cutoff,
            )
        else:
            electronic = {k: v for k, v in heads.items() if k not in _REP}
            # A purely repulsive direction has an exactly zero electronic source.
            if not electronic:
                electronic = {"shell.lev": np.zeros((self.n, 4))}
            response, ds, operator_moments = calc.evaluate_nn_parameter_cpscf(
                self.coordinates, electronic,
                return_overlap_direction=True, return_observable_direction=True,
            )
            cn_dot = self.radius_jacobian @ np.asarray(heads.get("atom.rcov", np.zeros((self.n, 1))))[:, 0]
            d4_cn_dot = np.zeros(self.n)
        self.maximum_residual = max(self.maximum_residual, response.residual_norm)
        qdensity, mdensity = dense_scc_observables(
            response.density_derivative, self.kernel.operators,
            self.partition.orbital_shell, np.zeros(self.partition.nshells), np.zeros(self.n),
        )
        qdot = qdensity - np.bincount(
            self.partition.orbital_shell, weights=np.sum(self.fixed.density * ds, axis=1),
            minlength=self.partition.nshells,
        )
        if operator_moments is None:
            operator_moments = AtomicMultipoles(np.zeros(self.n), np.zeros((self.n, 3)), np.zeros((self.n, 3, 3)))
        mdot = AtomicMultipoles(
            mdensity.charges + operator_moments.charges,
            mdensity.dipoles_bohr + operator_moments.dipoles_bohr,
            mdensity.quadrupoles_bohr2 + operator_moments.quadrupoles_bohr2,
        )
        subset = lambda names: {k: v for k, v in heads.items() if k in names}
        result = analytic_repulsion_head_adjoint_directional(
            calc.atomic_numbers, self.positions, self.parameters, dx, subset(_REP),
        )
        h0 = analytic_h0_head_adjoint_directional(
            calc.atomic_numbers, self.positions, self.parameters,
            self.orbitals.overlap, ds, self.fixed.density, response.density_derivative,
            self.orbitals.ao_atom, self.orbitals.ao_shell, subset(_H0),
            coordinate_direction=dx,
            coordination_direction=None if geometry else cn_dot,
        )
        result.update(h0)
        result.update(analytic_scc_head_adjoint_directional(
            self.positions, self.parameters, self.kernel.reference_shell_charges, qdot,
            self.partition.shell_atom, self.shell_labels, subset(_SCC), coordinate_direction=dx,
        ))
        aes = analytic_aes_head_adjoint_directional(
            self.positions, self.cn, self.parameters, self.moments, mdot, subset(_AES),
            coordinate_direction=dx, coordination_direction=cn_dot,
        )
        result.update(aes)
        result.update({f"global.{k}": np.array([[v]]) for k, v in d4_global_head_adjoint_direction_dense(
            self.positions, self.moments.charges, mdot.charges, self.kernel.d4_coordination,
            self.worker._d4_models, self.worker._d4_parameters, self.pair, self.kernel.d4_prepared,
            {k[7:]: float(v[0, 0]) for k, v in subset(_D4).items()},
            coordinate_direction=dx, coordination_direction=d4_cn_dot,
        ).items()})
        if geometry and "shell.exp" in self.shifts:
            result["shell.exp"] = calc.evaluate_nn_exponent_head_geometry_response(self.coordinates, dx)
        elif "shell.exp" in self.shifts:
            result["shell.exp"] = calc.evaluate_nn_exponent_head_adjoint_response(self.coordinates, electronic)
        result["atom.rcov"] = analytic_rcov_head_adjoint_directional(
            self.positions, self.parameters, self.h0["shell.lev"], self.aes["atom.mpvcn"],
            h0["shell.lev"], aes["atom.mpvcn"], heads, coordinate_direction=dx,
        )
        missing = self.shifts.keys() - result.keys()
        if missing:
            raise NotImplementedError(f"analytic NN Hessian heads missing: {sorted(missing)}")
        return result


def analytic_nn_hessian(calculator, coordinates_angstrom) -> NNXTBAnalyticHessian:
    """Assemble E_RR + E_Rp J + J.T E_pR + J.T E_pp J + E_p p_RR."""
    if not calculator.nn_enabled or calculator.nn_overlay is None:
        raise ValueError("analytic NN Hessian requires nn_enabled=True")
    context = NNAnalyticResponse(calculator, coordinates_angstrom)
    fixed = context.worker.ground_state_hessian(context.orbitals)
    if fixed.reciprocity_error_hartree_per_bohr2 > 2.0e-6:
        raise ArithmeticError(
            "fixed-parameter NN Hessian violates reciprocity: "
            f"{fixed.reciprocity_error_hartree_per_bohr2:.3e}")
    _, adjoints, _ = calculator._nn_parameter_response_state(context.coordinates)
    names = tuple(context.shifts)
    flatten = lambda values: np.concatenate([np.asarray(values[name]).ravel() for name in names])
    dimension = context.positions.size
    jacobian = np.empty((sum(context.shifts[name].size for name in names), dimension))
    mixed = np.empty_like(jacobian)
    parameter = np.empty_like(jacobian)
    curvature = np.empty((dimension, dimension))
    official_curvature = None
    if hasattr(calculator.nn_overlay, "parameter_shift_curvature_matrix"):
        official_curvature = calculator.nn_overlay.parameter_shift_curvature_matrix(
            calculator.atomic_numbers, context.positions, adjoints,
            charge=float(calculator.charge), unpaired=float(calculator.multiplicity - 1),
        )
    for column in range(dimension):
        direction = np.zeros_like(context.positions)
        direction.flat[column] = 1.0
        raw = calculator.nn_overlay.parameter_shift_jvp(
            calculator.atomic_numbers, context.positions, direction,
            charge=float(calculator.charge), unpaired=float(calculator.multiplicity - 1),
        )
        dp = {name: value * _MATRIX_SHIFT_FACTORS.get(name, 1.0) for name, value in raw.items()}
        jacobian[:, column] = flatten(dp)
        mixed[:, column] = flatten(context.head_response(coordinate_direction=direction))
        parameter[:, column] = flatten(context.head_response(parameter_direction=dp))
        if official_curvature is None:
            _, curve = calculator.nn_overlay.parameter_shift_vjp(
                calculator.atomic_numbers, context.positions, adjoints,
                charge=float(calculator.charge), unpaired=float(calculator.multiplicity - 1),
                hessian_direction=direction,
            )
            curvature[:, column] = curve.ravel()
        else:
            curvature[:, column] = official_curvature[:, column]
    cross = jacobian.T @ mixed
    cross = cross + cross.T
    parameter_block = jacobian.T @ parameter
    raw_hessian = fixed.hessian_hartree_per_bohr2 + cross + parameter_block + curvature
    if not np.all(np.isfinite(raw_hessian)):
        raise ArithmeticError("analytic NN Hessian contains nonfinite entries")
    reciprocity = float(np.max(np.abs(raw_hessian - raw_hessian.T)))
    if reciprocity > 2.0e-6:
        raise ArithmeticError(f"analytic NN Hessian violates reciprocity: {reciprocity:.3e}")
    symmetric = lambda x: 0.5 * (x + x.T)
    hessian = symmetric(raw_hessian)
    translation = float(np.max(np.abs(hessian.reshape(context.n, 3, context.n, 3).sum(axis=2))))
    return NNXTBAnalyticHessian(
        hessian, fixed.hessian_hartree_per_bohr2, cross,
        symmetric(parameter_block), symmetric(curvature), reciprocity, translation,
        max(context.maximum_residual, fixed.maximum_cpscf_residual),
        fixed.reciprocity_error_hartree_per_bohr2,
    )
