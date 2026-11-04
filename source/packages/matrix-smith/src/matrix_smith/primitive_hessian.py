"""SH on frozen ORACLE measurements, independent of the selected SONIC chart.

The model energy is 1/2 dp.T K dp. Realization in any admissible chart uses
dp = B J dq, not C K C.T or the unconstrained compliance C K^-1 C.T.
Only the diagonal K and the sparse primitive Jacobian are stored.

These local SH constants are provisional approximations to ZAFF curvatures
on the same ORACLE measurements, not a separately fitted force field. A
future ZAFF provider must use the same units, supports and redundancy weights.
The congruence is an initial harmonic model; it is not the exact nonlinear
ZAFF Hessian away from a stationary point (which also contains gradient terms).
"""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg

from .primitive_handoff import measurement_key
from .oracle_contract_consumer import load_validated_oracle_candidate_pool
from .contracts import GICForgeContractError
from .evaluation import compiled_gic_primitive_csr
from .sonic_force_constant_atlas import (
    DEFAULT_SH_ATLAS, sonic_primitive_environments_from_xyzin,
)


def primitive_model_family(primitive, covalent_bonds):
    """Chemical typing belongs to the measurement, not its SONIC label."""
    function = primitive.function
    if function == "R":
        if primitive.family == "FULLERENE_FACE_DIAGONAL":
            return primitive.family
        if tuple(sorted(primitive.atoms)) in covalent_bonds:
            return "STRETCH"
        return "HBOND_DISTANCE" if primitive.family == "HBOND_DISTANCE" else "PSEUDO_BOND_DISTANCE"
    if function == "D":
        return ("TORSION" if all(tuple(sorted(pair)) in covalent_bonds
                for pair in zip(primitive.atoms[:-1], primitive.atoms[1:]))
                else "PSEUDO_BOND_TORSION")
    if function == "A":
        return ("BEND" if all(tuple(sorted(pair)) in covalent_bonds
                for pair in zip(primitive.atoms[:-1], primitive.atoms[1:]))
                else "PSEUDO_BOND_BEND")
    families = {"A": "BEND", "L": "LINEAR_BEND",
                "IMPD": "IMPROPER_DIHEDRAL", "U": "OUT_OF_PLANE",
                # SOOP is the canonical selective out-of-plane primitive used
                # for cyclic puckering and related local distortions.
                "SOOP": "OUT_OF_PLANE",
                # H is an elementary local height.  Ring puckering is a
                # SMITH combination of such rows and must never receive an
                # independent SH parameter.
                "H": "OUT_OF_PLANE_HEIGHT", "FC_DIST": "FRAG_DISTANCE",
                "FCA_DIST": "FRAG_CENTER_ATOM_DISTANCE", "CENTER_ATOM_DIST": "CENTER_ATOM_DISTANCE",
                "FTRANS": "FRAG_TRANSLATION", "FLIN_TRANS": "FRAG_TRANSLATION",
                "FROT": "FRAG_ORIENTATION", "FAXIS": "FRAG_ORIENTATION"}
    if function in {"FACE_FLAP", "FACE_HINGE", "CENTER_FACE_DIST", "FACE_DIRECTION_ANGLE", "CONTACT_CENTER_ANGLE"}:
        return function
    if function not in families:
        raise ValueError(f"SH needs an elementary ORACLE measurement, got {function}")
    return families[function]


def normalized_primitive_curvatures(primitives, curvatures):
    """Split a physical local stiffness budget, never add it per duplicate.

    Exact functional aliases share their weight. Torsions share one central
    bond budget (already containing ni*nj); equivalent plane gauges share one
    local-plane budget. This does not assert invariance under arbitrary new
    nonlinear measurements or remove global geometric dependencies.
    """
    raw = np.asarray(curvatures, dtype=float)
    if not len(primitives) or raw.shape != (len(primitives),) or not np.all(np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("SH primitive curvatures must be finite and positive")
    aliases, budgets, key_budgets = {}, {}, {}
    for i, p in enumerate(primitives):
        key = measurement_key(p)
        aliases.setdefault(key, []).append(i)
        if p.function == "D":
            budget = ("CENTRAL_TORSION", p.family, tuple(sorted(p.atoms[1:3])))
        elif p.function == "U":
            budget = ("LOCAL_PLANE", p.atoms[0], tuple(sorted(p.atoms[1:])))
        else:
            budget = ("MEASUREMENT", key)
        if key in key_budgets and key_budgets[key] != budget:
            raise ValueError("conflicting SH family budgets for the same ORACLE measurement")
        key_budgets[key] = budget
        budgets.setdefault(budget, set()).add(key)
    result = np.empty_like(raw)
    for keys in budgets.values():
        for key in keys:
            indices = aliases[key]
            values = raw[indices]
            if not np.allclose(values, values[0], rtol=1.e-12, atol=0.):
                raise ValueError("conflicting SH curvature assignments for the same ORACLE measurement")
            result[indices] = values[0] / (len(keys) * len(indices))
    return result, {"raw_count": len(primitives), "unique_measurements": len(aliases),
                    "duplicate_count": len(primitives)-len(aliases), "local_budgets": len(budgets),
                    "rule": "EXACT_ALIAS_SHARING_AND_LOCAL_TORSION_PLANE_BUDGETS",
                    "arbitrary_redundant_reparametrization_invariant": False}


@dataclass(frozen=True)
class PrimitiveSH:
    primitives: tuple
    b_matrix: object
    curvatures: np.ndarray
    audit: dict

    def cartesian_action(self, displacement):
        vector = np.asarray(displacement, dtype=float).reshape(-1)
        if vector.shape != (self.b_matrix.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("invalid SH Cartesian displacement")
        return np.asarray(self.b_matrix.T @ (self.curvatures * (self.b_matrix @ vector))).ravel()

    def realize(self, forward, transpose):
        """J and J.T use angstrom per active-coordinate unit."""
        def action(vector):
            return np.asarray(transpose(self.cartesian_action(forward(vector))), dtype=float).ravel()
        return action

    def dense(self, cartesian_from_active):
        a = self.b_matrix @ np.asarray(cartesian_from_active, dtype=float)
        return a.T @ (self.curvatures[:, None] * a)


def build_primitive_sh(
    path,
    *,
    coordinates_angstrom=None,
    parameter_multipliers=None,
    simple_optimizer_model=False,
):
    """Build SH from ORACLE, with a read-only legacy-definition fallback.

    Production inputs always use the frozen ORACLE contract.  The fallback is
    deliberately limited to old serialized SONIC definitions: it retains the
    same compiled sparse primitive kernel but cannot apply topology-dependent
    SH typing because those historical files did not preserve SYNTHONS.
    """

    legacy_fallback = False
    try:
        pool = load_validated_oracle_candidate_pool(path)
        reference = np.asarray(pool.coordinates_angstrom, dtype=float)
        atom_symbols = pool.atom_symbols
        source_primitives = pool.candidates
        bonds = set(pool.contract.primary_topology.bonds)
    except GICForgeContractError as contract_error:
        from matrix_chem import read_enriched_xyz
        if not Path(path).is_file():
            raise contract_error
        # Resolve through the public package facade so legacy applications can
        # supply an already-frozen definition without inventing a #GIC block.
        from matrix_smith import read_gic_definition_from_xyzin

        geometry = read_enriched_xyz(path)
        definition = read_gic_definition_from_xyzin(path)
        reference = np.asarray(geometry.coordinates_angstrom, dtype=float)
        atom_symbols = tuple(geometry.atoms)
        source_primitives = tuple(definition.primitives)
        bonds = {
            tuple(sorted(primitive.atoms))
            for primitive in source_primitives
            if primitive.function == "R" and len(primitive.atoms) == 2
        }
        legacy_fallback = True
    coords = reference if coordinates_angstrom is None else np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != reference.shape or not np.isfinite(coords).all():
        raise ValueError("invalid primitive SH geometry")
    primitives = tuple(replace(p, family=primitive_model_family(p, bonds)) for p in source_primitives)
    if simple_optimizer_model:
        # The xTB optimization model contains distances (including frozen
        # pseudo-bonds), valence/linear bends and proper torsions.  Richer
        # ORACLE families remain available to SONIC, but must not be counted a
        # second time in the Cartesian SYCART seed.
        primitives = tuple(p for p in primitives if p.function in {"R", "A", "L", "D"})
    definition = SimpleNamespace(primitives=primitives)
    if legacy_fallback:
        raw = [DEFAULT_SH_ATLAS.entry(p.family).force_constant for p in primitives]
    else:
        environments = sonic_primitive_environments_from_xyzin(path, atom_symbols, definition)
        raw = [DEFAULT_SH_ATLAS.entry(p.family).resolve_force_constant(e)
               for p, e in zip(primitives, environments, strict=True)]
    constants, normalization = normalized_primitive_curvatures(primitives, raw)
    parameter_audit = None
    if parameter_multipliers is not None:
        from .sh_parameters import primitive_parameter_key, checked_parameter_multipliers
        if legacy_fallback:
            raise ValueError('calibrated SH parameters require the frozen ORACLE topology')
        parameters = checked_parameter_multipliers(parameter_multipliers)
        keys = tuple(primitive_parameter_key(p, atom_symbols, bonds) for p in primitives)
        constants *= np.asarray([parameters.get(key, 1.) for key in keys])
        parameter_audit = {'mode': 'EXPLICIT_LOCAL_MULTIPLIERS',
            'matched_types': sorted(set(keys) & parameters.keys()),
            'unfitted_types': sorted(set(keys) - parameters.keys()),
            'multipliers': parameters}
    result = compiled_gic_primitive_csr(primitives, coords, reference_coords=reference)
    if result is None:
        raise RuntimeError("SH requires the compiled complete ORACLE primitive kernel")
    _, data, indices, indptr = result
    b = csr_matrix((data, indices, indptr), shape=(len(primitives), coords.size))
    if not np.isfinite(b.data).all():
        raise FloatingPointError("nonfinite primitive SH Jacobian")
    constants.setflags(write=False)
    return PrimitiveSH(primitives, b, constants, {
        "schema": "matrix.smith.primitive_sh.v1",
        "source": "LEGACY_FROZEN_DEFINITION" if legacy_fallback else "ORACLE_FROZEN_PRIMITIVES",
        "parameter_role": "PROVISIONAL_ZAFF_LOCAL_CURVATURE_APPROXIMATION",
        "formula": "(B J)^T K_normalized (B J)", "gradient_evaluations": 0,
        "dense_matrix_materialized": False, "normalization": normalization,
        "primitive_count": len(primitives), "primitive_nnz": b.nnz,
        "curvature_families": sorted({p.family for p in primitives}),
        "optimizer_primitive_selection": (
            "XTB_SIMPLE_R_A_L_D" if simple_optimizer_model else "ALL_ORACLE_FAMILIES"
        ),
        "parameter_calibration": parameter_audit,
    })


def checked_inverse_action(action, dimension, *, relative_tolerance=1.e-8, maximum_iterations=None):
    """Residual-certified inverse; singular/indefinite models fail explicitly."""
    operator = LinearOperator((dimension, dimension), matvec=action, dtype=float)
    def inverse(rhs):
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if rhs.shape != (dimension,) or not np.isfinite(rhs).all():
            raise ValueError("invalid SH inverse right-hand side")
        if not np.any(rhs):
            return np.zeros_like(rhs)
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                solution, status = cg(operator, rhs, rtol=relative_tolerance, atol=0.,
                                      maxiter=maximum_iterations or max(100, 4*dimension))
        except FloatingPointError as exc:
            raise FloatingPointError("SH inverse failed certification: Krylov breakdown") from exc
        error = np.linalg.norm(action(solution)-rhs) / np.linalg.norm(rhs)
        if status != 0 or not np.isfinite(error) or error > 2*relative_tolerance or rhs @ solution <= 0:
            raise FloatingPointError(f"SH inverse failed certification: status={status}, residual={error}")
        return solution
    return inverse
