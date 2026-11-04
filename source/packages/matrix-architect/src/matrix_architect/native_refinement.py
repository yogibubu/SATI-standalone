"""Topology-defined native corrections; no anharmonic reference is consulted."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from scipy.linalg import null_space

from matrix_chem.primitive_coordinates import Primitive, primitive_b_matrix, eval_primitives
from matrix_chem.primitive_force_field import (
    PrimitiveMMForceField, evaluate_torsion_fourier_parameter,
)
from matrix_zaff.basic_primitive_derivative import analytic_basic_primitive_second_derivative

from .asymptotic_pic import AsymptoticPicFactor
from .native_sparse_fit import native_zaff_candidate_powers, fit_native_sparse_refinement


def _periodic_axes(primitives, periodic_indices):
    axes = set()
    for index in periodic_indices:
        if (not isinstance(index, (int, np.integer)) or isinstance(index, bool)
                or not 0 <= index < len(primitives)):
            raise ValueError("invalid periodic primitive index")
        primitive = primitives[index]
        if primitive.kind != "dihedral":
            raise ValueError("periodic refinement requires a dihedral primitive")
        axis = frozenset(primitive.atoms[1:3])
        if axis in axes:
            raise ValueError("only one periodic representative per central bond")
        axes.add(axis)
    return axes


def native_refinement_catalog(
    primitives: Sequence[Primitive],
    *,
    periodic_indices: Sequence[int] = (),
    maximum_active_primitives: int = 2,
    maximum_fourier_order: int = 8,
    periodic_coupling_shells: int = 1,
    maximum_periodic_pair_fourier_order: int = 0,
    periodic_window_start: float = 0.0,
    periodic_window_end: float = 0.0,
) -> tuple[tuple[AsymptoticPicFactor, ...], ...]:
    """Enumerate cubic/quartic native products and declared periodic refinements.

    Periodic representatives come from the existing ORACLE/SMITH contract:
    one primitive for each soft central bond. This function does not classify
    torsions or use frequencies, cubic constants, or quartic constants.
    Every returned product starts at order >= 3 at the reference. Evaluate it
    with ``asymptotic_pic_correction_design`` to include the bond switches.
    """
    if maximum_fourier_order < 2 or int(maximum_fourier_order) != maximum_fourier_order:
        raise ValueError("maximum Fourier order must be an integer of at least two")
    if (not isinstance(periodic_coupling_shells, (int, np.integer))
            or isinstance(periodic_coupling_shells, bool)
            or periodic_coupling_shells < 0):
        raise ValueError("periodic coupling shells must be a nonnegative integer")
    if (not isinstance(maximum_periodic_pair_fourier_order, (int, np.integer))
            or isinstance(maximum_periodic_pair_fourier_order, bool)
            or maximum_periodic_pair_fourier_order < 0):
        raise ValueError("periodic pair Fourier order must be a nonnegative integer")
    protected_periodic_neighbourhood = periodic_window_start != 0.0 or periodic_window_end != 0.0
    if protected_periodic_neighbourhood and not (
        np.isfinite(periodic_window_start)
        and np.isfinite(periodic_window_end)
        and 0.0 < periodic_window_start < periodic_window_end < np.pi
    ):
        raise ValueError("periodic protection window requires 0 < start < end < pi")
    _periodic_axes(primitives, periodic_indices)
    if any(p.kind not in {"bond", "angle", "dihedral", "out_of_plane"} for p in primitives):
        raise ValueError("native refinement catalog requires basic valence primitives")

    def form(index):
        return ("inverse_distance" if primitives[index].kind == "bond" else
                "native" if primitives[index].kind == "angle" else "sin")

    # Covalent graph locality is more general than sharing a rotor atom.  A
    # relaxed scan changes forces on coordinates in the first bonded shell of
    # the rotating axis (for example an O--H bond attached through a carbonyl
    # oxygen).  Build this neighborhood once from the existing bond
    # primitives; disconnected fragments remain excluded.
    adjacency: dict[int, set[int]] = {}
    for primitive in primitives:
        if primitive.kind == "bond":
            left, right = primitive.atoms
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)

    def within_rotor_shell(primitive, rotor) -> bool:
        frontier = set(primitives[rotor].atoms)
        visited = set(frontier)
        for _ in range(periodic_coupling_shells):
            frontier = {
                neighbor
                for atom in frontier
                for neighbor in adjacency.get(atom, ())
                if neighbor not in visited
            }
            visited.update(frontier)
        return bool(set(primitive.atoms) & visited)

    candidates = []
    for power in native_zaff_candidate_powers(
        [item.atoms for item in primitives],
        maximum_active_primitives=maximum_active_primitives,
    ):
        candidates.append(tuple(AsymptoticPicFactor(i, form(i), exponent)
                                for i, exponent in enumerate(power) if exponent))
    for rotor in periodic_indices:
        for harmonic in range(2, maximum_fourier_order + 1):
            for shape in ("jet_cos", "jet_sin"):
                candidates.append((AsymptoticPicFactor(rotor, shape, harmonic=harmonic),))
        for index, primitive in enumerate(primitives):
            # The representative fixes the periodic phase, while the other
            # redundant dihedrals around the same central bond remain valid
            # response coordinates.  Excluding the whole axis would make a
            # torsional force depend on only one terminal atom and is therefore
            # incorrect for rotors such as NH2 and CH3.  Exclude only the
            # representative itself.
            if index == rotor:
                continue
            if not within_rotor_shell(primitive, rotor):
                continue
            u = AsymptoticPicFactor(index, form(index))
            u2 = AsymptoticPicFactor(index, form(index), 2)
            sine = AsymptoticPicFactor(rotor, "sin")
            cosine = AsymptoticPicFactor(rotor, "one_minus_cos")
            candidates.extend(((u, cosine), (u, sine, cosine), (u2, sine), (u2, cosine)))
            # A relaxed rotor scan measures the periodic displacement and
            # curvature response of every coupled SAM coordinate.  A single
            # sine/cosine pair cannot represent that response in general.
            # Couple the same local coordinate to the higher Fourier shapes
            # after removing their central two-jet.  Consequently these
            # columns can improve finite-amplitude forces without changing
            # the ZAFF2 reference energy, gradient, or Hessian.  The common
            # asymptotic evaluator supplies all defining-bond switches.
            for harmonic in range(2, maximum_fourier_order + 1):
                jet_cosine = AsymptoticPicFactor(
                    rotor, "jet_cos", harmonic=harmonic,
                )
                jet_sine = AsymptoticPicFactor(
                    rotor, "jet_sin", harmonic=harmonic,
                )
                candidates.extend(
                    (
                        (u, jet_cosine),
                        (u, jet_sine),
                        (u2, jet_cosine),
                        (u2, jet_sine),
                    )
                )
        if maximum_periodic_pair_fourier_order:
            # If the linear LAM--SAM response does not pass the force gate,
            # the next physically ordered block is the periodic variation of
            # a local SAM--SAM curvature.  Restrict it a priori to bonded and
            # angular coordinates that share an atom.  This keeps the block
            # local and sparse; every column still has a zero central two-jet.
            local_sam = [
                index for index, primitive in enumerate(primitives)
                if primitive.kind in {"bond", "angle"}
                and within_rotor_shell(primitive, rotor)
            ]
            for position, left in enumerate(local_sam):
                for right in local_sam[position + 1:]:
                    if not set(primitives[left].atoms).intersection(primitives[right].atoms):
                        continue
                    left_factor = AsymptoticPicFactor(left, form(left))
                    right_factor = AsymptoticPicFactor(right, form(right))
                    for harmonic in range(1, maximum_periodic_pair_fourier_order + 1):
                        candidates.append((
                            left_factor,
                            right_factor,
                            AsymptoticPicFactor(
                                rotor, "one_minus_cos", harmonic=harmonic,
                            ),
                        ))
                        if harmonic >= 2:
                            candidates.append((
                                left_factor,
                                right_factor,
                                AsymptoticPicFactor(
                                    rotor, "jet_sin", harmonic=harmonic,
                                ),
                            ))
    if protected_periodic_neighbourhood:
        protected = []
        for term in candidates:
            rotors = {
                factor.primitive_index for factor in term
                if factor.primitive_index in periodic_indices
                and factor.form in {"jet_cos", "jet_sin"}
            }
            if len(rotors) == 1:
                rotor = next(iter(rotors))
                term = term + (AsymptoticPicFactor(
                    rotor,
                    "outer_window",
                    window_start=float(periodic_window_start),
                    window_end=float(periodic_window_end),
                ),)
            protected.append(term)
        candidates = protected

    # Canonicalize products so identical native factors are not fitted twice.
    def key(term):
        return tuple((x.primitive_index, x.form, x.power, x.harmonic,
                      x.window_start, x.window_end) for x in term)
    unique = {}
    for term in candidates:
        ordered = tuple(sorted(
            term,
            key=lambda x: (x.primitive_index, x.form, x.power, x.harmonic,
                           x.window_start, x.window_end),
        ))
        unique[key(ordered)] = ordered
    return tuple(unique[k] for k in sorted(unique))


@dataclass(frozen=True)
class PeriodicReferenceFreedom:
    """Periodic baseline replacement with exactly constrained central derivatives."""

    base_field: PrimitiveMMForceField
    factors: tuple[tuple[AsymptoticPicFactor, ...], ...]
    particular: np.ndarray
    null_basis: np.ndarray
    constraint_matrix: np.ndarray


@dataclass(frozen=True)
class NativePointFit:
    """Frozen E/G selection; acceptance concerns point data, not a spectrum."""

    coefficients: np.ndarray
    accepted: bool
    selected_size: int
    condition_number: float
    errors: dict[str, float]
    history: tuple[dict[str, float], ...]


def fit_native_point_observations(
    energy_design: np.ndarray,
    gradient_design: np.ndarray,
    residual_energies: np.ndarray,
    residual_gradients: np.ndarray,
    *,
    training_indices: Sequence[int],
    validation_indices: Sequence[int],
    energy_tolerance: float,
    gradient_tolerance: float,
    maximum_terms: int,
    condition_limit: float = 1e8,
    sample_groups: Sequence[str] | None = None,
) -> NativePointFit:
    """Select the smallest passing native support using only residual point E/G.

    Reference constraints must already be enforced in the columns, e.g. with
    ``periodic_reference_freedom``. E/G units are those of the supplied
    tolerances. Gradient rows are balanced by sqrt(3N) in the least-squares
    objective; acceptance still checks the maximum component independently.
    Symmetry-related samples must have the same group. No frequency or higher
    QM derivative is accepted by this interface.

    If the finite search fails, return the best tested support with accepted
    false and the actual residuals. It must not be promoted to a certified PES.
    """
    e, g, y, dy = (np.asarray(x, dtype=float) for x in
                   (energy_design, gradient_design, residual_energies, residual_gradients))
    if (e.ndim != 2 or not all(e.shape) or g.ndim != 3
            or g.shape[0] != e.shape[0] or g.shape[2] != e.shape[1]
            or y.shape != (len(e),) or dy.shape != g.shape[:2] or g.shape[1] == 0
            or not all(np.all(np.isfinite(x)) for x in (e, g, y, dy))):
        raise ValueError("incomplete or nonfinite native point observations")
    if (not np.isfinite(energy_tolerance) or energy_tolerance <= 0
            or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0):
        raise ValueError("native point tolerances must be finite and positive")
    partitions = []
    for raw in (training_indices, validation_indices):
        rows = np.asarray(raw)
        if (rows.ndim != 1 or len(rows) == 0 or not np.issubdtype(rows.dtype, np.integer)
                or np.any(rows < 0) or np.any(rows >= len(e)) or len(set(rows)) != len(rows)):
            raise ValueError("invalid native point partition")
        partitions.append(rows)
    train, valid = partitions
    if set(train) & set(valid):
        raise ValueError("training and validation points overlap")
    if sample_groups is not None:
        if len(sample_groups) != len(e):
            raise ValueError("sample groups do not match point count")
        if {sample_groups[i] for i in train} & {sample_groups[i] for i in valid}:
            raise ValueError("equivalent samples cross training and validation")
    if (not isinstance(maximum_terms, (int, np.integer)) or maximum_terms < 1
            or not np.isfinite(condition_limit) or condition_limit < 1):
        raise ValueError("invalid native selection limits")
    a = np.vstack((e[train]/energy_tolerance,
                   g[train].reshape(-1, e.shape[1])/(gradient_tolerance*np.sqrt(g.shape[1]))))
    target = np.r_[y[train]/energy_tolerance,
                   dy[train].ravel()/(gradient_tolerance*np.sqrt(g.shape[1]))]
    observed = np.flatnonzero(np.linalg.norm(a, axis=0) > 1e-14)
    if not len(observed):
        raise ValueError("point observations resolve no native terms")
    history = []
    designs = [(e[rows][:, observed], g[rows][:, :, observed], y[rows], dy[rows])
               for rows in partitions]

    def check(c):
        errors = {}
        for name, (ae, ag, ye, yg) in zip(("training", "validation"), designs):
            errors[name+"_energy"] = float(np.max(np.abs(ae@c-ye)))
            errors[name+"_gradient"] = float(np.max(np.abs(ag@c-yg)))
        score = max(errors[part+"_"+kind]/tol for part in ("training", "validation")
                    for kind, tol in (("energy", energy_tolerance), ("gradient", gradient_tolerance)))
        history.append({**errors, "normalized_maximum": score})
        return score <= 1

    path = fit_native_sparse_refinement(
        a[:, observed], target, max_terms=min(maximum_terms, len(observed), len(a)),
        condition_limit=condition_limit, stop_when=check,
    )
    best = min(range(len(history)), key=lambda i: (history[i]["normalized_maximum"], i))
    coefficients = np.zeros(e.shape[1])
    coefficients[observed] = path.coefficients(best+1, len(observed))
    return NativePointFit(coefficients, history[best]["normalized_maximum"] <= 1,
                          best+1, path.condition_by_size[best+1], history[best], tuple(history))


def periodic_reference_freedom(
    field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    reference_angstrom: np.ndarray,
    periodic_indices: Sequence[int],
) -> PeriodicReferenceFreedom:
    """Refit redundant periodic harmonic terms while preserving total G/H at x0.

    Only the reference geometry and the existing harmonic field enter this
    construction. Removed terms are replaced by sine/cosine and native pair
    terms with the same central first/second derivatives. Their null space
    allows the additional E/G observations to determine the finite-amplitude
    continuation. Each replacement uses the production asymptotic switches.
    """
    axes = _periodic_axes(primitives, periodic_indices)
    if not np.allclose(eval_primitives(primitives, reference_angstrom),
                       field.equilibrium_values, atol=1e-9, rtol=0):
        raise ValueError("periodic reparameterization requires the field reference geometry")
    rotors = {i for i, p in enumerate(primitives)
              if p.kind == "dihedral" and frozenset(p.atoms[1:3]) in axes}
    if not rotors:
        raise ValueError("no periodic dihedrals were supplied")
    ids = {field.primitive_type_ids[i] for i in rotors}
    if any(field.primitive_type_ids[i] in ids and i not in rotors for i in range(len(primitives))):
        raise ValueError("periodic and rigid primitives must have separate parameter types")
    torsions = {t.primitive_type_id: t for t in field.torsion_fourier_parameters}
    removed = {tuple(sorted(t.primitive_indices)): t.force_constant
               for t in field.retained_couplings if set(t.primitive_indices) & rotors}
    base = replace(
        field,
        torsion_fourier_parameters=tuple(replace(t, constant=0., terms=()) if
                                        t.primitive_type_id in ids else t
                                        for t in field.torsion_fourier_parameters),
        retained_couplings=tuple(t for t in field.retained_couplings
                                if not set(t.primitive_indices) & rotors),
    )
    b = primitive_b_matrix(primitives, reference_angstrom)
    upper = np.triu_indices(b.shape[1])
    weight = np.where(upper[0] == upper[1], 1., np.sqrt(2.))
    factors, particular, columns = [], [], []
    for i in sorted(rotors):
        _, first, second = evaluate_torsion_fourier_parameter(
            torsions[field.primitive_type_ids[i]], float(field.equilibrium_values[i]),
        )
        h = np.zeros((b.shape[1], b.shape[1]))
        primitive = SimpleNamespace(function="D", atoms=tuple(a+1 for a in primitives[i].atoms))
        for left, right, value in analytic_basic_primitive_second_derivative(
            primitive, reference_angstrom,
        ).entries:
            h[left, right] = value
        factors.extend(((AsymptoticPicFactor(i, "sin"),),
                        (AsymptoticPicFactor(i, "one_minus_cos"),)))
        particular.extend((first, second))
        columns.extend((np.r_[b[i], h[upper]*weight],
                        np.r_[np.zeros(b.shape[1]), np.outer(b[i], b[i])[upper]*weight]))
    for i in range(len(primitives)):
        for j in range(i+1, len(primitives)):
            if i not in rotors and j not in rotors:
                continue
            factors.append((AsymptoticPicFactor(i, "native"), AsymptoticPicFactor(j, "native")))
            particular.append(removed.get((i, j), 0.))
            h = np.outer(b[i], b[j]) + np.outer(b[j], b[i])
            columns.append(np.r_[np.zeros(b.shape[1]), h[upper]*weight])
    constraints = np.array(columns).T
    null = null_space(constraints, rcond=1e-10)
    return PeriodicReferenceFreedom(base, tuple(factors), np.array(particular), null, constraints)
