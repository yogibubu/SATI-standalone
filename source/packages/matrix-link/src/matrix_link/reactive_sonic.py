"""Hessian-adapted SONIC construction over a conserved/reactive source pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from matrix_chem import atomic_mass
from matrix_chem.topology.elements import atomic_number
from matrix_gf import (
    curvilinear_sonic_force_constants,
    delocalized_internal_basis_from_b_matrix,
    gf_from_cartesian_hessian_and_gic_b_matrix,
    sonic_reactive_modes,
)
from matrix_smith import (
    GICDefinition,
    build_gic_b_matrix,
    materialize_hessian_adapted_reactive_sonic_definition,
)


HESSIAN_ADAPTED_REACTIVE_SONIC_RESULT_SCHEMA = (
    "matrix.link.hessian_adapted_reactive_sonic_result.v1"
)
ANGSTROM_TO_BOHR = 1.8897261246257702


@dataclass(frozen=True)
class HessianAdaptedReactiveSonicResult:
    """Auditable exact-rank chart derived from a redundant reactive pool."""

    schema: str
    definition: GICDefinition
    common_count: int
    reactive_source_count: int
    reactive_rank: int
    source_pool_rank: int
    pool_coordinate_unit_scales: np.ndarray
    pool_to_delocalized: np.ndarray
    delocalized_to_reactive_modes: np.ndarray
    pool_to_reactive_modes: np.ndarray
    prioritized_mode_order: tuple[int, ...]
    reactive_eigenvalues: tuple[float, ...]
    reactive_frequencies_cm: tuple[float, ...]
    transition_mode_significance: tuple[float, ...]
    reaction_subspace_coverage: float
    reaction_direction_coverages: tuple[float, ...]
    reaction_direction_curvatures: tuple[float, ...]
    projected_reaction_direction_curvatures: tuple[float | None, ...]
    reactive_instability_captured: bool
    conserved_coupling_by_mode: tuple[float, ...]
    normalized_conserved_coupling_by_mode: tuple[float, ...]
    conserved_chart_condition: float
    source_pool_condition: float
    delocalized_chart_condition: float
    final_chart_condition: float
    reactive_metric_condition: float
    reactive_off_diagonal_norm_before: float
    reactive_off_diagonal_norm_after: float
    conserved_coupling_norm: float
    conserved_coupling_invariance_residual: float
    metric_orthogonality_residual: float
    common_reactive_metric_orthogonality_residual: float
    full_spectrum_residual: float
    degenerate_blocks: tuple[tuple[int, ...], ...]
    reference_subspace_overlaps: tuple[float, ...]
    curvature_correction_used: bool
    connectivity_change: bool
    reactive_family_mixing_policy: str

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-compatible numerical provenance without duplicating the chart."""

        from matrix_smith import sonic_definition_identity_sha256

        payload: dict[str, object] = {
            "schema": self.schema,
            "definition_identity_sha256": sonic_definition_identity_sha256(
                self.definition
            ),
        }
        for name in self.__dataclass_fields__:
            if name in {"schema", "definition"}:
                continue
            value = getattr(self, name)
            payload[name] = value.tolist() if isinstance(value, np.ndarray) else value
        return payload


def hessian_adapt_reactive_sonic_pool(
    reactive_pool_definition: GICDefinition,
    common_count: int,
    atom_symbols: Sequence[str],
    reference_coordinates_angstrom: np.ndarray,
    cartesian_hessian_hartree_per_bohr2: np.ndarray,
    *,
    cartesian_gradient_hartree_per_bohr: np.ndarray | None = None,
    transition_modes: int | tuple[int, ...] | list[int] | np.ndarray = 0,
    reaction_directions: np.ndarray | None = None,
    reference_reactive_basis: np.ndarray | None = None,
    pool_coordinate_unit_scales: np.ndarray | None = None,
    rank_tolerance: float = 1.0e-9,
    maximum_numerical_condition: float | None = None,
    connectivity_change: bool | None = None,
) -> HessianAdaptedReactiveSonicResult:
    """Construct reactive normal coordinates from every source-pool row.

    Conserved SONICs are copied unchanged. The complete redundant reactive
    pool is first reduced by a mass-weighted delocalized-internal-coordinate
    decomposition in the conserved orthogonal complement. Its exact-rank
    Hessian block is then diagonalized in the Wilson kinetic metric. The final
    reactive coordinates are linear combinations of the original primitive
    pool, rather than a selected primitive subset.

    When a Cartesian gradient is supplied, the existing curvilinear SONIC
    transformation adds the gradient-times-coordinate-curvature term before
    diagonalization. No benchmark-specific acceptance threshold is used.
    """

    pool = reactive_pool_definition
    declared_connectivity_change = any(
        "CONNECTIVITY_CHANGE=YES" in diagnostic
        for diagnostic in pool.semantic_diagnostics
    )
    connectivity_changed = (
        declared_connectivity_change
        if connectivity_change is None
        else bool(connectivity_change)
    )
    if connectivity_change is not None and pool.primitive_source in {
        "ENDPOINT_CONNECTIVITY_REACTIVE_POOL",
        "MULTITOPOLOGY_REACTIVE_POOL",
    } and connectivity_changed != declared_connectivity_change:
        raise ValueError("reactive SONIC connectivity declaration conflicts with the pool")
    if not connectivity_changed:
        raise ValueError(
            "unchanged connectivity requires family-preserving minimum SONIC rules"
        )
    coordinates = np.asarray(reference_coordinates_angstrom, dtype=float)
    symbols = tuple(str(symbol) for symbol in atom_symbols)
    count = int(common_count)
    tolerance = float(rank_tolerance)
    maximum_condition = (
        1.0 / np.sqrt(np.finfo(float).eps)
        if maximum_numerical_condition is None
        else float(maximum_numerical_condition)
    )
    if coordinates.shape != (len(symbols), 3):
        raise ValueError("reactive SONIC geometry and atom ordering are incompatible")
    if count < 0 or count >= pool.target_rank or count >= len(pool.gics):
        raise ValueError("common_count is incompatible with the reactive source pool")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank_tolerance must be finite and positive")
    if not np.isfinite(maximum_condition) or maximum_condition < 1.0:
        raise ValueError("maximum_numerical_condition must be finite and at least one")
    hessian = np.asarray(cartesian_hessian_hartree_per_bohr2, dtype=float)
    cartesian_size = coordinates.size
    if hessian.shape != (cartesian_size, cartesian_size) or not np.all(np.isfinite(hessian)):
        raise ValueError("Cartesian Hessian must be finite with shape (3N, 3N)")
    masses = np.asarray([_atomic_mass(symbol) for symbol in symbols], dtype=float)
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)

    pool_rows = np.asarray(
        build_gic_b_matrix(pool, coordinates_angstrom=coordinates).rows,
        dtype=float,
    )
    coordinate_scales = _coordinate_unit_scales(
        pool,
        pool_coordinate_unit_scales,
    )
    weighted_pool_rows = coordinate_scales[:, None] * pool_rows * inverse_sqrt_mass[None, :]
    common_rows = weighted_pool_rows[:count]
    source_rows = weighted_pool_rows[count:]
    common_pseudoinverse, conserved_chart_condition = _full_row_rank_pseudoinverse(
        common_rows,
        tolerance,
    )
    _require_usable_condition(
        "conserved SONIC block",
        conserved_chart_condition,
        maximum_condition,
    )
    if count:
        conserved_projection = source_rows @ common_pseudoinverse
    else:
        conserved_projection = np.zeros((source_rows.shape[0], 0), dtype=float)
    projected_sources = source_rows - conserved_projection @ common_rows
    source_pool_condition = _nonzero_singular_condition(projected_sources, tolerance)
    _require_usable_condition(
        "projected reactive source pool",
        source_pool_condition,
        maximum_condition,
    )

    delocalized = delocalized_internal_basis_from_b_matrix(
        projected_sources,
        rcond=tolerance,
    )
    reactive_rank = int(pool.target_rank) - count
    if delocalized.rank != reactive_rank:
        raise ValueError(
            "reactive source pool does not span the exact conserved complement: "
            f"rank {delocalized.rank}, expected {reactive_rank}"
        )
    source_to_delocalized = np.asarray(delocalized.u_matrix.T, dtype=float)
    pool_to_delocalized = source_to_delocalized @ np.hstack(
        (-conserved_projection, np.eye(source_rows.shape[0], dtype=float))
    )
    pool_to_delocalized *= coordinate_scales[None, :]
    delocalized_definition = materialize_hessian_adapted_reactive_sonic_definition(
        pool,
        count,
        pool_to_delocalized,
        symbols,
        coordinates,
        rank_tolerance=tolerance,
        allow_reactive_family_mixing=True,
    )
    delocalized_rows = np.asarray(
        build_gic_b_matrix(
            delocalized_definition,
            coordinates_angstrom=coordinates,
        ).rows,
        dtype=float,
    )
    gf = gf_from_cartesian_hessian_and_gic_b_matrix(
        hessian,
        delocalized_rows,
        masses,
        gic_labels=tuple(gic.identifier for gic in delocalized_definition.gics),
        gic_names=tuple(gic.name for gic in delocalized_definition.gics),
        coordinate_source="hessian-adapted-reactive-pool",
    )
    # Express both tensors in the canonical q/(bohr sqrt(amu)) convention.
    # The lower-level GF adapter deliberately combines a bohr Hessian with an
    # angstrom B matrix because the two scale factors cancel in GF.  Here the
    # eigenvectors themselves are materialized as coordinates, so the factors
    # must be made explicit rather than merely allowed to cancel.
    force_constants = (
        np.asarray(gf.force_constants, dtype=float) * ANGSTROM_TO_BOHR**2
    )
    working_g_matrix = (
        np.asarray(gf.g_matrix, dtype=float) / ANGSTROM_TO_BOHR**2
    )
    gradient = cartesian_gradient_hartree_per_bohr
    curvature_correction_used = False
    if gradient is not None:
        cartesian_gradient = np.asarray(gradient, dtype=float).reshape(-1)
        if cartesian_gradient.shape != (cartesian_size,) or not np.all(
            np.isfinite(cartesian_gradient)
        ):
            raise ValueError("Cartesian gradient must be finite with 3N components")
        if float(np.linalg.norm(cartesian_gradient)) > 1.0e-12:
            force_constants = curvilinear_sonic_force_constants(
                hessian * ANGSTROM_TO_BOHR**2,
                cartesian_gradient * ANGSTROM_TO_BOHR,
                coordinates,
                lambda trial: np.asarray(
                    build_gic_b_matrix(
                        delocalized_definition,
                        coordinates_angstrom=trial,
                    ).rows,
                    dtype=float,
                ),
                masses=masses,
            )
            curvature_correction_used = True

    modes = sonic_reactive_modes(
        force_constants,
        working_g_matrix,
        count,
        transition_modes=transition_modes,
        reaction_directions=reaction_directions,
        reference_reactive_basis=reference_reactive_basis,
    )
    _require_usable_condition(
        "reactive Wilson metric",
        modes.reactive_metric_condition,
        maximum_condition,
    )
    order = np.asarray(modes.priority_order, dtype=int)
    prioritized_basis = modes.reactive_basis[:, order]
    delocalized_to_reactive = np.linalg.solve(
        prioritized_basis,
        np.eye(reactive_rank, dtype=float),
    )
    pool_to_reactive = delocalized_to_reactive @ pool_to_delocalized
    definition = materialize_hessian_adapted_reactive_sonic_definition(
        pool,
        count,
        pool_to_reactive,
        symbols,
        coordinates,
        rank_tolerance=tolerance,
        allow_reactive_family_mixing=True,
    )
    final_rows = (
        np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
            dtype=float,
        )
        * inverse_sqrt_mass[None, :]
    )
    chart_row_scales = np.ones(pool.target_rank, dtype=float)
    chart_row_scales[:count] = coordinate_scales[:count]
    delocalized_chart_condition = float(
        np.linalg.cond(
            chart_row_scales[:, None]
            * delocalized_rows
            * inverse_sqrt_mass[None, :]
        )
    )
    final_chart_condition = float(
        np.linalg.cond(chart_row_scales[:, None] * final_rows)
    )
    _require_usable_condition(
        "intermediate delocalized chart",
        delocalized_chart_condition,
        maximum_condition,
    )
    _require_usable_condition(
        "final Hessian-adapted chart",
        final_chart_condition,
        maximum_condition,
    )
    coupling_by_mode = np.linalg.norm(modes.conserved_couplings[:, order], axis=0)
    return HessianAdaptedReactiveSonicResult(
        schema=HESSIAN_ADAPTED_REACTIVE_SONIC_RESULT_SCHEMA,
        definition=definition,
        common_count=count,
        reactive_source_count=len(pool.gics) - count,
        reactive_rank=reactive_rank,
        source_pool_rank=delocalized.rank,
        pool_coordinate_unit_scales=coordinate_scales,
        pool_to_delocalized=pool_to_delocalized,
        delocalized_to_reactive_modes=delocalized_to_reactive,
        pool_to_reactive_modes=pool_to_reactive,
        prioritized_mode_order=tuple(int(index) for index in order),
        reactive_eigenvalues=tuple(float(value) for value in modes.eigenvalues[order]),
        reactive_frequencies_cm=tuple(float(value) for value in modes.frequencies_cm[order]),
        transition_mode_significance=tuple(
            float(value) for value in modes.transition_mode_significance[order]
        ),
        reaction_subspace_coverage=modes.reaction_subspace_coverage,
        reaction_direction_coverages=modes.reaction_direction_coverages,
        reaction_direction_curvatures=modes.reaction_direction_curvatures,
        projected_reaction_direction_curvatures=(
            modes.projected_reaction_direction_curvatures
        ),
        reactive_instability_captured=modes.reactive_instability_captured,
        conserved_coupling_by_mode=tuple(float(value) for value in coupling_by_mode),
        normalized_conserved_coupling_by_mode=tuple(
            float(value)
            for value in modes.normalized_conserved_coupling_by_mode[order]
        ),
        conserved_chart_condition=conserved_chart_condition,
        source_pool_condition=source_pool_condition,
        delocalized_chart_condition=delocalized_chart_condition,
        final_chart_condition=final_chart_condition,
        reactive_metric_condition=modes.reactive_metric_condition,
        reactive_off_diagonal_norm_before=modes.reactive_off_diagonal_norm_before,
        reactive_off_diagonal_norm_after=modes.reactive_off_diagonal_norm_after,
        conserved_coupling_norm=modes.conserved_coupling_norm,
        conserved_coupling_invariance_residual=(modes.conserved_coupling_invariance_residual),
        metric_orthogonality_residual=modes.metric_orthogonality_residual,
        common_reactive_metric_orthogonality_residual=float(
            np.linalg.norm(final_rows[:count] @ final_rows[count:].T)
        ),
        full_spectrum_residual=modes.full_spectrum_residual,
        degenerate_blocks=modes.degenerate_blocks,
        reference_subspace_overlaps=modes.reference_subspace_overlaps,
        curvature_correction_used=curvature_correction_used,
        connectivity_change=True,
        reactive_family_mixing_policy=(
            "ALL_PRIMITIVE_FAMILIES_WITHIN_REACTIVE_COMPLEMENT_ONLY"
        ),
    )


def _full_row_rank_pseudoinverse(
    rows: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(rows, dtype=float)
    if matrix.size == 0:
        return np.zeros((matrix.shape[1], 0), dtype=float), 1.0
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(float(singular[0]) * tolerance, tolerance)
    rank = int(np.count_nonzero(singular > threshold))
    if rank != matrix.shape[0]:
        raise ValueError("conserved SONIC block is rank deficient")
    pseudoinverse = (right_t.T / singular) @ left.T
    return pseudoinverse, float(singular[0] / singular[-1])


def _nonzero_singular_condition(rows: np.ndarray, tolerance: float) -> float:
    singular = np.linalg.svd(np.asarray(rows, dtype=float), compute_uv=False)
    if not singular.size:
        return float("inf")
    threshold = max(float(singular[0]) * tolerance, tolerance)
    retained = singular[singular > threshold]
    if not retained.size:
        return float("inf")
    return float(retained[0] / retained[-1])


def _require_usable_condition(
    label: str,
    condition: float,
    maximum: float,
) -> None:
    if not np.isfinite(condition) or condition > maximum:
        raise ValueError(
            f"{label} is numerically ill-conditioned: "
            f"condition {condition:.6g} exceeds {maximum:.6g}"
        )


def _atomic_mass(symbol: str) -> float:
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise ValueError(f"unsupported atom symbol in reactive SONIC pool: {symbol}")
    return float(atomic_mass(int(number)))


def _coordinate_unit_scales(
    definition: GICDefinition,
    explicit: np.ndarray | None,
) -> np.ndarray:
    if explicit is not None:
        scales = np.asarray(explicit, dtype=float).reshape(-1)
        if scales.shape != (len(definition.gics),):
            raise ValueError("pool coordinate-unit scales must match the pool rows")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("pool coordinate-unit scales must be finite and positive")
        return scales
    from matrix_smith.policy import PRIMITIVE_POLICY_BY_FAMILY

    length_functions = {
        "R",
        "H",
        "FC_DIST",
        "FCA_DIST",
        "FTRANS",
        "CENTER_ATOM_DIST",
    }
    unit_functions = {
        "A",
        "L",
        "D",
        "U",
        "IMPD",
        "RPCB",
        "RPCK",
        "FROT",
        "DIMENSIONLESS",
    }
    scales: list[float] = []
    for gic in definition.gics:
        policy = PRIMITIVE_POLICY_BY_FAMILY.get(str(gic.family))
        if policy is None:
            raise ValueError(f"no coordinate-unit contract exists for pool family {gic.family!r}")
        function = str(policy.function).upper()
        if function in length_functions:
            scales.append(ANGSTROM_TO_BOHR)
        elif function in unit_functions:
            scales.append(1.0)
        else:
            raise ValueError(f"no coordinate-unit scale exists for pool function {function!r}")
    return np.asarray(scales, dtype=float)


__all__ = [
    "HESSIAN_ADAPTED_REACTIVE_SONIC_RESULT_SCHEMA",
    "HessianAdaptedReactiveSonicResult",
    "hessian_adapt_reactive_sonic_pool",
]
