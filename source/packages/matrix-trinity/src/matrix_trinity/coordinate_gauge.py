"""Transport resonant Fermi polyads between equivalent harmonic charts.

The quadratic chart connection generates a first-order unitary change of
oscillator representation. Bare cubic Fermi couplings are chart dependent by
the commutator [S, H0]. The nonresonant second-order energy is retained and
only the resonant terms are removed again in the selected common gauge. The
cubic chart jet supplies second-order field and property transport.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, combinations_with_replacement, permutations, product
from math import factorial, sqrt
from typing import Callable, Mapping

import numpy as np
from matrix_gf.harmonic import CM_PER_HARTREE
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM

from .curvilinear_vpt2 import (
    SonicGVPT2Result, SonicVPT2Input, _apply, _cubic_couplings, _modalize_sonic_field,
    _operator_element, solve_sonic_gvpt2,
)
from .gvpt2_spectroscopy import SonicPropertyDerivatives


@dataclass(frozen=True)
class QuadraticChartConnection:
    """Map native modes to a fixed Cartesian-anchored reference gauge."""

    second: np.ndarray
    reference_mode_signs: np.ndarray
    tangent_overlap: np.ndarray
    linear: np.ndarray | None = None
    degenerate_blocks: tuple[tuple[int, ...], ...] = ()
    canonical: np.ndarray | None = None
    third: np.ndarray | None = None


@dataclass(frozen=True)
class TransportedFermiPolyad:
    indices: tuple[int, ...]
    hamiltonian_cm: np.ndarray
    eigenvalues_cm: np.ndarray
    eigenvectors: np.ndarray
    pair_couplings_cm: tuple[tuple[int, int, float, float, float], ...]


@dataclass(frozen=True)
class TransportedResonancePolyad:
    """Second-order polyad in one real, symmetry-projected reference gauge."""

    states: tuple[tuple[int, ...], ...]
    hamiltonian_cm: np.ndarray
    eigenvalues_cm: np.ndarray
    eigenvectors: np.ndarray
    group_closure_residual: float
    group_invariance_residual_cm: float


@dataclass(frozen=True)
class CommonGaugeVP2TD:
    """Second-order field, property, and VP2TD result in one harmonic gauge."""

    field: SonicVPT2Input
    dipole: SonicPropertyDerivatives
    gvpt2: SonicGVPT2Result


def assess_large_amplitude_dvr(
    *, mode: int, harmonic_cm: float,
    potential_intermode_norm_native_cm: float,
    potential_intermode_norm_reference_cm: float,
    dvr_levels_relative_cm: np.ndarray, path_mode_overlap: float,
    grid_gap_difference_cm: float, grid_ir_difference_km_mol: float,
    grid_gap_tolerance_cm: float = 0.1,
    grid_ir_tolerance_km_mol: float = 0.1,
    minimum_path_mode_overlap: float = 0.95,
) -> dict[str, object]:
    """Assess a saved molecular DVR separately from semi-rigid VP2TD modes."""
    levels = np.asarray(dvr_levels_relative_cm, dtype=float)
    scalars = (
        harmonic_cm, potential_intermode_norm_native_cm,
        potential_intermode_norm_reference_cm, path_mode_overlap,
        grid_gap_difference_cm, grid_ir_difference_km_mol,
        grid_gap_tolerance_cm, grid_ir_tolerance_km_mol,
        minimum_path_mode_overlap,
    )
    if (mode < 0 or levels.ndim != 1 or len(levels) < 2
            or not np.all(np.isfinite(levels))
            or not np.isclose(levels[0], 0.0, atol=1e-7)
            or np.any(np.diff(levels) <= 0)
            or not all(np.isfinite(value) for value in scalars)
            or harmonic_cm <= 0
            or potential_intermode_norm_native_cm < 0
            or potential_intermode_norm_reference_cm < 0
            or not 0 <= path_mode_overlap <= 1
            or grid_gap_difference_cm < 0 or grid_ir_difference_km_mol < 0
            or grid_gap_tolerance_cm <= 0 or grid_ir_tolerance_km_mol <= 0
            or not 0 < minimum_path_mode_overlap <= 1):
        raise ValueError("invalid large-amplitude DVR evidence")
    localization_ratio = (
        potential_intermode_norm_native_cm / potential_intermode_norm_reference_cm
        if potential_intermode_norm_reference_cm > 0 else None
    )
    passed = (path_mode_overlap >= minimum_path_mode_overlap
              and grid_gap_difference_cm <= grid_gap_tolerance_cm
              and grid_ir_difference_km_mol <= grid_ir_tolerance_km_mol)
    return {
        "policy_id": "matrix-trinity.large-amplitude-dvr.v1",
        "mode": int(mode),
        "harmonic_cm": float(harmonic_cm),
        "first_dvr_gap_cm": float(levels[1]),
        "dvr_levels_relative_cm": levels.tolist(),
        "path_mode_overlap": float(path_mode_overlap),
        "potential_intermode_norm_native_cm": float(
            potential_intermode_norm_native_cm),
        "potential_intermode_norm_reference_cm": float(
            potential_intermode_norm_reference_cm),
        "localization_ratio_native_to_reference": localization_ratio,
        "grid_gap_difference_cm": float(grid_gap_difference_cm),
        "grid_ir_difference_km_mol": float(grid_ir_difference_km_mol),
        "grid_gap_tolerance_cm": float(grid_gap_tolerance_cm),
        "grid_ir_tolerance_km_mol": float(grid_ir_tolerance_km_mol),
        "minimum_path_mode_overlap": float(minimum_path_mode_overlap),
        "passed": bool(passed),
    }


def solve_common_gauge_vp2td(
    field: SonicVPT2Input,
    dipole: SonicPropertyDerivatives,
    connection: QuadraticChartConnection,
    **gvpt2_options: object,
) -> CommonGaugeVP2TD:
    """Transport the complete second-order model before selecting polyads.

    Fermi-only coupling transport does not include the chart-induced quartic
    and metric terms. This route applies the cubic inverse-chart jet to the
    force field and D1/D2/D3, then rebuilds VP2TD in the reference gauge.
    It requires a connection constructed with ``include_cubic=True``.
    """
    common_field = transport_sonic_field_second_order(field, connection)
    common_dipole = transport_sonic_property_second_order(dipole, connection)
    common_result = solve_sonic_gvpt2(common_field, **gvpt2_options)
    return CommonGaugeVP2TD(common_field, common_dipole, common_result)


def assess_chart_consistency(
    *, native_field: SonicVPT2Input, reference_field: SonicVPT2Input,
    native_fundamentals_cm: np.ndarray, reference_fundamentals_cm: np.ndarray,
    common_fundamentals_cm: np.ndarray,
    reference_ir_km_mol: np.ndarray, common_ir_km_mol: np.ndarray,
    frequency_tolerance_cm: float, ir_tolerance_km_mol: float,
    ir_relative_tolerance: float = 0.02,
    large_amplitude_modes: tuple[int, ...] = (),
    dvr_diagnostics: Mapping[int, object] | None = None,
) -> dict[str, object]:
    """Versioned I/C assessment after full second-order chart transport.

    Semi-rigid modes are checked in the common gauge. Explicitly declared
    large-amplitude modes are reported with potential intermode coupling
    norms and their DVR diagnostic, without imposing I/C equality on the
    finite-order local description. A missing DVR record leaves the
    assessment incomplete rather than silently accepting the mode.
    """
    native = np.asarray(native_fundamentals_cm, dtype=float)
    reference = np.asarray(reference_fundamentals_cm, dtype=float)
    common = np.asarray(common_fundamentals_cm, dtype=float)
    ir_reference = np.real_if_close(reference_ir_km_mol)
    ir_common = np.real_if_close(common_ir_km_mol)
    n = len(reference)
    if (native.shape != (n,) or common.shape != (n,)
            or ir_reference.shape != (n,) or ir_common.shape != (n,)
            or np.iscomplexobj(ir_reference) or np.iscomplexobj(ir_common)
            or not all(np.all(np.isfinite(value)) for value in (
                native, reference, common, ir_reference, ir_common))
            or not np.isfinite(frequency_tolerance_cm) or frequency_tolerance_cm <= 0
            or not np.isfinite(ir_tolerance_km_mol) or ir_tolerance_km_mol <= 0
            or not np.isfinite(ir_relative_tolerance) or ir_relative_tolerance < 0):
        raise ValueError("invalid common-gauge frequency or IR comparison")
    soft = tuple(int(mode) for mode in large_amplitude_modes)
    if len(set(soft)) != len(soft) or any(mode < 0 or mode >= n for mode in soft):
        raise ValueError("large-amplitude mode indices must be distinct and in range")
    dvr = {} if dvr_diagnostics is None else dict(dvr_diagnostics)
    if any(mode not in soft for mode in dvr):
        raise ValueError("DVR diagnostics must belong to declared large-amplitude modes")
    _, _, f3_native, _, _, _ = _modalize_sonic_field(native_field)
    _, _, f3_reference, _, _, _ = _modalize_sonic_field(reference_field)
    if f3_native.shape != (n, n, n) or f3_reference.shape != (n, n, n):
        raise ValueError("force-field modes disagree with the chart comparison")
    modes = []
    for mode in range(n):
        row = {
            "mode": mode,
            "native_frequency_difference_cm": float(native[mode] - reference[mode]),
            "transported_frequency_difference_cm": float(common[mode] - reference[mode]),
            "transported_ir_difference_km_mol": float(ir_common[mode] - ir_reference[mode]),
        }
        if mode in soft:
            def intermode_norm(f3: np.ndarray) -> float:
                couplings = f3[mode].copy()
                couplings[mode, mode] = 0.0
                return float(np.linalg.norm(couplings))
            row.update({
                "role": "large_amplitude",
                "potential_intermode_norm_native_cm": intermode_norm(f3_native),
                "potential_intermode_norm_reference_cm": intermode_norm(f3_reference),
                "dvr_diagnostic": dvr.get(mode),
            })
        else:
            ir_limit = max(ir_tolerance_km_mol, ir_relative_tolerance * max(
                abs(float(ir_reference[mode])), abs(float(ir_common[mode]))))
            row.update({
                "role": "semi_rigid",
                "ir_limit_km_mol": ir_limit,
                "within_tolerance": bool(
                    abs(row["transported_frequency_difference_cm"]) <= frequency_tolerance_cm
                    and abs(row["transported_ir_difference_km_mol"]) <= ir_limit
                ),
            })
        modes.append(row)
    checked = [row for row in modes if row["role"] == "semi_rigid"]
    complete = all(
        isinstance(row["dvr_diagnostic"], Mapping)
        and row["dvr_diagnostic"].get("policy_id")
        == "matrix-trinity.large-amplitude-dvr.v1"
        and row["dvr_diagnostic"].get("mode") == row["mode"]
        and row["dvr_diagnostic"].get("passed") is True
        for row in modes if row["role"] == "large_amplitude"
    )
    return {
        "policy_id": "matrix-trinity.chart-consistency.v1",
        "frequency_tolerance_cm": float(frequency_tolerance_cm),
        "ir_tolerance_km_mol": float(ir_tolerance_km_mol),
        "ir_relative_tolerance": float(ir_relative_tolerance),
        "semi_rigid_passed": all(row["within_tolerance"] for row in checked),
        "large_amplitude_dvr_complete": complete,
        "passed": all(row["within_tolerance"] for row in checked) and complete,
        "modes": modes,
    }


def quadratic_chart_connection(
    *, native_geometry_at_s: Callable[[np.ndarray], np.ndarray],
    reference_geometry_at_s: Callable[[np.ndarray], np.ndarray],
    native_displacement_per_q: np.ndarray,
    reference_displacement_per_q: np.ndarray,
    masses_amu: np.ndarray,
    step_dimensionless: float = 0.005,
    degenerate_blocks: tuple[tuple[int, ...], ...] = (),
    include_cubic: bool = False,
    cubic_step_dimensionless: float = 0.025,
) -> QuadraticChartConnection:
    """Derive the common-mode connection from two body-fixed geometry maps.

    Each declared degenerate block is aligned by the unique orthogonal polar
    factor of its mass-weighted tangent overlap. Other modes retain the
    original phase alignment. No point-group-specific formula is used.
    """
    native = np.asarray(native_displacement_per_q, dtype=float)
    reference = np.asarray(reference_displacement_per_q, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    n = len(native)
    h = float(step_dimensionless)
    if (native.shape != (n, n) or reference.shape != (n, n)
            or masses.size < 2 or np.any(masses <= 0)
            or not np.isfinite(h) or h <= 0):
        raise ValueError("invalid quadratic chart connection inputs")
    size = 3 * masses.size
    zero = np.zeros(n)

    def points(geometry, displacement, q):
        xyz = np.asarray(geometry(displacement @ q), dtype=float)
        if xyz.shape != (masses.size, 3) or not np.all(np.isfinite(xyz)):
            raise ValueError("chart returned invalid Cartesian geometry")
        return xyz.reshape(-1)

    origin_native = points(native_geometry_at_s, native, zero)
    origin_reference = points(reference_geometry_at_s, reference, zero)
    if np.max(abs(origin_native - origin_reference)) > 1e-7:
        raise ValueError("quadratic chart connection requires a common geometry")
    eye = np.eye(n)

    def tangent(geometry, displacement):
        return np.column_stack([
            (points(geometry, displacement, h * eye[j])
             - points(geometry, displacement, -h * eye[j])) / (2 * h)
            for j in range(n)
        ])

    tangent_native = tangent(native_geometry_at_s, native)
    tangent_reference = tangent(reference_geometry_at_s, reference)
    weights = np.sqrt(np.repeat(masses, 3))
    weighted_native = weights[:, None] * tangent_native
    weighted_reference = weights[:, None] * tangent_reference
    overlap = weighted_native.T @ weighted_reference / np.sqrt(np.outer(
        np.sum(weighted_native**2, axis=0),
        np.sum(weighted_reference**2, axis=0),
    ))
    seen: set[int] = set()
    blocks = []
    for declared in degenerate_blocks:
        block = tuple(sorted(int(index) for index in declared))
        if len(block) < 2 or any(index < 0 or index >= n or index in seen for index in block):
            raise ValueError("degenerate mode blocks must be disjoint and in range")
        seen.update(block)
        blocks.append(block)
    blocks.extend((index,) for index in range(n) if index not in seen)
    linear = np.zeros((n, n))
    signs = np.ones(n)
    for block in blocks:
        ix = np.asarray(block, dtype=int)
        cross = weighted_reference[:, ix].T @ weighted_native[:, ix]
        u, singular, vt = np.linalg.svd(cross)
        if np.min(singular) <= 0:
            raise ValueError("degenerate mode alignment has deficient rank")
        aligned = u @ vt
        linear[np.ix_(ix, ix)] = aligned
        if len(block) == 1:
            signs[block[0]] = aligned[0, 0]
    reference_tangent = tangent_reference @ linear
    residual = np.max(np.linalg.norm(
        weights[:, None] * (tangent_native - reference_tangent), axis=0,
    ) / np.maximum(np.linalg.norm(weighted_native, axis=0), 1e-12))
    if residual > 0.01:
        raise ValueError("charts do not share the declared harmonic mode blocks")

    # Fix the otherwise arbitrary partner orientation using ordered Cartesian
    # axes.  The pivot is chosen from the *subspace projector*, so rotating a
    # degenerate eigenspace before this step gives the same physical frame.
    canonical = np.eye(n)
    for block in (block for block in blocks if len(block) > 1):
        ix = np.asarray(block, dtype=int)
        tangent = weighted_reference[:, ix]
        orthonormal, _ = np.linalg.qr(tangent)
        projector = orthonormal @ orthonormal.T
        anchors: list[np.ndarray] = []
        for _ in block:
            scores = []
            for axis in range(size):
                vector = projector[:, axis].copy()
                for anchor in anchors:
                    vector -= anchor * np.dot(anchor, vector)
                scores.append(float(np.dot(vector, vector)))
            peak = max(scores)
            pivot = next(axis for axis, score in enumerate(scores)
                         if score >= peak - 1e-10)
            if scores[pivot] < 1e-10:
                raise ValueError("degenerate Cartesian anchor has deficient rank")
            vector = projector[:, pivot].copy()
            for anchor in anchors:
                vector -= anchor * np.dot(anchor, vector)
            anchors.append(vector / np.linalg.norm(vector))
        canonical_basis = np.column_stack(anchors)
        u, _, vt = np.linalg.svd(tangent, full_matrices=False)
        canonical[np.ix_(ix, ix)] = canonical_basis.T @ (u @ vt)
    if not np.allclose(canonical.T @ canonical, np.eye(n), atol=1e-5):
        raise ValueError("degenerate Cartesian anchor is not orthogonal")

    def mapped_native(q):
        return points(native_geometry_at_s, native, q)

    def mapped_reference(q):
        return points(reference_geometry_at_s, reference, linear @ q)

    projector = np.linalg.pinv(weights[:, None] * reference_tangent) * weights[None, :]
    connection = np.zeros((n, n, n))
    for j in range(n):
        for k in range(j, n):
            if j == k:
                difference = (
                    mapped_native(h * eye[j]) - 2 * origin_native
                    + mapped_native(-h * eye[j])
                    - mapped_reference(h * eye[j]) + 2 * origin_reference
                    - mapped_reference(-h * eye[j])
                ) / h**2
            else:
                def mixed(mapping):
                    return (
                        mapping(h * eye[j] + h * eye[k])
                        - mapping(h * eye[j] - h * eye[k])
                        - mapping(-h * eye[j] + h * eye[k])
                        + mapping(-h * eye[j] - h * eye[k])
                    ) / (4 * h**2)

                difference = mixed(mapped_native) - mixed(mapped_reference)
            connection[:, j, k] = connection[:, k, j] = projector @ difference
    if connection.shape != (n, n, n) or not np.all(np.isfinite(connection)):
        raise ValueError("quadratic chart connection is nonfinite")
    third = None
    if include_cubic:
        cubic_step = float(cubic_step_dimensionless)
        if not np.isfinite(cubic_step) or cubic_step <= 0:
            raise ValueError("cubic chart step must be positive and finite")
        origin = np.zeros(n)
        cache: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}

        def cached(mapping, label, coordinates):
            key = (label, tuple(float(value) for value in coordinates))
            if key not in cache:
                cache[key] = mapping(coordinates)
            return cache[key]

        def reference_at(coordinates):
            return cached(mapped_reference, "reference", coordinates)

        def difference_at(coordinates):
            return (cached(mapped_native, "native", coordinates)
                    - reference_at(coordinates))

        curvature = np.zeros((size, n, n))
        for j in range(n):
            for k in range(j, n):
                if j == k:
                    value = (reference_at(cubic_step * eye[j])
                             - 2 * reference_at(origin)
                             + reference_at(-cubic_step * eye[j])) / cubic_step**2
                else:
                    value = (
                        reference_at(cubic_step * (eye[j] + eye[k]))
                        - reference_at(cubic_step * (eye[j] - eye[k]))
                        - reference_at(cubic_step * (-eye[j] + eye[k]))
                        + reference_at(-cubic_step * (eye[j] + eye[k]))
                    ) / (4 * cubic_step**2)
                curvature[:, j, k] = curvature[:, k, j] = value

        def third_difference(j, k, l):
            counts = {index: (j, k, l).count(index) for index in {j, k, l}}
            if len(counts) == 1:
                direction = eye[j]
                return (
                    difference_at(2 * cubic_step * direction)
                    - 2 * difference_at(cubic_step * direction)
                    + 2 * difference_at(-cubic_step * direction)
                    - difference_at(-2 * cubic_step * direction)
                ) / (2 * cubic_step**3)
            if len(counts) == 2:
                repeated = next(index for index, count in counts.items() if count == 2)
                other = next(index for index, count in counts.items() if count == 1)

                def second_at(displacement):
                    return (difference_at(displacement + cubic_step * eye[repeated])
                            - 2 * difference_at(displacement)
                            + difference_at(displacement - cubic_step * eye[repeated]))

                return (second_at(cubic_step * eye[other])
                        - second_at(-cubic_step * eye[other])) / (2 * cubic_step**3)
            value = np.zeros(size)
            for signs3 in product((-1, 1), repeat=3):
                value += np.prod(signs3) * difference_at(cubic_step * (
                    signs3[0] * eye[j] + signs3[1] * eye[k] + signs3[2] * eye[l]
                ))
            return value / (8 * cubic_step**3)

        third = np.zeros((n, n, n, n))
        for j, k, l in combinations_with_replacement(range(n), 3):
            difference = third_difference(j, k, l)
            difference -= (curvature[:, j, :] @ connection[:, k, l]
                           + curvature[:, k, :] @ connection[:, j, l]
                           + curvature[:, l, :] @ connection[:, j, k])
            value = projector @ difference
            for ordered in set(permutations((j, k, l))):
                third[(slice(None), *ordered)] = value
        if not np.all(np.isfinite(third)):
            raise ValueError("cubic chart connection is nonfinite")
    return QuadraticChartConnection(connection, signs, overlap, linear,
                                    tuple(block for block in blocks if len(block) > 1),
                                    canonical, third)


def _inverse_chart_tensors(
    connection: QuadraticChartConnection, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second and third derivatives of native q with respect to reference q."""
    a = np.asarray(connection.second, dtype=float)
    b = None if connection.third is None else np.asarray(connection.third, dtype=float)
    linear = (np.asarray(connection.linear, dtype=float) if connection.linear is not None
              else np.diag(connection.reference_mode_signs))
    if (a.shape != (n,) * 3 or b is None or b.shape != (n,) * 4
            or linear.shape != (n, n)
            or not all(np.all(np.isfinite(value)) for value in (a, b, linear))
            or not np.allclose(linear.T @ linear, np.eye(n), atol=1e-7)):
        raise ValueError("second-order chart transport requires finite quadratic and cubic jets")
    inverse_second = -a
    inverse_third = (
        -b + np.einsum("iaj,jbc->iabc", a, a)
        + np.einsum("ibj,jac->iabc", a, a)
        + np.einsum("icj,jab->iabc", a, a)
    )
    return inverse_second, inverse_third, linear


def transport_sonic_property_second_order(
    derivatives: SonicPropertyDerivatives,
    connection: QuadraticChartConnection,
) -> SonicPropertyDerivatives:
    """Transform D1/D2/D3 by the full cubic inverse-chart chain rule."""
    first = np.asarray(derivatives.first, dtype=float)
    if first.ndim != 2:
        raise ValueError("property D1 must be components by modes")
    components, n = first.shape
    second = np.asarray(derivatives.second, dtype=float)
    third = np.asarray(derivatives.third, dtype=float)
    if (second.shape != (components, n, n) or third.shape != (components, n, n, n)
            or not all(np.all(np.isfinite(value)) for value in (first, second, third))):
        raise ValueError("second-order property transport requires finite D1/D2/D3")
    u2, u3, linear = _inverse_chart_tensors(connection, n)
    transformed_second = second + np.einsum("pi,iab->pab", first, u2)
    transformed_third = (
        third + np.einsum("pia,ibc->pabc", second, u2)
        + np.einsum("pib,iac->pabc", second, u2)
        + np.einsum("pic,iab->pabc", second, u2)
        + np.einsum("pi,iabc->pabc", first, u3)
    )
    return SonicPropertyDerivatives(
        first=np.einsum("pi,ai->pa", first, linear),
        second=np.einsum("pij,ai,bj->pab", transformed_second, linear, linear),
        third=np.einsum("pijk,ai,bj,ck->pabc", transformed_third,
                        linear, linear, linear, optimize=True),
    )


def transport_sonic_field_second_order(
    field: SonicVPT2Input,
    connection: QuadraticChartConnection,
) -> SonicVPT2Input:
    """Transform F3/F4 and G'/G'' into the reference normal-mode chart.

    At a stationary reference the inverse chart through cubic order is
    sufficient for the quartic potential. The contravariant metric follows
    from J G J^T through second order. No additional electronic points enter.
    """
    frequencies, _, f3, f4, g1, g2 = _modalize_sonic_field(field)
    n = len(frequencies)
    u2, u3, linear = _inverse_chart_tensors(connection, n)
    a = np.asarray(connection.second)
    b = np.asarray(connection.third)
    h0 = np.diag(frequencies)
    if np.max(abs(linear @ h0 @ linear.T - h0)) > 0.1:
        raise ValueError("chart rotation mixes nondegenerate harmonic modes")
    third = (
        f3 + np.einsum("ia,ibc->abc", h0, u2)
        + np.einsum("ib,iac->abc", h0, u2)
        + np.einsum("ic,iab->abc", h0, u2)
    )
    fourth = f4.copy()
    for axis in range(4):
        indices = tuple("abcd"[j] for j in range(4) if j != axis)
        fourth += np.einsum(f"i{'abcd'[axis]},i{''.join(indices)}->abcd", h0, u3)
    for pair in combinations(range(4), 2):
        remaining = tuple(index for index in range(4) if index not in pair)
        letters = "abcd"
        fourth += np.einsum(
            f"i{letters[remaining[0]]}{letters[remaining[1]]},"
            f"i{letters[pair[0]]}{letters[pair[1]]}->abcd", f3, u2,
        )
    for pair in ((0, 1), (0, 2), (0, 3)):
        remaining = tuple(index for index in range(4) if index not in pair)
        letters = "abcd"
        fourth += np.einsum(
            f"ij,i{letters[pair[0]]}{letters[pair[1]]},"
            f"j{letters[remaining[0]]}{letters[remaining[1]]}->abcd",
            h0, u2, u2,
        )
    jacobian_second = b + np.einsum("aim,mkl->aikl", a, u2)
    metric_second_native = g2 + np.einsum("ijm,mkl->ijkl", g1, u2)
    metric_first = (
        g1 + np.einsum("aik,ib->abk", a, h0)
        + np.einsum("aj,bjk->abk", h0, a)
    )
    metric_second = (
        metric_second_native
        + np.einsum("aikl,ib->abkl", jacobian_second, h0)
        + np.einsum("aj,bjkl->abkl", h0, jacobian_second)
        + np.einsum("aik,ibl->abkl", a, g1)
        + np.einsum("ail,ibk->abkl", a, g1)
        + np.einsum("ajk,bjl->abkl", g1, a)
        + np.einsum("ajl,bjk->abkl", g1, a)
        + np.einsum("aik,ij,bjl->abkl", a, h0, a)
        + np.einsum("ail,ij,bjk->abkl", a, h0, a)
    )
    f3_reference = np.einsum("ijk,ai,bj,ck->abc", third, linear, linear, linear,
                             optimize=True)
    f4_reference = np.einsum("ijkl,ai,bj,ck,dl->abcd", fourth,
                             linear, linear, linear, linear, optimize=True)
    g1_reference = np.einsum("ijk,ai,bj,ck->abc", metric_first,
                             linear, linear, linear, optimize=True)
    g2_reference = np.einsum("ijkl,ai,bj,ck,dl->abcd", metric_second,
                             linear, linear, linear, linear, optimize=True)
    gamma = np.sqrt(frequencies * CM_PER_HARTREE) / HESSIAN_EIGENVALUE_TO_CM
    metric_scale = HESSIAN_EIGENVALUE_TO_CM**2 / CM_PER_HARTREE
    return SonicVPT2Input(
        harmonic_sonic=np.diag((frequencies / HESSIAN_EIGENVALUE_TO_CM) ** 2),
        cubic_sonic=np.einsum("ijk,i,j,k->ijk", f3_reference,
                              gamma, gamma, gamma) / CM_PER_HARTREE,
        quartic_sonic=np.einsum("ijkl,i,j,k,l->ijkl", f4_reference,
                                gamma, gamma, gamma, gamma) / CM_PER_HARTREE,
        metric_sonic=np.eye(n),
        metric_first_sonic=np.einsum("ijk,k,i,j->ijk", g1_reference,
                                     gamma, 1 / gamma, 1 / gamma) / metric_scale,
        metric_second_sonic=np.einsum("ijkl,k,l,i,j->ijkl", g2_reference,
                                      gamma, gamma, 1 / gamma, 1 / gamma) / metric_scale,
        pseudopotential_cm=field.pseudopotential_cm,
        quartic_coverage=field.quartic_coverage,
    )


def _generator_element(
    second: np.ndarray, left: tuple[int, ...], right: tuple[int, ...],
) -> complex:
    """<left|S|right> for S = -i{f_i(q),p_i}/2, f_i=A_ijk q_j q_k/2."""
    value = 0j
    for i, j, k in np.argwhere(np.abs(second) > 1e-10):
        coefficient = -0.25j * second[i, j, k]
        value += coefficient * (
            _apply(right, (("q", int(j)), ("q", int(k)), ("p", int(i)))).get(left, 0j)
            + _apply(right, (("p", int(i)), ("q", int(j)), ("q", int(k)))).get(left, 0j)
        )
    return value


def transport_fermi_polyad(
    field: SonicVPT2Input,
    connection: QuadraticChartConnection,
    *, pairs: tuple[tuple[int, int], ...],
    max_quanta: int = 4,
    max_states: int = 300,
) -> TransportedFermiPolyad:
    """Build one Fermi-only polyad in the reference harmonic gauge.

    ``pairs`` are state indices of accepted 1:2 Fermi interactions in one
    connected polyad. The complete nonresonant VPT2 diagonal is used before
    removing these interactions in the common gauge. Darling--Dennison and
    other second-order off-diagonal terms require their own gauge transport.
    """
    if not pairs:
        raise ValueError("a transported Fermi polyad requires at least one pair")
    result = solve_sonic_gvpt2(
        field, max_quanta=max_quanta, max_states=max_states,
        resonance_policy="none",
    )
    frequencies, _, f3, _, g1, _ = _modalize_sonic_field(field)
    second = np.asarray(connection.second, dtype=float)
    if second.shape != (len(frequencies),) * 3:
        raise ValueError("connection dimension disagrees with the vibrational field")
    indices = tuple(sorted(set(i for pair in pairs for i in pair)))
    positions = {index: place for place, index in enumerate(indices)}
    local = np.diag(np.asarray(result.deperturbed_cm)[list(indices)].copy())
    corrections = []
    for a, b in pairs:
        if a not in positions or b not in positions or a == b:
            raise ValueError("invalid Fermi state pair")
        left, right = result.states[a], result.states[b]
        if sorted((sum(left), sum(right))) != [1, 2]:
            raise ValueError("coordinate-gauge transport currently supports 1:2 Fermi pairs")
        gap = float(np.dot(frequencies, np.subtract(left, right)))
        if abs(gap) < 1e-6:
            raise ValueError("zero harmonic Fermi denominator")
        native = _cubic_couplings(right, f3, g1, 0.0).get(left, 0j)
        generator = _generator_element(second, left, right)
        canonical = native - gap * generator
        if abs(native.imag) > 1e-8 or abs(canonical.imag) > 1e-8:
            raise ValueError("transported Fermi coupling must be real")
        coupling = float(canonical.real)
        ia, ib = positions[a], positions[b]
        local[ia, ia] -= coupling**2 / gap
        local[ib, ib] += coupling**2 / gap
        local[ia, ib] = local[ib, ia] = coupling
        corrections.append((a, b, float(native.real), coupling, float(generator.real)))
    eigenvalues, eigenvectors = np.linalg.eigh(local)
    return TransportedFermiPolyad(
        indices, local, eigenvalues, eigenvectors, tuple(corrections),
    )


def _generator_action(
    state: tuple[int, ...], second: np.ndarray,
) -> dict[tuple[int, ...], complex]:
    """Sparse action of the first-order quadratic-chart generator."""
    couplings: dict[tuple[int, ...], complex] = {}
    for i, j, k in np.argwhere(np.abs(second) > 1e-10):
        factor = -0.25j * second[i, j, k]
        for operators in (
            (("q", int(j)), ("q", int(k)), ("p", int(i))),
            (("p", int(i)), ("q", int(j)), ("q", int(k))),
        ):
            for target, value in _apply(state, operators).items():
                couplings[target] = couplings.get(target, 0j) + factor * value
    return {target: value for target, value in couplings.items() if abs(value) > 1e-12}


def _fock_support(
    mode_action: np.ndarray, source: tuple[int, ...],
) -> dict[tuple[int, ...], float]:
    """One sparse column of an orthogonal mode action in occupation space."""
    amplitudes: dict[tuple[int, ...], float] = {(0,) * len(source): 1.0}
    for mode, count in enumerate(source):
        for _ in range(count):
            updated: dict[tuple[int, ...], float] = {}
            for occupation, amplitude in amplitudes.items():
                for target_mode in np.flatnonzero(np.abs(mode_action[:, mode]) > 1e-12):
                    target = list(occupation)
                    target[target_mode] += 1
                    key = tuple(target)
                    updated[key] = updated.get(key, 0.0) + (
                        amplitude * mode_action[target_mode, mode]
                        * sqrt(target[target_mode])
                    )
            amplitudes = updated
    normalization = sqrt(np.prod([factorial(count) for count in source]))
    return {target: amplitude / normalization for target, amplitude in amplitudes.items()
            if abs(amplitude / normalization) > 1e-12}


def _fock_action(
    mode_action: np.ndarray, states: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, float]:
    """Lift any real orthogonal mode action to selected oscillator states."""
    index = {state: row for row, state in enumerate(states)}
    action = np.zeros((len(states), len(states)))
    escaped = 0.0
    for column, source in enumerate(states):
        for target, value in _fock_support(mode_action, source).items():
            if target in index:
                action[index[target], column] = value
            else:
                escaped = max(escaped, abs(value))
    return action, escaped


def _canonical_eigenvectors(
    roots: np.ndarray, vectors: np.ndarray, *, tolerance_cm: float = 1e-7,
) -> np.ndarray:
    """Fix phases and degenerate partner axes from ordered Fock projectors."""
    output = vectors.copy()
    start = 0
    while start < len(roots):
        stop = start + 1
        while stop < len(roots) and abs(roots[stop] - roots[start]) <= tolerance_cm:
            stop += 1
        projector = vectors[:, start:stop] @ vectors[:, start:stop].T
        anchors: list[np.ndarray] = []
        for _ in range(start, stop):
            scores = []
            for axis in range(len(roots)):
                column = projector[:, axis].copy()
                for anchor in anchors:
                    column -= anchor * np.dot(anchor, column)
                scores.append(float(np.dot(column, column)))
            peak = max(scores)
            pivot = next(axis for axis, score in enumerate(scores)
                         if score >= peak - 1e-10)
            if peak < 1e-10:
                raise ValueError("degenerate root projector has deficient rank")
            column = projector[:, pivot].copy()
            for anchor in anchors:
                column -= anchor * np.dot(anchor, column)
            anchors.append(column / np.linalg.norm(column))
        output[:, start:stop] = np.column_stack(anchors)
        start = stop
    return output


def transport_resonance_polyad(
    field: SonicVPT2Input,
    connection: QuadraticChartConnection,
    *,
    polyad_states: tuple[tuple[int, ...], ...],
    symmetry_actions: tuple[np.ndarray, ...],
    denominator_tolerance_cm: float = 1e-6,
    degenerate_gap_tolerance_cm: float = 0.1,
    closure_tolerance: float = 1e-7,
) -> TransportedResonancePolyad:
    """Transport cubic Fermi and second-order degenerate DD terms together.

    The quadratic coordinate generator transforms H1 and H2 through
    ``H1+[S,H0]`` and ``H2+[S,H1]+[S,[S,H0]]/2``. A third-order chart
    generator would add ``[T,H0]``; it vanishes between exactly degenerate
    partner states. Nondegenerate second-order polyad couplings are therefore
    rejected. The complete finite real point-group action is required so
    every degenerate manifold is closed and projected in the reference gauge.
    ``symmetry_actions`` act on the original reference normal modes; their
    order is immaterial. ``polyad_states`` must contain every symmetry partner
    and every occupation reached by the declared degenerate-block rotations.
    """
    frequencies, _, f3, f4, g1, g2 = _modalize_sonic_field(field)
    n = len(frequencies)
    states = tuple(sorted((tuple(int(v) for v in state) for state in polyad_states),
                          key=lambda state: (sum(state), state)))
    if (not states or len(set(states)) != len(states)
            or any(len(state) != n or any(v < 0 for v in state) for state in states)):
        raise ValueError("polyad requires distinct nonnegative occupation vectors")
    if not np.isfinite(denominator_tolerance_cm) or denominator_tolerance_cm <= 0:
        raise ValueError("invalid perturbative denominator tolerance")
    if (not np.isfinite(degenerate_gap_tolerance_cm)
            or degenerate_gap_tolerance_cm <= 0):
        raise ValueError("invalid degenerate-partner tolerance")
    second = np.asarray(connection.second, dtype=float)
    linear = (np.asarray(connection.linear, dtype=float) if connection.linear is not None
              else np.diag(connection.reference_mode_signs))
    canonical = (np.asarray(connection.canonical, dtype=float)
                 if connection.canonical is not None else np.eye(n))
    if (second.shape != (n, n, n) or linear.shape != (n, n)
            or canonical.shape != (n, n)
            or not np.all(np.isfinite(second)) or not np.all(np.isfinite(linear))
            or not np.all(np.isfinite(canonical))
            or not np.allclose(linear.T @ linear, np.eye(n), atol=closure_tolerance)
            or not np.allclose(canonical.T @ canonical, np.eye(n), atol=closure_tolerance)):
        raise ValueError("chart connection must have a finite orthogonal linear gauge")
    if not symmetry_actions:
        raise ValueError("a complete real point-group action is required")
    operations = tuple(np.asarray(action, dtype=float) for action in symmetry_actions)
    if any(action.shape != (n, n) or not np.all(np.isfinite(action))
           or not np.allclose(action.T @ action, np.eye(n), atol=closure_tolerance)
           for action in operations):
        raise ValueError("point-group mode actions must be real and orthogonal")
    if not any(np.allclose(action, np.eye(n), atol=closure_tolerance) for action in operations):
        raise ValueError("point-group action must contain the identity")
    if any(not any(np.allclose(a @ b, c, atol=closure_tolerance)
                   for c in operations) for a in operations for b in operations):
        raise ValueError("point-group mode actions are not closed")
    if any(np.allclose(a, b, atol=closure_tolerance)
           for index, a in enumerate(operations) for b in operations[index + 1:]):
        raise ValueError("point-group mode actions contain duplicates")
    nonabelian = any(not np.allclose(a @ b, b @ a, atol=closure_tolerance)
                     for a in operations for b in operations)
    if nonabelian and connection.canonical is None:
        raise ValueError("non-Abelian transport requires a Cartesian-anchored gauge")
    partner_block = np.arange(n)
    for block in connection.degenerate_blocks:
        if np.ptp(frequencies[list(block)]) > degenerate_gap_tolerance_cm:
            raise ValueError("declared partner block is not harmonically degenerate")
        partner_block[list(block)] = min(block)
    if any(abs(action[i, j]) > closure_tolerance and partner_block[i] != partner_block[j]
           for action in operations for i in range(n) for j in range(n)):
        raise ValueError("point-group partners require a declared degenerate block")

    zero = (0,) * n
    all_sources = (*states, zero)
    h1 = {state: _cubic_couplings(state, f3, g1, 0.0) for state in all_sources}
    generator = {state: _generator_action(state, second) for state in all_sources}

    def harmonic(state: tuple[int, ...]) -> float:
        return float(np.dot(frequencies, state))

    transported_h1 = {
        state: {
            target: h1[state].get(target, 0j)
            + (harmonic(state) - harmonic(target)) * generator[state].get(target, 0j)
            for target in h1[state].keys() | generator[state].keys()
        }
        for state in all_sources
    }

    def second_order(left: tuple[int, ...], right: tuple[int, ...]) -> complex:
        value = _operator_element(right, left, f3, f4, g1, g2, order=4)
        for middle in (generator[left].keys() | generator[right].keys()
                       | h1[left].keys() | h1[right].keys()):
            s_lm = -np.conj(generator[left].get(middle, 0j))
            s_mr = generator[right].get(middle, 0j)
            h_lm = np.conj(h1[left].get(middle, 0j))
            h_mr = h1[right].get(middle, 0j)
            value += s_lm * h_mr - h_lm * s_mr
            value += 0.5 * s_lm * s_mr * (
                harmonic(left) + harmonic(right) - 2 * harmonic(middle)
            )
        return value

    selected = set(states)

    def effective(left: tuple[int, ...], right: tuple[int, ...]) -> complex:
        value = (harmonic(left) if left == right else 0.0)
        value += transported_h1[right].get(left, 0j)
        value += second_order(left, right)
        for middle in transported_h1[left].keys() & transported_h1[right].keys():
            if middle in selected and (left != zero or right != zero):
                continue
            d_left = harmonic(left) - harmonic(middle)
            d_right = harmonic(right) - harmonic(middle)
            numerator = (np.conj(transported_h1[left][middle])
                         * transported_h1[right][middle])
            if abs(d_left) <= denominator_tolerance_cm or abs(d_right) <= denominator_tolerance_cm:
                if abs(numerator) > 1e-10:
                    raise ValueError("unresolved external resonance: enlarge the polyad")
                continue
            value += 0.5 * numerator * (1 / d_left + 1 / d_right)
        return value

    ground = effective(zero, zero)
    matrix = np.asarray([
        [effective(left, right) - (ground if left == right else 0.0)
         for right in states] for left in states
    ])
    if np.max(np.abs(matrix - matrix.conj().T)) > 1e-6 or np.max(abs(matrix.imag)) > 1e-6:
        raise ValueError("transported effective Hamiltonian is not real symmetric")
    matrix = matrix.real
    rotation, escape = _fock_action(canonical @ linear, states)
    if escape > closure_tolerance or not np.allclose(
        rotation.T @ rotation, np.eye(len(states)), atol=closure_tolerance,
    ):
        raise ValueError("polyad is not closed under the reference-gauge rotation")
    matrix = rotation @ matrix @ rotation.T
    group_actions = []
    for original in operations:
        action = canonical @ original @ canonical.T
        fock, leakage = _fock_action(action, states)
        escape = max(escape, leakage)
        if leakage > closure_tolerance or not np.allclose(
            fock.T @ fock, np.eye(len(states)), atol=closure_tolerance,
        ):
            raise ValueError("polyad does not contain complete point-group partners")
        group_actions.append(fock)
    projected = sum(action.T @ matrix @ action for action in group_actions) / len(group_actions)
    projected = (projected + projected.T) / 2
    invariance = max(float(np.max(abs(action.T @ projected @ action - projected)))
                     for action in group_actions)
    if invariance > closure_tolerance:
        raise ValueError("point-group projection did not produce an invariant polyad")
    for i, left in enumerate(states):
        for j, right in enumerate(states[:i]):
            if (sum(left) - sum(right)) % 2 != 0:
                continue
            if abs(harmonic(left) - harmonic(right)) > degenerate_gap_tolerance_cm:
                if abs(projected[i, j]) > degenerate_gap_tolerance_cm:
                    raise ValueError("nondegenerate DD transport requires a cubic chart connection")
    roots, vectors = np.linalg.eigh(projected)
    vectors = _canonical_eigenvectors(roots, vectors)
    return TransportedResonancePolyad(
        states, projected, roots, vectors, escape, invariance,
    )


def transport_gvpt2_polyads(
    result: SonicGVPT2Result,
    field: SonicVPT2Input,
    connection: QuadraticChartConnection,
    *,
    symmetry_actions: tuple[np.ndarray, ...],
) -> SonicGVPT2Result:
    """Close every selected PT2D block under the group and transport it.

    The returned blocks are mapped back to the native occupation basis so the
    existing PT2D property solver can mix its perturbatively dressed moments.
    The common Cartesian gauge and its invariance residual are retained in
    diagnostics; all unselected states preserve the original PT2D result.
    """
    states = result.states
    position = {state: index for index, state in enumerate(states)}
    selected: dict[int, frozenset[int]] = {}
    for block in result.blocks:
        members = frozenset(int(index) for index in block["indices"])
        for index in members:
            selected[index] = members
    if not selected:
        return result
    n = len(states[0])
    linear = (np.asarray(connection.linear, dtype=float) if connection.linear is not None
              else np.diag(connection.reference_mode_signs))
    canonical = (np.asarray(connection.canonical, dtype=float)
                 if connection.canonical is not None else np.eye(n))
    actions = (linear, canonical @ linear, *symmetry_actions)
    processed: set[int] = set()
    blocks: list[dict[str, object]] = []
    diagonal = np.asarray(result.deperturbed_cm, dtype=float).copy()
    energies = np.asarray(result.energies_cm, dtype=float).copy()
    max_leakage = 0.0
    max_invariance = 0.0
    for start in sorted(selected):
        if start in processed:
            continue
        component = set(selected[start])
        pending = list(component)
        while pending:
            index = pending.pop()
            for action in actions:
                for target, amplitude in _fock_support(action, states[index]).items():
                    if abs(amplitude) <= 1e-9:
                        continue
                    target_index = position.get(target)
                    if target_index is None:
                        raise ValueError("PT2D state basis omits a point-group partner")
                    for member in selected.get(target_index, frozenset((target_index,))):
                        if member not in component:
                            component.add(member)
                            pending.append(member)
        processed.update(component)
        members = tuple(sorted(component))
        transported = transport_resonance_polyad(
            field, connection, polyad_states=tuple(states[index] for index in members),
            symmetry_actions=symmetry_actions,
        )
        rotation, _ = _fock_action(canonical @ linear, transported.states)
        native_matrix = rotation.T @ transported.hamiltonian_cm @ rotation
        native_vectors = rotation.T @ transported.eigenvectors
        ordering = [transported.states.index(states[index]) for index in members]
        local = native_matrix[np.ix_(ordering, ordering)]
        vectors = native_vectors[ordering]
        roots = transported.eigenvalues_cm
        for row, index in enumerate(members):
            diagonal[index] = local[row, row]
            energies[index] = roots[int(np.argmax(abs(vectors[row]) ** 2))]
        blocks.append(dict(indices=members, hamiltonian_cm=local,
                           eigenvalues_cm=roots, eigenvectors=vectors))
        explicit_pairs = {
            (tuple(a), tuple(b))
            for old in result.blocks for a, b in old.get("direct_resonance_states", ())
            if position[tuple(a)] in component and position[tuple(b)] in component
        }
        if explicit_pairs:
            blocks[-1]["direct_resonance_states"] = tuple(sorted(explicit_pairs))
        max_leakage = max(max_leakage, transported.group_closure_residual)
        max_invariance = max(max_invariance,
                             transported.group_invariance_residual_cm)
    diagnostics = dict(result.diagnostics)
    diagnostics["common_gauge"] = dict(
        polyads=len(blocks), max_partner_leakage=max_leakage,
        max_group_invariance_residual_cm=max_invariance,
    )
    fundamentals = np.array([
        energies[position[tuple(int(j == mode) for j in range(n))]]
        for mode in range(n)
    ])
    return replace(result, blocks=tuple(blocks), deperturbed_cm=diagonal,
                   energies_cm=energies, fundamentals_cm=fundamentals,
                   diagnostics=diagnostics)


__all__ = [
    "QuadraticChartConnection", "CommonGaugeVP2TD", "TransportedFermiPolyad",
    "TransportedResonancePolyad", "quadratic_chart_connection",
    "transport_fermi_polyad", "transport_resonance_polyad",
    "transport_gvpt2_polyads", "transport_sonic_field_second_order",
    "transport_sonic_property_second_order", "solve_common_gauge_vp2td",
    "assess_chart_consistency", "assess_large_amplitude_dvr",
]
