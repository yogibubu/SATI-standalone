"""Reconstruct asymptotic ZAFF corrections from independent internal stencils.

SMITH realizes the nonredundant coordinates; LINK supplies Cartesian QM
derivatives. ARCHITECT matches differences of residual Cartesian Hessians to
the same differences of native ZAFF functions. This is an implicit nonlinear
derivative transformation on the physical manifold, not a pseudoinverse of a
redundant primitive tensor. TRINITY remains responsible for spectroscopy.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import lstsq
from scipy.linalg.lapack import get_lapack_funcs

from .asymptotic_pic import AsymptoticPicCorrection, asymptotic_pic_correction_design
from .asymptotic_pic_hessian import asymptotic_pic_correction_hessian_design
from .fit_stencil import paired_residual_blocks


def reconstructed_zaff_evaluator(field, primitives, reconstruction, baseline_evaluator):
    """Compose the frozen native correction with the full preceding ZAFF PES.

    The returned callback is directly usable by TRINITY's Cartesian acquisition
    driver. Both callbacks receive angstrom coordinates; output gradients and
    Hessians use bohr. ``baseline_evaluator`` must evaluate the preceding ZAFF
    surface, including nonbonded interactions, not launch new QM calculations.
    No dipole or other transition property is inferred from the PES.
    """
    from matrix_chem import BOHR_TO_ANGSTROM

    coefficients = np.array([t.coefficient for t in reconstruction.corrections])
    factors = tuple(t.factors for t in reconstruction.corrections)

    @dataclass(frozen=True)
    class Point:
        energy_hartree: float
        gradient_hartree_per_bohr: np.ndarray
        hessian_hartree_per_bohr2: np.ndarray

    def evaluate(xyz):
        base = baseline_evaluator(xyz)
        h = np.asarray(base.hessian_hartree_per_bohr2, dtype=float)
        g = np.asarray(base.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        if h.shape != (np.size(xyz),) * 2 or g.shape != (np.size(xyz),):
            raise ValueError("baseline ZAFF derivative dimensions differ from geometry")
        if (
            not np.isfinite(base.energy_hartree)
            or not np.all(np.isfinite(h))
            or not np.all(np.isfinite(g))
        ):
            raise ValueError("baseline ZAFF returned nonfinite energy or derivatives")
        if not factors:
            return Point(float(base.energy_hartree), g.copy(), (h + h.T) / 2)
        e, dg = asymptotic_pic_correction_design(field, primitives, xyz, factors)
        dh = asymptotic_pic_correction_hessian_design(field, primitives, xyz, factors)
        total_h = h + (dh @ coefficients) * BOHR_TO_ANGSTROM**2
        return Point(
            float(base.energy_hartree + e @ coefficients),
            g + (dg @ coefficients) * BOHR_TO_ANGSTROM,
            (total_h + total_h.T) / 2,
        )

    return evaluate


@dataclass(frozen=True)
class EsalciHessianStencil:
    """Canonical order: reference, +s1, -s1, +s2, -s2, ... .

    Hessians are Cartesian Eh/angstrom², already aligned to the same atom
    order and frame. The baseline includes the full preceding ZAFF field,
    electrostatics and vdW. The reference Hessian is not altered.
    Optional pair reference Hessians are independently measured numerical
    branch limits, used only as centres of even differences. They never
    replace the reference harmonic input or the stored displaced Hessians.
    """

    coordinate_system: str
    coordinates_angstrom: np.ndarray
    half_steps: np.ndarray
    qm_hessians: np.ndarray
    baseline_hessians: np.ndarray
    qm_pair_reference_hessians: np.ndarray | None = None


def esalci_tangent_projection(stencil: EsalciHessianStencil) -> np.ndarray:
    """Return dx/ds for the frozen ESALC-I paths from their central pairs.

    The columns use angstrom per native ESALC-I displacement unit.  The
    production reconstruction normally retains every component of the
    analytic Cartesian Hessian and differences it only along these paths.
    When a reduced projection is explicitly requested, this is the only
    admitted projection; Cartesian normal modes cannot replace it.
    """
    if stencil.coordinate_system != "ESALC-I":
        raise ValueError("ZAFF3 Hessian reconstruction requires ESALC-I paths")
    coordinates = np.asarray(stencil.coordinates_angstrom, dtype=float)
    steps = np.asarray(stencil.half_steps, dtype=float)
    if (
        coordinates.ndim != 3
        or coordinates.shape[2] != 3
        or steps.ndim != 1
        or len(coordinates) != 1 + 2 * len(steps)
        or np.any(steps <= 0)
        or not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(steps))
    ):
        raise ValueError("invalid ESALC-I Hessian stencil")
    pairs = coordinates[1:].reshape(len(steps), 2, -1)
    tangent = ((pairs[:, 0] - pairs[:, 1]) / (2 * steps[:, None])).T
    if np.linalg.matrix_rank(tangent) != tangent.shape[1]:
        raise ValueError("ESALC-I displacement tangents are rank deficient")
    return tangent


@dataclass(frozen=True)
class NativeResidualPointObservations:
    """Stationary-reference QM minus full baseline, in Eh and Eh/angstrom.

    Energies are relative to the same central geometry as the Hessian stencil.
    These rows supply global LAM information not determined by local Hessians.
    """

    coordinates_angstrom: np.ndarray
    residual_energies_hartree: np.ndarray
    residual_gradients_hartree_per_angstrom: np.ndarray
    energy_tolerance_hartree: float
    gradient_tolerance_hartree_per_angstrom: float


@dataclass(frozen=True)
class NativeHessianReconstruction:
    corrections: tuple[AsymptoticPicCorrection, ...]
    coefficients: np.ndarray
    resolved_rank: int
    candidate_count: int
    removed_terms: tuple[int, ...]
    normalized_residual_before_cut: float
    normalized_residual_after_cut: float
    truncation_bound: float
    local_derivatives_pass: bool
    reference_hessian_residual: float
    normalized_term_contributions: np.ndarray | None = None


def _independent_scaled_fit(design, target, relative_rank_cutoff, least_squares_driver):
    """Return a basic least-squares solution on a conditioned pivoted basis."""
    gelsy, gelsy_lwork = get_lapack_funcs(("gelsy", "gelsy_lwork"), (design,))
    rows, columns = design.shape
    workspace, info = gelsy_lwork(rows, columns, 1, relative_rank_cutoff)
    if info != 0:
        raise np.linalg.LinAlgError("rank-revealing workspace query failed")
    rhs = np.zeros((max(rows, columns), 1))
    rhs[:rows, 0] = target
    _, _, pivots, rank, info = gelsy(
        np.array(design, order="F"),
        rhs,
        np.zeros(columns, np.int32),
        relative_rank_cutoff,
        int(workspace),
        overwrite_a=True,
        overwrite_b=True,
    )
    if info != 0:
        raise np.linalg.LinAlgError("rank-revealing native-term selection failed")
    active = pivots[:rank] - 1
    selected, _, selected_rank, _ = lstsq(
        design[:, active],
        target,
        cond=relative_rank_cutoff,
        lapack_driver=least_squares_driver,
    )
    if selected_rank != rank:
        raise np.linalg.LinAlgError("selected native basis did not preserve numerical rank")
    coefficients = np.zeros(columns)
    coefficients[active] = selected
    return coefficients, int(rank), active


def acquire_esalci_hessian_stencil(
    smith_coordinates,
    half_steps: np.ndarray,
    *,
    qm_hessian_at_geometry: Callable[[np.ndarray], np.ndarray],
    baseline_hessian_at_geometry: Callable[[np.ndarray], np.ndarray],
) -> EsalciHessianStencil:
    """Acquire only centre and single-coordinate pairs using the SMITH provider.

    The callbacks can read caches or use an authorized LINK evaluator. No
    backend, symmetry, torsional classification or job launcher is recreated.
    Callers must supply a nonredundant SMITH definition, not the redundant
    field primitives. Steps have the native units of each frozen ESALC-I.
    """
    steps = np.asarray(half_steps, dtype=float)
    n = int(smith_coordinates.definition.rank)
    if len(smith_coordinates.definition.gics) != n:
        raise ValueError("acquisition requires nonredundant ESALC-I coordinates")
    if steps.shape != (n,) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("one finite positive step is required per ESALC-I coordinate")
    displacements = [np.zeros(n)]
    for i, step in enumerate(steps):
        for sign in (1, -1):
            s = np.zeros(n)
            s[i] = sign * step
            displacements.append(s)
    geometries, qm, baseline = [], [], []
    for s in displacements:
        xyz = smith_coordinates.geometry_at_displacement_angstrom(s)
        geometries.append(xyz)
        matrices = []
        for callback in (qm_hessian_at_geometry, baseline_hessian_at_geometry):
            h = np.asarray(callback(xyz.copy()), dtype=float)
            if h.shape != (xyz.size, xyz.size) or not np.all(np.isfinite(h)):
                raise ValueError("invalid Cartesian Hessian from acquisition provider")
            matrices.append((h + h.T) / 2)
        qm.append(matrices[0])
        baseline.append(matrices[1])
    return EsalciHessianStencil(
        "ESALC-I", np.array(geometries), steps.copy(), np.array(qm), np.array(baseline)
    )


def reconstruct_native_from_esalci_hessians(
    field,
    primitives,
    stencil: EsalciHessianStencil,
    factors: Sequence,
    *,
    odd_tolerance: float,
    even_tolerance: float,
    reference_hessian_tolerance: float,
    truncation_budget: float = 0.0,
    coefficient_scales: np.ndarray | None = None,
    relative_rank_cutoff: float = 1e-8,
    maximum_matrix_bytes: int = 512 * 1024**2,
    point_observations: NativeResidualPointObservations | None = None,
    least_squares_driver: str = "gelsd",
    hessian_projection: np.ndarray | None = None,
    independent_terms_only: bool = False,
    truncation_eligible: np.ndarray | None = None,
    maximum_retained_eligible: int | None = None,
    coefficient_prior: np.ndarray | None = None,
    coefficient_prior_strength: float = 0.0,
) -> NativeHessianReconstruction:
    """Fit observable third/fourth information, with correct tails term by term.

    Only factors with zero reference E/G/H are admitted, preserving ZAFF2.
    Unit corrections include the full native functions, switches and B prime.
    Cartesian Hessians are differenced along the independent internal paths;
    no linear-only tensor transformation is used. Missing four-distinct-index
    derivatives are not supplied as zero observations.

    Rank-revealing SVD or pivoted QR selects the minimum scaled coefficient
    norm in redundant directions.
    Unobserved directions receive zero weight in the solution, not invented
    QM measurements. Coefficient scales are fixed priors, not column norms.
    Tolerances apply to odd/even Hessian differences in Eh/angstrom² per
    internal-coordinate unit and its square, respectively. An optional
    full-column-rank ``hessian_projection`` must span the actual ESALC-I path
    tangents.  Only its orientation-independent Cartesian projector is used:
    observations and models become Pi @ H @ Pi.  Rotating the ESALC-I basis
    therefore leaves every fitted row unchanged.  The central Hessian and
    zero-reference-jet checks remain Cartesian and unprojected.

    Terms selected by the optional ``truncation_eligible`` mask are removed
    only while their summed maximum contributions stay within
    ``truncation_budget`` in tolerance-normalized observation units.
    ``maximum_retained_eligible`` can impose a hard family-size ceiling; the
    largest-contribution members are retained and all surviving coefficients
    are refitted, so the reduced field is not a zeroed full-fit expansion.
    Whole terms are removed, so periodicity and dissociation switches survive.
    An optional quadratic coefficient prior stabilizes a reduced refit around
    an already certified field. Its strength acts in scaled-coefficient space
    and prevents rotations through near-null primitive combinations.
    A local pass is not a global PES or spectroscopic certificate. Optional
    residual E/G observations constrain LAM scans in the same solve; no
    frequencies are used as targets. Independently verified numerical branch
    centres, when supplied, affect only the observed even differences. With
    ``independent_terms_only``, pivoted QR retains one deterministic native
    basis for the observable column space. This removes redundant terms before
    the bounded physical-contribution cut without changing that column space.
    """
    if least_squares_driver not in {"gelsd", "gelsy"}:
        raise ValueError("unsupported rank-revealing least-squares driver")
    x = np.asarray(stencil.coordinates_angstrom, dtype=float)
    steps = np.asarray(stencil.half_steps, dtype=float)
    qm = np.asarray(stencil.qm_hessians, dtype=float)
    baseline = np.asarray(stencil.baseline_hessians, dtype=float)
    factors = tuple(tuple(t) for t in factors)
    if stencil.coordinate_system != "ESALC-I":
        raise ValueError("ZAFF3 Hessian reconstruction requires ESALC-I paths")
    if (
        x.ndim != 3
        or x.shape[2] != 3
        or steps.ndim != 1
        or not len(steps)
        or len(x) != 1 + 2 * len(steps)
        or np.any(steps <= 0)
        or qm.shape != (len(x), x[0].size, x[0].size)
        or baseline.shape != qm.shape
        or not factors
        or not all(np.all(np.isfinite(a)) for a in (x, steps, qm, baseline))
    ):
        raise ValueError("invalid ESALC-I Hessian stencil")
    limits = (odd_tolerance, even_tolerance, reference_hessian_tolerance)
    if (
        not all(np.isfinite(t) and t > 0 for t in limits)
        or not np.isfinite(truncation_budget)
        or truncation_budget < 0
        or not np.isfinite(relative_rank_cutoff)
        or not 0 < relative_rank_cutoff < 1
    ):
        raise ValueError("invalid reconstruction tolerances")
    scales = (
        np.ones(len(factors))
        if coefficient_scales is None
        else np.asarray(coefficient_scales, float)
    )
    if scales.shape != (len(factors),) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("positive finite coefficient scales are required")
    eligible = (
        np.ones(len(factors), dtype=bool)
        if truncation_eligible is None
        else np.asarray(truncation_eligible, dtype=bool)
    )
    if eligible.shape != (len(factors),):
        raise ValueError("truncation eligibility must match the candidate terms")
    if maximum_retained_eligible is not None and (
        isinstance(maximum_retained_eligible, bool)
        or not isinstance(maximum_retained_eligible, (int, np.integer))
        or maximum_retained_eligible < 0
    ):
        raise ValueError("maximum retained eligible terms must be a nonnegative integer")
    prior = None if coefficient_prior is None else np.asarray(coefficient_prior, float)
    if (
        not np.isfinite(coefficient_prior_strength)
        or coefficient_prior_strength < 0
        or (prior is None) != (coefficient_prior_strength == 0)
        or (
            prior is not None and (prior.shape != (len(factors),) or not np.all(np.isfinite(prior)))
        )
    ):
        raise ValueError("coefficient prior requires matching finite values and positive strength")
    # All inputs are symmetrized before use, including saved acquisitions.
    residual = (qm + qm.swapaxes(1, 2) - baseline - baseline.swapaxes(1, 2)) / 2
    reference_error = float(np.max(abs(residual[0])))
    if reference_error > reference_hessian_tolerance:
        raise ValueError("ZAFF2 must reproduce the reference Hessian before reconstruction")
    tangent_projector = None
    if hessian_projection is not None:
        projection = np.asarray(hessian_projection, float)
        expected_projection = esalci_tangent_projection(stencil)
        if (
            projection.ndim != 2
            or projection.shape[0] != x[0].size
            or projection.shape[1] < 1
            or projection.shape[1] > x[0].size
            or not np.all(np.isfinite(projection))
            or np.linalg.matrix_rank(projection) != projection.shape[1]
        ):
            raise ValueError("only the actual ESALC-I path tangent may project Hessians")
        expected_q, _ = np.linalg.qr(expected_projection, mode="reduced")
        supplied_q, _ = np.linalg.qr(projection, mode="reduced")
        tangent_projector = expected_q @ expected_q.T
        supplied_projector = supplied_q @ supplied_q.T
        if not np.allclose(
            supplied_projector, tangent_projector, atol=1e-10, rtol=1e-8,
        ):
            raise ValueError("only the actual ESALC-I tangent space may project Hessians")
        residual = np.einsum(
            "ai,nij,jb->nab", tangent_projector, residual, tangent_projector,
            optimize=True,
        )
    upper = np.triu_indices(residual.shape[1])
    weights = np.where(upper[0] == upper[1], 1.0, np.sqrt(2.0))
    rows_per_point = len(upper[0])
    # Guard the dense rank solve; fail explicitly before allocating its arrays.
    extra_rows = 0
    if point_observations is not None:
        obs = point_observations
        px = np.asarray(obs.coordinates_angstrom, float)
        pe = np.asarray(obs.residual_energies_hartree, float)
        pg = np.asarray(obs.residual_gradients_hartree_per_angstrom, float)
        et = float(obs.energy_tolerance_hartree)
        gt = float(obs.gradient_tolerance_hartree_per_angstrom)
        if (
            px.ndim != 3
            or px.shape[1:] != x.shape[1:]
            or pe.shape != (len(px),)
            or pg.shape != (len(px), x[0].size)
            or not all(np.all(np.isfinite(t)) for t in (px, pe, pg))
            or not np.isfinite(et)
            or not np.isfinite(gt)
            or et <= 0
            or gt <= 0
        ):
            raise ValueError("invalid residual energy/gradient observations")
        extra_rows = len(px) * (1 + x[0].size)
    estimate = 8 * (len(x) * rows_per_point + extra_rows) * len(factors) * 6
    if estimate > maximum_matrix_bytes:
        raise ValueError("candidate block exceeds reconstruction memory budget; split the supports")
    designs = []
    for i, xyz in enumerate(x):
        h = asymptotic_pic_correction_hessian_design(field, primitives, xyz, factors)
        if i == 0:
            e, g = asymptotic_pic_correction_design(field, primitives, xyz, factors)
            if max(np.max(abs(e)), np.max(abs(g)), np.max(abs(h))) > 1e-10:
                raise ValueError(
                    "native corrections must preserve reference energy, gradient and Hessian"
                )
        if tangent_projector is not None:
            h = np.einsum(
                "ai,ijk,jb->abk", tangent_projector, h, tangent_projector,
                optimize=True,
            )
        designs.append(h[upper] * weights[:, None])
    packed = residual[:, upper[0], upper[1]] * weights
    pairs = np.arange(1, len(x)).reshape(-1, 2)
    blocks = paired_residual_blocks(
        np.array(designs), packed, center=0, pairs=pairs, half_steps=steps
    )
    even_target = blocks.even_target
    if stencil.qm_pair_reference_hessians is not None:
        centers = np.asarray(stencil.qm_pair_reference_hessians, float)
        if centers.shape != (len(steps), x[0].size, x[0].size) or not np.all(np.isfinite(centers)):
            raise ValueError("invalid independently measured QM pair reference Hessians")
        centers = (centers + centers.swapaxes(1, 2)) / 2
        centre_shift = centers - (qm[0] + qm[0].T) / 2
        if tangent_projector is not None:
            centre_shift = np.einsum(
                "ai,nij,jb->nab", tangent_projector, centre_shift, tangent_projector,
                optimize=True,
            )
        shift = centre_shift[:, upper[0], upper[1]] * weights
        even_target = even_target - 2 * shift / steps[:, None] ** 2
    a = np.concatenate(
        (
            blocks.odd_design.reshape(-1, len(factors)) / odd_tolerance,
            blocks.even_design.reshape(-1, len(factors)) / even_tolerance,
        )
    )
    y = np.r_[blocks.odd_target.ravel() / odd_tolerance, even_target.ravel() / even_tolerance]
    if point_observations is not None:
        energies, gradients = [], []
        center_e, _ = asymptotic_pic_correction_design(field, primitives, x[0], factors)
        for point in px:
            energy, gradient = asymptotic_pic_correction_design(field, primitives, point, factors)
            energies.append(energy - center_e)
            gradients.append(gradient)
        a = np.concatenate(
            (a, np.asarray(energies) / et, np.asarray(gradients).reshape(-1, len(factors)) / gt)
        )
        y = np.r_[y, pe / et, pg.ravel() / gt]
    observation_target = y
    scaled_design = a * scales
    if prior is not None:
        weight = np.sqrt(coefficient_prior_strength)
        scaled_design = np.concatenate((scaled_design, weight * np.eye(len(factors))))
        y = np.r_[y, weight * prior / scales]
    if independent_terms_only:
        scaled, rank, _ = _independent_scaled_fit(
            scaled_design,
            y,
            relative_rank_cutoff,
            least_squares_driver,
        )
    else:
        scaled, _, rank, _ = lstsq(
            scaled_design, y, cond=relative_rank_cutoff, lapack_driver=least_squares_driver
        )
    coefficients = scaled * scales
    forced_removed = []
    if maximum_retained_eligible is not None:
        initial_contribution = np.max(abs(a), axis=0) * abs(coefficients)
        active = coefficients != 0.0
        eligible_active = np.flatnonzero(eligible & active)
        excess = len(eligible_active) - int(maximum_retained_eligible)
        if excess > 0:
            order = eligible_active[
                np.argsort(initial_contribution[eligible_active], kind="stable")
            ]
            forced_removed = [int(index) for index in order[:excess]]
            active[forced_removed] = False
            active_indices = np.flatnonzero(active)
            reduced, rank, reduced_active = _independent_scaled_fit(
                scaled_design[:, active_indices],
                y,
                relative_rank_cutoff,
                least_squares_driver,
            )
            scaled = np.zeros(len(factors))
            scaled[active_indices] = reduced
            active[:] = False
            active[active_indices[reduced_active]] = True
            coefficients = scaled * scales
    before = float(np.max(abs(a @ coefficients - observation_target)))
    contribution = np.max(abs(a), axis=0) * abs(coefficients)
    removed, bound = forced_removed, 0.0
    forced_set = set(forced_removed)
    for index in np.argsort(contribution, kind="stable"):
        if not eligible[index] or index in forced_set:
            continue
        if bound + contribution[index] > truncation_budget:
            break
        bound += float(contribution[index])
        removed.append(int(index))
        coefficients[index] = 0.0
    after = float(np.max(abs(a @ coefficients - observation_target)))
    corrections = tuple(
        AsymptoticPicCorrection(float(c), term)
        for c, term in zip(coefficients, factors, strict=True)
        if c != 0.0
    )
    return NativeHessianReconstruction(
        corrections,
        coefficients,
        int(rank),
        len(factors),
        tuple(removed),
        before,
        after,
        bound,
        after <= 1.0,
        reference_error,
        contribution,
    )
