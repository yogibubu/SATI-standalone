"""Conditional radiative branching and correlated local uncertainty reports."""

import numpy as np
from .spectroscopic_inference import propagate_covariance


def conditional_branching(energies_cm, pairs, rates_s_inverse):
    """Normalize outgoing rates on the explicitly supplied endpoint inventory.

    Pair orientation is arbitrary. Dark sources have zero branch probabilities;
    no complete lifetime is inferred from a conditional endpoint list.
    """
    energy = np.asarray(energies_cm, float)
    pairs = np.asarray(pairs, int)
    rate = np.asarray(rates_s_inverse, float)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or rate.shape != (len(pairs),)
        or np.any(rate < 0)
        or not np.all(np.isfinite(rate))
    ):
        raise ValueError("matched finite nonnegative rates and endpoint pairs required")
    if len({tuple(sorted(p)) for p in pairs}) != len(pairs):
        raise ValueError("duplicate unordered endpoint pair")
    source = np.where(energy[pairs[:, 0]] >= energy[pairs[:, 1]], pairs[:, 0], pairs[:, 1])
    totals = np.bincount(source, weights=rate, minlength=len(energy))
    probability = np.divide(rate, totals[source], out=np.zeros_like(rate), where=totals[source] > 0)
    return dict(
        source_indices=source,
        source_total_selected_rate_s_inverse=totals,
        branching_probabilities=probability,
        scope="conditional_on_supplied_endpoints",
    )


def thermal_net_absorption_ratio(energies_cm, pairs, rates_s_inverse, hot, cold, temperature_K):
    """Resolved integrated hot/cold absorption ratio with population difference.

    Statistical multiplet sums must be resolved or summed explicitly before
    comparison. No experimental line-area ratio is treated as an Einstein-A
    ratio without these photon and population factors.
    """
    from scipy.constants import h, c, k

    energy = np.asarray(energies_cm, float)
    pairs = np.asarray(pairs, int)
    rate = np.asarray(rates_s_inverse, float)
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("positive finite absorption temperature required")
    gap = abs(energy[pairs[:, 0]] - energy[pairs[:, 1]])
    lower = np.minimum(energy[pairs[:, 0]], energy[pairs[:, 1]])
    if gap[hot] <= 0 or gap[cold] <= 0 or rate[cold] <= 0:
        raise ValueError("positive-frequency channels and a positive cold reference rate required")
    beta = h * c * 100 / (k * temperature_K)
    ratio = (
        rate[hot]
        / rate[cold]
        * (gap[cold] / gap[hot]) ** 2
        * np.exp(-beta * (lower[hot] - lower[cold]))
        * np.expm1(-beta * gap[hot])
        / np.expm1(-beta * gap[cold])
    )
    return float(ratio)


def infer_local_observable_uncertainty(center, plus, minus, steps, parameter_covariance):
    """Preserve the complete observable covariance, including cross correlations."""
    mean = np.asarray(center, float)
    p, m = np.asarray(plus, float), np.asarray(minus, float)
    step = np.asarray(steps, float)
    if (
        mean.ndim != 1
        or p.shape != m.shape
        or p.shape != (len(step), len(mean))
        or np.any(step <= 0)
    ):
        raise ValueError("one matched central difference per parameter required")
    jacobian = ((p - m) / (2 * step[:, None])).T
    propagated = propagate_covariance(jacobian, parameter_covariance)
    return dict(
        mean=mean,
        jacobian=jacobian,
        **propagated,
        scope="local_delta_method_conditional_on_reference_model_and_assignment",
    )


def transition_phase_factors(pairs, columns, reference_columns):
    """Transport endpoint phases before differentiating complex amplitudes.

    ``reference_columns`` maps resolved state indices to common reference
    vectors. A lost overlap is an assignment failure, not a phase convention.
    Degenerate subspaces require a separately transported resolved basis.
    """
    pairs = np.asarray(pairs, int)
    v = np.asarray(columns)
    phases = {}
    for index in np.unique(pairs):
        if int(index) not in reference_columns:
            raise ValueError("every amplitude endpoint requires a common reference column")
        anchor = np.vdot(np.asarray(reference_columns[int(index)]), v[:, index])
        if abs(anchor) <= 1e-8:
            raise ValueError(
                "lost state overlap; transport the state/subspace before differentiating"
            )
        phases[int(index)] = np.conj(anchor) / abs(anchor)
    return np.asarray([phases[int(a)] * np.conj(phases[int(b)]) for a, b in pairs])


def propagate_amplitude_quadrature(
    energies_cm,
    moments_au,
    pairs,
    energy_jacobian,
    moment_jacobian,
    parameter_covariance,
    *,
    order=5,
    maximum_nodes=10000,
):
    """Positive radiative observables from a correlated local amplitude model.

    Energies and complex moments are affine in the fit parameters. Gaussian
    quadrature then applies the exact photon cube, modulus square and outgoing
    normalization, retaining dark-state borrowing and rate/branch covariance.
    This is a local field approximation, not a new molecular calculation at
    each quadrature node. Discrete assignments remain separate hypotheses.
    """
    from itertools import product
    from numpy.polynomial.hermite import hermgauss
    from .spectroscopic_inference import electric_dipole_rates

    e = np.asarray(energies_cm, float)
    m = np.asarray(moments_au, complex)
    je = np.asarray(energy_jacobian, float)
    jm = np.asarray(moment_jacobian, complex)
    cov = np.asarray(parameter_covariance, float)
    dimension = cov.shape[0]
    if (
        cov.shape != (dimension, dimension)
        or je.shape != (len(e), dimension)
        or jm.shape != (*m.shape, dimension)
        or order**dimension > maximum_nodes
        or order < 2
    ):
        raise ValueError("matched energy/moment Jacobians and a bounded quadrature required")
    if not np.allclose(cov, cov.T) or np.min(np.linalg.eigvalsh(cov)) < 0:
        raise ValueError("positive semidefinite parameter covariance required")
    eigenvalues, rotation = np.linalg.eigh(cov)
    root = rotation * np.sqrt(np.maximum(0, eigenvalues))
    x, w = hermgauss(order)
    x = np.sqrt(2) * x
    w = w / np.sqrt(np.pi)
    values = []
    weights = []
    for selection in product(range(order), repeat=dimension):
        delta = root @ x[list(selection)]
        energy = e + je @ delta
        amplitude = m + np.einsum("ack,k->ac", jm, delta)
        orders = np.zeros((len(pairs), 3, 3), complex)
        orders[:, 0] = amplitude
        rate = electric_dipole_rates(energy, pairs, orders)
        branch = conditional_branching(energy, pairs, rate["rates_s_inverse"])
        values.append(
            np.r_[rate["gaps_cm"], rate["rates_s_inverse"], branch["branching_probabilities"]]
        )
        weights.append(float(np.prod(w[list(selection)])))
    values = np.asarray(values)
    weights = np.asarray(weights)
    weights /= weights.sum()
    mean = weights @ values
    centered = values - mean
    covariance = centered.T @ (weights[:, None] * centered)
    return dict(
        mean=mean,
        covariance=covariance,
        standard_deviations=np.sqrt(np.maximum(0, np.diag(covariance))),
        order=order,
        nodes=len(weights),
        parameter_count=dimension,
        scope="conditional correlated Gaussian parameters; locally affine energies and phase-aligned moments; exact positive rate and branch transformation",
    )


def transport_result_eigenframes(result, reference_columns, *, degeneracy_tolerance_cm=1e-8):
    """One-to-one overlap transport, including complete degenerate frames.

    Nondegenerate columns receive a phase. A polar factor aligns each complete
    numerical-degeneracy cluster, including partners in disconnected blocks.
    Physical eigenvalues and integer state-counting labels remain distinct.
    """
    from dataclasses import replace
    from scipy.optimize import linear_sum_assignment
    from .spectroscopic_inference import assigned_columns

    original = assigned_columns(result)
    reference = np.asarray(reference_columns)
    if reference.shape != original.shape or not np.all(np.isfinite(reference)):
        raise ValueError("a complete finite reference eigenframe is required")
    count = len(original)
    rr, cc = linear_sum_assignment(-(abs(reference.conj().T @ original) ** 2))
    permutation = np.empty(count, int)
    permutation[rr] = cc
    columns = original[:, permutation].copy()
    energy = np.asarray(result.energies_cm)[permutation].copy()
    ordering = np.argsort(energy, kind="stable")
    clusters = []
    current = []
    for index in ordering:
        if current and energy[index] - energy[current[0]] > degeneracy_tolerance_cm:
            clusters.append(current)
            current = []
        current.append(int(index))
    if current:
        clusters.append(current)
    minimum_singular = 1.0
    maximum_cluster_size = 1
    for cluster in clusters:
        overlap = columns[:, cluster].conj().T @ reference[:, cluster]
        u, singular, vh = np.linalg.svd(overlap)
        if np.min(singular) <= 1e-8:
            raise ValueError("reference and current eigenframes lost subspace overlap")
        columns[:, cluster] = columns[:, cluster] @ (u @ vh)
        minimum_singular = min(minimum_singular, float(np.min(singular)))
        maximum_cluster_size = max(maximum_cluster_size, len(cluster))
    owner = np.arange(count)
    for block_index, block in enumerate(result.blocks):
        owner[np.asarray(block["indices"])] = count + block_index
    cross_block = any(np.any(columns[owner != owner[i], i] != 0) for i in range(count))
    origin = result.zero_point_cm - float(np.sum(result.frequencies_cm) / 2)
    h = np.diag(result.deperturbed_cm + origin).astype(complex)
    for block in result.blocks:
        ix = np.asarray(block["indices"])
        h[np.ix_(ix, ix)] = block["hamiltonian_cm"]
    residual = float(np.linalg.norm(h @ columns - columns * (energy + origin)[None, :]))
    if residual > 2 * degeneracy_tolerance_cm * np.sqrt(count):
        raise ValueError("transported frame exceeds the declared eigenvalue-degeneracy precision")
    ground = result.states.index((0,) * len(result.frequencies_cm))
    ground_shift = float(energy[ground])
    energy -= ground_shift
    if cross_block:
        blocks = (
            dict(
                indices=list(range(count)),
                hamiltonian_cm=h,
                eigenvalues_cm=energy.copy(),
                eigenvectors=columns,
                basis_to_root=list(range(count)),
            ),
        )
    else:
        blocks = []
        for block in result.blocks:
            ix = np.asarray(block["indices"])
            blocks.append(
                dict(
                    block,
                    eigenvectors=columns[np.ix_(ix, ix)],
                    eigenvalues_cm=energy[ix],
                    basis_to_root=list(range(len(ix))),
                )
            )
        blocks = tuple(blocks)
    fundamentals = np.array(
        [
            energy[result.states.index(state)] if state in result.states else np.nan
            for state in (
                tuple(int(i == j) for i in range(len(result.frequencies_cm)))
                for j in range(len(result.frequencies_cm))
            )
        ]
    )
    diagnostics = dict(
        minimum_frame_singular_value=minimum_singular,
        maximum_degenerate_cluster_size=maximum_cluster_size,
        transported_eigenvector_residual_cm=residual,
        cross_block_frame=cross_block,
        reassigned_columns=int(np.count_nonzero(permutation != np.arange(count))),
    )
    return replace(
        result,
        blocks=blocks,
        energies_cm=energy,
        deperturbed_cm=result.deperturbed_cm - ground_shift,
        zero_point_cm=result.zero_point_cm + ground_shift,
        fundamentals_cm=fundamentals,
    ), diagnostics
