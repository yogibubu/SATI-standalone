"""SATI FC/HT transitions between completed SAM Hamiltonians, including hot bands."""

import numpy as np
from threadpoolctl import threadpool_limits
from .hdc_hamiltonian import solve_sam_hdcpt2, hdc_transition_spectrum, sam_thermal_populations
from .vibronic_harmonic import orient_harmonic_model, harmonic_gaussian_generator
from .vibronic_kernels import parallel_rows, worker_count
from .vibronic_observables import _operator_rows
from .vibronic_selection import PackedHermiteCache
from .vibronic_vpt2 import _positive_integer


def hdc_vibronic_ti(
    model,
    initial_field,
    final_field,
    *,
    final_levels,
    initial_levels=(0,),
    initial_populations=None,
    temperature_K=None,
    kind="absorption",
    operator="FCHT",
    maximum_total_quanta=8,
    maximum_states=4096,
    maximum_dense_bytes=512 * 1024**2,
    mechanical_scale=1.0,
    alpha=1.0,
    beta_cm2=5.0e5,
    mixing_weight=None,
    allow_unbounded_quartic=False,
    formulation="normal_form",
    regularization_width_cm=100.0,
    maximum_harmonic_pairs=1000000,
    maximum_integrals=1000000,
    integral_batch_size=256,
    workers=0,
):
    """Same Gaussian integrals, two unitary SAM transformations; no coefficient pruning.

    Final/initial levels are ENERGY-ORDERED EIGENLEVEL IDs, not occupations.
    Explicit populations are absolute and not thermally renormalized. Emission
    swaps both fields and the harmonic model consistently. Results keep their
    finite-model scope; no harmonic omission certificate is reused.
    """
    for name, value in dict(
        maximum_harmonic_pairs=maximum_harmonic_pairs,
        maximum_integrals=maximum_integrals,
        integral_batch_size=integral_batch_size,
    ).items():
        _positive_integer(value, name)
    if operator not in {"FC", "HT", "FCHT"}:
        raise ValueError("operator must be FC, HT or FCHT")
    for name, field, wi in (
        ("initial", initial_field, model.initial_frequencies_cm),
        ("final", final_field, model.final_frequencies_cm),
    ):
        if not np.array_equal(field.frequencies_cm, wi):
            raise ValueError("SAM frequencies/order must match the SATI harmonic basis exactly")
        expected = model.provenance.get(name + "_normal_basis_sha256")
        if field.basis_id and expected and field.basis_id != expected:
            raise ValueError(f"{name} SAM modal gauge does not match SATI")
    workers = worker_count(workers)
    controls = dict(
        maximum_total_quanta=maximum_total_quanta,
        maximum_states=maximum_states,
        maximum_dense_bytes=maximum_dense_bytes,
        mechanical_scale=mechanical_scale,
        alpha=alpha,
        beta_cm2=beta_cm2,
        mixing_weight=mixing_weight,
        allow_unbounded_quartic=allow_unbounded_quartic,
        formulation=formulation,
        regularization_width_cm=regularization_width_cm,
        workers=workers,
    )
    if kind == "emission":
        initial_field, final_field = final_field, initial_field
    model, direction = orient_harmonic_model(model, kind)
    initial = solve_sam_hdcpt2(initial_field, **controls)
    final = solve_sam_hdcpt2(final_field, **controls)

    # Validate selected levels/populations before any expensive overlap work.
    def selected(value, n, name):
        a = np.asarray(value)
        if (
            a.ndim != 1
            or not len(a)
            or a.dtype.kind not in "iu"
            or np.any(a < 0)
            or np.any(a >= n)
            or len(np.unique(a)) != len(a)
        ):
            raise ValueError(f"{name} must contain unique valid eigenlevel indices")
        return a

    selected(initial_levels, len(initial.energies_cm), "initial_levels")
    selected(final_levels, len(final.energies_cm), "final_levels")
    if temperature_K is not None:
        if initial_populations is not None:
            raise ValueError("choose temperature_K or explicit populations, not both")
        initial_populations = sam_thermal_populations(initial, temperature_K)[
            np.asarray(initial_levels)
        ]
    population = None if initial_populations is None else np.asarray(initial_populations, float)
    if population is None:
        if not np.array_equal(initial_levels, [0]):
            raise ValueError("hot bands require explicit absolute initial populations")
    elif (
        population.shape != (len(initial_levels),)
        or not np.all(np.isfinite(population))
        or np.any(population < 0)
        or population.sum() > 1 + 1.0e-12
    ):
        raise ValueError("absolute nonnegative populations summing to at most one required")
    ni, nf = len(initial.basis), len(final.basis)
    count = ni * nf
    if count > maximum_harmonic_pairs:
        raise MemoryError("completed SAM harmonic-pair budget exceeded before overlaps")
    if count * 16 * (3 + 2 * len(model.initial_frequencies_cm)) > maximum_dense_bytes:
        raise MemoryError("completed SAM overlap working-set budget exceeded")
    generator = harmonic_gaussian_generator(model)
    if generator["log_ground"] < np.log(np.finfo(float).tiny) + 4:
        raise FloatingPointError("harmonic overlap underflows; log-scaled recurrences required")
    constant = (model.transition_dipole_au if operator != "HT" else np.zeros(3)) + (
        generator["ht_constant"] if operator != "FC" else np.zeros(3)
    )
    linear = generator["ht_linear"] if operator != "FC" else np.zeros_like(generator["ht_linear"])
    cache = PackedHermiteCache(
        generator,
        linear,
        maximum_integrals,
        max(1, int(max(initial.basis.max(), final.basis.max()))),
    )
    # Flatten with final-basis index first, matching O(final, initial, component).
    moments = np.empty((count, 3), complex)
    for start in range(0, count, integral_batch_size):
        flat = np.arange(start, min(start + integral_batch_size, count))
        states = np.concatenate((initial.basis[flat % ni], final.basis[flat // ni]), axis=1)
        ids, exhausted = cache.evaluate(states)
        if exhausted:
            raise MemoryError("SAM harmonic overlap cache exhausted; no partial spectrum returned")
        parallel_rows(
            _operator_rows,
            len(flat),
            workers,
            states,
            ids,
            cache.values,
            constant,
            linear,
            moments[start : start + len(flat)],
        )
    with threadpool_limits(limits=workers, user_api="blas"):
        result = hdc_transition_spectrum(
            initial,
            final,
            moments.reshape(nf, ni, 3),
            initial_levels=initial_levels,
            final_levels=final_levels,
            initial_populations=initial_populations,
        )
    result["schema"] = "matrix.sati.hdc_vibronic_ti.v1"
    result["wavenumbers_cm"] = direction * (model.electronic_gap_cm + result["wavenumbers_cm"])
    result["kind"] = kind
    result["operator"] = operator
    result["initial_solver_diagnostics"] = initial.diagnostics
    result["final_solver_diagnostics"] = final.diagnostics
    result["initial_energies_cm"] = initial.energies_cm
    result["final_energies_cm"] = final.energies_cm
    result["diagnostics"].update(
        harmonic_pairs=count,
        cached_integrals=cache.count,
        workers=workers,
        harmonic_completeness_certificate_used=False,
        amplitude_pruning_used=False,
        strength_units="supplied FC/HT operator atomic units squared",
        population_model="finite_completed_hamiltonian_boltzmann"
        if temperature_K is not None
        else "explicit_absolute",
        temperature_K=temperature_K,
    )
    result["provenance"] = model.provenance
    return result


__all__ = ["hdc_vibronic_ti"]
