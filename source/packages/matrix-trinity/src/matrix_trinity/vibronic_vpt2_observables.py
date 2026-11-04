"""Hermitian observable banks on existing two-surface SAM wavefunctions."""

from dataclasses import replace
import numpy as np
from .vibronic_observables import _bank
from .vibronic_vpt2 import vpt2_vibronic_ti


def vpt2_operator_spectrum(
    model,
    initial_field,
    final_field,
    constants,
    forms,
    *,
    gradients=None,
    kind="absorption",
    **controls,
):
    """Evaluate arbitrary FC/linear-HT banks with strict mechanical order two.

    Reuses the native three-component VPT2 contraction in bounded blocks of
    three polynomial operators. Signed observable forms and strict perturbative
    strengths are retained. Positive completed squares are separate diagnostics;
    no renormalization, clipping or completeness certificate is introduced.
    """
    if kind != "absorption":
        raise ValueError("The VPT2 operator bank currently requires absorption orientation")
    _, _, _, _, w, _, _, _ = _bank(model, constants, gradients, forms, kind)
    c = np.asarray(constants, complex)
    h = (
        np.zeros((len(c), len(model.initial_frequencies_cm)), complex)
        if gradients is None
        else np.asarray(gradients, complex)
    )
    if "operator" in controls:
        raise ValueError("The explicit operator bank owns FC/linear-HT coefficients")
    blocks = []
    result = None
    for first in range(0, len(c), 3):
        count = min(3, len(c) - first)
        mu = np.zeros(3, complex)
        grad = np.zeros((3, h.shape[1]), complex)
        mu[:count] = c[first : first + count]
        grad[:count] = h[first : first + count]
        active = replace(
            model,
            transition_dipole_au=mu,
            transition_gradient_dimensionless_au=grad,
            transition_momentum_dimensionless_au=np.zeros_like(grad),
        )
        current = vpt2_vibronic_ti(
            active, initial_field, final_field, kind=kind, operator="FCHT", **controls
        )
        if result is not None:
            if not np.array_equal(
                current["photon_energy_orders_cm"], result["photon_energy_orders_cm"]
            ):
                raise RuntimeError("Inconsistent nuclear states across operator blocks")
        else:
            result = current
        blocks.append(current["amplitude_orders_au"][:, :, :count])
    amplitude = np.concatenate(blocks, axis=-1)
    orders = np.zeros((len(w), len(amplitude), 3))
    for order in range(3):
        for left in range(order + 1):
            orders[:, :, order] += np.einsum(
                "ta,jab,tb->jt",
                amplitude[:, left],
                w,
                amplitude[:, order - left].conj(),
                optimize=True,
            ).real
    completed = amplitude.sum(axis=1)
    positive = np.einsum("ta,jab,tb->jt", completed, w, completed.conj(), optimize=True).real
    return dict(
        schema="matrix.sati.vpt2_operator_spectrum.v1",
        kind=kind,
        wavenumbers_cm=result["wavenumbers_cm"],
        photon_energy_orders_cm=result["photon_energy_orders_cm"],
        initial_quanta=result["initial_quanta"],
        final_quanta=result["final_quanta"],
        population=result["population"],
        amplitude_orders=amplitude,
        strength_orders=orders,
        strengths=orders.sum(axis=-1),
        weighted_strengths=orders.sum(axis=-1) * result["population"][None, :],
        completed_strengths=positive,
        initial_state_diagnostics=result["initial_state_diagnostics"],
        final_state_diagnostics=result["final_state_diagnostics"],
        diagnostics={
            **{
                key: value
                for key, value in result["diagnostics"].items()
                if key
                not in {
                    "negative_truncated_strengths",
                    "maximum_positive_completion_difference_au2",
                }
            },
            "negative_truncated_observable_counts": np.count_nonzero(
                orders.sum(axis=-1) < 0, axis=1
            ),
            "maximum_completed_difference_by_observable": np.max(
                abs(positive - orders.sum(axis=-1)), axis=1
            ),
            "operator_count": len(c),
            "operator_blocks": len(blocks),
            "signed_observable_forms": True,
        },
    )


__all__ = ["vpt2_operator_spectrum"]
