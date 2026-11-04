"""Direct TRINITY GVPT2/polyad handoff to contracted LOCK rotation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from matrix_rovib import (
    ContractedRovibModel,
    ContractedVibrationalCoupling,
    ContractedVibrationalState,
    NativeRotorParameters,
    write_contracted_rovib_model,
)

CM_TO_MHZ = 29_979.2458


def build_lock_contracted_model(
    gvpt2: Any,
    rovibrational: Any,
    *,
    symmetries: Sequence[str] | None = None,
    dipole_debye: Sequence[float] = (0.0, 0.0, 0.0),
    distortion: NativeRotorParameters | None = None,
    representation: str = "Ir",
    coriolis_matrices_cm: Any = None,
    diabatic_rotational_constants_cm: Any = None,
    target_states: Sequence[int] | None = None,
    j_values: Sequence[int] | None = None,
) -> ContractedRovibModel:
    """Preserve the TRINITY diabatic polyad Hamiltonian for LOCK.

    The rotational constants of a basis state follow
    ``B(v) = B0 - sum_i alpha_i v_i``.  Scalar Fermi and
    Darling-Dennison matrix elements are copied from each GVPT2 block rather
    than rediagonalizing the vibrational problem before rotation is added.
    """
    states = tuple(tuple(int(value) for value in state) for state in gvpt2.states)
    alpha = np.asarray(rovibrational.alpha_cm, dtype=float)
    ground = np.asarray(rovibrational.ground_b_cm, dtype=float)
    if ground.shape != (3,) or alpha.shape != (len(gvpt2.frequencies_cm), 3):
        raise ValueError("contracted LOCK handoff currently requires a nonlinear three-axis rotor")
    if len(states) != len(gvpt2.deperturbed_cm):
        raise ValueError("GVPT2 state and deperturbed-energy counts differ")
    symmetry_labels = tuple("A" for _ in states) if symmetries is None else tuple(symmetries)
    if len(symmetry_labels) != len(states):
        raise ValueError("symmetries must contain one label per GVPT2 basis state")
    dipole = tuple(float(value) for value in dipole_debye)
    if len(dipole) != 3:
        raise ValueError("dipole_debye requires a,b,c components")
    quartic = (0.0,) * 5 if distortion is None else distortion.quartic_MHz
    sextic = (0.0,) * 7 if distortion is None else distortion.sextic_MHz
    octic = (0.0,) * 9 if distortion is None else distortion.octic_MHz
    reduction = "S" if distortion is None else distortion.reduction
    rep = representation if distortion is None else distortion.representation
    diabatic = None if diabatic_rotational_constants_cm is None else np.asarray(diabatic_rotational_constants_cm, float)
    if diabatic is not None:
        if diabatic.shape != (len(states), 3) or not np.all(np.isfinite(diabatic)):
            raise ValueError("one finite deperturbed rotational tensor diagonal per state required")
        if coriolis_matrices_cm is None:
            from matrix_trinity.contracted_coriolis import contract_modal_coriolis
            coriolis_matrices_cm = contract_modal_coriolis(states, rovibrational.metric_first_cm)
    if coriolis_matrices_cm is not None and diabatic is None:
        raise ValueError("explicit Coriolis requires diabatic rotational constants to avoid double counting alpha")

    contracted_states = []
    for index, quanta in enumerate(states):
        constants_cm = ground - np.asarray(quanta, dtype=float) @ alpha if diabatic is None else diabatic[index]
        if np.any(constants_cm <= 0.0):
            raise ValueError(f"nonpositive rotational constant for GVPT2 state {quanta}")
        label = "0" if not any(quanta) else "v=" + ",".join(str(value) for value in quanta)
        contracted_states.append(
            ContractedVibrationalState(
                label=label,
                energy_MHz=float(gvpt2.deperturbed_cm[index]) * CM_TO_MHZ,
                rotor=NativeRotorParameters(
                    *(float(value) * CM_TO_MHZ for value in constants_cm),
                    quartic_MHz=quartic,
                    sextic_MHz=sextic,
                    octic_MHz=octic,
                    reduction=reduction,
                    representation=rep,
                ),
                symmetry=str(symmetry_labels[index]),
                quanta=quanta,
                dipole_debye=dipole,
            )
        )

    coupling_values: dict[tuple[int, int], float] = {}
    for block in gvpt2.blocks:
        members = tuple(int(value) for value in block["indices"])
        hamiltonian = np.asarray(block["hamiltonian_cm"], dtype=float)
        if hamiltonian.shape != (len(members), len(members)):
            raise ValueError("GVPT2 polyad Hamiltonian has inconsistent dimensions")
        if (len(set(members)) != len(members) or any(i < 0 or i >= len(states) for i in members)
                or not np.all(np.isfinite(hamiltonian))
                or not np.allclose(hamiltonian, hamiltonian.T, rtol=1e-12, atol=1e-12)):
            raise ValueError("invalid Hermitian GVPT2 polyad")
        for row, left in enumerate(members):
            for column in range(row + 1, len(members)):
                right = members[column]
                value = float(hamiltonian[row, column])
                if abs(value) > 0.0:
                    key = (min(left, right), max(left, right))
                    if key in coupling_values and not np.isclose(coupling_values[key], value, rtol=1e-12, atol=1e-12):
                        raise ValueError("inconsistent overlapping GVPT2 polyads")
                    coupling_values[key] = value
    couplings = list(
        ContractedVibrationalCoupling(left, right, value * CM_TO_MHZ, "TRINITY GVPT2 polyad")
        for (left, right), value in sorted(coupling_values.items())
    )
    if coriolis_matrices_cm is not None:
        # Fully contracted coefficients of J_a, J_b, J_c in the SAME diabatic
        # basis and principal-axis frame as the supplied GVPT2 Hamiltonian.
        matrices = np.asarray(coriolis_matrices_cm, complex)
        if (matrices.shape != (3, len(states), len(states)) or not np.all(np.isfinite(matrices))
                or not np.allclose(matrices, matrices.conj().swapaxes(-1, -2), rtol=1e-12, atol=1e-12)):
            raise ValueError("Coriolis coefficients must be three Hermitian state matrices")
        for axis, matrix in enumerate(matrices):
            for left in range(len(states)):
                for right in range(left, len(states)):
                    value = matrix[left, right] * CM_TO_MHZ
                    if value:
                        couplings.append(ContractedVibrationalCoupling(left, right, value.real,
                            "TRINITY Coriolis", (axis,), value.imag))
    model = ContractedRovibModel(tuple(contracted_states), tuple(couplings))
    if target_states is not None:
        if j_values is None:
            raise ValueError("target manifold selection requires the production J range")
        from .contracted_rovib import retain_contracted_manifold
        model, _ = retain_contracted_manifold(model, target_states, j_values=j_values)
    return model


def write_lock_contracted_handoff(
    path: Path | str,
    gvpt2: Any,
    rovibrational: Any,
    **options,
) -> Path:
    return write_contracted_rovib_model(
        path, build_lock_contracted_model(gvpt2, rovibrational, **options)
    )


__all__ = ["CM_TO_MHZ", "build_lock_contracted_model", "write_lock_contracted_handoff"]
