"""APOC state densities contracted with backend-supplied EPOC modal integrals.

No population charges and no transition density approximate the state density.
The integral provider owns AO ordering, normalization and integral accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .state import ElectronicState


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _basis_frame(state):
    basis = None if state.basis is None else {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in vars(state.basis).items()
    }
    return _hash(dict(numbers=state.atomic_numbers.tolist(), geometry=state.coordinates_bohr.tolist(),
                      overlap=state.overlap_ao.tolist(), basis=basis, basis_label=state.basis_label))


@dataclass(frozen=True)
class ApocSolvationIntegrals:
    """I[m,mu,nu] = integral chi_mu(r) u_m(r) chi_nu(r) dr.

    ``u_m`` are the conductor's modal_values, in its exact order and frame.
    A bound integral artifact can be reused for states in the same AO space
    at the same geometry, including different electronic densities.
    """

    operator_sha256: str
    basis_frame_sha256: str
    geometry_id: str
    integrals: np.ndarray
    provenance: str

    def __post_init__(self):
        for name in ("operator_sha256", "basis_frame_sha256", "geometry_id", "provenance"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"explicit {name} required")
        if np.iscomplexobj(self.integrals):
            raise ValueError("real AO modal integrals required")
        values = np.asarray(self.integrals, dtype=float)
        if (values.ndim != 3 or not values.shape[0] or not values.shape[1]
                or values.shape[1] != values.shape[2] or not np.all(np.isfinite(values))
                or not np.allclose(values, values.swapaxes(1, 2), rtol=1e-10, atol=1e-12)):
            raise ValueError("finite symmetric (mode, AO, AO) integrals required")
        # Bytes-backed view cannot be made writable by the consumer.
        frozen = np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
        object.__setattr__(self, "integrals", frozen)

    @classmethod
    def bind(cls, model, state, integrals, *, provenance):
        if not isinstance(state, ElectronicState):
            raise TypeError("an APOC ElectronicState is required")
        result = cls(model.operator_sha256, _basis_frame(state),
                     model.geometry_identifier(state.coordinates_bohr), integrals, provenance)
        if result.integrals.shape != (len(model.conductor.modes), state.nao, state.nao):
            raise ValueError("AO modal integral dimensions do not match EPOC/APOC")
        return result


def apoc_solvation_source(model, state, operators, *, excited_state_id=None,
                          nuclear_charges_e=None):
    """Form total modal moments from a ground or excited state density.

    Excited sources require the APOC *difference* density, added to the
    ground-state density. ECP states require explicit effective nuclear charges;
    a mismatch with Tr(D S) is rejected rather than silently changing charge.
    This contracts an existing density; it performs no SCF relaxation.
    """
    if not isinstance(state, ElectronicState) or not isinstance(operators, ApocSolvationIntegrals):
        raise TypeError("APOC state and bound solvation integrals required")
    if (operators.operator_sha256 != model.operator_sha256
            or operators.basis_frame_sha256 != _basis_frame(state)
            or operators.geometry_id != model.geometry_identifier(state.coordinates_bohr)
            or operators.integrals.shape != (len(model.conductor.modes), state.nao, state.nao)):
        raise ValueError("AO ordering, geometry, basis or cavity differ from the integral artifact")
    density = sum(state.density_ao(i) for i in range(len(state.mo_coefficients)))
    label = state.label
    if excited_state_id is not None:
        if isinstance(excited_state_id, bool) or not isinstance(excited_state_id, int):
            raise ValueError("an integer APOC excited-state identifier is required")
        matches = [s for s in state.excited_states if s.state_id == excited_state_id]
        if len(matches) != 1 or matches[0].difference_density_ao is None:
            raise ValueError("one identified excited-state difference density is required")
        density = density + matches[0].difference_density_ao
        label = f"{state.label}/excited:{excited_state_id}"
    if (np.iscomplexobj(density) or not np.all(np.isfinite(density))
            or not np.allclose(density, density.T, atol=1e-10, rtol=1e-10)):
        raise ValueError("a finite real symmetric state density is required")
    nuclei = state.atomic_numbers if nuclear_charges_e is None else nuclear_charges_e
    if np.iscomplexobj(nuclei):
        raise ValueError("real effective nuclear charges required")
    nuclei = np.asarray(nuclei, dtype=float)
    if (nuclei.shape != state.atomic_numbers.shape or not np.all(np.isfinite(nuclei))
            or np.any(nuclei < 0) or np.any(nuclei > state.atomic_numbers)):
        raise ValueError("one effective nuclear charge per atom is required")
    electrons = float(np.einsum("ij,ji->", density, state.overlap_ao))
    if not np.isclose(nuclei.sum() - electrons, state.charge, atol=2e-6, rtol=0):
        raise ValueError("density electron count and nuclear charges disagree with molecular charge")
    moments = model.conductor.modal_moments(state.coordinates_bohr, nuclei)
    moments -= np.einsum("ij,mji->m", density, operators.integrals)
    source_id = _hash(dict(basis_frame=operators.basis_frame_sha256, state=label,
                           density=density.tolist(), nuclear_charges=nuclei.tolist(),
                           integrals=operators.integrals.tolist(), provenance=operators.provenance))
    return model.source(moments, state_id=label, source_id="apoc:" + source_id,
                        geometry_id=model.geometry_identifier(state.coordinates_bohr))


def solvation_fock_ao(model, source, frozen, operators):
    """Electronic one-body reaction operator, dF/dD, for an external SCF loop.

    This is minus the positive-test-charge potential, not the polarization
    free energy. The SCF owner must retain the corresponding work correction.
    """
    if (not isinstance(operators, ApocSolvationIntegrals) or operators.operator_sha256 != model.operator_sha256
            or operators.geometry_id != source.geometry_id
            or operators.integrals.shape[0] != len(model.conductor.modes)):
        raise ValueError("bound AO operators for this cavity are required")
    result = model.evaluate(source, frozen)
    return -np.einsum("m,mij->ij", result.reaction_coefficients, operators.integrals)


__all__ = ["ApocSolvationIntegrals", "apoc_solvation_source", "solvation_fock_ao"]
