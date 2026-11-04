"""TREXIO persistence for the native APOC electronic-state contract."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from matrix_chem.topology.elements import atomic_symbol

from .state import (
    APOC_ELECTRONIC_STATE_SCHEMA,
    ElectronicState,
    ExcitedState,
    GaussianBasis,
)


def write_electronic_state_trexio(
    path: Path | str,
    state: ElectronicState,
    *,
    text_backend: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write one complete APOC state to a TREXIO HDF5 or text container."""

    trexio = _trexio()
    target = Path(path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    backend = trexio.TREXIO_TEXT if text_backend else trexio.TREXIO_HDF5
    excited = tuple(sorted(state.excited_states, key=lambda item: item.state_id))
    if excited and state.spin_channels != 1:
        raise ValueError(
            "APOC TREXIO v1 stores TD transition densities for restricted states; "
            "spin-resolved excited-state densities require an explicit alpha/beta contract"
        )
    metadata = {
        "schema": APOC_ELECTRONIC_STATE_SCHEMA,
        "channel_sizes": [int(item.shape[1]) for item in state.mo_coefficients],
        "charge": int(state.charge),
        "multiplicity": int(state.multiplicity),
        "label": state.label,
        "method": state.method,
        "basis_label": state.basis_label,
        "source": state.source,
        "excited_states": [
            {
                "state_id": item.state_id,
                "label": item.label,
                "symmetry": item.symmetry,
                "energy_hartree": item.energy_hartree,
                "oscillator_strength": item.oscillator_strength,
            }
            for item in excited
        ],
    }
    with trexio.File(str(target), "w", backend) as handle:
        trexio.write_metadata_code_num(handle, 1)
        trexio.write_metadata_code(handle, ["APOC"])
        trexio.write_metadata_description(handle, json.dumps(metadata, sort_keys=True))
        trexio.write_nucleus_num(handle, state.natoms)
        trexio.write_nucleus_charge(handle, state.atomic_numbers.astype(float))
        trexio.write_nucleus_coord(handle, state.coordinates_bohr)
        trexio.write_nucleus_label(
            handle,
            [atomic_symbol(int(number)) for number in state.atomic_numbers],
        )
        nalpha, nbeta = _electron_spin_counts(state)
        trexio.write_electron_up_num(handle, nalpha)
        trexio.write_electron_dn_num(handle, nbeta)
        if state.basis is not None:
            _write_basis(trexio, handle, state.basis)
        trexio.write_ao_num(handle, state.nao)
        trexio.write_ao_cartesian(handle, int(state.basis.cartesian) if state.basis else 0)
        if state.basis is not None:
            trexio.write_ao_shell(handle, state.basis.ao_shell)
            trexio.write_ao_normalization(handle, state.basis.ao_normalization)
        trexio.write_ao_1e_int_overlap(handle, state.overlap_ao)

        coefficients = np.concatenate(state.mo_coefficients, axis=1)
        occupations = np.concatenate(state.mo_occupations)
        energies = (
            np.concatenate(state.mo_energies_hartree)
            if state.mo_energies_hartree
            else np.zeros(coefficients.shape[1], dtype=float)
        )
        spins = np.concatenate(
            [np.full(item.shape[1], index, dtype=int) for index, item in enumerate(state.mo_coefficients)]
        )
        trexio.write_mo_num(handle, coefficients.shape[1])
        trexio.write_mo_coefficient(handle, coefficients.T)
        trexio.write_mo_occupation(handle, occupations)
        trexio.write_mo_energy(handle, energies)
        trexio.write_mo_spin(handle, spins)
        trexio.write_mo_type(handle, "APOC canonical/natural orbital space")

        labels = [state.label or "S0"] + [item.label or f"S{item.state_id}" for item in excited]
        trexio.write_state_num(handle, len(labels))
        trexio.write_state_id(handle, 0)
        trexio.write_state_label(handle, labels)
        trexio.write_state_current_label(handle, labels[0])
        if state.total_energy_hartree is not None:
            trexio.write_state_energy(handle, float(state.total_energy_hartree))

        ground_mo = _block_mo_density(state)
        trexio.write_rdm_1e(handle, ground_mo)
        if excited:
            transition = np.zeros(
                (len(labels), len(labels), coefficients.shape[1], coefficients.shape[1]),
                dtype=float,
            )
            transition[0, 0] = ground_mo
            coefficient = state.mo_coefficients[0]
            for target_index, item in enumerate(excited, start=1):
                transition_mo = _ao_to_mo(item.transition_density_ao, coefficient, state.overlap_ao)
                transition[0, target_index] = np.real_if_close(transition_mo)
                transition[target_index, 0] = np.real_if_close(transition_mo.conj().T)
                if item.difference_density_ao is not None:
                    difference_mo = _ao_to_mo(
                        item.difference_density_ao,
                        coefficient,
                        state.overlap_ao,
                    )
                    transition[target_index, target_index] = np.real_if_close(
                        ground_mo + difference_mo
                    )
            trexio.write_rdm_1e_transition(handle, transition)
    return target


def read_electronic_state_trexio(path: Path | str) -> ElectronicState:
    """Restore an APOC electronic state from a TREXIO container."""

    trexio = _trexio()
    source = Path(path)
    backend = trexio.TREXIO_TEXT if source.is_dir() else trexio.TREXIO_HDF5
    with trexio.File(str(source), "r", backend) as handle:
        metadata = _read_metadata(trexio, handle)
        numbers = np.rint(np.asarray(trexio.read_nucleus_charge(handle), dtype=float)).astype(int)
        coordinates = np.asarray(trexio.read_nucleus_coord(handle), dtype=float)
        overlap = np.asarray(trexio.read_ao_1e_int_overlap(handle), dtype=float)
        coefficients_all = np.asarray(trexio.read_mo_coefficient(handle), dtype=float).T
        occupations_all = np.asarray(trexio.read_mo_occupation(handle), dtype=float)
        energies_all = (
            np.asarray(trexio.read_mo_energy(handle), dtype=float)
            if trexio.has_mo_energy(handle)
            else np.zeros(coefficients_all.shape[1], dtype=float)
        )
        channel_sizes = tuple(int(value) for value in metadata.get("channel_sizes", ()))
        if not channel_sizes:
            spins = (
                np.asarray(trexio.read_mo_spin(handle), dtype=int)
                if trexio.has_mo_spin(handle)
                else np.zeros(coefficients_all.shape[1], dtype=int)
            )
            channel_sizes = tuple(int(np.sum(spins == spin)) for spin in sorted(set(spins.tolist())))
        if sum(channel_sizes) != coefficients_all.shape[1] or len(channel_sizes) not in {1, 2}:
            raise ValueError("TREXIO MO spin/channel metadata are inconsistent")
        coefficients = _split_columns(coefficients_all, channel_sizes)
        occupations = _split_vector(occupations_all, channel_sizes)
        energies = _split_vector(energies_all, channel_sizes)
        rdm_mo = (
            np.asarray(trexio.read_rdm_1e(handle), dtype=float)
            if trexio.has_rdm_1e(handle)
            else np.diag(occupations_all)
        )
        densities = _block_ao_densities(rdm_mo, coefficients, channel_sizes)
        basis = _read_basis(trexio, handle) if trexio.has_basis(handle) else None
        excited = _read_excited_states(
            trexio,
            handle,
            metadata,
            coefficients,
            rdm_mo,
        )
        total_energy = (
            float(trexio.read_state_energy(handle)) if trexio.has_state_energy(handle) else None
        )
        nalpha = int(trexio.read_electron_up_num(handle))
        nbeta = int(trexio.read_electron_dn_num(handle))
    charge = int(metadata.get("charge", round(float(np.sum(numbers) - nalpha - nbeta))))
    multiplicity = int(metadata.get("multiplicity", abs(nalpha - nbeta) + 1))
    return ElectronicState(
        atomic_numbers=numbers,
        coordinates_bohr=coordinates,
        overlap_ao=overlap,
        mo_coefficients=coefficients,
        mo_occupations=occupations,
        mo_energies_hartree=energies,
        density_matrices_ao=densities,
        excited_states=excited,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        total_energy_hartree=total_energy,
        label=str(metadata.get("label", "S0")),
        method=str(metadata.get("method", "")),
        basis_label=str(metadata.get("basis_label", "")),
        source=str(metadata.get("source", source)),
    )


def _write_basis(trexio, handle, basis: GaussianBasis) -> None:
    trexio.write_basis_type(handle, "Gaussian")
    trexio.write_basis_shell_num(handle, basis.shell_count)
    trexio.write_basis_prim_num(handle, basis.primitive_count)
    trexio.write_basis_nucleus_index(handle, basis.shell_nucleus_index)
    trexio.write_basis_shell_ang_mom(handle, basis.shell_angular_momentum)
    trexio.write_basis_shell_factor(handle, basis.shell_factor)
    trexio.write_basis_r_power(handle, basis.shell_r_power)
    trexio.write_basis_shell_index(handle, basis.primitive_shell_index)
    trexio.write_basis_exponent(handle, basis.primitive_exponent)
    trexio.write_basis_coefficient(handle, basis.primitive_coefficient)
    trexio.write_basis_prim_factor(handle, basis.primitive_factor)


def _read_basis(trexio, handle) -> GaussianBasis:
    return GaussianBasis(
        cartesian=bool(trexio.read_ao_cartesian(handle)),
        shell_nucleus_index=np.asarray(trexio.read_basis_nucleus_index(handle), dtype=int),
        shell_angular_momentum=np.asarray(trexio.read_basis_shell_ang_mom(handle), dtype=int),
        shell_factor=np.asarray(trexio.read_basis_shell_factor(handle), dtype=float),
        shell_r_power=np.asarray(trexio.read_basis_r_power(handle), dtype=int),
        primitive_shell_index=np.asarray(trexio.read_basis_shell_index(handle), dtype=int),
        primitive_exponent=np.asarray(trexio.read_basis_exponent(handle), dtype=float),
        primitive_coefficient=np.asarray(trexio.read_basis_coefficient(handle), dtype=float),
        primitive_factor=np.asarray(trexio.read_basis_prim_factor(handle), dtype=float),
        ao_shell=np.asarray(trexio.read_ao_shell(handle), dtype=int),
        ao_normalization=np.asarray(trexio.read_ao_normalization(handle), dtype=float),
    )


def _read_metadata(trexio, handle) -> dict[str, object]:
    if not trexio.has_metadata_description(handle):
        return {}
    try:
        value = json.loads(trexio.read_metadata_description(handle))
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _electron_spin_counts(state: ElectronicState) -> tuple[int, int]:
    if state.spin_channels == 2:
        return tuple(int(round(np.sum(item))) for item in state.mo_occupations)  # type: ignore[return-value]
    electrons = int(round(state.electron_count))
    difference = int(state.multiplicity) - 1
    if (electrons + difference) % 2:
        raise ValueError("electron count and multiplicity have inconsistent parity")
    nalpha = (electrons + difference) // 2
    return nalpha, electrons - nalpha


def _block_mo_density(state: ElectronicState) -> np.ndarray:
    blocks = []
    for index, coefficient in enumerate(state.mo_coefficients):
        density = state.density_ao(index)
        blocks.append(_ao_to_mo(density, coefficient, state.overlap_ao))
    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=float)
    start = 0
    for block in blocks:
        stop = start + block.shape[0]
        result[start:stop, start:stop] = np.real_if_close(block)
        start = stop
    return result


def _ao_to_mo(matrix: np.ndarray, coefficient: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    return coefficient.conj().T @ overlap @ matrix @ overlap @ coefficient


def _mo_to_ao(matrix: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return coefficient @ matrix @ coefficient.conj().T


def _split_columns(matrix: np.ndarray, sizes: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    boundaries = np.cumsum((0,) + sizes)
    return tuple(matrix[:, boundaries[i] : boundaries[i + 1]] for i in range(len(sizes)))


def _split_vector(vector: np.ndarray, sizes: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    boundaries = np.cumsum((0,) + sizes)
    return tuple(vector[boundaries[i] : boundaries[i + 1]] for i in range(len(sizes)))


def _block_ao_densities(
    rdm_mo: np.ndarray,
    coefficients: tuple[np.ndarray, ...],
    sizes: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    boundaries = np.cumsum((0,) + sizes)
    return tuple(
        _mo_to_ao(
            rdm_mo[boundaries[i] : boundaries[i + 1], boundaries[i] : boundaries[i + 1]],
            coefficients[i],
        )
        for i in range(len(sizes))
    )


def _read_excited_states(trexio, handle, metadata, coefficients, ground_mo):
    records = metadata.get("excited_states", ())
    if not isinstance(records, list) or not records or not trexio.has_rdm_1e_transition(handle):
        return ()
    if len(coefficients) != 1:
        raise ValueError("spin-resolved TREXIO transition densities need an explicit channel map")
    tensor = np.asarray(trexio.read_rdm_1e_transition(handle), dtype=float)
    coefficient = coefficients[0]
    states = []
    for index, raw in enumerate(records, start=1):
        item = dict(raw)
        transition = _mo_to_ao(tensor[0, index], coefficient)
        difference = None
        if np.any(np.abs(tensor[index, index]) > 0.0):
            difference = _mo_to_ao(tensor[index, index] - ground_mo, coefficient)
        states.append(
            ExcitedState(
                state_id=int(item.get("state_id", index)),
                transition_density_ao=transition,
                difference_density_ao=difference,
                energy_hartree=item.get("energy_hartree"),
                oscillator_strength=item.get("oscillator_strength"),
                label=str(item.get("label", f"S{index}")),
                symmetry=str(item.get("symmetry", "")),
            )
        )
    return tuple(states)


def _trexio():
    try:
        import trexio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TREXIO support requires the optional 'trexio' Python package"
        ) from exc
    return trexio
