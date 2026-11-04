"""Translate native std2 eigenvectors to APOC states; tracking stays in APOC."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .state import ElectronicState, ExcitedState, gaussian_basis_from_pyscf


def electronic_state_from_std2(
    workdir: Path | str, *, ground_state: ElectronicState | None = None,
) -> ElectronicState:
    """Import restricted or unrestricted spin-conserving std2 eigenpairs.

    Native vectors are expanded by std2 into the full input MO ordering. APOC
    supplies cross-geometry AO metrics, matching, phase and ambiguity checks
    through its existing ``follow_excited_state`` / ``match_excited_states``.
    A supplied ground state must use the exact Molden MO order and phases.
    """
    directory = Path(workdir)
    result = json.loads((directory / "spectrum.json").read_text())
    if result.get("schema") != "matrix.link.std2.spectrum.v1":
        raise ValueError("unsupported std2 spectrum artifact")
    artifact = result["state_vectors"]
    filename = artifact["path"]
    if filename not in {"ciss_a", "sing_a", "ucis_a", "unrs_a"}:
        raise ValueError("unsupported std2 spin manifold")
    raw = (directory / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
        raise ValueError("std2 eigenvectors changed after execution")
    molden = directory / "reference.molden"
    if hashlib.sha256(molden.read_bytes()).hexdigest() != result["request"]["molden_sha256"]:
        raise ValueError("std2 Molden reference changed after execution")
    ground = ground_state if ground_state is not None else _ground_from_std2_molden(molden)
    unrestricted = filename in {"ucis_a", "unrs_a"}
    rpa = filename in {"sing_a", "unrs_a"}
    if len(ground.mo_coefficients) != (2 if unrestricted else 1):
        raise ValueError("std2 spin channels do not match the reference orbitals")
    channels = []
    for c, occ in zip(ground.mo_coefficients, ground.mo_occupations, strict=True):
        occupancy = 1.0 if unrestricted else 2.0
        nocc = int(np.count_nonzero(occ > occupancy - 0.01))
        if not np.allclose(occ, np.r_[np.full(nocc, occupancy), np.zeros(len(occ) - nocc)], atol=1e-6):
            raise ValueError("std2 requires occupied-then-virtual integral occupations per spin")
        if not np.allclose(c.conj().T @ ground.overlap_ao @ c, np.eye(c.shape[1]), atol=1e-4):
            raise ValueError("Molden MO normalization is incompatible with the APOC AO metric")
        channels.append((c[:, :nocc], c[:, nocc:], nocc * (c.shape[1] - nocc)))
    text = raw.decode("ascii")
    dimension_match = re.search(r"\$tensor space dimension\s+(\d+)", text)
    count_match = re.search(r"\$current subspace dimension\s+(\d+)", text)
    dimension = sum(channel[2] for channel in channels)
    if dimension_match is None or int(dimension_match[1]) != dimension or not dimension:
        raise ValueError("std2 eigenvector dimensions do not match the reference orbitals")
    tag = {"ciss_a": "ciss", "sing_a": "rpas", "ucis_a": "ucis", "unrs_a": "urpa"}[filename]
    if f"$scfinstab {tag}" not in text or "$eigenpairs" not in text or "$end" not in text:
        raise ValueError("invalid std2 eigenvector header")
    body = text.split("$eigenpairs", 1)[1].split("$end", 1)[0]
    blocks = list(re.finditer(r"^\s*(\d+)\s+eigenvalue\s*=\s*([^\s]+)\s*$", body, re.M))
    if count_match is None or len(blocks) != int(count_match[1]):
        raise ValueError("std2 eigenvector root count is incomplete")
    transitions = {row["state_id"]: row for row in result["transitions"]}
    states = []
    for i, match in enumerate(blocks):
        state_id = int(match[1])
        energy = float(match[2].replace("D", "E"))
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(body)
        values = []
        for line in body[match.end():end].splitlines():
            line = line.rstrip()
            for offset in range(0, len(line), 20):
                values.append(float(line[offset:offset + 20].replace("D", "E")))
        amplitudes = np.asarray(values)
        if amplitudes.size != dimension * (2 if rpa else 1) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("invalid or truncated std2 eigenvector")
        if not np.isfinite(energy) or energy <= 0 or state_id != i + 1:
            raise ValueError("invalid std2 eigenpair identity or energy")
        densities = []
        offset = 0
        for co, cv, size in channels:
            plus = amplitudes[offset:offset + size].reshape(co.shape[1], cv.shape[1])
            if rpa:
                minus = amplitudes[dimension + offset:dimension + offset + size].reshape(plus.shape)
                x, y = (plus + minus) / 2, (plus - minus) / 2
                density = co @ x @ cv.conj().T + cv @ y.conj().T @ co.conj().T
            else:
                density = co @ plus @ cv.conj().T
            densities.append(density)
            offset += size
        density = sum(densities)
        row = transitions.get(state_id)
        if row is not None and abs(energy * 27.21139 - row["excitation_energy_ev"]) > 0.00006:
            raise ValueError("std2 spectrum and state-vector energies disagree")
        states.append(ExcitedState(
            state_id=state_id, transition_density_ao=density,
            transition_density_spin_ao=tuple(densities) if unrestricted else (),
            energy_hartree=energy,
            oscillator_strength=None if row is None else row["oscillator_strength_length"],
            label=f"std2 root {state_id} (excitation energy)",
        ))
    if not states:
        raise ValueError("std2 exported no states for APOC")
    return replace(ground, excited_states=tuple(states), source=str(directory.resolve()))


def _ground_from_std2_molden(path: Path) -> ElectronicState:
    """Preserve Molden alpha/beta order; do not merge unrestricted orbitals."""
    from matrix_pyscf import load_molden_wavefunction

    try:
        mol, energies, coefficients, occupations, _labels, spins = load_molden_wavefunction(path)
    except (KeyError, NotImplementedError) as exc:
        raise ValueError("unsupported Molden convention; supply the producing backend's APOC ground_state") from exc
    if isinstance(coefficients, tuple):
        # Current PySCF loaders already separate unrestricted alpha/beta data.
        if len(coefficients) != 2 or len(occupations) != 2 or len(energies) != 2:
            raise ValueError("incomplete Molden alpha/beta channels")
        coeff = tuple(np.asarray(item) for item in coefficients)
        occ = tuple(np.asarray(item) for item in occupations)
        energy_channels = tuple(np.asarray(item) for item in energies)
    else:
        coefficients = np.asarray(coefficients)
        occupations = np.asarray(occupations)
        energies = np.asarray(energies)
        tags = np.asarray([str(value).strip().casefold() for value in spins])
        if tags.shape != occupations.shape:
            raise ValueError("Molden spin labels do not cover the complete MO space")
        if np.any(tags == "beta"):
            if not set(tags) <= {"alpha", "beta"}:
                raise ValueError("unsupported Molden spin label")
            indices = (np.flatnonzero(tags == "alpha"), np.flatnonzero(tags == "beta"))
        else:
            indices = (np.arange(len(occupations)),)
        coeff = tuple(coefficients[:, index] for index in indices)
        occ = tuple(occupations[index] for index in indices)
        energy_channels = tuple(energies[index] for index in indices)
    charge = int(round(sum(mol.atom_charges()) - sum(np.sum(item) for item in occ)))
    multiplicity = int(round(abs(np.sum(occ[0]) - np.sum(occ[1])))) + 1 if len(occ) == 2 else 1
    return ElectronicState(
        atomic_numbers=mol.atom_charges(), coordinates_bohr=mol.atom_coords(unit="Bohr"),
        overlap_ao=mol.intor_symmetric("int1e_ovlp"), mo_coefficients=coeff,
        mo_occupations=occ, mo_energies_hartree=energy_channels,
        basis=gaussian_basis_from_pyscf(mol), charge=charge, multiplicity=multiplicity,
        source=str(path),
    )


__all__ = ["electronic_state_from_std2"]
