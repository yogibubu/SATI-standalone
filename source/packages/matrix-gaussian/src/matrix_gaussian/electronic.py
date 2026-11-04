from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from matrix_qm import (
    ElectronicSection,
    ElectronicStateRecord,
    ElectronicTransitionRecord,
    TransitionsSection,
    merge_orbitals_section,
    orbital_file_record_from_path,
    write_electronic_section,
    write_transitions_section,
)

from .parsers import summarize_gaussian_log


_EXCITED_STATE_RE = re.compile(
    r"Excited State\s+(?P<index>\d+):\s+"
    r"(?P<label>\S+)\s+"
    r"(?P<energy>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+eV\s+"
    r"(?P<wavelength>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+nm\s+"
    r"f=(?P<osc>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)"
)


@dataclass(frozen=True)
class GaussianElectronicData:
    log_path: Path
    electronic: ElectronicSection
    transitions: TransitionsSection


@dataclass(frozen=True)
class GaussianElectronicPromotion:
    xyzin: Path
    log_path: Path
    wrote_electronic: bool
    wrote_transitions: bool
    wrote_orbitals: bool


GAUSSIAN_STATE_FINGERPRINT_REPRESENTATION = "gaussian-td-configurations-v1"
GAUSSIAN_EOM_STATE_FINGERPRINT_REPRESENTATION = (
    "gaussian-eom-ccsd-right-alpha-singles-v1"
)
_STATE_HEADER_RE = re.compile(
    r"^\s*Excited State\s+(?P<index>\d+):.*?"
    r"(?P<energy>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+eV",
    flags=re.IGNORECASE | re.MULTILINE,
)
_STATE_CONFIGURATION_RE = re.compile(
    r"^\s*(?P<occupied>\d+)\s*[-=]>\s*(?P<virtual>\d+)\s+"
    r"(?P<coefficient>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    flags=re.MULTILINE,
)
_GAUSSIAN_ORBITAL_COUNT_RE = re.compile(
    r"\b(?:NROrb|NBasis)\s*=\s*(?P<count>\d+)", flags=re.IGNORECASE
)
_EOM_FINAL_EIGENVALUE_HEADER_RE = re.compile(
    r"^\s*Final Eigenvalues for Irrep\s+\d+:\s*$", flags=re.IGNORECASE
)
_EOM_FINAL_EIGENVALUE_RE = re.compile(
    r"^\s*(?P<root>\d+)\s+"
    r"(?P<hartree>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)\s+"
    r"(?P<ev>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)\s+"
    r"(?P<nm>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)\s*$"
)
_EOM_SINGLES_AMPLITUDE_RE = re.compile(
    r"^\s*(?P<occupied>\d+)\s+\d+\s+(?P<virtual>\d+)\s+\d+\s+"
    r"(?P<coefficient>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)\s*$"
)
_EOM_TOTAL_ENERGY_RE = re.compile(
    r"Total Energy,\s*E\(EOM-CCSD\)\s*=\s*"
    r"(?P<energy>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class _GaussianStateManifold:
    state_ids: np.ndarray
    excitation_energies_hartree: np.ndarray
    vectors: np.ndarray
    representation: str
    reference_energy_hartree: float


def gaussian_state_fingerprint_data(path: Path | str, *, excitation_energies_hartree=None):
    """Return excited-state fingerprints from Gaussian amplitudes.

    TD methods use the printed occupied-to-virtual configurations.  EOM-CCSD
    uses the right alpha-singles amplitudes and the converged, full-precision
    EOM eigenvalues.  Both are embedded in a common orbital index space and
    therefore provide compact character vectors for APOC.
    Optional checkpoint excitation energies select the matching manifold in
    an optimization/frequency log instead of the default last manifold.
    """

    manifold = _gaussian_state_manifold(Path(path), excitation_energies_hartree=excitation_energies_hartree)
    if manifold is None:
        return None
    return (
        manifold.state_ids,
        manifold.excitation_energies_hartree,
        manifold.vectors,
        manifold.representation,
    )


def gaussian_state_reference_energy_hartree(path: Path | str) -> float | None:
    """Return the energy underlying a Gaussian excited-state manifold.

    For TD methods this is the final self-consistent-field energy.  For
    EOM-CCSD it is the CCSD energy reconstructed from the printed total energy
    of the requested root and that root's converged EOM eigenvalue.
    """

    manifold = _gaussian_state_manifold(Path(path))
    return None if manifold is None else float(manifold.reference_energy_hartree)


def _gaussian_state_manifold(target: Path, *, excitation_energies_hartree=None) -> _GaussianStateManifold | None:
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    headers = [
        (index, int(match.group("index")), float(match.group("energy")) / 27.211386245988)
        for index, line in enumerate(lines)
        if (match := _STATE_HEADER_RE.match(line)) is not None
    ]
    if not headers:
        return None
    all_headers = headers.copy()
    # Optimization and frequency logs may contain several TD manifolds.  A
    # LINK point must expose exactly one manifold, namely the last completed
    # one in the file, not the concatenation of all previous iterations.
    starts = [position for position, (_, state_id, _) in enumerate(headers) if state_id == 1]
    if starts:
        if excitation_energies_hartree is None:
            headers = headers[starts[-1] :]
        else:
            expected = np.asarray(excitation_energies_hartree, float)
            if expected.ndim != 1 or not len(expected) or not np.all(np.isfinite(expected)):
                raise ValueError("finite checkpoint excitation energies required")
            groups = [headers[a:b] for a, b in zip(starts, starts[1:]+[len(headers)])]
            candidates = [(max(abs(np.array([h[2] for h in group])-expected)), group)
                          for group in groups if len(group) == len(expected)]
            if not candidates:
                raise ValueError("no complete logged manifold matches the checkpoint")
            error, headers = min(candidates, key=lambda item: item[0])
            # Gaussian prints TD energies to four decimal places in eV.
            if error > 0.5001e-4/27.211386245988:
                raise ValueError("logged manifold differs from checkpoint energies")
    eom_eigenvalues = _last_eom_eigenvalues(lines, before=headers[0][0])
    is_eom = bool(eom_eigenvalues) and "EOM-CCSD" in text.upper()
    records: list[tuple[int, float, dict[tuple[int, int], float]]] = []
    # Use the orbital space declared by Gaussian rather than the largest
    # printed configuration.  Printed TD configurations are thresholded, so
    # their largest occupied/virtual index can change between geometries and
    # would otherwise make APOC fingerprints dimensionally incompatible.
    declared_orbitals = [
        int(match.group("count")) for match in _GAUSSIAN_ORBITAL_COUNT_RE.finditer(text)
    ]
    max_orbital = (max(declared_orbitals) - 1) if declared_orbitals else 0
    for position, (line_index, state_id, excitation) in enumerate(headers):
        end = next((i for i, _, _ in all_headers if i > line_index), len(lines))
        block = lines[line_index + 1 : end]
        if is_eom:
            contributions = _eom_right_alpha_singles(block)
            excitation = _match_eom_eigenvalue(excitation, eom_eigenvalues)
        else:
            contributions = {}
            for line in block:
                match = _STATE_CONFIGURATION_RE.match(line)
                if match is None:
                    continue
                occupied = int(match.group("occupied"))
                virtual = int(match.group("virtual"))
                coefficient = _float_token(match.group("coefficient"))
                contributions[(occupied, virtual)] = coefficient
        for occupied, virtual in contributions:
            max_orbital = max(max_orbital, occupied, virtual)
        if contributions:
            records.append((state_id, excitation, contributions))
    if not records:
        return None
    dimension = (max_orbital + 1) ** 2
    vectors = np.zeros((len(records), dimension), dtype=float)
    state_ids: list[int] = []
    excitations: list[float] = []
    for row, (state_id, excitation, contributions) in enumerate(records):
        state_ids.append(state_id)
        excitations.append(excitation)
        for (occupied, virtual), coefficient in contributions.items():
            vectors[row, occupied * (max_orbital + 1) + virtual] = coefficient
    if is_eom:
        reference_energy = _eom_reference_energy(headers, lines, eom_eigenvalues)
        representation = GAUSSIAN_EOM_STATE_FINGERPRINT_REPRESENTATION
    else:
        summary = summarize_gaussian_log(target)
        if not summary.scf_energies_hartree:
            return None
        reference_energy = float(summary.scf_energies_hartree[-1])
        representation = GAUSSIAN_STATE_FINGERPRINT_REPRESENTATION
    return _GaussianStateManifold(
        state_ids=np.asarray(state_ids, dtype=int),
        excitation_energies_hartree=np.asarray(excitations, dtype=float),
        vectors=vectors,
        representation=representation,
        reference_energy_hartree=reference_energy,
    )


def _last_eom_eigenvalues(lines: list[str], *, before: int) -> tuple[float, ...]:
    starts = [
        index
        for index, line in enumerate(lines[:before])
        if "Wavefunction amplitudes converged" in line
    ]
    section_start = starts[-1] if starts else 0
    blocks: list[list[float]] = []
    index = section_start
    while index < before:
        if _EOM_FINAL_EIGENVALUE_HEADER_RE.match(lines[index]) is None:
            index += 1
            continue
        values: list[float] = []
        index += 1
        while index < before:
            match = _EOM_FINAL_EIGENVALUE_RE.match(lines[index])
            if match is not None:
                values.append(_float_token(match.group("hartree")))
            elif values and not lines[index].strip():
                break
            index += 1
        if values:
            blocks.append(values)
        index += 1
    return tuple(value for block in blocks for value in block)


def _match_eom_eigenvalue(rounded_hartree: float, values: tuple[float, ...]) -> float:
    distances = np.abs(np.asarray(values, dtype=float) - float(rounded_hartree))
    order = np.argsort(distances)
    tolerance = (0.5e-4 + 1.0e-7) / 27.211386245988
    if distances[order[0]] > tolerance:
        raise ValueError("Gaussian EOM state has no matching final eigenvalue")
    if len(order) > 1 and distances[order[1]] <= tolerance:
        raise ValueError("Gaussian EOM state maps ambiguously to final eigenvalues")
    return float(values[int(order[0])])


def _eom_right_alpha_singles(lines: list[str]) -> dict[tuple[int, int], float]:
    contributions: dict[tuple[int, int], float] = {}
    in_right = False
    in_alpha_singles = False
    for line in lines:
        stripped = line.strip().casefold()
        if stripped == "right eigenvector":
            in_right = True
            in_alpha_singles = False
            continue
        if stripped == "left eigenvector":
            break
        if not in_right:
            continue
        if stripped == "alpha singles amplitudes":
            in_alpha_singles = True
            continue
        if in_alpha_singles and stripped.endswith("amplitudes"):
            break
        if not in_alpha_singles:
            continue
        match = _EOM_SINGLES_AMPLITUDE_RE.match(line)
        if match is None:
            continue
        occupied = int(match.group("occupied"))
        virtual = int(match.group("virtual"))
        contributions[(occupied, virtual)] = _float_token(match.group("coefficient"))
    return contributions


def _eom_reference_energy(
    headers: list[tuple[int, int, float]],
    lines: list[str],
    eigenvalues: tuple[float, ...],
) -> float:
    references: list[float] = []
    for position, (line_index, _state_id, rounded_excitation) in enumerate(headers):
        end = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
        total_matches = [
            _float_token(match.group("energy"))
            for line in lines[line_index + 1 : end]
            if (match := _EOM_TOTAL_ENERGY_RE.search(line)) is not None
        ]
        if not total_matches:
            continue
        excitation = _match_eom_eigenvalue(rounded_excitation, eigenvalues)
        references.append(float(total_matches[-1] - excitation))
    if not references:
        raise ValueError(
            "Gaussian EOM manifold has no root-resolved total energy from which "
            "to reconstruct the CCSD reference energy"
        )
    if max(references) - min(references) > 1.0e-8:
        raise ValueError("Gaussian EOM total energies imply inconsistent CCSD references")
    return float(np.mean(references))


def _float_token(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def write_gaussian_state_fingerprint_archive(
    log_path: Path | str,
    output: Path | str,
):
    """Write a compressed APOC state-manifold archive from a Gaussian log."""

    data = gaussian_state_fingerprint_data(log_path)
    if data is None:
        return None
    state_ids, excitations, vectors, representation = data
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        state_ids=state_ids,
        excitation_energies_hartree=excitations,
        vectors=vectors,
        representation=np.asarray([representation]),
    )
    return target


def parse_gaussian_electronic_log(path: Path | str) -> GaussianElectronicData:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="ignore")
    summary = summarize_gaussian_log(target)
    states: list[ElectronicStateRecord] = []
    if summary.scf_energies_hartree:
        states.append(
            ElectronicStateRecord(
                label="S0",
                energy_hartree=summary.scf_energies_hartree[-1],
                energy_ev=0.0,
                source="gaussian-log",
            )
        )
    transitions: list[ElectronicTransitionRecord] = []
    for match in _EXCITED_STATE_RE.finditer(text):
        index = int(match.group("index"))
        to_state = f"S{index}"
        label = match.group("label")
        energy_ev = float(match.group("energy"))
        multiplicity, symmetry = _split_excited_label(label)
        states.append(
            ElectronicStateRecord(
                label=to_state,
                energy_ev=energy_ev,
                multiplicity=multiplicity,
                symmetry=symmetry,
                source="gaussian-log",
            )
        )
        transitions.append(
            ElectronicTransitionRecord(
                from_state="S0",
                to_state=to_state,
                energy_ev=energy_ev,
                wavelength_nm=float(match.group("wavelength")),
                oscillator_strength=float(match.group("osc").replace("D", "E")),
                source="gaussian-log",
            )
        )
    return GaussianElectronicData(
        log_path=target,
        electronic=ElectronicSection(tuple(states)),
        transitions=TransitionsSection(tuple(transitions)),
    )


def promote_gaussian_electronic_log_to_xyzin(
    log_path: Path | str,
    xyzin: Path | str,
    *,
    write_electronic: bool = True,
    write_transitions: bool = True,
    orbital_files: tuple[Path | str, ...] = (),
) -> GaussianElectronicPromotion:
    target = Path(xyzin)
    data = parse_gaussian_electronic_log(log_path)
    wrote_electronic = False
    wrote_transitions = False
    wrote_orbitals = False
    if write_electronic and data.electronic.states:
        write_electronic_section(target, data.electronic)
        wrote_electronic = True
    if write_transitions and data.transitions.transitions:
        write_transitions_section(target, data.transitions)
        wrote_transitions = True
    if orbital_files:
        merge_orbitals_section(
            target,
            tuple(
                orbital_file_record_from_path(path, source="gaussian-log") for path in orbital_files
            ),
        )
        wrote_orbitals = True
    return GaussianElectronicPromotion(
        xyzin=target,
        log_path=data.log_path,
        wrote_electronic=wrote_electronic,
        wrote_transitions=wrote_transitions,
        wrote_orbitals=wrote_orbitals,
    )


def _split_excited_label(label: str) -> tuple[str, str]:
    if "-" not in label:
        return label, ""
    multiplicity, symmetry = label.split("-", 1)
    return multiplicity, symmetry
