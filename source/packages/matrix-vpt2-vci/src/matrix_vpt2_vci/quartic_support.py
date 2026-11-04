"""Chemical screening of normal-mode :math:`K_{iijj}` quartic constants.

The numerical normal-QFF projection can rank every semi-diagonal quartic,
but it cannot by itself decide which terms should become physical ZAFF
interactions.  This module supplies the deliberately local prior used by
ARCHITECT: X--H stretching / H--X--Y bending pairs are promoted only when
their internal-coordinate participation shares the same X--H bond.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM

from .vci import QuarticForceField


ZAFF_IIJJ_SUPPORT_SCHEMA = "matrix.trinity.zaff_iijj_support.v1"


@dataclass(frozen=True)
class NormalModeChemicalCharacter:
    """Internal-coordinate character of one dimensionless normal mode."""

    mode: int
    frequency_cm: float
    family_weights: Mapping[str, float]
    xh_bond_weights: Mapping[tuple[int, int], float]
    hxy_bond_weights: Mapping[tuple[int, int], float]
    dominant_coordinates: tuple[tuple[str, str, float], ...] = ()

    @property
    def xh_stretch_weight(self) -> float:
        return float(self.family_weights.get("XH_STRETCH", 0.0))

    @property
    def hxy_bend_weight(self) -> float:
        return float(self.family_weights.get("HXY_BEND", 0.0))

    def to_dict(self) -> dict[str, object]:
        def bonds(values: Mapping[tuple[int, int], float]) -> dict[str, float]:
            return {f"{a}-{b}": float(value) for (a, b), value in sorted(values.items())}

        return {
            "mode_one_based": self.mode + 1,
            "frequency_cm-1": self.frequency_cm,
            "family_weights": dict(self.family_weights),
            "xh_bond_weights": bonds(self.xh_bond_weights),
            "hxy_bond_weights": bonds(self.hxy_bond_weights),
            "dominant_coordinates": [
                {"identifier": identifier, "family": family, "weight": weight}
                for identifier, family, weight in self.dominant_coordinates
            ],
        }


def normal_mode_internal_ped(
    force_constants_internal: np.ndarray,
    b_matrix_per_angstrom: np.ndarray,
    displacement_bohr_per_dimensionless: np.ndarray,
) -> np.ndarray:
    """Return absolute normalized internal-coordinate PEDs for normal modes.

    The Cartesian displacement matrix has one dimensionless normal mode per
    column.  Wilson rows are per Angstrom, hence the explicit bohr-to-Angstrom
    conversion.  For a mode vector ``q`` the signed coordinate contribution is
    ``q * (F q)``; absolute normalization gives a robust chemical classifier
    even in a non-orthogonal local SONIC basis.
    """

    force = np.asarray(force_constants_internal, dtype=float)
    b_matrix = np.asarray(b_matrix_per_angstrom, dtype=float)
    displacement = np.asarray(displacement_bohr_per_dimensionless, dtype=float)
    if force.ndim != 2 or force.shape[0] != force.shape[1]:
        raise ValueError("internal force-constant matrix must be square")
    if b_matrix.ndim != 2 or b_matrix.shape[0] != force.shape[0]:
        raise ValueError("B-matrix row count must match the internal force constants")
    if displacement.ndim != 2 or displacement.shape[0] != b_matrix.shape[1]:
        raise ValueError("normal displacement rows must match Cartesian B-matrix columns")
    internal = b_matrix @ (displacement * BOHR_TO_ANGSTROM)
    contributions = np.abs(internal * (force @ internal))
    norms = np.sum(contributions, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("at least one normal mode has zero internal-coordinate PED")
    return (contributions / norms[None, :]).T


def zaff_normal_mode_chemical_characters(
    *,
    atoms: Sequence[str],
    definition: object,
    ped: np.ndarray,
    frequencies_cm: Sequence[float],
    dominant_coordinate_count: int = 3,
) -> tuple[NormalModeChemicalCharacter, ...]:
    """Classify local-SONIC PEDs and retain their X--H bond provenance."""

    atom_symbols = tuple(str(atom).upper() for atom in atoms)
    gics = tuple(getattr(definition, "gics"))
    primitives = tuple(getattr(definition, "primitives"))
    ped_array = np.asarray(ped, dtype=float)
    frequencies = np.asarray(frequencies_cm, dtype=float).reshape(-1)
    if ped_array.shape != (frequencies.size, len(gics)):
        raise ValueError("PED shape must be (normal modes, local SONICs)")
    primitive_by_id = {primitive.identifier: primitive for primitive in primitives}
    metadata = [
        _gic_chemical_metadata(gic, primitive_by_id, atom_symbols) for gic in gics
    ]
    characters: list[NormalModeChemicalCharacter] = []
    for mode, (frequency, weights) in enumerate(zip(frequencies, ped_array)):
        family_weights: dict[str, float] = {}
        xh_bonds: dict[tuple[int, int], float] = {}
        hxy_bonds: dict[tuple[int, int], float] = {}
        for weight, item in zip(weights, metadata):
            for family, fraction in item[0].items():
                family_weights[family] = family_weights.get(family, 0.0) + float(weight * fraction)
            for bond, fraction in item[1].items():
                xh_bonds[bond] = xh_bonds.get(bond, 0.0) + float(weight * fraction)
            for bond, fraction in item[2].items():
                hxy_bonds[bond] = hxy_bonds.get(bond, 0.0) + float(weight * fraction)
        order = np.argsort(-weights)[: max(0, dominant_coordinate_count)]
        dominant = tuple(
            (gics[index].identifier, gics[index].family, float(weights[index]))
            for index in order
        )
        characters.append(
            NormalModeChemicalCharacter(
                mode=mode,
                frequency_cm=float(frequency),
                family_weights=family_weights,
                xh_bond_weights=xh_bonds,
                hxy_bond_weights=hxy_bonds,
                dominant_coordinates=dominant,
            )
        )
    return tuple(characters)


def analyze_xh_bend_iijj_support(
    force_field: QuarticForceField,
    characters: Sequence[NormalModeChemicalCharacter],
    *,
    reference_quartic_cm: Mapping[tuple[int, int, int, int], float] | None = None,
    xh_threshold: float = 0.70,
    hxy_threshold: float = 0.35,
    shared_bond_threshold: float = 0.25,
) -> dict[str, object]:
    """Rank X--H stretch/H--X--Y bend ``K_iijj`` candidates.

    Candidate selection uses only the ZAFF chemical character.  An optional
    QM normal-QFF is reported solely as an external validation and never enters
    the decision, preventing reference-data leakage into the physical prior.
    ``K_iijj / 8`` is included because it is the direct first-order contribution
    of this term to either fundamental in dimensionless normal coordinates.
    """

    mode_count = len(characters)
    frequencies = np.asarray(force_field.harmonic_frequencies_cm, dtype=float)
    if frequencies.size != mode_count:
        raise ValueError("chemical-character count differs from normal-mode count")
    reference = reference_quartic_cm or {}
    rows: list[dict[str, object]] = []
    for i in range(mode_count):
        for j in range(i + 1, mode_count):
            key = (i, i, j, j)
            zaff_value = float(force_field.quartic_cm.get(key, 0.0))
            first, second = characters[i], characters[j]
            directional = (
                first.xh_stretch_weight * second.hxy_bend_weight,
                second.xh_stretch_weight * first.hxy_bend_weight,
            )
            family_score = float(max(directional))
            if directional[0] >= directional[1]:
                stretch, bend = first, second
            else:
                stretch, bend = second, first
            shared = float(
                sum(
                    weight * bend.hxy_bond_weights.get(bond, 0.0)
                    for bond, weight in stretch.xh_bond_weights.items()
                )
            )
            family_candidate = bool(
                stretch.xh_stretch_weight >= xh_threshold
                and bend.hxy_bend_weight >= hxy_threshold
            )
            local_candidate = bool(family_candidate and shared >= shared_bond_threshold)
            row: dict[str, object] = {
                "modes_one_based": [i + 1, j + 1],
                "frequencies_cm-1": [float(frequencies[i]), float(frequencies[j])],
                "stretch_mode_one_based": stretch.mode + 1,
                "bend_mode_one_based": bend.mode + 1,
                "xh_stretch_weight": stretch.xh_stretch_weight,
                "hxy_bend_weight": bend.hxy_bend_weight,
                "family_score": family_score,
                "shared_xh_bond_score": shared,
                "family_candidate": family_candidate,
                "local_shared_bond_candidate": local_candidate,
                "zaff_K_iijj_cm-1": zaff_value,
                "zaff_direct_fundamental_contribution_cm-1": zaff_value / 8.0,
                "selection_uses_reference": False,
            }
            if key in reference:
                value = float(reference[key])
                row.update(
                    {
                        "reference_K_iijj_cm-1": value,
                        "reference_direct_fundamental_contribution_cm-1": value / 8.0,
                        "absolute_reference_K_iijj_cm-1": abs(value),
                    }
                )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            not bool(row["local_shared_bond_candidate"]),
            -float(row.get("absolute_reference_K_iijj_cm-1", 0.0)),
            -abs(float(row["zaff_K_iijj_cm-1"])),
        )
    )
    candidates = [row for row in rows if row["local_shared_bond_candidate"]]
    validated = [row for row in candidates if "reference_K_iijj_cm-1" in row]
    return {
        "schema": ZAFF_IIJJ_SUPPORT_SCHEMA,
        "owner": "ARCHITECT/TRINITY",
        "policy": {
            "candidate_family": "XH_STRETCH_x_HXY_BEND",
            "xh_stretch_weight_minimum": xh_threshold,
            "hxy_bend_weight_minimum": hxy_threshold,
            "shared_xh_bond_score_minimum": shared_bond_threshold,
            "selection_uses_reference_qff": False,
            "torsional_large_amplitude_terms": "EXCLUDED_AND_ASSIGNED_TO_SCANS",
        },
        "summary": {
            "mode_count": mode_count,
            "all_iijj_pair_count": len(rows),
            "local_shared_bond_candidate_count": len(candidates),
            "validated_candidate_count": len(validated),
            "validated_candidates_above_100_cm-1": sum(
                abs(float(row["reference_K_iijj_cm-1"])) >= 100.0 for row in validated
            ),
        },
        "mode_characters": [character.to_dict() for character in characters],
        "ranked_iijj_pairs": rows,
    }


def _gic_chemical_metadata(
    gic: object,
    primitive_by_id: Mapping[str, object],
    atoms: Sequence[str],
) -> tuple[
    dict[str, float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    coefficients = tuple(getattr(gic, "coefficients")) or ((gic.primitive_id, 1.0),)
    norm = sum(float(coefficient) ** 2 for _identifier, coefficient in coefficients)
    norm = norm if norm > 0.0 else 1.0
    families: dict[str, float] = {}
    xh_bonds: dict[tuple[int, int], float] = {}
    hxy_bonds: dict[tuple[int, int], float] = {}
    for identifier, coefficient in coefficients:
        primitive = primitive_by_id.get(identifier)
        if primitive is None:
            continue
        weight = float(coefficient) ** 2 / norm
        function = str(primitive.function).upper()
        indices = tuple(int(atom) for atom in primitive.atoms)
        family = str(primitive.family).upper()
        if function == "R" and len(indices) == 2:
            bond = _xh_bond(indices[0], indices[1], atoms)
            label = "XH_STRETCH" if bond is not None else "HEAVY_STRETCH"
            if bond is not None:
                xh_bonds[bond] = xh_bonds.get(bond, 0.0) + weight
        elif function in {"A", "L"} and len(indices) >= 3:
            terminal_bonds = tuple(
                bond
                for bond in (
                    _xh_bond(indices[0], indices[1], atoms),
                    _xh_bond(indices[-1], indices[1], atoms),
                )
                if bond is not None
            )
            label = "HXY_BEND" if terminal_bonds else "HEAVY_BEND"
            for bond in set(terminal_bonds):
                hxy_bonds[bond] = hxy_bonds.get(bond, 0.0) + weight
        else:
            label = family
        families[label] = families.get(label, 0.0) + weight
    return families, xh_bonds, hxy_bonds


def _xh_bond(first: int, second: int, atoms: Sequence[str]) -> tuple[int, int] | None:
    first_h = atoms[first - 1] == "H"
    second_h = atoms[second - 1] == "H"
    if first_h == second_h:
        return None
    hydrogen, heavy = (first, second) if first_h else (second, first)
    return (heavy, hydrogen)
