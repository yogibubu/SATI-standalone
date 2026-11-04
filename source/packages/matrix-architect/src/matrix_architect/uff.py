"""ARCHITECT-owned implementation of the original Universal Force Field.

The implementation is deliberately independent of RDKit and SWITCH.  ORACLE
owns graph perception and continuous atomic descriptors; ARCHITECT freezes one
UFF model from that state and evaluates the original bonded plus 12-6 UFF
energy.  Cartesian gradients are analytic.  The Hessian used as a LINK seed is
the complete linear Wilson congruence plus the existing exact non-bonded
Hessian; the curvilinear B-prime term is intentionally omitted by the frozen
initialization protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
from itertools import combinations
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from matrix_chem import (
    BOHR_TO_ANGSTROM,
    Primitive,
    build_topology_objects,
    electronic_population_overrides_from_xyzin,
    eval_primitive,
    grad_primitive,
    nonbonded_cartesian_hessian_components,
    read_xyzin_geometry,
)
from matrix_chem.topology.elements import atomic_number, atomic_symbol


UFF_MODEL_SCHEMA = "matrix.architect.uff_model.v1"
UFF_EVALUATION_SCHEMA = "matrix.architect.uff_evaluation.v1"
UFF_REFERENCE = "Rappe et al., JACS 114, 10024-10035 (1992), DOI:10.1021/ja00051a040"
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631
UFF_LAMBDA = 0.1332
UFF_G = 332.06


@dataclass(frozen=True)
class UffAtomParameter:
    atom_type: str
    r1_angstrom: float
    theta0_radian: float
    effective_charge: float
    torsion_v_kcal_mol: float
    torsion_u_kcal_mol: float
    electronegativity: float


@dataclass(frozen=True)
class UffBondTerm:
    atoms: tuple[int, int]
    equilibrium_angstrom: float
    force_kcal_mol_angstrom2: float
    bond_order: float


@dataclass(frozen=True)
class UffAngleTerm:
    atoms: tuple[int, int, int]
    force_kcal_mol: float
    theta0_radian: float
    order: int
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0


@dataclass(frozen=True)
class UffTorsionTerm:
    atoms: tuple[int, int, int, int]
    force_kcal_mol: float
    order: int
    cosine_phase: float


@dataclass(frozen=True)
class UffInversionTerm:
    atoms: tuple[int, int, int, int]
    force_kcal_mol: float
    c0: float
    c1: float
    c2: float


@dataclass(frozen=True)
class UffModel:
    atoms: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    atom_types: tuple[str, ...]
    topology_bonds: tuple[tuple[int, int], ...]
    bonds: tuple[UffBondTerm, ...]
    angles: tuple[UffAngleTerm, ...]
    torsions: tuple[UffTorsionTerm, ...]
    inversions: tuple[UffInversionTerm, ...]
    reference_coordinates_angstrom: np.ndarray
    schema: str = UFF_MODEL_SCHEMA
    reference: str = UFF_REFERENCE
    atom_typing_owner: str = "ARCHITECT_FROM_ORACLE_TOPOLOGY_AND_SYNTHONS"
    runtime: str = "ORIGINAL_UFF_BONDED_PLUS_UFF_12_6_NO_ELECTROSTATICS"

    def __post_init__(self) -> None:
        if self.schema != UFF_MODEL_SCHEMA:
            raise ValueError(f"unsupported UFF model schema: {self.schema}")
        coordinates = np.asarray(self.reference_coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or np.any(~np.isfinite(coordinates)):
            raise ValueError("UFF reference coordinates do not match the atom list")
        if len(self.atomic_numbers) != len(self.atoms) or len(self.atom_types) != len(self.atoms):
            raise ValueError("UFF atom metadata do not match the atom list")
        object.__setattr__(self, "reference_coordinates_angstrom", coordinates)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "reference": self.reference,
            "atom_typing_owner": self.atom_typing_owner,
            "runtime": self.runtime,
            "atoms": list(self.atoms),
            "atomic_numbers": list(self.atomic_numbers),
            "atom_types": list(self.atom_types),
            "topology_bonds": [list(pair) for pair in self.topology_bonds],
            "reference_coordinates_angstrom": self.reference_coordinates_angstrom.tolist(),
            "bonds": [asdict(term) for term in self.bonds],
            "angles": [asdict(term) for term in self.angles],
            "torsions": [asdict(term) for term in self.torsions],
            "inversions": [asdict(term) for term in self.inversions],
            "hessian_contract": "LINEAR_WILSON_CONGRUENCE_WITHOUT_B_PRIME_PLUS_EXACT_UFF_VDW",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UffModel":
        if payload.get("schema") != UFF_MODEL_SCHEMA:
            raise ValueError("unsupported UFF model document")
        return cls(
            atoms=tuple(str(value) for value in payload["atoms"]),
            atomic_numbers=tuple(int(value) for value in payload["atomic_numbers"]),
            atom_types=tuple(str(value) for value in payload["atom_types"]),
            topology_bonds=tuple(tuple(int(value) for value in pair) for pair in payload["topology_bonds"]),
            bonds=tuple(_term_from_dict(UffBondTerm, row) for row in payload["bonds"]),
            angles=tuple(_term_from_dict(UffAngleTerm, row) for row in payload["angles"]),
            torsions=tuple(_term_from_dict(UffTorsionTerm, row) for row in payload["torsions"]),
            inversions=tuple(_term_from_dict(UffInversionTerm, row) for row in payload["inversions"]),
            reference_coordinates_angstrom=np.asarray(
                payload["reference_coordinates_angstrom"], dtype=float
            ),
            reference=str(payload.get("reference", UFF_REFERENCE)),
            atom_typing_owner=str(
                payload.get("atom_typing_owner", "ARCHITECT_FROM_ORACLE_TOPOLOGY_AND_SYNTHONS")
            ),
            runtime=str(
                payload.get("runtime", "ORIGINAL_UFF_BONDED_PLUS_UFF_12_6_NO_ELECTROSTATICS")
            ),
        )


@dataclass(frozen=True)
class UffEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None
    hessian_hartree_per_bohr2: np.ndarray | None
    components_hartree: Mapping[str, float]
    schema: str = UFF_EVALUATION_SCHEMA

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "energy_hartree": self.energy_hartree,
            "components_hartree": dict(self.components_hartree),
        }
        if self.gradient_hartree_per_bohr is not None:
            payload["gradient_hartree_per_bohr"] = self.gradient_hartree_per_bohr.tolist()
        if self.hessian_hartree_per_bohr2 is not None:
            payload["hessian_hartree_per_bohr2"] = self.hessian_hartree_per_bohr2.tolist()
        return payload


def load_uff_parameters() -> dict[str, UffAtomParameter]:
    """Load the bundled, attributed classic UFF atom table."""

    text = resources.files("matrix_architect").joinpath("data/uff_parameters.tsv").read_text(
        encoding="utf-8"
    )
    result: dict[str, UffAtomParameter] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 7:
            raise RuntimeError(f"invalid bundled UFF parameter row: {raw}")
        label = fields[0]
        if label in result:
            raise RuntimeError(f"duplicate bundled UFF atom type: {label}")
        result[label] = UffAtomParameter(
            atom_type=label,
            r1_angstrom=float(fields[1]),
            theta0_radian=math.radians(float(fields[2])),
            effective_charge=float(fields[3]),
            torsion_v_kcal_mol=float(fields[4]),
            torsion_u_kcal_mol=float(fields[5]),
            electronegativity=float(fields[6]),
        )
    if len(result) != 127:
        raise RuntimeError(f"bundled UFF table has {len(result)} rows, expected 127")
    return result


def compile_uff_model(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    bond_order_overrides: Mapping[tuple[int, int], float] | None = None,
    external_charges: Mapping[int, float] | None = None,
    charge_source: str = "ORACLE electronegativity estimate",
    bond_order_source: str = "ORACLE Pauling estimate",
) -> UffModel:
    """Compile one deterministic UFF model from the canonical ORACLE state."""

    symbols = tuple(str(atom) for atom in atoms)
    numbers = tuple(_required_atomic_number(atom) for atom in symbols)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(symbols), 3) or np.any(~np.isfinite(coords)):
        raise ValueError("UFF compilation coordinates must have shape (natoms, 3)")
    if any(number > 103 for number in numbers):
        raise ValueError("classic UFF parameters cover atomic numbers 1 through 103")
    _continuous, graph, _rings, synthons, aromaticity = build_topology_objects(
        coords,
        np.asarray(numbers, dtype=int),
        bond_order_overrides=bond_order_overrides,
        external_charges=external_charges,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
    )
    table = load_uff_parameters()
    grouped = _parameters_by_atomic_number(table)
    atom_types = tuple(
        _assign_atom_type(
            atom,
            numbers,
            graph,
            synthons,
            aromaticity,
            grouped.get(numbers[atom], ()),
        )
        for atom in range(len(symbols))
    )
    parameters = tuple(table[label] for label in atom_types)
    bond_orders = {
        tuple(sorted((int(left), int(right)))): _uff_bond_order(
            int(left), int(right), synthons, aromaticity
        )
        for left, right in graph.bonds
    }
    bond_terms = tuple(
        _bond_term(pair, bond_orders[pair], parameters)
        for pair in sorted(bond_orders)
    )
    angle_terms = _compile_angles(
        coords,
        graph,
        atom_types,
        parameters,
        bond_orders,
    )
    torsion_terms = _compile_torsions(
        numbers, graph, synthons, atom_types, parameters, bond_orders
    )
    inversion_terms = _compile_inversions(
        numbers, graph, atom_types, parameters, bond_orders
    )
    return UffModel(
        atoms=symbols,
        atomic_numbers=numbers,
        atom_types=atom_types,
        topology_bonds=tuple(
            (int(left) + 1, int(right) + 1) for left, right in sorted(graph.bonds)
        ),
        bonds=bond_terms,
        angles=angle_terms,
        torsions=torsion_terms,
        inversions=inversion_terms,
        reference_coordinates_angstrom=coords,
    )


def compile_uff_model_from_xyzin(path: Path | str) -> UffModel:
    """Compile UFF from the persisted ORACLE geometry and population contract."""

    source = Path(path)
    geometry = read_xyzin_geometry(source)
    electronic = electronic_population_overrides_from_xyzin(source)
    return compile_uff_model(
        geometry.atoms,
        geometry.coordinates_angstrom,
        bond_order_overrides=electronic.get("bond_orders"),
        external_charges=electronic.get("charges"),
        charge_source=str(
            electronic.get("charge_source", "ORACLE electronegativity estimate")
        ),
        bond_order_source=str(
            electronic.get("bond_order_source", "ORACLE Pauling estimate")
        ),
    )


def evaluate_uff_model(
    model: UffModel,
    coordinates_angstrom: np.ndarray,
    *,
    properties: Sequence[str] = ("energy", "gradient", "hessian"),
) -> UffEvaluation:
    """Evaluate classic UFF with analytic gradient and seed Hessian."""

    requested = {str(value).strip().casefold() for value in properties}
    unknown = requested - {"energy", "gradient", "hessian"}
    if unknown:
        raise ValueError(f"unsupported UFF properties: {sorted(unknown)}")
    need_gradient = bool({"gradient", "hessian"} & requested)
    need_hessian = "hessian" in requested
    coords = np.asarray(coordinates_angstrom, dtype=float)
    natoms = len(model.atoms)
    if coords.shape != (natoms, 3) or np.any(~np.isfinite(coords)):
        raise ValueError("UFF evaluation coordinates do not match the compiled model")

    bonded_energy = 0.0
    gradient_kcal_per_angstrom = np.zeros(3 * natoms, dtype=float)
    hessian_kcal_per_angstrom2 = np.zeros((3 * natoms, 3 * natoms), dtype=float)
    component_energy = {"bond": 0.0, "angle": 0.0, "torsion": 0.0, "inversion": 0.0}

    for term in model.bonds:
        primitive = Primitive("bond", term.atoms)
        distance = eval_primitive(primitive, coords)
        displacement = distance - term.equilibrium_angstrom
        energy = 0.5 * term.force_kcal_mol_angstrom2 * displacement**2
        first = term.force_kcal_mol_angstrom2 * displacement
        second = term.force_kcal_mol_angstrom2
        row = grad_primitive(primitive, coords).reshape(-1)
        bonded_energy += energy
        component_energy["bond"] += energy
        if need_gradient:
            gradient_kcal_per_angstrom += first * row
        if need_hessian:
            hessian_kcal_per_angstrom2 += second * np.outer(row, row)

    for term in model.angles:
        energy, derivative_cosine, second_theta = _angle_derivatives(term, coords)
        bonded_energy += energy
        component_energy["angle"] += energy
        if need_gradient:
            gradient_kcal_per_angstrom += derivative_cosine * _cosine_angle_gradient(
                term.atoms, coords
            )
        if need_hessian:
            if term.order == 1:
                for mode in (-1, 1):
                    row = grad_primitive(
                        Primitive("linear_bend", term.atoms, mode=mode), coords
                    ).reshape(-1)
                    hessian_kcal_per_angstrom2 += second_theta * np.outer(row, row)
            else:
                row = grad_primitive(Primitive("angle", term.atoms), coords).reshape(-1)
                hessian_kcal_per_angstrom2 += second_theta * np.outer(row, row)

    for term in model.torsions:
        primitive = Primitive("dihedral", term.atoms)
        angle = eval_primitive(primitive, coords)
        energy, first, second = _torsion_derivatives(term, angle)
        row = grad_primitive(primitive, coords).reshape(-1)
        bonded_energy += energy
        component_energy["torsion"] += energy
        if need_gradient:
            gradient_kcal_per_angstrom += first * row
        if need_hessian:
            hessian_kcal_per_angstrom2 += second * np.outer(row, row)

    for term in model.inversions:
        primitive = Primitive("out_of_plane", term.atoms)
        angle = eval_primitive(primitive, coords)
        energy, first, second = _inversion_derivatives(term, angle)
        row = grad_primitive(primitive, coords).reshape(-1)
        bonded_energy += energy
        component_energy["inversion"] += energy
        if need_gradient:
            gradient_kcal_per_angstrom += first * row
        if need_hessian:
            hessian_kcal_per_angstrom2 += second * np.outer(row, row)

    nonbonded = nonbonded_cartesian_hessian_components(
        coords / BOHR_TO_ANGSTROM,
        np.asarray(model.atomic_numbers, dtype=int),
        model.topology_bonds,
        electrostatic=False,
        uff_vdw=True,
        vdw_model="uff_12_6",
        one_four_scale=1.0,
        topology_scaling="discrete_graph",
    )
    bonded_hartree = bonded_energy * KCAL_MOL_TO_HARTREE
    energy_hartree = bonded_hartree + nonbonded.uff_vdw_energy
    gradient = None
    if need_gradient:
        gradient = (
            gradient_kcal_per_angstrom
            * KCAL_MOL_TO_HARTREE
            * BOHR_TO_ANGSTROM
            + nonbonded.uff_vdw_gradient
        )
    hessian = None
    if need_hessian:
        hessian = (
            hessian_kcal_per_angstrom2
            * KCAL_MOL_TO_HARTREE
            * BOHR_TO_ANGSTROM**2
            + nonbonded.uff_vdw
        )
        hessian = 0.5 * (hessian + hessian.T)
    components = {
        **{
            name: value * KCAL_MOL_TO_HARTREE
            for name, value in component_energy.items()
        },
        "uff_vdw": nonbonded.uff_vdw_energy,
    }
    return UffEvaluation(
        energy_hartree=float(energy_hartree),
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=hessian,
        components_hartree=components,
    )


def save_uff_model(path: Path | str, model: UffModel) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_uff_model(path: Path | str) -> UffModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("UFF model root must be a JSON object")
    return UffModel.from_dict(payload)


def _required_atomic_number(symbol: str) -> int:
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise ValueError(f"UFF cannot resolve element {symbol!r}")
    return int(number)


def _parameters_by_atomic_number(
    table: Mapping[str, UffAtomParameter],
) -> dict[int, tuple[str, ...]]:
    symbols = tuple(
        sorted(
            ((atomic_symbol(number), number) for number in range(1, 104)),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )
    grouped: dict[int, list[str]] = {}
    for label in table:
        number = 103 if label.startswith("Lw") else next(
            (number for symbol, number in symbols if label.startswith(symbol)), None
        )
        if number is None:
            raise RuntimeError(f"cannot assign UFF atom type {label} to an element")
        grouped.setdefault(number, []).append(label)
    return {number: tuple(labels) for number, labels in grouped.items()}


def _assign_atom_type(atom, numbers, graph, synthons, aromaticity, candidates) -> str:
    if not candidates:
        raise ValueError(f"classic UFF has no atom type for Z={numbers[atom]}")
    z = numbers[atom]
    neighbors = tuple(sorted(graph.adjacency[atom]))
    degree = len(neighbors)
    aromatic = atom in aromaticity.aromatic_atoms
    pi = float(synthons.pi_index(atom))
    pi_pi = float(synthons.pi_pi_index(atom))
    valence = sum(_uff_bond_order(atom, other, synthons, aromaticity) for other in neighbors)
    if z == 1:
        return _require_candidate("H_b" if degree == 2 else "H_", candidates)
    if z == 5:
        return _require_candidate("B_2" if degree <= 2 or pi >= 0.35 else "B_3", candidates)
    if z in {6, 7, 8}:
        prefix = atomic_symbol(z) + "_"
        if z == 8 and any(numbers[other] == 14 for other in neighbors) and pi < 0.35:
            return _require_candidate("O_3_z", candidates)
        suffix = "R" if aromatic else "1" if pi_pi >= 0.45 else "2" if pi >= 0.35 else "3"
        return _require_candidate(prefix + suffix, candidates)
    if z == 15:
        return _require_candidate("P_3+5" if valence >= 4.5 else "P_3+3", candidates)
    if z == 16:
        if aromatic:
            return _require_candidate("S_R", candidates)
        if pi >= 0.35:
            return _require_candidate("S_2", candidates)
        oxidation = 6 if valence >= 5.0 else 4 if valence >= 3.0 else 2
        return _require_candidate(f"S_3+{oxidation}", candidates)
    if len(candidates) == 1:
        return candidates[0]
    desired_geometry = 6 if degree >= 5 else 4 if degree == 4 else 3 if degree >= 2 else 1

    def score(label: str) -> tuple[float, float, int]:
        geometry = _type_geometry_code(label)
        oxidation = _type_oxidation(label)
        geometry_score = 8.0 if geometry is None else abs(geometry - desired_geometry)
        oxidation_score = 0.0 if oxidation is None else abs(oxidation - valence)
        return geometry_score, oxidation_score, candidates.index(label)

    return min(candidates, key=score)


def _require_candidate(label: str, candidates: Sequence[str]) -> str:
    if label not in candidates:
        raise ValueError(f"classic UFF atom type {label} is unavailable")
    return label


def _type_geometry_code(label: str) -> int | None:
    match = re.search(r"(?:_|^[A-Z][a-z]?)([1-6])", label)
    return None if match is None else int(match.group(1))


def _type_oxidation(label: str) -> int | None:
    match = re.search(r"\+(\d+)$", label)
    return None if match is None else int(match.group(1))


def _hybridization(label: str) -> str:
    if "_R" in label or "_2" in label:
        return "sp2"
    if "_1" in label or label.endswith("1+1") or label.startswith(("Ag1", "Hg1")):
        return "sp"
    if "6" in label:
        return "sp3d2"
    if "5" in label:
        return "sp3d"
    if "4" in label:
        return "sp2d"
    return "sp3"


def _uff_bond_order(left, right, synthons, aromaticity) -> float:
    pair = tuple(sorted((int(left), int(right))))
    if pair in aromaticity.aromatic_bonds:
        return 1.5
    components = synthons.bond_order_components(int(left), int(right))
    order = 1 + int(float(components.pi) >= 0.50) + int(float(components.pi_pi) >= 0.50)
    return float(order)


def _bond_rest_length(order, left, right) -> float:
    rsum = left.r1_angstrom + right.r1_angstrom
    bond_order_correction = -UFF_LAMBDA * rsum * math.log(float(order))
    difference = math.sqrt(left.electronegativity) - math.sqrt(right.electronegativity)
    electronegativity_correction = (
        left.r1_angstrom
        * right.r1_angstrom
        * difference**2
        / (
            left.electronegativity * left.r1_angstrom
            + right.electronegativity * right.r1_angstrom
        )
    )
    return rsum + bond_order_correction - electronegativity_correction


def _bond_term(pair, order, parameters) -> UffBondTerm:
    left, right = pair
    equilibrium = _bond_rest_length(order, parameters[left], parameters[right])
    force = (
        2.0
        * UFF_G
        * parameters[left].effective_charge
        * parameters[right].effective_charge
        / equilibrium**3
    )
    return UffBondTerm(tuple(pair), equilibrium, force, float(order))


def _angle_term(atoms, order12, order23, parameters, hybridization) -> UffAngleTerm:
    left, center, right = atoms
    theta0 = parameters[center].theta0_radian
    r12 = _bond_rest_length(order12, parameters[left], parameters[center])
    r23 = _bond_rest_length(order23, parameters[center], parameters[right])
    cosine0 = math.cos(theta0)
    r13 = math.sqrt(r12**2 + r23**2 - 2.0 * r12 * r23 * cosine0)
    force = (
        (2.0 * UFF_G / (r12 * r23))
        * parameters[left].effective_charge
        * parameters[right].effective_charge
        / r13**5
        * (r12 * r23)
        * (3.0 * r12 * r23 * (1.0 - cosine0**2) - r13**2 * cosine0)
    )
    periodic_order = {
        "sp": 1,
        "trigonal_bipyramid_axial": 2,
        "sp2": 3,
        "sp3d2": 4,
    }.get(hybridization, 0)
    if periodic_order:
        return UffAngleTerm(tuple(atoms), force, theta0, periodic_order)
    sine0 = math.sin(theta0)
    c2 = 1.0 / (4.0 * max(sine0**2, 1.0e-8))
    c1 = -4.0 * c2 * cosine0
    c0 = c2 * (2.0 * cosine0**2 + 1.0)
    return UffAngleTerm(tuple(atoms), force, theta0, 0, c0, c1, c2)


def _compile_angles(coords, graph, atom_types, parameters, bond_orders):
    terms: list[UffAngleTerm] = []
    for center in range(len(atom_types)):
        neighbors = tuple(sorted(graph.adjacency[center]))
        hybridization = _hybridization(atom_types[center])
        axial: frozenset[int] = frozenset()
        if len(neighbors) == 5:
            axial_pair = min(
                combinations(neighbors, 2),
                key=lambda pair: float(
                    np.dot(
                        _unit_vector(coords[pair[0]] - coords[center]),
                        _unit_vector(coords[pair[1]] - coords[center]),
                    )
                ),
            )
            axial = frozenset(axial_pair)
        for left, right in combinations(neighbors, 2):
            local_hybridization = hybridization
            if axial:
                pair = {left, right}
                if pair == set(axial):
                    local_hybridization = "trigonal_bipyramid_axial"
                elif not pair.intersection(axial):
                    local_hybridization = "sp2"
                else:
                    local_hybridization = "sp3"
            terms.append(
                _angle_term(
                    (left, center, right),
                    bond_orders[tuple(sorted((left, center)))],
                    bond_orders[tuple(sorted((center, right)))],
                    parameters,
                    local_hybridization,
                )
            )
    return tuple(terms)


def _compile_torsions(numbers, graph, synthons, atom_types, parameters, bond_orders):
    terms: list[UffTorsionTerm] = []
    for center1, center2 in sorted(tuple(sorted(pair)) for pair in graph.bonds):
        hybrid1 = _hybridization(atom_types[center1])
        hybrid2 = _hybridization(atom_types[center2])
        if hybrid1 not in {"sp2", "sp3"} or hybrid2 not in {"sp2", "sp3"}:
            continue
        combinations_about_bond = [
            (outer1, center1, center2, outer2)
            for outer1 in sorted(set(graph.adjacency[center1]) - {center2})
            for outer2 in sorted(set(graph.adjacency[center2]) - {center1})
            if outer1 != outer2
        ]
        if not combinations_about_bond:
            continue
        for atoms in combinations_about_bond:
            force, order, phase = _torsion_parameters(
                center1,
                center2,
                atoms,
                numbers,
                synthons,
                atom_types,
                parameters,
                bond_orders[(center1, center2)],
            )
            terms.append(
                UffTorsionTerm(
                    tuple(atoms), force / len(combinations_about_bond), order, phase
                )
            )
    return tuple(terms)


def _torsion_parameters(
    center1,
    center2,
    torsion_atoms,
    numbers,
    synthons,
    atom_types,
    parameters,
    bond_order,
):
    del synthons
    hybrid1 = _hybridization(atom_types[center1])
    hybrid2 = _hybridization(atom_types[center2])
    group6 = {8, 16, 34, 52, 84}
    if hybrid1 == hybrid2 == "sp3":
        force = math.sqrt(
            parameters[center1].torsion_v_kcal_mol
            * parameters[center2].torsion_v_kcal_mol
        )
        if bond_order == 1.0 and numbers[center1] in group6 and numbers[center2] in group6:
            left = 2.0 if numbers[center1] == 8 else 6.8
            right = 2.0 if numbers[center2] == 8 else 6.8
            return math.sqrt(left * right), 2, -1.0
        return force, 3, -1.0
    if hybrid1 == hybrid2 == "sp2":
        force = 5.0 * math.sqrt(
            parameters[center1].torsion_u_kcal_mol
            * parameters[center2].torsion_u_kcal_mol
        ) * (1.0 + 4.18 * math.log(bond_order))
        return force, 2, 1.0
    if bond_order == 1.0:
        sp3_center, sp2_center = (
            (center1, center2) if hybrid1 == "sp3" else (center2, center1)
        )
        if numbers[sp3_center] in group6 and numbers[sp2_center] not in group6:
            force = 5.0 * math.sqrt(
                parameters[center1].torsion_u_kcal_mol
                * parameters[center2].torsion_u_kcal_mol
            )
            return force, 2, -1.0
        if any(
            _hybridization(atom_types[outer]) == "sp2"
            for outer in (torsion_atoms[0], torsion_atoms[3])
        ):
            return 2.0, 3, -1.0
    return 1.0, 6, 1.0


def _compile_inversions(numbers, graph, atom_types, parameters, bond_orders):
    terms: list[UffInversionTerm] = []
    eligible = {6, 7, 8, 15, 33, 51, 83}
    for center, number in enumerate(numbers):
        neighbors = tuple(sorted(graph.adjacency[center]))
        if number not in eligible or len(neighbors) != 3:
            continue
        if number in {6, 7, 8} and _hybridization(atom_types[center]) != "sp2":
            continue
        carbonyl = number == 6 and any(
            numbers[neighbor] == 8
            and _hybridization(atom_types[neighbor]) == "sp2"
            and bond_orders[tuple(sorted((center, neighbor)))] >= 2.0
            for neighbor in neighbors
        )
        force, c0, c1, c2 = _inversion_parameters(number, carbonyl)
        first, second, third = neighbors
        permutations = (
            (center, first, second, third),
            (center, first, third, second),
            (center, second, third, first),
        )
        terms.extend(UffInversionTerm(atoms, force, c0, c1, c2) for atoms in permutations)
    return tuple(terms)


def _inversion_parameters(atomic_number_value, carbonyl):
    if atomic_number_value in {6, 7, 8}:
        return (50.0 if carbonyl else 6.0) / 3.0, 1.0, -1.0, 0.0
    reference_degrees = {15: 84.4339, 33: 86.9735, 51: 87.7047, 83: 90.0}
    reference = math.radians(reference_degrees[atomic_number_value])
    c2 = 1.0
    c1 = -4.0 * math.cos(reference)
    c0 = -(c1 * math.cos(reference) + c2 * math.cos(2.0 * reference))
    return (22.0 / (c0 + c1 + c2)) / 3.0, c0, c1, c2


def _angle_derivatives(term: UffAngleTerm, coords):
    left, center, right = term.atoms
    u = coords[left] - coords[center]
    v = coords[right] - coords[center]
    cosine = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))
    theta = math.acos(cosine)
    if term.order == 0:
        energy = term.force_kcal_mol * (
            term.c0 + term.c1 * cosine + term.c2 * (2.0 * cosine**2 - 1.0)
        )
        derivative_cosine = term.force_kcal_mol * (term.c1 + 4.0 * term.c2 * cosine)
        sine2 = max(0.0, 1.0 - cosine**2)
        second = term.force_kcal_mol * (
            4.0 * term.c2 * sine2 - cosine * (term.c1 + 4.0 * term.c2 * cosine)
        )
        return energy, derivative_cosine, second
    if term.order == 1:
        return (
            term.force_kcal_mol * (1.0 + cosine),
            term.force_kcal_mol,
            -term.force_kcal_mol * cosine,
        )
    order = term.order
    cos_order = math.cos(order * theta)
    energy = term.force_kcal_mol / order**2 * (1.0 - cos_order)
    derivative_cosine = -term.force_kcal_mol / order**2 * _chebyshev_derivative(
        order, cosine
    )
    return energy, derivative_cosine, term.force_kcal_mol * cos_order


def _chebyshev_derivative(order, cosine):
    if order == 2:
        return 4.0 * cosine
    if order == 3:
        return 12.0 * cosine**2 - 3.0
    if order == 4:
        return 32.0 * cosine**3 - 16.0 * cosine
    raise ValueError(f"unsupported UFF angle periodicity: {order}")


def _cosine_angle_gradient(atoms, coords):
    left, center, right = atoms
    first = coords[left] - coords[center]
    second = coords[right] - coords[center]
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        raise ValueError("UFF angle contains coincident atoms")
    u = first / first_norm
    v = second / second_norm
    cosine = float(np.clip(np.dot(u, v), -1.0, 1.0))
    result = np.zeros_like(coords)
    result[left] = (v - cosine * u) / first_norm
    result[right] = (u - cosine * v) / second_norm
    result[center] = -(result[left] + result[right])
    return result.reshape(-1)


def _unit_vector(vector):
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        raise ValueError("UFF coordination geometry contains coincident atoms")
    return values / norm


def _torsion_derivatives(term, angle):
    multiple = term.order * angle
    energy = 0.5 * term.force_kcal_mol * (
        1.0 - term.cosine_phase * math.cos(multiple)
    )
    first = (
        0.5
        * term.force_kcal_mol
        * term.cosine_phase
        * term.order
        * math.sin(multiple)
    )
    second = (
        0.5
        * term.force_kcal_mol
        * term.cosine_phase
        * term.order**2
        * math.cos(multiple)
    )
    return energy, first, second


def _inversion_derivatives(term, angle):
    energy = term.force_kcal_mol * (
        term.c0 + term.c1 * math.cos(angle) + term.c2 * math.cos(2.0 * angle)
    )
    first = -term.force_kcal_mol * (
        term.c1 * math.sin(angle) + 2.0 * term.c2 * math.sin(2.0 * angle)
    )
    second = -term.force_kcal_mol * (
        term.c1 * math.cos(angle) + 4.0 * term.c2 * math.cos(2.0 * angle)
    )
    return energy, first, second


def _term_from_dict(term_type, payload):
    data = dict(payload)
    data["atoms"] = tuple(int(value) for value in data["atoms"])
    return term_type(**data)


__all__ = [
    "UFF_EVALUATION_SCHEMA",
    "UFF_MODEL_SCHEMA",
    "UFF_REFERENCE",
    "UffAtomParameter",
    "UffEvaluation",
    "UffModel",
    "compile_uff_model",
    "compile_uff_model_from_xyzin",
    "evaluate_uff_model",
    "load_uff_model",
    "load_uff_parameters",
    "save_uff_model",
]
