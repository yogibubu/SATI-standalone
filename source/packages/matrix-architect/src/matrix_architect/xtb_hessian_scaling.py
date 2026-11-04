"""Pareto selection of transferable primitive-Hessian scale factors for xTB.

GF and SMITH own the Cartesian/primitive transformation. ARCHITECT fits only
positive, shared scale factors.  If primitive ``i`` belongs to class ``c(i)``,
the complete primitive Hessian is transformed by the congruence

    K' = D K D,  D_ii = sqrt(s_c(i)).

Consequently every off-diagonal element is scaled by the geometric mean of
its two diagonal factors and the inertia of the primitive Hessian is
preserved.  No independently fitted off-diagonal parameters are permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment, minimize

from matrix_chem.topology.elements import atomic_number
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM, HessianInput, hessian_input_from_engine
from matrix_gf.harmonic_field import transform_cartesian_hessian


@dataclass(frozen=True)
class XtbHessianScaleRequest:
    reference_hessian_path: Path
    xtb_hessian_path: Path
    primitive_manifest_path: Path
    oracle_topology_path: Path
    xtb_observables_path: Path
    output_path: Path
    protocol_manifest: Mapping[str, Any]
    protocol_sha256: str
    reference_engine: str = "gdv"
    excluded_mode_indices: tuple[int, ...] = (0,)


def _reference_hessian(request: XtbHessianScaleRequest) -> HessianInput:
    """Load a native backend Hessian or an audited LCB26 harmonic record."""
    if str(request.reference_engine).strip().lower() != "lcb26":
        return hessian_input_from_engine(
            request.reference_engine, request.reference_hessian_path
        )
    payload = json.loads(Path(request.reference_hessian_path).read_text())
    if payload.get("schema") != "matrix.lcb26.harmonic_record.v1":
        raise ValueError("LCB26 reference requires matrix.lcb26.harmonic_record.v1")
    hessian = payload.get("cartesian_hessian", {})
    if hessian.get("units") != "hartree/bohr^2":
        raise ValueError("LCB26 Hessian must be expressed in hartree/bohr^2")
    atoms = tuple(str(atom) for atom in payload.get("atoms", ()))
    coordinate_key = (
        "optimized_coordinates_angstrom_in_lcb26_frame"
        if "optimized_coordinates_angstrom_in_lcb26_frame" in payload
        else "mp2_optimized_coordinates_angstrom_in_lcb26_frame"
    )
    return HessianInput(
        atomic_numbers=np.asarray([atomic_number(atom) for atom in atoms]),
        cartesian_coordinates_bohr=np.asarray(
            payload[coordinate_key], dtype=float
        ) / 0.52917721092,
        masses_amu=np.asarray(payload["masses_amu"], dtype=float),
        cartesian_hessian=np.asarray(hessian["values"], dtype=float),
        harmonic_frequencies_cm=np.asarray(
            payload["harmonic_frequencies_cm-1"], dtype=float
        ),
        source=str(payload.get("identifier", "LCB26 harmonic record")),
        provenance={"SCHEMA": str(payload["schema"])},
    )


def scale_primitive_hessian_geometric(
    primitive_hessian: np.ndarray, primitive_scale_factors: np.ndarray
) -> np.ndarray:
    """Apply positive diagonal factors and their geometric-mean couplings."""
    matrix = np.asarray(primitive_hessian, dtype=float)
    factors = np.asarray(primitive_scale_factors, dtype=float).reshape(-1)
    if matrix.shape != (len(factors), len(factors)) or np.any(factors <= 0.0):
        raise ValueError("primitive Hessian and positive scale factors are inconsistent")
    roots = np.sqrt(factors)
    return roots[:, None] * matrix * roots[None, :]


def _normal_modes(hessian: np.ndarray, masses: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse_sqrt_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    weighted = inverse_sqrt_mass[:, None] * hessian * inverse_sqrt_mass[None, :]
    reduced = basis.T @ (0.5 * (weighted + weighted.T)) @ basis
    eigenvalues, modes = np.linalg.eigh(reduced)
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * HESSIAN_EIGENVALUE_TO_CM
    return frequencies, basis @ modes


def _proper_alignment(moving: np.ndarray, reference: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, float]:
    weights = masses / np.sum(masses)
    left = moving - np.sum(weights[:, None] * moving, axis=0)
    right = reference - np.sum(weights[:, None] * reference, axis=0)
    u, _singular, vt = np.linalg.svd((weights[:, None] * left).T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    residual = left @ rotation - right
    rmsd = float(np.sqrt(np.sum(weights * np.sum(residual * residual, axis=1))))
    return rotation, rmsd


def _rotate_modes(modes: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    shaped = modes.reshape((-1, 3, modes.shape[1]))
    return np.einsum("aik,ij->ajk", shaped, rotation).reshape(modes.shape)


def _oracle_atom_types(
    atoms: tuple[str, ...], topology_payload: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return compact, deterministic ORACLE environment types.

    Aromatic atoms form element-specific types.  Hydrogens inherit the type
    of their single neighbour.  Other atoms are classified by element,
    valence degree, and the sorted element/aromaticity signatures of their
    neighbours.  The definition is graph based and contains no molecule-
    specific atom indices.
    """
    entries = topology_payload.get("entries", ())
    if len(entries) != 1:
        raise ValueError("ORACLE topology artifact must contain one molecular entry")
    entry = entries[0]
    aromatic = {int(value) - 1 for value in entry.get("aromatic_atoms", ())}
    neighbours: list[list[int]] = [[] for _ in atoms]
    for raw_i, raw_j in entry.get("bonds", ()):
        i, j = int(raw_i) - 1, int(raw_j) - 1
        neighbours[i].append(j); neighbours[j].append(i)
    preliminary: list[str] = []
    for atom, element in enumerate(atoms):
        if atom in aromatic:
            preliminary.append(f"{element}_ar")
        elif element == "H":
            preliminary.append("H")
        else:
            environment = sorted(
                f"{atoms[other]}{'_ar' if other in aromatic else ''}"
                for other in neighbours[atom]
            )
            preliminary.append(f"{element}_d{len(neighbours[atom])}_{'-'.join(environment)}")
    result = list(preliminary)
    for atom, element in enumerate(atoms):
        if element == "H":
            if len(neighbours[atom]) != 1:
                raise ValueError("hydrogen atom type requires exactly one bonded neighbour")
            result[atom] = f"H_on_{preliminary[neighbours[atom][0]]}"
    return tuple(result)


def _xtb_scaling_observables(
    payload: Mapping[str, Any], atoms: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Read self-consistent SCC charges and Mayer orders from LINK-xTB only."""
    if payload.get("schema") != "matrix.gfn2.scaling_observables.v1":
        raise ValueError("xTB descriptors require matrix.gfn2.scaling_observables.v1")
    if payload.get("backend") != "LINK-XTB/MATRIX-GFN2":
        raise ValueError("xTB descriptors must originate from LINK-XTB/MATRIX-GFN2")
    if str(payload.get("charge_model", "")).upper() != "GFN2-SCC":
        raise ValueError("xTB geometry/Hessian scaling requires native GFN2-SCC charges")
    if str(payload.get("bond_order_model", "")).upper() != "MAYER":
        raise ValueError("xTB geometry/Hessian scaling requires Mayer bond orders")
    if tuple(str(atom) for atom in payload.get("atoms", ())) != atoms:
        raise ValueError("xTB descriptor and Hessian atom ordering differ")
    charges = np.asarray(payload.get("charges_electron", ()), dtype=float).reshape(-1)
    if charges.shape != (len(atoms),):
        raise ValueError("xTB SCC charge vector has inconsistent dimensions")
    mayer = np.zeros((len(atoms), len(atoms)), dtype=float)
    for row in payload.get("mayer_bond_orders", ()):
        left, right, value = row
        i, j = int(left) - 1, int(right) - 1
        if not (0 <= i < len(atoms) and 0 <= j < len(atoms) and i != j):
            raise ValueError("invalid one-based Mayer pair in xTB descriptor artifact")
        mayer[i, j] = mayer[j, i] = float(value)
    if not np.all(np.isfinite(charges)) or not np.all(np.isfinite(mayer)):
        raise ValueError("xTB descriptor artifact contains non-finite values")
    return charges, mayer


def _oracle_perceived_bond_orders(topology_payload: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    """Read perception bond orders with the artifact's one-based atom indices."""
    entries = topology_payload.get("entries", ())
    if len(entries) != 1:
        raise ValueError("ORACLE topology artifact must contain one molecular entry")
    orders = {}
    for row in entries[0].get("bond_order_components", ()):
        key = tuple(sorted(int(v) for v in row["atoms"]))
        order = float(row["bond_order"])
        if len(key) != 2 or not np.isfinite(order):
            raise ValueError("invalid ORACLE perception bond order")
        if key in orders:
            raise ValueError("duplicate ORACLE perception bond order")
        orders[key] = order
    return orders


def xtb_primitive_features(
    manifest: Mapping[str, Any],
    atoms: tuple[str, ...],
    atom_types: tuple[str, ...],
    charges: np.ndarray,
    mayer: np.ndarray,
    perceived_bond_orders: Mapping[tuple[int, int], float] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build role-aware, continuous primitive descriptors.

    Atom-type modifiers are categorical, whereas bond order and optional SCC
    charge response remain continuous.  Missing learned type modifiers can
    therefore fall back exactly to the primitive-family intercept.
    """
    coordinates = manifest["coordinates"]
    bond_degree = [0] * len(atoms)
    for left, right in coordinates.get("R", ()):
        bond_degree[int(left) - 1] += 1
        bond_degree[int(right) - 1] += 1

    def record(family: str, indices: tuple[int, ...], bonds: tuple[tuple[int, int], ...]) -> dict[str, Any]:
        orders = np.asarray([max(float(mayer[i, j]), 1.0e-6) for i, j in bonds])
        return {
            "family": family,
            "atom_indices": [value + 1 for value in indices],
            "elements": [atoms[value] for value in indices],
            "atom_types": [atom_types[value] for value in indices],
            "perceived_bond_order": (
                perceived_bond_orders.get(tuple(sorted(value + 1 for value in indices)))
                if family == "R" and perceived_bond_orders is not None else None
            ),
            "terminal_sulfur_oxygen_stretch": bool(
                family == "R" and sorted(atoms[value] for value in indices) == ["O", "S"]
                and any(atoms[value] == "O" and bond_degree[value] == 1 for value in indices)
            ),
            "mayer_bond_orders": orders.tolist(),
            "geometric_mean_mayer_order": float(np.exp(np.mean(np.log(orders)))),
            "mean_abs_scc_charge": float(np.mean(np.abs(charges[list(indices)]))),
        }

    result: list[dict[str, Any]] = []
    for raw_i, raw_j in coordinates.get("R", ()):
        i, j = int(raw_i) - 1, int(raw_j) - 1
        result.append(record("R", (i, j), ((i, j),)))
    for raw_i, raw_j, raw_k in coordinates.get("A", ()):
        i, j, k = int(raw_i) - 1, int(raw_j) - 1, int(raw_k) - 1
        result.append(record("A", (i, j, k), ((i, j), (j, k))))
    for raw in coordinates.get("L", ()):
        raw_i, raw_j, raw_k = raw["atoms"] if isinstance(raw, Mapping) else raw
        i, j, k = int(raw_i) - 1, int(raw_j) - 1, int(raw_k) - 1
        feature = record("L", (i, j, k), ((i, j), (j, k)))
        result.extend((dict(feature, component="x"), dict(feature, component="y")))
    for raw in coordinates.get("SOOP", ()):
        center, moving = int(raw["center"]) - 1, int(raw["moving"]) - 1
        supports = tuple(int(value) - 1 for value in raw["plane"] if int(value) - 1 != center)
        partners = tuple(dict.fromkeys((moving, *supports)))
        result.append(record("SOOP", (center, moving, *supports), tuple((center, value) for value in partners)))
    for raw_i, raw_j, raw_k, raw_l in coordinates.get("D", ()):
        i, j, k, l = (int(value) - 1 for value in (raw_i, raw_j, raw_k, raw_l))
        result.append(record("D", (i, j, k, l), ((i, j), (j, k), (k, l))))
    return tuple(result)


def _class_keys(
    manifest: Mapping[str, Any], atoms: tuple[str, ...], atom_types: tuple[str, ...]
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    coordinates = manifest["coordinates"]
    bonds = [tuple(int(v) - 1 for v in pair) for pair in coordinates.get("R", ())]
    degree = [0] * len(atoms)
    for i, j in bonds:
        degree[i] += 1; degree[j] += 1
    rings = [set(int(v) - 1 for v in ring) for ring in manifest.get("rings", ())]

    families: list[tuple[Any, ...]] = []
    chemical: list[tuple[Any, ...]] = []
    topological: list[tuple[Any, ...]] = []
    for raw_i, raw_j in coordinates.get("R", ()):
        i, j = int(raw_i) - 1, int(raw_j) - 1
        element_pair = tuple(sorted((atom_types[i], atom_types[j])))
        shared_ring = any(i in ring and j in ring for ring in rings)
        families.append(("R",)); chemical.append(("R", *element_pair))
        topological.append(("R", *element_pair, shared_ring, *sorted((degree[i], degree[j]))))
    for raw_i, raw_j, raw_k in coordinates.get("A", ()):
        i, j, k = int(raw_i) - 1, int(raw_j) - 1, int(raw_k) - 1
        outer = tuple(sorted((atom_types[i], atom_types[k])))
        same_ring = any(i in ring and j in ring and k in ring for ring in rings)
        families.append(("A",)); chemical.append(("A", atom_types[j], *outer))
        topological.append(("A", atom_types[j], *outer, degree[j], same_ring))
    for raw in coordinates.get("L", ()):
        raw_i, raw_j, raw_k = raw["atoms"] if isinstance(raw, Mapping) else raw
        i, j, k = int(raw_i) - 1, int(raw_j) - 1, int(raw_k) - 1
        outer = tuple(sorted((atom_types[i], atom_types[k])))
        for component in ("x", "y"):
            families.append(("L",)); chemical.append(("L", atom_types[j], *outer))
            topological.append(("L", atom_types[j], *outer, component))
    for record in coordinates.get("SOOP", ()):
        center, moving = int(record["center"]) - 1, int(record["moving"]) - 1
        supports = [int(v) - 1 for v in record["plane"] if int(v) - 1 != center]
        support_elements = tuple(sorted(atom_types[v] for v in supports))
        same_ring = any(center in ring and moving in ring for ring in rings)
        families.append(("SOOP",))
        chemical.append(("SOOP", atom_types[center], atom_types[moving], *support_elements))
        topological.append(("SOOP", atom_types[center], atom_types[moving], *support_elements, degree[center], same_ring))
    for raw_i, raw_j, raw_k, raw_l in coordinates.get("D", ()):
        i, j, k, l = (int(value) - 1 for value in (raw_i, raw_j, raw_k, raw_l))
        terminal = tuple(sorted((atom_types[i], atom_types[l])))
        central = tuple(sorted((atom_types[j], atom_types[k])))
        families.append(("D",)); chemical.append(("D", *central, *terminal))
        topological.append(("D", *central, *terminal, any(j in ring and k in ring for ring in rings)))
    global_keys = tuple(("ALL",) for _ in families)
    return {
        "global": global_keys,
        "family": tuple(families),
        "chemical": tuple(chemical),
        "topological": tuple(topological),
    }


def _groups_from_keys(keys: tuple[tuple[Any, ...], ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    index: dict[tuple[Any, ...], int] = {}
    labels: list[str] = []
    groups = np.empty(len(keys), dtype=int)
    for primitive, key in enumerate(keys):
        if key not in index:
            index[key] = len(index)
            labels.append(":".join(str(value) for value in key))
        groups[primitive] = index[key]
    return groups, tuple(labels)


def _nested_chemical_keys(
    family_keys: tuple[tuple[Any, ...], ...],
    chemical_keys: tuple[tuple[Any, ...], ...],
    diagonal_reference: np.ndarray,
    diagonal_raw: np.ndarray,
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    """Build a deterministic 3..N hierarchy of chemically typed partitions.

    Within each primitive family one chemical type remains the shared baseline.
    The other types are released in descending order of their robust diagonal
    correction relative to that family.  Thus factor count is a genuine Pareto
    complexity variable rather than a jump from three factors to one factor per
    chemical type.
    """
    valid = (diagonal_reference > 1.0e-10) & (diagonal_raw > 1.0e-10)
    log_ratio = np.zeros(len(diagonal_raw))
    log_ratio[valid] = np.log(diagonal_reference[valid] / diagonal_raw[valid])
    release: list[tuple[float, tuple[Any, ...]]] = []
    baselines: set[tuple[Any, ...]] = set()
    for family in sorted(set(family_keys)):
        family_members = np.asarray([key == family for key in family_keys])
        types = sorted(set(key for key, member in zip(chemical_keys, family_members) if member))
        family_values = log_ratio[family_members & valid]
        family_center = float(np.median(family_values)) if family_values.size else 0.0
        scored: list[tuple[float, tuple[Any, ...]]] = []
        for key in types:
            members = np.asarray([candidate == key for candidate in chemical_keys]) & valid
            center = float(np.median(log_ratio[members])) if np.any(members) else family_center
            score = abs(center - family_center) * np.sqrt(max(1, int(np.sum(members))))
            scored.append((score, key))
        baseline = min(scored, key=lambda item: (item[0], item[1]))[1]
        baselines.add(baseline)
        release.extend(item for item in scored if item[1] != baseline)
    release.sort(key=lambda item: (-item[0], item[1]))
    hierarchy: dict[str, tuple[tuple[Any, ...], ...]] = {}
    selected: set[tuple[Any, ...]] = set()
    base_count = len(set(family_keys))
    for extra, (_score, key) in enumerate(release, start=1):
        selected.add(key)
        keys = tuple(
            ("TYPE", *chemical) if chemical in selected else ("BASE", *family)
            for family, chemical in zip(family_keys, chemical_keys)
        )
        hierarchy[f"chemical-{base_count + extra:02d}"] = keys
    return hierarchy


def _first_complexity_plateau(
    candidates: list[dict[str, Any]],
    *,
    minimum_factor_count: int = 5,
    consecutive_small_gains: int = 2,
    maximum_relative_rms_gain: float = 0.06,
    minimum_mean_overlap: float = 0.85,
) -> dict[str, Any] | None:
    """Return the first pre-registered parsimonious point on a fit path.

    The curve is inspected strictly in increasing model complexity.  Once the
    best frequency fit has retained acceptable modal identity and the last
    ``consecutive_small_gains`` additions each improve frequency RMS by no
    more than ``maximum_relative_rms_gain``, later partitions are not opened.
    This is an early-stopping rule, not a post-hoc choice among all larger
    models, so a later training-set jump cannot induce chemical overfitting.
    """
    best_by_count: list[dict[str, Any]] = []
    for count in sorted({int(item["factor_count"]) for item in candidates if item["factor_count"]}):
        subset = [item for item in candidates if int(item["factor_count"]) == count]
        best_by_count.append(min(subset, key=lambda item: item["frequency_rms_error_cm-1"]))
    if not best_by_count or int(best_by_count[-1]["factor_count"]) < minimum_factor_count:
        return None
    current = best_by_count[-1]
    if float(current["mean_assigned_mode_overlap"]) < minimum_mean_overlap:
        return None
    if len(best_by_count) <= consecutive_small_gains:
        return None
    gains = []
    for left, right in zip(
        best_by_count[-consecutive_small_gains - 1 : -1],
        best_by_count[-consecutive_small_gains :],
        strict=True,
    ):
        previous = max(float(left["frequency_rms_error_cm-1"]), 1.0e-12)
        gains.append(max(0.0, (previous - float(right["frequency_rms_error_cm-1"])) / previous))
    if all(gain <= maximum_relative_rms_gain for gain in gains):
        return {**current, "plateau_relative_rms_gains": tuple(gains)}
    return None


def fit_xtb_hessian_scales(request: XtbHessianScaleRequest) -> dict:
    """Fit and Pareto-select the smallest useful xTB scaling partition."""
    protocol = dict(request.protocol_manifest)
    if protocol.get("protocol_id") != "xtb-primitive-hessian-scaling.v2" or protocol.get("status") != "approved":
        raise ValueError("Keymaker requires the approved xTB Hessian-scaling protocol")
    if len(request.protocol_sha256) != 64:
        raise ValueError("Keymaker requires the installed protocol artifact hash")

    reference = _reference_hessian(request)
    xtb_payload = json.loads(Path(request.xtb_hessian_path).read_text())
    if xtb_payload.get("schema") != "matrix.link.final_cartesian_hessian.v1":
        raise ValueError("xTB Hessian must use matrix.link.final_cartesian_hessian.v1")
    atoms = tuple(str(atom) for atom in xtb_payload["atoms"])
    numbers = np.asarray([atomic_number(atom) for atom in atoms], dtype=int)
    if not np.array_equal(numbers, reference.atomic_numbers):
        raise ValueError("reference and xTB atom ordering differ")
    xtb_coordinates = np.asarray(xtb_payload["coordinates_angstrom"], dtype=float) / 0.52917721092
    xtb_hessian = np.asarray(xtb_payload["hessian"], dtype=float)
    if xtb_hessian.shape != reference.cartesian_hessian.shape:
        raise ValueError("reference and xTB Hessian dimensions differ")
    manifest = json.loads(Path(request.primitive_manifest_path).read_text())
    topology_payload = json.loads(Path(request.oracle_topology_path).read_text())
    atom_types = _oracle_atom_types(atoms, topology_payload)
    observables_payload = json.loads(Path(request.xtb_observables_path).read_text())
    xtb_charges, xtb_mayer = _xtb_scaling_observables(observables_payload, atoms)
    primitive_features = xtb_primitive_features(
        manifest, atoms, atom_types, xtb_charges, xtb_mayer,
        perceived_bond_orders=_oracle_perceived_bond_orders(topology_payload),
    )
    ref_transform = transform_cartesian_hessian(
        reference.cartesian_hessian, reference.cartesian_coordinates_bohr,
        reference.masses_amu, manifest,
    )
    xtb_transform = transform_cartesian_hessian(
        xtb_hessian, xtb_coordinates, reference.masses_amu, manifest,
    )
    if ref_transform.primitive_labels != xtb_transform.primitive_labels:
        raise ValueError("reference and xTB primitive contracts differ")

    target_frequencies, target_modes = _normal_modes(
        reference.cartesian_hessian, reference.masses_amu,
        ref_transform.redundant_hessian.vibrational_basis,
    )
    rotation, geometry_rmsd = _proper_alignment(
        xtb_coordinates, reference.cartesian_coordinates_bohr, reference.masses_amu
    )
    included = np.ones(len(target_frequencies), dtype=bool)
    included[list(request.excluded_mode_indices)] = False
    ordinary = np.flatnonzero(included)
    target_ordinary_modes = target_modes[:, ordinary]
    raw_k = xtb_transform.redundant_hessian.force_constants
    reference_k = ref_transform.redundant_hessian.force_constants
    b_matrix = xtb_transform.b_matrix
    basis = xtb_transform.redundant_hessian.vibrational_basis

    def evaluate(group_index: np.ndarray, log_scales: np.ndarray) -> tuple[float, float, float, dict[str, Any]]:
        primitive_scales = np.exp(log_scales[group_index])
        scaled_k = scale_primitive_hessian_geometric(raw_k, primitive_scales)
        hessian = b_matrix.T @ scaled_k @ b_matrix
        frequencies, modes = _normal_modes(hessian, reference.masses_amu, basis)
        rotated_modes = _rotate_modes(modes, rotation)
        overlap = np.abs(target_ordinary_modes.T @ rotated_modes[:, ordinary])
        rows, columns = linear_sum_assignment(-overlap)
        assigned = overlap[rows, columns]
        frequency_rms = float(np.sqrt(np.mean((frequencies[included] - target_frequencies[included]) ** 2)))
        mean_overlap = float(np.mean(assigned))
        displacement = float(np.sqrt(np.mean(log_scales * log_scales))) if len(log_scales) else 0.0
        details = {
            "frequencies": frequencies, "modes": rotated_modes, "assigned": assigned,
            "scaled_k": scaled_k, "hessian": hessian,
        }
        return frequency_rms, mean_overlap, displacement, details

    profiles = (
        ("frequency", 1.0, 0.0, 0.0),
        ("balanced", 1.0, 1.0, 0.01),
        ("transition_3", 1.0, 3.0, 0.01),
        ("mode", 1.0, 10.0, 0.01),
    )
    candidates: list[dict[str, Any]] = []
    zero_groups = np.zeros(len(raw_k), dtype=int)
    raw_metrics = evaluate(zero_groups, np.zeros(1))
    candidates.append({
        "partition": "unscaled", "profile": "raw", "factor_count": 0,
        "group_index": zero_groups, "group_labels": (), "log_scales": np.zeros(0),
        "frequency_rms_error_cm-1": raw_metrics[0],
        "mean_assigned_mode_overlap": raw_metrics[1],
        "normalized_scale_displacement": 0.0, "solver_success": True,
    })

    diagonal_ref = np.diag(reference_k)
    diagonal_raw = np.diag(raw_k)
    schemes = _class_keys(manifest, atoms, atom_types)
    nested = _nested_chemical_keys(
        schemes["family"], schemes["chemical"], diagonal_ref, diagonal_raw
    )
    ordered_schemes: dict[str, tuple[tuple[Any, ...], ...]] = {
        "global": schemes["global"],
        "family": schemes["family"],
        **nested,
        "chemical": schemes["chemical"],
        "topological": schemes["topological"],
    }
    plateau_selected: dict[str, Any] | None = None
    for partition, keys in ordered_schemes.items():
        group_index, group_labels = _groups_from_keys(keys)
        count = len(group_labels)
        initial = np.zeros(count)
        valid_ratio = (diagonal_ref > 1.0e-10) & (diagonal_raw > 1.0e-10)
        for group in range(count):
            members = (group_index == group) & valid_ratio
            if np.any(members):
                initial[group] = np.log(np.clip(np.median(diagonal_ref[members] / diagonal_raw[members]), 0.2, 5.0))
        for profile, wf, wo, wd in profiles:
            def objective(values: np.ndarray) -> float:
                frequency_rms, mean_overlap, displacement, _details = evaluate(group_index, values)
                return wf * (frequency_rms / 10.0) ** 2 + wo * ((1.0 - mean_overlap) / 0.20) ** 2 + wd * displacement**2
            if profile == "frequency":
                def spectral_residual(values: np.ndarray) -> np.ndarray:
                    primitive_scales = np.exp(values[group_index])
                    scaled_k = scale_primitive_hessian_geometric(raw_k, primitive_scales)
                    hessian = b_matrix.T @ scaled_k @ b_matrix
                    frequencies, _modes = _normal_modes(hessian, reference.masses_amu, basis)
                    return (frequencies[included] - target_frequencies[included]) / 10.0
                solution = least_squares(
                    spectral_residual, initial, bounds=(-2.0, 2.0), method="trf",
                    max_nfev=2000, ftol=1.0e-10, xtol=1.0e-10, gtol=1.0e-10,
                )
            else:
                solution = minimize(
                    objective, initial, method="L-BFGS-B", bounds=[(-2.0, 2.0)] * count,
                    options={"maxiter": 500, "maxfun": 50000, "ftol": 1.0e-9, "gtol": 1.0e-7},
                )
            frequency_rms, mean_overlap, displacement, _details = evaluate(group_index, solution.x)
            candidates.append({
                "partition": partition, "profile": profile, "factor_count": count,
                "group_index": group_index, "group_labels": group_labels,
                "log_scales": solution.x,
                "frequency_rms_error_cm-1": frequency_rms,
                "mean_assigned_mode_overlap": mean_overlap,
                "normalized_scale_displacement": displacement,
                "solver_success": bool(solution.success),
            })
        plateau_selected = _first_complexity_plateau(candidates)
        if plateau_selected is not None:
            break

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        weak = (
            left["frequency_rms_error_cm-1"] <= right["frequency_rms_error_cm-1"]
            and left["mean_assigned_mode_overlap"] >= right["mean_assigned_mode_overlap"]
            and left["normalized_scale_displacement"] <= right["normalized_scale_displacement"]
            and left["factor_count"] <= right["factor_count"]
        )
        strict = (
            left["frequency_rms_error_cm-1"] < right["frequency_rms_error_cm-1"]
            or left["mean_assigned_mode_overlap"] > right["mean_assigned_mode_overlap"]
            or left["normalized_scale_displacement"] < right["normalized_scale_displacement"]
            or left["factor_count"] < right["factor_count"]
        )
        return bool(weak and strict)

    for candidate in candidates:
        candidate["accepted"] = bool(
            candidate["frequency_rms_error_cm-1"] <= 10.0
            and candidate["mean_assigned_mode_overlap"] >= 0.85
        )
        candidate["dominated"] = any(
            other is not candidate and dominates(other, candidate) for other in candidates
        )
    frontier = [candidate for candidate in candidates if not candidate["dominated"]]
    accepted = [candidate for candidate in frontier if candidate["accepted"]]
    if plateau_selected is not None:
        selected = next(
            candidate for candidate in candidates
            if candidate["partition"] == plateau_selected["partition"]
            and candidate["profile"] == plateau_selected["profile"]
        )
    elif accepted:
        selected = min(accepted, key=lambda item: (
            item["factor_count"], item["normalized_scale_displacement"], item["frequency_rms_error_cm-1"]
        ))
    else:
        selected = min(frontier, key=lambda item: (
            (item["frequency_rms_error_cm-1"] / 10.0) ** 2
            + ((max(0.0, 0.85 - item["mean_assigned_mode_overlap"])) / 0.15) ** 2
            + 0.01 * item["normalized_scale_displacement"] ** 2
            + 0.01 * item["factor_count"]
        ))
    if selected["factor_count"]:
        selected_metrics = evaluate(selected["group_index"], selected["log_scales"])
        selected_scales = np.exp(selected["log_scales"])
    else:
        selected_metrics = raw_metrics
        selected_scales = np.ones(0)
    _rms, _overlap, _disp, selected_details = selected_metrics
    primitive_scale = (
        selected_scales[selected["group_index"]] if selected["factor_count"] else np.ones(len(raw_k))
    )
    diagonal_relative_error = float(
        np.linalg.norm(np.diag(selected_details["scaled_k"]) - diagonal_ref)
        / max(np.linalg.norm(diagonal_ref), 1.0e-30)
    )

    def serializable(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "partition": candidate["partition"], "profile": candidate["profile"],
            "factor_count": candidate["factor_count"],
            "frequency_rms_error_cm-1": candidate["frequency_rms_error_cm-1"],
            "mean_assigned_mode_overlap": candidate["mean_assigned_mode_overlap"],
            "normalized_scale_displacement": candidate["normalized_scale_displacement"],
            "accepted": candidate["accepted"], "dominated": candidate["dominated"],
            "solver_success": candidate["solver_success"],
        }

    payload = {
        "schema": "matrix.architect.xtb_primitive_hessian_scaling.v2",
        "status": "COMPLETED",
        "protocol": {"id": protocol["protocol_id"], "sha256": request.protocol_sha256},
        "inputs": {
            "reference_hessian": str(request.reference_hessian_path),
            "xtb_hessian": str(request.xtb_hessian_path),
            "primitive_manifest": str(request.primitive_manifest_path),
            "oracle_topology": str(request.oracle_topology_path),
            "xtb_observables": str(request.xtb_observables_path),
        },
        "primitive_count": len(raw_k),
        "primitive_rank": xtb_transform.redundant_hessian.primitive_rank,
        "primitive_labels": list(xtb_transform.primitive_labels),
        "atom_types": list(atom_types),
        "unique_atom_types": sorted(set(atom_types)),
        "geometry_mass_weighted_rmsd_bohr": geometry_rmsd,
        "excluded_large_amplitude_modes": list(request.excluded_mode_indices),
        "reference_frequencies_cm-1": target_frequencies.tolist(),
        "raw_xtb_frequencies_cm-1": raw_metrics[3]["frequencies"].tolist(),
        "scaled_xtb_frequencies_cm-1": selected_details["frequencies"].tolist(),
        "raw_metrics": {
            "frequency_rms_error_cm-1": raw_metrics[0],
            "mean_assigned_mode_overlap": raw_metrics[1],
        },
        "selected_metrics": {
            "frequency_rms_error_cm-1": selected_metrics[0],
            "mean_assigned_mode_overlap": selected_metrics[1],
            "minimum_assigned_mode_overlap": float(np.min(selected_details["assigned"])),
            "primitive_diagonal_relative_l2_error": diagonal_relative_error,
        },
        "selected_model": {
            "partition": selected["partition"], "profile": selected["profile"],
            "factor_count": selected["factor_count"],
            "class_labels": list(selected["group_labels"]),
            "class_scale_factors": selected_scales.tolist(),
            "primitive_class_index": selected["group_index"].tolist() if selected["factor_count"] else [-1] * len(raw_k),
            "primitive_scale_factors": primitive_scale.tolist(),
        },
        "self_consistent_descriptors": {
            "source_schema": observables_payload["schema"],
            "backend": observables_payload["backend"],
            "charge_model": observables_payload["charge_model"],
            "bond_order_model": observables_payload["bond_order_model"],
            "atom_types": list(atom_types),
            "primitive_features": list(primitive_features),
            "external_qm_observables_used": False,
        },
        "pareto": {
            "descriptors": ["frequency_rms_error_cm-1", "mean_assigned_mode_overlap", "normalized_scale_displacement", "factor_count"],
            "acceptance": {"maximum_frequency_rms_cm-1": 10.0, "minimum_mean_mode_overlap": 0.85},
            "candidates": [serializable(item) for item in candidates],
            "frontier": [f"{item['partition']}:{item['profile']}" for item in frontier],
            "selection_rule": (
                "first sequential complexity plateau with modal identity retained; "
                "otherwise least factor count among accepted non-dominated candidates"
            ),
            "plateau_stop": {
                "minimum_factor_count": 5,
                "consecutive_small_gains": 2,
                "maximum_relative_frequency_rms_gain": 0.06,
                "minimum_mean_mode_overlap": 0.85,
                "detected": plateau_selected is not None,
                "selected_factor_count": (
                    int(plateau_selected["factor_count"])
                    if plateau_selected is not None else None
                ),
                "observed_relative_frequency_rms_gains": (
                    list(plateau_selected["plateau_relative_rms_gains"])
                    if plateau_selected is not None else []
                ),
            },
        },
        "off_diagonal_policy": "K_scaled[i,j] = sqrt(s[class(i)]*s[class(j)]) * K_xtb[i,j]",
        "scaled_primitive_hessian": selected_details["scaled_k"].tolist(),
        "scaled_cartesian_hessian_hartree_per_bohr2": selected_details["hessian"].tolist(),
    }
    Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(request.output_path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = [
    "XtbHessianScaleRequest", "fit_xtb_hessian_scales",
    "scale_primitive_hessian_geometric", "xtb_primitive_features",
]
