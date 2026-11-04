from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from matrix_chem import (
    BOHR_TO_ANGSTROM,
    MolecularGeometry,
    atomic_mass,
    expected_vibrational_mode_count,
    is_linear_geometry,
)
from matrix_chem.geometry_io import GeometryParseError, normalize_atom_symbol
from matrix_chem.topology.elements import atomic_number

from .dispersion import resolve_pyscf_dispersion_contract


PYSCF_OUTPUT_FORMAT = "pyscf_output"
PYSCF_RESULT_PREFIX = "MATRIX_PYSCF_RESULT "
_SCF_ENERGY_RE = re.compile(
    r"converged\s+SCF\s+energy\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)",
    re.I,
)


@dataclass(frozen=True)
class PySCFOutputSummary:
    path: Path
    geometry: MolecularGeometry | None
    charge: int
    multiplicity: int
    method: str
    reference: str
    basis: str
    dispersion_contract: str | None
    final_energy_hartree: float | None
    gradient_hartree_per_bohr: np.ndarray | None
    converged: bool
    normal_termination: bool
    scf_cycles: int = 0


@dataclass(frozen=True)
class PySCFPointResult:
    path: Path
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None
    normal_termination: bool
    scf_cycles: int = 0
    hessian_hartree_per_bohr2: np.ndarray | None = None


def write_pyscf_point_input(
    output: Path | str,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    method: str = "HF",
    reference: str | None = None,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    dispersion_contract: str | None = None,
    accelerator: str = "density_fitting",
    _include_gradient: bool = True,
    _include_hessian: bool = False,
    _include_population: bool = False,
    _scf_guess_path: Path | str | None = None,
    _scf_state_path: Path | str | None = None,
) -> Path:
    """Write a self-reporting PySCF energy/gradient job for MATRIX workflows."""

    target = Path(output)
    if target.name.lower() == "pyscf.py":
        raise GeometryParseError(
            "PySCF input cannot be named pyscf.py because it would shadow the installed pyscf package"
        )
    coords = np.asarray(coordinates_angstrom, dtype=float)
    atoms_tuple = tuple(normalize_atom_symbol(atom) for atom in atoms)
    if coords.shape != (len(atoms_tuple), 3):
        raise GeometryParseError("PySCF point coordinates must have shape natoms x 3")
    payload = [
        [atom, [float(xyz[0]), float(xyz[1]), float(xyz[2])]]
        for atom, xyz in zip(atoms_tuple, coords, strict=True)
    ]
    dispersion = (
        None
        if dispersion_contract is None
        else resolve_pyscf_dispersion_contract(dispersion_contract).to_dict()
    )
    resolved_reference = _resolve_pyscf_reference(
        method=method,
        multiplicity=multiplicity,
        reference=reference,
    )
    resolved_accelerator = str(accelerator).strip().casefold() or "density_fitting"
    if resolved_accelerator not in {"none", "density_fitting"}:
        raise GeometryParseError(
            "unsupported PySCF accelerator in the frozen protocol: "
            f"{accelerator!r}"
        )
    requested_symmetry = os.environ.get("MATRIX_PYSCF_SYMMETRY_GROUP", "").strip()
    symmetry_literal = "True" if not requested_symmetry else repr(requested_symmetry)
    schema = "matrix.pyscf.result.v1"
    hessian_source = ""
    hessian_result = ""
    population_source = ""
    population_result = ""
    if _include_hessian:
        schema = "matrix.pyscf.hessian_result.v1"
        hessian_source = '''hessian_tensor = np.asarray(mf.Hessian().kernel(), dtype=float)
hessian_matrix_raw = hessian_tensor.transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
hessian_antisymmetry_max = float(np.max(np.abs(hessian_matrix_raw - hessian_matrix_raw.T)))
hessian_matrix = 0.5 * (hessian_matrix_raw + hessian_matrix_raw.T)
analysis = thermo.harmonic_analysis(mol, hessian_tensor, imaginary_freq=False)
frequencies = np.asarray(analysis["freq_wavenumber"], dtype=float).reshape(-1)
'''
        hessian_result = '''    "hessian_hartree_per_bohr2": hessian_matrix.tolist(),
    "raw_hessian_antisymmetry_max_hartree_per_bohr2": hessian_antisymmetry_max,
    "harmonic_frequencies_cm": frequencies.tolist(),
'''
    if _include_population:
        population_source = '''from matrix_pyscf.population import population_observables_from_pyscf
population = population_observables_from_pyscf(mol, mf.make_rdm1())
'''
        population_result = '''    "hirshfeld_charges": population.hirshfeld_charges.tolist(),
    "cm5_charges": population.cm5_charges.tolist(),
    "mayer_bond_orders": population.mayer_bond_orders.tolist(),
    "electron_count": population.electron_count,
    "population_charge": population.charge,
'''
    gradient_source = (
        "gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)"
        if _include_gradient
        else "gradient = None"
    )
    gradient_result = (
        '    "gradient_hartree_per_bohr": gradient.tolist(),\n'
        if _include_gradient
        else ""
    )
    guess_path = "" if _scf_guess_path is None else str(Path(_scf_guess_path).resolve())
    state_path = "" if _scf_state_path is None else str(Path(_scf_state_path).resolve())
    source = f'''from __future__ import annotations

import json
import numpy as np
import pyscf
from pyscf import dft, gto, scf
from pyscf.hessian import thermo
{"import dftd3.pyscf as d3disp" if dispersion is not None else ""}

atoms = {json.dumps(payload)}
method = {json.dumps(str(method).strip() or "HF")}
reference = {json.dumps(resolved_reference)}
basis = {json.dumps(str(basis).strip() or "sto-3g")}
charge = {int(charge)}
multiplicity = {int(multiplicity)}
dispersion = json.loads({json.dumps(json.dumps(dispersion, sort_keys=True))})

protocol_id = "matrix-qm-backend-execution-v2"
protocol_version = "1.0.0"
mol = gto.M(
    atom=atoms,
    basis=basis,
    charge=charge,
    spin=multiplicity - 1,
    unit="Angstrom",
    symmetry={symmetry_literal},
)
if {"True" if requested_symmetry else "False"}:
    # PySCF's gradient projector reconstructs the molecule from ``topgroup``;
    # keep that projector on the explicitly authorized Abelian subgroup.
    mol.topgroup = {json.dumps(requested_symmetry)}
if reference == "RHF":
    mf = scf.RHF(mol)
elif reference == "ROHF":
    mf = scf.ROHF(mol)
elif reference == "UHF":
    mf = scf.UHF(mol)
elif reference == "RKS":
    mf = dft.RKS(mol)
    mf.xc = method
elif reference == "ROKS":
    mf = dft.ROKS(mol)
    mf.xc = method
elif reference == "UKS":
    mf = dft.UKS(mol)
    mf.xc = method
else:
    raise RuntimeError(f"unsupported validated PySCF reference: {{reference}}")
if {json.dumps(resolved_accelerator)} == "density_fitting":
    mf = mf.density_fit()
if dispersion is not None:
    mf = d3disp.d3_energy(
        mf,
        method=method,
        version=dispersion["version"],
        atm=dispersion["atm"],
        param=dispersion["parameters"],
    )
scf_guess_path = {json.dumps(guess_path)}
dm0 = None
if scf_guess_path:
    with np.load(scf_guess_path, allow_pickle=False) as guess:
        density_alpha = np.asarray(guess["density_alpha"], dtype=float)
        density_beta = np.asarray(guess["density_beta"], dtype=float) if "density_beta" in guess else None
        fock_alpha = np.asarray(guess["fock_alpha"], dtype=float) if "fock_alpha" in guess else None
        fock_beta = np.asarray(guess["fock_beta"], dtype=float) if "fock_beta" in guess else None
    dm0 = density_alpha if density_beta is None else np.asarray((density_alpha, density_beta))
    if fock_alpha is not None:
        overlap = mf.get_ovlp()
        fock_guess = fock_alpha if fock_beta is None else np.asarray((fock_alpha, fock_beta))
        mo_energy, mo_coeff = mf.eig(fock_guess, overlap)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
energy = float(mf.kernel(dm0=dm0))
{gradient_source}
{hessian_source}{population_source}state_path = {json.dumps(state_path)}
if state_path:
    density_state = np.asarray(mf.make_rdm1(), dtype=float)
    fock_state = np.asarray(mf.get_fock(dm=density_state), dtype=float)
    state_payload = {{
        "density_alpha": density_state if density_state.ndim == 2 else density_state[0],
        "fock_alpha": fock_state if fock_state.ndim == 2 else fock_state[0],
        "overlap": np.asarray(mf.get_ovlp(), dtype=float),
        "orbitals": np.asarray(mf.mo_coeff, dtype=float),
        "occupations": np.asarray(mf.mo_occ, dtype=float),
        "coordinates_angstrom": np.asarray(mol.atom_coords(unit="Angstrom"), dtype=float),
        "electron_count": np.asarray([float(mol.nelectron)], dtype=float),
    }}
    if density_state.ndim == 3:
        state_payload["density_beta"] = density_state[1]
    if fock_state.ndim == 3:
        state_payload["fock_beta"] = fock_state[1]
    np.savez(state_path, **state_payload)
result = {{
    "schema": "{schema}",
    "pyscf_version": pyscf.__version__,
    "atoms": [item[0] for item in atoms],
    "coordinates_angstrom": np.asarray(mol.atom_coords(unit="Angstrom"), dtype=float).tolist(),
    "charge": charge,
    "multiplicity": multiplicity,
    "method": method,
    "reference": reference,
    "basis": basis,
    "dispersion": dispersion,
    "execution_protocol_id": protocol_id,
    "execution_protocol_version": protocol_version,
    "symmetry": {json.dumps(requested_symmetry or "auto")},
    "accelerator": {json.dumps(resolved_accelerator)},
    "energy_hartree": energy,
    "scf_cycles": int(getattr(mf, "cycles", 0)),
{gradient_result}
{hessian_result}{population_result}    "converged": bool(mf.converged),
}}
print("{PYSCF_RESULT_PREFIX}" + json.dumps(result, sort_keys=True))
if not mf.converged:
    raise RuntimeError("PySCF SCF did not converge")
print("MATRIX PYSCF TERMINATED NORMALLY")
'''
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def write_pyscf_hessian_input(
    output: Path | str,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    method: str = "HF",
    reference: str | None = None,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    dispersion_contract: str | None = None,
    accelerator: str = "density_fitting",
    include_population: bool = False,
    _scf_guess_path: Path | str | None = None,
    _scf_state_path: Path | str | None = None,
) -> Path:
    """Write a self-reporting PySCF Hessian and harmonic-frequency job.

    The default script is standalone in a plain PySCF environment. APOC
    population observables may be requested explicitly when ``matrix-pyscf``
    is also installed in the execution environment.
    """

    return write_pyscf_point_input(
        output,
        atoms,
        coordinates_angstrom,
        method=method,
        reference=reference,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        dispersion_contract=dispersion_contract,
        accelerator=accelerator,
        _include_hessian=True,
        _include_population=bool(include_population),
        _scf_guess_path=_scf_guess_path,
        _scf_state_path=_scf_state_path,
    )


def population_observables_from_pyscf_output(path: Path | str):
    """Read the canonical APOC observables from structured PySCF output."""

    from matrix_qm import QMPopulationObservables

    target = Path(path)
    data = _last_structured_result(
        target.read_text(encoding="utf-8", errors="replace")
    )
    if data is None:
        raise GeometryParseError(f"PySCF output contains no structured result: {target}")
    required = (
        "hirshfeld_charges",
        "cm5_charges",
        "mayer_bond_orders",
        "electron_count",
        "population_charge",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise GeometryParseError(
            "PySCF output lacks APOC population fields: " + ", ".join(missing)
        )
    return QMPopulationObservables(
        hirshfeld_charges=np.asarray(data["hirshfeld_charges"], dtype=float),
        cm5_charges=np.asarray(data["cm5_charges"], dtype=float),
        mayer_bond_orders=np.asarray(data["mayer_bond_orders"], dtype=float),
        electron_count=float(data["electron_count"]),
        charge=float(data["population_charge"]),
    )


def summarize_pyscf_output(path: Path | str) -> PySCFOutputSummary:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    data = _last_structured_result(text)
    geometry = _geometry_from_result(data, target) if data is not None else None
    energy = None
    if data is not None and data.get("energy_hartree") is not None:
        energy = float(data["energy_hartree"])
    else:
        matches = _SCF_ENERGY_RE.findall(text)
        if matches:
            energy = float(matches[-1].replace("D", "E").replace("d", "e"))
    gradient = None
    if data is not None and data.get("gradient_hartree_per_bohr") is not None:
        gradient = np.asarray(data["gradient_hartree_per_bohr"], dtype=float)
    normal = "MATRIX PYSCF TERMINATED NORMALLY" in text
    return PySCFOutputSummary(
        path=target,
        geometry=geometry,
        charge=int(data.get("charge", 0)) if data else 0,
        multiplicity=int(data.get("multiplicity", 1)) if data else 1,
        method=str(data.get("method", "")) if data else "",
        reference=str(data.get("reference", "")) if data else "",
        basis=str(data.get("basis", "")) if data else "",
        dispersion_contract=(
            str(data["dispersion"].get("contract_id"))
            if data and isinstance(data.get("dispersion"), dict)
            else None
        ),
        final_energy_hartree=energy,
        gradient_hartree_per_bohr=gradient,
        converged=bool(data.get("converged", normal)) if data else normal,
        normal_termination=normal,
        scf_cycles=int(data.get("scf_cycles", 0)) if data else 0,
    )


def read_pyscf_output_geometry(path: Path | str) -> MolecularGeometry:
    summary = summarize_pyscf_output(path)
    if summary.geometry is None:
        raise GeometryParseError(
            f"PySCF output has no {PYSCF_RESULT_PREFIX.strip()} structured geometry: {path}"
        )
    return summary.geometry


def read_pyscf_point_result(path: Path | str) -> PySCFPointResult:
    summary = summarize_pyscf_output(path)
    if summary.final_energy_hartree is None:
        raise GeometryParseError(f"PySCF output contains no parseable energy: {path}")
    return PySCFPointResult(
        path=summary.path,
        energy_hartree=summary.final_energy_hartree,
        gradient_hartree_per_bohr=(
            None
            if summary.gradient_hartree_per_bohr is None
            else np.asarray(summary.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        ),
        normal_termination=summary.normal_termination and summary.converged,
        scf_cycles=summary.scf_cycles,
    )


def hessian_input_from_pyscf_output(
    path: Path | str,
    *,
    input_path: Path | str | None = None,
):
    """Read a MATRIX structured PySCF Hessian/frequency result."""

    from matrix_qm import HessianInput

    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    data = _last_structured_result(text)
    if data is None or data.get("schema") != "matrix.pyscf.hessian_result.v1":
        raise GeometryParseError(f"PySCF output contains no structured Hessian result: {target}")
    geometry = _geometry_from_result(data, target)
    atomic_numbers = tuple(_required_atomic_number(atom) for atom in geometry.atoms)
    size = 3 * geometry.natoms
    hessian = np.asarray(data.get("hessian_hartree_per_bohr2", []), dtype=float)
    if hessian.shape != (size, size):
        raise GeometryParseError(
            f"PySCF Cartesian Hessian has shape {hessian.shape}; expected ({size}, {size})"
        )
    if not np.all(np.isfinite(hessian)):
        raise GeometryParseError("PySCF Cartesian Hessian contains non-finite values")
    if not np.allclose(hessian, hessian.T, rtol=1.0e-8, atol=1.0e-10):
        mismatch = float(np.max(np.abs(hessian - hessian.T)))
        raise GeometryParseError(
            f"PySCF Cartesian Hessian is not symmetric (maximum mismatch {mismatch:.3e})"
        )
    coordinates = np.asarray(geometry.coordinates_angstrom, dtype=float)
    frequencies = np.asarray(data.get("harmonic_frequencies_cm", []), dtype=float).reshape(-1)
    expected_modes = expected_vibrational_mode_count(coordinates)
    linear = is_linear_geometry(coordinates)
    if frequencies.size != expected_modes:
        raise GeometryParseError(
            f"PySCF result contains {frequencies.size} vibrational modes; expected "
            f"{expected_modes} (3N-{'5' if linear else '6'}) for this "
            f"{'linear' if linear else 'nonlinear'} {geometry.natoms}-atom geometry"
        )
    version = str(data.get("pyscf_version", "")).strip()
    if not version:
        raise GeometryParseError("PySCF Hessian result is missing pyscf_version provenance")
    provenance = {
        "PROGRAM": "PySCF",
        "PROGRAM_VERSION": version,
        "METHOD": str(data.get("method", "")),
        "REFERENCE": str(data.get("reference", "")),
        "BASIS": str(data.get("basis", "")),
        "OUTPUT_SHA256": _sha256(target),
        "EXPECTED_MODE_COUNT": str(expected_modes),
        "LINEAR": str(linear).lower(),
    }
    dispersion = data.get("dispersion")
    if isinstance(dispersion, dict) and dispersion.get("contract_id"):
        provenance["DISPERSION_CONTRACT"] = str(dispersion["contract_id"])
        provenance["DISPERSION"] = json.dumps(dispersion, sort_keys=True)
    if data.get("raw_hessian_antisymmetry_max_hartree_per_bohr2") is not None:
        provenance["RAW_HESSIAN_ANTISYMMETRY_MAX_HARTREE_PER_BOHR2"] = str(
            float(data["raw_hessian_antisymmetry_max_hartree_per_bohr2"])
        )
    if input_path is not None:
        source_input = Path(input_path)
        if not source_input.is_file():
            raise GeometryParseError(f"PySCF input file not found: {source_input}")
        provenance["INPUT_SHA256"] = _sha256(source_input)
    result = HessianInput(
        atomic_numbers=np.asarray(atomic_numbers, dtype=int),
        cartesian_coordinates_bohr=coordinates / BOHR_TO_ANGSTROM,
        masses_amu=np.asarray([atomic_mass(value) for value in atomic_numbers], dtype=float),
        cartesian_hessian=(hessian + hessian.T) * 0.5,
        harmonic_frequencies_cm=frequencies,
        source="pyscf-structured-hessian",
        provenance=provenance,
    )
    result.validate()
    return result


def _last_structured_result(text: str) -> dict[str, object] | None:
    values: list[dict[str, object]] = []
    for raw in text.splitlines():
        if not raw.startswith(PYSCF_RESULT_PREFIX):
            continue
        try:
            parsed = json.loads(raw[len(PYSCF_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise GeometryParseError("invalid MATRIX PySCF JSON result") from exc
        if parsed.get("schema") not in {
            "matrix.pyscf.result.v1",
            "matrix.pyscf.hessian_result.v1",
        }:
            raise GeometryParseError("unsupported MATRIX PySCF result schema")
        values.append(parsed)
    return values[-1] if values else None


def _geometry_from_result(data: dict[str, object], path: Path) -> MolecularGeometry:
    atoms = tuple(normalize_atom_symbol(item) for item in data.get("atoms", []))
    coords = np.asarray(data.get("coordinates_angstrom", []), dtype=float)
    if coords.shape != (len(atoms), 3) or not atoms:
        raise GeometryParseError("MATRIX PySCF result has invalid geometry")
    return MolecularGeometry(
        atoms=atoms,
        coordinates_angstrom=coords,
        comment=path.name,
        source_format=PYSCF_OUTPUT_FORMAT,
        source_path=path,
        charge=int(data.get("charge", 0)),
        multiplicity=int(data.get("multiplicity", 1)),
        metadata={"method": str(data.get("method", "")), "basis": str(data.get("basis", ""))},
    )


def _required_atomic_number(symbol: str) -> int:
    value = atomic_number(symbol)
    if value is None:
        raise GeometryParseError(f"unknown atom symbol in PySCF result: {symbol}")
    return int(value)


def _resolve_pyscf_reference(
    *,
    method: str,
    multiplicity: int,
    reference: str | None,
) -> str:
    """Resolve one explicit PySCF SCF reference without changing spin semantics."""

    if int(multiplicity) < 1:
        raise GeometryParseError("PySCF multiplicity must be a positive integer")
    normalized_method = str(method).strip().upper() or "HF"
    requested = "" if reference is None else str(reference).strip().upper()
    hf_method = normalized_method in {"HF", "RHF", "ROHF", "UHF"}
    if not requested:
        if normalized_method in {"RHF", "ROHF", "UHF"}:
            requested = normalized_method
        elif hf_method:
            requested = "RHF" if int(multiplicity) == 1 else "UHF"
        else:
            requested = "RKS" if int(multiplicity) == 1 else "UKS"
    allowed = {"RHF", "ROHF", "UHF"} if hf_method else {"RKS", "ROKS", "UKS"}
    if requested not in allowed:
        family = "HF" if hf_method else "DFT"
        raise GeometryParseError(
            f"PySCF reference {requested or '<empty>'} is incompatible with {family} method {method}"
        )
    restricted_closed = requested in {"RHF", "RKS"}
    restricted_open = requested in {"ROHF", "ROKS"}
    if restricted_closed and int(multiplicity) != 1:
        raise GeometryParseError(f"PySCF {requested} requires multiplicity 1")
    if restricted_open and int(multiplicity) < 2:
        raise GeometryParseError(f"PySCF {requested} requires multiplicity at least 2")
    return requested


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
