"""APOC-owned SCF adapter registry for electronic-structure backends.

The registry deliberately separates capability declaration from electronic
structure implementation.  A backend without a verified state representation
is usable by a client driver, but receives the standard SCF guess and an audit event.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .state_reuse import (
    SCFBackendRequest,
    SCFGuess,
    SCFState,
    SCFStateCapabilities,
    SCFStateContractError,
    SCFStateManager,
    extrapolate_scf_state,
    project_scf_state,
    reuse_same_basis_state,
    extrapolate_same_basis_state,
)


SCF_BACKENDS = ("pyscf", "pyfock", "et", "orca", "gaussian", "psi4", "mrcc", "molpro")


@dataclass(frozen=True)
class MRCCDensities:
    """Reduced MRCC densities as stored in the ASCII ``CCDENSITIES`` file."""

    one_body: np.ndarray
    two_body: dict[tuple[int, int, int, int], float]


def read_mrcc_scf_densities(
    artifact: str | Path,
    *,
    dimension: int | None = None,
) -> dict[int, np.ndarray]:
    """Read MRCC's binary ``SCFDENSITIES`` Fortran-sequential records.

    Each record contains ``(n_entries, spin_block)`` followed by ``n_entries``
    sparse lower-triangular records ``(value: float64, i: int32, j: int32)``.
    MRCC uses one-based AO indices.  The return value maps the MRCC spin-block
    identifier to a full symmetric AO matrix, without decimal truncation.
    """
    path = Path(artifact)
    if path.is_dir():
        path = path / "SCFDENSITIES"
    if not path.is_file():
        raise SCFStateContractError(f"MRCC density artifact not found: {path}")
    raw = path.read_bytes()
    records: list[tuple[int, list[tuple[float, int, int]]]] = []
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise SCFStateContractError("truncated MRCC SCFDENSITIES record marker")
        record_size = struct.unpack_from("<i", raw, offset)[0]
        if record_size < 8 or offset + record_size + 8 > len(raw):
            raise SCFStateContractError("invalid MRCC SCFDENSITIES record length")
        payload_start = offset + 4
        payload_end = payload_start + record_size
        trailer = struct.unpack_from("<i", raw, payload_end)[0]
        if trailer != record_size:
            raise SCFStateContractError("inconsistent MRCC SCFDENSITIES record markers")
        if record_size < 8:
            raise SCFStateContractError("empty MRCC SCFDENSITIES record")
        n_entries, spin_block = struct.unpack_from("<ii", raw, payload_start)
        expected = 8 + 16 * n_entries
        if n_entries < 0 or expected != record_size:
            raise SCFStateContractError("invalid MRCC sparse density record layout")
        entries: list[tuple[float, int, int]] = []
        cursor = payload_start + 8
        for _ in range(n_entries):
            value = struct.unpack_from("<d", raw, cursor)[0]
            i, j = struct.unpack_from("<ii", raw, cursor + 8)
            if i < 1 or j < 1 or not np.isfinite(value):
                raise SCFStateContractError("invalid MRCC density entry")
            entries.append((value, i, j))
            cursor += 16
        records.append((spin_block, entries))
        offset = payload_end + 4
    if not records:
        raise SCFStateContractError("MRCC SCFDENSITIES contains no records")
    inferred = max(max(max(i, j) for _, i, j in entries) for _, entries in records)
    n_ao = inferred if dimension is None else int(dimension)
    if n_ao < inferred or n_ao < 1:
        raise SCFStateContractError("MRCC density dimension is smaller than an AO index")
    matrices: dict[int, np.ndarray] = {}
    for spin_block, entries in records:
        matrix = matrices.setdefault(spin_block, np.zeros((n_ao, n_ao), dtype=float))
        for value, i, j in entries:
            matrix[i - 1, j - 1] = value
            matrix[j - 1, i - 1] = value
    return matrices


def scf_state_from_mrcc_scf_densities(
    artifact: str | Path,
    *,
    method: str,
    basis: str,
    coordinates: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    electronic_state: int = 0,
    symmetry: str = "",
    dimension: int | None = None,
) -> SCFState:
    """Create a validated MRCC state from ``SCFDENSITIES``."""
    path = Path(artifact)
    densities = read_mrcc_scf_densities(path, dimension=dimension)
    alpha = densities[min(densities)]
    beta = densities.get(1) if 1 in densities and min(densities) != 1 else None
    return SCFState(
        backend="mrcc",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        electronic_state=electronic_state,
        coordinates=coordinates,
        density_alpha=alpha,
        density_beta=beta,
        symmetry=symmetry,
        provenance={
            "artifact": str((path / "SCFDENSITIES" if path.is_dir() else path).resolve()),
            "format": "mrcc-sparse-fortran",
            "spin_blocks": tuple(sorted(densities)),
        },
    )


def _unsupported(operation: str, backend: str) -> None:
    raise SCFStateContractError(
        f"{backend} does not expose a verified SCF {operation} representation"
    )


class BackendSCFAdapter:
    """Callback-based adapter shared by in-process and external backends."""

    def __init__(
        self,
        backend: str,
        capabilities: SCFStateCapabilities,
        *,
        export: Callable[[Any], SCFState] | None = None,
        project: Callable[[SCFState, Any, str], SCFGuess] | None = None,
        extrapolate: Callable[[Sequence[SCFState], Any, str], SCFGuess] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    ) -> None:
        self.backend = str(backend).strip().casefold()
        if self.backend not in SCF_BACKENDS:
            raise ValueError(f"unknown APOC SCF backend: {backend}")
        self._capabilities = capabilities
        self._export = export
        self._project = project
        self._extrapolate = extrapolate
        self._evaluate = evaluate

    def capabilities(self) -> SCFStateCapabilities:
        return self._capabilities

    def export_state(self, result: Any) -> SCFState:
        if self._export is None:
            _unsupported("state export", self.backend)
        state = self._export(result)
        if not isinstance(state, SCFState):
            raise SCFStateContractError(f"{self.backend} adapter returned an invalid SCF state")
        return state

    def project_state(self, state: SCFState, target_geometry: Any, target_basis: str) -> SCFGuess:
        if self._project is None:
            _unsupported("projection", self.backend)
        guess = self._project(state, target_geometry, target_basis)
        if not isinstance(guess, SCFGuess):
            raise SCFStateContractError(f"{self.backend} adapter returned an invalid SCF guess")
        return guess

    def extrapolate_state(
        self, history: Sequence[SCFState], target_geometry: Any, target_basis: str
    ) -> SCFGuess:
        if self._extrapolate is None:
            _unsupported("extrapolation", self.backend)
        guess = self._extrapolate(history, target_geometry, target_basis)
        if not isinstance(guess, SCFGuess):
            raise SCFStateContractError(f"{self.backend} adapter returned an invalid SCF guess")
        return guess

    def evaluate(self, request: SCFBackendRequest, guess: SCFGuess | None = None) -> Any:
        if self._evaluate is None:
            raise SCFStateContractError(f"{self.backend} adapter has no evaluator")
        return self._evaluate(request, guess)

    def evaluate_with_state(
        self,
        request: SCFBackendRequest,
        *,
        manager: SCFStateManager | None = None,
        branch: str = "root",
    ) -> tuple[Any, SCFGuess, SCFState | None]:
        """Run the common APOC lifecycle, including audited fallback."""
        if manager is None:
            manager = SCFStateManager(self)
        guess = manager.prepare_guess(request, branch=branch)
        result = self.evaluate(request, guess=guess)
        if self._export is None:
            manager.events.append(
                {
                    "event": "state_not_recorded",
                    "backend": self.backend,
                    "branch": str(branch),
                    "reason": "backend_does_not_export_scf_state",
                }
            )
            return result, guess, None
        return result, guess, manager.record_converged(result, branch=branch)


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def scf_state_from_result(
    result: Any,
    *,
    backend: str,
    method: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    electronic_state: int = 0,
) -> SCFState:
    """Build a validated state from an in-process result or external restart record."""
    if isinstance(result, SCFState):
        return result
    coordinates = _result_field(result, "coordinates")
    density_alpha = _result_field(result, "density_alpha")
    if coordinates is None or density_alpha is None:
        raise SCFStateContractError(
            f"{backend} result does not contain the matrices required for SCF state export"
        )
    provenance = _result_field(result, "provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    return SCFState(
        backend=backend,
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        electronic_state=electronic_state,
        coordinates=coordinates,
        density_alpha=density_alpha,
        density_beta=_result_field(result, "density_beta"),
        fock_alpha=_result_field(result, "fock_alpha"),
        fock_beta=_result_field(result, "fock_beta"),
        orbitals=_result_field(result, "orbitals"),
        occupations=_result_field(result, "occupations"),
        overlap=_result_field(result, "overlap"),
        electron_count=_result_field(result, "electron_count"),
        spin=str(_result_field(result, "spin", "")).strip(),
        symmetry=str(_result_field(result, "symmetry", "")).strip(),
        orbital_order=tuple(_result_field(result, "orbital_order", ())),
        provenance={
            "backend_result_type": type(result).__name__,
            **dict(provenance),
        },
    )


def read_mrcc_ccdensities(artifact: str | Path) -> MRCCDensities:
    """Read MRCC's ASCII correlated one- and two-particle densities.

    Records are ``value i j k l``.  Records with ``k=l=0`` are one-particle
    elements; all other records are two-particle elements.  Indices remain
    one-based in the returned two-particle map.
    """
    path = Path(artifact)
    if path.is_dir():
        path = path / "CCDENSITIES"
    if not path.is_file():
        raise SCFStateContractError(f"MRCC correlated density not found: {path}")
    one_body_entries: list[tuple[float, int, int]] = []
    two_body: dict[tuple[int, int, int, int], float] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise SCFStateContractError(f"invalid MRCC CCDENSITIES record at line {line_number}")
        try:
            value = float(fields[0].replace("D", "E").replace("d", "e"))
            indices = tuple(int(field) for field in fields[1:])
        except ValueError as exc:
            raise SCFStateContractError(
                f"invalid MRCC CCDENSITIES record at line {line_number}"
            ) from exc
        if not np.isfinite(value) or any(index < 0 for index in indices):
            raise SCFStateContractError("invalid value or index in MRCC CCDENSITIES")
        i, j, k, l = indices
        if k == 0 and l == 0:
            if i < 1 or j < 1:
                raise SCFStateContractError("invalid one-particle MRCC orbital index")
            one_body_entries.append((value, i, j))
        else:
            if min(i, j, k, l) < 1:
                raise SCFStateContractError("invalid two-particle MRCC orbital index")
            two_body[(i, j, k, l)] = value
    if not one_body_entries:
        raise SCFStateContractError("MRCC CCDENSITIES contains no one-particle density")
    n_orb = max(max(i, j) for _, i, j in one_body_entries)
    one_body = np.zeros((n_orb, n_orb), dtype=float)
    for value, i, j in one_body_entries:
        one_body[i - 1, j - 1] = value
    return MRCCDensities(one_body=one_body, two_body=two_body)


def scf_state_from_mrcc_ccdensities(
    artifact: str | Path,
    *,
    method: str,
    basis: str,
    coordinates: np.ndarray,
    mo_coefficients: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    electronic_state: int = 0,
    symmetry: str = "",
) -> SCFState:
    """Create an AO state from MRCC's correlated one-body density.

    ``CCDENSITIES`` is in the MRCC orbital representation; MO coefficients
    are mandatory so an orbital-space density cannot be mistaken for AO data.
    """
    path = Path(artifact)
    densities = read_mrcc_ccdensities(path)
    coefficients = np.asarray(mo_coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] != densities.one_body.shape[0]:
        raise SCFStateContractError("MRCC MO coefficients do not match CCDENSITIES")
    density_ao = coefficients @ densities.one_body @ coefficients.T
    resolved = (path / "CCDENSITIES" if path.is_dir() else path).resolve()
    return SCFState(
        backend="mrcc",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        electronic_state=electronic_state,
        coordinates=coordinates,
        density_alpha=density_ao,
        symmetry=symmetry,
        electron_count=float(np.trace(densities.one_body)),
        provenance={
            "artifact": str(resolved),
            "format": "mrcc-ccdensities",
            "two_body_elements": len(densities.two_body),
        },
    )


def pyscf_cross_overlap_transform(
    state: SCFState,
    target_geometry: np.ndarray,
    target_basis: str,
    *,
    atoms: Sequence[str],
    target_charge: int | None = None,
    target_multiplicity: int | None = None,
) -> np.ndarray:
    """Return the AO coefficient transformation built with PySCF cross overlap."""
    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SCFStateContractError("PySCF is required for AO cross-overlap projection") from exc
    source_atoms = tuple(state.provenance.get("atoms", ()))
    target_atoms = tuple(str(atom) for atom in atoms)
    if source_atoms != target_atoms:
        raise SCFStateContractError("PySCF cross-overlap requires identical atom ordering")
    source = gto.Mole(
        atom=[
            (atom, tuple(xyz)) for atom, xyz in zip(source_atoms, state.coordinates, strict=True)
        ],
        basis=state.basis,
        charge=state.charge,
        spin=max(0, state.multiplicity - 1),
        unit="Angstrom",
        verbose=0,
    ).build()
    target = gto.Mole(
        atom=[
            (atom, tuple(xyz))
            for atom, xyz in zip(
                target_atoms, np.asarray(target_geometry, dtype=float), strict=True
            )
        ],
        basis=target_basis,
        charge=state.charge if target_charge is None else int(target_charge),
        spin=max(0, state.multiplicity - 1)
        if target_multiplicity is None
        else max(0, int(target_multiplicity) - 1),
        unit="Angstrom",
        verbose=0,
    ).build()
    cross = gto.intor_cross("int1e_ovlp", target, source)
    overlap = target.intor("int1e_ovlp")
    if state.density_alpha is not None and cross.shape[1] != state.density_alpha.shape[0]:
        raise SCFStateContractError("PySCF source AO dimension differs from density dimension")
    if not np.all(np.isfinite(cross)):
        raise SCFStateContractError("PySCF cross-overlap contains non-finite values")
    try:
        return np.linalg.solve(overlap, cross)
    except np.linalg.LinAlgError as exc:
        raise SCFStateContractError("PySCF target AO overlap is singular") from exc


def scf_state_from_gaussian_fchk(
    fchk: str | Path,
    *,
    method: str,
    basis: str,
    symmetry: str = "",
) -> SCFState:
    """Read a Gaussian FCHK SCF density without reducing its precision."""
    from matrix_gaussian.fchk import _first_array, _read_fchk_blocks, _optional_int_scalar

    blocks = _read_fchk_blocks(Path(fchk))
    density_name = next(
        (name for name in ("Total SCF Density", "Alpha Density Matrix") if name in blocks),
        None,
    )
    if density_name is None:
        raise SCFStateContractError("Gaussian FCHK contains no SCF density matrix")
    packed = np.asarray(_first_array(blocks, density_name), dtype=float)
    dimension = int((np.sqrt(8.0 * packed.size + 1.0) - 1.0) / 2.0)
    if dimension * (dimension + 1) // 2 != packed.size:
        raise SCFStateContractError("Gaussian FCHK density matrix is not packed triangular data")
    density = np.zeros((dimension, dimension), dtype=float)
    lower = np.tril_indices(dimension)
    density[lower] = packed
    density[(lower[1], lower[0])] = packed
    coords = _first_array(blocks, "Current cartesian coordinates").reshape((-1, 3))
    charge = int(_optional_int_scalar(blocks, "Charge") or 0)
    multiplicity = int(_optional_int_scalar(blocks, "Multiplicity") or 1)
    return SCFState(
        backend="gaussian",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        electronic_state=0,
        coordinates=coords,
        density_alpha=density,
        symmetry=symmetry,
        electron_count=float(_first_array(blocks, "Number of electrons")[0])
        if "Number of electrons" in blocks
        else None,
        provenance={"artifact": str(Path(fchk).resolve()), "format": "fchk"},
    )


def scf_state_from_orca_molden(
    molden: str | Path,
    *,
    method: str,
    basis: str,
    coordinates: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    symmetry: str = "",
) -> SCFState:
    """Build an ORCA state from Molden orbitals and occupations."""
    try:
        from matrix_pyscf import load_molden_wavefunction
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SCFStateContractError("ORCA Molden state export requires matrix-pyscf") from exc
    _mol, _energies, coefficients, occupations, _labels, _spins = load_molden_wavefunction(molden)
    if isinstance(coefficients, (tuple, list)):
        if not isinstance(occupations, (tuple, list)) or len(coefficients) != 2:
            raise SCFStateContractError("ORCA Molden spin blocks are incomplete")
        densities = tuple(
            np.asarray(coeff, dtype=float)
            @ np.diag(np.asarray(occ, dtype=float))
            @ np.asarray(coeff, dtype=float).T
            for coeff, occ in zip(coefficients, occupations, strict=True)
        )
        density_alpha, density_beta = densities
    else:
        coefficients = np.asarray(coefficients, dtype=float)
        occupations = np.asarray(occupations, dtype=float)
        density_alpha = coefficients @ np.diag(occupations) @ coefficients.T
        density_beta = None
    return SCFState(
        backend="orca",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        electronic_state=0,
        coordinates=coordinates,
        density_alpha=density_alpha,
        density_beta=density_beta,
        symmetry=symmetry,
        electron_count=float(
            np.trace(density_alpha) + (0.0 if density_beta is None else np.trace(density_beta))
        ),
        provenance={"artifact": str(Path(molden).resolve()), "format": "molden"},
    )


class PySCFSCFAdapter(BackendSCFAdapter):
    """PySCF adapter for direct in-process density/Fock/orbital results."""

    def __init__(
        self,
        *,
        method: str = "HF",
        basis: str = "sto-3g",
        charge: int = 0,
        multiplicity: int = 1,
        electronic_state: int = 0,
        atoms: Sequence[str] | None = None,
        basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    ):
        super().__init__(
            "pyscf",
            SCFStateCapabilities(in_process=True),
            export=lambda result: _pyscf_export_state(
                result,
                method=method,
                basis=basis,
                charge=charge,
                multiplicity=multiplicity,
                electronic_state=electronic_state,
                atoms=atoms,
            ),
            evaluate=evaluate,
        )
        if atoms is not None and basis_transform is None:
            basis_transform = lambda state, geometry, target_basis: pyscf_cross_overlap_transform(
                state, geometry, target_basis, atoms=atoms
            )
        if basis_transform is not None:
            self._project = lambda state, geometry, target_basis: (
                reuse_same_basis_state(state, geometry, target_basis)
                if target_basis == state.basis
                else project_scf_state(state, geometry, target_basis, basis_transform)
            )
            self._extrapolate = lambda history, geometry, target_basis: extrapolate_scf_state(
                history, geometry, target_basis, self._project
            )
            self._capabilities = SCFStateCapabilities(
                density_projection=True,
                density_extrapolation=True,
                fock_projection=True,
                fock_extrapolation=True,
                in_process=True,
            )


class PyFockSCFAdapter(BackendSCFAdapter):
    """PyFock adapter using the general APOC SCF-state lifecycle.

    The native PyFock projection callback is used only for a true AO-basis
    change.  Within one fixed basis, densities and operators are copied or
    extrapolated directly, so no artificial cross-overlap is evaluated.
    """

    def __init__(
        self,
        *,
        method: str,
        basis: str,
        charge: int = 0,
        multiplicity: int = 1,
        electronic_state: int = 0,
        atoms: Sequence[str] | None = None,
        basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
        density_projector: Callable[[SCFState, Any, str], np.ndarray] | None = None,
        operator_projector: Callable[[SCFState, Any, str], np.ndarray] | None = None,
        export: Callable[[Any], SCFState] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    ) -> None:
        super().__init__(
            "pyfock",
            SCFStateCapabilities(
                density_projection=True,
                density_extrapolation=True,
                fock_projection=True,
                fock_extrapolation=True,
                in_process=True,
            ),
            evaluate=evaluate,
            export=export
            or (
                lambda result: _pyfock_export_state(
                    result,
                    method=method,
                    basis=basis,
                    charge=charge,
                    multiplicity=multiplicity,
                    electronic_state=electronic_state,
                    atoms=atoms,
                )
            ),
        )
        self._project = reuse_same_basis_state
        self._extrapolate = extrapolate_same_basis_state
        if density_projector is not None:

            def native_project(state, geometry, target_basis):
                density = density_projector(state, geometry, target_basis)
                fock = (
                    None
                    if operator_projector is None
                    else operator_projector(state, geometry, target_basis)
                )
                return SCFGuess(
                    density_alpha=np.asarray(density, dtype=float),
                    fock_alpha=None if fock is None else np.asarray(fock, dtype=float),
                    status="projected",
                    reason="backend_native_basis_projection",
                )

            self._project = lambda state, geometry, target_basis: (
                reuse_same_basis_state(state, geometry, target_basis)
                if target_basis == state.basis
                else native_project(state, geometry, target_basis)
            )
            self._extrapolate = lambda history, geometry, target_basis: extrapolate_scf_state(
                history, geometry, target_basis, self._project
            )
        if basis_transform is not None:
            self._project = lambda state, geometry, target_basis: (
                reuse_same_basis_state(state, geometry, target_basis)
                if target_basis == state.basis
                else project_scf_state(state, geometry, target_basis, basis_transform)
            )
            self._extrapolate = lambda history, geometry, target_basis: extrapolate_scf_state(
                history, geometry, target_basis, self._project
            )


def _pyfock_export_state(
    result: Any,
    *,
    method: str,
    basis: str,
    charge: int,
    multiplicity: int,
    electronic_state: int,
    atoms: Sequence[str] | None,
) -> SCFState:
    state = scf_state_from_result(
        result,
        backend="pyfock",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        electronic_state=electronic_state,
    )
    if atoms is None:
        return state
    return replace(state, provenance={**state.provenance, "atoms": tuple(atoms)})


def _pyscf_export_state(
    result: Any,
    *,
    method: str,
    basis: str,
    charge: int,
    multiplicity: int,
    electronic_state: int,
    atoms: Sequence[str] | None,
) -> SCFState:
    state = scf_state_from_result(
        result,
        backend="pyscf",
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        electronic_state=electronic_state,
    )
    if atoms is None:
        return state
    return replace(state, provenance={**state.provenance, "atoms": tuple(atoms)})


class ExternalCheckpointSCFAdapter(BackendSCFAdapter):
    """Adapter for external ORCA/Gaussian restart or checkpoint records."""

    def __init__(
        self,
        backend: str,
        *,
        method: str,
        basis: str,
        basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
        export: Callable[[Any], SCFState] | None = None,
    ):
        if backend not in {"orca", "gaussian"}:
            raise ValueError("external checkpoint adapter is restricted to ORCA/Gaussian")
        super().__init__(
            backend,
            SCFStateCapabilities(orbital_restart=True),
            export=export
            or (
                lambda result: scf_state_from_result(
                    result,
                    backend=backend,
                    method=method,
                    basis=basis,
                )
            ),
            evaluate=evaluate,
        )
        # The callback is installed after construction so the capability flags
        # remain conservative when no verified AO cross-overlap is available.
        if basis_transform is not None:
            self._project = lambda state, geometry, target_basis: project_scf_state(
                state, geometry, target_basis, basis_transform
            )


class GaussianFCHKSCFAdapter(ExternalCheckpointSCFAdapter):
    """Gaussian adapter whose state source is a formatted checkpoint file."""

    def __init__(
        self,
        *,
        method: str,
        basis: str,
        symmetry: str = "",
        basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    ):
        super().__init__(
            "gaussian",
            method=method,
            basis=basis,
            basis_transform=basis_transform,
            evaluate=evaluate,
            export=lambda result: scf_state_from_gaussian_fchk(
                result, method=method, basis=basis, symmetry=symmetry
            ),
        )


class OrcaMoldenSCFAdapter(ExternalCheckpointSCFAdapter):
    """ORCA adapter whose state source is an ``orca_2mkl`` Molden file."""

    def __init__(
        self,
        *,
        method: str,
        basis: str,
        coordinates: np.ndarray,
        charge: int = 0,
        multiplicity: int = 1,
        symmetry: str = "",
        basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    ):
        super().__init__(
            "orca",
            method=method,
            basis=basis,
            basis_transform=basis_transform,
            evaluate=evaluate,
            export=lambda result: scf_state_from_orca_molden(
                result,
                method=method,
                basis=basis,
                coordinates=coordinates,
                charge=charge,
                multiplicity=multiplicity,
                symmetry=symmetry,
            ),
        )


class MolproMoldenSCFAdapter(BackendSCFAdapter):
    """Molpro adapter using an electronic Molden export and occupations."""

    def __init__(
        self,
        *,
        method: str,
        basis: str,
        coordinates: np.ndarray,
        charge: int = 0,
        multiplicity: int = 1,
        symmetry: str = "",
        basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    ):
        super().__init__(
            "molpro",
            SCFStateCapabilities(
                density_projection=True,
                density_extrapolation=True,
                fock_projection=True,
                fock_extrapolation=True,
            ),
            evaluate=evaluate,
            export=lambda result: replace(
                scf_state_from_orca_molden(
                    result,
                    method=method,
                    basis=basis,
                    coordinates=coordinates,
                    charge=charge,
                    multiplicity=multiplicity,
                    symmetry=symmetry,
                ),
                backend="molpro",
            ),
        )
        self._project = reuse_same_basis_state
        self._extrapolate = extrapolate_same_basis_state
        if basis_transform is not None:
            self._project = lambda state, geometry, target_basis: project_scf_state(
                state, geometry, target_basis, basis_transform
            )
            self._extrapolate = lambda history, geometry, target_basis: extrapolate_scf_state(
                history, geometry, target_basis, self._project
            )
            self._capabilities = SCFStateCapabilities(
                density_projection=True,
                density_extrapolation=True,
                fock_projection=True,
                fock_extrapolation=True,
            )


class MRCCDensSCFAdapter(BackendSCFAdapter):
    """MRCC adapter with an explicit loader for the backend density artifact."""

    def __init__(
        self,
        *,
        method: str,
        basis: str,
        state_loader: Callable[[Any], SCFState],
        basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
        evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    ):
        super().__init__(
            "mrcc",
            SCFStateCapabilities(
                density_projection=True,
                density_extrapolation=True,
                fock_projection=True,
                fock_extrapolation=True,
            ),
            export=state_loader,
            evaluate=evaluate,
        )
        self._project = reuse_same_basis_state
        self._extrapolate = extrapolate_same_basis_state
        if basis_transform is not None:
            self._project = lambda state, geometry, target_basis: project_scf_state(
                state, geometry, target_basis, basis_transform
            )
            self._extrapolate = lambda history, geometry, target_basis: extrapolate_scf_state(
                history, geometry, target_basis, self._project
            )
            self._capabilities = SCFStateCapabilities(
                density_projection=True,
                density_extrapolation=True,
                fock_projection=True,
                fock_extrapolation=True,
            )


def adapter_for_backend(
    backend: str,
    *,
    method: str = "HF",
    basis: str = "sto-3g",
    atoms: Sequence[str] | None = None,
    export: Callable[[Any], SCFState] | None = None,
    project: Callable[[SCFState, Any, str], SCFGuess] | None = None,
    extrapolate: Callable[[Sequence[SCFState], Any, str], SCFGuess] | None = None,
    evaluate: Callable[[SCFBackendRequest, SCFGuess | None], Any] | None = None,
    basis_transform: Callable[[SCFState, Any, str], Any] | None = None,
) -> BackendSCFAdapter:
    """Build the declared adapter for any non-PyFock QM backend.

    State reuse is enabled only when implementation callbacks are supplied.
    ORCA/Gaussian/eT restart artifacts are therefore opt-in and cannot be
    mistaken for validated density/Fock projections.
    """
    name = str(backend).strip().casefold()
    if name not in SCF_BACKENDS:
        raise ValueError(f"unknown APOC SCF backend: {backend}")
    if name == "pyfock":
        if export is None and basis_transform is None:
            # A factory call without a PyFock result exporter is deliberately
            # fail-closed; the concrete adapter is used once the native
            # evaluator is wired to return its density/Fock matrices.
            return BackendSCFAdapter(
                "pyfock", SCFStateCapabilities(in_process=True), evaluate=evaluate
            )
        return PyFockSCFAdapter(
            method=method,
            basis=basis,
            atoms=atoms,
            basis_transform=basis_transform,
            export=export,
            evaluate=evaluate,
        )
    external_restart = name in {"orca", "gaussian", "et", "mrcc", "molpro"}
    if project is None and basis_transform is not None:
        project = lambda state, geometry, basis: project_scf_state(
            state, geometry, basis, basis_transform
        )
    if extrapolate is None and project is not None:
        extrapolate = lambda history, geometry, basis: extrapolate_scf_state(
            history, geometry, basis, project
        )
    has_projection = project is not None
    has_extrapolation = extrapolate is not None
    capabilities = SCFStateCapabilities(
        density_projection=has_projection,
        density_extrapolation=has_extrapolation,
        fock_projection=has_projection,
        fock_extrapolation=has_extrapolation,
        orbital_restart=external_restart,
        in_process=name in {"pyscf", "psi4"},
    )
    return BackendSCFAdapter(
        name,
        capabilities,
        export=export,
        project=project,
        extrapolate=extrapolate,
        evaluate=evaluate,
    )


__all__ = [
    "SCF_BACKENDS",
    "MRCCDensities",
    "BackendSCFAdapter",
    "PySCFSCFAdapter",
    "PyFockSCFAdapter",
    "ExternalCheckpointSCFAdapter",
    "adapter_for_backend",
    "GaussianFCHKSCFAdapter",
    "OrcaMoldenSCFAdapter",
    "MolproMoldenSCFAdapter",
    "MRCCDensSCFAdapter",
    "scf_state_from_result",
    "scf_state_from_gaussian_fchk",
    "scf_state_from_orca_molden",
    "read_mrcc_scf_densities",
    "scf_state_from_mrcc_scf_densities",
    "read_mrcc_ccdensities",
    "scf_state_from_mrcc_ccdensities",
    "pyscf_cross_overlap_transform",
]
