"""Canonical Cartesian-Hessian contract and electronic-structure dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HessianInput:
    """Backend-independent Cartesian harmonic input used throughout MATRIX.

    Coordinates are in bohr, masses in unified atomic-mass units and force
    constants in hartree/bohr**2.  Backend parsers may be strict about their
    native files; this class provides the final common numerical contract.
    """

    atomic_numbers: np.ndarray
    cartesian_coordinates_bohr: np.ndarray
    masses_amu: np.ndarray
    cartesian_hessian: np.ndarray
    harmonic_frequencies_cm: np.ndarray
    source: str = "matrix-qm"
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        atomic_numbers = np.asarray(self.atomic_numbers, dtype=int).reshape(-1)
        coordinates = np.asarray(self.cartesian_coordinates_bohr, dtype=float)
        masses = np.asarray(self.masses_amu, dtype=float).reshape(-1)
        hessian = np.asarray(self.cartesian_hessian, dtype=float)
        frequencies = np.asarray(self.harmonic_frequencies_cm, dtype=float).reshape(-1)
        provenance = {
            str(key).strip().upper(): str(value).strip()
            for key, value in dict(self.provenance or {}).items()
            if str(key).strip() and str(value).strip()
        }
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "cartesian_coordinates_bohr", coordinates)
        object.__setattr__(self, "masses_amu", masses)
        object.__setattr__(self, "cartesian_hessian", 0.5 * (hessian + hessian.T))
        object.__setattr__(self, "harmonic_frequencies_cm", frequencies)
        object.__setattr__(self, "source", str(self.source).strip() or "matrix-qm")
        object.__setattr__(self, "provenance", provenance)
        self.validate(raw_hessian=hessian)

    def validate(self, *, raw_hessian: np.ndarray | None = None) -> None:
        natoms = len(self.atomic_numbers)
        if natoms < 1 or np.any(self.atomic_numbers <= 0):
            raise ValueError("atomic numbers must be positive")
        if self.cartesian_coordinates_bohr.shape != (natoms, 3):
            raise ValueError("Cartesian coordinates must have shape (natoms, 3)")
        if self.masses_amu.shape != (natoms,) or np.any(self.masses_amu <= 0.0):
            raise ValueError("masses must be a positive vector with shape (natoms,)")
        expected = (3 * natoms, 3 * natoms)
        if self.cartesian_hessian.shape != expected:
            raise ValueError(f"Cartesian Hessian must have shape {expected}")
        arrays = (
            self.cartesian_coordinates_bohr,
            self.masses_amu,
            self.cartesian_hessian,
            self.harmonic_frequencies_cm,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("Hessian input contains non-finite numerical values")
        candidate = self.cartesian_hessian if raw_hessian is None else raw_hessian
        if not np.allclose(candidate, candidate.T, rtol=1.0e-8, atol=1.0e-10):
            mismatch = float(np.max(np.abs(candidate - candidate.T)))
            raise ValueError(
                "Cartesian Hessian must be symmetric "
                f"(maximum mismatch {mismatch:.3e} hartree/bohr^2)"
            )
        if any(not str(key).strip() for key in self.provenance):
            raise ValueError("Hessian provenance keys must be non-empty")


def canonicalize_hessian_input(data) -> HessianInput:
    """Return a defensive canonical copy of any HessianInput-compatible object."""

    return HessianInput(
        atomic_numbers=data.atomic_numbers,
        cartesian_coordinates_bohr=data.cartesian_coordinates_bohr,
        masses_amu=data.masses_amu,
        cartesian_hessian=data.cartesian_hessian,
        harmonic_frequencies_cm=data.harmonic_frequencies_cm,
        source=data.source,
        provenance=dict(getattr(data, "provenance", {}) or {}),
    )


def hessian_input_from_engine(
    engine: str,
    path: Path | str,
    *,
    grd: Path | str | None = None,
    output: Path | str | None = None,
    geometry: Path | str | None = None,
    spectrum: Path | str | None = None,
    input_path: Path | str | None = None,
) -> HessianInput:
    """Read and normalize a Hessian through the best native MATRIX adapter."""

    name = _normalized_engine(engine)
    target = Path(path)
    if name == "xyzin":
        from .sections import hessian_input_from_xyzin

        data = hessian_input_from_xyzin(target)
    elif name == "gaussian":
        if target.suffix.lower() in {".fchk", ".fch"}:
            from matrix_gaussian.fchk import hessian_input_from_gaussian_fchk

            data = hessian_input_from_gaussian_fchk(target)
        else:
            from matrix_gaussian.parsers import hessian_input_from_gaussian_log

            data = hessian_input_from_gaussian_log(target)
    elif name == "orca":
        from matrix_orca import hessian_input_from_orca_hessian_file, hessian_input_from_orca_output

        data = (
            hessian_input_from_orca_hessian_file(target)
            if target.suffix.lower() == ".hess"
            else hessian_input_from_orca_output(target)
        )
    elif name == "molpro":
        from matrix_molpro import hessian_input_from_molpro_output

        data = hessian_input_from_molpro_output(target)
    elif name == "mrcc":
        from matrix_mrcc import hessian_input_from_mrcc_output

        data = hessian_input_from_mrcc_output(target)
    elif name == "cfour":
        from matrix_cfour import hessian_input_from_cfour_files

        fcmfinal = target / "FCMFINAL" if target.is_dir() else target
        data = hessian_input_from_cfour_files(
            fcmfinal,
            grd=grd or fcmfinal.with_name("GRD"),
            output=output or fcmfinal.with_name("cfour.out"),
        )
    elif name == "xtb":
        from matrix_xtb import hessian_input_from_xtb_files

        data = hessian_input_from_xtb_files(
            target,
            geometry=geometry,
            spectrum=spectrum,
            output=output,
        )
    elif name == "pyscf":
        from matrix_pyscf import hessian_input_from_pyscf_output

        data = hessian_input_from_pyscf_output(target, input_path=input_path)
    else:  # pragma: no cover - guarded by _normalized_engine
        raise ValueError(f"unsupported Hessian engine: {engine}")
    return canonicalize_hessian_input(data)


def supported_hessian_engines() -> tuple[str, ...]:
    return ("xyzin", "gaussian", "orca", "molpro", "mrcc", "cfour", "xtb", "pyscf")


def _normalized_engine(engine: str) -> str:
    name = str(engine).strip().lower().replace("_", "-")
    aliases = {
        "oracle": "xyzin",
        # GDV and Gaussian 16 write the same Gaussian-family log/FCHK Hessian
        # contract.  The public CLI exposes both backend names, so Hessian
        # dispatch must normalize them before selecting the shared parser.
        "g16": "gaussian",
        "gdv": "gaussian",
        "gaussian-fchk": "gaussian",
        "gaussian-log": "gaussian",
        "orca-output": "orca",
        "orca-hess": "orca",
        "molpro-output": "molpro",
        "mrcc-output": "mrcc",
        "cfour-files": "cfour",
        "xtb-native": "xtb",
        "pyscf-structured": "pyscf",
    }
    name = aliases.get(name, name)
    if name not in supported_hessian_engines():
        choices = ", ".join(supported_hessian_engines())
        raise ValueError(f"unsupported Hessian engine {engine!r}; choose one of {choices}")
    return name
