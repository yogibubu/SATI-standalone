"""Restartable point-calculator adapters for LINK path optimizations."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .scan import QMScanBackend, ScanPoint, _run_gaussian_scan_point, read_point_result, write_point_result


class GaussianPointCalculator:
    """Expose a Gaussian-compatible backend through LINK's ``evaluate`` boundary.

    Geometry hashes make electronic evaluations reusable after interruption.
    The most recent checkpoint is also passed to the next geometry, allowing
    Gaussian/GDV to reuse the converged wavefunction without coupling the
    optimizer to a molecule-specific driver.
    """

    def __init__(
        self,
        backend: QMScanBackend,
        atoms: Sequence[str],
        run_directory: Path | str,
    ) -> None:
        normalized = str(backend.name).strip().casefold()
        if normalized not in {"gaussian", "g16", "gdv"}:
            raise ValueError("GaussianPointCalculator requires gaussian, g16, or gdv")
        self.backend = backend
        self.atoms = tuple(str(atom) for atom in atoms)
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._restart_artifact: Path | None = None
        self.electronic_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _geometry_key(coordinates_angstrom: np.ndarray) -> str:
        canonical = np.asarray(coordinates_angstrom, dtype="<f8")
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()[:24]

    def evaluate(self, coordinates_angstrom: np.ndarray):
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("point-calculator coordinates must have shape natoms x 3")
        key = self._geometry_key(coordinates)
        point_dir = self.run_directory / f"point-{key}"
        result_path = point_dir / "point.json"
        if result_path.is_file():
            self.cache_hits += 1
            result = read_point_result(result_path)
            checkpoint = point_dir / "gauin.chk"
            if checkpoint.is_file():
                self._restart_artifact = checkpoint
            return result

        point_dir.mkdir(parents=True, exist_ok=True)
        backend = self.backend
        if self._restart_artifact is not None:
            backend = replace(
                backend,
                restart_artifact=self._restart_artifact,
                gaussian_native_checkpoint_restart=True,
            )
        point = ScanPoint(
            index=self.electronic_calls,
            displacement=0.0,
            coordinates_angstrom=coordinates,
        )
        result = _run_gaussian_scan_point(backend, self.atoms, point, point_dir)
        write_point_result(result_path, result)
        checkpoint = point_dir / "gauin.chk"
        if checkpoint.is_file():
            self._restart_artifact = checkpoint
        self.electronic_calls += 1
        return result

    def evaluate_hessian(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        """Return a restartable analytic Cartesian Hessian in hartree/bohr²."""

        from matrix_gaussian.fchk import lower_to_symmetric

        return lower_to_symmetric(self.evaluate_hessian_data(coordinates_angstrom).cartesian_hessian_lower)

    def evaluate_hessian_data(self, coordinates_angstrom: np.ndarray):
        """Return the complete formatted-checkpoint record for a Hessian point."""

        from matrix_gaussian import formchk_checkpoint
        from matrix_gaussian.fchk import read_gaussian_fchk

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("Hessian coordinates must have shape natoms x 3")
        key = self._geometry_key(coordinates)
        point_dir = self.run_directory / f"hessian-{key}"
        fchk_path = point_dir / "gauin.fchk"
        if fchk_path.is_file():
            self.cache_hits += 1
            return read_gaussian_fchk(fchk_path)
        point_dir.mkdir(parents=True, exist_ok=True)
        route = str(self.backend.route).strip()
        if "freq" not in route.casefold():
            route = f"{route} Freq".strip()
        backend = replace(self.backend, route=route, gradient_mode="energy")
        if self._restart_artifact is not None:
            backend = replace(
                backend,
                restart_artifact=self._restart_artifact,
                gaussian_native_checkpoint_restart=True,
            )
        point = ScanPoint(
            index=self.electronic_calls,
            displacement=0.0,
            coordinates_angstrom=coordinates,
        )
        _run_gaussian_scan_point(backend, self.atoms, point, point_dir)
        checkpoint = point_dir / "gauin.chk"
        if not checkpoint.is_file():
            raise RuntimeError("Gaussian frequency calculation produced no checkpoint")
        formchk_checkpoint(checkpoint, fchk_path, timeout=self.backend.timeout)
        self._restart_artifact = checkpoint
        self.electronic_calls += 1
        return read_gaussian_fchk(fchk_path)


__all__ = ["GaussianPointCalculator"]
