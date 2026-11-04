"""Immutable loader for versioned ZAFF artifacts produced by ARCHITECT."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .compatibility import normalize_legacy_zaff_payload


ZAFF_FORCE_FIELD_SCHEMA = "matrix.zaff.force_field.v1"


@dataclass(frozen=True)
class ZaffZoomArtifact:
    identifier: str
    hessian_hartree_per_bohr2: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier = str(self.identifier).strip()
        hessian = np.asarray(self.hessian_hartree_per_bohr2, dtype=float)
        if not identifier:
            raise ValueError("ZAFF zoom identifier must be nonempty")
        if (
            hessian.ndim != 2
            or hessian.shape[0] != hessian.shape[1]
            or np.any(~np.isfinite(hessian))
            or not np.allclose(hessian, hessian.T, atol=1.0e-12, rtol=0.0)
        ):
            raise ValueError("ZAFF zoom Hessian must be finite, square and symmetric")
        hessian = 0.5 * (hessian + hessian.T)
        hessian.setflags(write=False)
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "hessian_hartree_per_bohr2", hessian)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata or {})),
        )


@dataclass(frozen=True)
class ZaffArtifact:
    """Runtime-ready subset of a compiled ZAFF force-field artifact."""

    atoms: tuple[str, ...]
    reference_coordinates_angstrom: np.ndarray
    energy_reference_hartree: float
    gradient_reference_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    zoom_levels: tuple[ZaffZoomArtifact, ...] = ()
    active_zoom_level: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    anharmonic_model: Mapping[str, Any] | None = None
    anharmonic_model_present: bool = False
    geminal_xh_couplings_present: bool = False
    source_path: Path | None = None
    sha256: str = ""
    schema: str = ZAFF_FORCE_FIELD_SCHEMA

    def __post_init__(self) -> None:
        atoms = tuple(str(atom).strip() for atom in self.atoms)
        coordinates = np.asarray(self.reference_coordinates_angstrom, dtype=float)
        gradient = np.asarray(self.gradient_reference_hartree_per_bohr, dtype=float).reshape(-1)
        hessian = np.asarray(self.hessian_hartree_per_bohr2, dtype=float)
        dimension = 3 * len(atoms)
        if (
            not atoms
            or any(not atom for atom in atoms)
            or coordinates.shape != (len(atoms), 3)
            or gradient.shape != (dimension,)
            or hessian.shape != (dimension, dimension)
        ):
            raise ValueError("ZAFF artifact dimensions are inconsistent")
        if (
            np.any(~np.isfinite(coordinates))
            or np.any(~np.isfinite(gradient))
            or np.any(~np.isfinite(hessian))
            or not np.isfinite(self.energy_reference_hartree)
        ):
            raise ValueError("ZAFF artifact contains non-finite values")
        if not np.allclose(hessian, hessian.T, atol=1.0e-12, rtol=0.0):
            raise ValueError("ZAFF artifact Hessian must be symmetric")
        if self.schema != ZAFF_FORCE_FIELD_SCHEMA:
            raise ValueError(f"unsupported ZAFF force-field schema: {self.schema}")
        zooms = tuple(self.zoom_levels)
        identifiers = tuple(item.identifier for item in zooms)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ZAFF zoom identifiers must be unique")
        if any(item.hessian_hartree_per_bohr2.shape != hessian.shape for item in zooms):
            raise ValueError("ZAFF zoom Hessian dimension differs from the artifact")
        active = str(self.active_zoom_level).strip()
        if active and active not in identifiers:
            raise ValueError(f"unknown active ZAFF zoom level: {active}")
        coordinates = coordinates.copy()
        gradient = gradient.copy()
        hessian = 0.5 * (hessian + hessian.T)
        coordinates.setflags(write=False)
        gradient.setflags(write=False)
        hessian.setflags(write=False)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "reference_coordinates_angstrom", coordinates)
        object.__setattr__(self, "energy_reference_hartree", float(self.energy_reference_hartree))
        object.__setattr__(self, "gradient_reference_hartree_per_bohr", gradient)
        object.__setattr__(self, "hessian_hartree_per_bohr2", hessian)
        object.__setattr__(self, "zoom_levels", zooms)
        object.__setattr__(self, "active_zoom_level", active)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata or {})),
        )
        object.__setattr__(
            self,
            "anharmonic_model",
            (
                None
                if self.anharmonic_model is None
                else MappingProxyType(dict(self.anharmonic_model))
            ),
        )
        object.__setattr__(
            self,
            "anharmonic_model_present",
            bool(self.anharmonic_model_present or self.anharmonic_model is not None),
        )
        object.__setattr__(
            self,
            "geminal_xh_couplings_present",
            bool(self.geminal_xh_couplings_present),
        )
        object.__setattr__(
            self,
            "source_path",
            None if self.source_path is None else Path(self.source_path).resolve(),
        )
        object.__setattr__(self, "sha256", str(self.sha256))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        source_path: Path | None = None,
        sha256: str = "",
    ) -> "ZaffArtifact":
        payload = normalize_legacy_zaff_payload(payload)
        if payload.get("schema") != ZAFF_FORCE_FIELD_SCHEMA:
            raise ValueError(f"unsupported ZAFF force-field schema: {payload.get('schema')!r}")
        atoms = tuple(str(atom) for atom in payload["atoms"])
        dimension = 3 * len(atoms)
        gradient = np.asarray(
            payload.get("gradient_reference_hartree_per_bohr", np.zeros(dimension)),
            dtype=float,
        )
        zooms = tuple(
            ZaffZoomArtifact(
                identifier=str(item["identifier"]),
                hessian_hartree_per_bohr2=np.asarray(
                    item["hessian_hartree_per_bohr2"], dtype=float
                ),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("zoom_levels", ())
        )
        return cls(
            atoms=atoms,
            reference_coordinates_angstrom=np.asarray(
                payload["reference_coordinates_angstrom"], dtype=float
            ),
            energy_reference_hartree=float(payload.get("energy_reference_hartree", 0.0)),
            gradient_reference_hartree_per_bohr=gradient,
            hessian_hartree_per_bohr2=np.asarray(
                payload["hessian_hartree_per_bohr2"], dtype=float
            ),
            zoom_levels=zooms,
            active_zoom_level=str(payload.get("active_zoom_level", "")),
            metadata=dict(payload.get("metadata", {})),
            anharmonic_model=(
                None
                if payload.get("anharmonic_model") is None
                else dict(payload["anharmonic_model"])
            ),
            anharmonic_model_present=payload.get("anharmonic_model") is not None,
            geminal_xh_couplings_present=bool(payload.get("geminal_xh_couplings")),
            source_path=source_path,
            sha256=sha256,
        )

    def hessian(self, zoom_level: str | None = None) -> np.ndarray:
        # FAST_RIGID_TORSIONAL is a runtime policy, not a Hessian zoom level.
        if zoom_level is not None and str(zoom_level).strip().upper().replace("-", "_") == "FAST_RIGID_TORSIONAL":
            zoom_level = None
        selected = self.active_zoom_level if zoom_level is None else str(zoom_level).strip()
        if not selected:
            return self.hessian_hartree_per_bohr2
        for level in self.zoom_levels:
            if level.identifier == selected:
                return level.hessian_hartree_per_bohr2
        raise ValueError(f"unknown ZAFF zoom level: {selected}")


def load_zaff_artifact(path: Path | str) -> ZaffArtifact:
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    payload = normalize_legacy_zaff_payload(json.loads(raw))
    if not isinstance(payload, Mapping):
        raise ValueError("ZAFF artifact root must be a JSON object")
    return ZaffArtifact.from_dict(
        payload,
        source_path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "ZAFF_FORCE_FIELD_SCHEMA",
    "ZaffArtifact",
    "ZaffZoomArtifact",
    "load_zaff_artifact",
]
