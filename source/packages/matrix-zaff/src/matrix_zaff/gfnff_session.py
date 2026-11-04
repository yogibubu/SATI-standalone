"""Native GFN-FF session and analytic curvature provider for LINK/SYCART."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .gfnff_runtime import GFNFFDynamicsRuntime
from .nonbonded import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class NativeGFNFFEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    charges: np.ndarray | None
    execution: Mapping[str, object]


class NativeGFNFFCurvatures:
    """Positive model curvatures from the same typed terms used for E/G.

    Torsional contributions are summed over all explicit outer-atom pairs
    before applying the positive floor. They must not be expanded again by
    LINK. No Cartesian Hessian or gradient evaluations are needed.
    """

    def __init__(self, runtime: GFNFFDynamicsRuntime):
        self.runtime = runtime

    @staticmethod
    def _phi(xyz, atoms):
        a, b, c, d = xyz[np.asarray(atoms, dtype=int)]
        b1, b2, b3 = a - b, c - b, d - c
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        if min(np.linalg.norm(n1), np.linalg.norm(n2), np.linalg.norm(b2)) <= 1e-14:
            raise ValueError("singular native GFN-FF torsional curvature geometry")
        return np.arctan2(np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2)), np.dot(n1, n2))

    def primitive_curvatures(self, primitives, coordinates_angstrom):
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        if xyz.shape != (self.runtime.natoms, 3) or not np.all(np.isfinite(xyz)):
            raise ValueError("native GFN-FF curvature geometry does not match the model")
        bonds = {
            tuple(sorted(record["atoms"])):
                -2.0 * record["amplitude_hartree"] * record["exponent_per_angstrom2"]
            for record in self.runtime.bond_records
        }
        angles, torsions = {}, {}
        valence = self.runtime.valence
        for atoms, parameters in zip(valence.angle_atoms, valence.angle_parameters, strict=True):
            left, center, right = map(int, atoms)
            theta, force = parameters[:2]
            value = 2.0 * force
            if abs(np.pi - theta) >= 1e-6:
                value *= np.sin(theta)**2
            key = (center, *sorted((left, right)))
            angles[key] = angles.get(key, 0.0) + value
        for atoms, parameters in zip(valence.torsion_atoms, valence.torsion_parameters, strict=True):
            mode, period, phase, amplitude = parameters[:4]
            if mode <= 0:
                continue
            key = tuple(sorted(map(int, atoms[1:3])))
            value = period**2 * amplitude * np.cos(period * (self._phi(xyz, atoms) - phase))
            torsions[key] = torsions.get(key, 0.0) + value
        for atoms, amplitude in zip(
            valence.special_torsion_atoms, valence.special_torsion_amplitudes_hartree, strict=True
        ):
            key = tuple(sorted(map(int, atoms[1:3])))
            torsions[key] = torsions.get(key, 0.0) + 4.0 * amplitude * np.cos(2 * self._phi(xyz, atoms))
        output, matched = [], {"R": 0, "A": 0, "D": 0}
        for primitive in primitives:
            function = str(primitive.function).upper()
            atoms = tuple(int(atom) - 1 for atom in primitive.atoms)
            if function == "R" and len(atoms) == 2:
                value = bonds[tuple(sorted(atoms))]
            elif function in {"A", "L"} and len(atoms) >= 3:
                value = angles[(atoms[1], *sorted((atoms[0], atoms[2])))]
            elif function == "D" and len(atoms) >= 4:
                value = torsions[tuple(sorted(atoms[1:3]))]
            else:
                raise KeyError(f"native GFN-FF has no local curvature for {function}/{atoms}")
            if not np.isfinite(value):
                raise FloatingPointError("non-finite native GFN-FF curvature")
            output.append(max(float(value), 1e-4))
            matched["A" if function == "L" else function] += 1
        return np.asarray(output), matched


class NativeGFNFFSession:
    """Prepared internal calculator; a complete ARCHITECT model is mandatory."""

    source = "GFN-FF (native MATRIX resident calculator)"
    engine = "matrix_gfnff"

    def __init__(
        self, atoms: Sequence[str], *, force_field: Path | str | None,
        charge: int = 0, multiplicity: int = 1, processors: int = 1,
    ):
        if force_field is None:
            raise ValueError(
                "native GFN-FF requires force_field=<complete ARCHITECT model JSON>; "
                "libxtb is only available through the explicit gfnff_reference backend"
            )
        if not 1 <= int(processors) <= 8:
            raise ValueError("GFN-FF processors must be between 1 and 8")
        payload = Path(force_field).expanduser().read_bytes()
        record = json.loads(payload)
        # Accept either a compact compiled model or the model within an
        # existing artifact. Do not construct the artifact's dense Hessian.
        model = record.get("metadata", {}).get("diagonal_seed_model", record)
        if tuple(model.get("atoms", ())) != tuple(atoms):
            raise ValueError("native GFN-FF model atom ordering differs from LINK")
        if abs(sum(model.get("gfnff_eeq", {}).get("fragment_charges", ())) - charge) > 1e-10:
            raise ValueError("native GFN-FF model charge differs from LINK")
        metadata = record.get("metadata", {})
        if int(metadata.get("multiplicity", model.get("multiplicity", multiplicity))) != multiplicity:
            raise ValueError("native GFN-FF model multiplicity differs from LINK")
        self.runtime = GFNFFDynamicsRuntime(model)
        self.topology_curvatures = NativeGFNFFCurvatures(self.runtime)
        self.model_sha256 = hashlib.sha256(payload).hexdigest()
        self._charges_pending = True
        from threadpoolctl import threadpool_limits

        self._threadpool_limiter = threadpool_limits(limits=int(processors))

    def evaluate(self, coordinates_angstrom: np.ndarray) -> NativeGFNFFEvaluation:
        result = self.runtime.evaluate(np.asarray(coordinates_angstrom) / BOHR_TO_ANGSTROM)
        charges = result.eeq_charges if self._charges_pending else None
        self._charges_pending = False
        return NativeGFNFFEvaluation(
            result.energy_hartree, result.gradient_hartree_per_bohr, charges,
            {**result.execution, "engine": self.engine, "model_sha256": self.model_sha256},
        )
