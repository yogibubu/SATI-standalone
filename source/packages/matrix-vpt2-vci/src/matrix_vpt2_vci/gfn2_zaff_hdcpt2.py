"""KEYMAKER acquisition of a harmonic ZAFF / GFN2 anharmonic QFF."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path

import numpy as np

from .hdcpt2 import solve_hdcpt2
from .vci import QuarticForceField


def _factorial_divisor(indices: tuple[int, ...]) -> int:
    divisor = 1
    for count in Counter(indices).values():
        divisor *= math.factorial(count)
    return divisor


def _raw_gfn2_hessian(
    atoms: tuple[str, ...], coordinates: np.ndarray, charge: int,
    multiplicity: int, temperature_kelvin: float,
) -> np.ndarray:
    from matrix_gfn2 import GFN2Calculator

    calculator = GFN2Calculator(
        atoms, charge=charge, multiplicity=multiplicity,
        temperature_kelvin=temperature_kelvin,
    )
    value = calculator.evaluate(coordinates, retain_orbitals=True)
    if not value.converged or value.orbital_state is None:
        raise RuntimeError("GFN2-XTB SCC did not converge at a displaced geometry")
    hessian = calculator.ground_state_hessian(value.orbital_state)
    if hessian.maximum_cpscf_residual > 1.0e-8:
        raise RuntimeError("GFN2-XTB analytic Hessian CPSCF residual exceeds 1e-8")
    return np.asarray(hessian.hessian_hartree_per_bohr2, dtype=float)


def _compute_mode_stencil(
    mode: int, atoms: tuple[str, ...], geometry: np.ndarray,
    directions: np.ndarray, center: np.ndarray, steps: tuple[float, float, float],
    charge: int, multiplicity: int, temperature_kelvin: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    projected = []
    for step in steps:
        displacement = step * directions[:, mode].reshape(geometry.shape) * 0.529177210903
        plus_hessian = _raw_gfn2_hessian(
            atoms, geometry + displacement, charge, multiplicity, temperature_kelvin,
        )
        minus_hessian = _raw_gfn2_hessian(
            atoms, geometry - displacement, charge, multiplicity, temperature_kelvin,
        )
        projected.append((directions.T @ plus_hessian @ directions,
                          directions.T @ minus_hessian @ directions))
    cubic_slice = (projected[0][0] - projected[0][1]) / (2.0 * steps[0])
    coarse_column = np.diag((projected[1][0] - 2.0 * center + projected[1][1]) / steps[1]**2)
    fine_column = np.diag((projected[2][0] - 2.0 * center + projected[2][1]) / steps[2]**2)
    return mode, cubic_slice, coarse_column, fine_column


def run_gfn2_zaff_hdcpt2(
    force_field_path: Path | str,
    output_dir: Path | str,
    *,
    harmonic_modes: tuple[int, ...] = (),
    temperature_kelvin: float | None = None,
    cubic_step_sqrt_amu_bohr: float = 0.02,
    quartic_steps_sqrt_amu_bohr: tuple[float, float] = (0.08, 0.04),
    workers: int = 1,
) -> dict[str, object]:
    """Fit F2 only with ZAFF, acquire GFN2 F3/F4, and solve HDCPT2.

    Mode indices in ``harmonic_modes`` are zero based. Displaced Hessians are
    cached in ``output_dir`` and reused only for the identical field, mode
    vector, electronic temperature and stencil settings.
    """
    from matrix_chem.physical_constants import Phy, get_physical_constants
    from matrix_core import load_keymaker_protocol_library
    from matrix_gfn2 import GFN2Calculator
    from matrix_link.gfn2_zaff_composite import Gfn2ZaffCompositeSession
    from matrix_qm import cartesian_normal_modes_from_hessian
    from matrix_rovib import isotopic_masses_amu

    field_path = Path(force_field_path)
    out = Path(output_dir)
    steps = (float(cubic_step_sqrt_amu_bohr), *(float(v) for v in quartic_steps_sqrt_amu_bohr))
    if len(steps) != 3 or any(not np.isfinite(step) or step <= 0.0 for step in steps):
        raise ValueError("cubic and quartic stencil steps must be finite and positive")
    if steps[1] <= steps[2]:
        raise ValueError("quartic steps must be ordered coarse, fine")
    if workers < 1:
        raise ValueError("workers must be positive")
    raw_field = field_path.read_bytes()
    field_hash = hashlib.sha256(raw_field).hexdigest()
    field_data = json.loads(raw_field)
    residual = field_data.get("metadata", {}).get("primitive_residual_model", {})
    fit_protocol = field_data.get("metadata", {}).get("protocol_id")
    if fit_protocol not in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"}:
        raise ValueError("HDCPT2 requires an approved two-Morse GFN2-XTB + Delta(ZAFF) field")
    if (residual.get("stretch_form") != "QM_ZAFF_MORSE_MINUS_GFN2_DERIVED_MORSE"
            or "gfn2_morse_alpha_per_angstrom" not in residual):
        raise ValueError("HDCPT2 requires separate QM and GFN2 Morse terms")
    if temperature_kelvin is None:
        temperature_kelvin = 0.0 if fit_protocol.endswith(".v4") else 300.0
    if not np.isfinite(temperature_kelvin) or temperature_kelvin < 0.0:
        raise ValueError("electronic temperature must be finite and non-negative")
    if fit_protocol.endswith(".v4") and temperature_kelvin != 0.0:
        raise ValueError("v4 two-Morse HDCPT2 requires the fitted 0 K GFN2 temperature")
    manifest = load_keymaker_protocol_library().resolve_action(
        "prediction_assignment", "vpt2_vci.solve_hdcpt2_gfn2_zaff",
        options={"model": (
            "gfn2_zaff_harmonic_residual_energy_xh"
            if fit_protocol.endswith(".v4") else "gfn2_zaff_harmonic_residual"
        )},
    )
    atoms = tuple(field_data["atoms"])
    charge = int(field_data["metadata"]["gfn2_charge"])
    multiplicity = int(field_data["metadata"]["gfn2_multiplicity"])
    session = Gfn2ZaffCompositeSession(
        atoms, force_field=field_path, charge=charge, multiplicity=multiplicity,
        required_fit_protocol_id=fit_protocol,
    )
    session.calculator = GFN2Calculator(
        atoms, charge=charge, multiplicity=multiplicity,
        temperature_kelvin=temperature_kelvin,
    )
    geometry = np.asarray(session.force_field.reference_coordinates_angstrom, dtype=float)
    masses = isotopic_masses_amu(atoms)
    composite = session.evaluate(geometry, hessian=True)
    if composite.hessian_hartree_per_bohr2 is None:
        raise RuntimeError("composite F2 analytic Hessian is unavailable")
    modes = cartesian_normal_modes_from_hessian(
        composite.hessian_hartree_per_bohr2,
        masses,
        geometry / 0.529177210903,
        source="GFN2-XTB + Delta(ZAFF) F2",
    )
    frequencies = np.asarray(modes.frequencies_cm, dtype=float)
    if np.any(frequencies <= 0.0):
        raise ValueError("HDCPT2 requires real positive composite harmonic modes")
    n_modes = len(frequencies)
    if any(mode < 0 or mode >= n_modes for mode in harmonic_modes):
        raise ValueError("harmonic mode index out of range")
    directions = modes.mass_weighted_modes.reshape(n_modes, 3 * len(atoms)).T
    directions = directions / np.sqrt(np.repeat(masses, 3))[:, None]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "gfn2_hessian_stencils"
    cache.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(
        json.dumps({"field_sha256": field_hash, "temperature_kelvin": temperature_kelvin,
                    "steps": steps}, sort_keys=True).encode()
    ).hexdigest()

    center = directions.T @ _raw_gfn2_hessian(
        atoms, geometry, charge, multiplicity, temperature_kelvin,
    ) @ directions
    cubic = np.zeros((n_modes, n_modes, n_modes), dtype=float)
    quartic_coarse = np.zeros((n_modes, n_modes), dtype=float)
    quartic_fine = np.zeros((n_modes, n_modes), dtype=float)
    missing = []
    for mode in range(n_modes):
        cache_file = cache / f"mode_{mode + 1:03d}.npz"
        cached = None
        if cache_file.is_file():
            with np.load(cache_file) as candidate:
                if (str(candidate["cache_key"]) == cache_key
                        and np.allclose(candidate["mode_vector"], directions[:, mode],
                                        atol=1.0e-12, rtol=0.0)):
                    cached = (candidate["cubic"], candidate["quartic_coarse"], candidate["quartic_fine"])
        if cached is None:
            missing.append(mode)
        else:
            cubic[:, :, mode], quartic_coarse[:, mode], quartic_fine[:, mode] = cached

    def accept_stencil(computed: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> None:
        mode, cubic_slice, coarse_column, fine_column = computed
        cache_file = cache / f"mode_{mode + 1:03d}.npz"
        temporary = cache_file.with_suffix(".tmp.npz")
        try:
            np.savez(temporary, cache_key=cache_key, mode_vector=directions[:, mode],
                     cubic=cubic_slice, quartic_coarse=coarse_column, quartic_fine=fine_column)
            temporary.replace(cache_file)
        finally:
            temporary.unlink(missing_ok=True)
        cubic[:, :, mode] = cubic_slice
        quartic_coarse[:, mode] = coarse_column
        quartic_fine[:, mode] = fine_column

    if workers == 1:
        for mode in missing:
            accept_stencil(_compute_mode_stencil(
                mode, atoms, geometry, directions, center, steps,
                charge, multiplicity, temperature_kelvin,
            ))
    elif missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            futures = [pool.submit(
                _compute_mode_stencil, mode, atoms, geometry, directions, center,
                steps, charge, multiplicity, temperature_kelvin,
            ) for mode in missing]
            for future in as_completed(futures):
                accept_stencil(future.result())
    permutations = (cubic, cubic.transpose(0, 2, 1), cubic.transpose(1, 0, 2),
                    cubic.transpose(1, 2, 0), cubic.transpose(2, 0, 1), cubic.transpose(2, 1, 0))
    cubic_symmetric = np.mean(np.stack(permutations), axis=0)
    cubic_residual = float(np.linalg.norm(cubic - cubic_symmetric) / max(np.linalg.norm(cubic_symmetric), 1.0e-30))
    quartic_extrapolated = (steps[1]**2 * quartic_fine - steps[2]**2 * quartic_coarse) / (steps[1]**2 - steps[2]**2)
    quartic_residual = float(np.linalg.norm(quartic_extrapolated - quartic_extrapolated.T) / max(np.linalg.norm(quartic_extrapolated), 1.0e-30))
    quartic_symmetric = 0.5 * (quartic_extrapolated + quartic_extrapolated.T)
    if cubic_residual > 0.02 or quartic_residual > 0.02:
        raise RuntimeError("GFN2-XTB normal-mode derivative symmetry gate failed")
    const = get_physical_constants()
    hbar_au = const[Phy.PLANCK] / (const[Phy.TO_KG] * const[Phy.M_PER_B]**2) / (2.0 * math.pi)
    scale = np.sqrt(hbar_au / (2.0 * math.pi * const[Phy.C_LIGHT] * frequencies))
    cm_per_hartree = const[Phy.HARTREE] / (const[Phy.PLANCK] * const[Phy.C_LIGHT])
    cubic_rows = []
    cubic_terms = {}
    for key in combinations_with_replacement(range(n_modes), 3):
        value = float(cubic_symmetric[key] * math.prod(scale[index] for index in key) * cm_per_hartree)
        if abs(value) > 1.0e-10:
            cubic_rows.append({"modes_zero_based": key, "derivative_cm-1": value})
            cubic_terms[key] = value / _factorial_divisor(key)
    quartic_rows = []
    quartic_terms = {}
    for i in range(n_modes):
        for j in range(i, n_modes):
            key = tuple(sorted((i, i, j, j)))
            value = float(quartic_symmetric[i, j] * scale[i]**2 * scale[j]**2 * cm_per_hartree)
            quartic_rows.append({"modes_zero_based": key, "derivative_cm-1": value})
            quartic_terms[key] = value / _factorial_divisor(key)
    result = solve_hdcpt2(
        QuarticForceField(frequencies, cubic_terms, quartic_terms),
        harmonic_modes=harmonic_modes,
    )
    qff_path = out / "gfn2_zaff_hdcpt2_qff.json"
    result_path = out / "gfn2_zaff_hdcpt2_result.json"
    qff_path.write_text(json.dumps({"schema": "matrix.vpt2_vci.normal_mode_qff.v1",
                                    "harmonic_frequencies_cm-1": frequencies.tolist(),
                                    "cubic_derivatives": cubic_rows,
                                    "semidiagonal_quartic_derivatives": quartic_rows}, indent=2) + "\n")
    payload = {"schema": "matrix.vpt2_vci.hdcpt2.v1",
               "protocol_id": manifest.protocol_id, "protocol_sha256": manifest.sha256,
               "field_sha256": field_hash, "temperature_kelvin": temperature_kelvin,
               "harmonic_modes_one_based": [mode + 1 for mode in harmonic_modes],
               "harmonic_frequencies_cm-1": frequencies.tolist(),
               "fundamentals_cm-1": result.fundamentals_cm.tolist(),
               "effective_x_matrix_cm-1": result.x_matrix_cm.tolist(),
               "overtone_residuals_cm-1": result.overtone_residuals_cm.tolist(),
               "cubic_symmetry_relative_residual": cubic_residual,
               "quartic_symmetry_relative_residual": quartic_residual,
               "qff_path": str(qff_path)}
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = ["run_gfn2_zaff_hdcpt2"]
