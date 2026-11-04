"""TRINITY isotope DeltaBvib acquisition from the resident v4 composite."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM, read_xyzin_geometry
from matrix_chem.topology.elements import atomic_symbol
from matrix_qm import cartesian_normal_modes_from_hessian, hessian_input_from_engine
from matrix_rovib import isotopic_masses_amu
from matrix_smith import build_gic_b_matrix, read_gic_definition_from_xyzin
from matrix_trinity.anharmonic_sampling import semidiagonal_cubic_from_modal_gradient_stencil
from matrix_trinity.computed_cubic_job import (
    acquire_computed_cubic_job_from_hessians,
    write_computed_cubic_job,
)
from matrix_trinity.curvilinear_rovib import (
    CurvilinearIsotopologueDefinition,
    curvilinear_deltabvib_from_semidiagonal_cubic,
    curvilinear_deltabvib_to_dict,
    curvilinear_isotopologue_state_from_internal_field,
)
from matrix_trinity.isotopic_internal_qff import internal_cubic_from_parent_normal_field


PROTOCOL_ID = "trinity-isotope-deltabvib-acquisition.v1"
RESULT_SCHEMA = "matrix.trinity.isotope_deltabvib_acquisition.v1"
METHODS = {"auto", "isotope_energy_gradient", "parent_analytic_hessian"}
BOHR = float(BOHR_TO_ANGSTROM)


def choose_isotope_acquisition_method(
    *,
    isotope_count: int,
    mode_count: int,
    gradient_seconds: float,
    hessian_seconds: float,
    require_full_cubic: bool = False,
) -> str:
    """Choose by requested derivative coverage and measured point costs."""

    if isotope_count < 1 or mode_count < 1:
        raise ValueError("isotope and mode counts must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in (gradient_seconds, hessian_seconds)):
        raise ValueError("measured derivative point costs must be positive and finite")
    if require_full_cubic:
        return "parent_analytic_hessian"
    return (
        "isotope_energy_gradient"
        if 4 * mode_count * isotope_count * gradient_seconds
        <= 4 * mode_count * hessian_seconds
        else "parent_analytic_hessian"
    )


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def acquire_isotope_deltabvib(
    *,
    geometry_path: Path | str,
    hessian_path: Path | str,
    force_field_path: Path | str,
    msr_path: Path | str,
    output_dir: Path | str,
    method: str = "auto",
    step_sqrt_amu_bohr: float = 0.02,
    require_full_cubic: bool = False,
) -> Path:
    """Acquire once from parent Hessians or per isotope from E/G, then contract."""

    from matrix_core import ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID, load_keymaker_protocol_library
    from matrix_link import prepare_modal_gradient_batch
    from matrix_link.gfn2_zaff_composite import Gfn2ZaffCompositeSession
    from matrix_morpheus import read_msr_input
    from matrix_morpheus.dual_deltavib import combine_isotopic_deltabvib_channels

    if method not in METHODS:
        raise ValueError(f"unsupported TRINITY acquisition method: {method}")
    if require_full_cubic and method == "isotope_energy_gradient":
        raise ValueError("complete cubic coverage requires parent Hessian acquisition")
    step = float(step_sqrt_amu_bohr)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("TRINITY displacement step must be positive and finite")
    geometry_path = Path(geometry_path).resolve()
    hessian_path = Path(hessian_path).resolve()
    force_field_path = Path(force_field_path).resolve()
    msr_path = Path(msr_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "geometry": geometry_path,
        "hessian": hessian_path,
        "force_field": force_field_path,
        "msr": msr_path,
    }
    source_sha256 = {name: _sha(path) for name, path in source_paths.items()}
    protocol = load_keymaker_protocol_library().resolve(PROTOCOL_ID)
    geometry = read_xyzin_geometry(geometry_path)
    xyz = np.asarray(geometry.coordinates_angstrom, dtype=float)
    hessian_source = hessian_input_from_engine("xyzin", hessian_path)
    if tuple(atomic_symbol(int(number)) for number in hessian_source.atomic_numbers) != tuple(geometry.atoms):
        raise ValueError("TRINITY Hessian and SMITH geometry have different atoms")
    if not np.allclose(xyz / BOHR, hessian_source.cartesian_coordinates_bohr, atol=1e-8):
        raise ValueError("TRINITY geometry and frozen composite Hessian disagree")
    parsed = read_msr_input(msr_path)
    if tuple(parsed.geometry.atoms) != tuple(geometry.atoms):
        raise ValueError("MSR isotope observations have another atom order")
    msr_xyz = np.asarray(parsed.geometry.coordinates_angstrom, dtype=float)
    if np.max(np.abs(
        np.linalg.norm(msr_xyz[:, None] - msr_xyz[None, :], axis=2)
        - np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    )) > 0.01:
        raise ValueError("MSR isotope observations differ from the frozen molecular geometry")
    definitions = tuple(
        CurvilinearIsotopologueDefinition(row.label, row.substitutions)
        for row in parsed.observations
    )
    if not definitions:
        raise ValueError("TRINITY requires at least one isotopologue")
    session = Gfn2ZaffCompositeSession(
        geometry.atoms,
        force_field=force_field_path,
        required_fit_protocol_id=ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID,
    )
    reference = session.evaluate(xyz, hessian=True)
    reference_hessian = np.asarray(reference.hessian_hartree_per_bohr2)
    frozen_hessian = np.asarray(hessian_source.cartesian_hessian)
    reference_hessian = 0.5 * (reference_hessian + reference_hessian.T)
    frozen_hessian = 0.5 * (frozen_hessian + frozen_hessian.T)
    relative = float(np.linalg.norm(reference_hessian - frozen_hessian) / np.linalg.norm(frozen_hessian))
    if relative > 1e-8:
        raise ValueError("TRINITY composite Hessian differs from its frozen source")
    if np.max(np.abs(reference.gradient_hartree_per_bohr)) > 1e-6:
        raise ValueError("TRINITY composite reference gradient is not stationary")
    smith = read_gic_definition_from_xyzin(geometry_path)

    def b_matrix(at_xyz: np.ndarray) -> np.ndarray:
        return np.asarray(
            build_gic_b_matrix(smith, coordinates_angstrom=at_xyz).rows,
            dtype=float,
        ) * BOHR

    b = b_matrix(xyz)
    nmode = b.shape[0]
    if b.shape[1] != xyz.size or nmode != xyz.size - 6 or np.linalg.matrix_rank(b) != nmode:
        raise ValueError("TRINITY requires a full-rank nonredundant SMITH basis")
    pilot = None
    selected = method
    if method == "auto" and require_full_cubic:
        selected = "parent_analytic_hessian"
    elif method == "auto":
        started = perf_counter()
        session.evaluate(xyz, hessian=False)
        gradient_seconds = perf_counter() - started
        started = perf_counter()
        session.evaluate(xyz, hessian=True)
        hessian_seconds = perf_counter() - started
        estimated_gradient = 4 * nmode * len(definitions) * gradient_seconds
        estimated_hessian = 4 * nmode * hessian_seconds
        selected = choose_isotope_acquisition_method(
            isotope_count=len(definitions), mode_count=nmode,
            gradient_seconds=gradient_seconds,
            hessian_seconds=hessian_seconds,
        )
        pilot = {
            "gradient_seconds": gradient_seconds,
            "hessian_seconds": hessian_seconds,
            "estimated_gradient_route_seconds": estimated_gradient,
            "estimated_hessian_route_seconds": estimated_hessian,
        }
    modes = cartesian_normal_modes_from_hessian(
        reference_hessian,
        np.asarray(hessian_source.masses_amu),
        xyz / BOHR,
    )
    cartesian_per_q = modes.mass_weighted_modes.reshape((nmode, -1)).T
    cartesian_per_q /= np.sqrt(np.repeat(hessian_source.masses_amu, 3))[:, None]
    started = perf_counter()
    if selected == "isotope_energy_gradient":
        harmonic = internal_cubic_from_parent_normal_field(
            np.diag(modes.eigenvalues),
            np.zeros((nmode, nmode, nmode)),
            b,
            np.zeros((nmode, xyz.size, xyz.size)),
            cartesian_per_q,
        )
        rows = []
        derivatives = []
        for definition in definitions:
            state = curvilinear_isotopologue_state_from_internal_field(
                harmonic, b,
                isotopic_masses_amu(tuple(geometry.atoms), dict(definition.substitutions)),
                definition,
            )
            samples = []
            for displacement in (step, 2.0 * step):
                batch = prepare_modal_gradient_batch(
                    xyz / BOHR, state.modes_mw, state.masses_amu,
                    np.full(nmode, displacement),
                )
                plus = [session.evaluate(point * BOHR) for point in batch.plus_bohr]
                minus = [session.evaluate(point * BOHR) for point in batch.minus_bohr]
                gp = np.asarray([item.gradient_hartree_per_bohr for item in plus])
                gm = np.asarray([item.gradient_hartree_per_bohr for item in minus])
                cubic = semidiagonal_cubic_from_modal_gradient_stencil(
                    reference.gradient_hartree_per_bohr,
                    gp, gm, state.modes_mw, state.masses_amu,
                    batch.steps_sqrt_amu_bohr,
                ).cubic_fjkk_qmw
                samples.append((plus, minus, gp, gm, cubic))
            fine, coarse = samples
            cubic = (4 * fine[4] - coarse[4]) / 3
            relative_step = float(np.linalg.norm(fine[4] - coarse[4]) / max(np.linalg.norm(cubic), 1e-15))
            if relative_step > 0.01:
                raise ValueError(f"TRINITY isotope {state.label} cubic stencil is not step stable")
            rows.append(curvilinear_deltabvib_from_semidiagonal_cubic(xyz, state, cubic))
            directions = state.modes_mw.reshape((nmode, -1)).T
            directions /= np.sqrt(np.repeat(state.masses_amu, 3))[:, None]
            quartic_energy = []
            quartic_gradient = []
            for index in range(nmode):
                e1 = fine[0][index].energy_hartree + fine[1][index].energy_hartree - 2 * reference.energy_hartree
                e2 = coarse[0][index].energy_hartree + coarse[1][index].energy_hartree - 2 * reference.energy_hartree
                quartic_energy.append(float((e2 - 4 * e1) / step**4))
                d1 = float((fine[2][index] - fine[3][index]) @ directions[:, index]) / (2 * step)
                d2 = float((coarse[2][index] - coarse[3][index]) @ directions[:, index]) / (4 * step)
                quartic_gradient.append(float(2 * (d2 - d1) / step**2))
            if np.max(np.abs(np.asarray(quartic_energy) - quartic_gradient)) > 0.01:
                raise ValueError(f"TRINITY isotope {state.label} diagonal quartics disagree")
            derivatives.append({
                "label": state.label,
                "cubic_fjkk_qmw": cubic.tolist(),
                "quartic_fiiii_from_energy": quartic_energy,
                "quartic_fiiii_from_gradient": quartic_gradient,
                "cubic_step_relative_difference": relative_step,
            })
        derivative_path = output_dir / "isotope-mode-derivatives.json"
        _json(derivative_path, {
            "schema": "matrix.trinity.isotope_mode_derivatives.v1",
            "method": selected,
            "steps_sqrt_amu_bohr": [step, 2 * step],
            "isotopologues": derivatives,
        })
        evidence = {
            "coverage": "isotope-specific F_jkk and diagonal F_iiii",
            "evaluation_count": 4 * nmode * len(definitions),
            "derivative_path": str(derivative_path),
            "derivative_sha256": _sha(derivative_path),
        }
    else:
        b_prime = np.empty((nmode, xyz.size, xyz.size))
        step_bohr = 1e-3
        for column in range(xyz.size):
            plus = xyz.copy().reshape(-1)
            minus = xyz.copy().reshape(-1)
            plus[column] += step_bohr * BOHR
            minus[column] -= step_bohr * BOHR
            b_prime[:, :, column] = (
                b_matrix(plus.reshape(xyz.shape)) - b_matrix(minus.reshape(xyz.shape))
            ) / (2 * step_bohr)
        b_prime = 0.5 * (b_prime + b_prime.transpose(0, 2, 1))
        job, stencil = acquire_computed_cubic_job_from_hessians(
            symbols=tuple(geometry.atoms),
            coordinates_angstrom=xyz,
            masses_amu=np.asarray(hessian_source.masses_amu),
            cartesian_hessian=reference_hessian,
            b_matrix_per_bohr=b,
            b_prime_per_bohr2=b_prime,
            evaluator=lambda point: session.evaluate(point, hessian=True),
            step_sqrt_amu_bohr=step,
        )
        if stencil["step_relative_difference"] > 0.01 or stencil["raw_symmetry_residual"] > 0.01:
            raise ValueError("TRINITY parent Hessian cubic stencil is unstable")
        job_path = output_dir / "computed-f2-f3.json"
        write_computed_cubic_job(
            job_path, job,
            method="resident GFN2-XTB+Delta(ZAFF) v4 analytic Hessian stencil",
            method_artifact_sha256=_sha(force_field_path),
            hessian_sha256=_sha(hessian_path),
            acquisition_evidence=stencil,
        )
        rows = job.calculate(definitions)
        evidence = {
            "coverage": "complete parent F3 transformed to isotope GF modes",
            "evaluation_count": stencil["hessian_evaluations"],
            "computed_job_path": str(job_path),
            "computed_job_sha256": _sha(job_path),
            "stencil": stencil,
        }
    result_path = output_dir / "isotope-deltabvib.json"
    observation_path = output_dir / "corrected-observations.csv"
    comparison_path = output_dir / "isotope-comparison.json"
    combined = combine_isotopic_deltabvib_channels(
        parsed.observations, rows,
        output_csv=observation_path,
        report_json=comparison_path,
    )
    if len(combined.observations) != len(definitions):
        raise ValueError("MORPHEUS corrected observations omit an isotopologue")
    if any(_sha(path) != source_sha256[name] for name, path in source_paths.items()):
        raise ValueError("TRINITY acquisition input changed while evaluating")
    _json(result_path, {
        "schema": RESULT_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "method": selected,
        "requested_method": method,
        "require_full_cubic": bool(require_full_cubic),
        "steps_sqrt_amu_bohr": [step, 2 * step] if selected == "isotope_energy_gradient" else [step, step / 2],
        "independent_scf_per_point": True,
        "reference_hessian_relative_error": relative,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": source_sha256,
        "normal_mode_count": nmode,
        "isotopologue_count": len(rows),
        "pilot_cost": pilot,
        "sampling_wall_seconds": perf_counter() - started,
        "evidence": evidence,
        "corrected_observations": {
            "path": str(observation_path), "sha256": _sha(observation_path),
        },
        "isotope_comparison": {
            "path": str(comparison_path), "sha256": _sha(comparison_path),
        },
        "isotopologue_corrections": [curvilinear_deltabvib_to_dict(row) for row in rows],
    })
    return result_path


__all__ = [
    "METHODS", "PROTOCOL_ID", "RESULT_SCHEMA",
    "acquire_isotope_deltabvib", "choose_isotope_acquisition_method",
]
