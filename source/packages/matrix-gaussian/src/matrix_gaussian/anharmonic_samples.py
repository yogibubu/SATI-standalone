"""Recover already calculated finite E/G/H samples from Gaussian anharmonic logs.

No electronic jobs, F3/F4 differencing, or model-dependent coordinate guesses
are performed. The archived Cartesian displacement vectors and printed nuclear
repulsions determine and validate the common finite displacement amplitude.
Only complete paired first-step Cartesian stencils in the verified FCHK frame
are accepted. Raw electronic derivatives remain Cartesian in atomic units.
"""

from hashlib import sha256
from pathlib import Path
import re
import numpy as np
from scipy.optimize import least_squares
from .fchk import _read_fchk_blocks, lower_to_symmetric, BOHR_TO_ANGSTROM
from .parsers import _parse_last_point_energy, _parse_last_force_block
from .parsers import _parse_cartesian_hessian_block, _parse_last_orientation


def _nuclear_repulsion(geometry, charges):
    return sum(charges[i]*charges[j]/np.linalg.norm(geometry[i]-geometry[j])
               for i in range(len(charges)) for j in range(i))


def _fchk_roundoff(path, label, shape):
    """Half a last printed decimal unit for each archived real value."""
    source = Path(path).read_text().splitlines()
    header = next(i for i, line in enumerate(source)
                  if line.startswith(label + " ") and "N=" in line)
    count = int(source[header].split("N=")[1])
    tokens = []
    for line in source[header+1:]:
        tokens.extend(line.split())
        if len(tokens) >= count:
            break
    bounds = []
    for token in tokens[:count]:
        mantissa, exponent = token.upper().replace("D", "E").split("E")
        decimals = len(mantissa.split(".")[1])
        bounds.append(0.5 * 10.0**(int(exponent)-decimals))
    return np.asarray(bounds).reshape(shape)


def _nuclear_roundoff_bound(geometry, charges, coordinate_bounds):
    gradient = np.zeros_like(geometry)
    for i in range(len(charges)):
        for j in range(i):
            delta = geometry[i]-geometry[j]
            term = -charges[i]*charges[j]*delta/np.linalg.norm(delta)**3
            gradient[i] += term
            gradient[j] -= term
    return float(np.sum(abs(gradient)*coordinate_bounds))


def read_gaussian_anharmonic_samples(log_path, fchk_path, *, nuclear_tolerance_hartree=1e-8,
                                    allow_postprocessing_failure=False):
    """Return finite cached samples and source-integrity evidence, never F3/F4.

    The explicit failure-recovery option admits only the terminal L716 RotETr
    error after the numerical stencil. All paired geometry, E/G/H and frame
    checks still apply; the electronic job remains recorded as failed.
    A new fit may reuse these observations only within its explicitly frozen
    finite sample budget. Parsing an archive does not authorize acquisition.
    """
    log_path = Path(log_path)
    fchk_path = Path(fchk_path)
    text = log_path.read_text(errors="replace")
    blocks = _read_fchk_blocks(fchk_path)
    if not isinstance(allow_postprocessing_failure, bool):
        raise ValueError("Explicit boolean postprocessing-recovery policy required")
    normally_terminated = text.rfind("Normal termination of Gaussian") > text.rfind("Error termination")
    recoverable_failure = (
        not normally_terminated and allow_postprocessing_failure
        and re.search(r"Wrong NETrnD in RotETr\.\s*Error termination via Lnk1e in [^\n]*/l716\.exe", text) is not None
        and text.rfind("Wrong NETrnD in RotETr.") > text.rfind("Re-enter D2Numr:") >= 0
    )
    if not normally_terminated and not recoverable_failure:
        raise ValueError("Normally terminated source or explicit completed-stencil RotETr recovery required")
    xyz = np.asarray(blocks["Current cartesian coordinates"], float).reshape(-1, 3)
    charges = np.asarray(blocks["Nuclear charges"], float)
    flags = np.asarray(blocks["Anharmonic displacement flags"], int)
    vectors = np.asarray(blocks["Anharmonic displacements"], float).reshape(len(flags), *xyz.shape)
    if len(set(flags)) != len(flags) or charges.shape != (len(xyz),):
        raise ValueError("Unique archived displacement identities and nuclear charges required")
    marker = re.compile(r"Re-enter D2Numr:\s+normal mode\s+(\d+) step-(up|down)\s+number\s+(\d+)")
    markers = list(marker.finditer(text))
    starts = list(re.finditer(r"D2Numr \.\.\. symmetry will not be used\.", text))
    if not starts or not markers or any(int(m[3]) != 1 for m in markers):
        raise ValueError("Only complete paired first-step Cartesian stencils are supported")
    start = next((s.start() for s in starts if s.start() < markers[0].start()), None)
    if start is None:
        raise ValueError("Stencil start is missing")
    prefix = text[:start].splitlines()
    orientation = _parse_last_orientation(prefix, source_path=log_path)
    if orientation is None:
        raise ValueError("Parent orientation missing before displacement sampling")
    hstarts = [
        i for i, line in enumerate(prefix) if "Force constants in Cartesian coordinates" in line
    ]
    if not hstarts:
        raise ValueError("Parent Hessian missing before displacement sampling")
    parent = _parse_cartesian_hessian_block(prefix[hstarts[-1] + 1 :], size=xyz.size)
    reference_hessian = lower_to_symmetric(blocks["Cartesian Force Constants"])
    recovered_reference = False
    last_hessian_error = None
    coordinate_bounds = np.zeros_like(xyz)
    vector_bounds = np.zeros_like(vectors)
    last_vector_bound = np.zeros_like(xyz)
    if recoverable_failure:
        # An aborted final L716 may leave the last displaced geometry/Hessian
        # in /Mol/. Infer the origin from that actual geometry and all printed
        # nuclear repulsions, instead of relabeling it as the equilibrium.
        observations = []
        previous = start
        for mark in markers:
            mode = int(mark[1]); sign = 1 if mark[2] == "up" else -1
            if mode not in flags:
                raise ValueError("Unknown mode in completed-stencil recovery")
            segment = text[previous:mark.start()]
            values = re.findall(r"nuclear repulsion energy\s+([-+\d.EeDd]+)", segment)
            if len(values) != 1:
                raise ValueError("Unambiguous nuclear repulsion required for every displacement")
            observations.append((sign*vectors[int(np.flatnonzero(flags == mode)[0])],
                                 float(values[0].replace("D", "E")), segment))
            previous = mark.end()
        last_vector = observations[-1][0]
        def origin_residual(step):
            origin = xyz-float(step[0])*last_vector
            return np.array([_nuclear_repulsion(origin+float(step[0])*v, charges)-e
                             for v,e,_ in observations])
        fit = least_squares(origin_residual,[0.02],bounds=(1e-6,0.2),
                            xtol=1e-14,ftol=1e-14,gtol=1e-14)
        coordinate_bounds = _fchk_roundoff(fchk_path, "Current cartesian coordinates", xyz.shape)
        vector_bounds = _fchk_roundoff(fchk_path, "Anharmonic displacements", vectors.shape)
        last_mode = int(markers[-1][1])
        last_vector_bound = vector_bounds[int(np.flatnonzero(flags == last_mode)[0])]
        origin = xyz-float(fit.x[0])*last_vector
        recovery_bounds = []
        for mark, (vector, _, _) in zip(markers, observations):
            bound = coordinate_bounds + float(fit.x[0])*(
                last_vector_bound + vector_bounds[int(np.flatnonzero(flags == int(mark[1]))[0])])
            recovery_bounds.append(_nuclear_roundoff_bound(
                origin+float(fit.x[0])*vector, charges, bound))
        if not fit.success or np.any(abs(origin_residual(fit.x)) >
                                      nuclear_tolerance_hartree+np.asarray(recovery_bounds)):
            raise ValueError("Last displaced checkpoint cannot reconstruct the common reference")
        last_lines = observations[-1][2].splitlines()
        last_starts = [i for i,line in enumerate(last_lines)
                       if "Force constants in Cartesian coordinates" in line]
        if len(last_starts) != 1:
            raise ValueError("Last displaced checkpoint Hessian is ambiguous")
        last_hessian = _parse_cartesian_hessian_block(last_lines[last_starts[0]+1:],size=xyz.size)
        last_hessian_error = float(np.max(abs(last_hessian-reference_hessian)))
        if last_hessian_error > 5e-6:
            raise ValueError("Checkpoint does not match the last completed displaced Hessian")
        xyz = xyz-float(fit.x[0])*last_vector
        reference_hessian = parent.copy()
        recovered_reference = True
    if np.max(abs(orientation.coordinates_angstrom - xyz * BOHR_TO_ANGSTROM)) > 2e-6:
        raise ValueError("Displaced derivatives are not in the verified FCHK Cartesian frame")
    masses = np.asarray(blocks["Real atomic weights"], float)[: len(xyz)]
    centered = xyz * BOHR_TO_ANGSTROM - np.average(xyz * BOHR_TO_ANGSTROM, axis=0, weights=masses)
    parent_errors = [float(np.max(abs(parent - reference_hessian)))]
    # Gaussian restores input axes for the equilibrium printout; the finite
    # stencil is normally printed in standard axes. Verify both conventions.
    for begin, line in enumerate(prefix):
        if "Input orientation:" not in line and "Standard orientation:" not in line:
            continue
        separators = [i for i in range(begin + 1, len(prefix)) if set(prefix[i].strip()) == {"-"}]
        if len(separators) < 3:
            continue
        frame = _parse_last_orientation(prefix[begin : separators[2] + 1], source_path=log_path)
        if frame is None or frame.coordinates_angstrom.shape != xyz.shape:
            continue
        original = frame.coordinates_angstrom - np.average(
            frame.coordinates_angstrom, axis=0, weights=masses
        )
        left, _, right = np.linalg.svd((original * masses[:, None]).T @ centered)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(left @ right)
        rotation = left @ correction @ right
        if np.max(abs(original @ rotation - centered)) > 3e-6:
            continue
        transform = np.kron(np.eye(len(xyz)), rotation.T)
        parent_errors.append(
            float(np.max(abs(transform @ parent @ transform.T - reference_hessian)))
        )
    parent_error = min(parent_errors)
    if parent_error > 5e-6:
        raise ValueError("Parent log/FCHK Hessians differ in all verified Cartesian frames")
    records = []
    seen = set()
    previous = start
    for mark in markers:
        mode = int(mark[1])
        sign = 1 if mark[2] == "up" else -1
        if mode not in flags or (mode, sign) in seen:
            raise ValueError("Missing or duplicate archived displacement mode")
        seen.add((mode, sign))
        segment = text[previous : mark.start()]
        lines = segment.splitlines()
        repulsions = re.findall(r"nuclear repulsion energy\s+([-+\d.EeDd]+)", segment)
        hstarts = [
            i for i, line in enumerate(lines) if "Force constants in Cartesian coordinates" in line
        ]
        if len(repulsions) != 1 or len(hstarts) != 1:
            raise ValueError("Ambiguous finite-point energy/geometry/Hessian association")
        energy = _parse_last_point_energy(segment)
        forces = _parse_last_force_block(lines)
        if energy is None or forces is None or forces.shape != xyz.shape:
            raise ValueError("Complete finite E/G observations required")
        hessian = _parse_cartesian_hessian_block(lines[hstarts[0] + 1 :], size=xyz.size)
        vector = vectors[int(np.flatnonzero(flags == mode)[0])] * sign
        records.append(
            dict(
                mode=mode,
                direction=mark[2],
                vector=vector,
                nuclear_repulsion=float(repulsions[0].replace("D", "E")),
                energy_hartree=float(energy),
                gradient_hartree_per_bohr=-forces.reshape(-1),
                hessian_hartree_per_bohr2=hessian,
                marker_line=text.count("\n", 0, mark.start()) + 1,
            )
        )
        previous = mark.end()
    if seen != {(int(mode), sign) for mode in flags for sign in [-1, 1]}:
        raise ValueError("Incomplete paired finite stencil")

    paired_hessian_errors = []
    for mode in flags:
        pair = [row["hessian_hartree_per_bohr2"] for row in records if row["mode"] == mode]
        paired_hessian_errors.append(
            float(np.max(abs((pair[0] + pair[1]) / 2 - reference_hessian)))
        )
    if max(paired_hessian_errors) > 2e-3:
        raise ValueError("Finite Hessians do not share the archived FCHK Cartesian frame")

    def residual(step):
        return np.array(
            [
                _nuclear_repulsion(xyz + float(step[0]) * row["vector"], charges) - row["nuclear_repulsion"]
                for row in records
            ]
        )

    inferred = least_squares(
        residual, [0.02], bounds=(1e-6, 0.2), xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    step = float(inferred.x[0])
    errors = residual([step])
    rounding_bounds = np.asarray([
        _nuclear_roundoff_bound(xyz+step*row["vector"], charges,
            coordinate_bounds+step*(last_vector_bound+
                vector_bounds[int(np.flatnonzero(flags == row["mode"])[0])]))
        for row in records])
    if not inferred.success or np.any(abs(errors) > nuclear_tolerance_hartree+rounding_bounds):
        raise ValueError("A common archived displacement does not reproduce nuclear repulsions")
    samples = []
    for row, error in zip(records, errors):
        sample = dict(row)
        vector = sample.pop("vector")
        sample["coordinates_bohr"] = xyz + step * vector
        sample["nuclear_repulsion_error_hartree"] = float(error)
        samples.append(sample)
    provenance = dict(
        schema="matrix.gaussian.cached_anharmonic_samples.v1",
        log_sha256=sha256(log_path.read_bytes()).hexdigest(),
        fchk_sha256=sha256(fchk_path.read_bytes()).hexdigest(),
        common_displacement_step_bohr=step,
        sample_count=len(samples),
        parent_hessian_frame_error=parent_error,
        parent_hessian_checked_against_reference_fchk=not recovered_reference,
        last_displaced_hessian_checkpoint_error=last_hessian_error,
        paired_hessian_center_errors=paired_hessian_errors,
        maximum_nuclear_repulsion_error_hartree=float(np.max(abs(errors))),
        nuclear_repulsion_roundoff_bounds_hartree=rounding_bounds.tolist(),
        nuclear_repulsion_absolute_tolerance_hartree=nuclear_tolerance_hartree,
        derivative_orders_observed=[0, 1, 2],
        higher_derivative_targets_used=False,
        new_electronic_evaluations=0,
        electronic_job_normally_terminated=normally_terminated,
        reference_reconstructed_from_last_displaced_checkpoint=recovered_reference,
        reference_coordinates_bohr=xyz.tolist(),
        parent_hessian_source="verified parent log before the complete stencil" if recovered_reference else "FCHK checked against parent log",
        completed_stencil_recovered_after_postprocessing_failure=recoverable_failure,
        postprocessing_failure="Wrong NETrnD in RotETr" if recoverable_failure else None,
        geometry_source="archived Cartesian displacement vectors and common step inferred from printed nuclear repulsions; parent frame/Hessian verified",
    )
    return samples, provenance
