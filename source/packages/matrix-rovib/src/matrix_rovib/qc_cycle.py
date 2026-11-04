"""Restartable GDV preparation/import and provenance-bound QC PES refinement."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.linalg import cho_factor, cho_solve, block_diag
from .tunneling_coordinates import HARTREE_TO_MHZ


def _json_numeric(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"nonportable QC artifact value: {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_numeric
        ).encode()
    ).hexdigest()


def _write(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_numeric) + "\n")
    temporary.replace(target)


def prepare_qc_cycle(
    directory,
    candidates,
    selected_ids,
    *,
    atoms,
    route,
    reference,
    qm_prior_values,
    qm_prior_covariance,
    producer_specification,
    charge=0,
    multiplicity=1,
    link0=(),
    noise_covariance=None,
    reference_energy_sigma_MHz=0.0,
    qm_source_id="QM_PES",
):
    """Prepare immutable force inputs and a resumable ledger; never execute QC.

    Refinement uses the original QM prior, not a spectral posterior recycled
    as an independent QM restraint. The latter may guide acquisition only.
    """
    from matrix_gaussian.writers import write_gaussian_point_input
    from .physical_producer import build_physical_lam_producer
    from .prediction_budget import _covariance

    import re

    if re.search(r"\b(opt|irc|scan)\b", str(route), re.IGNORECASE):
        raise ValueError("QC cycle requires fixed-geometry force calculations")
    records = {str(r["id"]): deepcopy(r) for r in candidates}
    ids = [str(k) for k in selected_ids]
    if (
        len(records) != len(candidates)
        or len(set(ids)) != len(ids)
        or not ids
        or set(ids) - set(records)
    ):
        raise ValueError("unique selected candidate identities required")
    selected = [records[k] for k in ids]
    names = list(qm_prior_values)
    values = [float(qm_prior_values[n]) for n in names]
    prior = _covariance(qm_prior_covariance, len(names), "original QM prior")
    np.linalg.cholesky(prior)
    sigma = float(reference_energy_sigma_MHz)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("invalid shared reference-energy uncertainty")
    producer = build_physical_lam_producer(producer_specification)
    spec = producer.to_specification()
    if any(record["potential"].get("energy_kind") != "electronic" for record in spec["systems"]):
        raise ValueError(
            "QC refinement requires an explicitly electronic PES component; effective bath/ZPE contributions must remain separate"
        )
    # Bind points to the common reference before writing any files.
    ref = deepcopy(reference)
    refxyz = np.asarray(ref["geometry_angstrom"], float)
    for record in selected:
        job = record["job"]
        xyz = np.asarray(job["geometry_angstrom"], float)
        q = np.asarray(job["coordinate"], float)
        tangent = np.asarray(job["cartesian_tangents_angstrom"], float)
        if (
            xyz.shape != (len(atoms), 3)
            or refxyz.shape != xyz.shape
            or tangent.shape != xyz.shape + (len(q),)
            or not all(np.all(np.isfinite(a)) for a in (xyz, refxyz, tangent))
        ):
            raise ValueError("QC geometry/tangent shape mismatch")
        if not np.allclose(job["reference_coordinate"], ref["coordinate"], rtol=0.0, atol=1.0e-12):
            raise ValueError("QC candidate references differ")
        if np.asarray(record["noise_covariance"]).shape != (len(q) + 1, len(q) + 1):
            raise ValueError("QC energy/gradient noise shape mismatch")
    noise = (
        block_diag(*(np.asarray(r["noise_covariance"], float) for r in selected))
        if noise_covariance is None
        else np.asarray(noise_covariance, float)
    )
    size = sum(len(r["job"]["coordinate"]) + 1 for r in selected)
    noise = _covariance(noise, size, "QC acquisition")
    shared = np.zeros(size)
    offset = 0
    for record in selected:
        shared[offset] = 1.0
        offset += len(record["job"]["coordinate"]) + 1
    noise = noise + sigma * sigma * np.outer(shared, shared)
    np.linalg.cholesky(noise)
    contract = dict(
        atoms=list(atoms),
        route=str(route),
        reference=ref,
        candidates=selected,
        selected_ids=ids,
        charge=int(charge),
        multiplicity=int(multiplicity),
        link0=list(link0),
        parameter_names=names,
        qm_prior_values=dict(zip(names, values, strict=True)),
        qm_prior_covariance=prior.tolist(),
        noise_covariance=noise.tolist(),
        producer=spec,
        qm_source_id=str(qm_source_id),
    )
    digest = _digest(contract)
    directory = Path(directory)
    path = directory / "cycle.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old is not None and (
        old.get("contract_sha256") != digest or _digest(old["contract"]) != digest
    ):
        raise ValueError("existing QC cycle has a different physical contract")
    directory.mkdir(parents=True, exist_ok=True)
    from tempfile import TemporaryDirectory

    jobs, contents = [], {}
    with TemporaryDirectory(prefix=".prepare-", dir=directory) as staging:
        for index, job in enumerate([ref, *[r["job"] for r in selected]]):
            stem = "reference" if index == 0 else f"point-{index - 1:05d}"
            target = Path(staging) / (stem + ".gjf")
            checkpoint = stem + ".chk"
            links = [*link0]
            if any(str(line).lower().startswith("%chk=") for line in links):
                raise ValueError("QC cycle owns each point checkpoint filename")
            write_gaussian_point_input(
                target,
                atoms,
                job["geometry_angstrom"],
                route=route,
                title="LOCK spectroscopy-directed QC " + stem,
                charge=charge,
                multiplicity=multiplicity,
                link0=tuple(links + [f"%chk={checkpoint}"]),
                ensure_force=True,
            )
            content = target.read_bytes()
            contents[target.name] = content
            jobs.append(
                dict(
                    id="reference" if index == 0 else ids[index - 1],
                    input=target.name,
                    log=stem + ".log",
                    fchk=stem + ".fchk",
                    input_sha256=hashlib.sha256(content).hexdigest(),
                    command=["gdv", target.name],
                    formchk_command=["formchk", checkpoint, stem + ".fchk"],
                )
            )
        if old is not None and old["jobs"] != jobs:
            raise ValueError("QC cycle job registry or Gaussian writer convention changed")
        for job in jobs:
            target = directory / job["input"]
            if target.exists():
                if old is None:
                    raise FileExistsError(f"refusing to replace unregistered QC input: {target}")
                if hashlib.sha256(target.read_bytes()).hexdigest() != job["input_sha256"]:
                    raise ValueError("QC cycle input changed after preparation")
        ledger = (
            old
            if old is not None
            else dict(
                schema="matrix.lock.qc_cycle.v1",
                contract_sha256=digest,
                contract=contract,
                jobs=jobs,
                status="preparing",
                results={},
                launch="Explicit execution required; this interface prepares and imports only",
            )
        )
        # Register ownership before writing any final inputs. A restart safely
        # restores missing inputs from the identical registered physical contract.
        _write(path, ledger)
        for filename, content in contents.items():
            target = directory / filename
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(content)
                temporary.replace(target)
        if ledger["status"] == "preparing":
            ledger["status"] = "prepared"
            _write(path, ledger)
    return path


def collect_qc_cycle(path, *, geometry_tolerance_angstrom=1.0e-5):
    """Import normal GDV/FCHK results, align frames, project force gradients."""
    from matrix_gaussian.fchk import read_gaussian_fchk_point_result, BOHR_TO_ANGSTROM
    from matrix_chem.topology.elements import atomic_number

    path = Path(path)
    ledger = json.loads(path.read_text())
    contract = ledger["contract"]
    if ledger.get("schema") != "matrix.lock.qc_cycle.v1" or ledger["contract_sha256"] != _digest(
        contract
    ):
        raise ValueError("invalid QC cycle contract")
    expected = np.array([atomic_number(a) for a in contract["atoms"]])
    results = dict(ledger.get("results", {}))
    missing = []
    for index, job in enumerate(ledger["jobs"]):
        if (
            hashlib.sha256((path.parent / job["input"]).read_bytes()).hexdigest()
            != job["input_sha256"]
        ):
            raise ValueError("QC input identity mismatch")
        log = path.parent / job["log"]
        fchk = path.parent / job["fchk"]
        if not log.exists() or not fchk.exists():
            missing.append(job["id"])
            continue
        text = log.read_text(errors="replace")
        normal = text.rfind("Normal termination")
        error = text.rfind("Error termination")
        if normal < 0 or error > normal:
            raise ValueError(f"GDV calculation did not terminate normally: {job['id']}")
        from matrix_gaussian.parsers import read_gaussian_point_result

        parsed_log = read_gaussian_point_result(log, require_gradient=False)
        raw = read_gaussian_fchk_point_result(fchk)
        if not np.array_equal(raw.atomic_numbers, expected) or raw.total_energy_hartree is None:
            raise ValueError("QC atom identities or energy missing")
        if abs(float(raw.total_energy_hartree) - parsed_log.energy_hartree) > 5.0e-8:
            raise ValueError("Gaussian log and FCHK energies differ")
        input_job = (
            contract["reference"] if index == 0 else contract["candidates"][index - 1]["job"]
        )
        target = np.asarray(input_job["geometry_angstrom"])
        xyz = np.asarray(raw.cartesian_coordinates_bohr) * BOHR_TO_ANGSTROM
        left = xyz - xyz.mean(axis=0)
        right = target - target.mean(axis=0)
        u, _, vh = np.linalg.svd(left.T @ right)
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vh)]) @ vh
        residual = np.max(np.linalg.norm(left @ rotation - right, axis=1), initial=0.0)
        if residual > geometry_tolerance_angstrom:
            raise ValueError("QC geometry moved outside the frozen input chart")
        gradient = np.asarray(raw.cartesian_gradient_hartree_per_bohr).reshape(-1, 3) @ rotation
        coordinate_gradient = None
        if index:
            tangent = np.asarray(input_job["cartesian_tangents_angstrom"])
            coordinate_gradient = (
                np.einsum("ac,acd->d", gradient / BOHR_TO_ANGSTROM, tangent) * HARTREE_TO_MHZ
            )
        results[job["id"]] = dict(
            energy_hartree=float(raw.total_energy_hartree),
            coordinate_gradient_MHz=None
            if coordinate_gradient is None
            else coordinate_gradient.tolist(),
            geometry_alignment_error_angstrom=float(residual),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            fchk_sha256=hashlib.sha256(fchk.read_bytes()).hexdigest(),
        )
    ledger["results"] = results
    ledger["missing"] = missing
    ledger["status"] = "collected" if not missing else "partial"
    if not missing:
        reference = results["reference"]["energy_hartree"]
        values = []
        for candidate in contract["candidates"]:
            r = results[str(candidate["id"])]
            values.extend(
                [(r["energy_hartree"] - reference) * HARTREE_TO_MHZ, *r["coordinate_gradient_MHz"]]
            )
        ledger["observations_MHz"] = values
    _write(path, ledger)
    return ledger


def physical_qc_predictions(specification, parameters, candidates, reference, *, _parameter_direction=None):
    """Evaluate the actual represented potential at fixed QC geometries."""
    from .physical_producer import build_physical_lam_producer
    from scipy.interpolate import CubicSpline

    producer = build_physical_lam_producer(specification)

    def point(record, job, gradient=True):
        spec = record["potential"]
        q = np.asarray(job["coordinate"], float)
        xyz = np.asarray(job["geometry_angstrom"])
        t = np.asarray(job.get("cartesian_tangents_angstrom", np.zeros(xyz.shape + (len(q),))))
        value = float(spec.get("constant_MHz", 0.0)) if _parameter_direction is None else 0.0
        g = np.zeros(len(q))
        kind = spec["representation"]
        if kind == "grid" and _parameter_direction is None:

            def interpolate(derivative=None):
                a = np.asarray(spec["values_MHz"], complex)
                for axis, (grid, periodic) in enumerate(
                    zip(record["coordinates"], record["periodic"], strict=True)
                ):
                    grid = np.asarray(grid)
                    order = int(derivative == axis)
                    if periodic:
                        waves = 2 * np.pi * np.fft.fftfreq(len(grid), grid[1] - grid[0])
                        phase = np.exp(1j * waves * (q[axis] - grid[0])) * (1j * waves) ** order
                        a = np.tensordot(phase, np.fft.fft(a, axis=0) / len(grid), axes=(0, 0))
                    else:
                        if q[axis] < grid[0] or q[axis] > grid[-1]:
                            raise ValueError("QC point outside the represented PES")
                        a = CubicSpline(grid, a, axis=0)(q[axis], nu=order)
                if abs(np.imag(a)) > 1.0e-8 * max(1.0, abs(a)):
                    raise ValueError("unresolved QC PES Nyquist interpolation")
                return float(np.real(a))

            value += interpolate()
            g += np.array([interpolate(axis) for axis in range(len(q))])
        elif kind == "trinity_qff" and _parameter_direction is None:
            from math import factorial

            from .physical_producer import qff_normal_coordinates

            normal, dq = qff_normal_coordinates(spec, xyz, t.transpose(2, 0, 1))
            jac = dq.T
            n = len(normal)
            terms = [
                ([2 if j == i else 0 for j in range(n)], 0.5 * w * 29979.2458)
                for i, w in enumerate(spec["frequencies_cm"])
            ]
            for term in spec.get("derivatives_cm", []):
                powers = [term["indices"].count(i) for i in range(n)]
                terms.append(
                    (
                        powers,
                        term["value_cm"] * 29979.2458 / np.prod([factorial(k) for k in powers]),
                    )
                )
            for powers, coefficient in terms:
                value += coefficient * np.prod(normal ** np.array(powers))
                grad = np.zeros(n)
                for i, power in enumerate(powers):
                    if power:
                        reduced = np.array(powers)
                        reduced[i] -= 1
                        grad[i] = power * np.prod(normal**reduced)
                g += coefficient * grad @ jac
        elif kind not in {"terms","grid","trinity_qff"}:
            raise ValueError("unsupported QC potential representation")
        for term in spec.get("terms", []):
            parts = []
            ds = []
            for axis, factor in enumerate(term["factors"]):
                order = int(factor.get("order", 0))
                v = q[axis]
                kind = factor["function"]
                if kind == "constant":
                    f, df = 1.0, 0.0
                elif kind == "power":
                    f, df = v**order, (order * v ** (order - 1) if order else 0.0)
                elif kind == "cos":
                    f, df = np.cos(order * v), -order * np.sin(order * v)
                elif kind == "sin":
                    f, df = np.sin(order * v), order * np.cos(order * v)
                elif kind == "one_minus_cos":
                    f, df = 1 - np.cos(order * v), order * np.sin(order * v)
                else:
                    raise ValueError("unsupported PES factor")
                parts.append(f)
                ds.append(df)
            if _parameter_direction is None:
                coefficient = float(term.get("coefficient_MHz", 0.0))
                if term.get("parameter") is not None:
                    coefficient += term.get("multiplier_MHz", 1.0) * parameters[term["parameter"]]
            else:
                coefficient = (float(term.get("multiplier_MHz",1.0))
                               if term.get("parameter")==_parameter_direction else 0.0)
            value += coefficient * np.prod(parts)
            for axis in range(len(q)):
                g[axis] += coefficient * ds[axis] * np.prod(parts[:axis] + parts[axis + 1 :])
        return value, g

    output = []
    for candidate in candidates:
        job = candidate["job"]
        system = job.get("system")
        if system is None:
            if len(producer.systems) != 1:
                raise ValueError("QC jobs in a multi-system producer require system identities")
            system = next(iter(producer.systems))
        record, _ = producer.systems[system]
        v, g = point(record, job)
        ref, _ = point(record, reference, False)
        output.extend([v - ref, *g])
    return np.asarray(output, float)


def _physical_qc_design(specification, parameter_names, candidates, reference):
    """Exact affine PES E/G coefficients without subtracting large energies."""
    return np.column_stack([
        physical_qc_predictions(specification,{},candidates,reference,_parameter_direction=name)
        for name in parameter_names
    ])


def refine_qc_pes_cycle(path, *, output=None):
    """Update the original QM prior with collected QC E/G observations once.

    All supported producer parameter coefficients are affine at fixed QC
    geometries. Analytic coefficient extraction avoids subtractive cancellation,
    and the accepted update
    is checked by reevaluating the physical potential, not just its design map.
    """
    path = Path(path)
    ledger = json.loads(path.read_text())
    contract = ledger["contract"]
    if ledger["contract_sha256"] != _digest(contract) or ledger.get("status") not in {
        "collected",
        "refined",
    }:
        raise ValueError("QC cycle is incomplete or incompatible")
    for job in ledger["jobs"]:
        result = ledger["results"][job["id"]]
        for key in ("log", "fchk"):
            if (
                hashlib.sha256((path.parent / job[key]).read_bytes()).hexdigest()
                != result[key + "_sha256"]
            ):
                raise ValueError("collected QC result changed before refinement")
    names = contract["parameter_names"]
    x = np.array([contract["qm_prior_values"][name] for name in names])
    cov = np.asarray(contract["qm_prior_covariance"])
    noise = np.asarray(contract["noise_covariance"])
    y = np.asarray(ledger["observations_MHz"])

    def predict(z):
        return physical_qc_predictions(
            contract["producer"],
            dict(zip(names, map(float, z), strict=True)),
            contract["candidates"],
            contract["reference"],
        )

    base = predict(x)
    h = _physical_qc_design(contract["producer"],names,contract["candidates"],contract["reference"])
    cross = cov @ h.T
    factor = cho_factor(noise + h @ cross, lower=True)
    updated = x + cross @ cho_solve(factor, y - base)
    gain = cho_solve(factor, cross.T).T
    residual_map = np.eye(len(x)) - gain @ h
    posterior = residual_map @ cov @ residual_map.T + gain @ noise @ gain.T
    posterior = (posterior + posterior.T) / 2
    np.linalg.cholesky(posterior)
    actual = predict(updated)
    expected = base + h @ (updated - x)
    if not np.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-7):
        raise ValueError("QC PES update is not affine under the declared producer")
    prior = cho_factor(cov, lower=True)
    measurement = cho_factor(noise, lower=True)
    before = float((base - y) @ cho_solve(measurement, base - y))
    after = float((actual - y) @ cho_solve(measurement, actual - y))
    shift = updated - x
    objective = after + float(shift @ cho_solve(prior, shift))
    if objective > before + 1.0e-8 * max(1.0, before) or not np.all(np.isfinite(posterior)):
        raise RuntimeError("QC PES refinement failed objective/covariance verification")
    result = dict(
        schema="matrix.lock.qc_pes_refinement.v1",
        verified=True,
        cycle_contract_sha256=ledger["contract_sha256"],
        qc_results_sha256=_digest(ledger["results"]),
        refinement_id=_digest(dict(contract=ledger["contract_sha256"], results=ledger["results"])),
        qm_source_id=contract["qm_source_id"],
        parameter_names=names,
        qm_prior_values=contract["qm_prior_values"],
        updated_values=dict(zip(names, map(float, updated), strict=True)),
        updated_qm_covariance=posterior.tolist(),
        qc_chi_square_before=before,
        qc_chi_square_after=after,
        total_objective_after=objective,
        maximum_affine_verification_error_MHz=float(np.max(abs(actual - expected), initial=0.0)),
        producer=contract["producer"],
        information_treatment="QC updates original QM information only. Replace that prior block in a spectral refit; never recycle a spectral posterior as independent QM information.",
    )
    target = Path(output) if output is not None else path.parent / "refined_pes.json"
    _write(target, result)
    ledger["status"] = "refined"
    ledger["refinement"] = dict(
        path=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest()
    )
    _write(path, ledger)
    return result


def apply_qc_prior(request, refinement):
    """Replace the original matching QM prior block, without double counting."""
    if not refinement.get("verified"):
        raise ValueError("unverified QC refinement")
    result = deepcopy(request)
    names = refinement["parameter_names"]
    source = refinement["qm_source_id"]
    variables = {v["name"]: v for v in result["variables"]}
    if set(names) - set(variables):
        raise ValueError("QC and spectral parameter registries differ")
    predicates = result.get("predicates", [])
    overlap = [i for i, p in enumerate(predicates) if set(p.get("coefficients", {})) & set(names)]
    if overlap:
        if any(predicates[i].get("source_id") != source for i in overlap):
            raise ValueError("overlapping QM predicates require the matching original QC source_id")
        if any(set(predicates[i].get("coefficients", {})) - set(names) for i in overlap):
            raise ValueError(
                "a coupled QM predicate requires all its physical variables in the QC update"
            )
        removed = overlap
        retained = [i for i in range(len(predicates)) if i not in removed]
        if "predicate_covariance" in result:
            original_covariance = np.asarray(result["predicate_covariance"], float)
            cross = original_covariance[np.ix_(removed, retained)]
            diagonal = np.diag(original_covariance)
            if np.any(diagonal <= 0):
                raise ValueError("QM predicate covariance has nonpositive variances")
            correlation = cross / np.sqrt(diagonal[removed, None] * diagonal[None, retained])
            if np.max(abs(correlation), initial=0.0) > 1.0e-12:
                raise ValueError("correlated QM sources must be updated together in the QC cycle")
            if retained:
                result["predicate_covariance"] = original_covariance[
                    np.ix_(retained, retained)
                ].tolist()
            else:
                result.pop("predicate_covariance", None)
        result["predicates"] = [predicates[i] for i in retained]
    existing = [v["name"] for v in result["variables"] if "prior" in v]
    if set(existing) - set(names):
        raise ValueError("a correlated QC prior cannot silently replace unrelated QM prior blocks")
    order = [v["name"] for v in result["variables"] if v["name"] in names]
    positions = [names.index(n) for n in order]
    cov = np.asarray(refinement["updated_qm_covariance"])[np.ix_(positions, positions)]
    for k, name in enumerate(order):
        variables[name]["prior"] = dict(
            value=refinement["updated_values"][name],
            sigma=float(np.sqrt(cov[k, k])),
            source_id=source,
        )
        variables[name]["initial"] = refinement["updated_values"][name]
    result["prior_covariance"] = cov.tolist()
    result.setdefault("qm_information_sources", {})[source] = refinement["refinement_id"]
    return result
