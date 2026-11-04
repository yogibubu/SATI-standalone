"""Typed export of LOCK spectroscopic fits to MORPHEUS observations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


LOCK_MORPHEUS_HANDOFF_REQUEST_SCHEMA = "matrix.lock.morpheus_handoff_request.v1"
LOCK_MORPHEUS_HANDOFF_SCHEMA = "matrix.lock.morpheus_handoff.v1"


def build_morpheus_handoff(
    request: Mapping[str, object], *, base_directory: Path | str = "."
) -> dict[str, object]:
    """Convert independent LOCK isotopologue fits into MORPHEUS observations.

    The complete A/B/C covariance propagated from the LOCK fit variables is
    retained in ``lock_provenance`` and supplied to MORPHEUS for correlated
    generalized least squares.
    """

    if request.get("schema") != LOCK_MORPHEUS_HANDOFF_REQUEST_SCHEMA:
        raise ValueError("unsupported LOCK-to-MORPHEUS handoff request schema")
    if request.get('joint_fit_result'):
        target=Path(str(request['joint_fit_result']))
        if not target.is_absolute():target=Path(base_directory)/target
        return build_lam_morpheus_handoff(json.loads(target.read_text()),request)
    raw = request.get("isotopologues")
    if not isinstance(raw, list) or not raw:
        raise ValueError("LOCK-to-MORPHEUS handoff requires isotopologues")
    root = Path(base_directory).expanduser().resolve()
    observations = []
    provenance = []
    labels: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"isotopologue {index} must be an object")
        label = str(entry.get("label", "")).strip()
        if not label or label in labels:
            raise ValueError("LOCK-to-MORPHEUS isotopologue labels must be nonempty and unique")
        labels.add(label)
        fit_path = Path(str(entry.get("fit_result", ""))).expanduser()
        if not fit_path.is_absolute():
            fit_path = root / fit_path
        if not fit_path.is_file():
            raise FileNotFoundError(f"LOCK fit result not found: {fit_path}")
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        constants, fitted_covariance, covariance_note = _rotational_fit_projection(fit)
        override = entry.get("sigma_MHz")
        if override is None:
            if fitted_covariance is None:
                raise ValueError(
                    f"LOCK fit {label!r} has no usable A/B/C covariance; "
                    "supply sigma_MHz explicitly"
                )
            diagonal = np.diag(fitted_covariance)
            if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
                raise ValueError(
                    f"LOCK fit {label!r} does not determine positive A/B/C uncertainties; "
                    "supply sigma_MHz explicitly"
                )
            sigmas = np.sqrt(diagonal)
            uncertainty_source = "LOCK propagated covariance"
            input_covariance = fitted_covariance
        else:
            sigmas = _axis_values(override, field=f"{label}.sigma_MHz", positive=True)
            uncertainty_source = "explicit handoff override"
            input_covariance = np.diag(sigmas * sigmas)
        raw_correction = entry.get("vibrational_correction")
        if not isinstance(raw_correction, Mapping):
            raise ValueError(
                f"isotopologue {label!r} requires an explicit vibrational_correction object"
            )
        if not str(raw_correction.get("source", "")).strip():
            raise ValueError(
                f"isotopologue {label!r} requires a vibrational-correction source"
            )
        correction = _correction(raw_correction, "unspecified")
        electronic = _correction(entry.get("electronic_correction"), "unspecified")
        substitutions = entry.get("substitutions", {})
        if not isinstance(substitutions, (dict, list, str)):
            raise ValueError(f"invalid substitutions for isotopologue {label!r}")
        observations.append({
            "label": label,
            "constants": dict(zip(("A_MHz", "B_MHz", "C_MHz"), constants, strict=True)),
            "substitutions": substitutions,
            "vibrational_correction": correction,
            "electronic_correction": electronic,
            "sigma_MHz": dict(zip(("A_MHz", "B_MHz", "C_MHz"), sigmas, strict=True)),
            "covariance_MHz2": input_covariance.tolist(),
        })
        provenance.append({
            "label": label,
            "lock_fit_result": str(fit_path.resolve()),
            "lock_fit_schema": fit.get("schema"),
            "lock_fit_status": fit.get("status"),
            "lock_rotational_covariance_MHz2": (
                None if fitted_covariance is None else fitted_covariance.tolist()
            ),
            "morpheus_input_covariance_MHz2": input_covariance.tolist(),
            "covariance_note": covariance_note,
            "uncertainty_source": uncertainty_source,
            "lock_correlations_preserved_in_handoff": True,
            "correlations_used_by_morpheus": override is None,
        })
    return {
        "schema": LOCK_MORPHEUS_HANDOFF_SCHEMA,
        "isotopologues": observations,
        "lock_provenance": provenance,
        "morpheus_weighting": "correlated generalized least squares by isotopologue",
    }


def write_morpheus_handoff(path: Path | str, payload: Mapping[str, object]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _rotational_fit_projection(
    payload: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if payload.get("schema") != "matrix.lock.spectral_fit_result.v1":
        raise ValueError("MORPHEUS handoff requires a native LOCK spectral-fit result")
    if payload.get("status") not in {"converged", "rank_deficient"}:
        raise ValueError("LOCK spectral fit did not converge")
    rotor = payload.get("fitted_rotor")
    variables = payload.get("variables")
    raw_covariance = payload.get("covariance")
    if not isinstance(rotor, Mapping) or not isinstance(variables, list):
        raise ValueError("LOCK fit result lacks fitted rotor or variable metadata")
    constants = np.asarray([float(rotor[f"{axis}_MHz"]) for axis in "ABC"], dtype=float)
    if np.any(~np.isfinite(constants)) or np.any(constants <= 0.0):
        raise ValueError("LOCK fitted A/B/C constants must be positive and finite")
    try:
        covariance = np.asarray(raw_covariance, dtype=float)
    except (TypeError, ValueError):
        covariance = np.empty((0, 0), dtype=float)
    if covariance.shape != (len(variables), len(variables)) or not np.all(np.isfinite(covariance)):
        return constants, None, "LOCK fit variable covariance is missing or malformed"
    projection = np.zeros((3, len(variables)), dtype=float)
    paths = {f"rotor.{axis}_MHz": row for row, axis in enumerate("ABC")}
    for column, variable in enumerate(variables):
        if not isinstance(variable, Mapping):
            raise ValueError("LOCK fit variable metadata is malformed")
        targets = variable.get("targets", ())
        if not isinstance(targets, list):
            raise ValueError("LOCK fit variable targets are malformed")
        for target in targets:
            if not isinstance(target, Mapping):
                continue
            row = paths.get(str(target.get("path", "")))
            if row is not None:
                projection[row, column] += float(target.get("factor", 1.0))
    projected = projection @ covariance @ projection.T
    projected = 0.5 * (projected + projected.T)
    return constants, projected, str(payload.get("covariance_note", ""))


def _axis_values(value: object, *, field: str, positive: bool = False) -> np.ndarray:
    if isinstance(value, Mapping):
        result = np.asarray([value.get(f"{axis}_MHz") for axis in "ABC"], dtype=float)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = np.asarray(value, dtype=float)
    else:
        raise ValueError(f"{field} requires A/B/C values")
    if result.shape != (3,) or np.any(~np.isfinite(result)) or (positive and np.any(result <= 0.0)):
        raise ValueError(f"{field} requires three {'positive ' if positive else ''}finite values")
    return result


def _correction(value: object, default_source: str) -> dict[str, object]:
    if value is None:
        return {
            "delta_A_MHz": 0.0,
            "delta_B_MHz": 0.0,
            "delta_C_MHz": 0.0,
            "source": default_source,
            "convention": "subtract",
        }
    if not isinstance(value, Mapping):
        raise ValueError("MORPHEUS corrections must be objects")
    result = {
        f"delta_{axis}_MHz": float(value.get(f"delta_{axis}_MHz", 0.0))
        for axis in "ABC"
    }
    if not all(math.isfinite(number) for number in result.values()):
        raise ValueError("MORPHEUS corrections must be finite")
    result["source"] = str(value.get("source", default_source))
    result["convention"] = str(value.get("convention", "subtract"))
    if not str(result["source"]).strip():
        raise ValueError("MORPHEUS correction source must be nonempty")
    if result["convention"] not in {"add", "subtract"}:
        raise ValueError("MORPHEUS correction convention must be add or subtract")
    return result


__all__ = [
    "LOCK_MORPHEUS_HANDOFF_REQUEST_SCHEMA",
    "LOCK_MORPHEUS_HANDOFF_SCHEMA",
    "build_morpheus_handoff",
    "write_morpheus_handoff",
]


def build_lam_morpheus_handoff(fit,request):
    """Export defined LAM-band observables with full isotope covariance."""
    if not fit.get('success') or fit.get('rotational_observables') is None:
        raise ValueError('a converged native fit with physical rotational projections is required')
    projection=fit['rotational_observables'];records=projection['records']
    systems=[r['system'] for r in records];entries=request['isotopologues']
    labels=[str(e.get('label',e['system'])) for e in entries]
    if len(set(labels))!=len(labels):raise ValueError('duplicate isotope label')
    order=[];observations=[]
    for entry,label in zip(entries,labels,strict=True):
        system=entry['system']
        if system not in systems:raise ValueError('missing physical rotational projection')
        index=systems.index(system);record=records[index];order.extend(range(3*index,3*index+3))
        correction=entry.get('vibrational_correction')
        if not isinstance(correction,Mapping) or not str(correction.get('source','')).strip():
            raise ValueError('an explicit sourced vibrational correction is required')
        observations.append(dict(label=label,constants=dict(zip(('A_MHz','B_MHz','C_MHz'),record['constants_MHz'],strict=True)),
            substitutions=entry.get('substitutions',{}),vibrational_correction=_correction(correction,'unspecified'),
            electronic_correction=_correction(entry.get('electronic_correction'),'unspecified')))
    covariance=np.asarray(projection['covariance_MHz2'])[np.ix_(order,order)]
    if request.get('correction_covariance_MHz2') is not None:
        correction=np.asarray(request['correction_covariance_MHz2'],float)
        if correction.shape!=covariance.shape or not np.all(np.isfinite(correction)) or not np.allclose(correction,correction.T):
            raise ValueError('invalid joint correction covariance')
        if np.min(np.linalg.eigvalsh(correction))<-1.e-12*max(1.,np.max(abs(correction))):raise ValueError('correction covariance is not positive semidefinite')
        covariance=covariance+correction
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance)<=0):
        raise ValueError('physical observables lack determined uncertainties; provide independent correction covariance where justified')
    eigen=np.linalg.eigvalsh(covariance);threshold=128*np.finfo(float).eps*len(eigen)*max(np.max(abs(eigen)),np.finfo(float).tiny)
    if np.min(eigen)<=threshold:
        raise ValueError('joint rotational covariance is singular; add justified correction uncertainty or use identifiable structural contrasts')
    for k,observation in enumerate(observations):
        block=covariance[3*k:3*k+3,3*k:3*k+3]
        observation['covariance_MHz2']=block.tolist()
        observation['sigma_MHz']=dict(zip(('A_MHz','B_MHz','C_MHz'),np.sqrt(np.diag(block)),strict=True))
    return dict(schema=LOCK_MORPHEUS_HANDOFF_SCHEMA,isotopologues=observations,
        joint_covariance_MHz2=covariance.tolist(),joint_covariance_labels=labels,
        morpheus_weighting='joint correlated generalized least squares across isotopologues',
        lock_provenance=[dict(label=label,lock_fit_result=request.get('joint_fit_result','in-memory joint native fit'),
            lock_rotational_covariance_MHz2=np.asarray(projection['covariance_MHz2'])[np.ix_(order[3*k:3*k+3],order[3*k:3*k+3])].tolist(),
            morpheus_input_covariance_MHz2=covariance[3*k:3*k+3,3*k:3*k+3].tolist(),uncertainty_source='joint LOCK propagated covariance',
            definition='isolated_band_second_order_rotational_tensor',shared_fit_covariance_preserved=True,
            independent_correction_covariance_added=request.get('correction_covariance_MHz2') is not None) for k,label in enumerate(labels)])
