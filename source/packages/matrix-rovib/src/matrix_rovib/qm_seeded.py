"""Reproducible QM-seeded LOCK refinement requests and blind data splits."""

from __future__ import annotations

import hashlib
import json
import math
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Sequence


QM_SEED_SCHEMA = "matrix.lock.qm_seed.v1"
QM_SEEDED_REFINEMENT_SCHEMA = "matrix.lock.qm_seeded_refinement.v1"


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_qm_seed(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != QM_SEED_SCHEMA:
        raise ValueError(f"unsupported QM seed schema: {payload.get('schema')!r}")
    validate_qm_seed(payload)
    return payload


def validate_qm_seed(payload: Mapping[str, Any]) -> None:
    """Validate constants, surfaces, tensors and their theoretical errors."""

    if payload.get("schema") != QM_SEED_SCHEMA:
        raise ValueError(f"unsupported QM seed schema: {payload.get('schema')!r}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("a QM seed requires a nonempty parameter array")
    paths = []
    for item in parameters:
        if not isinstance(item, Mapping):
            raise ValueError("QM seed parameters must be objects")
        path = str(item.get("path", "")).strip()
        value = float(item["value"])
        sigma = float(item["sigma"])
        if not path or not math.isfinite(value) or not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("QM seed parameters require a path, finite value and positive sigma")
        lower = float(item.get("lower", -math.inf))
        upper = float(item.get("upper", math.inf))
        if math.isnan(lower) or math.isnan(upper) or lower >= upper or not lower <= value <= upper:
            raise ValueError(f"invalid QM bounds for {path}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("QM seed parameter paths must be unique")
    covariance = payload.get("covariance")
    if covariance is not None:
        import numpy as np

        matrix = np.asarray(covariance, dtype=float)
        if matrix.shape != (len(parameters), len(parameters)) or not np.allclose(matrix, matrix.T):
            raise ValueError("QM seed covariance must be a symmetric parameter square matrix")
        if float(np.min(np.linalg.eigvalsh(matrix))) <= 0.0:
            raise ValueError("QM seed covariance must be positive definite")
    _expanded_qm_predicates(payload, require_fitted_match=False)


def _expanded_qm_predicates(
    payload: Mapping[str, Any], *, require_fitted_match: bool
) -> list[dict[str, Any]]:
    """Expand MORPHEUS-style soft QM predicates onto exact LOCK paths."""

    parameters = payload.get("parameters", ())
    answer: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("predicates", ())):
        if not isinstance(raw, Mapping):
            raise ValueError("QM predicates must be objects")
        pattern = str(raw.get("path_pattern", "")).strip()
        sigma = float(raw.get("sigma", 0.0))
        if not pattern or not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("QM predicates require a path_pattern and positive finite sigma")
        center = raw.get("value")
        if center is not None and not math.isfinite(float(center)):
            raise ValueError("QM predicate values must be finite")
        matches = [
            item
            for item in parameters
            if fnmatchcase(str(item["path"]), pattern)
            or fnmatchcase(str(item.get("name", item["path"])), pattern)
        ]
        fitted = [item for item in matches if bool(item.get("fit", True))]
        required = bool(raw.get("required", True))
        if required and not matches:
            raise ValueError(f"QM predicate {pattern!r} matches no input parameter")
        if require_fitted_match and required and not fitted:
            raise ValueError(f"QM predicate {pattern!r} matches no fitted parameter")
        for item in fitted if require_fitted_match else matches:
            answer.append(
                {
                    "name": str(raw.get("name", f"qm-predicate-{index + 1}")),
                    "path": str(item["path"]),
                    "value": float(item["value"] if center is None else center),
                    "sigma": sigma,
                    "source": str(raw.get("source", "qm")),
                    "path_pattern": pattern,
                }
            )
    return answer


def deterministic_blind_split(
    observations: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.2,
    salt: str = "LOCK-QM-v1",
) -> list[dict[str, Any]]:
    """Assign stable train/validation roles without using measured frequencies."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    rows = []
    for index, observation in enumerate(observations):
        item = dict(observation)
        label = str(item.get("label", f"line-{index + 1}"))
        if "role" not in item:
            digest = hashlib.sha256(f"{salt}\0{label}".encode("utf-8")).digest()
            draw = int.from_bytes(digest[:8], "big") / 2**64
            item["role"] = "validation" if draw < validation_fraction else "training"
        rows.append(item)
    if rows and not any(row["role"] == "training" for row in rows):
        rows[0]["role"] = "training"
    return rows


def build_qm_seeded_lam_request(
    seed: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    lam_states: int = 24,
    validation_fraction: float = 0.2,
    workers: int = 1,
) -> dict[str, Any]:
    """Translate ab-initio surface/tensor estimates into a LOCK LAM fit."""

    validate_qm_seed(seed)
    variables = []
    fixed = []
    for index, item in enumerate(seed["parameters"]):
        row = {
            "name": str(item.get("name", item["path"])),
            "path": str(item["path"]),
            "initial": float(item["value"]),
            "scale": float(item.get("scale", max(abs(float(item["value"])), float(item["sigma"])))),
            "multiplier": float(item.get("multiplier", 1.0)),
            "qm_index": index,
        }
        if bool(item.get("use_as_prior", True)):
            row["prior"] = {"value": float(item["value"]), "sigma": float(item["sigma"])}
        if "lower" in item:
            row["lower"] = float(item["lower"])
        if "upper" in item:
            row["upper"] = float(item["upper"])
        if bool(item.get("fit", True)):
            variables.append(row)
        else:
            fixed.append(row)
    if not variables:
        raise ValueError("QM-seeded refinement requires at least one fitted parameter")
    return {
        "schema": "matrix.lock.lam_spectral_fit.v1",
        "workflow_schema": QM_SEEDED_REFINEMENT_SCHEMA,
        "qm_seed_sha256": _canonical_digest(seed),
        "qm_method": seed.get("method"),
        "qm_provenance": seed.get("provenance", {}),
        "variables": variables,
        "fixed_qm_parameters": fixed,
        "predicates": _expanded_qm_predicates(seed, require_fitted_match=True),
        "observations": deterministic_blind_split(
            observations,
            validation_fraction=validation_fraction,
            salt=_canonical_digest(seed),
        ),
        "lam_states": int(lam_states),
        "options": {
            "jacobian_scheme": "analytic",
            "tracking": "overlap",
            "loss": "soft_l1",
            "workers": int(workers),
            "jacobian_workers": int(workers),
            "block_workers": 1,
        },
    }


def qm_seeded_ram36_priors(
    seed: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, float]], list[list[float]] | None]:
    """Return RAM36 prior values/errors and optional correlated covariance."""

    validate_qm_seed(seed)
    priors = {
        str(item["path"]): (float(item["value"]), float(item["sigma"]))
        for item in seed["parameters"]
        if bool(item.get("fit", True)) and bool(item.get("use_as_prior", True))
    }
    covariance = seed.get("covariance")
    if covariance is None:
        return priors, None
    fitted = [
        index
        for index, item in enumerate(seed["parameters"])
        if bool(item.get("fit", True)) and bool(item.get("use_as_prior", True))
    ]
    if not fitted:
        return priors, None
    subset = [[float(covariance[row][column]) for column in fitted] for row in fitted]
    return priors, subset


def write_qm_seeded_request(path: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "QM_SEED_SCHEMA",
    "QM_SEEDED_REFINEMENT_SCHEMA",
    "build_qm_seeded_lam_request",
    "deterministic_blind_split",
    "qm_seeded_ram36_priors",
    "read_qm_seed",
    "validate_qm_seed",
    "write_qm_seeded_request",
]
