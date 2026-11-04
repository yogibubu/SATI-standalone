"""Simple assigned-line input and SPFIT compatibility import for LOCK."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .native_rotor import NativeRotorParameters
from .spin_hyperfine import SpinHyperfineModel


SIMPLE_LINE_FIELDS = (
    "frequency_MHz",
    "uncertainty_MHz",
    "lower_N",
    "lower_J",
    "lower_F",
    "lower_Ka",
    "lower_Kc",
    "upper_N",
    "upper_J",
    "upper_F",
    "upper_Ka",
    "upper_Kc",
)


def read_assigned_spectrum(path: Path | str, *, format: str = "auto") -> list[dict[str, Any]]:
    """Read native CSV/TSV, JSON, or a conventional SPFIT ``.lin`` file.

    Native tables use ordinary J and F values (including halves), not doubled
    integers.  For closed-shell spectra N/J/F columns may be omitted in favor
    of a single J column; LOCK expands the missing values automatically.
    """
    source = Path(path)
    selected = format.lower()
    if selected == "auto":
        selected = (
            "spfit"
            if source.suffix.lower() == ".lin"
            else "json"
            if source.suffix.lower() == ".json"
            else "table"
        )
    if selected == "json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("observations", payload.get("lines"))
        if not isinstance(payload, list):
            raise ValueError("LOCK JSON line input requires an observations array")
        return [dict(_mapping(item, "observation")) for item in payload]
    if selected == "spfit":
        return _read_spfit_lin(source)
    if selected != "table":
        raise ValueError("line format must be auto, table, json, or spfit")
    return _read_native_table(source)


def build_simple_fit_request(
    observations: Sequence[Mapping[str, Any]],
    vary: Sequence[str],
    parameters: NativeRotorParameters,
    model: SpinHyperfineModel,
    *,
    loss: str = "linear",
    outlier_threshold: float = 4.0,
    assembly_backend: str = "compiled",
) -> dict[str, Any]:
    """Compile a short list of familiar names into the full fit contract."""
    variables = []
    for raw in vary:
        token = raw.strip()
        if not token:
            continue
        path = simple_parameter_path(token, model)
        value = _parameter_value(path, parameters, model)
        scale = _default_scale(path, value)
        item: dict[str, Any] = {"name": token, "path": path, "scale": scale}
        if path in {"rotor.A_MHz", "rotor.B_MHz", "rotor.C_MHz"}:
            item["lower"] = max(1.0e-9, 0.05 * value)
            item["upper"] = 20.0 * value
        variables.append(item)
    if not variables:
        raise ValueError("select at least one parameter to vary")
    n_values: set[int] = set()
    for observation in observations:
        states = [observation.get("state"), observation.get("lower"), observation.get("upper")]
        for component in observation.get("components", ()):
            if isinstance(component, Mapping):
                states.extend((component.get("lower"), component.get("upper")))
        for state in states:
            if isinstance(state, Mapping) and "N" in state:
                n_values.add(int(state["N"]))
    if not n_values:
        raise ValueError("assigned lines do not contain any N quantum numbers")
    return {
        "schema": "matrix.lock.spectral_fit.v1",
        "n_values": sorted(n_values),
        "variables": variables,
        "observations": [dict(item) for item in observations],
        "options": {
            "loss": loss,
            "outlier_threshold": float(outlier_threshold),
            "assembly_backend": assembly_backend,
        },
    }


def simple_parameter_path(name: str, model: SpinHyperfineModel) -> str:
    """Resolve concise KEYMAKER/CLI parameter names to stable LOCK paths."""
    token = name.strip()
    upper = token.upper()
    rotor = {"A": "A_MHz", "B": "B_MHz", "C": "C_MHz"}
    if upper in rotor:
        return f"rotor.{rotor[upper]}"
    quartic = {key: index for index, key in enumerate(("DJ", "DJK", "DK", "DJ_SMALL", "DK_SMALL"))}
    aliases = {"D1": "DJ_SMALL", "D2": "DK_SMALL", "DELTAJ": "DJ_SMALL", "DELTAK": "DK_SMALL"}
    upper = aliases.get(upper, upper)
    if upper in quartic:
        return f"rotor.quartic_MHz.{quartic[upper]}"
    sextic = {key: index for index, key in enumerate(("HJ", "HJK", "HKJ", "HK", "H1", "H2", "H3"))}
    if upper in sextic:
        return f"rotor.sextic_MHz.{sextic[upper]}"
    octic = {
        key: index for index, key in enumerate(
            ("LJ", "LJJK", "LJK", "LKKJ", "LK", "L1", "L2", "L3", "L4")
        )
    }
    if upper in octic:
        return f"rotor.octic_MHz.{octic[upper]}"
    if upper in {"EPS", "EPSILON", "SPIN_ROTATION"}:
        return "electron.isotropic_spin_rotation_MHz"
    match = re.fullmatch(
        r"(?i)(AF|FERMI|CHI_AA|CHI_BB|CHI_CC|CHI_AB|CHI_AC|CHI_BC|T_AA|T_BB|T_CC|T_AB|T_AC|T_BC):(.+)",
        token,
    )
    if match:
        kind, label = match.group(1).upper(), match.group(2).strip()
        labels = {nucleus.label.casefold(): nucleus.label for nucleus in model.nuclei}
        try:
            canonical = labels[label.casefold()]
        except KeyError as exc:
            raise ValueError(f"unknown nuclear label in parameter {name!r}") from exc
        if kind in {"AF", "FERMI"}:
            return f"nucleus:{canonical}.fermi_contact_MHz"
        component = kind[-2:].lower()
        tensor = (
            "quadrupole_tensor_MHz"
            if kind.startswith("CHI")
            else "electron_nuclear_dipolar_tensor_MHz"
        )
        return f"nucleus:{canonical}.{tensor}.{component}"
    if token.startswith(("rotor.", "electron.", "nucleus:", "pair:")):
        return token
    raise ValueError(f"unknown simple LOCK parameter name: {name}")


def write_native_line_table(path: Path | str, observations: Sequence[Mapping[str, Any]]) -> Path:
    """Write the compact table used by KEYMAKER."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ("label", "dataset", *SIMPLE_LINE_FIELDS, "weight", "include")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for observation in observations:
            if "components" in observation or "state" in observation:
                raise ValueError("the compact table writer supports resolved transition lines")
            row: dict[str, Any] = {
                "label": observation.get("label", ""),
                "dataset": observation.get("dataset", "default"),
                "frequency_MHz": observation["frequency_MHz"],
                "uncertainty_MHz": observation["uncertainty_MHz"],
                "weight": observation.get("weight", 1.0),
                "include": 1,
            }
            for side in ("lower", "upper"):
                state = _mapping(observation[side], side)
                row[f"{side}_N"] = state["N"]
                row[f"{side}_J"] = 0.5 * int(state["two_J"])
                row[f"{side}_F"] = 0.5 * int(state["two_F"])
                row[f"{side}_Ka"] = state["Ka"]
                row[f"{side}_Kc"] = state["Kc"]
            writer.writerow(row)
    return target


def _read_native_table(path: Path) -> list[dict[str, Any]]:
    sample = path.read_text(encoding="utf-8")[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    result = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle, dialect=dialect)):
            normalized = {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            if normalized.get("include", "1").lower() in {"0", "false", "no", "n"}:
                continue
            uncertainty = float(
                normalized.get("uncertainty_MHz")
                or normalized.get("sigma_MHz")
                or normalized.get("sigma")
                or ""
            )
            if (normalized.get("kind") or "transition").lower() in {"term", "term_value", "level"}:
                result.append(
                    {
                        "label": normalized.get("label", f"term-{index + 1}"),
                        "dataset": normalized.get("dataset", "default"),
                        "energy_MHz": float(
                            normalized.get("energy_MHz") or normalized.get("frequency_MHz") or ""
                        ),
                        "uncertainty_MHz": uncertainty,
                        "state": _table_state(normalized, "state"),
                        "weight": float(normalized.get("weight") or 1.0),
                    }
                )
                continue
            result.append(
                {
                    "label": normalized.get("label", f"line-{index + 1}"),
                    "dataset": normalized.get("dataset", "default"),
                    "frequency_MHz": float(normalized["frequency_MHz"]),
                    "uncertainty_MHz": uncertainty,
                    "lower": _table_state(normalized, "lower"),
                    "upper": _table_state(normalized, "upper"),
                    "weight": float(normalized.get("weight") or 1.0),
                }
            )
    if not result:
        raise ValueError(f"no included assigned observations in {path}")
    return result


def _read_spfit_lin(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        content = raw.split("/", 1)[0]
        if not content.strip() or content.lstrip().startswith(("#", "!")):
            continue
        if len(content) < 51:
            raise ValueError(f"short SPFIT line at {path}:{line_number}")
        quanta = [content[index : index + 3].strip() for index in range(0, 36, 3)]
        values = [int(value) for value in quanta if value]
        fields = content[36:].split()
        if len(fields) < 2 or len(values) not in {6, 8}:
            raise ValueError(f"unsupported SPFIT quantum-number layout at {path}:{line_number}")
        frequency, uncertainty = float(fields[0]), float(fields[1])
        weight = float(fields[2].replace("D", "E")) if len(fields) > 2 else 1.0
        if len(values) == 6:
            ju, kau, kcu, jl, kal, kcl = values
            fu, fl = ju, jl
        else:
            ju, kau, kcu, fu, jl, kal, kcl, fl = values
        result.append(
            {
                "label": f"{path.stem}:{line_number}",
                "dataset": path.stem,
                "frequency_MHz": frequency,
                "uncertainty_MHz": uncertainty,
                "weight": weight,
                "upper": _state(ju, 2 * ju, 2 * fu, kau, kcu),
                "lower": _state(jl, 2 * jl, 2 * fl, kal, kcl),
            }
        )
    if not result:
        raise ValueError(f"no usable SPFIT lines in {path}")
    return result


def _table_state(row: Mapping[str, str], prefix: str) -> dict[str, int]:
    def get(name: str) -> str:
        return row.get(f"{prefix}_{name}", "")

    j_text = get("J") or get("N")
    n_text = get("N") or j_text
    f_text = get("F") or j_text
    return _state(
        int(float(n_text)), _twice(j_text), _twice(f_text), int(get("Ka")), int(get("Kc"))
    )


def _state(n: int, two_j: int, two_f: int, ka: int, kc: int) -> dict[str, int]:
    return {"N": int(n), "two_J": int(two_j), "two_F": int(two_f), "Ka": int(ka), "Kc": int(kc)}


def _twice(value: str | int | float) -> int:
    numeric = float(value)
    doubled = round(2.0 * numeric)
    if not math.isclose(2.0 * numeric, doubled, abs_tol=1.0e-8):
        raise ValueError(f"J/F value is not integer or half-integer: {value}")
    return int(doubled)


def _parameter_value(
    path: str, parameters: NativeRotorParameters, model: SpinHyperfineModel
) -> float:
    from .spectral_fit import _get_parameter

    return _get_parameter(parameters, model, path)


def _default_scale(path: str, value: float) -> float:
    if path.startswith("rotor.") and path.count(".") == 1:
        return max(abs(value) * 0.05, 100.0)
    return max(abs(value) * 0.2, 0.1)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


__all__ = [
    "SIMPLE_LINE_FIELDS",
    "build_simple_fit_request",
    "read_assigned_spectrum",
    "simple_parameter_path",
    "write_native_line_table",
]
