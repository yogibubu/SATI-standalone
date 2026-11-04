"""Automatic, reproducible discovery of CALPGM projects for LOCK.

The routines in this module inspect files and operator identifiers.  They do
not contain molecule names or molecule-specific numerical constants.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calpgm_native import calpgm_operator_inventory


_WAVENUMBER_TO_MHZ = 29979.2458


CALPGM_PROJECT_SCHEMA = "matrix.lock.calpgm_project.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _infer_line_layout_rows(rows: Sequence[str], source: str) -> dict[str, int]:
    if not rows:
        raise ValueError(f"no CALPGM assigned lines in {source}")
    valid: list[int] = []
    for per_state in range(1, 7):
        width = 6 * per_state
        accepted = True
        for row in rows:
            if len(row) <= width:
                accepted = False
                break
            quantum = [row[index:index + 3].strip() for index in range(0, width, 3)]
            try:
                if any(not value for value in quantum):
                    raise ValueError
                tuple(int(value) for value in quantum)
                fields = row[width:].split()
                if len(fields) < 2:
                    raise ValueError
                float(fields[0].replace("D", "E"))
                float(fields[1].replace("D", "E"))
            except ValueError:
                accepted = False
                break
        if accepted:
            valid.append(per_state)
    if not valid:
        raise ValueError(f"cannot infer CALPGM quantum-number layout in {source}")
    per_state = max(valid)
    return {"quantum_numbers_per_state": per_state, "prefix_width": 6 * per_state}


def infer_calpgm_line_layout(path: Path | str) -> dict[str, int]:
    """Infer the fixed-width quantum-number prefix of a CALPGM line file.

    Every quantum number occupies three columns.  Candidate layouts are
    checked over all data rows, so a coincidental integer at the beginning of
    a frequency cannot silently select a larger layout.
    """
    source = Path(path)
    rows = [
        raw.partition("/")[0].rstrip("\n")
        for raw in source.read_text(encoding="utf-8", errors="replace").splitlines()
        if raw.strip() and not raw.lstrip().startswith(("#", "!"))
    ]
    return _infer_line_layout_rows(rows, str(source))


def read_calpgm_assigned_lines_auto(
    path: Path | str, *, quantum_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read CALPGM lines after discovering their number of quantum labels."""
    source = Path(path)
    return parse_calpgm_assigned_lines_auto(
        source.read_text(encoding="utf-8"), source_name=str(source),
        quantum_names=quantum_names,
    )


def parse_calpgm_assigned_lines_auto(
    text: str, *, source_name: str = "CALPGM", quantum_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse CALPGM assigned-line text with automatic fixed-width discovery."""
    raw_lines = text.splitlines()
    data_rows = [
        raw.partition("/")[0].rstrip("\n") for raw in raw_lines
        if raw.strip() and not raw.lstrip().startswith(("#", "!"))
    ]
    layout = _infer_line_layout_rows(data_rows, source_name)
    count = layout["quantum_numbers_per_state"]
    names = tuple(quantum_names or (f"q{index}" for index in range(count)))
    if len(names) != count:
        raise ValueError(f"expected {count} quantum names, obtained {len(names)}")
    width = layout["prefix_width"]
    result: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_lines, 1):
        content, _, reference = raw.partition("/")
        if not content.strip() or content.lstrip().startswith(("#", "!")):
            continue
        values = tuple(int(content[index:index + 3]) for index in range(0, width, 3))
        fields = content[width:].split()
        frequency = float(fields[0].replace("D", "E"))
        uncertainty = float(fields[1].replace("D", "E"))
        if uncertainty == 0.0:
            raise ValueError(f"zero line uncertainty at {source_name}:{line_number}")
        wavenumber_input = uncertainty < 0.0
        if wavenumber_input:
            frequency *= _WAVENUMBER_TO_MHZ
            uncertainty *= _WAVENUMBER_TO_MHZ
        result.append({
            "label": f"{Path(source_name).stem}:{line_number}",
            "source_line": line_number,
            "upper": dict(zip(names, values[:count], strict=True)),
            "lower": dict(zip(names, values[count:], strict=True)),
            "frequency_MHz": frequency,
            "uncertainty_MHz": abs(uncertainty),
            "wavenumber_input": wavenumber_input,
            "weight": float(fields[2].replace("D", "E")) if len(fields) > 2 else None,
            "reference": reference.strip(),
            "raw_quantum_numbers": list(values),
        })
    return result


def infer_calpgm_local_channel(model: Mapping[str, Any]) -> dict[str, Any]:
    """Infer the contracted local-channel topology from a parsed PAR model.

    This backend represents a pair of local states coupled by one or more
    reduced-axis operators.  Unsupported topologies are rejected with a
    diagnostic and can be routed to LOCK's general native CALPGM backend.
    """
    inventory = calpgm_operator_inventory(model)
    options = model.get("options", ())
    state_count = abs(int(options[0]["nvib"])) if options else 0
    if state_count != 2:
        raise ValueError(
            f"local-channel backend requires two states, found {state_count}; "
            "use the native CALPGM backend for this topology"
        )
    labels = {
        str(item.get("label", "")).strip().upper()
        for item in model.get("parameters", ())
        if not bool(item.get("ratio_constrained", False))
    }
    if not {"A", "B", "C", "E"}.issubset(labels):
        missing = sorted({"A", "B", "C", "E"} - labels)
        raise ValueError("local-channel model lacks " + ", ".join(missing))
    distortion = 4
    if labels & {"HJ", "HJK", "HKJ", "HK", "H1", "H2", "H3"}:
        distortion = 6
    if labels & {"LJ", "LJJK", "LJK", "LKKJ", "LK", "L1", "L2", "L3", "L4"}:
        distortion = 8
    interactions: dict[tuple[int, int, str], int] = {}
    canonical_ras = {
        "F{axis}": 2,
        "F{axis}K": 4, "F{axis}J": 4, "F2{axis}": 4,
        "F{axis}KK": 6, "F{axis}JK": 6, "F2{axis}J": 6,
        "F{axis}JKK": 8, "F{axis}JJK": 8,
    }
    for label in labels:
        compact = label.replace("_", "")
        for axes in ("AB", "AC", "BC"):
            orders = {
                template.format(axis=axes): order
                for template, order in canonical_ras.items()
            }
            if compact in orders:
                key = (0, 1, axes.lower())
                interactions[key] = max(interactions.get(key, 0), orders[compact])
    return {
        "backend": "contracted-local-channel",
        "state_count": state_count,
        "reduction": str(options[0].get("reduction", "")),
        "distortion_order": distortion,
        "ras_interactions": [
            {"left": left, "right": right, "axes": axes, "order": order}
            for (left, right, axes), order in sorted(interactions.items())
        ],
        "operator_inventory": inventory,
    }


def inspect_calpgm_project(
    parameter_path: Path | str,
    line_path: Path | str | None = None,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a machine-checkable provenance and model-discovery manifest."""
    from .calpgm_adapter import read_calpgm_parameters

    parameter_source = Path(parameter_path)
    model = read_calpgm_parameters(parameter_source)
    files = {"parameters": {"path": str(parameter_source), "sha256": _sha256(parameter_source)}}
    payload: dict[str, Any] = {
        "schema": CALPGM_PROJECT_SCHEMA,
        "files": files,
        "model": {
            "title": model["title"],
            "state_count": abs(int(model["options"][0]["nvib"])),
            "reduction": model["options"][0]["reduction"],
            "operator_inventory": calpgm_operator_inventory(model),
        },
    }
    try:
        payload["model"]["local_channel"] = infer_calpgm_local_channel(model)
    except ValueError as exc:
        payload["model"]["local_channel"] = {"backend": "native-calpgm", "reason": str(exc)}
    if line_path is not None:
        line_source = Path(line_path)
        records = read_calpgm_assigned_lines_auto(line_source)
        files["lines"] = {"path": str(line_source), "sha256": _sha256(line_source)}
        frequencies = {(row["frequency_MHz"], row["uncertainty_MHz"]) for row in records}
        payload["lines"] = {
            **infer_calpgm_line_layout(line_source),
            "transition_count": len(records),
            "frequency_count": len(frequencies),
        }
    expected = dict(expected_sha256 or {})
    for role, digest in expected.items():
        if role not in files:
            raise ValueError(f"checksum declared for absent CALPGM file role {role!r}")
        if files[role]["sha256"].lower() != digest.lower():
            raise ValueError(
                f"CALPGM {role} checksum mismatch: expected {digest}, "
                f"obtained {files[role]['sha256']}"
            )
    payload["verified_checksums"] = sorted(expected)
    return payload


def local_channel_operator_frontier(
    model: Mapping[str, Any], fit_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare an active LOCK fit with every family declared by CALPGM."""
    declared = infer_calpgm_local_channel(model)
    active_distortion = int(fit_result.get("distortion_order", 0))
    selection = fit_result.get("selection", {})
    active_rows = selection.get("ras_interactions", ()) if isinstance(selection, Mapping) else ()
    active_ras = max(
        (int(row.get("order", 0)) for row in active_rows if isinstance(row, Mapping)),
        default=0,
    )
    declared_ras = max(
        (int(row["order"]) for row in declared["ras_interactions"]), default=0,
    )
    complete = active_distortion >= int(declared["distortion_order"]) and active_ras >= declared_ras
    return {
        "schema": "matrix.lock.local_channel_operator_frontier.v1",
        "declared_calpgm": {
            "distortion_order": int(declared["distortion_order"]),
            "ras_order": declared_ras,
        },
        "active_lock": {
            "distortion_order": active_distortion,
            "ras_order": active_ras,
        },
        "declared_inventory_complete": complete,
        "next_systematic_families": (
            [] if not complete else [
                {
                    "family": "diagonal centrifugal distortion",
                    "order": active_distortion + 2,
                    "status": "requires independent implementation and qualification",
                },
                {
                    "family": "rotation-dependent reduced-axis coupling",
                    "order": active_ras + 2,
                    "status": "requires independent implementation and qualification",
                },
            ]
        ),
    }


__all__ = [
    "CALPGM_PROJECT_SCHEMA",
    "infer_calpgm_line_layout",
    "infer_calpgm_local_channel",
    "inspect_calpgm_project",
    "local_channel_operator_frontier",
    "parse_calpgm_assigned_lines_auto",
    "read_calpgm_assigned_lines_auto",
]
