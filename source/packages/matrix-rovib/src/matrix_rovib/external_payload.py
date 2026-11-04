from __future__ import annotations

"""Compatibility adapters for CeDiTT and alpha-resonance result payloads."""

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path

from .contracts import (
    DeltaBVibAlphaRow,
    DeltaBVibSection,
    read_rotational_section,
    write_deltabvib_section,
    write_rotational_section,
)


EXTERNAL_ROVIB_SCHEMA = "matrix.external.rovib.v1"


@dataclass(frozen=True)
class ExternalRovibPayload:
    source: str
    representation: str = ""
    watson_reduction: str = ""
    rotational_constants_MHz: tuple[float, float, float] | None = None
    quartic_distortion_MHz: tuple[float, ...] = ()
    sextic_distortion_MHz: tuple[float, ...] = ()
    octic_distortion_MHz: tuple[float, ...] = ()
    alpha_rows_MHz: tuple[DeltaBVibAlphaRow, ...] = ()
    excluded_modes: tuple[int, ...] = ()
    schema: str = EXTERNAL_ROVIB_SCHEMA


def read_external_rovib_payload(path: Path | str) -> ExternalRovibPayload:
    target = Path(path)
    if target.suffix.lower() == ".csv":
        return _read_ceditt_csv(target)
    data = json.loads(target.read_text(encoding="utf-8"))
    return _payload_from_json(data, source_path=target)


def promote_external_rovib_payload(
    payload_path: Path | str,
    xyzin: Path | str,
) -> ExternalRovibPayload:
    """Merge external constants into canonical ``#ROTATIONAL/#DELTABVIB`` sections."""
    payload = read_external_rovib_payload(payload_path)
    target = Path(xyzin)
    rotational = read_rotational_section(target)
    abc = payload.rotational_constants_MHz
    updated = replace(
        rotational,
        representation=payload.representation or rotational.representation,
        watson_reduction=payload.watson_reduction or rotational.watson_reduction,
        A_MHz=rotational.A_MHz if abc is None else abc[0],
        B_MHz=rotational.B_MHz if abc is None else abc[1],
        C_MHz=rotational.C_MHz if abc is None else abc[2],
        quartic_distortion_MHz=(
            payload.quartic_distortion_MHz or rotational.quartic_distortion_MHz
        ),
        sextic_distortion_MHz=(
            payload.sextic_distortion_MHz or rotational.sextic_distortion_MHz
        ),
        octic_distortion_MHz=(
            payload.octic_distortion_MHz or rotational.octic_distortion_MHz
        ),
        distortion_source=(
            payload.source
            if (payload.quartic_distortion_MHz or payload.sextic_distortion_MHz
                or payload.octic_distortion_MHz)
            else rotational.distortion_source
        ),
    )
    write_rotational_section(target, updated)
    if payload.alpha_rows_MHz:
        kept = [row for row in payload.alpha_rows_MHz if row.mode not in payload.excluded_modes]
        delta = tuple(
            0.5 * sum(getattr(row, attribute) for row in kept)
            for attribute in ("a_MHz", "b_MHz", "c_MHz")
        )
        write_deltabvib_section(
            target,
            DeltaBVibSection(
                delta_A_MHz=delta[0],
                delta_B_MHz=delta[1],
                delta_C_MHz=delta[2],
                source=payload.source,
                alpha_rows_MHz=payload.alpha_rows_MHz,
                excluded_modes=payload.excluded_modes,
            ),
        )
    return payload


def _payload_from_json(data: dict[str, object], *, source_path: Path) -> ExternalRovibPayload:
    source = str(data.get("source", f"external:{source_path.name}"))
    output = data.get("output", {}) if isinstance(data.get("output"), dict) else {}
    representation = str(output.get("representation", data.get("representation", "")))
    reduction = str(output.get("reduction", data.get("watson_reduction", data.get("reduction", ""))))
    quartic = data.get("quartic", {}) if isinstance(data.get("quartic"), dict) else {}
    sextic = data.get("sextic", {}) if isinstance(data.get("sextic"), dict) else {}
    octic = data.get("octic", {}) if isinstance(data.get("octic"), dict) else {}
    abc_raw = (
        quartic.get("output_abc_mhz") or sextic.get("output_abc_mhz")
        or data.get("abc_mhz") or data.get("rotational_constants_MHz")
    )
    q_raw, q_scale = _constant_vector(quartic, data, "quartic", 5)
    s_raw, s_scale = _constant_vector(sextic, data, "sextic", 7)
    o_raw, o_scale = _constant_vector(octic, data, "octic", 9)
    alpha = data.get("alpha", data)
    rows = _alpha_rows(alpha if isinstance(alpha, dict) else {})
    excluded_raw = alpha.get("excluded_modes", ()) if isinstance(alpha, dict) else ()
    return ExternalRovibPayload(
        source=source,
        representation=representation,
        watson_reduction=reduction,
        rotational_constants_MHz=_triplet(abc_raw),
        quartic_distortion_MHz=tuple(value * q_scale for value in q_raw),
        sextic_distortion_MHz=tuple(value * s_scale for value in s_raw),
        octic_distortion_MHz=tuple(value * o_scale for value in o_raw),
        alpha_rows_MHz=rows,
        excluded_modes=tuple(int(value) for value in excluded_raw),
    )


def _constant_vector(section, root, label, size):
    for key, scale in (
        ("output_constants_mhz", 1.0), ("output_constants_MHz", 1.0),
        ("output_constants_khz", 1.0e-3), ("output_constants_kHz", 1.0e-3),
    ):
        raw = section.get(key)
        if raw is not None:
            values = tuple(float(value) for value in raw)
            if len(values) != size:
                raise ValueError(f"external {label} payload requires {size} constants")
            return values, scale
    raw = root.get(f"{label}_distortion_MHz", ())
    values = tuple(float(value) for value in raw)
    if values and len(values) != size:
        raise ValueError(f"external {label} payload requires {size} constants")
    return values, 1.0


def _alpha_rows(data: dict[str, object]) -> tuple[DeltaBVibAlphaRow, ...]:
    rows = data.get("rows")
    if rows is not None:
        return tuple(
            DeltaBVibAlphaRow(
                int(row["mode"]), float(row.get("a_MHz", row.get("A_MHz"))),
                float(row.get("b_MHz", row.get("B_MHz"))),
                float(row.get("c_MHz", row.get("C_MHz"))),
            )
            for row in rows
        )
    matrix = data.get("alpha_mhz", data.get("alpha_MHz", ()))
    modes = data.get("mode_indices", range(1, len(matrix) + 1))
    return tuple(
        DeltaBVibAlphaRow(int(mode), float(row[0]), float(row[1]), float(row[2]))
        for mode, row in zip(modes, matrix)
    )


def _read_ceditt_csv(path: Path) -> ExternalRovibPayload:
    with path.open(encoding="utf-8", newline="") as handle:
        fields = {row["field"].strip(): row["value"].strip() for row in csv.DictReader(handle)}
    reduction = fields.get("output_red", "")
    names = ("DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK") if reduction.upper() == "A" else ("DJ", "DJK", "DK", "d1", "d2")
    quartic = tuple(float(fields[f"out_{name}"]) * 1.0e-3 for name in names)
    abc = tuple(float(fields[key]) for key in ("A_MHz", "B_MHz", "C_MHz"))
    return ExternalRovibPayload(
        source=f"ceditt:{path.name}", representation=fields.get("output_rep", ""),
        watson_reduction=reduction, rotational_constants_MHz=abc,
        quartic_distortion_MHz=quartic,
    )


def _triplet(raw) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    values = tuple(float(value) for value in raw)
    if len(values) != 3:
        raise ValueError("external rovib ABC payload requires three constants")
    return values
