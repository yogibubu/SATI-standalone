"""Versioned CALPGM/SPFIT interoperability for assigned radical spectra.

This adapter preserves CALPGM quantum numbers that cannot be reduced to the
ordinary ``N, J, F`` LOCK selector, including the aggregate spin index ``nn``.
It also provides a reproducible external-engine qualification route while the
same operators are implemented in the native variational Hamiltonian.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


_WAVENUMBER_TO_MHZ = 29979.2458


CALPGM_MODEL_SCHEMA = "matrix.lock.calpgm_model.v1"
CALPGM_FIT_RESULT_SCHEMA = "matrix.lock.calpgm_fit_result.v1"
CALPGM_AGGREGATE_MAP_SCHEMA = "matrix.lock.calpgm_aggregate_map.v1"


def infer_calpgm_scalar_aggregate_map(model: Mapping[str, Any]) -> dict[str, Any]:
    """Construct GETQQ symmetry blocks for a spin-scalar CALPGM model.

    The option cards contain all information required in the absence of a
    nontrivial spin-coupling tree: K limits, signed-l pairs, Wang weights and
    molecular symmetry.  Building this map internally prevents l-doublets
    from being treated as independent full signed-K product spaces.
    """

    from .calpgm_native import (
        calpgm_spin_twice_from_model,
        calpgm_vibrational_l_values,
    )

    spin_twice = calpgm_spin_twice_from_model(model)
    if any(spin_twice):
        raise ValueError("automatic aggregate inference is limited to scalar spin")
    options = tuple(model.get("options", ()))
    if not options:
        raise ValueError("CALPGM aggregate inference requires option cards")
    state_count = max(1, abs(int(options[0]["nvib"])))
    l_values = calpgm_vibrational_l_values(options, state_count)
    by_vibration = {0: options[0]}
    for index, option in enumerate(options[1:], 1):
        vibration = abs(int(option.get("nvib", index)))
        if vibration < state_count:
            by_vibration[vibration] = option
    normalized_options = []
    for vibration in range(state_count):
        option = by_vibration.get(vibration, options[0])
        normalized_options.append({
            "vibration": vibration,
            "k_min": int(option.get("k_min", 0)),
            "k_max": int(option.get("k_max", 99)),
            "weight_positive": int(option.get("weight_positive", 0)),
            "weight_negative": int(option.get("weight_negative", 0)),
            "energy_symmetry_weight": int(option.get("e_state_weight", 0)),
            "nuclear_symmetry_order": abs(int(option.get("symmetry_axis", 0))),
            "spin_quantum_numbers": [],
        })

    entries = []
    block = 0
    consumed: set[int] = set()
    for vibration in range(state_count):
        if vibration in consumed:
            continue
        partner = None
        if (
            l_values[vibration]
            and vibration + 1 < state_count
            and l_values[vibration + 1] == -l_values[vibration]
        ):
            partner = vibration + 1
            consumed.add(partner)
        family = (vibration,) if partner is None else (vibration, partner)
        consumed.add(vibration)
        for symbols, weight_name in (
            (("x", "b"), "weight_positive"),
            (("c", "a"), "weight_negative"),
        ):
            weights = [
                int(normalized_options[state][weight_name]) for state in family
            ]
            if not any(weights):
                continue
            block += 1
            for state, weight in zip(family, weights, strict=True):
                if weight <= 0:
                    continue
                for symbol in symbols:
                    entries.append({
                        "block": block,
                        "weight": weight,
                        "wang_symmetry": symbol,
                        "vibration": state,
                        "nn": 0,
                        "offsets_relative_to_F": [0.0],
                        "N_minus_F": 0.0,
                        "intermediate_angular_momenta_minus_F": [],
                        "nuclear_permutation_symmetry": 0,
                        "I_tot": 0.0,
                    })
        present = {
            item["wang_symmetry"] for item in entries
            if item["vibration"] == vibration
        }
        for symbol in ("x", "c", "b", "a"):
            if symbol in present:
                continue
            entries.append({
                "block": 0,
                "weight": 0,
                "wang_symmetry": symbol,
                "vibration": vibration,
                "nn": 0,
                "offsets_relative_to_F": [0.0],
                "N_minus_F": 0.0,
                "intermediate_angular_momenta_minus_F": [],
                "nuclear_permutation_symmetry": 0,
                "I_tot": 0.0,
            })
    return {
        "schema": CALPGM_AGGREGATE_MAP_SCHEMA,
        "source": "CALPGM option-card inference",
        "itot_basis": False,
        "vibrational_options": normalized_options,
        "entries": entries,
        "unique_state_count": len({
            (item["vibration"], item["nn"]) for item in entries
        }),
    }


def read_calpgm_parameters(path: Path | str) -> dict[str, Any]:
    """Read the reproducibility-critical fields of an SPFIT ``.par`` file."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError(f"short CALPGM parameter file: {source}")
    control = lines[1].split()
    if len(control) < 3:
        raise ValueError("CALPGM control line requires NPAR, NLINE, and NITR")
    options = []
    parameters = []
    for raw in lines[2:]:
        content, _, label = raw.partition("/")
        fields = content.split()
        if not fields:
            continue
        first = fields[0]
        if first[:1].isalpha():
            fields = fields[1:]
            if len(fields) < 10:
                raise ValueError(f"invalid CALPGM option line: {raw}")
            options.append(_option_payload(first[0], fields))
            continue
        if len(fields) >= 10 and not parameters:
            # Additional vibronic option lines omit the leading character.
            try:
                int(fields[0])
                int(fields[1])
                int(fields[2])
                if len(options) and len(options) < abs(options[0]["nvib"]):
                    options.append(_option_payload(options[0]["reduction"], fields))
                    continue
            except ValueError:
                pass
        if len(fields) < 3:
            raise ValueError(f"invalid CALPGM parameter line: {raw}")
        identifier = int(fields[0])
        parameters.append({
            "identifier": identifier,
            "value": float(fields[1].replace("D", "E")),
            "uncertainty_or_constraint": float(fields[2].replace("D", "E")),
            "label": label.strip() or str(abs(identifier)),
            "ratio_constrained": identifier < 0,
        })
    if not parameters:
        raise ValueError("CALPGM parameter file contains no Hamiltonian parameters")
    return {
        "schema": CALPGM_MODEL_SCHEMA,
        "title": lines[0].strip(),
        "control": {
            "parameter_count": int(control[0]),
            "line_count": abs(int(control[1])),
            "iterations": abs(int(control[2])),
        },
        "options": options,
        "parameters": parameters,
        "sha256": _sha256(source),
    }


def read_calpgm_assigned_lines(
    path: Path | str,
    *,
    quantum_names: Sequence[str] = ("N", "K", "v", "nn", "F"),
) -> list[dict[str, Any]]:
    """Read fixed-width SPFIT lines without discarding aggregate spin labels."""
    source = Path(path)
    names = tuple(str(value) for value in quantum_names)
    if not names or 2 * len(names) > 12:
        raise ValueError("CALPGM supports one to six printed quantum numbers per state")
    result = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        content, _, reference = raw.partition("/")
        if not content.strip() or content.lstrip().startswith(("#", "!")):
            continue
        if len(content) < 38:
            raise ValueError(f"short CALPGM line at {source}:{line_number}")
        printed = [content[index:index + 3].strip() for index in range(0, 36, 3)]
        count = 2 * len(names)
        missing = [index for index, value in enumerate(printed[:count]) if not value]
        if missing and not ("0.0" in content[36:] and missing == list(range(missing[0], count))):
            raise ValueError(f"missing CALPGM quantum number at {source}:{line_number}")
        values = tuple(int(value) if value else None for value in printed[:count])
        fields = content[36:].split()
        if len(fields) < 2:
            raise ValueError(f"missing CALPGM frequency/error at {source}:{line_number}")
        frequency = float(fields[0].replace("D", "E"))
        uncertainty = float(fields[1].replace("D", "E"))
        if uncertainty == 0.0 or not math.isfinite(frequency) or not math.isfinite(uncertainty):
            raise ValueError(f"invalid CALPGM frequency/error at {source}:{line_number}")
        wavenumber_input = uncertainty < 0.0
        if wavenumber_input:
            frequency *= _WAVENUMBER_TO_MHZ
            uncertainty *= _WAVENUMBER_TO_MHZ
        result.append({
            "label": f"{source.stem}:{line_number}",
            "upper": dict(zip(names, values[:len(names)], strict=True)),
            "lower": dict(zip(names, values[len(names):], strict=True)),
            "frequency_MHz": frequency,
            "uncertainty_MHz": abs(uncertainty),
            "wavenumber_input": wavenumber_input,
            "weight": float(fields[2].replace("D", "E")) if len(fields) > 2 else 1.0,
            "reference": reference.strip(),
            "raw_quantum_numbers": list(values),
        })
    if not result:
        raise ValueError(f"no CALPGM assigned lines in {source}")
    return result


def parse_calpgm_fit(path: Path | str) -> dict[str, Any]:
    """Extract final parameters and statistically relevant SPFIT metrics."""
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    rms = [float(value) for value in re.findall(
        r"MICROWAVE RMS\s*=\s*([-+0-9.Ee]+)\s*MHz", text
    )]
    normalized = [tuple(map(float, values)) for values in re.findall(
        r"OLD, NEW RMS ERROR=\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", text
    )]
    if not rms or not normalized:
        raise ValueError(f"SPFIT convergence metrics are absent from {source}")
    parameters: dict[str, dict[str, float | int | str]] = {}
    pattern = re.compile(
        r"^\s*\d+\s+\d+\s+(-?\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+(\S+)\s*$"
    )
    for raw in text.splitlines():
        match = pattern.match(raw)
        if match:
            identifier, value, uncertainty, label = match.groups()
            parameters[label] = {
                "identifier": int(identifier),
                "value": float(value),
                "reported_uncertainty": float(uncertainty),
            }
    return {
        "rms_residual_MHz": rms[-1],
        "normalized_rms_before_update": normalized[-1][0],
        "normalized_rms_after_update": normalized[-1][1],
        "parameters": parameters,
    }


def read_calpgm_aggregate_map(path: Path | str) -> dict[str, Any]:
    """Read the authoritative aggregate-spin state map printed by CALPGM.

    CALPGM prints angular momenta as offsets relative to ``F=0``.  The first
    value is always ``N-F``.  With an ``I(TOT)`` basis the final value is
    ``I_tot`` and the preceding value is the permutation-symmetry code.  All
    intermediate offsets are retained so the representation also covers
    coupling chains longer than the CH3O ``N,J,I_tot,F`` chain.
    """

    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    marker = "BLOCK - WT - SYM - V - TSP - N  -  other quanta"
    try:
        start = next(index for index, raw in enumerate(lines) if marker in raw) + 1
    except StopIteration as exc:
        raise ValueError(f"aggregate-spin state table is absent from {source}") from exc

    option_rows: list[dict[str, Any]] = []
    itot_basis = any("I(TOT) IS LAST QUANTUM BEFORE F" in raw for raw in lines[:start])
    option_header = next(
        (index for index, raw in enumerate(lines[:start]) if "V KMIN KMAX WTPL WTMN ESYMWT NSYM SPINS" in raw),
        None,
    )
    if option_header is not None:
        for raw in lines[option_header + 1:start - 1]:
            fields = raw.split()
            if not fields or not _is_int(fields[0]) or len(fields) < 8:
                continue
            try:
                option_rows.append({
                    "vibration": int(fields[0]),
                    "k_min": int(fields[1]),
                    "k_max": int(fields[2]),
                    "weight_positive": int(fields[3]),
                    "weight_negative": int(fields[4]),
                    "energy_symmetry_weight": int(fields[5]),
                    "nuclear_symmetry_order": int(fields[6]),
                    "spin_quantum_numbers": [float(value) for value in fields[7:]],
                })
            except ValueError:
                continue

    entries: list[dict[str, Any]] = []
    for raw in lines[start:]:
        if raw.lstrip().startswith("Maximum Dimension for Hamiltonian"):
            break
        fields = raw.split()
        if len(fields) < 7 or not all(_is_int(value) for value in fields[:2]):
            continue
        if not _is_int(fields[3]) or not _is_int(fields[4]):
            continue
        try:
            offsets = [float(value) for value in fields[5:]]
        except ValueError:
            continue
        if not offsets:
            continue
        entry: dict[str, Any] = {
            "block": int(fields[0]),
            "weight": int(fields[1]),
            "wang_symmetry": fields[2],
            "vibration": int(fields[3]),
            "nn": int(fields[4]),
            "offsets_relative_to_F": offsets,
            "N_minus_F": offsets[0],
        }
        if itot_basis:
            if len(offsets) < 3:
                raise ValueError(f"short I(TOT) aggregate row in {source}: {raw}")
            entry.update({
                "intermediate_angular_momenta_minus_F": offsets[1:-2],
                "nuclear_permutation_symmetry": int(round(offsets[-2])),
                "I_tot": offsets[-1],
            })
        else:
            entry["intermediate_angular_momenta_minus_F"] = offsets[1:]
        entries.append(entry)
    if not entries:
        raise ValueError(f"aggregate-spin state rows are absent from {source}")

    signatures: dict[tuple[int, int], tuple[float, ...]] = {}
    for entry in entries:
        key = entry["vibration"], entry["nn"]
        signature = tuple(entry["offsets_relative_to_F"])
        previous = signatures.setdefault(key, signature)
        if previous != signature:
            raise ValueError(
                f"inconsistent aggregate mapping for v={key[0]}, nn={key[1]} in {source}"
            )
    return {
        "schema": CALPGM_AGGREGATE_MAP_SCHEMA,
        "source": str(source),
        "itot_basis": itot_basis,
        "vibrational_options": option_rows,
        "entries": entries,
        "unique_state_count": len(signatures),
    }


def decode_calpgm_aggregate_label(
    label: Mapping[str, Any], aggregate_map: Mapping[str, Any]
) -> dict[str, Any]:
    """Expand one printed ``N,K,v,nn,F`` label into coupled quantum numbers."""

    required = {"N", "K", "v", "nn", "F"}
    missing = required.difference(label)
    if missing:
        raise ValueError(f"aggregate CALPGM label is missing {sorted(missing)}")
    undefined = sorted(name for name in required if label[name] is None)
    if undefined:
        raise ValueError(f"aggregate CALPGM label has undefined {undefined}")
    vibration, nn = int(label["v"]), int(label["nn"])
    candidates = [
        entry for entry in aggregate_map.get("entries", ())
        if int(entry["vibration"]) == vibration and int(entry["nn"]) == nn
    ]
    if not candidates:
        raise KeyError(f"no aggregate mapping for v={vibration}, nn={nn}")
    signatures = {tuple(entry["offsets_relative_to_F"]) for entry in candidates}
    if len(signatures) != 1:
        raise ValueError(f"ambiguous aggregate mapping for v={vibration}, nn={nn}")
    template = candidates[0]
    F = float(label["F"])
    expected_N = F + float(template["N_minus_F"])
    if not math.isclose(expected_N, float(label["N"]), abs_tol=1.0e-9):
        raise ValueError(
            f"inconsistent aggregate label N={label['N']}, F={label['F']}, nn={nn}; "
            f"the state table gives N-F={template['N_minus_F']}"
        )
    expanded = dict(label)
    expanded["F"] = F
    expanded["coupled_angular_momenta"] = [
        F + float(offset)
        for offset in template.get("intermediate_angular_momenta_minus_F", ())
    ]
    if aggregate_map.get("itot_basis"):
        expanded["nuclear_permutation_symmetry"] = int(
            template["nuclear_permutation_symmetry"]
        )
        expanded["I_tot"] = float(template["I_tot"])
        if "nuclear_spin_multiplicity" in template:
            expanded["nuclear_spin_multiplicity"] = int(
                template["nuclear_spin_multiplicity"]
            )
        if "nuclear_spin_groups" in template:
            expanded["nuclear_spin_groups"] = list(template["nuclear_spin_groups"])
    expanded["aggregate_signature"] = list(template["offsets_relative_to_F"])
    return expanded


def calpgm_aggregate_state_key(label: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    """Return the canonical integer key for a printed aggregate state."""

    values = tuple(label.get(name) for name in ("N", "K", "v", "nn", "F"))
    if any(value is None for value in values):
        raise ValueError("a CALPGM aggregate state requires finite N,K,v,nn,F labels")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def build_calpgm_aggregate_state_registry(
    lines: Sequence[Mapping[str, Any]], aggregate_map: Mapping[str, Any]
) -> dict[tuple[int, int, int, int, int], dict[str, Any]]:
    """Expand and deduplicate every physical upper/lower state in a line list."""

    registry: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    for line in lines:
        if float(line.get("frequency_MHz", 0.0)) == 0.0:
            continue
        for side in ("upper", "lower"):
            raw = line.get(side)
            if not isinstance(raw, Mapping):
                raise ValueError(f"CALPGM line has no {side} state")
            expanded = decode_calpgm_aggregate_label(raw, aggregate_map)
            key = calpgm_aggregate_state_key(expanded)
            previous = registry.setdefault(key, expanded)
            if previous != expanded:
                raise ValueError(f"inconsistent expanded CALPGM state {key}")
    return registry


def build_calpgm_coupled_basis(
    aggregate_map: Mapping[str, Any],
    F: int,
    *,
    n_max: int | None = None,
) -> list[dict[str, Any]]:
    """Build the full signed-K coupled basis represented by one CALPGM F block."""

    if isinstance(F, bool) or int(F) != F or F < 0:
        raise ValueError("a CALPGM F block requires a nonnegative integer")
    if n_max is not None and (
        isinstance(n_max, bool) or int(n_max) != n_max or n_max < 0
    ):
        raise ValueError("n_max must be a nonnegative integer")
    unique: dict[tuple[int, int], Mapping[str, Any]] = {}
    for entry in aggregate_map.get("entries", ()):
        key = int(entry["vibration"]), int(entry["nn"])
        unique.setdefault(key, entry)
    basis = []
    for (vibration, nn), entry in sorted(unique.items()):
        N_float = float(F) + float(entry["N_minus_F"])
        if not N_float.is_integer() or N_float < 0:
            continue
        N = int(N_float)
        if n_max is not None and N > n_max:
            continue
        intermediates = [
            float(F) + float(offset)
            for offset in entry.get("intermediate_angular_momenta_minus_F", ())
        ]
        if any(value < 0.0 for value in intermediates):
            continue
        if aggregate_map.get("itot_basis"):
            I_tot = float(entry["I_tot"])
            last = intermediates[-1] if intermediates else float(N)
            if not abs(last - I_tot) <= F <= last + I_tot:
                continue
        for K in range(-N, N + 1):
            label = {"N": N, "K": K, "v": vibration, "nn": nn, "F": F}
            basis.append(decode_calpgm_aggregate_label(label, aggregate_map))
    return assign_calpgm_symmetry_labels(basis, aggregate_map)


_CALPGM_WANG_CODES = {"x": 0, "c": 1, "b": 2, "a": 3}


def assign_calpgm_symmetry_labels(
    basis: Sequence[Mapping[str, Any]],
    aggregate_map: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Attach the exact GETQQ block and Wang irrep to signed-K labels."""

    entries = {
        (int(entry["vibration"]), int(entry["nn"]), str(entry["wang_symmetry"]).lower()): entry
        for entry in aggregate_map.get("entries", ())
    }
    option_by_vibration = {
        int(row["vibration"]): row
        for row in aggregate_map.get("vibrational_options", ())
    }
    l_values = _calpgm_l_values(option_by_vibration)
    result = []
    for raw in basis:
        label = dict(raw)
        vibration = int(label["v"])
        nn = int(label["nn"])
        N = int(label["N"])
        K = int(label["K"])
        l_value = l_values.get(vibration, 0)
        if K > 0:
            symbol = "x" if not (K & 1) else "b"
        elif K < 0:
            symbol = "a" if not ((-K) & 1) else "c"
        elif l_value > 0:
            # The positive-l K=0 vector is the printed partner of the
            # negative-l a-symmetry state (GETQN's K=0 swap).
            symbol = "a"
        elif l_value < 0:
            symbol = "x"
        else:
            symbol = "a" if N & 1 else "x"
        entry = entries.get((vibration, nn, symbol))
        if entry is None:
            raise KeyError(
                f"no CALPGM symmetry entry for v={vibration}, nn={nn}, symmetry={symbol}"
            )
        label.update({
            "calpgm_block": int(entry["block"]),
            "wang_symmetry": symbol,
            "wang_symmetry_code": _CALPGM_WANG_CODES[symbol],
            "l_value": l_value,
            "K_magnitude": abs(K),
        })
        result.append(label)
    return result


def build_calpgm_symmetry_subblocks(
    aggregate_map: Mapping[str, Any],
    F: int,
    *,
    n_max: int | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct CALPGM's discrete ``GETQQ`` subblocks for one ``F``.

    The aggregate table lists a Wang irrep but omits the actual K sequence.
    ``GETQQ`` obtains that sequence from the four ``KMIN`` values and reverses
    the irrep at odd N.  Reconstructing it here prevents l-doublet K=0 states
    and the two K parities from being folded into an undifferentiated signed-K
    group.
    """

    if isinstance(F, bool) or int(F) != F or F < 0:
        raise ValueError("a CALPGM F block requires a nonnegative integer")
    if n_max is not None and (
        isinstance(n_max, bool) or int(n_max) != n_max or n_max < 0
    ):
        raise ValueError("n_max must be a nonnegative integer")

    option_by_vibration = {
        int(row["vibration"]): row
        for row in aggregate_map.get("vibrational_options", ())
    }
    l_values = _calpgm_l_values(option_by_vibration)
    result: list[dict[str, Any]] = []
    for entry in aggregate_map.get("entries", ()):
        if int(entry.get("weight", 1)) <= 0:
            continue
        N_float = float(F) + float(entry["N_minus_F"])
        if not N_float.is_integer() or N_float < 0:
            continue
        N = int(N_float)
        if n_max is not None and N > n_max:
            continue
        intermediates = [
            float(F) + float(offset)
            for offset in entry.get("intermediate_angular_momenta_minus_F", ())
        ]
        if any(value < 0.0 for value in intermediates):
            continue
        if aggregate_map.get("itot_basis"):
            I_tot = float(entry["I_tot"])
            last = intermediates[-1] if intermediates else float(N)
            if not abs(last - I_tot) <= F <= last + I_tot:
                continue

        vibration = int(entry["vibration"])
        option = option_by_vibration.get(vibration, {})
        minimum = int(option.get("k_min", 0))
        maximum = min(N, int(option.get("k_max", N)))
        l_value = l_values.get(vibration, 0)
        kmin = [0, 1, 1, 2]
        if l_value < 0:
            kmin[3] = 0
        elif l_value > 0:
            kmin[0] = kmin[3] = 2
        for index, value in enumerate(kmin):
            while value < minimum:
                value += 2
            kmin[index] = value

        symbol = str(entry["wang_symmetry"]).lower()
        if symbol not in _CALPGM_WANG_CODES:
            raise ValueError(f"unknown CALPGM Wang symmetry {symbol!r}")
        symmetry = _CALPGM_WANG_CODES[symbol]
        ksym = 3 - symmetry if N & 1 else symmetry
        k_start = kmin[ksym]
        if k_start > maximum:
            continue
        subblock = dict(entry)
        subblock.update({
            "N": N,
            "F": int(F),
            "wang_symmetry_code": symmetry,
            "l_value": l_value,
            "k_start": k_start,
            "k_values": list(range(k_start, maximum + 1, 2)),
            "dimension": 1 + ((maximum - k_start) >> 1),
        })
        result.append(subblock)
    return result


def _calpgm_l_values(
    option_by_vibration: Mapping[int, Mapping[str, Any]],
) -> dict[int, int]:
    """Decode the signed l pair represented by the printed ESYMWT field."""

    vibrations = sorted(option_by_vibration)
    unsigned: dict[int, int] = {}
    orders: dict[int, int] = {}
    for vibration in vibrations:
        row = option_by_vibration[vibration]
        order = abs(int(row.get("nuclear_symmetry_order", 0)))
        # SPINV applies cyclic l reduction only for genuine E symmetries
        # (NSYM >= 3).  NSYM=2 is an asymmetric-top axis code, not C2
        # modular arithmetic.
        if order < 3:
            order = 0
        code = abs(int(row.get("energy_symmetry_weight", 0)))
        # The aggregate listing has already combined EWT1 and EWT0.  EWT1 is
        # the hundreds digit in the normal format and the thousands digit in
        # the extended format.
        divisor = 1000 if code >= 1000 and order and code // 100 > order else 100
        value = code // divisor
        if value > 10:
            value -= 10
        if order and value > order:
            value = (value - 1) % order + 1
        unsigned[vibration] = value
        orders[vibration] = order

    signed = dict(unsigned)
    index = 0
    while index + 1 < len(vibrations):
        left, right = vibrations[index:index + 2]
        order = orders[left] or orders[right]
        lleft, lright = unsigned[left], unsigned[right]
        paired = lleft and lright and (not order or (lleft + lright) % order == 0)
        if paired:
            if lleft > lright:
                signed[left] = -lright
            else:
                signed[right] = -lleft
            index += 2
        else:
            index += 1
    return signed


def calpgm_getmask(
    bra: Mapping[str, Any],
    ket: Mapping[str, Any],
    *,
    delta_k: int,
    delta_l: int,
    alpha: int = 0,
    nuclear_symmetry_order: int = 0,
    identity_itot: bool = False,
    spin_symmetry_odd: bool = False,
) -> int:
    """Return SPINV's three-bit ``GETMASK`` branch selector.

    Bit 0 is the ``+delta_k,+delta_l`` branch, bit 2 is its reversed
    counterpart, and bit 1 is the quasi-diagonal branch
    ``K_bra=delta_k-K_ket, l_bra=delta_l-l_ket``.  Keeping bit 1 explicit is
    essential for l doublets: it is the branch that carries the K=0 member
    and couples the two signs of l without mixing distinct Hamiltonian blocks.
    """

    kd = abs(int(delta_k))
    ldel = int(delta_l)
    mask = 7
    nitot = abs(int(nuclear_symmetry_order))
    bra_symmetry, bra_iqn, bra_esym = _calpgm_itot_symmetry_fields(bra, nitot)
    ket_symmetry, ket_iqn, ket_esym = _calpgm_itot_symmetry_fields(ket, nitot)
    if nitot >= 3:
        parity_change = (bra_iqn ^ ket_iqn) >> 1
        if identity_itot:
            if parity_change != 0:
                return 0
        elif not (parity_change & 1):
            if spin_symmetry_odd:
                return 0
        else:
            if spin_symmetry_odd:
                return 0
            if bra_esym == 0 and ket_esym == 0:
                return 0
        mask = 0
        difference = ket_symmetry - bra_symmetry
        if alpha == 0:
            if difference == 0:
                mask = 5
        elif alpha < nitot:
            if (difference + alpha) % nitot == 0:
                mask = 1
            if (difference - alpha) % nitot == 0:
                mask += 4
        elif (bra_symmetry + ket_symmetry - alpha) % nitot == 0:
            mask = 2
        if mask == 0:
            return 0

    lbra = int(bra.get("l_value", 0))
    lket = int(ket.get("l_value", 0))
    difference = lbra - lket
    if ldel == 0:
        if difference != 0:
            mask &= 2
    elif difference == 0:
        mask &= 2
    elif difference != ldel:
        mask &= 6
    elif -difference != ldel:
        mask &= 3
    if not (mask & 2):
        return mask

    mask &= 5
    if lbra + lket != ldel:
        return mask
    kbra = int(bra.get("k_start", 0))
    kket = int(ket.get("k_start", 0))
    remaining = kd - kbra - kket
    if remaining < 0:
        return mask
    if not (ket_iqn & 1):
        remaining -= 2
    if not (bra_iqn & 1):
        remaining -= 2
    if remaining >= 0:
        mask += 2
    return mask


def _calpgm_itot_symmetry_fields(
    label: Mapping[str, Any], nitot: int
) -> tuple[int, int, int]:
    """Reconstruct GETQS' XISYM, XIQN and ``is_esym`` fields."""

    # Aggregate-spin offsets are printed in ordinary angular-momentum units,
    # whereas GETQS stores the same packed quantum number doubled.  Restore
    # the integer representation before extracting XISYM and XIQN.
    raw = 2 * int(label.get("nuclear_permutation_symmetry", 0))
    if nitot < 3:
        symmetry = 0
        iqn = 0
        esym = 0
    else:
        symmetry = (raw >> 2) % nitot
        iqn = raw - (symmetry << 2)
        partner = nitot - symmetry
        esym = 0 if symmetry == 0 or symmetry == partner else (1 if symmetry < partner else -1)
        if esym:
            iqn += 1
    if int(label.get("l_value", 0)) != 0 or int(label.get("k_start", 0)) != 0:
        iqn |= 1
    return symmetry, iqn, esym


def calpgm_wang_transform(N: int) -> tuple[tuple[int, ...], Any]:
    """Return the raw-K to signed-K Wang transform for a symmetric-top block.

    Positive and negative printed K label the +1 and -1 eigenvalues,
    respectively, of a pi rotation about the body-fixed b axis.  K=0 remains
    unsigned.  Columns are ordered by the returned signed-K labels.
    """

    import numpy as np

    if isinstance(N, bool) or int(N) != N or N < 0:
        raise ValueError("a Wang transform requires a nonnegative integer N")
    N = int(N)
    columns: dict[int, Any] = {0: np.eye(2 * N + 1, dtype=float)[:, N]}
    for magnitude in range(1, N + 1):
        positive = np.zeros(2 * N + 1, dtype=float)
        negative = np.zeros(2 * N + 1, dtype=float)
        phase = -1.0 if ((N - magnitude) & 1) else 1.0
        positive[N + magnitude] = 1.0 / math.sqrt(2.0)
        positive[N - magnitude] = 1.0 / math.sqrt(2.0)
        negative[N + magnitude] = 1.0 / math.sqrt(2.0)
        negative[N - magnitude] = -1.0 / math.sqrt(2.0)
        columns[magnitude if phase > 0.0 else -magnitude] = positive
        columns[-magnitude if phase > 0.0 else magnitude] = negative
    labels = tuple(range(-N, N + 1))
    transform = np.column_stack([columns[label] for label in labels])
    # SPINV's real Wang convention carries an additional (-1)^N phase on
    # the negative signed-K branch.  It is invisible within one N manifold
    # but fixes the relative sign of odd-rank N<->N+1 matrix elements.
    if N & 1:
        transform[:, :N] *= -1.0
    transform.setflags(write=False)
    return labels, transform


def assign_calpgm_aggregate_eigenvectors(
    vectors: Any,
    basis_labels: Sequence[Mapping[str, Any]],
    *,
    block: int | None = None,
    minimum_overlap: float = 0.0,
    allow_duplicate_labels: bool = False,
) -> list[dict[str, Any]]:
    """Assign aggregate labels with CALPGM's ``IDIAG=1`` ``ORDBLK`` rule.

    With one requested block per basis function, ``ORDBLK`` repeatedly fixes
    the largest remaining projection.  A displaced eigenvector is then
    rescored after the occupied row is removed.  This greedy ordering is
    intentionally different from a maximum-sum (Hungarian) assignment.
    Returned records are in IDIAG order; ``eigenvector_index`` retains the
    column index produced by the eigensolver.
    """

    import numpy as np
    matrix = np.asarray(vectors, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("CALPGM projection sorting requires a square eigenvector matrix")
    if len(basis_labels) != matrix.shape[0]:
        raise ValueError("one aggregate label is required for every coupled-basis row")
    if not math.isfinite(float(minimum_overlap)) or not 0.0 <= minimum_overlap <= 1.0:
        raise ValueError("minimum_overlap must lie between zero and one")
    keys = [calpgm_aggregate_state_key(label) for label in basis_labels]
    if len(set(keys)) != len(keys) and not allow_duplicate_labels:
        raise ValueError("coupled-basis aggregate labels must be unique within a block")
    gram_error = float(np.max(np.abs(matrix.conj().T @ matrix - np.eye(matrix.shape[1]))))
    if gram_error > 1.0e-8:
        raise ValueError(f"eigenvectors are not orthonormal: {gram_error:.3e}")
    overlaps = np.abs(matrix) ** 2
    import heapq

    size = matrix.shape[0]
    # Each column keeps its rows in descending projection order.  A heap then
    # exposes the best currently valid column candidate.  This is exactly the
    # ORDBLK greedy rule below, including its column and row tie breaking, but
    # avoids rescanning the shrinking dense matrix at every assignment.
    row_order = np.argsort(-overlaps, axis=0, kind="stable")
    row_available = np.ones(size, dtype=bool)
    column_available = np.ones(size, dtype=bool)
    pointers = np.zeros(size, dtype=np.intp)
    heap = [
        (-float(overlaps[int(row_order[0, column]), column]), -column,
         int(row_order[0, column]))
        for column in range(size)
    ]
    heapq.heapify(heap)
    by_row: dict[int, int] = {}
    while len(by_row) < size:
        negative_projection, negative_column, row = heapq.heappop(heap)
        column = -negative_column
        if not column_available[column]:
            continue
        if not row_available[row]:
            pointer = int(pointers[column]) + 1
            while pointer < size and not row_available[int(row_order[pointer, column])]:
                pointer += 1
            pointers[column] = pointer
            if pointer < size:
                candidate_row = int(row_order[pointer, column])
                heapq.heappush(
                    heap,
                    (-float(overlaps[candidate_row, column]), -column, candidate_row),
                )
            continue
        by_row[row] = column
        row_available[row] = False
        column_available[column] = False
    assignments = []
    occurrences: dict[tuple[int, int, int, int, int], int] = {}
    for row in range(matrix.shape[0]):
        column = by_row[row]
        ranked = np.sort(overlaps[:, column])
        purity = float(overlaps[row, column])
        runner_up = float(ranked[-2]) if len(ranked) > 1 else 0.0
        if purity < minimum_overlap:
            raise ValueError(
                f"aggregate assignment purity {purity:.6f} is below {minimum_overlap:.6f}"
            )
        assignment = {
            "idiag_index": row,
            "eigenvector_index": column,
            "basis_index": row,
            "state_key": list(keys[row]),
            "label": dict(basis_labels[row]),
            "projection": purity,
            "runner_up_projection": runner_up,
            "projection_gap": purity - runner_up,
            "phase": 1.0 if matrix[row, column].real >= 0.0 else -1.0,
        }
        occurrence = occurrences.get(keys[row], 0)
        occurrences[keys[row]] = occurrence + 1
        if allow_duplicate_labels:
            assignment["label_occurrence"] = occurrence
        if block is not None:
            assignment["block"] = int(block)
        assignments.append(assignment)
    return assignments


def build_calpgm_native_fit_observations(
    lines: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    *,
    minimum_projection: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert aggregate line labels to the explicit native level indices."""

    if not 0.0 <= minimum_projection <= 1.0:
        raise ValueError("minimum_projection must lie between zero and one")
    lookup: dict[tuple[int, int, int, int, int], tuple[int, int]] = {}
    for assignment in assignments:
        key = tuple(int(value) for value in assignment["state_key"])
        if len(key) != 5:
            raise ValueError("an aggregate assignment state_key must contain N,K,v,nn,F")
        if float(assignment.get("projection", 1.0)) < minimum_projection:
            raise ValueError(f"aggregate assignment {key} is below the projection threshold")
        location = int(assignment["block"]), int(assignment["eigenvector_index"])
        previous = lookup.setdefault(key, location)
        if previous != location:
            raise ValueError(f"aggregate state {key} has multiple native assignments")

    observations = []
    for line in lines:
        if float(line.get("frequency_MHz", 0.0)) == 0.0:
            continue
        upper_key = calpgm_aggregate_state_key(line["upper"])
        lower_key = calpgm_aggregate_state_key(line["lower"])
        try:
            upper = lookup[upper_key]
            lower = lookup[lower_key]
        except KeyError as exc:
            raise KeyError(f"unassigned aggregate CALPGM state {exc.args[0]}") from exc
        observations.append({
            "label": str(line.get("label", len(observations) + 1)),
            "upper": {"F": upper[0], "level": upper[1]},
            "lower": {"F": lower[0], "level": lower[1]},
            "upper_state_key": list(upper_key),
            "lower_state_key": list(lower_key),
            "frequency_MHz": float(line["frequency_MHz"]),
            "uncertainty_MHz": float(line["uncertainty_MHz"]),
            "blend_weight": float(line.get("weight", 1.0)),
        })
    return observations


def collapse_calpgm_native_blends(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse contiguous SPFIT components sharing one measured frequency.

    CALPGM represents an unresolved experimental feature by consecutive
    components with the same frequency and uncertainty.  The calculated
    frequency is their normalized intensity-weighted mean.
    """

    groups: list[list[Mapping[str, Any]]] = []
    for observation in observations:
        key = (
            float(observation["frequency_MHz"]),
            float(observation["uncertainty_MHz"]),
        )
        previous_key = None
        if groups:
            previous = groups[-1][0]
            previous_key = (
                float(previous["frequency_MHz"]),
                float(previous["uncertainty_MHz"]),
            )
        if key != previous_key:
            groups.append([])
        groups[-1].append(observation)

    collapsed: list[dict[str, Any]] = []
    for group in groups:
        if len(group) == 1:
            item = dict(group[0])
            item.pop("blend_weight", None)
            item["component_count"] = 1
            collapsed.append(item)
            continue
        weights = np.asarray(
            [float(item.get("blend_weight", 1.0)) for item in group], dtype=float
        )
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("CALPGM blend weights must be finite and positive")
        weights /= float(np.sum(weights))
        components = []
        for item, weight in zip(group, weights, strict=True):
            components.append({
                "label": str(item.get("label", len(components) + 1)),
                "upper": dict(item["upper"]),
                "lower": dict(item["lower"]),
                "weight": float(weight),
                **(
                    {"upper_state_key": list(item["upper_state_key"])}
                    if "upper_state_key" in item else {}
                ),
                **(
                    {"lower_state_key": list(item["lower_state_key"])}
                    if "lower_state_key" in item else {}
                ),
            })
        collapsed.append({
            "label": "+".join(str(item.get("label", "")) for item in group),
            "components": components,
            "component_count": len(components),
            "frequency_MHz": float(group[0]["frequency_MHz"]),
            "uncertainty_MHz": float(group[0]["uncertainty_MHz"]),
        })
    return collapsed


def qualify_calpgm_native_reproduction(
    observations: Sequence[Mapping[str, Any]],
    level_energies: Mapping[tuple[int, int], float],
    *,
    rms_limit_MHz: float,
) -> dict[str, Any]:
    """Apply a fail-closed numerical criterion to an assigned native spectrum."""

    import numpy as np

    if not math.isfinite(rms_limit_MHz) or rms_limit_MHz <= 0.0:
        raise ValueError("rms_limit_MHz must be finite and positive")
    residuals = []
    missing = []
    for observation in observations:
        components = observation.get("components")
        if components is None:
            components = ({
                "upper": observation["upper"],
                "lower": observation["lower"],
                "weight": 1.0,
            },)
        calculated = 0.0
        complete = True
        weights = np.asarray(
            [float(component.get("weight", 1.0)) for component in components], dtype=float
        )
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("CALPGM blend weights must be finite and positive")
        weights /= float(np.sum(weights))
        for component, weight in zip(components, weights, strict=True):
            upper = component["upper"]
            lower = component["lower"]
            upper_key = int(upper.get("F", upper.get("J"))), int(upper["level"])
            lower_key = int(lower.get("F", lower.get("J"))), int(lower["level"])
            if upper_key not in level_energies or lower_key not in level_energies:
                complete = False
                break
            calculated += float(weight) * (
                float(level_energies[upper_key]) - float(level_energies[lower_key])
            )
        if not complete:
            missing.append(str(observation.get("label", len(missing) + 1)))
            continue
        residuals.append(float(observation["frequency_MHz"]) - calculated)
    values = np.asarray(residuals, dtype=float)
    rms = float(np.sqrt(np.mean(values * values))) if len(values) else float("inf")
    return {
        "physical_line_count": len(observations),
        "component_count": sum(int(item.get("component_count", 1)) for item in observations),
        "assigned_line_count": len(residuals),
        "missing_lines": missing,
        "rms_residual_MHz": rms,
        "maximum_absolute_residual_MHz": (
            float(np.max(np.abs(values))) if len(values) else float("inf")
        ),
        "rms_limit_MHz": float(rms_limit_MHz),
        "native_reproduction_qualified": (
            len(residuals) == len(observations) and math.isfinite(rms) and rms <= rms_limit_MHz
        ),
    }


def run_calpgm_blind_fit(
    parameter_file: Path | str,
    line_file: Path | str,
    *,
    executable: Path | str,
    validation_above_MHz: float,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Fit a low-frequency subset and predict the held-out high-frequency set."""
    parameter_source, line_source = Path(parameter_file), Path(line_file)
    engine = Path(executable).expanduser().resolve()
    if not engine.is_file():
        raise FileNotFoundError(engine)
    parsed_lines = read_calpgm_assigned_lines(line_source)
    raw_lines = [
        raw for raw in line_source.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith(("#", "!"))
    ]
    paired = list(zip(parsed_lines, raw_lines, strict=True))
    training = [raw for line, raw in paired if line["frequency_MHz"] == 0.0 or line["frequency_MHz"] <= validation_above_MHz]
    validation = [raw for line, raw in paired if line["frequency_MHz"] > validation_above_MHz]
    if len(training) < 2 or not validation:
        raise ValueError("blind CALPGM split requires training and validation observations")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="matrix-lock-calpgm-") as directory:
        work = Path(directory)
        shutil.copy2(parameter_source, work / "training.par")
        (work / "training.lin").write_text("\n".join(training) + "\n", encoding="utf-8")
        _rewrite_control(work / "training.par", len(training), None)
        environment = dict(os.environ)
        environment.setdefault("SPECNAME", str(engine.parent) + os.sep)
        _execute_spfit(engine, "training", work, environment, timeout)
        training_fit = parse_calpgm_fit(work / "training.fit")
        shutil.copy2(work / "training.par", work / "validation.par")
        (work / "validation.lin").write_text("\n".join(validation) + "\n", encoding="utf-8")
        _rewrite_control(work / "validation.par", len(validation), 0)
        _execute_spfit(engine, "validation", work, environment, timeout)
        validation_fit = parse_calpgm_fit(work / "validation.fit")
    return {
        "schema": CALPGM_FIT_RESULT_SCHEMA,
        "backend": "CALPGM/SPFIT external compatibility backend",
        "model": read_calpgm_parameters(parameter_source),
        "assigned_lines_sha256": _sha256(line_source),
        "quantum_number_format": {
            "kind": "symmetric-top aggregate-spin",
            "fields": ["N", "K", "v", "nn", "F"],
            "aggregate_spin_preserved": True,
        },
        "split": {
            "validation_above_MHz": float(validation_above_MHz),
            "training_records": sum(
                line["frequency_MHz"] != 0.0
                for line, _raw in paired
                if line["frequency_MHz"] <= validation_above_MHz
            ),
            "validation_records": len(validation),
        },
        "training": training_fit,
        "validation": validation_fit,
        "timing_seconds": time.perf_counter() - started,
        "engine": {"path": str(engine), "sha256": _sha256(engine)},
        "native_lock_hamiltonian": False,
    }


def write_calpgm_fit_result(path: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _option_payload(reduction: str, fields: Sequence[str]) -> dict[str, Any]:
    spind, nvib = int(fields[0]), int(fields[1])
    payload = {
        "reduction": reduction.lower(),
        "symmetric_top": spind < 0,
        "spin_degeneracy_code": abs(spind),
        "nvib": nvib,
        "k_min": int(fields[2]),
        "k_max": int(fields[3]),
        "interaction_flags": int(fields[4]),
        "itot_basis": int(fields[5]) < 0,
        "symmetry_axis": abs(int(fields[5])),
        "weight_positive": int(fields[6]),
        "weight_negative": int(fields[7]),
        "vibrational_symmetry": float(fields[8].replace("D", "E")),
        "e_state_weight": int(fields[9]),
    }
    if len(fields) > 10:
        payload["diagonalization"] = int(fields[10])
    if len(fields) > 11:
        payload["extended_options"] = int(fields[11])
    return payload


def _rewrite_control(path: Path, line_count: int, iterations: int | None) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    fields = lines[1].split()
    fields[1] = str(line_count)
    if iterations is not None:
        fields[2] = str(iterations)
    lines[1] = " ".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute_spfit(
    engine: Path, stem: str, work: Path, environment: Mapping[str, str], timeout: float
) -> None:
    completed = subprocess.run(
        [str(engine), stem], cwd=work, env=dict(environment), timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"SPFIT failed ({completed.returncode}): {completed.stdout[-2000:]}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


__all__ = [
    "CALPGM_AGGREGATE_MAP_SCHEMA",
    "CALPGM_FIT_RESULT_SCHEMA",
    "CALPGM_MODEL_SCHEMA",
    "assign_calpgm_symmetry_labels",
    "assign_calpgm_aggregate_eigenvectors",
    "build_calpgm_aggregate_state_registry",
    "build_calpgm_coupled_basis",
    "build_calpgm_symmetry_subblocks",
    "build_calpgm_native_fit_observations",
    "calpgm_aggregate_state_key",
    "calpgm_getmask",
    "calpgm_wang_transform",
    "decode_calpgm_aggregate_label",
    "parse_calpgm_fit",
    "qualify_calpgm_native_reproduction",
    "read_calpgm_aggregate_map",
    "read_calpgm_assigned_lines",
    "read_calpgm_parameters",
    "run_calpgm_blind_fit",
    "write_calpgm_fit_result",
]
