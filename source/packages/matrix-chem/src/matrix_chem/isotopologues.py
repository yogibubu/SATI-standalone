from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import shlex

from matrix_core.sectioned_xyz import (
    has_section,
    read_sectioned_lines,
    replace_section,
    section_content,
)


XYZIN_ISOTOPOLOGUES_SECTION = "ISOTOPOLOGUES"
ORACLE_XYZ_ISOTOPOLOGUES_SCHEMA = "oracle.xyz.isotopologues.v1"
XYZIN_ISOTOPOLOGUES_SCHEMA = ORACLE_XYZ_ISOTOPOLOGUES_SCHEMA
SUPPORTED_XYZIN_ISOTOPOLOGUES_SCHEMAS = {ORACLE_XYZ_ISOTOPOLOGUES_SCHEMA}


@dataclass(frozen=True)
class XyzinIsotopologueRecord:
    label: str
    substitutions: dict[int, int] = field(default_factory=dict)
    rotational_MHz: tuple[float, float, float] | None = None
    deltavib_MHz: tuple[float, float, float] | None = None
    deltavib_source: str = "unspecified"
    deltavib_convention: str = "subtract"
    deltael_MHz: tuple[float, float, float] | None = None
    deltael_source: str = "unspecified"
    deltael_convention: str = "subtract"
    sigma_MHz: tuple[float, float, float] | None = None
    covariance_MHz2: tuple[tuple[float, float, float], ...] | None = None


@dataclass(frozen=True)
class XyzinIsotopologueValidationIssue:
    severity: str
    code: str
    message: str
    context: str = ""


def mass_number(value) -> int:
    text = str(value).strip()
    aliases = {"D": 2, "T": 3}
    if text.upper() in aliases:
        return aliases[text.upper()]
    return int(text)


def parse_substitutions(text: str) -> dict[int, int]:
    result: dict[int, int] = {}
    text = (text or "").strip()
    if not text:
        return result
    for chunk in text.split(";"):
        atom_text, isotope_text = chunk.split(":", 1)
        atom_index = int(atom_text.strip())
        isotope_a = mass_number(isotope_text.strip())
        if atom_index < 1:
            raise ValueError("Substitution atom indexes are one-based")
        if atom_index in result:
            raise ValueError(f"Duplicate substitution for atom {atom_index}")
        result[atom_index] = isotope_a
    return result


def format_substitutions(substitutions: dict[int, int]) -> str:
    return ";".join(f"{atom}:{mass}" for atom, mass in sorted(substitutions.items()))


def sigma_text_from_weight(weight: float) -> str:
    if weight <= 0.0:
        return ""
    return f"{(1.0 / weight) ** 0.5:.12g}"


def has_xyzin_isotopologues(path: Path) -> bool:
    return has_section(Path(path), XYZIN_ISOTOPOLOGUES_SECTION)


def read_xyzin_isotopologue_records(path: Path) -> tuple[XyzinIsotopologueRecord, ...]:
    lines = section_content(read_sectioned_lines(Path(path)), XYZIN_ISOTOPOLOGUES_SECTION)
    return parse_xyzin_isotopologue_records(lines)


def validate_xyzin_isotopologue_file(
    path: Path,
    *,
    atom_count: int | None = None,
    require_rotational: bool = False,
) -> tuple[XyzinIsotopologueValidationIssue, ...]:
    return validate_xyzin_isotopologue_records(
        read_xyzin_isotopologue_records(Path(path)),
        atom_count=atom_count,
        require_rotational=require_rotational,
    )


def validate_xyzin_isotopologue_records(
    records: tuple[XyzinIsotopologueRecord, ...],
    *,
    atom_count: int | None = None,
    require_rotational: bool = False,
) -> tuple[XyzinIsotopologueValidationIssue, ...]:
    issues: list[XyzinIsotopologueValidationIssue] = []

    def add(severity: str, code: str, message: str, context: str = "") -> None:
        issues.append(XyzinIsotopologueValidationIssue(severity, code, message, context))

    if not records:
        add("error", "empty_isotopologue_section", "#ISOTOPOLOGUES contains no records")
        return tuple(issues)
    labels: dict[str, int] = {}
    definitions: dict[tuple[tuple[int, int], ...], str] = {}
    for record in records:
        label = record.label.strip()
        if not label:
            add("error", "empty_isotopologue_label", "Isotopologue label cannot be empty")
        elif label in labels:
            add(
                "error",
                "duplicate_isotopologue_label",
                "Duplicate isotopologue label in #ISOTOPOLOGUES",
                f"label={label}",
            )
        labels[label] = labels.get(label, 0) + 1
        normalized_substitutions: dict[int, int] = {}
        for atom, mass in record.substitutions.items():
            try:
                normalized_substitutions[int(atom)] = int(mass)
            except (TypeError, ValueError):
                add(
                    "error",
                    "invalid_substitution_value",
                    "Substitution atom indexes and isotope masses must be integers.",
                    f"label={label};atom={atom};mass={mass}",
                )
        definition_key = tuple(sorted(normalized_substitutions.items()))
        previous = definitions.get(definition_key)
        if previous is not None and label:
            add(
                "warning",
                "duplicate_isotopologue_definition",
                "Two isotopologue records use the same substitution definition.",
                f"labels={previous},{label};definition={format_substitutions(dict(definition_key)) or 'parent'}",
            )
        else:
            definitions[definition_key] = label
        for atom_index, isotope in sorted(normalized_substitutions.items()):
            if atom_index < 1:
                add(
                    "error",
                    "invalid_substitution_index",
                    "Substitution atom indexes are one-based.",
                    f"label={label};atom={atom_index}",
                )
            if atom_count is not None and atom_index > atom_count:
                add(
                    "error",
                    "substitution_index_out_of_range",
                    "Substitution atom index is outside the parent geometry.",
                    f"label={label};atom={atom_index};natoms={atom_count}",
                )
            if isotope <= 0:
                add(
                    "error",
                    "invalid_isotope_mass",
                    "Isotope mass number must be positive.",
                    f"label={label};atom={atom_index};mass={isotope}",
                )
        if require_rotational and record.rotational_MHz is None:
            add(
                "error",
                "missing_rotational_constants",
                "SEfit requires ROTATIONAL_MHZ for every isotopologue.",
                f"label={label}",
            )
        _validate_optional_triple(
            add,
            label,
            "rotational_constants",
            "ROTATIONAL_MHZ",
            record.rotational_MHz,
            positive=True,
        )
        _validate_optional_triple(add, label, "deltavib", "DELTAVIB_MHZ", record.deltavib_MHz)
        _validate_optional_triple(add, label, "deltael", "DELTAEL_MHZ", record.deltael_MHz)
        _validate_optional_triple(add, label, "sigma", "SIGMA_MHZ", record.sigma_MHz, positive=True)
        _validate_covariance(add, label, record.covariance_MHz2)
        _validate_correction_convention(add, label, "DELTAVIB_MHZ", record.deltavib_convention)
        _validate_correction_convention(add, label, "DELTAEL_MHZ", record.deltael_convention)
    return tuple(issues)


def xyzin_isotopologue_validation_errors(
    issues: tuple[XyzinIsotopologueValidationIssue, ...],
) -> tuple[XyzinIsotopologueValidationIssue, ...]:
    return tuple(item for item in issues if item.severity == "error")


def format_xyzin_isotopologue_issues(
    issues: tuple[XyzinIsotopologueValidationIssue, ...],
) -> str:
    return "; ".join(
        f"{item.severity}:{item.code}:{item.message}"
        + (f" ({item.context})" if item.context else "")
        for item in issues
    )


def write_xyzin_isotopologue_records(
    path: Path, records: tuple[XyzinIsotopologueRecord, ...]
) -> Path:
    target = Path(path)
    replace_section(target, XYZIN_ISOTOPOLOGUES_SECTION, xyzin_isotopologue_section_lines(records))
    return target


def merge_xyzin_isotopologue_records(
    path: Path, records: tuple[XyzinIsotopologueRecord, ...]
) -> Path:
    target = Path(path)
    existing = read_xyzin_isotopologue_records(target) if has_xyzin_isotopologues(target) else ()
    merged = _merge_records(existing, records)
    return write_xyzin_isotopologue_records(target, merged)


def parse_xyzin_isotopologue_records(lines: list[str]) -> tuple[XyzinIsotopologueRecord, ...]:
    records: list[XyzinIsotopologueRecord] = []
    current: dict[str, object] | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        tokens = shlex.split(line)
        if not tokens:
            continue
        keyword = tokens[0].upper()
        if keyword == "SCHEMA":
            schema = tokens[1] if len(tokens) > 1 else ""
            if schema not in SUPPORTED_XYZIN_ISOTOPOLOGUES_SCHEMAS:
                raise ValueError(f"Unsupported xyzin isotopologue schema: {schema!r}")
            continue
        if keyword in {"UNITS", "INDEXING"}:
            continue
        if keyword == "BEGIN":
            if current is not None:
                raise ValueError("Nested BEGIN in #ISOTOPOLOGUES")
            if len(tokens) < 2:
                raise ValueError("BEGIN in #ISOTOPOLOGUES needs a label")
            current = {
                "label": tokens[1],
                "substitutions": {},
                "rotational": None,
                "deltavib": None,
                "deltavib_source": "unspecified",
                "deltavib_convention": "subtract",
                "deltael": None,
                "deltael_source": "unspecified",
                "deltael_convention": "subtract",
                "sigma": None,
                "covariance": None,
            }
            continue
        if keyword == "END":
            if current is None:
                raise ValueError("END without BEGIN in #ISOTOPOLOGUES")
            records.append(_record_from_mapping(current))
            current = None
            continue
        if current is None:
            raise ValueError(f"#ISOTOPOLOGUES entry outside BEGIN/END: {line}")
        if keyword == "DEFINITION":
            definition = " ".join(tokens[1:]).strip()
            current["substitutions"] = (
                {}
                if not definition or definition.lower() == "parent"
                else parse_substitutions(definition)
            )
            continue
        values = _assignment_dict(tokens[1:])
        if keyword == "ROTATIONAL_MHZ":
            current["rotational"] = _abc(values)
        elif keyword == "DELTAVIB_MHZ":
            current["deltavib"] = _abc_default(values)
            current["deltavib_source"] = str(values.get("SOURCE", "unspecified"))
            current["deltavib_convention"] = str(values.get("CONVENTION", "subtract"))
        elif keyword == "DELTAEL_MHZ":
            current["deltael"] = _abc_default(values)
            current["deltael_source"] = str(values.get("SOURCE", "unspecified"))
            current["deltael_convention"] = str(values.get("CONVENTION", "subtract"))
        elif keyword == "SIGMA_MHZ":
            sigmas = _abc(values)
            if any(sigma <= 0.0 for sigma in sigmas):
                raise ValueError("SIGMA_MHZ values in #ISOTOPOLOGUES must be positive")
            current["sigma"] = sigmas
        elif keyword == "COVARIANCE_MHZ2":
            current["covariance"] = _covariance(values)
        else:
            raise ValueError(f"Unsupported #ISOTOPOLOGUES keyword: {keyword}")

    if current is not None:
        raise ValueError("Unclosed BEGIN in #ISOTOPOLOGUES")
    if not records:
        raise ValueError("#ISOTOPOLOGUES contains no records")
    return tuple(records)


def xyzin_isotopologue_section_lines(records: tuple[XyzinIsotopologueRecord, ...]) -> list[str]:
    lines = [
        f"SCHEMA {XYZIN_ISOTOPOLOGUES_SCHEMA}",
        "UNITS ROTATIONAL=MHz DELTAVIB=MHz DELTAEL=MHz SIGMA=MHz COVARIANCE=MHz2",
        "INDEXING ATOMS=ONE_BASED",
    ]
    for record in records:
        label = record.label.strip() or "unnamed"
        definition = (
            format_substitutions(record.substitutions) if record.substitutions else "parent"
        )
        lines.extend(
            [
                f"BEGIN {shlex.quote(label)}",
                f"DEFINITION {definition}",
            ]
        )
        if record.rotational_MHz is not None:
            lines.append(_triple_line("ROTATIONAL_MHZ", record.rotational_MHz))
        if record.deltavib_MHz is not None:
            lines.append(
                _triple_line(
                    "DELTAVIB_MHZ",
                    record.deltavib_MHz,
                    source=record.deltavib_source,
                    convention=record.deltavib_convention,
                )
            )
        if record.deltael_MHz is not None:
            lines.append(
                _triple_line(
                    "DELTAEL_MHZ",
                    record.deltael_MHz,
                    source=record.deltael_source,
                    convention=record.deltael_convention,
                )
            )
        if record.sigma_MHz is not None:
            lines.append(_triple_line("SIGMA_MHZ", record.sigma_MHz))
        if record.covariance_MHz2 is not None:
            lines.append(_covariance_line(record.covariance_MHz2))
        lines.append("END")
    return lines


def _merge_records(
    existing: tuple[XyzinIsotopologueRecord, ...],
    incoming: tuple[XyzinIsotopologueRecord, ...],
) -> tuple[XyzinIsotopologueRecord, ...]:
    ordered: list[XyzinIsotopologueRecord] = list(existing)
    by_label = {record.label: idx for idx, record in enumerate(ordered)}
    for record in incoming:
        idx = by_label.get(record.label)
        if idx is None:
            by_label[record.label] = len(ordered)
            ordered.append(record)
        else:
            ordered[idx] = _merge_record(ordered[idx], record)
    return tuple(ordered)


def _merge_record(
    old: XyzinIsotopologueRecord, new: XyzinIsotopologueRecord
) -> XyzinIsotopologueRecord:
    return XyzinIsotopologueRecord(
        label=new.label or old.label,
        substitutions=new.substitutions,
        rotational_MHz=new.rotational_MHz if new.rotational_MHz is not None else old.rotational_MHz,
        deltavib_MHz=new.deltavib_MHz if new.deltavib_MHz is not None else old.deltavib_MHz,
        deltavib_source=new.deltavib_source
        if new.deltavib_MHz is not None
        else old.deltavib_source,
        deltavib_convention=new.deltavib_convention
        if new.deltavib_MHz is not None
        else old.deltavib_convention,
        deltael_MHz=new.deltael_MHz if new.deltael_MHz is not None else old.deltael_MHz,
        deltael_source=new.deltael_source if new.deltael_MHz is not None else old.deltael_source,
        deltael_convention=new.deltael_convention
        if new.deltael_MHz is not None
        else old.deltael_convention,
        sigma_MHz=new.sigma_MHz if new.sigma_MHz is not None else old.sigma_MHz,
        covariance_MHz2=(
            new.covariance_MHz2
            if new.covariance_MHz2 is not None
            else old.covariance_MHz2
        ),
    )


def _record_from_mapping(item: dict[str, object]) -> XyzinIsotopologueRecord:
    return XyzinIsotopologueRecord(
        label=str(item["label"]),
        substitutions=dict(item.get("substitutions", {})),
        rotational_MHz=item.get("rotational")
        if isinstance(item.get("rotational"), tuple)
        else None,
        deltavib_MHz=item.get("deltavib") if isinstance(item.get("deltavib"), tuple) else None,
        deltavib_source=str(item.get("deltavib_source", "unspecified")),
        deltavib_convention=str(item.get("deltavib_convention", "subtract")),
        deltael_MHz=item.get("deltael") if isinstance(item.get("deltael"), tuple) else None,
        deltael_source=str(item.get("deltael_source", "unspecified")),
        deltael_convention=str(item.get("deltael_convention", "subtract")),
        sigma_MHz=item.get("sigma") if isinstance(item.get("sigma"), tuple) else None,
        covariance_MHz2=(
            item.get("covariance")
            if isinstance(item.get("covariance"), tuple)
            else None
        ),
    )


def _assignment_dict(tokens: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Expected KEY=VALUE assignment, got {token!r}")
        key, value = token.split("=", 1)
        values[key.strip().upper()] = value.strip()
    return values


def _abc(values: dict[str, str]) -> tuple[float, float, float]:
    return float(values["A"]), float(values["B"]), float(values["C"])


def _covariance(values: dict[str, str]) -> tuple[tuple[float, float, float], ...]:
    keys = (("AA", "AB", "AC"), ("AB", "BB", "BC"), ("AC", "BC", "CC"))
    return tuple(tuple(float(values[key]) for key in row) for row in keys)


def _covariance_line(covariance: tuple[tuple[float, float, float], ...]) -> str:
    names = (("AA", 0, 0), ("AB", 0, 1), ("AC", 0, 2), ("BB", 1, 1), ("BC", 1, 2), ("CC", 2, 2))
    fields = " ".join(f"{name}={covariance[row][column]:.12g}" for name, row, column in names)
    return f"COVARIANCE_MHZ2 {fields}"


def _validate_covariance(add, label: str, covariance) -> None:
    if covariance is None:
        return
    if len(covariance) != 3 or any(len(row) != 3 for row in covariance):
        add("error", "invalid_covariance_shape", "COVARIANCE_MHZ2 must be 3 by 3.", f"label={label}")
        return
    if any(not math.isfinite(float(value)) for row in covariance for value in row):
        add("error", "invalid_covariance_value", "COVARIANCE_MHZ2 must be finite.", f"label={label}")
        return
    scale = max(1.0, max(abs(float(value)) for row in covariance for value in row))
    if any(abs(float(covariance[i][j]) - float(covariance[j][i])) > 1.0e-10 * scale for i in range(3) for j in range(3)):
        add("error", "asymmetric_covariance", "COVARIANCE_MHZ2 must be symmetric.", f"label={label}")
    if any(float(covariance[i][i]) <= 0.0 for i in range(3)):
        add("error", "invalid_covariance_diagonal", "COVARIANCE_MHZ2 diagonal must be positive.", f"label={label}")


def _abc_default(values: dict[str, str]) -> tuple[float, float, float]:
    return float(values.get("A", 0.0)), float(values.get("B", 0.0)), float(values.get("C", 0.0))


def _triple_line(
    keyword: str,
    values: tuple[float, float, float],
    *,
    source: str | None = None,
    convention: str | None = None,
) -> str:
    line = f"{keyword} A={values[0]:.12g} B={values[1]:.12g} C={values[2]:.12g}"
    if source is not None:
        line += f" SOURCE={shlex.quote(source or 'unspecified')}"
    if convention is not None:
        line += f" CONVENTION={shlex.quote(convention or 'subtract')}"
    return line


def _validate_optional_triple(
    add,
    label: str,
    code: str,
    keyword: str,
    values: tuple[float, float, float] | None,
    *,
    positive: bool = False,
) -> None:
    if values is None:
        return
    for component, value in zip(("A", "B", "C"), values):
        if not math.isfinite(float(value)):
            add(
                "error",
                f"nonfinite_{code}",
                f"{keyword} values must be finite.",
                f"label={label};component={component};value={value}",
            )
        elif positive and float(value) <= 0.0:
            add(
                "error",
                f"nonpositive_{code}",
                f"{keyword} values must be positive.",
                f"label={label};component={component};value={value}",
            )


def _validate_correction_convention(add, label: str, keyword: str, convention: str) -> None:
    text = str(convention or "subtract").strip().lower().replace("-", "_")
    allowed = {
        "subtract",
        "subtractive",
        "observed_minus_delta",
        "b0_minus_delta",
        "add",
        "additive",
        "observed_plus_delta",
        "b0_plus_delta",
        "msr",
    }
    if text not in allowed:
        add(
            "error",
            "invalid_correction_convention",
            f"{keyword} correction convention is not supported.",
            f"label={label};convention={convention}",
        )
