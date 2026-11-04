from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from .definition import write_sonic_build_sections_from_cartesian
from .models import GICDefinition
from .runtime.gicforge_python import LocalSALCSettings


SMITH_INPUT_SCHEMA = "matrix.smith.input.v1"


def write_smith_build_sections_from_input(
    input_path: Path,
    output: Path | None = None,
) -> GICDefinition:
    """Build a SONIC coordinate contract from one SMITH extended-XYZ input file."""
    source = Path(input_path)
    xyz_lines, options = _read_smith_input(source)
    target = Path(output) if output is not None else source.with_suffix(".xyzin")
    source_kind = _normalized_source_kind(options.get("source_kind", "auto"))
    local_salc_settings = LocalSALCSettings(
        zeff_tolerance=_optional_float(options.get("local_zeff_tolerance"), 5.0e-4),
        distance_tolerance_angstrom=_optional_float(
            options.get("local_distance_tolerance"), 1.0e-3
        ),
        template_rms_threshold=_optional_float(
            options.get("local_template_rms_threshold"), 0.12
        ),
        template_min_margin=_optional_float(options.get("local_template_margin"), 0.02),
        angle_class_tolerance=_optional_float(
            options.get("local_angle_class_tolerance"), 0.02
        ),
    )
    with tempfile.TemporaryDirectory(prefix="smith-") as scratch:
        geometry_source = Path(scratch) / "geometry.xyz"
        geometry_source.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
        regime = str(options.get("task_regime", "minimum")).strip().lower().replace("-", "_")
        if regime == "reaction_path":
            from .reaction_path import write_reaction_path_build_sections_from_cartesian

            group = _optional_string(options.get("symmetry_group"))
            if not group:
                raise ValueError("reaction_path requires an a priori symmetry_group")
            if "symmetrize" in options and not options["symmetrize"]:
                raise ValueError("reaction_path requires symmetry-adapted coordinates")
            return write_reaction_path_build_sections_from_cartesian(
                geometry_source, target, symmetry_group=group,
                source_kind=source_kind, sycart=bool(options.get("sycart", False)),
            )
        return write_sonic_build_sections_from_cartesian(
            geometry_source,
            target,
            source_kind=source_kind,
            symmetrize=bool(options.get("symmetrize", False)),
            sycart=bool(options.get("sycart", False)),
            symmetry_group=_optional_string(options.get("symmetry_group")),
            improper_dihedrals=False,
            fragment_mode=_optional_string(options.get("fragment_mode")),
            xh_stretch_policy=_optional_string(options.get("xh_stretch_policy")),
            local_xh_bonds=_pairs(options.get("local_xh_bonds")),
            local_xh_classes=_strings(options.get("local_xh_classes")),
            local_salc=bool(options.get("local_salc", False)),
            local_salc_settings=local_salc_settings,
        )


def _read_smith_input(source: Path) -> tuple[list[str], dict[str, Any]]:
    raw_lines = source.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) < 2:
        raise ValueError("SMITH input must start with a standard XYZ block")
    try:
        atom_count = int(raw_lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("SMITH input first line must be an XYZ atom count") from exc
    xyz_end = atom_count + 2
    if len(raw_lines) < xyz_end:
        raise ValueError("SMITH input XYZ block is shorter than the atom count")
    xyz_lines = [line.rstrip() for line in raw_lines[:xyz_end]]
    options = _parse_directives(raw_lines[xyz_end:])
    schema = options.pop("schema", None)
    if schema is not None and str(schema).strip() != SMITH_INPUT_SCHEMA:
        raise ValueError(f"unsupported SMITH input schema {schema!r}")
    return xyz_lines, options


def _parse_directives(lines: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for raw in lines:
        text = raw.strip()
        if not text or text.upper() in {"#SMITH", "#SONIC", "$SMITH", "$SONIC"}:
            continue
        if text.startswith("#"):
            text = text[1:].strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
        else:
            parts = text.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        options[key.strip().lower().replace("-", "_")] = _parse_value(value)
    return options


def _parse_value(text: str) -> object:
    value = text.strip().strip("\"'")
    lower = value.lower()
    if lower in {"true", "yes", "on", "1"}:
        return True
    if lower in {"false", "no", "off", "0"}:
        return False
    if "," in value:
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_source_kind(value: object) -> str:
    text = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "gaussian_cartesian_input": "auto",
        "gaussian_cartesian": "auto",
        "cartesian_input": "auto",
    }
    return aliases.get(text, text)


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _optional_float(value: object, default: float) -> float:
    return default if value is None else float(value)


def _strings(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())  # type: ignore[union-attr]


def _pairs(value: object) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    pairs = []
    if isinstance(value, str):
        value = tuple(item.strip() for item in value.split(",") if item.strip())
    for item in value:  # type: ignore[union-attr]
        if isinstance(item, str) and "-" in item:
            left, right = item.split("-", 1)
            pairs.append((int(left), int(right)))
            continue
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("local_xh_bonds must be a list of two-integer pairs")
        pairs.append((int(item[0]), int(item[1])))
    return tuple(pairs)
