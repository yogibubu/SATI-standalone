"""Replace an explicit Cartesian geometry without rewriting approved job options."""

from pathlib import Path
import re

import numpy as np


def write_gaussian_cartesian_template(template, output, coordinates_angstrom):
    """Preserve all template lines except its ordinary four-column coordinates.

    Supports ExtraOverlays because the geometry is identified by its explicit
    charge/multiplicity line, not by counting blank-line sections. Atom labels,
    Link0 resources, routes, overlays and trailing input remain byte-for-byte
    unchanged. One job and one Cartesian geometry are required. Jobs in separate
    directories may keep the same relative checkpoint name. No keyword, symmetry
    projection or coordinate alignment is introduced.
    """
    source, target = Path(template), Path(output)
    if source.resolve() == target.resolve():
        raise ValueError("The approved template must not be overwritten")
    text = source.read_text()
    if re.search(r"^\s*--link1--\s*$", text, re.MULTILINE | re.IGNORECASE):
        raise ValueError("Cartesian template replacement supports one job only")
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*[+-]?\d+\s+\d+\s*", line)]
    if len(matches) != 1:
        raise ValueError("Template requires one explicit charge/multiplicity line")
    start = matches[0] + 1
    stop = start
    labels = []
    while stop < len(lines) and lines[stop].strip():
        fields = lines[stop].split()
        if len(fields) != 4:
            raise ValueError("Template geometry requires atom and three Cartesian values")
        try:
            [float(value.replace("D", "E").replace("d", "e")) for value in fields[1:]]
        except ValueError as error:
            raise ValueError("Template geometry is not Cartesian") from error
        labels.append(fields[0])
        stop += 1
    raw = np.asarray(coordinates_angstrom)
    if np.iscomplexobj(raw):
        raise ValueError("Cartesian coordinates must be real")
    coords = np.asarray(raw, float)
    if not labels or coords.shape != (len(labels), 3) or not np.all(np.isfinite(coords)):
        raise ValueError("New coordinates must match the template atom count")
    replacement = [
        f"{label} {x:.16f} {y:.16f} {z:.16f}\n"
        for label, (x, y, z) in zip(labels, coords, strict=True)
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(lines[:start] + replacement + lines[stop:]))
    return target
