#!/usr/bin/env python3
"""Cheap deterministic Cartesian PES used only for protocol verification."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    xyz_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    point_index = int(sys.argv[3])
    coordinates = [
        float(value)
        for line in xyz_path.read_text(encoding="utf-8").splitlines()[2:]
        for value in line.split()[1:4]
    ]
    # A positive quadratic Cartesian surface. Its purpose is to make every
    # exchanged E/G/H value exactly reproducible, not to model water accuracy.
    force = 0.02
    coordinates_bohr = [value / 0.529177210903 for value in coordinates]
    energy = -76.0 + 0.5 * force * sum(value * value for value in coordinates_bohr)
    gradient = [force * value for value in coordinates_bohr]
    hessian = [
        [force if row == column else 0.0 for column in range(len(coordinates))]
        for row in range(len(coordinates))
    ]
    payload = {
        "schema": "oracle.link.point_result.v1",
        "point_index": point_index,
        "displacement": 0.0,
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": gradient,
        "hessian_hartree_per_bohr2": hessian,
        "status": "completed",
        "source": "water-2d-analytic-pes",
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
