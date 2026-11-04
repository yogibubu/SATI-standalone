"""Single ORACLE perception and frozen-coordinate handoff for KEYMAKER."""

from __future__ import annotations

from pathlib import Path

from matrix_chem import read_oracle_coordinate_atlas_contract, read_oracle_sonic_contract

from .api import analyze_structure
from .sonic_contract_builder import write_oracle_sonic_contract_from_xyzin


def analyze_and_freeze(source: Path | str, output: Path | str) -> Path:
    """Perceive once and validate both contracts consumed by SMITH."""

    result = Path(output).expanduser().resolve()
    analyze_structure(Path(source).expanduser().resolve(), result)
    write_oracle_sonic_contract_from_xyzin(result)
    read_oracle_sonic_contract(result)
    read_oracle_coordinate_atlas_contract(result)
    return result
