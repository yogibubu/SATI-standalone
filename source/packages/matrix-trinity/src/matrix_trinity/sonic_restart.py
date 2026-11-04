"""Versioned, checked restart for a complete SONIC GVPT2 numerical field."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .curvilinear_vpt2 import SonicAcquiredField, SonicVPT2Input, _validate_field


SONIC_GVPT2_RESTART_SCHEMA = "matrix.trinity.sonic-gvpt2-field.v1"
_ARRAYS = (
    "harmonic_sonic", "cubic_sonic", "quartic_sonic", "metric_sonic",
    "metric_first_sonic", "metric_second_sonic",
)


def _digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    return sha256(str(contiguous.shape).encode() + contiguous.tobytes()).hexdigest()


def write_sonic_gvpt2_restart(path: Path | str, acquired: SonicAcquiredField) -> Path:
    """Atomically persist the complete field, acquisition audit and checksums."""
    if acquired.acquisition.get("quartic_coverage") != "complete F_ijkl":
        raise ValueError("GVPT2 restart requires complete quartic coverage")
    arrays = _validate_field(acquired.field)
    payload = {name: value for name, value in zip(_ARRAYS, arrays)}
    header = {
        "schema": SONIC_GVPT2_RESTART_SCHEMA,
        "pseudopotential_cm": float(acquired.field.pseudopotential_cm),
        "acquisition": acquired.acquisition,
        "checksums": {name: _digest(value) for name, value in payload.items()},
    }
    target = Path(path)
    with NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".npz",
                            delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary, header=np.array(json.dumps(header, sort_keys=True)),
                                **payload)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, target)
    return target


def read_sonic_gvpt2_restart(path: Path | str) -> SonicAcquiredField:
    """Load a checked restart without pickle or executable data."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {*_ARRAYS, "header"}:
            raise ValueError("SONIC GVPT2 restart has unexpected fields")
        header = json.loads(str(archive["header"].item()))
        if header.get("schema") != SONIC_GVPT2_RESTART_SCHEMA:
            raise ValueError("unsupported SONIC GVPT2 restart schema")
        arrays = {name: np.asarray(archive[name], dtype=float) for name in _ARRAYS}
    for name, value in arrays.items():
        if header["checksums"].get(name) != _digest(value):
            raise ValueError(f"SONIC GVPT2 restart checksum mismatch for {name}")
    field = SonicVPT2Input(
        **arrays, pseudopotential_cm=float(header["pseudopotential_cm"]),
        quartic_coverage="complete" if header["acquisition"].get("quartic_coverage")
        == "complete F_ijkl" else "semidiagonal",
    )
    _validate_field(field)
    acquisition = header["acquisition"]
    if acquisition.get("quartic_coverage") != "complete F_ijkl":
        raise ValueError("SONIC GVPT2 restart lacks complete quartic coverage")
    return SonicAcquiredField(field, acquisition)


__all__ = ["SONIC_GVPT2_RESTART_SCHEMA", "write_sonic_gvpt2_restart", "read_sonic_gvpt2_restart"]
