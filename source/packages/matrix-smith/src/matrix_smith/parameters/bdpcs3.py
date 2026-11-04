from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Bdpcs3HbondParameters:
    angle_threshold_deg: float
    distance_cutoff_ang: float
    distance_width_ang: float
    search_cutoff_ang: float
    donor_atomic_numbers: tuple[int, ...]
    acceptor_atomic_numbers: tuple[int, ...]
    average_mixed_n_o: bool
    corrections_ang: dict[tuple[int, int], float]

    def correction_ang(self, donor_z: int, acceptor_z: int) -> float:
        key = (int(donor_z), int(acceptor_z))
        if key in self.corrections_ang:
            return self.corrections_ang[key]
        if self.average_mixed_n_o and key[0] in (7, 8) and key[1] in (7, 8):
            donor_ref = self.corrections_ang.get((key[0], key[0]))
            acceptor_ref = self.corrections_ang.get((key[1], key[1]))
            if donor_ref is not None and acceptor_ref is not None:
                return 0.5 * (donor_ref + acceptor_ref)
        return 0.0


@dataclass(frozen=True)
class Bdpcs3WeightParameters:
    profile: str
    stretch: float
    angle: float
    hbond: float
    torsion_min: float
    fragment: float


@dataclass(frozen=True)
class Bdpcs3Parameters:
    hbond: Bdpcs3HbondParameters
    weights: Bdpcs3WeightParameters


def _default_path() -> Path:
    return Path(__file__).with_name("bdpcs3_hbond.toml")


def _parse_pair_key(key: str) -> tuple[int, int]:
    left, right = key.split("-", 1)
    return int(left), int(right)


@lru_cache(maxsize=1)
def load_bdpcs3_parameters(path: str | Path | None = None) -> Bdpcs3Parameters:
    param_path = Path(path) if path is not None else _default_path()
    data = tomllib.loads(param_path.read_text(encoding="utf-8"))
    hbond = data["hbond"]
    corrections = {
        _parse_pair_key(key): float(value)
        for key, value in hbond.get("corrections_ang", {}).items()
    }
    return Bdpcs3Parameters(
        hbond=Bdpcs3HbondParameters(
            angle_threshold_deg=float(hbond["angle_threshold_deg"]),
            distance_cutoff_ang=float(hbond["distance_cutoff_ang"]),
            distance_width_ang=float(hbond["distance_width_ang"]),
            search_cutoff_ang=float(hbond["search_cutoff_ang"]),
            donor_atomic_numbers=tuple(int(z) for z in hbond["donor_atomic_numbers"]),
            acceptor_atomic_numbers=tuple(int(z) for z in hbond["acceptor_atomic_numbers"]),
            average_mixed_n_o=bool(hbond.get("average_mixed_n_o", True)),
            corrections_ang=corrections,
        ),
        weights=Bdpcs3WeightParameters(
            profile=str(data["weights"]["profile"]),
            stretch=float(data["weights"]["stretch"]),
            angle=float(data["weights"]["angle"]),
            hbond=float(data["weights"]["hbond"]),
            torsion_min=float(data["weights"]["torsion_min"]),
            fragment=float(data["weights"]["fragment"]),
        ),
    )
