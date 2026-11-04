"""Stewart STO-nG expansions used by the GFN2 minimal basis."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json

from .basis import GaussianPrimitive


_TYPE_INDEX = {
    (1, 0): 0, (2, 0): 1, (3, 0): 2, (4, 0): 3, (5, 0): 4,
    (2, 1): 5, (3, 1): 6, (4, 1): 7, (5, 1): 8,
    (3, 2): 9, (4, 2): 10, (5, 2): 11,
}


@lru_cache(maxsize=1)
def _tables() -> dict[str, object]:
    resource = files("matrix_gfn2").joinpath("data/sto-ng.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def slater_to_gaussians(
    gaussian_count: int, principal: int, angular: int, slater_exponent: float
) -> tuple[GaussianPrimitive, ...]:
    """Expand an STO into normalized-primitives-compatible GTO coefficients."""

    if gaussian_count not in (3, 4, 6):
        raise ValueError("GFN2 supports STO-3G, STO-4G, or STO-6G expansions")
    if slater_exponent <= 0.0:
        raise ValueError("Slater exponent must be positive")
    tables = _tables()
    if principal == 6 and gaussian_count == 6 and angular in (0, 1):
        suffix = "s" if angular == 0 else "p"
        alpha = tables[f"pAlpha6{suffix}"]
        coefficients = tables[f"pCoeff6{suffix}"]
    else:
        try:
            index = _TYPE_INDEX[(principal, angular)]
        except KeyError as exc:
            raise ValueError("unsupported Slater principal/angular combination") from exc
        group = tables[str(gaussian_count)]
        alpha = group["alpha"][index]
        coefficients = group["coeff"][index]
    return tuple(
        GaussianPrimitive(float(exponent) * slater_exponent**2, float(coefficient))
        for exponent, coefficient in zip(alpha, coefficients, strict=True)
    )


def shell_quantum_numbers(label: str) -> tuple[int, int]:
    angular = {"s": 0, "p": 1, "d": 2}
    try:
        return int(label[:-1]), angular[label[-1]]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid GFN2 shell label: {label}") from exc


__all__ = ["shell_quantum_numbers", "slater_to_gaussians"]
