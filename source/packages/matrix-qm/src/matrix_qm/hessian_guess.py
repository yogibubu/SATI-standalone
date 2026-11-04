"""Canonical lower-level Hessian guess for a final LINK optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class HessianGuessLevel:
    """Electronic-structure level used only for the initial Hessian guess."""

    target_family: str
    level: str
    method: str
    basis: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_COUPLED_CLUSTER = re.compile(r"^(?:DLPNO)?CC(?:2|3|SD|SDT|SDTQ)")
_DOUBLE_HYBRID_MARKERS = (
    "DOUBLEHYBRID",
    "B2PLYP",
    "B2GPPLYP",
    "DSDPBEP86",
    "REVDSDPBEP86",
    "PWPB95",
    "XYG3",
)
_HYBRID_MARKERS = (
    "B3LYP",
    "PBE0",
    "PBE1PBE",
    "BHLYP",
    "CAMB3LYP",
    "M062X",
    "TPSSH",
    "WB97",
)


def default_initial_hessian_level(
    final_method: str,
    final_basis: str = "",
) -> HessianGuessLevel | None:
    """Return the canonical ``L(i-1)`` Hessian level for final method ``Li``.

    The policy is deliberately method-based rather than backend-based.  LINK
    therefore obtains the same initial curvature model with every final
    electronic-structure program.
    """

    method = str(final_method).strip()
    compact = re.sub(r"[^A-Z0-9]", "", method.upper())
    basis = str(final_basis).strip()
    if not compact:
        return None
    if compact in {"L3", "L4", "FL4", "PL4"} or _COUPLED_CLUSTER.search(compact):
        return HessianGuessLevel(
            target_family="coupled_cluster",
            level="L2",
            method="MP2",
            basis=basis or "def2-TZVP",
        )
    if (
        compact in {"L2", "DL2", "MP2", "RMP2", "UMP2"}
        or "MP2" in compact
        or any(marker in compact for marker in _DOUBLE_HYBRID_MARKERS)
    ):
        return HessianGuessLevel(
            target_family="double_hybrid_or_mp2",
            level="L1",
            method="PBE0",
            basis="def2-SVP",
        )
    if compact == "L1" or any(marker in compact for marker in _HYBRID_MARKERS):
        return HessianGuessLevel(
            target_family="hybrid_dft",
            level="L0",
            method="HF",
            basis="STO-3G",
        )
    return None


__all__ = ["HessianGuessLevel", "default_initial_hessian_level"]
