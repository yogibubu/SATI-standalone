"""Canonical capability registry for SONIC primitive B-prime kernels."""

from __future__ import annotations

from dataclasses import dataclass


B_PRIME_CAPABILITY_SCHEMA = "matrix.smith.b_prime_capabilities.v1"
B_PRIME_ANALYTIC = "ANALYTIC"
B_PRIME_ANALYTIC_ZERO = "ANALYTIC_ZERO"
B_PRIME_NUMERICAL_AUDIT = "NUMERICAL_AUDIT"


@dataclass(frozen=True)
class PrimitiveBPrimeCapability:
    primitive_id: str
    function: str
    mode: str
    reason: str
    schema: str = B_PRIME_CAPABILITY_SCHEMA

    @property
    def analytic(self) -> bool:
        return self.mode in {B_PRIME_ANALYTIC, B_PRIME_ANALYTIC_ZERO}


_ANALYTIC_FUNCTIONS = frozenset(
    {
        "R", "A", "D", "U", "H", "L", "FC_DIST", "FCA_DIST",
        "CENTER_ATOM_DIST", "RPCB", "RPCK", "IMPD", "FROT",
    }
)
_NUMERICAL_AUDIT_FUNCTIONS = frozenset({"FLIN_TRANS", "RPU", "FAXIS"})


def primitive_b_prime_capability(primitive) -> PrimitiveBPrimeCapability:
    function = str(primitive.function).strip().upper()
    identifier = str(primitive.identifier)
    if function in {"SOOP", "SYCART"}:
        return PrimitiveBPrimeCapability(
            identifier, function, B_PRIME_ANALYTIC_ZERO,
            "frozen-reference SOOP is affine in Cartesian coordinates",
        )
    if function == "FTRANS":
        if tuple(getattr(primitive, "ref_frame_atoms", ())):
            return PrimitiveBPrimeCapability(
                identifier, function, B_PRIME_ANALYTIC,
                "second-order AD of fragment-center translation in its anchored reference frame",
            )
        return PrimitiveBPrimeCapability(
            identifier, function, B_PRIME_ANALYTIC_ZERO,
            "laboratory-frame fragment translation is affine",
        )
    if function in _ANALYTIC_FUNCTIONS:
        return PrimitiveBPrimeCapability(
            identifier, function, B_PRIME_ANALYTIC,
            "resident second-order forward-AD kernel",
        )
    if function in _NUMERICAL_AUDIT_FUNCTIONS:
        return PrimitiveBPrimeCapability(
            identifier, function, B_PRIME_NUMERICAL_AUDIT,
            "explicitly inventoried numerical audit kernel",
        )
    return PrimitiveBPrimeCapability(
        identifier, function, B_PRIME_NUMERICAL_AUDIT,
        "unregistered primitive family; fail-closed production capability",
    )


def audit_b_prime_capabilities(definition) -> tuple[PrimitiveBPrimeCapability, ...]:
    return tuple(primitive_b_prime_capability(item) for item in definition.primitives)


ANALYTIC_B_PRIME_FAMILIES = frozenset(
    _ANALYTIC_FUNCTIONS | {"SOOP", "SYCART", "FTRANS"}
)


__all__ = [
    "ANALYTIC_B_PRIME_FAMILIES",
    "B_PRIME_ANALYTIC",
    "B_PRIME_ANALYTIC_ZERO",
    "B_PRIME_CAPABILITY_SCHEMA",
    "B_PRIME_NUMERICAL_AUDIT",
    "PrimitiveBPrimeCapability",
    "audit_b_prime_capabilities",
    "primitive_b_prime_capability",
]
