"""Table-driven Hermitian RAM36-style torsion--rotation operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np


RAM36_OPERATOR_TERM_SCHEMA = "matrix.lock.ram36_operator_term.v1"
RAM36_OPERATOR_PARAMETER_SCHEMA = "matrix.lock.ram36_operator_parameter.v1"


@dataclass(frozen=True)
class RAM36OperatorTerm:
    """One coefficient in the general RAM36 operator expansion.

    The integer tuple follows the published ``k,n,p,q,r,s,t`` convention for
    ``J^(2k) Ja^n Jb^p Jc^q p_alpha^r cos(3s alpha)`` or its sine analogue.
    LOCK forms the half-sum with the reverse-ordered adjoint, so the resulting
    matrix is Hermitian by construction.
    """

    coefficient_MHz: float
    k: int = 0
    n: int = 0
    p: int = 0
    q: int = 0
    r: int = 0
    s: int = 0
    t: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        indices = (self.k, self.n, self.p, self.q, self.r, self.s, self.t)
        if any(int(value) != value or value < 0 for value in indices):
            raise ValueError("RAM36 operator indices must be nonnegative integers")
        if self.s and self.t:
            raise ValueError("a RAM36 term cannot contain both cosine and sine harmonics")
        if not math.isfinite(float(self.coefficient_MHz)):
            raise ValueError("RAM36 operator coefficient must be finite")
        if (self.n + self.p + self.q + self.r) % 2:
            raise ValueError("RAM36 operator must be invariant under time reversal")

    @property
    def momentum_order(self) -> int:
        return int(2 * self.k + self.n + self.p + self.q + self.r)

    @property
    def canonical_id(self) -> str:
        return f"{self.k},{self.n},{self.p},{self.q},{self.r},{self.s},{self.t}"

    def allowed_by(self, symmetry: "RAM36SymmetryProfile") -> bool:
        return symmetry.allows(self)

    def to_dict(self) -> dict[str, object]:
        return {"schema": RAM36_OPERATOR_TERM_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RAM36OperatorTerm":
        values = dict(payload)
        schema = values.pop("schema", RAM36_OPERATOR_TERM_SCHEMA)
        if schema != RAM36_OPERATOR_TERM_SCHEMA:
            raise ValueError(f"unsupported RAM36 operator schema: {schema}")
        return cls(**values)


@dataclass(frozen=True)
class RAM36SymmetryProfile:
    """Parity action of one molecular-symmetry generator on RAM coordinates."""

    name: str
    Ja: int
    Jb: int
    Jc: int
    p_alpha: int
    cosine: int = 1
    sine: int = -1

    def __post_init__(self) -> None:
        if not self.name.strip() or any(
            value not in (-1, 1)
            for value in (self.Ja, self.Jb, self.Jc, self.p_alpha, self.cosine, self.sine)
        ):
            raise ValueError("RAM36 symmetry profiles require a name and +/-1 parities")

    def allows(self, term: RAM36OperatorTerm) -> bool:
        parity = (
            self.Ja**term.n
            * self.Jb**term.p
            * self.Jc**term.q
            * self.p_alpha**term.r
            * (self.cosine if term.s else 1)
            * (self.sine if term.t else 1)
        )
        return parity == 1


RAM36_CS_AB = RAM36SymmetryProfile(
    name="Cs(ab)", Ja=-1, Jb=-1, Jc=1, p_alpha=-1
)


@dataclass(frozen=True)
class RAM36OperatorContribution:
    """One dimensionless monomial multiplier in a shared RAM parameter."""

    multiplier: float = 1.0
    k: int = 0
    n: int = 0
    p: int = 0
    q: int = 0
    r: int = 0
    s: int = 0
    t: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.multiplier)):
            raise ValueError("RAM36 contribution multiplier must be finite")
        RAM36OperatorTerm(
            0.0, k=self.k, n=self.n, p=self.p, q=self.q, r=self.r, s=self.s, t=self.t
        )

    @property
    def canonical_id(self) -> str:
        return f"{self.k},{self.n},{self.p},{self.q},{self.r},{self.s},{self.t}"

    def as_term(self, parameter_value_MHz: float, *, label: str = "") -> RAM36OperatorTerm:
        return RAM36OperatorTerm(
            float(parameter_value_MHz) * float(self.multiplier),
            k=self.k,
            n=self.n,
            p=self.p,
            q=self.q,
            r=self.r,
            s=self.s,
            t=self.t,
            label=label,
        )


@dataclass(frozen=True)
class RAM36OperatorParameter:
    """A refinable coefficient shared by one or more RAM36 monomials."""

    name: str
    value_MHz: float
    contributions: tuple[RAM36OperatorContribution, ...]
    source_operator: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not math.isfinite(float(self.value_MHz)):
            raise ValueError("RAM36 parameters require a name and finite value")
        if not self.contributions:
            raise ValueError("RAM36 parameters require at least one contribution")
        if len({item.canonical_id for item in self.contributions}) != len(self.contributions):
            raise ValueError("duplicate monomials in one RAM36 parameter")

    @property
    def derivative_name(self) -> str:
        return f"ram36:{self.name}"

    def terms(self) -> tuple[RAM36OperatorTerm, ...]:
        return tuple(
            item.as_term(self.value_MHz, label=self.name) for item in self.contributions
        )

    def derivative_terms(self) -> tuple[RAM36OperatorTerm, ...]:
        return tuple(item.as_term(1.0, label=self.name) for item in self.contributions)

    def with_value(self, value_MHz: float) -> "RAM36OperatorParameter":
        return replace(self, value_MHz=float(value_MHz))

    def allowed_by(self, symmetry: RAM36SymmetryProfile) -> bool:
        return all(term.allowed_by(symmetry) for term in self.derivative_terms())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RAM36_OPERATOR_PARAMETER_SCHEMA,
            "name": self.name,
            "value_MHz": self.value_MHz,
            "source_operator": self.source_operator,
            "contributions": [asdict(item) for item in self.contributions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RAM36OperatorParameter":
        values = dict(payload)
        schema = values.pop("schema", RAM36_OPERATOR_PARAMETER_SCHEMA)
        if schema != RAM36_OPERATOR_PARAMETER_SCHEMA:
            raise ValueError(f"unsupported RAM36 parameter schema: {schema}")
        raw = values.pop("contributions")
        if not isinstance(raw, list):
            raise ValueError("RAM36 parameter contributions must be a list")
        return cls(
            contributions=tuple(RAM36OperatorContribution(**item) for item in raw),
            **values,
        )


def _matrix_power(operator: np.ndarray, exponent: int) -> np.ndarray:
    return np.linalg.matrix_power(operator, int(exponent))


def _torsional_harmonic(size: int, *, cosine: int, sine: int) -> np.ndarray:
    harmonic = int(cosine or sine)
    result = np.eye(size, dtype=complex) if harmonic == 0 else np.zeros((size, size), complex)
    if harmonic == 0:
        return result
    for left in range(size - harmonic):
        right = left + harmonic
        if cosine:
            result[left, right] = 0.5
            result[right, left] = 0.5
        else:
            result[left, right] = 0.5j
            result[right, left] = -0.5j
    return result


def ram36_factorized_terms(
    term: RAM36OperatorTerm,
    *,
    torsional_momentum: np.ndarray,
    angular_momenta: Sequence[np.ndarray],
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    """Return the two factorized halves of one Hermitian RAM36 term."""

    if len(angular_momenta) != 3:
        raise ValueError("RAM36 assembly requires Ja, Jb, and Jc")
    Ja, Jb, Jc = (np.asarray(value, dtype=complex) for value in angular_momenta)
    if Ja.shape != Jb.shape or Ja.shape != Jc.shape or Ja.ndim != 2 or Ja.shape[0] != Ja.shape[1]:
        raise ValueError("RAM36 angular-momentum matrices must share one square shape")
    torsional_momentum = np.asarray(torsional_momentum, dtype=complex)
    if torsional_momentum.ndim != 2 or torsional_momentum.shape[0] != torsional_momentum.shape[1]:
        raise ValueError("RAM36 torsional momentum must be square")

    J2 = Ja @ Ja + Jb @ Jb + Jc @ Jc
    rotational_left = (
        _matrix_power(J2, term.k)
        @ _matrix_power(Ja, term.n)
        @ _matrix_power(Jb, term.p)
        @ _matrix_power(Jc, term.q)
    )
    rotational_right = (
        _matrix_power(Jc, term.q)
        @ _matrix_power(Jb, term.p)
        @ _matrix_power(Ja, term.n)
        @ _matrix_power(J2, term.k)
    )
    torsional_function = _torsional_harmonic(
        torsional_momentum.shape[0], cosine=term.s, sine=term.t
    )
    momentum_power = _matrix_power(torsional_momentum, term.r)
    torsional_left = momentum_power @ torsional_function
    torsional_right = torsional_function @ momentum_power
    coefficient = 0.5 * float(term.coefficient_MHz)
    return (
        (coefficient, torsional_left, rotational_left),
        (coefficient, torsional_right, rotational_right),
    )


def ram36_factorized_terms_many(
    terms: Iterable[RAM36OperatorTerm],
    *,
    torsional_momentum: np.ndarray,
    angular_momenta: Sequence[np.ndarray],
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    """Build many RAM36 terms while caching every repeated matrix power."""

    terms = tuple(terms)
    if not terms:
        return ()
    if len(angular_momenta) != 3:
        raise ValueError("RAM36 assembly requires Ja, Jb, and Jc")
    Ja, Jb, Jc = (np.asarray(value, dtype=complex) for value in angular_momenta)
    torsional_momentum = np.asarray(torsional_momentum, dtype=complex)
    J2 = Ja @ Ja + Jb @ Jb + Jc @ Jc

    def powers(operator: np.ndarray, exponents: set[int]) -> dict[int, np.ndarray]:
        return {value: _matrix_power(operator, value) for value in exponents}

    J2p = powers(J2, {term.k for term in terms})
    Jap = powers(Ja, {term.n for term in terms})
    Jbp = powers(Jb, {term.p for term in terms})
    Jcp = powers(Jc, {term.q for term in terms})
    pp = powers(torsional_momentum, {term.r for term in terms})
    harmonics = {
        (term.s, term.t): _torsional_harmonic(
            torsional_momentum.shape[0], cosine=term.s, sine=term.t
        )
        for term in terms
    }
    factors = []
    for term in terms:
        rotational_left = J2p[term.k] @ Jap[term.n] @ Jbp[term.p] @ Jcp[term.q]
        rotational_right = Jcp[term.q] @ Jbp[term.p] @ Jap[term.n] @ J2p[term.k]
        function = harmonics[(term.s, term.t)]
        torsional_left = pp[term.r] @ function
        torsional_right = function @ pp[term.r]
        coefficient = 0.5 * float(term.coefficient_MHz)
        factors.extend(
            (
                (coefficient, torsional_left, rotational_left),
                (coefficient, torsional_right, rotational_right),
            )
        )
    return tuple(factors)


__all__ = [
    "RAM36_OPERATOR_TERM_SCHEMA",
    "RAM36_OPERATOR_PARAMETER_SCHEMA",
    "RAM36_CS_AB",
    "RAM36OperatorContribution",
    "RAM36OperatorParameter",
    "RAM36OperatorTerm",
    "RAM36SymmetryProfile",
    "ram36_factorized_terms",
    "ram36_factorized_terms_many",
]
