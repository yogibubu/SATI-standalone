"""Native, versioned interpretation of CALPGM/SPFIT parameter identifiers.

The decimal fields follow Pickett's public SPINV convention.  Decoding is
kept separate from matrix-element providers so rigid, rovibronic, LAM and spin
bases consume exactly the same operator description.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


CALPGM_OPERATOR_SCHEMA = "matrix.lock.calpgm_operator.v1"


@dataclass(frozen=True)
class CALPGMOperatorCode:
    identifier: int
    fourier_flag: int
    spin_2: int
    spin_1: int
    n_dot_s_power: int
    projection_type: int
    k_square_power: int
    n_square_power: int
    vibrational_upper: int
    vibrational_lower: int
    n_z_s_z: bool = False
    ratio_constrained: bool = False

    def __post_init__(self) -> None:
        fields = (
            self.fourier_flag,
            self.spin_2,
            self.spin_1,
            self.n_dot_s_power,
            self.projection_type,
            self.k_square_power,
            self.n_square_power,
            self.vibrational_upper,
            self.vibrational_lower,
        )
        if self.identifier == 0 or any(value < 0 for value in fields):
            raise ValueError("invalid CALPGM parameter identifier")

    @property
    def symmetry(self) -> str:
        projection = self.projection_type
        if 20 <= projection < 40:
            return "a"
        if 40 <= projection < 60:
            return "b"
        if 60 <= projection < 80:
            return "c"
        if 12 <= projection < 20:
            return "x"
        return "scalar"

    @property
    def projection_rank(self) -> int:
        projection = self.projection_type
        if projection <= 3:
            return 0 if projection == 0 else 2
        if projection < 12:
            return 2 * projection - 6
        if projection < 20:
            return 2 * projection - 21
        if projection < 80:
            return projection % 20 + 1
        return 0

    @property
    def delta_k(self) -> int | None:
        projection = self.projection_type
        if projection <= 3:
            return 0
        if projection < 12:
            return 2 * projection - 6
        if projection < 20:
            return 2 * projection - 22
        if projection < 80:
            family, rank = projection // 20, projection % 20 + 1
            return rank - (1 if ((family >> 1) + rank) % 2 else 0)
        if projection < 90:
            return 0
        return 2 * (projection - 90)

    @property
    def operator_family(self) -> str:
        if self.projection_type in range(91, 100, 2):
            return "series-constant"
        if self.projection_type >= 90:
            return "euler-series"
        if 80 <= self.projection_type < 90:
            return "selected-K-energy"
        if self.spin_1 and self.spin_2:
            return "spin-spin"
        if self.spin_1 or self.spin_2:
            return "rotation-spin"
        if self.n_dot_s_power or self.n_z_s_z:
            return "spin-rotation-power"
        if self.vibrational_upper != self.vibrational_lower:
            return "vibronic"
        if self.fourier_flag:
            return "fourier-vibration-rotation"
        return "rotation"

    @property
    def implicit_imaginary(self) -> bool:
        projection = self.projection_type
        sine = bool(self.fourier_flag and _calpgm_fourier_mode(self.fourier_flag)[1])
        if sine:
            return projection < 20 or (20 <= projection < 80 and projection % 2 == 1)
        return 20 <= projection < 80 and projection % 2 == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CALPGM_OPERATOR_SCHEMA,
            **asdict(self),
            "symmetry": self.symmetry,
            "projection_rank": self.projection_rank,
            "delta_k": self.delta_k,
            "operator_family": self.operator_family,
            "implicit_imaginary": self.implicit_imaginary,
        }


def decode_calpgm_identifier(identifier: int, *, state_count: int) -> CALPGMOperatorCode:
    """Decode one signed IDPAR without discarding ratio constraints."""

    signed = int(identifier)
    value = abs(signed)
    if value == 0 or state_count < 1 or state_count > 999:
        raise ValueError("CALPGM identifier and state_count must be positive")
    width = 1 if state_count < 10 else 2 if state_count < 100 else 3
    minimum_total = 8 + 2 * width
    digits = str(value).zfill(minimum_total)
    # IDPAR's leading Fourier/tunnelling field is open ended. Ordinary
    # rotational terms use one digit, whereas general torsion--inversion
    # models use values such as 11--14. Every field to its right has fixed
    # width, so the leading width is determined without a molecule-specific
    # rule.
    fourier_width = len(digits) - (7 + 2 * width)
    if fourier_width < 1:
        raise ValueError(f"CALPGM identifier {identifier} is shorter than its state fields")
    offset = 0

    def take(count: int = 1) -> int:
        nonlocal offset
        result = int(digits[offset : offset + count])
        offset += count
        return result

    fourier, spin_2, spin_1, ns = take(fourier_width), take(), take(), take()
    n_z_s_z = ns >= 5
    if n_z_s_z:
        ns -= 5
    projection, ksq, nsq = take(2), take(), take()
    upper, lower = take(width), take(width)
    return CALPGMOperatorCode(
        value,
        fourier,
        spin_2,
        spin_1,
        ns,
        projection,
        ksq,
        nsq,
        upper,
        lower,
        n_z_s_z,
        signed < 0,
    )


def calpgm_operator_inventory(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return a complete native operator inventory for a parsed PAR model."""

    options = model.get("options", ())
    if not options:
        raise ValueError("a CALPGM model requires option records")
    state_count = max(1, abs(int(options[0]["nvib"])))
    rho = calpgm_fourier_rho_values(model)
    nofc, _g12 = calpgm_fourier_options(model)
    rows = []
    families: dict[str, int] = {}
    for parameter in model.get("parameters", ()):
        code = decode_calpgm_identifier(int(parameter["identifier"]), state_count=state_count)
        row = code.to_dict()
        row.update(
            {
                "label": str(parameter.get("label", code.identifier)),
                "value": float(parameter["value"]),
                "uncertainty_or_constraint": float(parameter["uncertainty_or_constraint"]),
                "native_support": calpgm_native_support(
                    code, fourier_context=bool(rho), nofc=nofc
                ),
            }
        )
        rows.append(row)
        families[code.operator_family] = families.get(code.operator_family, 0) + 1
    declared = int(model.get("control", {}).get("parameter_count", len(rows)))
    if declared != len(rows):
        raise ValueError(f"CALPGM declared {declared} parameters but supplied {len(rows)}")
    complete = sum(row["native_support"]["complete_matrix"] for row in rows)
    return {
        "schema": "matrix.lock.calpgm_operator_inventory.v1",
        "source_schema": model.get("schema"),
        "spin_twice": list(calpgm_spin_twice_from_model(model)),
        "parameter_count": len(rows),
        "all_identifiers_decoded": True,
        "native_matrix_coverage": {
            "complete": complete,
            "partial": len(rows) - complete,
            "fraction": complete / len(rows) if rows else 1.0,
        },
        "operator_families": dict(sorted(families.items())),
        "operators": rows,
    }


def calpgm_native_support(
    code: CALPGMOperatorCode,
    *,
    fourier_context: bool = False,
    nofc: bool = False,
) -> dict[str, Any]:
    """Describe which factors of one decoded CALPGM term LOCK can build.

    The status is deliberately mechanical: semantic decoding, the rotational
    carrier, and the vibrational projector are independent from the coupled
    spin tensor.  This prevents a parsed identifier from being mistaken for a
    numerically qualified native SPFIT operator.
    """

    series_constant = code.operator_family == "series-constant"
    rotational = series_constant or 0 <= code.projection_type < 80
    vibrational = (
        series_constant
        or code.fourier_flag == 0
        or nofc
        or fourier_context
    )
    explicit_spins = int(bool(code.spin_1)) + int(bool(code.spin_2))
    rank = code.projection_rank
    spin = (
        explicit_spins == 0
        or code.projection_type == 0
        or (explicit_spins == 1 and rank >= 1)
        or (explicit_spins == 2 and rank >= 2)
    )
    missing = []
    if not rotational:
        missing.append("general-spherical-rotational-provider")
    if not vibrational:
        missing.append(
            "selected-k-provider" if code.fourier_flag and nofc
            else "fourier-rotation-provider"
        )
    if not spin:
        missing.append("coupled-spin-tensor-provider")
    return {
        "identifier_decoded": True,
        "rotational_carrier": rotational,
        "vibrational_projector": vibrational,
        "coupled_spin_tensor": spin,
        "complete_matrix": rotational and vibrational and spin,
        "missing_components": missing,
    }


def calpgm_fourier_options(model: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return SPINV's ``NOFC`` and ``G12`` Fourier switches."""

    options = tuple(model.get("options", ()))
    extended = abs(int(options[0].get("extended_options", 0))) if options else 0
    return bool((extended // 10) & 2), bool((extended // 10) & 4)


def calpgm_fourier_rho_values(model: Mapping[str, Any]) -> tuple[float, ...]:
    """Extract signed state-specific rho constants encoded by ``9100vv``."""

    import math

    options = tuple(model.get("options", ()))
    if not options:
        raise ValueError("a CALPGM model requires option records")
    state_count = max(1, abs(int(options[0]["nvib"])))
    wildcard = 9 if state_count < 10 else 99 if state_count < 100 else 999
    values = [math.nan] * state_count
    specific: list[tuple[int, float]] = []
    for parameter in model.get("parameters", ()):
        code = decode_calpgm_identifier(int(parameter["identifier"]), state_count=state_count)
        is_rho = (
            code.projection_type == 91
            and code.spin_1 == code.spin_2 == code.n_dot_s_power == 0
            and code.k_square_power == code.n_square_power == 0
            and code.vibrational_upper == code.vibrational_lower
        )
        if not is_rho:
            continue
        value = float(parameter["value"])
        if code.vibrational_upper == wildcard:
            values[:] = [value] * state_count
        elif code.vibrational_upper < state_count:
            specific.append((code.vibrational_upper, value))
    for state, value in specific:
        values[state] = value
    return tuple(values) if any(math.isfinite(value) for value in values) else ()


def _calpgm_fourier_mode(flag: int) -> tuple[int, bool]:
    """Map the external FF field to SPINV's harmonic and sine flag."""

    if flag <= 0:
        raise ValueError("a Fourier flag must be positive")
    decade = (flag - 1) // 10
    internal = flag - (decade // 2) * 10
    if decade & 1:
        internal = 9 - internal
    sine = internal < 0
    return (-1 - internal if sine else internal), sine


def calpgm_fourier_modulate(
    code: CALPGMOperatorCode,
    matrix: Any,
    k_bra: Any,
    k_ket: Any,
    rho_upper: float,
    rho_lower: float,
    *,
    g12_phase: bool = False,
) -> Any:
    """Apply SPINV's general Fourier coefficient to a K-space matrix.

    This vectorizes ``specfc`` and covers unequal or opposite-sign rho,
    arbitrary sine/cosine decades, and the optional odd-K phase.  Supplying
    K labels keeps the implementation independent of molecule and basis.
    """

    import math
    import numpy as np

    result = np.asarray(matrix, dtype=complex).copy()
    if not code.fourier_flag:
        return result
    if not math.isfinite(rho_upper) or not math.isfinite(rho_lower):
        raise ValueError("Fourier CALPGM operators require rho for both vibronic states")
    bra = np.asarray(k_bra, dtype=int)
    ket = np.asarray(k_ket, dtype=int)
    if result.shape != (bra.size, ket.size):
        raise ValueError("K labels do not match the Fourier operator matrix")
    harmonic, sine = _calpgm_fourier_mode(code.fourier_flag)
    multiplier = float(harmonic)
    if (rho_upper < 0.0) != (rho_lower < 0.0):
        multiplier -= 0.5
    ki = bra[:, None]
    kj_original = ket[None, :]
    delta = np.abs(ki - kj_original)
    kj = np.where(delta == code.delta_k, kj_original, -kj_original)
    angle = multiplier * (math.pi / 3.0) * (
        abs(rho_upper) * ki + abs(rho_lower) * kj
    )
    factor = np.sin(angle) if sine else np.cos(angle)
    if g12_phase and (harmonic & 1):
        factor *= np.where((np.minimum(ki, kj_original) & 1) != 0, -1.0, 1.0)
    result *= factor
    return result


def calpgm_selected_k_modulate(
    code: CALPGMOperatorCode,
    matrix: Any,
    k_bra: Any,
    k_ket: Any,
) -> Any:
    r"""Apply SPINV's ``NOFC`` fixed-:math:`K_\mathrm{avg}` selector.

    With ``NOFC`` set, the external ``FF`` field is a sampled K value rather
    than a Fourier order.  SPINV selects ``Kavg`` for even ``delta K`` and
    ``Kavg + 1/2`` for odd ``delta K``.  For a quasi-diagonal Wang element the
    ket has the opposite signed-K branch; using the same effective ket as the
    Fourier provider reproduces both DIRCOS cases without a molecule-specific
    convention.
    """

    import numpy as np

    result = np.asarray(matrix, dtype=complex).copy()
    if not code.fourier_flag:
        return result
    selected_k, _sine = _calpgm_fourier_mode(code.fourier_flag)
    bra = np.asarray(k_bra, dtype=int)
    ket = np.asarray(k_ket, dtype=int)
    if result.shape != (bra.size, ket.size):
        raise ValueError("K labels do not match the selected-K operator matrix")
    if selected_k <= 0:
        return result
    ki = bra[:, None]
    kj_original = ket[None, :]
    delta_k = code.delta_k
    if delta_k is None:
        raise ValueError("selected-K CALPGM operators require a defined delta K")
    kj = np.where(np.abs(ki - kj_original) == delta_k, kj_original, -kj_original)
    target = 2 * selected_k - (delta_k & 1)
    result *= np.abs(ki + kj) == target
    return result


def calpgm_spin_twice_from_model(model: Mapping[str, Any]) -> tuple[int, ...]:
    """Return declared ``2I`` values from a CALPGM spin-multiplicity code.

    CALPGM stores one decimal digit ``2I + 1`` per spin in ``SPIND``.  The
    conversion is independent of the molecule and retains every declared
    spin, including a digit of one for a scalar spin space.
    """

    options = model.get("options", ())
    if not options:
        raise ValueError("a CALPGM model requires option records")
    packed = abs(int(options[0]["spin_degeneracy_code"]))
    if packed == 0:
        return ()
    return tuple(int(digit) - 1 for digit in str(packed) if digit != "0")


def calpgm_spin_angular_momentum_operators(two_spin: int) -> tuple[Any, Any, Any]:
    """Return Cartesian angular-momentum matrices for a CALPGM spin."""

    import numpy as np

    if isinstance(two_spin, bool) or int(two_spin) != two_spin or two_spin < 0:
        raise ValueError("CALPGM twice-spin values must be nonnegative integers")
    two_spin = int(two_spin)
    spin = 0.5 * two_spin
    m = np.arange(-spin, spin + 1.0, 1.0)
    raising = np.zeros((len(m), len(m)), dtype=complex)
    for index, value in enumerate(m[:-1]):
        raising[index + 1, index] = (spin * (spin + 1.0) - value * (value + 1.0)) ** 0.5
    lowering = raising.T
    z = np.diag(m).astype(complex)
    x = 0.5 * (raising + lowering)
    y = (lowering - raising) / (2.0j)
    return z, x, y


def _embedded_spin_components(spin_twice: tuple[int, ...], active: int) -> tuple[Any, Any, Any]:
    import numpy as np

    if active < 1 or active > len(spin_twice):
        raise ValueError(f"CALPGM spin identifier {active} has no declared spin")
    components = calpgm_spin_angular_momentum_operators(spin_twice[active - 1])
    embedded = []
    for component in components:
        result = np.ones((1, 1), dtype=complex)
        for index, two_spin in enumerate(spin_twice, 1):
            factor = component if index == active else np.eye(two_spin + 1, dtype=complex)
            result = np.kron(result, factor)
        embedded.append(result)
    return tuple(embedded)


def _half_anticommutator(left: Any, right: Any) -> Any:
    return 0.5 * (left @ right + right @ left)


@dataclass(frozen=True)
class _OperatorJet:
    """Matrix polynomial truncated after one power of each spin marker."""

    coefficients: dict[tuple[int, int], Any]

    def __add__(self, other: Any) -> _OperatorJet:
        if not isinstance(other, _OperatorJet):
            return NotImplemented
        keys = self.coefficients.keys() | other.coefficients.keys()
        return _OperatorJet(
            {
                key: self.coefficients.get(key, 0.0) + other.coefficients.get(key, 0.0)
                for key in keys
            }
        )

    def __sub__(self, other: Any) -> _OperatorJet:
        return self + (-1.0 * other)

    def __mul__(self, scalar: Any) -> _OperatorJet:
        return _OperatorJet({key: value * scalar for key, value in self.coefficients.items()})

    def __rmul__(self, scalar: Any) -> _OperatorJet:
        return self * scalar

    def __truediv__(self, scalar: Any) -> _OperatorJet:
        return self * (1.0 / scalar)

    def __matmul__(self, other: Any) -> _OperatorJet:
        if not isinstance(other, _OperatorJet):
            return NotImplemented
        result: dict[tuple[int, int], Any] = {}
        for (left_1, left_2), left in self.coefficients.items():
            for (right_1, right_2), right in other.coefficients.items():
                degree = left_1 + right_1, left_2 + right_2
                if degree[0] > 1 or degree[1] > 1:
                    continue
                product = left @ right
                result[degree] = result.get(degree, 0.0) + product
        return _OperatorJet(result)


def _operator_power(operator: Any, power: int) -> Any:
    import numpy as np

    if power < 0:
        raise ValueError("operator powers must be nonnegative")
    if isinstance(operator, _OperatorJet):
        dimension = next(iter(operator.coefficients.values())).shape[0]
        result: Any = _OperatorJet({(0, 0): np.eye(dimension, dtype=complex)})
    else:
        result = np.eye(operator.shape[0], dtype=complex)
    factor = operator
    exponent = power
    while exponent:
        if exponent & 1:
            result = result @ factor
        exponent >>= 1
        if exponent:
            factor = factor @ factor
    return result


def _calpgm_raw_projection(code: CALPGMOperatorCode, axes: tuple[Any, Any, Any]) -> Any:
    """Build the homogeneous tesseral carrier before scalar modifiers."""

    import numpy as np

    z, x, y = axes
    if isinstance(z, _OperatorJet):
        dimension = next(iter(z.coefficients.values())).shape[0]
        identity: Any = _OperatorJet({(0, 0): np.eye(dimension, dtype=complex)})
    else:
        identity = np.eye(z.shape[0], dtype=complex)
    projection = code.projection_type
    if projection == 0:
        return identity
    if projection in {1, 2, 3}:
        # TYP=1,2,3 are NaNa,NbNb,NcNc.  In SPINV's internal frame these
        # are z*z,y*y,x*x for a prolate rotor.
        axis = (z, y, x)[projection - 1]
        return axis @ axis
    if projection in {4, 5, 6, 7, 8, 9, 10, 11}:
        power = 2 * (projection - 3)
        plus, minus = x + 1.0j * y, x - 1.0j * y
        return _operator_power(plus, power) + _operator_power(minus, power)
    if projection == 20:
        return z
    if projection == 21:
        return x @ y + y @ x
    if projection == 40:
        return y
    if projection == 41:
        return z @ x + x @ z
    if projection == 60:
        return x
    if projection == 61:
        return z @ y + y @ z
    if 12 <= projection < 80:
        rank = code.projection_rank
        delta_k = code.delta_k
        if delta_k is None or delta_k < 0 or delta_k > rank:
            raise ValueError(f"invalid CALPGM rank/delta-K pair for type {projection}")
        plus, minus = x + 1.0j * y, x - 1.0j * y
        if delta_k == 0:
            return _operator_power(z, rank)
        positive = _operator_power(plus, delta_k)
        negative = _operator_power(minus, delta_k)
        sine_component = code.symmetry == "c"
        if (rank - delta_k) % 2:
            sine_component = not sine_component
        base = (positive - negative) / (2.0j) if sine_component else 0.5 * (positive + negative)
        for _ in range(rank - delta_k):
            base = _half_anticommutator(z, base)
        return base
    raise NotImplementedError(
        f"projection type {projection} requires the general spherical-tensor provider"
    )


def _spin_replaced_projection(
    code: CALPGMOperatorCode,
    rotation: tuple[Any, Any, Any],
    replacements: tuple[tuple[Any, Any, Any], ...],
) -> Any:
    """Replace one or two N factors in an arbitrary CALPGM tensor."""

    if not replacements:
        return _calpgm_raw_projection(code, rotation)
    if len(replacements) > 2:
        raise ValueError("CALPGM identifiers encode at most two explicit spin replacements")
    if code.projection_type == 0:
        if len(replacements) == 1:
            return sum(
                n_axis @ spin_axis
                for n_axis, spin_axis in zip(rotation, replacements[0], strict=True)
            )
        return sum(
            left @ right
            for left, right in zip(replacements[0], replacements[1], strict=True)
        )
    rank = code.projection_rank
    if rank < len(replacements):
        raise ValueError(
            f"projection type {code.projection_type} has rank {rank}, "
            f"too small for {len(replacements)} spin replacements"
        )
    jets = []
    for axis in range(3):
        coefficients = {(0, 0): rotation[axis], (1, 0): replacements[0][axis]}
        if len(replacements) == 2:
            coefficients[(0, 1)] = replacements[1][axis]
        jets.append(_OperatorJet(coefficients))
    expanded = _calpgm_raw_projection(code, tuple(jets))
    marker = (1, 0) if len(replacements) == 1 else (1, 1)
    normalization = rank if len(replacements) == 1 else rank * (rank - 1)
    result = expanded.coefficients[marker] / normalization
    if len(replacements) == 2 and code.projection_type in {1, 2, 3}:
        scalar = sum(
            left @ right
            for left, right in zip(replacements[0], replacements[1], strict=True)
        )
        result = result - scalar / 3.0
    return result


def calpgm_spin_rotational_projection(
    code: CALPGMOperatorCode,
    angular_momenta: tuple[Any, Any, Any],
    spin_twice: tuple[int, ...],
    *,
    prolate: bool = True,
) -> Any:
    """Build a CALPGM rotational/spin operator in an uncoupled product basis.

    One or two rotational factors in any homogeneous ``TYP < 80`` carrier are
    replaced by automatic matrix-polynomial differentiation.  The algorithm
    therefore supports arbitrary CALPGM tensor rank, spin quantum numbers and
    spin identifiers without molecule-specific operator tables.  Type 1--3
    two-spin terms are made traceless as prescribed by SPINV.
    """

    import numpy as np

    if any(
        isinstance(value, bool) or int(value) != value or value < 0
        for value in spin_twice
    ):
        raise ValueError("CALPGM twice-spin values must be nonnegative integers")
    spin_twice = tuple(int(value) for value in spin_twice)
    if not spin_twice:
        if code.spin_1 or code.spin_2 or code.n_dot_s_power or code.n_z_s_z:
            raise ValueError("CALPGM spin operators require declared spins")
        return calpgm_rotational_projection(code, angular_momenta, prolate=prolate)
    Ja, Jb, Jc = (np.asarray(value, dtype=complex) for value in angular_momenta)
    # SPINV defines prolate (z,y,x)=(a,b,c) and oblate (z,y,x)=(c,b,a).
    z, x, y = (Ja, Jc, Jb) if prolate else (Jc, Ja, Jb)
    spin_dimension = int(np.prod([value + 1 for value in spin_twice]))
    rotation_identity = np.eye(z.shape[0], dtype=complex)
    spin_identity = np.eye(spin_dimension, dtype=complex)
    N = tuple(np.kron(value, spin_identity) for value in (z, x, y))
    explicit = tuple(value for value in (code.spin_1, code.spin_2) if value)
    required = set(explicit)
    if code.n_dot_s_power or code.n_z_s_z:
        required.add(1)
    spins = {
        index: tuple(np.kron(rotation_identity, value) for value in _embedded_spin_components(spin_twice, index))
        for index in required
    }
    base = _spin_replaced_projection(code, N, tuple(spins[index] for index in explicit))

    if code.n_z_s_z:
        spin = spins[1]
        base = _half_anticommutator(base, N[0] @ spin[0])
    n2 = sum(value @ value for value in N)
    if code.k_square_power:
        base = _half_anticommutator(
            base, _operator_power(N[0] @ N[0], code.k_square_power)
        )
    if code.n_square_power:
        base = _half_anticommutator(
            base, _operator_power(n2, code.n_square_power)
        )
    if code.n_dot_s_power:
        spin = spins[1]
        n_dot_s = sum(left @ right for left, right in zip(N, spin, strict=True))
        base = _half_anticommutator(
            base, _operator_power(n_dot_s, code.n_dot_s_power)
        )
    base = 0.5 * (base + base.conj().T)
    if not np.all(np.isfinite(base)):
        raise ValueError("CALPGM spin-rotational projection is not finite")
    return base


def calpgm_native_matrix(
    code: CALPGMOperatorCode,
    angular_momenta: tuple[Any, Any, Any],
    *,
    state_count: int,
    spin_twice: tuple[int, ...] = (),
    prolate: bool = True,
    rho_values: tuple[float, ...] = (),
    g12_phase: bool = False,
) -> Any:
    """Assemble one complete native CALPGM operator matrix."""

    import numpy as np

    vibration = calpgm_vibrational_projection(code, state_count)
    rotation_spin = calpgm_spin_rotational_projection(
        code, angular_momenta, spin_twice, prolate=prolate
    )
    if code.fourier_flag:
        dimension = angular_momenta[0].shape[0]
        J = (dimension - 1) // 2
        spin_dimension = rotation_spin.shape[0] // dimension
        k_labels = np.repeat(np.arange(-J, J + 1), spin_dimension)
        matrix = np.zeros(
            (state_count * rotation_spin.shape[0],) * 2, dtype=complex
        )
        for upper, lower in np.argwhere(np.abs(vibration) > 1.0e-15):
            selector = np.zeros_like(vibration)
            selector[upper, lower] = vibration[upper, lower]
            modulated = calpgm_fourier_modulate(
                code,
                rotation_spin,
                k_labels,
                k_labels,
                rho_values[upper],
                rho_values[lower],
                g12_phase=g12_phase,
            )
            matrix += np.kron(selector, modulated)
    else:
        matrix = np.kron(vibration, rotation_spin)
    return 0.5 * (matrix + matrix.conj().T)


def read_calpgm_native_inventory(path: Path | str) -> dict[str, Any]:
    """Parse a PAR file and decode every requested operator natively."""

    from .calpgm_adapter import read_calpgm_parameters

    return calpgm_operator_inventory(read_calpgm_parameters(path))


def calpgm_rotational_projection(
    code: CALPGMOperatorCode,
    angular_momenta: tuple[Any, Any, Any],
    *,
    prolate: bool = True,
) -> Any:
    """Build documented spin-free projections used by ordinary rotor terms.

    General spin and vibronic terms are assembled by the shared tensor engine;
    this helper supplies the Cartesian low-order projections and a normalized
    Hermitian tesseral construction for higher projection ranks.
    """

    import numpy as np

    Ja, Jb, Jc = (np.asarray(value, dtype=complex) for value in angular_momenta)
    # SPINV defines prolate (z,y,x)=(a,b,c) and oblate (z,y,x)=(c,b,a).
    z, x, y = (Ja, Jc, Jb) if prolate else (Jc, Ja, Jb)
    base = _calpgm_raw_projection(code, (z, x, y))
    n2 = Ja @ Ja + Jb @ Jb + Jc @ Jc
    if code.k_square_power:
        k_power = _operator_power(z @ z, code.k_square_power)
        base = 0.5 * (k_power @ base + base @ k_power)
    if code.n_square_power:
        n_power = _operator_power(n2, code.n_square_power)
        base = 0.5 * (n_power @ base + base @ n_power)
    base = 0.5 * (base + base.conj().T)
    if not np.all(np.isfinite(base)):
        raise ValueError("CALPGM rotational projection is not finite")
    return base


def calpgm_vibrational_aliases(options: Any, state_count: int) -> tuple[int, ...]:
    """Return SPINV's canonical parameter index for each vibronic state.

    SPINV stores l-doublets as adjacent physical states.  During ``pasort``
    the second member's parameter index is decremented, so a diagonal
    parameter on the first member applies to both components.  EWT values on
    the option cards define these pairs.
    """

    if state_count < 1:
        raise ValueError("state_count must be positive")
    aliases = list(range(state_count))
    l_values = calpgm_vibrational_l_values(options, state_count)
    index = 0
    while index + 1 < state_count:
        left, right = l_values[index], l_values[index + 1]
        if left and right and left == -right:
            aliases[index + 1] = index
            index += 2
        else:
            index += 1
    return tuple(aliases)


def calpgm_vibrational_l_values(options: Any, state_count: int) -> tuple[int, ...]:
    """Return signed l values reconstructed from CALPGM option records."""

    if state_count < 1:
        raise ValueError("state_count must be positive")
    rows = tuple(options or ())
    if not rows:
        return (0,) * state_count
    first = rows[0]
    divisor = 1000 if int(first.get("e_state_weight", 0)) < 0 else 100
    symmetry_axis = int(first.get("symmetry_axis", 0))
    symmetry_order = {
        4: 2, 5: 2, 6: 3, 7: 4, 8: 4, 9: 5, 10: 6, 11: 6,
    }.get(symmetry_axis, 0)
    ewt = [int(first.get("e_state_weight", 0))] * state_count
    for card_index, row in enumerate(rows):
        vibration = 0 if card_index == 0 else abs(int(row.get("nvib", card_index)))
        if 0 <= vibration < state_count:
            ewt[vibration] = int(row.get("e_state_weight", ewt[vibration]))
    l_values = []
    for value in ewt:
        l_value = abs(value) // divisor
        if l_value > 10:
            l_value -= 10
        if symmetry_order and l_value > symmetry_order:
            l_value = (l_value - 1) % symmetry_order + 1
        l_values.append(l_value)
    index = 0
    while index + 1 < state_count:
        left, right = l_values[index], l_values[index + 1]
        paired = left and right and (
            not symmetry_order or (left + right) % symmetry_order == 0
        )
        if paired:
            if left > right:
                l_values[index] = -right
            else:
                l_values[index + 1] = -left
            index += 2
        else:
            index += 1
    return tuple(l_values)


def calpgm_vibrational_projection(
    code: CALPGMOperatorCode,
    state_count: int,
    *,
    aliases: tuple[int, ...] | None = None,
) -> Any:
    """Return the Hermitian state projector encoded by a CALPGM identifier.

    Diagonal terms select a single vibronic state.  Off-diagonal terms use the
    real or imaginary Hermitian component selected by the CALPGM convention.
    Fourier flags modulate the rotational K-space carrier and therefore leave
    this state projector unchanged.
    """

    import numpy as np

    if state_count < 1:
        raise ValueError("state_count must be positive")
    upper, lower = code.vibrational_upper, code.vibrational_lower
    if upper == lower == 9 and state_count <= 9:
        return np.eye(state_count, dtype=complex)
    if upper >= state_count or lower >= state_count:
        raise ValueError("CALPGM vibrational index exceeds state_count")
    if aliases is None:
        aliases = tuple(range(state_count))
    if len(aliases) != state_count or any(not 0 <= value < state_count for value in aliases):
        raise ValueError("CALPGM vibrational aliases must index every physical state")
    matrix = np.zeros((state_count, state_count), dtype=complex)
    if upper == lower:
        canonical = aliases[upper]
        for state, root in enumerate(aliases):
            if root == canonical:
                matrix[state, state] = 1.0
    elif code.implicit_imaginary:
        matrix[upper, lower] = -0.5j
        matrix[lower, upper] = 0.5j
    else:
        matrix[upper, lower] = 0.5
        matrix[lower, upper] = 0.5
    return matrix


__all__ = [
    "CALPGM_OPERATOR_SCHEMA",
    "CALPGMOperatorCode",
    "calpgm_fourier_modulate",
    "calpgm_fourier_options",
    "calpgm_fourier_rho_values",
    "calpgm_selected_k_modulate",
    "calpgm_native_support",
    "calpgm_native_matrix",
    "calpgm_operator_inventory",
    "calpgm_rotational_projection",
    "calpgm_spin_angular_momentum_operators",
    "calpgm_spin_rotational_projection",
    "calpgm_spin_twice_from_model",
    "calpgm_vibrational_aliases",
    "calpgm_vibrational_l_values",
    "calpgm_vibrational_projection",
    "decode_calpgm_identifier",
    "read_calpgm_native_inventory",
]
