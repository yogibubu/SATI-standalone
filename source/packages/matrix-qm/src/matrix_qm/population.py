"""Backend-independent CM5 and Mayer population analysis.

The CM5 constants are transcribed from the Apache-2.0 CM5PAC 2015 reference
implementation by Duanmu, Wang, Marenich, Cramer and Truhlar. Distances are in
angstrom and charges are in units of the elementary charge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content


CM5_ALPHA_INV_ANGSTROM = 2.4740
CM5_COVALENT_RADII_ANGSTROM = (
    0.3200,
    0.3700,
    1.3000,
    0.9900,
    0.8400,
    0.7500,
    0.7100,
    0.6400,
    0.6000,
    0.6200,
    1.6000,
    1.4000,
    1.2400,
    1.1400,
    1.0900,
    1.0400,
    1.0000,
    1.0100,
    2.0000,
    1.7400,
    1.5900,
    1.4800,
    1.4400,
    1.3000,
    1.2900,
    1.2400,
    1.1800,
    1.1700,
    1.2200,
    1.2000,
    1.2300,
    1.2000,
    1.2000,
    1.1800,
    1.1700,
    1.1600,
    2.1500,
    1.9000,
    1.7600,
    1.6400,
    1.5600,
    1.4600,
    1.3800,
    1.3600,
    1.3400,
    1.3000,
    1.3600,
    1.4000,
    1.4200,
    1.4000,
    1.4000,
    1.3700,
    1.3600,
    1.3600,
    2.3800,
    2.0600,
    1.9400,
    1.8400,
    1.9000,
    1.8800,
    1.8600,
    1.8500,
    1.8300,
    1.8200,
    1.8100,
    1.8000,
    1.7900,
    1.7700,
    1.7700,
    1.7800,
    1.7400,
    1.6400,
    1.5800,
    1.5000,
    1.4100,
    1.3600,
    1.3200,
    1.3000,
    1.3000,
    1.3200,
    1.4400,
    1.4500,
    1.5000,
    1.4200,
    1.4800,
    1.4600,
    2.4200,
    2.1100,
    2.0100,
    1.9000,
    1.8400,
    1.8300,
    1.8000,
    1.8000,
    1.7300,
    1.6800,
    1.6800,
    1.6800,
    1.6500,
    1.6700,
    1.7300,
    1.7600,
    1.6100,
    1.5700,
    1.4900,
    1.4300,
    1.4100,
    1.3400,
    1.2900,
    1.2800,
    1.2100,
    1.2200,
    1.3600,
    1.4300,
    1.6200,
    1.7500,
    1.6500,
    1.5700,
)
CM5_ATOMIC_D = (
    0.0056,
    -0.1543,
    0.0000,
    0.0333,
    -0.1030,
    -0.0446,
    -0.1072,
    -0.0802,
    -0.0629,
    -0.1088,
    0.0184,
    0.0000,
    -0.0726,
    -0.0790,
    -0.0756,
    -0.0565,
    -0.0444,
    -0.0767,
    0.0130,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    -0.0512,
    -0.0557,
    -0.0533,
    -0.0399,
    -0.0313,
    -0.0541,
    0.0092,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    -0.0361,
    -0.0393,
    -0.0376,
    -0.0281,
    -0.0220,
    -0.0381,
    0.0065,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    -0.0255,
    -0.0277,
    -0.0265,
    -0.0198,
    -0.0155,
    -0.0269,
    0.0046,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    -0.0179,
    -0.0195,
    -0.0187,
    -0.0140,
    -0.0110,
    -0.0189,
)
CM5_PAIR_D = {
    (1, 6): 0.0502,
    (1, 7): 0.1747,
    (1, 8): 0.1671,
    (6, 7): 0.0556,
    (6, 8): 0.0234,
    (7, 8): -0.0346,
}
MATRIX_XYZ_QM_POPULATION_SCHEMA = "matrix.qm.population.v1"
QM_POPULATION_SECTION = "QM_POPULATION"


@dataclass(frozen=True)
class QMPopulationObservables:
    """The mandatory default electronic observables used by ZAFF."""

    hirshfeld_charges: np.ndarray
    cm5_charges: np.ndarray
    mayer_bond_orders: np.ndarray
    electron_count: float
    charge: float
    ecp_core_electrons: np.ndarray | None = None
    spin_populations: np.ndarray | None = None

    def __post_init__(self) -> None:
        hirshfeld = np.asarray(self.hirshfeld_charges, dtype=float).reshape(-1)
        cm5 = np.asarray(self.cm5_charges, dtype=float).reshape(-1)
        mayer = np.asarray(self.mayer_bond_orders, dtype=float)
        if hirshfeld.shape != cm5.shape:
            raise ValueError("Hirshfeld and CM5 charge vectors must have the same shape")
        if mayer.shape != (cm5.size, cm5.size):
            raise ValueError("Mayer matrix must have shape natoms x natoms")
        if any(not np.all(np.isfinite(values)) for values in (hirshfeld, cm5, mayer)):
            raise ValueError("QM population observables contain non-finite values")
        if not math.isfinite(float(self.electron_count)) or not math.isfinite(float(self.charge)):
            raise ValueError("electron count and molecular charge must be finite")
        core = (
            np.zeros(cm5.size, dtype=int)
            if self.ecp_core_electrons is None
            else np.asarray(self.ecp_core_electrons, dtype=int).reshape(-1)
        )
        if core.shape != cm5.shape or np.any(core < 0):
            raise ValueError("ECP core-electron counts must be nonnegative and per atom")
        spin = (
            None
            if self.spin_populations is None
            else np.asarray(self.spin_populations, dtype=float).reshape(-1)
        )
        if spin is not None and (spin.shape != cm5.shape or not np.all(np.isfinite(spin))):
            raise ValueError("spin populations must be finite and per atom")
        object.__setattr__(self, "hirshfeld_charges", hirshfeld)
        object.__setattr__(self, "cm5_charges", cm5)
        object.__setattr__(self, "mayer_bond_orders", 0.5 * (mayer + mayer.T))
        object.__setattr__(self, "ecp_core_electrons", core)
        object.__setattr__(self, "spin_populations", spin)

    @property
    def natoms(self) -> int:
        return int(self.cm5_charges.size)

    def to_dict(self, *, source: str = "") -> dict[str, object]:
        pairs = [
            [left + 1, right + 1, float(self.mayer_bond_orders[left, right])]
            for left in range(self.natoms)
            for right in range(left + 1, self.natoms)
        ]
        payload = {
            "schema": "matrix.apoc.analysis.v1",
            "source": str(source),
            "charge_model": "CM5",
            "bond_order_model": "Mayer",
            "electron_count": float(self.electron_count),
            "molecular_charge": float(self.charge),
            "ecp_core_electrons": self.ecp_core_electrons.tolist(),
            "electron_space": (
                "ECP_VALENCE_PLUS_FROZEN_CORE"
                if np.any(self.ecp_core_electrons)
                else "ALL_ELECTRON"
            ),
            "hirshfeld_charges": self.hirshfeld_charges.tolist(),
            "cm5_charges": self.cm5_charges.tolist(),
            "mayer_bond_orders": pairs,
            "spin_population_present": self.spin_populations is not None,
        }
        if self.spin_populations is not None:
            payload["spin_populations"] = self.spin_populations.tolist()
        return payload


def population_observables_from_dict(payload: dict[str, object]) -> QMPopulationObservables:
    """Restore the fixed APOC v1 CM5/Mayer contract from JSON-compatible data."""

    if payload.get("schema") != "matrix.apoc.analysis.v1":
        raise ValueError("unsupported APOC population schema")
    if str(payload.get("charge_model", "")).upper() != "CM5":
        raise ValueError("APOC v1 requires charge_model=CM5")
    if str(payload.get("bond_order_model", "")).upper() != "MAYER":
        raise ValueError("APOC v1 requires bond_order_model=Mayer")
    cm5 = np.asarray(payload.get("cm5_charges", []), dtype=float).reshape(-1)
    hirshfeld = np.asarray(payload.get("hirshfeld_charges", []), dtype=float).reshape(-1)
    mayer = np.zeros((cm5.size, cm5.size), dtype=float)
    for row in payload.get("mayer_bond_orders", []):
        left, right, value = row
        i = int(left) - 1
        j = int(right) - 1
        if not (0 <= i < cm5.size and 0 <= j < cm5.size and i != j):
            raise ValueError("invalid one-based Mayer pair in APOC data")
        mayer[i, j] = mayer[j, i] = float(value)
    spin_payload = payload.get("spin_populations")
    if payload.get("spin_population_present") is False:
        spin_payload = None
    return QMPopulationObservables(
        hirshfeld_charges=hirshfeld,
        cm5_charges=cm5,
        mayer_bond_orders=mayer,
        electron_count=float(payload.get("electron_count", 0.0)),
        charge=float(payload.get("molecular_charge", np.sum(cm5))),
        ecp_core_electrons=np.asarray(
            payload.get("ecp_core_electrons", np.zeros(cm5.size)),
            dtype=int,
        ),
        spin_populations=(None if spin_payload is None else np.asarray(spin_payload, dtype=float)),
    )


def qm_population_section_lines(
    observables: QMPopulationObservables,
    *,
    source: str,
) -> list[str]:
    """Serialize the backend-independent APOC observables into an xyzin section."""

    lines = [
        f"SCHEMA {MATRIX_XYZ_QM_POPULATION_SCHEMA}",
        "INDEXING ATOMS=ONE_BASED",
        f"SOURCE {str(source).strip() or 'unspecified'}",
        "CHARGE_MODEL CM5",
        "BOND_ORDER_MODEL Mayer",
        f"ELECTRON_COUNT {float(observables.electron_count):.12g}",
        f"MOLECULAR_CHARGE {float(observables.charge):.12g}",
        "ECP_CORE_ELECTRONS "
        + ",".join(str(int(value)) for value in observables.ecp_core_electrons),
        "[ATOMIC_CHARGES]",
        "COLUMNS ATOM HIRSHFELD CM5",
    ]
    lines.extend(
        f"{atom + 1} {observables.hirshfeld_charges[atom]:.12g} "
        f"{observables.cm5_charges[atom]:.12g}"
        for atom in range(observables.natoms)
    )
    lines.extend(["[BOND_ORDERS]", "COLUMNS ATOM_I ATOM_J MAYER"])
    lines.extend(
        f"{left + 1} {right + 1} {observables.mayer_bond_orders[left, right]:.12g}"
        for left in range(observables.natoms)
        for right in range(left + 1, observables.natoms)
    )
    return lines


def parse_qm_population_section(lines: list[str]) -> tuple[QMPopulationObservables, str]:
    """Parse one canonical ``#QM_POPULATION`` section."""

    if not lines or lines[0].strip() != f"SCHEMA {MATRIX_XYZ_QM_POPULATION_SCHEMA}":
        raise ValueError("missing or unsupported #QM_POPULATION schema")
    metadata: dict[str, str] = {}
    hirshfeld: dict[int, float] = {}
    cm5: dict[int, float] = {}
    pairs: list[tuple[int, int, float]] = []
    block = ""
    for raw in lines[1:]:
        text = raw.strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]"):
            block = text.upper()
            continue
        if text.upper().startswith(("INDEXING ", "COLUMNS ")):
            continue
        parts = text.split()
        if block == "[ATOMIC_CHARGES]" and len(parts) == 3:
            index = int(parts[0]) - 1
            hirshfeld[index] = float(parts[1])
            cm5[index] = float(parts[2])
        elif block == "[BOND_ORDERS]" and len(parts) == 3:
            pairs.append((int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])))
        elif len(parts) >= 2:
            metadata[parts[0].upper()] = " ".join(parts[1:])
    if metadata.get("CHARGE_MODEL", "").upper() != "CM5":
        raise ValueError("#QM_POPULATION charge model must be CM5")
    if metadata.get("BOND_ORDER_MODEL", "").upper() != "MAYER":
        raise ValueError("#QM_POPULATION bond-order model must be Mayer")
    natoms = len(cm5)
    if set(cm5) != set(range(natoms)) or set(hirshfeld) != set(range(natoms)):
        raise ValueError("#QM_POPULATION atomic charge vectors are incomplete")
    mayer = np.zeros((natoms, natoms), dtype=float)
    expected_pairs = natoms * (natoms - 1) // 2
    if len(pairs) != expected_pairs:
        raise ValueError("#QM_POPULATION Mayer matrix is incomplete")
    for left, right, value in pairs:
        if not (0 <= left < natoms and 0 <= right < natoms and left < right):
            raise ValueError("#QM_POPULATION contains an invalid Mayer pair")
        mayer[left, right] = mayer[right, left] = value
    observables = QMPopulationObservables(
        hirshfeld_charges=np.asarray([hirshfeld[i] for i in range(natoms)]),
        cm5_charges=np.asarray([cm5[i] for i in range(natoms)]),
        mayer_bond_orders=mayer,
        electron_count=float(metadata["ELECTRON_COUNT"]),
        charge=float(metadata["MOLECULAR_CHARGE"]),
        ecp_core_electrons=np.asarray(
            [
                int(value)
                for value in metadata.get(
                    "ECP_CORE_ELECTRONS",
                    ",".join("0" for _ in range(natoms)),
                ).split(",")
            ],
            dtype=int,
        ),
    )
    return observables, metadata.get("SOURCE", "unspecified")


def write_qm_population_section(
    path: Path | str,
    observables: QMPopulationObservables,
    *,
    source: str,
) -> Path:
    target = Path(path)
    replace_section(
        target,
        QM_POPULATION_SECTION,
        qm_population_section_lines(observables, source=source),
    )
    return target


def read_qm_population_section(
    path: Path | str,
) -> tuple[QMPopulationObservables, str]:
    content = section_content(read_sectioned_lines(Path(path)), QM_POPULATION_SECTION)
    return parse_qm_population_section(content)


def cm5_charges_from_hirshfeld(
    atomic_numbers,
    coordinates_angstrom,
    hirshfeld_charges,
) -> np.ndarray:
    """Apply the unique CM5PAC-2015 mapping to Hirshfeld charges."""

    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    charges = np.asarray(hirshfeld_charges, dtype=float).reshape(-1)
    if coords.shape != (numbers.size, 3) or charges.shape != (numbers.size,):
        raise ValueError("CM5 input shapes must be natoms, natoms x 3, natoms")
    if np.any(numbers < 1) or np.any(numbers > 118):
        raise ValueError("CM5 supports atomic numbers 1 through 118")
    if any(not np.all(np.isfinite(values)) for values in (coords, charges)):
        raise ValueError("CM5 inputs must be finite")
    result = charges.copy()
    for left in range(numbers.size):
        z_left = int(numbers[left])
        for right in range(numbers.size):
            z_right = int(numbers[right])
            if left == right or z_left == z_right:
                continue
            distance = float(np.linalg.norm(coords[left] - coords[right]))
            radial = math.exp(
                -CM5_ALPHA_INV_ANGSTROM
                * (
                    distance
                    - CM5_COVALENT_RADII_ANGSTROM[z_left - 1]
                    - CM5_COVALENT_RADII_ANGSTROM[z_right - 1]
                )
            )
            result[left] += radial * _cm5_pair_parameter(z_left, z_right)
    if not np.isclose(np.sum(result), np.sum(charges), rtol=0.0, atol=5.0e-12):
        raise ArithmeticError("CM5 antisymmetric correction did not conserve charge")
    return result


def hirshfeld_charges_from_grid(
    atomic_numbers,
    grid_weights,
    molecular_density,
    proatom_densities,
    *,
    expected_charge: float | None = None,
    charge_tolerance: float = 2.0e-4,
) -> np.ndarray:
    """Integrate Hirshfeld stockholder charges on a common numerical grid."""

    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    weights = np.asarray(grid_weights, dtype=float).reshape(-1)
    density = np.asarray(molecular_density, dtype=float).reshape(-1)
    proatoms = np.asarray(proatom_densities, dtype=float)
    if proatoms.shape != (numbers.size, weights.size) or density.shape != weights.shape:
        raise ValueError("Hirshfeld grid arrays have incompatible shapes")
    if any(not np.all(np.isfinite(values)) for values in (weights, density, proatoms)):
        raise ValueError("Hirshfeld grid arrays must be finite")
    if np.any(weights < 0.0) or np.any(density < -1.0e-12) or np.any(proatoms < -1.0e-12):
        raise ValueError("Hirshfeld densities and quadrature weights must be non-negative")
    denominator = np.sum(np.maximum(proatoms, 0.0), axis=0)
    active = denominator > np.finfo(float).tiny
    stockholder = np.zeros_like(proatoms)
    stockholder[:, active] = proatoms[:, active] / denominator[active]
    populations = np.sum(stockholder * (weights * np.maximum(density, 0.0))[None, :], axis=1)
    charges = numbers.astype(float) - populations
    if expected_charge is not None:
        residual = float(expected_charge) - float(np.sum(charges))
        if abs(residual) > float(charge_tolerance):
            raise ValueError(
                f"Hirshfeld quadrature violates the molecular charge by {residual:.3e} e"
            )
        charges += residual / numbers.size
    return charges


def mayer_bond_order_matrix(
    overlap,
    ao_atom_indices,
    *,
    density_total=None,
    density_alpha=None,
    density_beta=None,
) -> np.ndarray:
    """Calculate Mayer bond orders from AO density and overlap matrices.

    Restricted input uses the spin-summed density. Unrestricted input uses the
    standard spin-resolved expression with the factor of two that recovers the
    restricted definition when alpha and beta densities are equal.
    """

    metric = np.asarray(overlap, dtype=float)
    owners = np.asarray(ao_atom_indices, dtype=int).reshape(-1)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1] or owners.size != metric.shape[0]:
        raise ValueError("overlap and AO ownership arrays have incompatible shapes")
    if density_total is not None and (density_alpha is not None or density_beta is not None):
        raise ValueError("provide either a total density or alpha and beta densities")
    if density_total is not None:
        density = _square_density(density_total, metric.shape)
        products = ((density @ metric),)
        factors = (1.0,)
    else:
        if density_alpha is None or density_beta is None:
            raise ValueError("both alpha and beta density matrices are required")
        products = (
            _square_density(density_alpha, metric.shape) @ metric,
            _square_density(density_beta, metric.shape) @ metric,
        )
        factors = (2.0, 2.0)
    natoms = int(np.max(owners)) + 1 if owners.size else 0
    if natoms < 1 or np.any(owners < 0) or set(owners) != set(range(natoms)):
        raise ValueError("AO atom indices must be contiguous and zero-based")
    result = np.zeros((natoms, natoms), dtype=float)
    atom_aos = [np.flatnonzero(owners == atom) for atom in range(natoms)]
    for left in range(natoms):
        mu = atom_aos[left]
        for right in range(left + 1, natoms):
            nu = atom_aos[right]
            value = 0.0
            for factor, product in zip(factors, products, strict=True):
                value += factor * float(
                    np.einsum(
                        "ij,ji->",
                        product[np.ix_(mu, nu)],
                        product[np.ix_(nu, mu)],
                        optimize=True,
                    )
                )
            result[left, right] = result[right, left] = value
    return result


def mayer_spin_bond_order_matrix(
    overlap,
    ao_atom_indices,
    *,
    density_alpha,
    density_beta,
) -> np.ndarray:
    """Return the signed alpha-minus-beta Mayer bond-order contribution.

    The definition uses the same AO partition and normalization as the
    unrestricted total Mayer matrix, but subtracts the beta-channel
    contribution from the alpha-channel contribution.  It is therefore zero
    for a restricted closed shell and changes sign when alpha and beta labels
    are exchanged.  The matrix measures spin-selective covalency on bonds; it
    is complementary to, and not a replacement for, an atomic spin density.
    """

    metric = np.asarray(overlap, dtype=float)
    owners = np.asarray(ao_atom_indices, dtype=int).reshape(-1)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1] or owners.size != metric.shape[0]:
        raise ValueError("overlap and AO ownership arrays have incompatible shapes")
    alpha_product = _square_density(density_alpha, metric.shape) @ metric
    beta_product = _square_density(density_beta, metric.shape) @ metric
    natoms = int(np.max(owners)) + 1 if owners.size else 0
    if natoms < 1 or np.any(owners < 0) or set(owners) != set(range(natoms)):
        raise ValueError("AO atom indices must be contiguous and zero-based")
    result = np.zeros((natoms, natoms), dtype=float)
    atom_aos = [np.flatnonzero(owners == atom) for atom in range(natoms)]
    for left in range(natoms):
        mu = atom_aos[left]
        for right in range(left + 1, natoms):
            nu = atom_aos[right]
            alpha = 2.0 * float(
                np.einsum(
                    "ij,ji->",
                    alpha_product[np.ix_(mu, nu)],
                    alpha_product[np.ix_(nu, mu)],
                    optimize=True,
                )
            )
            beta = 2.0 * float(
                np.einsum(
                    "ij,ji->",
                    beta_product[np.ix_(mu, nu)],
                    beta_product[np.ix_(nu, mu)],
                    optimize=True,
                )
            )
            result[left, right] = result[right, left] = alpha - beta
    return result


def _cm5_pair_parameter(left: int, right: int) -> float:
    if left == right:
        return 0.0
    pair = (left, right) if left < right else (right, left)
    value = CM5_PAIR_D.get(pair)
    if value is None:
        value = CM5_ATOMIC_D[pair[0] - 1] - CM5_ATOMIC_D[pair[1] - 1]
    return float(value if left < right else -value)


def _square_density(value, shape: tuple[int, int]) -> np.ndarray:
    density = np.asarray(value, dtype=float)
    if density.shape != shape or not np.all(np.isfinite(density)):
        raise ValueError(f"density matrix must be finite with shape {shape}")
    return density
