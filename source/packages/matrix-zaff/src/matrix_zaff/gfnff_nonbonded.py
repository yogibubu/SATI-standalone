"""Frozen GFN-FF non-bonded pair conversion to the ZAFF radial form.

GFN-FF does not define a transferable element-only van der Waals table.  Its
repulsive exponent and D3 dispersion coefficients depend on the prepared
topology, coordination and frozen topological charges.  This module therefore
accepts the *resolved atom-pair parameters* produced for one fixed pair and
maps their radial well to ZAFF Exp-PE without pretending that they are
element-only constants.

Electrostatics and directional corrections remain separate. H-bond and XB
radial terms are compiled to Exp-PE as well, so they can share the same ZAFF
lookup-bank implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from math import erf, exp, isfinite
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .construction import (
    ResolvedExpPEPairTable,
    evaluate_exppe_dimensionless,
    hydrogen_bond_radial_switch,
)
from .radial import (
    DampedExpPEPotential,
    MinimumDerivatives,
    damped_exppe_from_minimum,
)


BOHR_TO_ANGSTROM = 0.52917721092
HARTREE_TO_KCAL_PER_MOL = 627.5094740631
_D3_FREQUENCIES = np.asarray(
    (
        0.000001, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
        0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50,
        3.00, 4.00, 5.00, 7.50, 10.00,
    )
)
_D3_WEIGHTS = np.empty_like(_D3_FREQUENCIES)
_D3_WEIGHTS[0] = 0.5 * (_D3_FREQUENCIES[1] - _D3_FREQUENCIES[0])
_D3_WEIGHTS[-1] = 0.5 * (_D3_FREQUENCIES[-1] - _D3_FREQUENCIES[-2])
_D3_WEIGHTS[1:-1] = 0.5 * (
    _D3_FREQUENCIES[2:] - _D3_FREQUENCIES[:-2]
)


@dataclass(frozen=True)
class FrozenGFNFFNonbondedPair:
    """Resolved GFN-FF repulsion plus two-body D3 radial parameters.

    All distances and energies may use any internally consistent unit system.
    ``damping_r0_squared`` is the squared BJ damping radius used by xTB;
    ``c8_coefficient`` corresponds to ``2 * 3*r4r2_i*r4r2_j`` in the
    non-periodic GFN-FF implementation.
    """

    repulsion_prefactor: float
    repulsion_alpha: float
    c6: float
    zeta_scale: float
    dispersion_scale: float
    damping_r0_squared: float
    c8_coefficient: float

    def __post_init__(self) -> None:
        values = (
            self.repulsion_prefactor,
            self.repulsion_alpha,
            self.c6,
            self.zeta_scale,
            self.dispersion_scale,
            self.damping_r0_squared,
            self.c8_coefficient,
        )
        if any(not isfinite(float(value)) for value in values):
            raise ValueError("frozen GFN-FF pair parameters must be finite")
        if (
            self.repulsion_prefactor <= 0.0
            or self.repulsion_alpha <= 0.0
            or self.c6 <= 0.0
            or self.zeta_scale <= 0.0
            or self.dispersion_scale <= 0.0
            or self.damping_r0_squared <= 0.0
            or self.c8_coefficient < 0.0
        ):
            raise ValueError("frozen GFN-FF pair parameters are not physical")

    def energy(self, distance: float) -> float:
        """Evaluate the frozen two-body repulsion plus damped D3 energy."""

        r = float(distance)
        if not isfinite(r) or r <= 0.0:
            raise ValueError("pair distance must be finite and positive")
        r2 = r * r
        repulsion = (
            self.repulsion_prefactor
            * exp(-self.repulsion_alpha * r2**0.75)
            / r
        )
        dispersion_kernel = 1.0 / (
            r2**3 + self.damping_r0_squared**3
        ) + self.c8_coefficient / (
            r2**4 + self.damping_r0_squared**4
        )
        dispersion = (
            -self.c6
            * self.zeta_scale
            * self.dispersion_scale
            * dispersion_kernel
        )
        return float(repulsion + dispersion)

    def minimum_derivatives(
        self,
        lower_distance: float,
        upper_distance: float,
    ) -> MinimumDerivatives:
        """Locate the attractive minimum and obtain its local curvature."""

        lower = float(lower_distance)
        upper = float(upper_distance)
        if not (isfinite(lower) and isfinite(upper) and 0.0 < lower < upper):
            raise ValueError("GFN-FF minimum bounds must satisfy 0 < lower < upper")
        result = minimize_scalar(
            self.energy,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": max(1.0e-12, 1.0e-10 * (upper - lower))},
        )
        if not result.success:
            raise RuntimeError(f"GFN-FF radial minimum search failed: {result.message}")
        r_min = float(result.x)
        energy_min = self.energy(r_min)
        if energy_min >= 0.0:
            raise ValueError("resolved GFN-FF pair has no attractive well in the bounds")
        margin = min(r_min - lower, upper - r_min)
        step = min(max(1.0e-5 * r_min, 1.0e-7), 0.2 * margin)
        if step <= 0.0:
            raise ValueError("GFN-FF radial minimum lies on a search boundary")
        f0 = energy_min
        fm1 = self.energy(r_min - step)
        fp1 = self.energy(r_min + step)
        fm2 = self.energy(r_min - 2.0 * step)
        fp2 = self.energy(r_min + 2.0 * step)
        curvature = (
            -fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2
        ) / (12.0 * step * step)
        return MinimumDerivatives(
            epsilon=-energy_min,
            r_min=r_min,
            second=curvature,
        )

    def to_exppe(
        self,
        lower_distance: float,
        upper_distance: float,
    ) -> DampedExpPEPotential:
        """Map the frozen GFN-FF well preserving depth, minimum and curvature."""

        return damped_exppe_from_minimum(
            self.minimum_derivatives(lower_distance, upper_distance)
        )


def gfnff_log_coordination_numbers(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    covalent_radii_bohr_by_atomic_number: Sequence[float],
    *,
    bonded_pairs: Sequence[tuple[int, int]] | None = None,
    covalent_radius_scale: float = 1.0,
    cn_max: float = 4.4,
    steepness: float = -7.5,
) -> np.ndarray:
    """Reproduce the GFN-FF log-CN mapping on intrinsic covalent pairs.

    When ``bonded_pairs`` is supplied, all other pairs are rigorously excluded.
    This is the construction-time contract: other fragments and intramolecular
    non-covalent contacts cannot change coordination or downstream typing.
    """

    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    radii = np.asarray(covalent_radii_bohr_by_atomic_number, dtype=float)
    if coordinates.shape != (len(numbers), 3):
        raise ValueError("GFN-FF coordinates and atomic numbers are inconsistent")
    if (
        len(numbers)
        and (
            np.min(numbers) < 1
            or np.max(numbers) > len(radii)
            or np.any(radii[numbers - 1] <= 0.0)
        )
    ):
        raise ValueError("GFN-FF covalent radii do not cover all atoms")
    raw = np.zeros(len(numbers), dtype=float)
    radius_scale = float(covalent_radius_scale)
    if not isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("GFN-FF covalent-radius scale must be positive")
    allowed = (
        None
        if bonded_pairs is None
        else {
            tuple(sorted((int(left), int(right))))
            for left, right in bonded_pairs
        }
    )
    if allowed is not None and any(
        left == right or left < 0 or right >= len(numbers)
        for left, right in allowed
    ):
        raise ValueError("GFN-FF intrinsic coordination contains an invalid bond")
    for left in range(1, len(numbers)):
        for right in range(left):
            if allowed is not None and (right, left) not in allowed:
                continue
            r0 = radius_scale * (
                radii[numbers[left] - 1] + radii[numbers[right] - 1]
            )
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            contribution = 0.5 * (
                1.0 + erf(float(steepness) * (distance - r0) / r0)
            )
            raw[left] += contribution
            raw[right] += contribution
    maximum = float(cn_max)
    return np.logaddexp(0.0, maximum) - np.logaddexp(0.0, maximum - raw)


@lru_cache(maxsize=128)
def cached_gfnff_fragment_metadata(
    elements: tuple[str, ...],
    coordinates_key: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Return reusable intrinsic topology metadata for a rigid fragment.

    The cache is shared by all ZAFF clients.  Coordinates are part of the key
    because coordination is geometry dependent; inter-fragment contacts are
    never included in the intrinsic bond list.
    """

    from matrix_chem.topology.elements import atomic_number
    from matrix_chem.topology.covalent_radii import covalent_radius

    numbers = np.asarray([atomic_number(symbol) or 0 for symbol in elements], dtype=int)
    coordinates = np.asarray(coordinates_key, dtype=float).reshape((-1, 3))
    if np.any(numbers < 1) or coordinates.shape != (len(numbers), 3):
        raise ValueError("cannot prepare frozen GFN-FF metadata for the fragment")
    adjacency = np.zeros((len(numbers), len(numbers)), dtype=bool)
    for left in range(len(numbers)):
        radius_left = float(covalent_radius(int(numbers[left])) or 0.75)
        for right in range(left):
            radius_right = float(covalent_radius(int(numbers[right])) or 0.75)
            if np.linalg.norm(coordinates[left] - coordinates[right]) <= 1.25 * (
                radius_left + radius_right
            ):
                adjacency[left, right] = adjacency[right, left] = True
    bundle = load_gfnff_parameter_bundle()
    bonds = tuple(
        (right, left)
        for left in range(len(numbers))
        for right in range(left)
        if adjacency[left, right]
    )
    coordination = gfnff_log_coordination_numbers(
        numbers,
        coordinates / BOHR_TO_ANGSTROM,
        bundle.covalent_radii_bohr,
        bonded_pairs=bonds,
    )
    return numbers, np.sum(adjacency, axis=1).astype(int), coordination, bonds


def gfnff_reference_weights(
    coordination_number: float,
    reference_coordination_numbers: Sequence[float],
    *,
    weighting_factor: float = 4.0,
) -> np.ndarray:
    """Return the normalized GFN-FF Gaussian reference-CN weights."""

    references = np.asarray(reference_coordination_numbers, dtype=float).reshape(-1)
    if not len(references) or np.any(~np.isfinite(references)):
        raise ValueError("GFN-FF reference coordination numbers are invalid")
    log_weights = -float(weighting_factor) * (
        float(coordination_number) - references
    ) ** 2
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / float(np.sum(weights))


@dataclass(frozen=True)
class GFNFFReferenceDispersion:
    """Reference-CN and C6 tables used by the GFN-FF D3 interpolation."""

    reference_cn_by_atomic_number: Mapping[int, Sequence[float]]
    reference_c6_by_pair: Mapping[tuple[int, int], np.ndarray]
    weighting_factor: float = 4.0

    def cross_c6(
        self,
        candidate_atomic_numbers: Sequence[int],
        candidate_coordination_numbers: Sequence[float],
        environment_atomic_numbers: Sequence[int],
        environment_coordination_numbers: Sequence[float],
    ) -> np.ndarray:
        candidate_z = np.asarray(candidate_atomic_numbers, dtype=int).reshape(-1)
        environment_z = np.asarray(environment_atomic_numbers, dtype=int).reshape(-1)
        candidate_cn = np.asarray(candidate_coordination_numbers, dtype=float).reshape(-1)
        environment_cn = np.asarray(environment_coordination_numbers, dtype=float).reshape(-1)
        if candidate_cn.shape != candidate_z.shape or environment_cn.shape != environment_z.shape:
            raise ValueError("GFN-FF atomic numbers and coordination arrays disagree")
        output = np.empty((len(candidate_z), len(environment_z)), dtype=float)
        candidate_weights = [
            gfnff_reference_weights(
                cn,
                self.reference_cn_by_atomic_number[int(z)],
                weighting_factor=self.weighting_factor,
            )
            for z, cn in zip(candidate_z, candidate_cn, strict=True)
        ]
        environment_weights = [
            gfnff_reference_weights(
                cn,
                self.reference_cn_by_atomic_number[int(z)],
                weighting_factor=self.weighting_factor,
            )
            for z, cn in zip(environment_z, environment_cn, strict=True)
        ]
        for left, (zi, wi) in enumerate(zip(candidate_z, candidate_weights, strict=True)):
            for right, (zj, wj) in enumerate(
                zip(environment_z, environment_weights, strict=True)
            ):
                key = (min(int(zi), int(zj)), max(int(zi), int(zj)))
                reference = np.asarray(self.reference_c6_by_pair[key], dtype=float)
                expected = (
                    (len(wi), len(wj))
                    if int(zi) <= int(zj)
                    else (len(wj), len(wi))
                )
                if reference.shape != expected:
                    raise ValueError(f"GFN-FF reference C6 dimensions disagree for {key}")
                if int(zi) > int(zj):
                    reference = reference.T
                output[left, right] = float(wi @ reference @ wj)
        return output


@dataclass(frozen=True)
class GFNFFParameterBundle:
    """Compact xTB-derived atomic and reference-polarizability parameters."""

    repan: np.ndarray
    torsion_central: np.ndarray
    torsion_terminal: np.ndarray
    repz: np.ndarray
    effective_nuclear_charge: np.ndarray
    hardness: np.ndarray
    sqrt_z_r4_over_r2: np.ndarray
    covalent_radii_bohr: np.ndarray
    reference_count: np.ndarray
    reference_cn: np.ndarray
    reference_alpha_iw: np.ndarray
    provenance: str

    def reference_dispersion(
        self, atomic_numbers: Sequence[int]
    ) -> GFNFFReferenceDispersion:
        unique = sorted({int(value) for value in atomic_numbers})
        cn = {
            z: self.reference_cn[z - 1, : int(self.reference_count[z - 1])].copy()
            for z in unique
        }
        c6: dict[tuple[int, int], np.ndarray] = {}
        for zi in unique:
            ni = int(self.reference_count[zi - 1])
            ai = self.reference_alpha_iw[zi - 1, :ni]
            for zj in unique:
                if zj < zi:
                    continue
                nj = int(self.reference_count[zj - 1])
                aj = self.reference_alpha_iw[zj - 1, :nj]
                c6[(zi, zj)] = (
                    3.0
                    / np.pi
                    * np.einsum("if,jf,f->ij", ai, aj, _D3_WEIGHTS)
                )
        return GFNFFReferenceDispersion(cn, c6)


@lru_cache(maxsize=1)
def load_gfnff_parameter_bundle() -> GFNFFParameterBundle:
    """Load the resident xTB-6.7.1 comparison source parameters."""

    resource = files("matrix_zaff").joinpath("data/gfnff_reference_v671.npz")
    with as_file(resource) as path, np.load(path, allow_pickle=False) as data:
        if data["schema"].item() != "matrix.zaff.gfnff_reference.v1":
            raise ValueError("unsupported resident GFN-FF reference bundle")
        return GFNFFParameterBundle(
            repan=data["repan"].copy(),
            torsion_central=data["torsion_central"].copy(),
            torsion_terminal=data["torsion_terminal"].copy(),
            repz=data["repz"].copy(),
            effective_nuclear_charge=data["effective_nuclear_charge"].copy(),
            hardness=data["hardness"].copy(),
            sqrt_z_r4_over_r2=data["sqrt_z_r4_over_r2"].copy(),
            covalent_radii_bohr=data["covalent_radii_bohr"].copy(),
            reference_count=data["reference_count"].copy(),
            reference_cn=data["reference_cn"].copy(),
            reference_alpha_iw=data["reference_alpha_iw"].copy(),
            provenance="xTB 6.7.1 GFN-FF numerical parameters; LGPL-3.0-or-later",
        )


def gfnff_charge_scale(
    atomic_number: int,
    charge_e: float,
    effective_nuclear_charge_by_atomic_number: Sequence[float],
    hardness_by_atomic_number: Sequence[float],
) -> float:
    """GFN-FF charge scaling factor used for the pair C6 coefficient."""

    z = int(atomic_number)
    zeff = float(effective_nuclear_charge_by_atomic_number[z - 1])
    hardness = float(hardness_by_atomic_number[z - 1])
    charged = zeff + float(charge_e)
    if charged < 0.0:
        return exp(3.0)
    if charged == 0.0:
        return 0.0
    return exp(3.0 * (1.0 - exp(hardness * (1.0 - zeff / charged))))


def compile_gfnff_exppe_pair_table(
    candidate_atomic_numbers: Sequence[int],
    candidate_charges_e: Sequence[float],
    candidate_bonded_neighbor_counts: Sequence[int],
    candidate_coordination_numbers: Sequence[float],
    environment_atomic_numbers: Sequence[int],
    environment_charges_e: Sequence[float],
    environment_bonded_neighbor_counts: Sequence[int],
    environment_coordination_numbers: Sequence[float],
    *,
    repan_by_atomic_number: Sequence[float],
    repz_by_atomic_number: Sequence[float],
    sqrt_z_r4_over_r2_by_atomic_number: Sequence[float],
    effective_nuclear_charge_by_atomic_number: Sequence[float],
    hardness_by_atomic_number: Sequence[float],
    reference_dispersion: GFNFFReferenceDispersion,
    metal_atomic_numbers: Sequence[int] = (),
) -> ResolvedExpPEPairTable:
    """Compile charge/CN-corrected GFN-FF pairs into the common ZAFF Exp-PE table."""

    candidate_z = np.asarray(candidate_atomic_numbers, dtype=int).reshape(-1)
    environment_z = np.asarray(environment_atomic_numbers, dtype=int).reshape(-1)
    candidate_q = np.asarray(candidate_charges_e, dtype=float).reshape(-1)
    environment_q = np.asarray(environment_charges_e, dtype=float).reshape(-1)
    candidate_nb = np.asarray(candidate_bonded_neighbor_counts, dtype=int).reshape(-1)
    environment_nb = np.asarray(environment_bonded_neighbor_counts, dtype=int).reshape(-1)
    candidate_cn = np.asarray(candidate_coordination_numbers, dtype=float).reshape(-1)
    environment_cn = np.asarray(environment_coordination_numbers, dtype=float).reshape(-1)
    if not (
        candidate_z.shape == candidate_q.shape == candidate_nb.shape == candidate_cn.shape
        and environment_z.shape
        == environment_q.shape
        == environment_nb.shape
        == environment_cn.shape
    ):
        raise ValueError("GFN-FF atom-resolved input arrays have inconsistent dimensions")
    repan = np.asarray(repan_by_atomic_number, dtype=float)
    repz = np.asarray(repz_by_atomic_number, dtype=float)
    r4r2 = np.asarray(sqrt_z_r4_over_r2_by_atomic_number, dtype=float)
    c6 = reference_dispersion.cross_c6(
        candidate_z, candidate_cn, environment_z, environment_cn
    )
    alpha_atom_candidate = repan[candidate_z - 1] * (
        1.0 + 0.3480 * candidate_q
    ) * (1.0 - 0.1270 / (1.0 + candidate_nb.astype(float) ** 2))
    alpha_atom_environment = repan[environment_z - 1] * (
        1.0 + 0.3480 * environment_q
    ) * (1.0 - 0.1270 / (1.0 + environment_nb.astype(float) ** 2))
    zeta_candidate = np.asarray(
        [
            gfnff_charge_scale(z, q, effective_nuclear_charge_by_atomic_number, hardness_by_atomic_number)
            for z, q in zip(candidate_z, candidate_q, strict=True)
        ]
    )
    zeta_environment = np.asarray(
        [
            gfnff_charge_scale(z, q, effective_nuclear_charge_by_atomic_number, hardness_by_atomic_number)
            for z, q in zip(environment_z, environment_q, strict=True)
        ]
    )
    epsilon = np.empty_like(c6)
    r_min = np.empty_like(c6)
    alpha_exppe = np.empty_like(c6)
    metals = {int(value) for value in metal_atomic_numbers}
    for left, zi in enumerate(candidate_z):
        for right, zj in enumerate(environment_z):
            pair_scale = 1.0
            pair = {int(zi), int(zj)}
            if int(zi) == int(zj) == 1:
                pair_scale = 0.6290
            elif pair == {1, 6}:
                pair_scale = 0.91
            elif pair == {1, 8}:
                pair_scale = 1.04
            elif 1 in pair and any(value in metals for value in pair):
                pair_scale = 0.85
            repulsion_alpha = (
                float(np.sqrt(alpha_atom_candidate[left] * alpha_atom_environment[right]))
                * pair_scale
            )
            r4r2_pair = r4r2[int(zi) - 1] * r4r2[int(zj) - 1]
            source = FrozenGFNFFNonbondedPair(
                repulsion_prefactor=0.4270 * repz[int(zi) - 1] * repz[int(zj) - 1],
                repulsion_alpha=repulsion_alpha,
                c6=float(c6[left, right]),
                zeta_scale=float(zeta_candidate[left] * zeta_environment[right]),
                dispersion_scale=1.0,
                damping_r0_squared=(0.58 * np.sqrt(3.0 * r4r2_pair) + 4.80) ** 2,
                c8_coefficient=6.0 * r4r2_pair,
            )
            damping_radius = np.sqrt(source.damping_r0_squared)
            translated = source.to_exppe(
                max(0.25, 0.20 * damping_radius),
                4.0 * damping_radius,
            )
            epsilon[left, right] = translated.epsilon * HARTREE_TO_KCAL_PER_MOL
            r_min[left, right] = translated.r_min * BOHR_TO_ANGSTROM
            alpha_exppe[left, right] = translated.alpha
    return ResolvedExpPEPairTable(
        epsilon_kcal_per_mol=epsilon,
        r_min_angstrom=r_min,
        alpha=alpha_exppe,
        source="GFN-FF charge/CN corrected and compiled to ZAFF Exp-PE",
    )


@dataclass(frozen=True)
class DirectionalExpPEContact:
    """Exp-PE/Morse-polynomial directional contact.

    ``charge_beta`` and ``synthon_beta`` are fitted response coefficients.
    They modulate the radial well multiplicatively through a bounded
    exponential, so CM5 charge response and the intrinsic synthon identity
    affect both the energy and its gradient without changing the radial
    Morse-polynomial shape.  A zero coefficient reproduces the legacy
    directional contact exactly.
    """

    kind: str
    radial: DampedExpPEPotential
    angular_power: int
    source: str = "ZAFF-fast"
    charge_beta: float = 0.0
    synthon_beta: float = 0.0
    reference_charge_product: float = 0.0
    reference_synthon_score: float = 0.0
    charge_product: float = 0.0
    synthon_score: float = 0.0
    switch_on_angstrom: float | None = None
    switch_off_angstrom: float | None = None

    def __post_init__(self) -> None:
        kind = _normalise_directional_kind(self.kind)
        if kind not in {
            "hydrogen_bond", "alkali_bond", "alkaline_earth_bond", "regium_bond",
            "spodium_bond", "triel_bond", "tetrel_bond", "pnictogen_bond",
            "chalcogen_bond", "halogen_bond", "aerogen_bond", "cation_pi",
            "anion_pi", "lone_pair_pi", "pi_pi", "orthogonal",
        }:
            raise ValueError("unsupported directional noncovalent interaction kind")
        if (
            not isinstance(self.radial, DampedExpPEPotential)
            or int(self.angular_power) != self.angular_power
            or not 1 <= int(self.angular_power) <= 16
        ):
            raise ValueError("directional Exp-PE parameters are invalid")
        for name in (
            "charge_beta", "synthon_beta", "reference_charge_product",
            "reference_synthon_score", "charge_product", "synthon_score",
        ):
            if not isfinite(float(getattr(self, name))):
                raise ValueError("directional CM5/synthon modulation is invalid")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "angular_power", int(self.angular_power))
        if kind == "hydrogen_bond":
            on = self.radial.r_min + 1.0 if self.switch_on_angstrom is None else float(self.switch_on_angstrom)
            off = self.radial.r_min + 2.0 if self.switch_off_angstrom is None else float(self.switch_off_angstrom)
            if not isfinite(on) or not isfinite(off) or not self.radial.r_min < on < off:
                raise ValueError("invalid hydrogen-bond switch interval")
            object.__setattr__(self, "switch_on_angstrom", on)
            object.__setattr__(self, "switch_off_angstrom", off)

    def energy(
        self,
        distance: float | np.ndarray,
        donor_contact_acceptor_angle_radian: float | np.ndarray,
        *,
        radial_evaluation: str = "analytic",
        charge_product: float | np.ndarray | None = None,
        synthon_score: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Evaluate Exp-PE times an axial envelope without inverse trigonometry."""

        r = np.asarray(distance, dtype=float)
        angle = np.asarray(donor_contact_acceptor_angle_radian, dtype=float)
        if np.any(~np.isfinite(r)) or np.any(r <= 0.0) or np.any(~np.isfinite(angle)):
            raise ValueError("directional contact coordinates are invalid")
        radial = self.radial.epsilon * evaluate_exppe_dimensionless(
            r / self.radial.r_min,
            self.radial.alpha,
            radial_evaluation=radial_evaluation,
        )
        if self.kind == "hydrogen_bond":
            radial *= hydrogen_bond_radial_switch(r, self.switch_on_angstrom, self.switch_off_angstrom)
        axial = np.maximum(0.0, -np.cos(angle)) ** self.angular_power
        modulation = _directional_modulation(
            self.charge_product if charge_product is None else charge_product,
            self.synthon_score if synthon_score is None else synthon_score,
            charge_beta=self.charge_beta,
            synthon_beta=self.synthon_beta,
            reference_charge_product=self.reference_charge_product,
            reference_synthon_score=self.reference_synthon_score,
        )
        result = radial * axial * modulation
        return float(result) if result.ndim == 0 else result

    def energy_from_cosine(
        self,
        distance: float | np.ndarray,
        donor_contact_acceptor_cosine: float | np.ndarray,
        *,
        radial_evaluation: str = "lookup",
        charge_product: float | np.ndarray | None = None,
        synthon_score: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Fast path using a precomputed cosine; no ``acos`` is evaluated."""

        r = np.asarray(distance, dtype=float)
        cosine = np.asarray(donor_contact_acceptor_cosine, dtype=float)
        radial = self.radial.epsilon * evaluate_exppe_dimensionless(
            r / self.radial.r_min,
            self.radial.alpha,
            radial_evaluation=radial_evaluation,
        )
        if self.kind == "hydrogen_bond":
            radial *= hydrogen_bond_radial_switch(r, self.switch_on_angstrom, self.switch_off_angstrom)
        axial = np.maximum(0.0, -cosine)
        angular = axial.copy()
        for _ in range(1, self.angular_power):
            angular *= axial
        modulation = _directional_modulation(
            self.charge_product if charge_product is None else charge_product,
            self.synthon_score if synthon_score is None else synthon_score,
            charge_beta=self.charge_beta,
            synthon_beta=self.synthon_beta,
            reference_charge_product=self.reference_charge_product,
            reference_synthon_score=self.reference_synthon_score,
        )
        result = radial * angular * modulation
        return float(result) if result.ndim == 0 else result

    def energy_from_vectors(
        self,
        distance: float | np.ndarray,
        contact_to_donor: np.ndarray,
        contact_to_acceptor: np.ndarray,
        *,
        radial_evaluation: str = "lookup",
        charge_product: float | np.ndarray = 0.0,
        synthon_score: float | np.ndarray = 0.0,
    ) -> float | np.ndarray:
        """Vectorized dot-product path for H-bond and XB axes."""

        donor = np.asarray(contact_to_donor, dtype=float)
        acceptor = np.asarray(contact_to_acceptor, dtype=float)
        if donor.shape != acceptor.shape or donor.shape[-1] != 3:
            raise ValueError("directional contact vectors must share shape (..., 3)")
        denominator = np.linalg.norm(donor, axis=-1) * np.linalg.norm(
            acceptor, axis=-1
        )
        if np.any(denominator <= 0.0):
            raise ValueError("directional contact vectors cannot have zero length")
        cosine = np.sum(donor * acceptor, axis=-1) / denominator
        return self.energy_from_cosine(
            distance,
            np.clip(cosine, -1.0, 1.0),
            radial_evaluation=radial_evaluation,
            charge_product=charge_product,
            synthon_score=synthon_score,
        )


@dataclass(frozen=True)
class ZAFFDirectionalParameters:
    """Resident ZAFF-fast H-bond and sigma/pi-hole shape parameters.

    These values are part of the ZAFF-fast model definition.  They are not
    loaded from xTB or any other external force field at runtime and can be
    optimized in a future ZAFF parameterization campaign.
    """

    hbacut: float = 49.0
    hbscut: float = 22.0
    xbacut: float = 70.0
    xbscut: float = 5.0
    hbalp: float = 6.0
    hblongcut: float = 85.0
    hblongcut_xb: float = 70.0
    hbst: float = 15.0
    hbsf: float = 1.0
    xbst: float = 15.0
    xbsf: float = 0.03

    def __post_init__(self) -> None:
        if any(not isfinite(float(value)) or float(value) <= 0.0 for value in self.__dict__.values()):
            raise ValueError("ZAFF directional constants must be positive and finite")


def zaff_directional_charge_factor(
    kind: str,
    donor_charge: float,
    acceptor_charge: float,
    hydrogen_charge: float | None = None,
    *,
    parameters: ZAFFDirectionalParameters | None = None,
) -> float:
    """Return the resident ZAFF CM5 multiplier for directional terms."""

    p = ZAFFDirectionalParameters() if parameters is None else parameters
    normalized = _normalise_directional_kind(kind)
    charges = (donor_charge, acceptor_charge, hydrogen_charge)
    if any(value is not None and not isfinite(float(value)) for value in charges):
        raise ValueError("ZAFF directional charges must be finite")
    if normalized == "hydrogen_bond":
        qscale = p.hbst
        floor = p.hbsf
        donor = exp(-qscale * float(donor_charge)) / (exp(-qscale * float(donor_charge)) + floor)
        acceptor = exp(-qscale * float(acceptor_charge)) / (exp(-qscale * float(acceptor_charge)) + floor)
        if hydrogen_charge is None:
            return float(donor * acceptor)
        hterm = exp(qscale * float(hydrogen_charge)) / (exp(qscale * float(hydrogen_charge)) + floor)
        return float(donor * acceptor * hterm)
    if normalized in {
        "halogen_bond",
        "chalcogen_bond",
        "pnictogen_bond",
        "tetrel_bond",
    }:
        donor = exp(-p.xbst * float(donor_charge)) / (exp(-p.xbst * float(donor_charge)) + p.xbsf)
        acceptor = exp(-p.xbst * float(acceptor_charge)) / (exp(-p.xbst * float(acceptor_charge)) + p.xbsf)
        return float(donor * acceptor)
    raise ValueError(
        "ZAFF CM5 scaling is implemented for H-bond and X-bond plus "
        "chalcogen-, pnictogen- and tetrel-bond only"
    )


def compile_zaff_directional_exppe_contact(
    kind: str,
    *,
    radial_scale_angstrom: float,
    amplitude_kcal_per_mol: float,
    angular_power: int = 4,
    parameters: ZAFFDirectionalParameters | None = None,
    charge_beta: float = 0.0,
    synthon_beta: float = 0.0,
    charge_product: float = 0.0,
    synthon_score: float = 0.0,
) -> DirectionalExpPEContact:
    """Compile the resident ZAFF directional damping to Morse-polynomial form.

    The resident radial damping is sampled analytically, its minimum and second
    derivative are obtained numerically, and ``damped_exppe_from_minimum``
    preserves both observables.  No external force-field call or
    geometry-specific fit is introduced.  CM5 and synthon dependence remain
    explicit modulation terms.
    """

    p = ZAFFDirectionalParameters() if parameters is None else parameters
    normalized = _normalise_directional_kind(kind)
    supported = {
        "hydrogen_bond",
        "halogen_bond",
        "chalcogen_bond",
        "pnictogen_bond",
        "tetrel_bond",
    }
    if normalized not in supported:
        raise ValueError("unsupported ZAFF directional interaction kind")
    scale = float(radial_scale_angstrom)
    amplitude = float(amplitude_kcal_per_mol)
    if not isfinite(scale) or scale <= 0.0 or not isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("directional radial scale and amplitude must be positive")
    short_parameter = p.hbscut if normalized == "hydrogen_bond" else p.xbscut
    long_parameter = p.hblongcut if normalized == "hydrogen_bond" else p.hblongcut_xb
    # xTB stores these constants in its internal squared-distance convention.
    # Normalize their ratio to the supplied atom-pair length so the translated
    # potential is unit-consistent in Angstrom and has the same dimensionless
    # preserve the dimensionless ZAFF damping balance.
    ratio = (short_parameter / long_parameter) ** 0.25
    short = scale * ratio
    long = scale / ratio
    alpha = p.hbalp

    def native(distance: float) -> float:
        r = float(distance)
        short_term = 1.0 / (1.0 + (short / r) ** (2.0 * alpha))
        long_term = 1.0 / (1.0 + (r / long) ** (2.0 * alpha))
        return -amplitude * short_term * long_term

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(native, bounds=(0.35 * scale, 4.5 * scale), method="bounded")
    if not result.success:
        raise RuntimeError("ZAFF directional radial minimum search failed")
    rmin = float(result.x)
    h = min(1.0e-4 * rmin, 0.01 * scale)
    curvature = (native(rmin + h) - 2.0 * native(rmin) + native(rmin - h)) / (h * h)
    depth = -native(rmin)
    # The native XB damping is sharper than the bounded Exp-PE
    # parameter domain.  Preserve the well and position exactly and cap only
    # the curvature at the representable ZAFF range.
    curvature = min(max(curvature, 0.25 * depth / rmin**2), 100.0 * depth / rmin**2)
    translated = damped_exppe_from_minimum(
        MinimumDerivatives(epsilon=depth, r_min=rmin, second=curvature)
    )
    return DirectionalExpPEContact(
        kind=normalized,
        radial=translated,
        angular_power=angular_power,
        charge_beta=charge_beta,
        synthon_beta=synthon_beta,
        charge_product=charge_product,
        synthon_score=synthon_score,
        source=(
            "ZAFF-fast resident directional shape translated to Exp-PE; "
            "interaction radial scale and amplitude supplied explicitly"
        ),
    )


def fit_directional_exppe_contact(
    kind: str,
    distances: Sequence[float],
    donor_contact_acceptor_angles_radian: Sequence[float],
    source_total_energies: Sequence[float],
    *,
    baseline_energies: Sequence[float],
    reference_distance: float | None = None,
    charge_products: Sequence[float] | None = None,
    synthon_scores: Sequence[float] | None = None,
) -> DirectionalExpPEContact:
    """Fit a directional residual to the ZAFF Exp-PE form.

    ``baseline_energies`` must contain electrostatics plus the isotropic
    non-bonded model evaluated for the same samples. Requiring it explicitly
    prevents those contributions from being absorbed a second time into the
    H-bond or XB term.
    """

    r = np.asarray(distances, dtype=float).reshape(-1)
    angle = np.asarray(donor_contact_acceptor_angles_radian, dtype=float).reshape(-1)
    source = np.asarray(source_total_energies, dtype=float).reshape(-1)
    baseline = np.asarray(baseline_energies, dtype=float).reshape(-1)
    energy = source - baseline
    has_modulation = charge_products is not None or synthon_scores is not None
    charge = np.zeros_like(r) if charge_products is None else np.asarray(charge_products, dtype=float).reshape(-1)
    synthon = np.zeros_like(r) if synthon_scores is None else np.asarray(synthon_scores, dtype=float).reshape(-1)
    if (
        len(r) < 6
        or angle.shape != r.shape
        or source.shape != r.shape
        or baseline.shape != r.shape
        or charge.shape != r.shape
        or synthon.shape != r.shape
        or np.any(~np.isfinite(r))
        or np.any(r <= 0.0)
        or np.any(~np.isfinite(angle))
        or np.any(~np.isfinite(source))
        or np.any(~np.isfinite(baseline))
        or np.any(~np.isfinite(charge))
        or np.any(~np.isfinite(synthon))
    ):
        raise ValueError("directional fit samples are insufficient or invalid")
    r0 = (
        float(r[int(np.argmin(energy))])
        if reference_distance is None
        else float(reference_distance)
    )
    if not isfinite(r0) or r0 <= 0.0:
        raise ValueError("directional contact reference distance is invalid")
    depth = max(1.0e-10, -float(np.min(energy)))
    initial = np.asarray((np.log(depth), np.log(r0), np.log(8.0), np.log(4.0)))
    if has_modulation:
        initial = np.concatenate((initial, (0.0, 0.0)))

    def residual(parameters: np.ndarray) -> np.ndarray:
        epsilon = np.exp(parameters[0])
        fitted_r0 = np.exp(parameters[1])
        alpha = 4.0 + np.exp(parameters[2])
        angular_power = np.exp(parameters[3])
        radial = epsilon * evaluate_exppe_dimensionless(
            r / fitted_r0,
            alpha,
            radial_evaluation="analytic",
        )
        if _normalise_directional_kind(kind) == "hydrogen_bond":
            radial *= hydrogen_bond_radial_switch(
                r, fitted_r0 + 1.0, fitted_r0 + 2.0
            )
        axial = np.maximum(0.0, -np.cos(angle)) ** angular_power
        modulation = (
            _directional_modulation(
                charge, synthon,
                charge_beta=parameters[4], synthon_beta=parameters[5],
                reference_charge_product=0.0, reference_synthon_score=0.0,
            )
            if has_modulation else 1.0
        )
        return axial * radial * modulation - energy

    fit = least_squares(
        residual,
        initial,
        bounds=(
            np.asarray((np.log(1.0e-12), np.log(0.5 * r0), np.log(0.05), np.log(1.0), -8.0, -8.0) if has_modulation else (np.log(1.0e-12), np.log(0.5 * r0), np.log(0.05), np.log(1.0))),
            np.asarray((np.log(1.0e3 * depth), np.log(1.5 * r0), np.log(60.0), np.log(16.0), 8.0, 8.0) if has_modulation else (np.log(1.0e3 * depth), np.log(1.5 * r0), np.log(60.0), np.log(16.0))),
        ),
        max_nfev=5000,
    )
    if not fit.success:
        raise RuntimeError(f"directional fit failed: {fit.message}")
    fitted_power = int(np.clip(np.rint(np.exp(fit.x[3])), 1, 16))
    return DirectionalExpPEContact(
        kind=kind,
        radial=DampedExpPEPotential(
            epsilon=float(np.exp(fit.x[0])),
            r_min=float(np.exp(fit.x[1])),
            alpha=float(4.0 + np.exp(fit.x[2])),
        ),
        angular_power=fitted_power,
        charge_beta=float(fit.x[4]) if has_modulation else 0.0,
        synthon_beta=float(fit.x[5]) if has_modulation else 0.0,
        reference_charge_product=0.0,
        reference_synthon_score=0.0,
        source="ZAFF directional residual after electrostatics/Exp-PE baseline",
    )


# Public compatibility name retained for callers predating the generalized
# ZAFF directional fit.  Keep it as an alias so signature introspection and
# serialized callable references continue to resolve without a wrapper layer.
fit_gfnff_directional_exppe_contact = fit_directional_exppe_contact


def _directional_modulation(
    charge_product: float | np.ndarray,
    synthon_score: float | np.ndarray,
    *,
    charge_beta: float,
    synthon_beta: float,
    reference_charge_product: float,
    reference_synthon_score: float,
) -> float | np.ndarray:
    """Bounded CM5/synthon response multiplier for directional wells."""

    charge = np.asarray(charge_product, dtype=float)
    synthon = np.asarray(synthon_score, dtype=float)
    if charge.shape != synthon.shape or np.any(~np.isfinite(charge)) or np.any(~np.isfinite(synthon)):
        raise ValueError("directional CM5/synthon descriptors have inconsistent shapes")
    exponent = float(charge_beta) * (charge - float(reference_charge_product))
    exponent += float(synthon_beta) * (synthon - float(reference_synthon_score))
    result = np.exp(np.clip(exponent, -4.0, 4.0))
    return float(result) if result.ndim == 0 else result


def _normalise_directional_kind(kind: str) -> str:
    """Canonicalise H-bond/X-bond aliases for the shared ZAFF library."""

    value = str(kind).strip().lower().replace("-", "_")
    return {
        "hbond": "hydrogen_bond",
        "xbond": "halogen_bond",
        "x_bond": "halogen_bond",
    }.get(value, value)
