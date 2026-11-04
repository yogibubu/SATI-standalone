"""Analytic non-bonded Cartesian Hessians shared by ORACLE and TRINITY."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from matrix_core import read_sectioned_lines, section_content

from .continuous_nonbonded import continuous_topology_scaled_pair_derivatives, continuous_topology_pair_scales
from .spatial_regions import bounded_topological_distances
from .topology.covalent_radii import covalent_radius
from .topology.vdw_radii import uff_vdw_radius, uff_well_depth_kcal


BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631
UFF_PARAMETER_REFERENCE = "Rappe et al., JACS 114, 10024 (1992), DOI:10.1021/ja00051a040"


@dataclass(frozen=True)
class NonbondedHessianComponents:
    """Analytic radial energies, gradients and Cartesian Hessians."""

    electrostatic: np.ndarray
    uff_vdw: np.ndarray
    electrostatic_energy: float
    uff_vdw_energy: float
    electrostatic_gradient: np.ndarray
    uff_vdw_gradient: np.ndarray
    charge_source: str = "NONE"
    one_four_scale: float = 0.5
    vdw_model: str = "uff_12_6"
    vdw_radius_scale: float = 1.0
    vdw_well_depth_scale: float = 1.0
    topology_scaling: str = "continuous_erf_paths"

    @property
    def total(self) -> np.ndarray:
        return np.asarray(self.electrostatic) + np.asarray(self.uff_vdw)

    @property
    def total_energy(self) -> float:
        return float(self.electrostatic_energy + self.uff_vdw_energy)

    @property
    def total_gradient(self) -> np.ndarray:
        return np.asarray(self.electrostatic_gradient) + np.asarray(self.uff_vdw_gradient)


@dataclass(frozen=True)
class _RadialDerivative:
    energy: float
    first: float
    second: float


def synthon_charges_from_xyzin(path: Path | str, natoms: int) -> tuple[np.ndarray, str]:
    """Read the complete canonical charge vector from ``#SYNTHONS``."""
    content = section_content(read_sectioned_lines(Path(path)), "SYNTHONS")
    charges = np.full(int(natoms), np.nan, dtype=float)
    source = "ORACLE electronegativity estimate"
    atom_column = 0
    charge_column = 3
    for raw in content:
        text = raw.strip()
        if not text:
            continue
        upper = text.upper()
        if upper.startswith("CHARGE_SOURCE"):
            fields = text.split(None, 1)
            source = fields[1].strip() if len(fields) == 2 else source
            continue
        if upper.startswith("COLUMNS "):
            columns = [field.upper() for field in text.split()[1:]]
            if "ATOM" in columns:
                atom_column = columns.index("ATOM")
            if "CHARGE" in columns:
                charge_column = columns.index("CHARGE")
            continue
        if upper.startswith(
            ("SCHEMA ", "INDEXING ", "BOND_ORDER_SOURCE", "BOND_MULTIPLICITY_SOURCE", "COLUMNS ")
        ):
            continue
        parts = text.split()
        if len(parts) <= max(atom_column, charge_column):
            continue
        try:
            idx = int(parts[atom_column]) - 1
            charge = float(parts[charge_column])
        except ValueError:
            continue
        if 0 <= idx < len(charges):
            charges[idx] = charge
    if np.any(~np.isfinite(charges)):
        raise ValueError("#SYNTHONS does not contain a complete charge column")
    return charges, source


def preferred_atomic_charges_from_xyzin(
    path: Path | str,
    natoms: int,
    *,
    hessian_charges: np.ndarray | tuple[float, ...] | None = None,
    hessian_charge_model: str = "",
) -> tuple[np.ndarray, str]:
    """Return CM5 from the Hessian or the canonical ORACLE charge vector.

    Non-CM5 QM populations are never promoted to a competing charge model.
    The frozen synthon vector is already CM5 when the complete paired QM level
    exists and the electronegativity estimate otherwise.
    """
    if hessian_charges is not None:
        values = np.asarray(hessian_charges, dtype=float).reshape(-1)
        if values.size:
            if values.size != int(natoms) or np.any(~np.isfinite(values)):
                raise ValueError("#CARTESIAN_HESSIAN atomic charges are incomplete")
            model = str(hessian_charge_model).strip() or "unspecified charges"
            if "CM5" in model.upper():
                return values, f"#CARTESIAN_HESSIAN {model}"

    values, source = synthon_charges_from_xyzin(path, natoms)
    return values, f"#SYNTHONS {source}"


def topology_bonds_from_xyzin(path: Path | str) -> tuple[tuple[int, int], ...]:
    """Read one-based covalent topology bonds from a MATRIX xyzin file."""
    topology = section_content(read_sectioned_lines(Path(path)), "TOPOLOGY")
    bonds: list[tuple[int, int]] = []
    in_bonds = False
    for raw in topology:
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            in_bonds = text.upper() == "[BONDS]"
            continue
        if not in_bonds or not text or text.upper() == "NONE":
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        try:
            left, right = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if left != right:
            bonds.append(tuple(sorted((left, right))))
    return tuple(dict.fromkeys(bonds))


def nonbonded_cartesian_hessian_components(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray | tuple[int, ...],
    topology_bonds: tuple[tuple[int, int], ...],
    *,
    charges: np.ndarray | None = None,
    charge_source: str = "NONE",
    electrostatic: bool = False,
    uff_vdw: bool = False,
    vdw_model: str = "uff_12_6",
    one_four_scale: float = 0.5,
    vdw_radius_scale: float = 1.0,
    vdw_well_depth_scale: float = 1.0,
    vdw_damping_scale: float = 0.80,
    vdw_damping_exponent: int = 24,
    electrostatic_model: str = "topology_scaled_point_charges",
    electrostatic_gaussian_width_scale: float = 1.0,
    topology_scaling: str = "continuous_erf_paths",
    topology_switch_alpha_per_angstrom: float = 8.0,
) -> NonbondedHessianComponents:
    """Return radial energy, Cartesian gradient and Hessian components.

    All three derivative orders are evaluated analytically from the same
    pair potential.  The Cartesian derivatives use the exact radial chain
    rule; no finite differences enter the production path.
    """
    coords = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    natoms = len(numbers)
    if coords.shape != (natoms, 3):
        raise ValueError("non-bonded coordinates must have shape (natoms, 3)")
    if np.any(~np.isfinite(coords)):
        raise ValueError("non-bonded coordinates must be finite")
    scale_14 = float(one_four_scale)
    if not np.isfinite(scale_14) or scale_14 < 0.0:
        raise ValueError("1-4 non-bonded scale factor must be finite and non-negative")
    radius_scale = float(vdw_radius_scale)
    well_depth_scale = float(vdw_well_depth_scale)
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("van der Waals radius scale must be finite and positive")
    if not np.isfinite(well_depth_scale) or well_depth_scale <= 0.0:
        raise ValueError("van der Waals well-depth scale must be finite and positive")
    normalized_topology_scaling = str(topology_scaling).strip().lower()
    if normalized_topology_scaling not in {
        "collective_all_pairs",
        "all_pairs_radial_damping",
        "continuous_erf_paths",
        "discrete_graph",
    }:
        raise ValueError(
            "topology scaling must be collective_all_pairs, all_pairs_radial_damping, "
            "continuous_erf_paths or discrete_graph"
        )
    distances = None
    if normalized_topology_scaling == "discrete_graph":
        graph = _zero_based_graph(natoms, topology_bonds)
        distances = _topological_distances(graph, natoms)
    zero_hessian = np.zeros((3 * natoms, 3 * natoms), dtype=float)
    zero_gradient = np.zeros(3 * natoms, dtype=float)

    electrostatic_energy = 0.0
    electrostatic_gradient = zero_gradient.copy()
    electrostatic_hessian = zero_hessian.copy()
    if electrostatic:
        if charges is None:
            raise ValueError("electrostatic Hessian correction requires atomic charges")
        charge_values = np.asarray(charges, dtype=float).reshape(-1)
        if charge_values.size != natoms or np.any(~np.isfinite(charge_values)):
            raise ValueError("electrostatic Hessian correction requires one finite charge per atom")
        normalized_electrostatic_model = str(electrostatic_model).strip().lower()
        if normalized_electrostatic_model == "gaussian_erf_all_pairs":
            electrostatic_energy, electrostatic_gradient, electrostatic_hessian = (
                _gaussian_electrostatic_derivatives(
                    coords,
                    numbers,
                    charge_values,
                    width_scale=float(electrostatic_gaussian_width_scale),
                )
            )
        elif normalized_electrostatic_model != "topology_scaled_point_charges":
            raise ValueError(
                "electrostatic model must be gaussian_erf_all_pairs or "
                "topology_scaled_point_charges"
            )
        elif normalized_topology_scaling == "continuous_erf_paths":

            def electrostatic_factory(i: int, j: int, distance: float):
                product = float(charge_values[i]) * float(charge_values[j])
                return (
                    None
                    if abs(product) <= 1.0e-16
                    else _electrostatic_radial_derivative(distance, product, 1.0)
                )

            electrostatic_energy, electrostatic_gradient, electrostatic_hessian = (
                continuous_topology_scaled_pair_derivatives(
                    coords,
                    numbers,
                    electrostatic_factory,
                    one_four_scale=scale_14,
                    switch_alpha_per_angstrom=topology_switch_alpha_per_angstrom,
                )
            )
        else:
            electrostatic_energy, electrostatic_gradient, electrostatic_hessian = (
                _electrostatic_derivatives(
                    coords,
                    charge_values,
                    distances or {},
                    one_four_scale=scale_14,
                )
            )
    uff_energy = 0.0
    uff_gradient = zero_gradient.copy()
    uff_hessian = zero_hessian.copy()
    normalized_vdw_model = str(vdw_model).strip().lower()
    if normalized_vdw_model not in {
        "uff_12_6",
        "exppe_from_uff",
        "damped_exppe_from_uff",
    }:
        raise ValueError(
            "vdW model must be uff_12_6, exppe_from_uff or damped_exppe_from_uff"
        )
    if uff_vdw:
        if normalized_topology_scaling == "continuous_erf_paths":

            def vdw_factory(i: int, j: int, distance: float):
                xij_bohr, dij_hartree = uff_pair_parameters(int(numbers[i]), int(numbers[j]))
                xij_bohr *= radius_scale
                dij_hartree *= well_depth_scale
                if normalized_vdw_model == "damped_exppe_from_uff":
                    return _damped_exppe_radial_derivative(
                        distance,
                        xij_bohr,
                        dij_hartree,
                        1.0,
                        damping_scale=float(vdw_damping_scale),
                        damping_exponent=int(vdw_damping_exponent),
                    )
                return (
                    _exppe_radial_derivative(distance, xij_bohr, dij_hartree, 1.0)
                    if normalized_vdw_model == "exppe_from_uff"
                    else _uff_radial_derivative(distance, xij_bohr, dij_hartree, 1.0)
                )

            uff_energy, uff_gradient, uff_hessian = continuous_topology_scaled_pair_derivatives(
                coords,
                numbers,
                vdw_factory,
                one_four_scale=scale_14,
                switch_alpha_per_angstrom=topology_switch_alpha_per_angstrom,
            )
        else:
            uff_energy, uff_gradient, uff_hessian = _uff_vdw_derivatives(
                coords,
                numbers,
                distances or {},
                one_four_scale=scale_14,
                vdw_model=normalized_vdw_model,
                vdw_radius_scale=radius_scale,
                vdw_well_depth_scale=well_depth_scale,
                vdw_damping_scale=float(vdw_damping_scale),
                vdw_damping_exponent=int(vdw_damping_exponent),
            )
    return NonbondedHessianComponents(
        electrostatic=_symmetrize(electrostatic_hessian),
        uff_vdw=_symmetrize(uff_hessian),
        electrostatic_energy=float(electrostatic_energy),
        uff_vdw_energy=float(uff_energy),
        electrostatic_gradient=np.asarray(electrostatic_gradient, dtype=float),
        uff_vdw_gradient=np.asarray(uff_gradient, dtype=float),
        charge_source=str(charge_source) if electrostatic else "NONE",
        one_four_scale=scale_14,
        vdw_model=normalized_vdw_model,
        vdw_radius_scale=radius_scale,
        vdw_well_depth_scale=well_depth_scale,
        topology_scaling=normalized_topology_scaling,
    )


def nonbonded_cartesian_energy(coordinates_bohr, atomic_numbers, topology_bonds, **settings):
    """Same pair energy as the E/G/H owner, without differentiating path weights.

    Continuous-topology Gaussian/point electrostatics and UFF/Exp-PE forms use
    their existing radial kernels. Other policies delegate to the reference
    owner. No derivative or Hessian is built for an energy-only request.
    """
    supported={'charges','charge_source','electrostatic','uff_vdw','vdw_model',
        'one_four_scale','vdw_radius_scale','vdw_well_depth_scale','vdw_damping_scale',
        'vdw_damping_exponent','electrostatic_model','electrostatic_gaussian_width_scale',
        'topology_scaling','topology_switch_alpha_per_angstrom'}
    fallback=lambda:nonbonded_cartesian_hessian_components(
        coordinates_bohr,atomic_numbers,topology_bonds,**settings).total_energy
    if (set(settings)-supported or
            str(settings.get('topology_scaling','continuous_erf_paths')).strip().lower()!='continuous_erf_paths'):
        return fallback()
    coords=np.asarray(coordinates_bohr,dtype=float)
    numbers=np.asarray(atomic_numbers,dtype=int).reshape(-1)
    if coords.shape!=(len(numbers),3) or not np.all(np.isfinite(coords)):
        return fallback()
    scale14=float(settings.get('one_four_scale',.5))
    radius=float(settings.get('vdw_radius_scale',1.))
    well=float(settings.get('vdw_well_depth_scale',1.))
    if not np.isfinite(scale14) or scale14<0 or not np.isfinite(radius) or radius<=0 or not np.isfinite(well) or well<=0:
        return fallback()
    vdw=bool(settings.get('uff_vdw',False))
    electrostatic=bool(settings.get('electrostatic',False))
    model=str(settings.get('vdw_model','uff_12_6')).strip().lower()
    emodel=str(settings.get('electrostatic_model','topology_scaled_point_charges')).strip().lower()
    if model not in {'uff_12_6','exppe_from_uff','damped_exppe_from_uff'}:
        return fallback()
    charges=None;widths=None
    if electrostatic:
        if settings.get('charges') is None:return fallback()
        charges=np.asarray(settings['charges'],dtype=float).reshape(-1)
        if len(charges)!=len(numbers) or not np.all(np.isfinite(charges)):
            return fallback()
        if emodel not in {'gaussian_erf_all_pairs','topology_scaled_point_charges'}:
            return fallback()
        if emodel=='gaussian_erf_all_pairs':
            width_scale=float(settings.get('electrostatic_gaussian_width_scale',1.))
            if not np.isfinite(width_scale) or width_scale<=0:return fallback()
            widths=np.array([width_scale*float(covalent_radius(int(z)) or .75)/BOHR_TO_ANGSTROM for z in numbers])
    if not vdw and not electrostatic:return 0.
    scales=(continuous_topology_pair_scales(coords,numbers,one_four_scale=scale14,
        switch_alpha_per_angstrom=float(settings.get('topology_switch_alpha_per_angstrom',8.)))
        if vdw or emodel=='topology_scaled_point_charges' else {})
    ee=0.;ev=0.
    for i in range(len(numbers)):
        for j in range(i+1,len(numbers)):
            product=float(charges[i])*float(charges[j]) if electrostatic else 0.
            if not vdw and emodel=='gaussian_erf_all_pairs' and abs(product)<=1e-16:
                continue
            distance=_pair_distance(coords[i]-coords[j],i,j)
            if electrostatic and abs(product)>1e-16:
                if emodel=='gaussian_erf_all_pairs':
                    ee+=_gaussian_electrostatic_radial_derivative(distance,product,widths[i],widths[j]).energy
                else:
                    ee+=scales[(i+1,j+1)]*_electrostatic_radial_derivative(distance,product,1.).energy
            if vdw:
                rmin,epsilon=uff_pair_parameters(int(numbers[i]),int(numbers[j]))
                rmin*=radius;epsilon*=well
                if model=='exppe_from_uff':
                    value=_exppe_radial_derivative(distance,rmin,epsilon,1.).energy
                elif model=='damped_exppe_from_uff':
                    value=_damped_exppe_radial_derivative(distance,rmin,epsilon,1.,
                        damping_scale=float(settings.get('vdw_damping_scale',.8)),
                        damping_exponent=int(settings.get('vdw_damping_exponent',24))).energy
                else:
                    value=_uff_radial_derivative(distance,rmin,epsilon,1.).energy
                ev+=scales[(i+1,j+1)]*value
    return float(ee+ev)


def continuous_electrostatic_pair_components(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray | tuple[int, ...],
    *,
    one_four_scale: float = 0.5,
    topology_switch_alpha_per_angstrom: float = 8.0,
) -> dict[tuple[int, int], tuple[float, np.ndarray, np.ndarray]]:
    """Cache unit-charge pair E/G/H contributions at one fixed geometry."""

    coords = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    components: dict[tuple[int, int], tuple[float, np.ndarray, np.ndarray]] = {}

    def unit_coulomb_factory(_i: int, _j: int, distance: float):
        return _electrostatic_radial_derivative(distance, 1.0, 1.0)

    continuous_topology_scaled_pair_derivatives(
        coords,
        numbers,
        unit_coulomb_factory,
        one_four_scale=float(one_four_scale),
        switch_alpha_per_angstrom=float(topology_switch_alpha_per_angstrom),
        pair_components=components,
    )
    return components


def nonbonded_cartesian_hessian_correction(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray | tuple[int, ...],
    topology_bonds: tuple[tuple[int, int], ...],
    *,
    charges: np.ndarray | None = None,
    electrostatic: bool = False,
    uff_vdw: bool = False,
    vdw_model: str = "uff_12_6",
    one_four_scale: float = 0.5,
    vdw_radius_scale: float = 1.0,
    vdw_well_depth_scale: float = 1.0,
    vdw_damping_scale: float = 0.80,
    vdw_damping_exponent: int = 24,
    electrostatic_model: str = "topology_scaled_point_charges",
    electrostatic_gaussian_width_scale: float = 1.0,
    topology_scaling: str = "continuous_erf_paths",
    topology_switch_alpha_per_angstrom: float = 8.0,
) -> np.ndarray:
    """Build an analytic radial Cartesian Hessian for later basis projection."""
    return nonbonded_cartesian_hessian_components(
        coordinates_bohr,
        atomic_numbers,
        topology_bonds,
        charges=charges,
        electrostatic=electrostatic,
        uff_vdw=uff_vdw,
        vdw_model=vdw_model,
        one_four_scale=one_four_scale,
        vdw_radius_scale=vdw_radius_scale,
        vdw_well_depth_scale=vdw_well_depth_scale,
        vdw_damping_scale=vdw_damping_scale,
        vdw_damping_exponent=vdw_damping_exponent,
        electrostatic_model=electrostatic_model,
        electrostatic_gaussian_width_scale=electrostatic_gaussian_width_scale,
        topology_scaling=topology_scaling,
        topology_switch_alpha_per_angstrom=topology_switch_alpha_per_angstrom,
    ).total


def _electrostatic_derivatives(
    coordinates_bohr: np.ndarray,
    charges: np.ndarray,
    distances: dict[tuple[int, int], int],
    *,
    one_four_scale: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    natoms = len(charges)
    energy = 0.0
    gradient = np.zeros(3 * natoms, dtype=float)
    hessian = np.zeros((3 * natoms, 3 * natoms), dtype=float)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            pair_scale = _nonbonded_pair_scale(distances.get((i, j), 999999), one_four_scale)
            if pair_scale == 0.0:
                continue
            qiqj = float(charges[i]) * float(charges[j])
            if abs(qiqj) <= 1.0e-16:
                continue
            rvec = coordinates_bohr[i] - coordinates_bohr[j]
            r = _pair_distance(rvec, i, j)
            radial = _electrostatic_radial_derivative(r, qiqj, pair_scale)
            energy += radial.energy
            _accumulate_pair_derivatives(gradient, hessian, i, j, rvec, radial)
    return energy, gradient, hessian


def _uff_vdw_derivatives(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    distances: dict[tuple[int, int], int],
    *,
    one_four_scale: float,
    vdw_model: str,
    vdw_radius_scale: float,
    vdw_well_depth_scale: float,
    vdw_damping_scale: float = 0.80,
    vdw_damping_exponent: int = 24,
) -> tuple[float, np.ndarray, np.ndarray]:
    natoms = len(atomic_numbers)
    energy = 0.0
    gradient = np.zeros(3 * natoms, dtype=float)
    hessian = np.zeros((3 * natoms, 3 * natoms), dtype=float)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            pair_scale = _nonbonded_pair_scale(distances.get((i, j), 999999), one_four_scale)
            if pair_scale == 0.0:
                continue
            xij_bohr, dij_hartree = uff_pair_parameters(
                int(atomic_numbers[i]), int(atomic_numbers[j])
            )
            xij_bohr *= float(vdw_radius_scale)
            dij_hartree *= float(vdw_well_depth_scale)
            rvec = coordinates_bohr[i] - coordinates_bohr[j]
            r = _pair_distance(rvec, i, j)
            if vdw_model == "damped_exppe_from_uff":
                radial = _damped_exppe_radial_derivative(
                    r,
                    xij_bohr,
                    dij_hartree,
                    pair_scale,
                    damping_scale=vdw_damping_scale,
                    damping_exponent=vdw_damping_exponent,
                )
            elif vdw_model == "exppe_from_uff":
                radial = _exppe_radial_derivative(r, xij_bohr, dij_hartree, pair_scale)
            else:
                radial = _uff_radial_derivative(r, xij_bohr, dij_hartree, pair_scale)
            energy += radial.energy
            _accumulate_pair_derivatives(gradient, hessian, i, j, rvec, radial)
    return energy, gradient, hessian


def _electrostatic_radial_derivative(
    distance_bohr: float,
    charge_product: float,
    pair_scale: float,
) -> _RadialDerivative:
    """Return V, dV/dr and d²V/dr² for q_i q_j/r in atomic units."""
    r = float(distance_bohr)
    factor = float(pair_scale) * float(charge_product)
    return _RadialDerivative(factor / r, -factor / r**2, 2.0 * factor / r**3)


def _gaussian_electrostatic_derivatives(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    charges: np.ndarray,
    *,
    width_scale: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    scale = float(width_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("electrostatic Gaussian width scale must be finite and positive")
    widths = np.asarray(
        [
            scale * float(covalent_radius(int(number)) or 0.75) / BOHR_TO_ANGSTROM
            for number in atomic_numbers
        ],
        dtype=float,
    )
    natoms = len(atomic_numbers)
    energy = 0.0
    gradient = np.zeros(3 * natoms, dtype=float)
    hessian = np.zeros((3 * natoms, 3 * natoms), dtype=float)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            product = float(charges[i]) * float(charges[j])
            if abs(product) <= 1.0e-16:
                continue
            rvec = coordinates_bohr[i] - coordinates_bohr[j]
            distance = _pair_distance(rvec, i, j)
            radial = _gaussian_electrostatic_radial_derivative(
                distance,
                product,
                widths[i],
                widths[j],
            )
            energy += radial.energy
            _accumulate_pair_derivatives(gradient, hessian, i, j, rvec, radial)
    return energy, gradient, hessian


def _gaussian_electrostatic_radial_derivative(
    distance_bohr: float,
    charge_product: float,
    width_i_bohr: float,
    width_j_bohr: float,
) -> _RadialDerivative:
    """Derivatives for Coulomb interaction between normalized Gaussian charges."""

    r = float(distance_bohr)
    factor = float(charge_product)
    beta = 1.0 / math.sqrt(
        2.0 * (float(width_i_bohr) ** 2 + float(width_j_bohr) ** 2)
    )
    argument = beta * r
    gaussian = math.exp(-(argument**2))
    error_function = math.erf(argument)
    root_pi = math.sqrt(math.pi)
    first = (
        2.0 * beta * gaussian / (root_pi * r)
        - error_function / r**2
    )
    second = (
        -4.0 * beta**3 * gaussian / root_pi
        - 4.0 * beta * gaussian / (root_pi * r**2)
        + 2.0 * error_function / r**3
    )
    return _RadialDerivative(
        factor * error_function / r,
        factor * first,
        factor * second,
    )


def _uff_radial_derivative(
    distance_bohr: float,
    xij_bohr: float,
    dij_hartree: float,
    pair_scale: float,
) -> _RadialDerivative:
    """Return V, dV/dr and d²V/dr² for the UFF 6-12 radial potential."""
    r = float(distance_bohr)
    x6 = float(xij_bohr) ** 6
    x12 = x6 * x6
    factor = float(pair_scale) * float(dij_hartree)
    return _RadialDerivative(
        factor * (x12 / r**12 - 2.0 * x6 / r**6),
        factor * (-12.0 * x12 / r**13 + 12.0 * x6 / r**7),
        factor * (156.0 * x12 / r**14 - 84.0 * x6 / r**8),
    )


def _exppe_radial_derivative(
    distance_bohr: float,
    r_min_bohr: float,
    well_depth_hartree: float,
    pair_scale: float,
) -> _RadialDerivative:
    """Return analytic Exp-PE derivatives after curvature-preserving UFF conversion."""
    r = float(distance_bohr)
    r_min = float(r_min_bohr)
    x = r / r_min
    alpha = float(np.sqrt(160.0))
    exp_full = float(np.exp(alpha * (1.0 - x)))
    exp_half = float(np.exp(0.5 * alpha * (1.0 - x)))
    polynomial = x**4 - 2.0 * x**2 + 3.0
    polynomial_1 = 4.0 * x**3 - 4.0 * x
    polynomial_2 = 12.0 * x**2 - 4.0
    energy = exp_full - polynomial * exp_half
    first = -alpha * exp_full - exp_half * (polynomial_1 - 0.5 * alpha * polynomial)
    second = alpha**2 * exp_full + exp_half * (
        -polynomial_2 + alpha * polynomial_1 - 0.25 * alpha**2 * polynomial
    )
    factor = float(pair_scale) * float(well_depth_hartree)
    return _RadialDerivative(
        factor * energy,
        factor * first / r_min,
        factor * second / r_min**2,
    )


def _damped_exppe_radial_derivative(
    distance_bohr: float,
    r_min_bohr: float,
    well_depth_hartree: float,
    pair_scale: float,
    *,
    damping_scale: float,
    damping_exponent: int,
) -> _RadialDerivative:
    """Apply a rational short-range switch to the complete Exp-PE term.

    The radial non-bonded field is attenuated throughout both the covalent and
    geminal-distance regions so that it supplies neither a hidden stretching
    nor a hidden bending curvature. Those contributions belong to the local
    topology-dependent corrections.
    """

    scale = float(damping_scale)
    exponent = int(damping_exponent)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("vdW damping scale must be finite and positive")
    if exponent < 4 or exponent % 2:
        raise ValueError("vdW damping exponent must be an even integer of at least four")
    base = _exppe_radial_derivative(
        distance_bohr,
        r_min_bohr,
        well_depth_hartree,
        pair_scale,
    )
    r = float(distance_bohr)
    damping_radius = scale * float(r_min_bohr)
    ratio = (damping_radius / r) ** exponent
    damping = 1.0 / (1.0 + ratio)
    first = exponent * damping * (1.0 - damping) / r
    second = (
        exponent
        * damping
        * (1.0 - damping)
        / r**2
        * (exponent * (1.0 - 2.0 * damping) - 1.0)
    )
    return _RadialDerivative(
        damping * base.energy,
        first * base.energy + damping * base.first,
        second * base.energy + 2.0 * first * base.first + damping * base.second,
    )


def uff_pair_parameters(zi: int, zj: int) -> tuple[float, float]:
    """Return the mixed UFF minimum distance (bohr) and well depth (hartree)."""
    ri = uff_vdw_radius(zi)
    rj = uff_vdw_radius(zj)
    di = uff_well_depth_kcal(zi)
    dj = uff_well_depth_kcal(zj)
    if ri is None or rj is None or di is None or dj is None:
        raise ValueError(f"UFF vdW parameters are not available for pair Z={zi}, Z={zj}")
    xij_angstrom = 2.0 * float(np.sqrt(float(ri) * float(rj)))
    dij_hartree = float(np.sqrt(float(di) * float(dj))) * KCAL_MOL_TO_HARTREE
    return xij_angstrom * ANGSTROM_TO_BOHR, dij_hartree


def _nonbonded_pair_scale(topological_distance: int, one_four_scale: float) -> float:
    if topological_distance <= 2:
        return 0.0
    if topological_distance == 3:
        return float(one_four_scale)
    return 1.0


def _accumulate_pair_derivatives(
    gradient: np.ndarray,
    hessian: np.ndarray,
    i: int,
    j: int,
    rvec: np.ndarray,
    radial: _RadialDerivative,
) -> None:
    """Apply the analytic chain rule from radial distance to Cartesians.

    For e = (x_i-x_j)/r, grad_i V = V' e and
    H_ii = V'' ee^T + (V'/r)(I-ee^T); the j blocks follow by sign.
    """
    r = float(np.linalg.norm(rvec))
    unit = rvec / r
    block = (radial.second - radial.first / r) * np.outer(unit, unit)
    block += (radial.first / r) * np.eye(3)
    si = slice(3 * i, 3 * i + 3)
    sj = slice(3 * j, 3 * j + 3)
    gradient[si] += radial.first * unit
    gradient[sj] -= radial.first * unit
    hessian[si, si] += block
    hessian[sj, sj] += block
    hessian[si, sj] -= block
    hessian[sj, si] -= block


def _pair_distance(rvec: np.ndarray, i: int, j: int) -> float:
    distance = float(np.linalg.norm(rvec))
    if distance <= 1.0e-12:
        raise ValueError(f"non-bonded atoms {i + 1} and {j + 1} are coincident")
    return distance


def _zero_based_graph(natoms: int, bonds: tuple[tuple[int, int], ...]) -> dict[int, set[int]]:
    graph = {idx: set() for idx in range(natoms)}
    for left, right in bonds:
        i = int(left) - 1
        j = int(right) - 1
        if 0 <= i < natoms and 0 <= j < natoms and i != j:
            graph[i].add(j)
            graph[j].add(i)
    return graph


def _topological_distances(graph: dict[int, set[int]], natoms: int) -> dict[tuple[int, int], int]:
    del natoms
    return bounded_topological_distances(graph, maximum_distance=3)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return 0.5 * (values + values.T)
