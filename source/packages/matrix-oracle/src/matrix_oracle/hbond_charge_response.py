"""Closed-form hydrogen-bond charge response for ZAFF force fields.

The model reuses ORACLE's continuous hydrogen-bond perception but does not
solve a fluctuating-charge system.  Each perceived D--H...A contact carries a
local, charge-conserving response map.  Sparse second-order forward
differentiation provides exact Cartesian charge Jacobians and Hessians.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, exp, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from matrix_chem import HydrogenBondPairList, perceive_hydrogen_bonds
from matrix_chem.topology.vdw_radii import uff_vdw_radius

ZAFF_HBOND_CHARGE_RESPONSE_SCHEMA = "matrix.zaff.hbond_charge_response.v1"
ANGSTROM_TO_BOHR = 1.8897261254578281
# PBE0/def2-TZVP L0 CM5 water reference.  The raw calculation gives
# H1/H2 = 0.3371643551/0.3370162080 e; ORACLE symmetry projection replaces
# them by their mean while retaining the raw vector in the validation archive.
WATER_CM5_REFERENCE = (
    -0.6741805631786746,
    0.3370902815893373,
    0.3370902815893373,
)
WATER_TIP3P_FB_REFERENCE = (-0.84844, 0.42422, 0.42422)
# Barone et al., ACS Omega 2022, 7, 13382-13394, eqs 18-20.
# The published alpha is absorbed below into the exact CM5 -> TIP3P-FB
# endpoint constraint; beta remains the signed per-contact fragment transfer.
WATER_HBOND_BOUNDARY_ALPHA = 0.20
WATER_HBOND_CHARGE_TRANSFER_E = 0.10


@dataclass(frozen=True)
class HydrogenBondStrengthParameters:
    """Parameter-free interpolation through ORACLE's existing bond order."""

    interpolation: str = "MAYER_OR_ORACLE_PAULING_RATIO"

    def __post_init__(self) -> None:
        if self.interpolation != "MAYER_OR_ORACLE_PAULING_RATIO":
            raise ValueError("unsupported hydrogen-bond interpolation contract")


@dataclass(frozen=True)
class HydrogenBondChargeContact:
    """One contact with separate polarization and charge-transfer channels."""

    donor: int
    hydrogen: int
    acceptor: int
    donor_polarization_response_e: tuple[tuple[int, float], ...]
    acceptor_polarization_response_e: tuple[tuple[int, float], ...]
    charge_transfer_e: float = 0.0
    label: str = ""
    reference_distance_angstrom: float = 1.85
    reference_mayer_bond_order: float | None = None
    vdw_radius_sum_angstrom: float | None = None

    def __post_init__(self) -> None:
        if min(self.donor, self.hydrogen, self.acceptor) < 0:
            raise ValueError("hydrogen-bond atom indices must be nonnegative")
        combined = (
            self.donor_polarization_response_e
            + self.acceptor_polarization_response_e
        )
        if not combined:
            raise ValueError("a hydrogen-bond contact needs a charge response")
        if any(index < 0 or not np.isfinite(value) for index, value in combined):
            raise ValueError("hydrogen-bond response entries must be finite")
        for name, response in (
            ("donor", self.donor_polarization_response_e),
            ("acceptor", self.acceptor_polarization_response_e),
        ):
            if not np.isclose(
                sum(value for _index, value in response),
                0.0,
                atol=2.0e-14,
                rtol=0.0,
            ):
                raise ValueError(
                    f"{name} polarization response must conserve fragment charge"
                )
        if not np.isfinite(self.charge_transfer_e):
            raise ValueError("hydrogen-bond charge transfer must be finite")
        if (
            not np.isfinite(self.reference_distance_angstrom)
            or self.reference_distance_angstrom <= 0.0
        ):
            raise ValueError("hydrogen-bond reference distance must be positive")
        if self.reference_mayer_bond_order is not None and (
            not np.isfinite(self.reference_mayer_bond_order)
            or self.reference_mayer_bond_order <= 0.0
        ):
            raise ValueError("reference Mayer bond order must be positive")
        if self.vdw_radius_sum_angstrom is not None and (
            not np.isfinite(self.vdw_radius_sum_angstrom)
            or self.vdw_radius_sum_angstrom <= 0.0
        ):
            raise ValueError("the H...A van der Waals radius sum must be positive")

    @property
    def polarization_response_e(self) -> tuple[tuple[int, float], ...]:
        return (
            self.donor_polarization_response_e
            + self.acceptor_polarization_response_e
        )

    @property
    def charge_transfer_response_e(self) -> tuple[tuple[int, float], ...]:
        return (
            (self.donor, float(self.charge_transfer_e)),
            (self.acceptor, -float(self.charge_transfer_e)),
        )

    @property
    def response_e(self) -> tuple[tuple[int, float], ...]:
        return self.polarization_response_e + self.charge_transfer_response_e


@dataclass(frozen=True)
class EllipsoidalBoundaryResponse:
    """Frozen spherical/ellipsoidal boundary used to restore missing partners."""

    center_bohr: np.ndarray
    semiaxes_bohr: np.ndarray
    rotation: np.ndarray
    onset_fraction: float = 0.82

    def __post_init__(self) -> None:
        center = np.asarray(self.center_bohr, dtype=float)
        axes = np.asarray(self.semiaxes_bohr, dtype=float)
        rotation = np.asarray(self.rotation, dtype=float)
        if center.shape != (3,) or axes.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("ellipsoidal boundary dimensions are inconsistent")
        if np.any(~np.isfinite(center)) or np.any(~np.isfinite(axes)) or np.min(axes) <= 0:
            raise ValueError("ellipsoidal boundary values must be finite and positive")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10):
            raise ValueError("ellipsoidal boundary rotation must be orthonormal")
        if not 0.0 < float(self.onset_fraction) < 1.0:
            raise ValueError("boundary onset fraction must lie between zero and one")
        object.__setattr__(self, "center_bohr", center)
        object.__setattr__(self, "semiaxes_bohr", axes)
        object.__setattr__(self, "rotation", rotation)


@dataclass(frozen=True)
class WaterHydrogenBondResponseParameters:
    """Published water response adapted to CM5 and the MATRIX H-bond coordinate.

    The isolated and four-coordinate limits are imposed exactly.  The
    published ``beta`` value supplies the signed intermolecular transfer; it
    cancels on the central water of a symmetric two-donor/two-acceptor
    pentamer.
    """

    isolated_cm5_e: tuple[float, float, float] = WATER_CM5_REFERENCE
    condensed_reference_e: tuple[float, float, float] = WATER_TIP3P_FB_REFERENCE
    condensed_model: str = "TIP3P-FB"
    donor_fraction: float = 0.5
    charge_transfer_e: float = WATER_HBOND_CHARGE_TRANSFER_E

    def __post_init__(self) -> None:
        isolated = tuple(float(value) for value in self.isolated_cm5_e)
        condensed = tuple(float(value) for value in self.condensed_reference_e)
        if (
            len(isolated) != 3
            or len(condensed) != 3
            or not all(np.isfinite(isolated + condensed))
        ):
            raise ValueError("water endpoint references must contain three finite charges")
        if not np.isclose(sum(isolated), 0.0, atol=2.0e-8, rtol=0.0):
            raise ValueError("water CM5 reference must be neutral")
        if not np.isclose(sum(condensed), 0.0, atol=2.0e-8, rtol=0.0):
            raise ValueError("condensed-water reference must be neutral")
        if not 0.0 <= float(self.donor_fraction) <= 1.0:
            raise ValueError("water donor fraction must lie between zero and one")
        if not np.isfinite(self.charge_transfer_e):
            raise ValueError("water hydrogen-bond charge transfer must be finite")
        if not str(self.condensed_model).strip():
            raise ValueError("condensed-water endpoint model must be named")

    @property
    def hydrogen_endpoint_shift_e(self) -> float:
        return float(self.condensed_reference_e[1] - self.isolated_cm5_e[1])


@dataclass(frozen=True)
class HydrogenBondResponseCalibration:
    """Fitted reference/target response for one donor--acceptor synthon pair."""

    contact: HydrogenBondChargeContact
    donor_synthon: str
    acceptor_synthon: str
    reference_strength: float
    reference_source: str
    target_source: str
    teacher_model: str
    boundary_contract: str
    rms_charge_change_e: float
    maximum_charge_change_e: float
    polarization_locality: str = "FULL_FRAGMENT"

    @property
    def charge_transfer_e(self) -> float:
        return float(self.contact.charge_transfer_e)

    @property
    def donor_polarization_displaced_charge_e(self) -> float:
        """Charge displaced within the donor while its net charge stays fixed."""

        return 0.5 * float(
            sum(
                abs(value)
                for _atom, value in self.contact.donor_polarization_response_e
            )
        )

    @property
    def acceptor_polarization_displaced_charge_e(self) -> float:
        """Charge displaced within the acceptor while its net charge stays fixed."""

        return 0.5 * float(
            sum(
                abs(value)
                for _atom, value in self.contact.acceptor_polarization_response_e
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema": ZAFF_HBOND_CHARGE_RESPONSE_SCHEMA,
            "donor_synthon": self.donor_synthon,
            "acceptor_synthon": self.acceptor_synthon,
            "reference_strength": self.reference_strength,
            "reference_source": self.reference_source,
            "target_source": self.target_source,
            "teacher_model": self.teacher_model,
            "boundary_contract": self.boundary_contract,
            "rms_charge_change_e": self.rms_charge_change_e,
            "maximum_charge_change_e": self.maximum_charge_change_e,
            "polarization_locality": self.polarization_locality,
            "donor_polarization_displaced_charge_e": (
                self.donor_polarization_displaced_charge_e
            ),
            "acceptor_polarization_displaced_charge_e": (
                self.acceptor_polarization_displaced_charge_e
            ),
            "charge_transfer_e": self.charge_transfer_e,
            "contact": {
                "donor": self.contact.donor,
                "hydrogen": self.contact.hydrogen,
                "acceptor": self.contact.acceptor,
                "donor_polarization_response_e": [
                    list(item)
                    for item in self.contact.donor_polarization_response_e
                ],
                "acceptor_polarization_response_e": [
                    list(item)
                    for item in self.contact.acceptor_polarization_response_e
                ],
                "charge_transfer_e": self.contact.charge_transfer_e,
                "label": self.contact.label,
                "reference_distance_angstrom": (
                    self.contact.reference_distance_angstrom
                ),
                "reference_mayer_bond_order": (
                    self.contact.reference_mayer_bond_order
                ),
                "vdw_radius_sum_angstrom": (
                    self.contact.vdw_radius_sum_angstrom
                ),
            },
        }


@dataclass(frozen=True)
class HydrogenBondChargeResponseResult:
    """Charges and exact sparse first/second Cartesian derivatives."""

    charges_e: np.ndarray
    polarization_delta_e: np.ndarray
    charge_transfer_delta_e: np.ndarray
    boundary_delta_e: np.ndarray
    charge_jacobian_e_per_bohr: csr_matrix
    charge_hessian_entries_e_per_bohr2: tuple[tuple[int, int, int, float], ...]
    contact_strengths: np.ndarray
    boundary_exposures: np.ndarray
    schema: str = ZAFF_HBOND_CHARGE_RESPONSE_SCHEMA
    backend: str = "LOCAL_CLOSED_FORM_SPARSE_SECOND_ORDER"

    def charge_hessian(self, atom: int) -> csr_matrix:
        """Return one atomic charge Hessian without storing dense 3N matrices."""

        size = self.charge_jacobian_e_per_bohr.shape[1]
        rows, cols, data = [], [], []
        for charge_atom, left, right, value in self.charge_hessian_entries_e_per_bohr2:
            if charge_atom != int(atom):
                continue
            rows.append(left)
            cols.append(right)
            data.append(value)
            if left != right:
                rows.append(right)
                cols.append(left)
                data.append(value)
        return coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

    def charge_directional_derivative(self, vector_bohr: np.ndarray) -> np.ndarray:
        direction = np.asarray(vector_bohr, dtype=float).reshape(-1)
        return np.asarray(self.charge_jacobian_e_per_bohr @ direction).reshape(-1)

    def charge_second_directional_derivative(
        self, vector_bohr: np.ndarray
    ) -> np.ndarray:
        direction = np.asarray(vector_bohr, dtype=float).reshape(-1)
        result = np.zeros(len(self.charges_e))
        for atom, left, right, value in self.charge_hessian_entries_e_per_bohr2:
            factor = 1.0 if left == right else 2.0
            result[atom] += factor * value * direction[left] * direction[right]
        return result


def fit_hydrogen_bond_response(
    reference_charges_e: Sequence[float],
    target_charges_e: Sequence[float],
    *,
    donor: int,
    hydrogen: int,
    acceptor: int,
    donor_fragment_atoms: Iterable[int],
    acceptor_fragment_atoms: Iterable[int],
    donor_polarization_atoms: Iterable[int] | None = None,
    acceptor_polarization_atoms: Iterable[int] | None = None,
    donor_synthon: str,
    acceptor_synthon: str,
    reference_strength: float = 1.0,
    reference_source: str = "unpolarized reference charges",
    target_source: str = "polarized one-bridge target charges",
    teacher_model: str = "VARIATIONAL_QEQ_SQE",
    boundary_contract: str = "NOT_APPLICABLE_ISOLATED_CLUSTER",
    locality_threshold_e: float = 1.0e-7,
    reference_distance_angstrom: float = 1.85,
    reference_mayer_bond_order: float | None = None,
    vdw_radius_sum_angstrom: float | None = None,
) -> HydrogenBondResponseCalibration:
    """Fit a reusable contact rule from paired reference/teacher charges.

    The atom order and total charge must be identical in the two calculations.
    Dividing by ``reference_strength`` makes the stored increments correspond
    to a unit-strength bridge; runtime values are then obtained by continuous
    interpolation.  Donor and acceptor fragment sums expose any fitted charge
    transfer without mixing it with the local polarization pattern.
    """

    unbound = np.asarray(reference_charges_e, dtype=float).reshape(-1)
    bonded = np.asarray(target_charges_e, dtype=float).reshape(-1)
    if (
        unbound.shape != bonded.shape
        or not len(unbound)
        or np.any(~np.isfinite(unbound))
        or np.any(~np.isfinite(bonded))
    ):
        raise ValueError("paired charge vectors must have the same finite nonzero length")
    if not np.isclose(np.sum(unbound), np.sum(bonded), atol=2.0e-7, rtol=0.0):
        raise ValueError("reference and teacher states must have the same total charge")
    boundary_label = str(boundary_contract).strip()
    if not boundary_label:
        raise ValueError("hydrogen-bond fitting requires an explicit boundary contract")
    if (
        "QEQ" in str(teacher_model).upper()
        and boundary_label == "UNSPECIFIED"
    ):
        raise ValueError("a variational teacher cannot use an unspecified boundary")
    strength = float(reference_strength)
    if not np.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("reference hydrogen-bond strength must lie in (0, 1]")
    donor_atoms = tuple(sorted({int(atom) for atom in donor_fragment_atoms}))
    acceptor_atoms = tuple(sorted({int(atom) for atom in acceptor_fragment_atoms}))
    if (
        not donor_atoms
        or not acceptor_atoms
        or set(donor_atoms) & set(acceptor_atoms)
        or min(donor_atoms + acceptor_atoms) < 0
        or max(donor_atoms + acceptor_atoms) >= len(unbound)
    ):
        raise ValueError("donor and acceptor fragment atom sets must be valid and disjoint")
    if donor not in donor_atoms or hydrogen not in donor_atoms or acceptor not in acceptor_atoms:
        raise ValueError("D--H and A atoms must belong to their declared fragments")
    difference = bonded - unbound
    covered = set(donor_atoms) | set(acceptor_atoms)
    omitted = [
        atom
        for atom, value in enumerate(difference)
        if atom not in covered and abs(value) > float(locality_threshold_e)
    ]
    if omitted:
        raise ValueError(
            "hydrogen-bond response omits changed atoms: "
            + ", ".join(str(atom) for atom in omitted)
        )
    scaled = difference / strength
    donor_total = {
        atom: float(scaled[atom])
        for atom in donor_atoms
        if abs(scaled[atom]) > float(locality_threshold_e)
    }
    acceptor_total = {
        atom: float(scaled[atom])
        for atom in acceptor_atoms
        if abs(scaled[atom]) > float(locality_threshold_e)
    }
    printed_residual = float(sum(donor_total.values()) + sum(acceptor_total.values()))
    acceptor_total[int(acceptor)] = (
        acceptor_total.get(int(acceptor), 0.0) - printed_residual
    )
    transfer = float(sum(donor_total.values()))
    donor_total[int(donor)] = donor_total.get(int(donor), 0.0) - transfer
    acceptor_total[int(acceptor)] = acceptor_total.get(int(acceptor), 0.0) + transfer
    donor_allowed = (
        donor_atoms
        if donor_polarization_atoms is None
        else tuple(sorted({int(atom) for atom in donor_polarization_atoms}))
    )
    acceptor_allowed = (
        acceptor_atoms
        if acceptor_polarization_atoms is None
        else tuple(sorted({int(atom) for atom in acceptor_polarization_atoms}))
    )
    if (
        not donor_allowed
        or not acceptor_allowed
        or not set(donor_allowed) <= set(donor_atoms)
        or not set(acceptor_allowed) <= set(acceptor_atoms)
        or donor not in donor_allowed
        or hydrogen not in donor_allowed
        or acceptor not in acceptor_allowed
    ):
        raise ValueError(
            "polarization atoms must be fragment subsets containing D--H and A"
        )
    donor_total = _project_local_neutral_response(donor_total, donor_allowed)
    acceptor_total = _project_local_neutral_response(
        acceptor_total, acceptor_allowed
    )
    donor_response = tuple(
        sorted((atom, value) for atom, value in donor_total.items() if value != 0.0)
    )
    acceptor_response = tuple(
        sorted((atom, value) for atom, value in acceptor_total.items() if value != 0.0)
    )
    contact = HydrogenBondChargeContact(
        donor=int(donor),
        hydrogen=int(hydrogen),
        acceptor=int(acceptor),
        donor_polarization_response_e=donor_response,
        acceptor_polarization_response_e=acceptor_response,
        charge_transfer_e=transfer,
        label=f"{str(donor_synthon)}...{str(acceptor_synthon)}",
        reference_distance_angstrom=float(reference_distance_angstrom),
        reference_mayer_bond_order=reference_mayer_bond_order,
        vdw_radius_sum_angstrom=vdw_radius_sum_angstrom,
    )
    return HydrogenBondResponseCalibration(
        contact=contact,
        donor_synthon=str(donor_synthon),
        acceptor_synthon=str(acceptor_synthon),
        reference_strength=strength,
        reference_source=str(reference_source),
        target_source=str(target_source),
        teacher_model=str(teacher_model),
        boundary_contract=boundary_label,
        rms_charge_change_e=float(np.sqrt(np.mean(difference**2))),
        maximum_charge_change_e=float(np.max(np.abs(difference))),
        polarization_locality=(
            "FULL_FRAGMENT"
            if donor_polarization_atoms is None
            and acceptor_polarization_atoms is None
            else "LOCAL_XH_CO_NCO"
        ),
    )


def _project_local_neutral_response(
    full_response: Mapping[int, float],
    allowed_atoms: tuple[int, ...],
) -> dict[int, float]:
    """Minimum-L2 projection onto a supported, fragment-neutral response."""

    values = np.asarray(
        [float(full_response.get(atom, 0.0)) for atom in allowed_atoms],
        dtype=float,
    )
    values -= float(np.sum(values)) / len(values)
    return {
        atom: float(value)
        for atom, value in zip(allowed_atoms, values, strict=True)
        if abs(value) > 1.0e-15
    }


def fit_cm5_hydrogen_bond_response(
    unbound_cm5_e: Sequence[float],
    bonded_cm5_e: Sequence[float],
    **kwargs: object,
) -> HydrogenBondResponseCalibration:
    """Fit the paired-CM5 special case of :func:`fit_hydrogen_bond_response`."""

    options = dict(kwargs)
    options.setdefault("reference_source", "CM5 without hydrogen bond")
    options.setdefault("target_source", "CM5 with one hydrogen bond")
    options.setdefault("teacher_model", "PAIRED_CM5")
    return fit_hydrogen_bond_response(
        unbound_cm5_e,
        bonded_cm5_e,
        **options,
    )


def hydrogen_bond_strength(
    coordinates_angstrom: np.ndarray,
    *,
    donor: int,
    hydrogen: int,
    acceptor: int,
    parameters: HydrogenBondStrengthParameters = HydrogenBondStrengthParameters(),
) -> float:
    """Return the scalar continuous strength used by the analytic response."""

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or np.any(~np.isfinite(xyz)):
        raise ValueError("hydrogen-bond coordinates must have shape (natoms, 3)")
    if min(donor, hydrogen, acceptor) < 0 or max(donor, hydrogen, acceptor) >= len(xyz):
        raise IndexError("hydrogen-bond atom is outside the coordinate array")
    dummy = HydrogenBondChargeContact(
        donor=int(donor),
        hydrogen=int(hydrogen),
        acceptor=int(acceptor),
        donor_polarization_response_e=((int(donor), 0.0),),
        acceptor_polarization_response_e=((int(acceptor), 0.0),),
    )
    return float(
        _hydrogen_bond_strength(
            xyz * ANGSTROM_TO_BOHR,
            dummy,
            parameters,
        ).value
    )


def qmmm_mm_charge_response_contacts(
    contacts: Sequence[HydrogenBondChargeContact],
    qm_atoms: Iterable[int],
) -> tuple[HydrogenBondChargeContact, ...]:
    """Project contact rules onto the MM side of electrostatic embedding.

    QM--QM contacts are handled by the electronic-structure program.  For a
    contact crossing the boundary, only the fragment-neutral MM polarization
    response survives and net charge transfer is disabled.  MM--MM contacts
    retain both polarization and charge transfer.
    """

    qm = {int(atom) for atom in qm_atoms}
    if qm and min(qm) < 0:
        raise ValueError("QM atom indices must be nonnegative")
    projected: list[HydrogenBondChargeContact] = []
    for contact in contacts:
        donor_is_qm = contact.donor in qm or contact.hydrogen in qm
        acceptor_is_qm = contact.acceptor in qm
        if donor_is_qm and acceptor_is_qm:
            continue
        if donor_is_qm:
            donor_response: tuple[tuple[int, float], ...] = ()
            acceptor_response = contact.acceptor_polarization_response_e
            transfer = 0.0
        elif acceptor_is_qm:
            donor_response = contact.donor_polarization_response_e
            acceptor_response = ()
            transfer = 0.0
        else:
            donor_response = contact.donor_polarization_response_e
            acceptor_response = contact.acceptor_polarization_response_e
            transfer = contact.charge_transfer_e
        if not donor_response and not acceptor_response and transfer == 0.0:
            continue
        projected.append(
            HydrogenBondChargeContact(
                donor=contact.donor,
                hydrogen=contact.hydrogen,
                acceptor=contact.acceptor,
                donor_polarization_response_e=donor_response,
                acceptor_polarization_response_e=acceptor_response,
                charge_transfer_e=transfer,
                label=contact.label,
                reference_distance_angstrom=contact.reference_distance_angstrom,
                reference_mayer_bond_order=contact.reference_mayer_bond_order,
                vdw_radius_sum_angstrom=contact.vdw_radius_sum_angstrom,
            )
        )
    return tuple(projected)


@dataclass(frozen=True)
class _Jet:
    value: float
    gradient: dict[int, float]
    hessian: dict[tuple[int, int], float]


def evaluate_hydrogen_bond_charge_response(
    coordinates_bohr: np.ndarray,
    reference_charges_e: Sequence[float],
    contacts: Sequence[HydrogenBondChargeContact],
    *,
    strength_parameters: HydrogenBondStrengthParameters = HydrogenBondStrengthParameters(),
) -> HydrogenBondChargeResponseResult:
    """Apply arbitrary local D--H...A response maps with analytic E/G/H data."""

    xyz = np.asarray(coordinates_bohr, dtype=float)
    reference = np.asarray(reference_charges_e, dtype=float).reshape(-1)
    if xyz.shape != (len(reference), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("charge-response coordinates and reference charges are inconsistent")
    charges = [_constant(value) for value in reference]
    polarization = [_constant(0.0) for _value in reference]
    charge_transfer = [_constant(0.0) for _value in reference]
    boundary_delta = [_constant(0.0) for _value in reference]
    strengths: list[float] = []
    for contact in contacts:
        if max(contact.donor, contact.hydrogen, contact.acceptor) >= len(reference):
            raise IndexError("hydrogen-bond contact atom is outside the system")
        strength = _hydrogen_bond_strength(xyz, contact, strength_parameters)
        strengths.append(strength.value)
        for atom, increment in contact.polarization_response_e:
            if atom >= len(reference):
                raise IndexError("hydrogen-bond response atom is outside the system")
            contribution = _scale(strength, float(increment))
            charges[atom] = _add(charges[atom], contribution)
            polarization[atom] = _add(polarization[atom], contribution)
        for atom, increment in contact.charge_transfer_response_e:
            contribution = _scale(strength, float(increment))
            charges[atom] = _add(charges[atom], contribution)
            charge_transfer[atom] = _add(charge_transfer[atom], contribution)
    return _pack_result(
        charges,
        strengths,
        (),
        polarization,
        charge_transfer,
        boundary_delta,
    )


def evaluate_water_hydrogen_bond_charge_response(
    coordinates_bohr: np.ndarray,
    waters: Sequence[tuple[int, int, int]],
    *,
    reference_charges_e: Sequence[float] | None = None,
    response_parameters: WaterHydrogenBondResponseParameters = (
        WaterHydrogenBondResponseParameters()
    ),
    strength_parameters: HydrogenBondStrengthParameters = HydrogenBondStrengthParameters(),
    boundary: EllipsoidalBoundaryResponse | None = None,
    recognition: HydrogenBondPairList | None = None,
) -> HydrogenBondChargeResponseResult:
    """Evaluate noniterative water polarization from ORACLE H-bond perception.

    ``waters`` stores ``(O, H1, H2)`` indices.  Donor and acceptor responses
    are separately fragment-neutral when ``charge_transfer_e`` is zero.  A
    fully four-coordinate water reaches the resident TIP3P-FB charges exactly.
    """

    xyz = np.asarray(coordinates_bohr, dtype=float)
    water_tuples = tuple(tuple(int(value) for value in water) for water in waters)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or np.any(~np.isfinite(xyz)):
        raise ValueError("water charge-response coordinates must have shape (natoms, 3)")
    flat = tuple(index for water in water_tuples for index in water)
    if len(flat) != len(set(flat)) or (flat and (min(flat) < 0 or max(flat) >= len(xyz))):
        raise ValueError("water atoms must be unique valid zero-based indices")
    if reference_charges_e is None:
        reference = np.zeros(len(xyz))
        for oxygen, h1, h2 in water_tuples:
            reference[[oxygen, h1, h2]] = response_parameters.isolated_cm5_e
    else:
        reference = np.asarray(reference_charges_e, dtype=float).reshape(-1)
        if len(reference) != len(xyz):
            raise ValueError("water reference charges have the wrong dimension")

    numbers = np.zeros(len(xyz), dtype=int)
    bonds: list[tuple[int, int]] = []
    hydrogen_to_water: dict[int, tuple[int, int, int]] = {}
    oxygen_to_water: dict[int, tuple[int, int, int]] = {}
    for water in water_tuples:
        oxygen, h1, h2 = water
        numbers[oxygen], numbers[h1], numbers[h2] = 8, 1, 1
        bonds.extend(((oxygen, h1), (oxygen, h2)))
        hydrogen_to_water[h1] = water
        hydrogen_to_water[h2] = water
        oxygen_to_water[oxygen] = water
    perceived = (
        perceive_hydrogen_bonds(
            numbers,
            xyz / ANGSTROM_TO_BOHR,
            bonds,
            selector_threshold=1.0e-8,
            minimum_angle_degrees=90.0,
        )
        if recognition is None
        else recognition.perceive(xyz / ANGSTROM_TO_BOHR)
    )
    contacts: list[HydrogenBondChargeContact] = []
    donor_occupancy = {hydrogen: _constant(0.0) for hydrogen in hydrogen_to_water}
    acceptor_occupancy = {oxygen: _constant(0.0) for oxygen in oxygen_to_water}
    delta_h = response_parameters.hydrogen_endpoint_shift_e
    donor_share = float(response_parameters.donor_fraction) * delta_h
    acceptor_share = (1.0 - float(response_parameters.donor_fraction)) * delta_h
    transfer = float(response_parameters.charge_transfer_e)
    charges = [_constant(value) for value in reference]
    polarization = [_constant(0.0) for _value in reference]
    charge_transfer_delta = [_constant(0.0) for _value in reference]
    boundary_delta = [_constant(0.0) for _value in reference]
    strengths: list[float] = []
    for perceived_contact in perceived:
        donor_water = hydrogen_to_water.get(perceived_contact.hydrogen)
        acceptor_water = oxygen_to_water.get(perceived_contact.acceptor)
        if donor_water is None or acceptor_water is None or donor_water == acceptor_water:
            continue
        donor_o, _dh1, _dh2 = donor_water
        acceptor_o, acceptor_h1, acceptor_h2 = acceptor_water
        contact = HydrogenBondChargeContact(
            donor=donor_o,
            hydrogen=perceived_contact.hydrogen,
            acceptor=acceptor_o,
            donor_polarization_response_e=(
                (donor_o, -donor_share),
                (perceived_contact.hydrogen, donor_share),
            ),
            acceptor_polarization_response_e=(
                (acceptor_o, -acceptor_share),
                (acceptor_h1, 0.5 * acceptor_share),
                (acceptor_h2, 0.5 * acceptor_share),
            ),
            charge_transfer_e=transfer,
            label="WATER_OH...O",
            reference_distance_angstrom=1.85,
            vdw_radius_sum_angstrom=float(
                (uff_vdw_radius(1) or 0.0) + (uff_vdw_radius(8) or 0.0)
            ),
        )
        contacts.append(contact)
        strength = _hydrogen_bond_strength(xyz, contact, strength_parameters)
        strengths.append(strength.value)
        donor_occupancy[contact.hydrogen] = _add(
            donor_occupancy[contact.hydrogen], strength
        )
        acceptor_occupancy[contact.acceptor] = _add(
            acceptor_occupancy[contact.acceptor], strength
        )
        for atom, increment in contact.polarization_response_e:
            contribution = _scale(strength, increment)
            charges[atom] = _add(charges[atom], contribution)
            polarization[atom] = _add(polarization[atom], contribution)
        for atom, increment in contact.charge_transfer_response_e:
            contribution = _scale(strength, increment)
            charges[atom] = _add(charges[atom], contribution)
            charge_transfer_delta[atom] = _add(
                charge_transfer_delta[atom], contribution
            )

    exposures: list[float] = []
    if boundary is not None:
        for oxygen, h1, h2 in water_tuples:
            exposure = _boundary_exposure(xyz, oxygen, boundary)
            exposures.append(exposure.value)
            for hydrogen in (h1, h2):
                occupancy = donor_occupancy[hydrogen]
                if occupancy.value < 1.0:
                    missing = _multiply(exposure, _subtract(_constant(1.0), occupancy))
                    charges[oxygen] = _add(
                        charges[oxygen], _scale(missing, -donor_share)
                    )
                    polarization[oxygen] = _add(
                        polarization[oxygen], _scale(missing, -donor_share)
                    )
                    boundary_delta[oxygen] = _add(
                        boundary_delta[oxygen], _scale(missing, -donor_share)
                    )
                    charges[hydrogen] = _add(
                        charges[hydrogen], _scale(missing, donor_share)
                    )
                    polarization[hydrogen] = _add(
                        polarization[hydrogen], _scale(missing, donor_share)
                    )
                    boundary_delta[hydrogen] = _add(
                        boundary_delta[hydrogen], _scale(missing, donor_share)
                    )
            occupancy = acceptor_occupancy[oxygen]
            if occupancy.value < 2.0:
                missing = _multiply(exposure, _subtract(_constant(2.0), occupancy))
                charges[oxygen] = _add(
                    charges[oxygen], _scale(missing, -acceptor_share)
                )
                polarization[oxygen] = _add(
                    polarization[oxygen], _scale(missing, -acceptor_share)
                )
                boundary_delta[oxygen] = _add(
                    boundary_delta[oxygen], _scale(missing, -acceptor_share)
                )
                charges[h1] = _add(
                    charges[h1], _scale(missing, 0.5 * acceptor_share)
                )
                polarization[h1] = _add(
                    polarization[h1], _scale(missing, 0.5 * acceptor_share)
                )
                boundary_delta[h1] = _add(
                    boundary_delta[h1], _scale(missing, 0.5 * acceptor_share)
                )
                charges[h2] = _add(
                    charges[h2], _scale(missing, 0.5 * acceptor_share)
                )
                polarization[h2] = _add(
                    polarization[h2], _scale(missing, 0.5 * acceptor_share)
                )
                boundary_delta[h2] = _add(
                    boundary_delta[h2], _scale(missing, 0.5 * acceptor_share)
                )
    return _pack_result(
        charges,
        strengths,
        exposures,
        polarization,
        charge_transfer_delta,
        boundary_delta,
    )


def _hydrogen_bond_strength(
    xyz: np.ndarray,
    contact: HydrogenBondChargeContact,
    parameters: HydrogenBondStrengthParameters,
) -> _Jet:
    coordinates = [
        [_variable(3 * atom + axis, xyz[atom, axis]) for axis in range(3)]
        for atom in (contact.hydrogen, contact.acceptor)
    ]
    hydrogen, acceptor = coordinates
    ah = [_subtract(acceptor[k], hydrogen[k]) for k in range(3)]
    distance = _sqrt(_dot(ah, ah))
    if distance.value <= 1.0e-12:
        raise ValueError("hydrogen-bond strength is undefined for coincident atoms")
    reference = contact.reference_distance_angstrom * ANGSTROM_TO_BOHR
    decay = (
        contact.vdw_radius_sum_angstrom
        if contact.vdw_radius_sum_angstrom is not None
        else contact.reference_distance_angstrom
    ) * ANGSTROM_TO_BOHR
    return _exp(_scale(_subtract(_constant(reference), distance), 1.0 / decay))


def _boundary_exposure(
    xyz: np.ndarray, oxygen: int, boundary: EllipsoidalBoundaryResponse
) -> _Jet:
    delta = [
        _subtract(
            _variable(3 * oxygen + axis, xyz[oxygen, axis]),
            _constant(boundary.center_bohr[axis]),
        )
        for axis in range(3)
    ]
    local = [
        _scale(
            _add(
                _add(
                    _scale(delta[0], boundary.rotation[0, axis]),
                    _scale(delta[1], boundary.rotation[1, axis]),
                ),
                _scale(delta[2], boundary.rotation[2, axis]),
            ),
            1.0 / boundary.semiaxes_bohr[axis],
        )
        for axis in range(3)
    ]
    rho = _sqrt(_dot(local, local))
    onset = float(boundary.onset_fraction)
    if rho.value <= onset:
        return _constant(0.0)
    if rho.value >= 1.0:
        return _constant(1.0)
    x = _scale(_subtract(rho, _constant(onset)), 1.0 / (1.0 - onset))
    x2, x3 = _multiply(x, x), _multiply(_multiply(x, x), x)
    x4, x5 = _multiply(x3, x), _multiply(x3, x2)
    return _add(_scale(x3, 10.0), _add(_scale(x4, -15.0), _scale(x5, 6.0)))


def _pack_result(
    charges: Sequence[_Jet],
    strengths: Sequence[float],
    exposures: Sequence[float],
    polarization: Sequence[_Jet],
    charge_transfer: Sequence[_Jet],
    boundary_delta: Sequence[_Jet],
) -> HydrogenBondChargeResponseResult:
    natoms = len(charges)
    jac_rows, jac_cols, jac_data = [], [], []
    hessian_entries: list[tuple[int, int, int, float]] = []
    for atom, charge in enumerate(charges):
        for coordinate, value in charge.gradient.items():
            if value != 0.0:
                jac_rows.append(atom)
                jac_cols.append(coordinate)
                jac_data.append(value)
        for (left, right), value in charge.hessian.items():
            if value != 0.0:
                hessian_entries.append((atom, left, right, value))
    jacobian = coo_matrix(
        (jac_data, (jac_rows, jac_cols)), shape=(natoms, 3 * natoms)
    ).tocsr()
    values = np.asarray([charge.value for charge in charges])
    if np.max(np.abs(np.asarray(jacobian.sum(axis=0)).reshape(-1)), initial=0.0) > 2.0e-11:
        raise ArithmeticError("hydrogen-bond charge Jacobian violates charge conservation")
    hessian_sums: dict[tuple[int, int], float] = {}
    for _atom, left, right, value in hessian_entries:
        hessian_sums[(left, right)] = hessian_sums.get((left, right), 0.0) + value
    if max((abs(value) for value in hessian_sums.values()), default=0.0) > 2.0e-10:
        raise ArithmeticError("hydrogen-bond charge Hessian violates charge conservation")
    return HydrogenBondChargeResponseResult(
        charges_e=values,
        polarization_delta_e=np.asarray([item.value for item in polarization]),
        charge_transfer_delta_e=np.asarray(
            [item.value for item in charge_transfer]
        ),
        boundary_delta_e=np.asarray([item.value for item in boundary_delta]),
        charge_jacobian_e_per_bohr=jacobian,
        charge_hessian_entries_e_per_bohr2=tuple(hessian_entries),
        contact_strengths=np.asarray(strengths),
        boundary_exposures=np.asarray(exposures),
    )


def _constant(value: float) -> _Jet:
    return _Jet(float(value), {}, {})


def _variable(index: int, value: float) -> _Jet:
    return _Jet(float(value), {int(index): 1.0}, {})




def _add(left: _Jet, right: _Jet) -> _Jet:
    gradient = dict(left.gradient)
    for index, value in right.gradient.items():
        gradient[index] = gradient.get(index, 0.0) + value
    hessian = dict(left.hessian)
    for key, value in right.hessian.items():
        hessian[key] = hessian.get(key, 0.0) + value
    return _Jet(left.value + right.value, gradient, hessian)


def _scale(item: _Jet, factor: float) -> _Jet:
    return _Jet(
        item.value * factor,
        {index: factor * value for index, value in item.gradient.items()},
        {key: factor * value for key, value in item.hessian.items()},
    )


def _subtract(left: _Jet, right: _Jet) -> _Jet:
    return _add(left, _scale(right, -1.0))


def _multiply(left: _Jet, right: _Jet) -> _Jet:
    gradient = {
        index: right.value * left.gradient.get(index, 0.0)
        + left.value * right.gradient.get(index, 0.0)
        for index in set(left.gradient) | set(right.gradient)
    }
    hessian: dict[tuple[int, int], float] = {}
    for key in set(left.hessian) | set(right.hessian):
        hessian[key] = (
            right.value * left.hessian.get(key, 0.0)
            + left.value * right.hessian.get(key, 0.0)
        )
    for i, gi in left.gradient.items():
        for j, gj in right.gradient.items():
            key = (i, j) if i <= j else (j, i)
            hessian[key] = hessian.get(key, 0.0) + (
                (2.0 if i == j else 1.0) * gi * gj
            )
    return _Jet(left.value * right.value, gradient, hessian)


def _unary(item: _Jet, value: float, first: float, second: float) -> _Jet:
    gradient = {index: first * derivative for index, derivative in item.gradient.items()}
    hessian = {key: first * derivative for key, derivative in item.hessian.items()}
    indices = tuple(item.gradient)
    for offset, i in enumerate(indices):
        for j in indices[offset:]:
            key = (i, j) if i <= j else (j, i)
            hessian[key] = (
                hessian.get(key, 0.0)
                + second * item.gradient[i] * item.gradient[j]
            )
    return _Jet(float(value), gradient, hessian)


def _exp(item: _Jet) -> _Jet:
    value = exp(item.value)
    return _unary(item, value, value, value)


def _sqrt(item: _Jet) -> _Jet:
    if item.value <= 0.0:
        raise ValueError("square-root jet requires a positive value")
    value = sqrt(item.value)
    return _unary(item, value, 0.5 / value, -0.25 / (item.value * value))






def _acos(item: _Jet) -> _Jet:
    denominator = sqrt(max(1.0e-30, 1.0 - item.value**2))
    return _unary(
        item,
        acos(item.value),
        -1.0 / denominator,
        -item.value / denominator**3,
    )


def _dot(left: Sequence[_Jet], right: Sequence[_Jet]) -> _Jet:
    result = _constant(0.0)
    for left_item, right_item in zip(left, right, strict=True):
        result = _add(result, _multiply(left_item, right_item))
    return result


__all__ = [
    "ZAFF_HBOND_CHARGE_RESPONSE_SCHEMA",
    "WATER_CM5_REFERENCE",
    "WATER_TIP3P_FB_REFERENCE",
    "HydrogenBondResponseCalibration",
    "EllipsoidalBoundaryResponse",
    "HydrogenBondChargeContact",
    "HydrogenBondChargeResponseResult",
    "HydrogenBondStrengthParameters",
    "WaterHydrogenBondResponseParameters",
    "evaluate_hydrogen_bond_charge_response",
    "evaluate_water_hydrogen_bond_charge_response",
    "fit_cm5_hydrogen_bond_response",
    "fit_hydrogen_bond_response",
    "hydrogen_bond_strength",
    "qmmm_mm_charge_response_contacts",
]
