"""Composable, charge-conserving CM5 response for directional interactions.

The CM5 vector stored in LCB26 is an intrinsic electronic baseline.  Runtime
interactions may polarize that vector and transfer charge, but they must never
overwrite it.  This module supplies the common composition and audit layer;
individual interaction models remain responsible for recognizing contacts and
computing their own response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .directional_contacts import DirectionalContact, perceive_directional_contacts


MATRIX_CM5_DIRECTIONAL_RESPONSE_SCHEMA = "matrix.chem.cm5_directional_response.v1"


@dataclass(frozen=True)
class CM5ChargeResponseContribution:
    """Resolved response of one interaction channel at one geometry."""

    kind: str
    polarization_delta_e: np.ndarray
    charge_transfer_delta_e: np.ndarray
    response: object


@dataclass(frozen=True)
class CM5ChargeResponseResult:
    """Intrinsic CM5 charges plus the audited sum of all response channels."""

    charges_e: np.ndarray
    intrinsic_charges_e: np.ndarray
    polarization_delta_e: np.ndarray
    charge_transfer_delta_e: np.ndarray
    contributions: tuple[CM5ChargeResponseContribution, ...]
    schema: str = MATRIX_CM5_DIRECTIONAL_RESPONSE_SCHEMA


@dataclass(frozen=True)
class CM5ChargeResponseChannel:
    """Named adapter around one geometry-dependent CM5 response model."""

    kind: str
    model: object

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower().replace("_", "-")
        if not kind:
            raise ValueError("a CM5 response channel needs a non-empty kind")
        if not callable(getattr(self.model, "evaluate", None)):
            raise TypeError("a CM5 response channel model must define evaluate()")
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class CM5ChargeResponseModel:
    """Compose independent directional responses over an immutable baseline."""

    intrinsic_charges_e: np.ndarray
    channels: tuple[CM5ChargeResponseChannel, ...]
    charge_tolerance_e: float = 1.0e-12

    def __post_init__(self) -> None:
        intrinsic = np.asarray(self.intrinsic_charges_e, dtype=float).reshape(-1).copy()
        channels = tuple(self.channels)
        tolerance = float(self.charge_tolerance_e)
        if intrinsic.size == 0 or np.any(~np.isfinite(intrinsic)):
            raise ValueError("intrinsic CM5 charges must be a finite non-empty vector")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("CM5 response charge tolerance must be positive and finite")
        kinds = tuple(channel.kind for channel in channels)
        if len(set(kinds)) != len(kinds):
            raise ValueError("CM5 response channel kinds must be unique")
        intrinsic.setflags(write=False)
        object.__setattr__(self, "intrinsic_charges_e", intrinsic)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "charge_tolerance_e", tolerance)

    def evaluate(self, coordinates_angstrom: np.ndarray) -> CM5ChargeResponseResult:
        """Evaluate and sum channels without allowing baseline replacement."""

        xyz = np.asarray(coordinates_angstrom, dtype=float)
        if xyz.shape != (self.intrinsic_charges_e.size, 3) or np.any(
            ~np.isfinite(xyz)
        ):
            raise ValueError("runtime coordinates must be a finite (natoms, 3) array")
        polarization = np.zeros_like(self.intrinsic_charges_e)
        transfer = np.zeros_like(self.intrinsic_charges_e)
        contributions = []
        for channel in self.channels:
            channel_intrinsic = getattr(channel.model, "intrinsic_charges_e", None)
            if channel_intrinsic is not None and not np.allclose(
                np.asarray(channel_intrinsic, dtype=float).reshape(-1),
                self.intrinsic_charges_e,
                atol=self.charge_tolerance_e,
                rtol=0.0,
            ):
                raise ValueError(
                    f"CM5 response channel {channel.kind!r} uses a different baseline"
                )
            response = channel.model.evaluate(xyz)
            channel_polarization = _response_vector(
                response,
                "polarization_delta_e",
                self.intrinsic_charges_e.shape,
            )
            channel_transfer = _response_vector(
                response,
                "charge_transfer_delta_e",
                self.intrinsic_charges_e.shape,
            )
            _require_charge_conservation(
                channel_polarization,
                self.charge_tolerance_e,
                channel.kind,
                "polarization",
            )
            _require_charge_conservation(
                channel_transfer,
                self.charge_tolerance_e,
                channel.kind,
                "charge transfer",
            )
            polarization += channel_polarization
            transfer += channel_transfer
            contributions.append(
                CM5ChargeResponseContribution(
                    kind=channel.kind,
                    polarization_delta_e=channel_polarization,
                    charge_transfer_delta_e=channel_transfer,
                    response=response,
                )
            )
        charges = self.intrinsic_charges_e + polarization + transfer
        if abs(float(np.sum(charges) - np.sum(self.intrinsic_charges_e))) > (
            2.0 * self.charge_tolerance_e
        ):
            raise ArithmeticError("composed CM5 response does not conserve total charge")
        return CM5ChargeResponseResult(
            charges_e=charges,
            intrinsic_charges_e=self.intrinsic_charges_e.copy(),
            polarization_delta_e=polarization,
            charge_transfer_delta_e=transfer,
            contributions=tuple(contributions),
        )


@dataclass(frozen=True)
class CM5DirectionalContactResponseResult:
    """Charge-conserving response and contacts resolved by one generic model."""

    charges_e: np.ndarray
    intrinsic_charges_e: np.ndarray
    polarization_delta_e: np.ndarray
    charge_transfer_delta_e: np.ndarray
    contacts: tuple[DirectionalContact, ...]


@dataclass(frozen=True)
class CM5DirectionalContactResponseModel:
    """Apply explicitly calibrated transfer amplitudes to perceived contacts."""

    atomic_numbers: tuple[int, ...]
    bonded_pairs: tuple[tuple[int, int], ...]
    intrinsic_charges_e: np.ndarray
    transfer_by_kind_e: Mapping[str, float]

    def __post_init__(self) -> None:
        numbers = tuple(int(value) for value in self.atomic_numbers)
        charges = np.asarray(self.intrinsic_charges_e, dtype=float).reshape(-1).copy()
        transfers = {
            str(kind).strip().lower().replace("_", "-"): float(value)
            for kind, value in self.transfer_by_kind_e.items()
        }
        bonds = tuple(
            sorted(
                {
                    tuple(sorted((int(left), int(right))))
                    for left, right in self.bonded_pairs
                }
            )
        )
        if charges.shape != (len(numbers),) or np.any(~np.isfinite(charges)):
            raise ValueError("directional CM5 baseline must contain one finite charge per atom")
        if not transfers or any(
            not np.isfinite(value) or value < 0.0 for value in transfers.values()
        ):
            raise ValueError("directional CM5 transfer amplitudes must be finite and non-negative")
        charges.setflags(write=False)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "bonded_pairs", bonds)
        object.__setattr__(self, "intrinsic_charges_e", charges)
        object.__setattr__(self, "transfer_by_kind_e", transfers)

    def evaluate(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> CM5DirectionalContactResponseResult:
        contacts = perceive_directional_contacts(
            self.atomic_numbers,
            coordinates_angstrom,
            self.bonded_pairs,
            kinds=tuple(self.transfer_by_kind_e),
        )
        transfer = np.zeros_like(self.intrinsic_charges_e)
        for contact in contacts:
            amount = self.transfer_by_kind_e[contact.kind] * contact.strength
            transfer[contact.center] += amount
            transfer[contact.acceptor] -= amount
        polarization = np.zeros_like(transfer)
        return CM5DirectionalContactResponseResult(
            charges_e=self.intrinsic_charges_e + transfer,
            intrinsic_charges_e=self.intrinsic_charges_e.copy(),
            polarization_delta_e=polarization,
            charge_transfer_delta_e=transfer,
            contacts=contacts,
        )


def prepare_cm5_directional_contact_response_model(
    atomic_numbers: Sequence[int],
    bonded_pairs: Iterable[tuple[int, int]],
    intrinsic_charges_e: Sequence[float],
    *,
    transfer_by_kind_e: Mapping[str, float],
) -> CM5DirectionalContactResponseModel:
    """Prepare generic NCI response; amplitudes must be supplied explicitly."""

    return CM5DirectionalContactResponseModel(
        atomic_numbers=tuple(int(value) for value in atomic_numbers),
        bonded_pairs=tuple(bonded_pairs),
        intrinsic_charges_e=np.asarray(intrinsic_charges_e, dtype=float),
        transfer_by_kind_e=transfer_by_kind_e,
    )


def prepare_cm5_charge_response_model(
    intrinsic_charges_e: Sequence[float],
    channels: Sequence[CM5ChargeResponseChannel | tuple[str, object]],
    *,
    charge_tolerance_e: float = 1.0e-12,
) -> CM5ChargeResponseModel:
    """Prepare the common response layer from named interaction models."""

    normalized = tuple(
        channel
        if isinstance(channel, CM5ChargeResponseChannel)
        else CM5ChargeResponseChannel(str(channel[0]), channel[1])
        for channel in channels
    )
    return CM5ChargeResponseModel(
        intrinsic_charges_e=np.asarray(intrinsic_charges_e, dtype=float),
        channels=normalized,
        charge_tolerance_e=charge_tolerance_e,
    )


def _response_vector(
    response: object,
    field: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = (
        response.get(field)
        if isinstance(response, Mapping)
        else getattr(response, field, None)
    )
    if value is None:
        raise TypeError(f"CM5 response result must provide {field}")
    vector = np.asarray(value, dtype=float).reshape(-1).copy()
    if vector.shape != shape or np.any(~np.isfinite(vector)):
        raise ValueError(f"CM5 response {field} has an invalid shape or value")
    return vector


def _require_charge_conservation(
    vector: np.ndarray,
    tolerance: float,
    kind: str,
    component: str,
) -> None:
    if abs(float(np.sum(vector))) > tolerance:
        raise ArithmeticError(
            f"CM5 {component} in channel {kind!r} does not conserve charge"
        )


__all__ = [
    "MATRIX_CM5_DIRECTIONAL_RESPONSE_SCHEMA",
    "CM5ChargeResponseChannel",
    "CM5ChargeResponseContribution",
    "CM5ChargeResponseModel",
    "CM5ChargeResponseResult",
    "CM5DirectionalContactResponseModel",
    "CM5DirectionalContactResponseResult",
    "prepare_cm5_charge_response_model",
    "prepare_cm5_directional_contact_response_model",
]
