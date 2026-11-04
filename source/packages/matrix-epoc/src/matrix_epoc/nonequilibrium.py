"""State-specific fast/slow polarization on the frozen spectral CPCM cavity.

The conductor kernel and dielectric scaling are exactly those of EPOC CPCM.
This is a free-energy functional, not a time-dependent dielectric model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np

from .contracts import EPOC_REACTION_FIELD_SCHEMA
from .cpcm import CPCMConfinement, CPCMReactionField, CPCMSecondOrderResult


def _vector(value, size, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} requires {size} finite components")
    return array


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"an explicit {name} is required")
    return value


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SolvationSource:
    """Total (nuclei minus electrons) moments in one identified EPOC basis.

    ``source_id`` identifies the density/geometry artifact; ``state_id`` names
    its electronic state. No transition density may substitute a state density.
    """

    operator_sha256: str
    state_id: str
    source_id: str
    geometry_id: str
    moments: tuple[float, ...]

    def __post_init__(self):
        for name in ("operator_sha256", "state_id", "source_id", "geometry_id"):
            _text(getattr(self, name), name)
        values = _vector(self.moments, len(self.moments), "modal moments")
        if not len(values):
            raise ValueError("modal moments cannot be empty")
        object.__setattr__(self, "moments", tuple(float(v) for v in values))


@dataclass(frozen=True)
class FrozenSlowPolarization:
    """Immutable slow field, including its preparation free energy."""

    reference: SolvationSource
    static_dielectric: float
    optical_dielectric: float
    coefficients: tuple[float, ...]
    preparation_energy_hartree: float

    def __post_init__(self):
        if not isinstance(self.reference, SolvationSource):
            raise TypeError("a SolvationSource reference is required")
        _dielectrics(self.static_dielectric, self.optical_dielectric)
        coefficients = _vector(self.coefficients, len(self.reference.moments), "slow coefficients")
        if not math.isfinite(self.preparation_energy_hartree):
            raise ValueError("finite slow preparation energy required")
        object.__setattr__(self, "coefficients", tuple(float(v) for v in coefficients))

    def to_record(self):
        payload = dict(
            schema=EPOC_REACTION_FIELD_SCHEMA, kind="frozen_slow_polarization", version=1,
            operator_sha256=self.reference.operator_sha256,
            reference_state_id=self.reference.state_id, reference_source_id=self.reference.source_id,
            reference_geometry_id=self.reference.geometry_id,
            reference_moments=list(self.reference.moments),
            static_dielectric="INFINITY" if math.isinf(self.static_dielectric) else self.static_dielectric,
            optical_dielectric=self.optical_dielectric, coefficients=list(self.coefficients),
            preparation_energy_hartree=self.preparation_energy_hartree,
            energy_unit="hartree", model="spectral_conductor_scaling",
        )
        return {**payload, "sha256": _digest(payload)}

    @property
    def sha256(self):
        return self.to_record()["sha256"]


@dataclass(frozen=True)
class NonequilibriumSolvationResult:
    """Electrostatic free energy only; electronic vacuum energy is separate."""

    energy_hartree: float
    fast_energy_hartree: float
    slow_interaction_hartree: float
    slow_preparation_hartree: float
    equilibrium_energy_hartree: float
    reorganization_energy_hartree: float
    reaction_coefficients: tuple[float, ...]
    state_id: str
    source_id: str
    frozen_reference_sha256: str


def _dielectrics(static, optical):
    if (isinstance(static, bool) or isinstance(optical, bool)
            or not math.isfinite(optical) or optical < 1.0
            or math.isnan(static) or static < optical):
        raise ValueError("require 1 <= finite optical dielectric <= static dielectric")


@dataclass(frozen=True)
class NonequilibriumCPCM:
    """Fast response relaxed to the current source, slow response frozen.

    With negative conductor weights W, F(m|a) = 1/2 f_inf m.W.m
    + m.s_a - 1/2 a.s_a, where s_a=(f_0-f_inf)W a.
    Thus F(a|a)=F_eq(a), and F(m|a)-F_eq(m) is nonnegative.
    """

    conductor: CPCMReactionField
    static_dielectric: float
    optical_dielectric: float

    def __post_init__(self):
        _dielectrics(self.static_dielectric, self.optical_dielectric)
        if not isinstance(self.conductor, CPCMReactionField):
            raise TypeError("a compiled CPCMReactionField is required")
        if not math.isinf(self.conductor.confinement.dielectric):
            raise ValueError("nonequilibrium CPCM requires a conductor kernel")
        if any(not math.isfinite(m.reaction_weight) or m.reaction_weight > 0 for m in self.conductor.modes):
            raise ValueError("conductor kernel must be finite and negative semidefinite")

    @classmethod
    def compile(cls, cavity: CPCMConfinement, *, static_dielectric, optical_dielectric):
        _dielectrics(static_dielectric, optical_dielectric)
        # Copy the cavity arrays: later mutation of a caller's cavity must not
        # change the frame of an already prepared slow field.
        arrays = {}
        for name in ("center_bohr", "semiaxes_bohr", "rotation"):
            arrays[name] = np.array(getattr(cavity, name), copy=True)
            arrays[name].setflags(write=False)
        frozen = replace(cavity, dielectric=math.inf, **arrays)
        for name in arrays:
            getattr(frozen, name).setflags(write=False)
        return cls(CPCMReactionField.compile(frozen), float(static_dielectric), float(optical_dielectric))

    @property
    def operator_sha256(self):
        cavity = self.conductor.confinement
        return _digest(dict(
            model="spectral_conductor_scaling", backend=self.conductor.modal_backend,
            center=cavity.center_bohr.tolist(), axes=cavity.semiaxes_bohr.tolist(),
            rotation=cavity.rotation.tolist(), modes=[dict(
                weight=float(m.reaction_weight), exponents=m.exponents.tolist(),
                coefficients=m.coefficients.tolist()) for m in self.conductor.modes],
        ))

    @property
    def static_factor(self):
        return 1.0 - 1.0 / self.static_dielectric

    @property
    def optical_factor(self):
        return 1.0 - 1.0 / self.optical_dielectric

    @property
    def weights(self):
        return np.array([m.reaction_weight for m in self.conductor.modes])

    @staticmethod
    def geometry_identifier(coordinates_bohr):
        """Exact Cartesian fingerprint; callers use the same bohr conversion."""
        if np.iscomplexobj(coordinates_bohr):
            raise ValueError("geometry must be real")
        xyz = np.asarray(coordinates_bohr, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.all(np.isfinite(xyz)):
            raise ValueError("finite nonempty (atom,3) geometry in bohr required")
        return _digest(xyz.tolist())

    def source(self, moments, *, state_id, source_id, geometry_id):
        values = _vector(moments, len(self.conductor.modes), "modal moments")
        return SolvationSource(self.operator_sha256, state_id, source_id, geometry_id, tuple(values))

    def site_source(self, coordinates_bohr, charges, *, state_id, source_id):
        """Explicit site-charge approximation, with charge in electron units."""
        if np.iscomplexobj(coordinates_bohr) or np.iscomplexobj(charges):
            raise ValueError("site coordinates and charges must be real")
        return self.source(self.conductor.modal_moments(coordinates_bohr, charges),
                           state_id=state_id, source_id=source_id,
                           geometry_id=self.geometry_identifier(coordinates_bohr))

    def _moments(self, source):
        if not isinstance(source, SolvationSource) or source.operator_sha256 != self.operator_sha256:
            raise ValueError("source and frozen cavity/modal basis differ")
        return _vector(source.moments, len(self.conductor.modes), "modal moments")

    def freeze(self, reference):
        moments = self._moments(reference)
        slow = (self.static_factor - self.optical_factor) * self.weights * moments
        return FrozenSlowPolarization(reference, self.static_dielectric, self.optical_dielectric,
                                      tuple(slow), -0.5 * float(moments @ slow))

    def validate_frozen(self, frozen):
        if not isinstance(frozen, FrozenSlowPolarization):
            raise TypeError("a FrozenSlowPolarization is required")
        expected = self.freeze(frozen.reference)
        if frozen.to_record() != expected.to_record():
            raise ValueError("frozen field does not match this dielectric model/reference")

    def restore_frozen(self, record):
        """Restore only an untampered record matching this compiled operator."""
        payload = dict(record)
        checksum = payload.pop("sha256", None)
        if checksum != _digest(payload):
            raise ValueError("frozen polarization checksum mismatch")
        reference = SolvationSource(payload["operator_sha256"], payload["reference_state_id"],
                                    payload["reference_source_id"], payload["reference_geometry_id"],
                                    tuple(payload["reference_moments"]))
        expected = self.freeze(reference)
        if expected.to_record() != record:
            raise ValueError("incompatible frozen polarization record")
        return expected

    def evaluate(self, source, frozen):
        self.validate_frozen(frozen)
        m = self._moments(source)
        a = np.asarray(frozen.reference.moments)
        slow = np.asarray(frozen.coefficients)
        fast_coefficients = self.optical_factor * self.weights * m
        fast = 0.5 * float(m @ fast_coefficients)
        interaction = float(m @ slow)
        equilibrium = 0.5 * self.static_factor * float(m @ (self.weights * m))
        delta = m - a
        reorganization = -0.5 * (self.static_factor - self.optical_factor) * float(delta @ (self.weights * delta))
        return NonequilibriumSolvationResult(
            fast + interaction + frozen.preparation_energy_hartree, fast, interaction,
            frozen.preparation_energy_hartree, equilibrium, reorganization,
            tuple(fast_coefficients + slow), source.state_id, source.source_id, frozen.sha256,
        )

    def modal_derivatives(self, source, frozen, moment_jacobian, moment_hessian):
        """Chain-rule E/G/H for total moments, including supplied density response.

        Jacobian is (mode, coordinate), Hessian (mode, coordinate, coordinate).
        Units are hartree and the caller's explicit differentiation coordinate.
        The reference moments/cavity stay fixed during differentiation.
        """
        result = self.evaluate(source, frozen)
        if np.iscomplexobj(moment_jacobian) or np.iscomplexobj(moment_hessian):
            raise ValueError("moment derivatives must be real")
        j = np.asarray(moment_jacobian, dtype=float)
        h = np.asarray(moment_hessian, dtype=float)
        if (j.ndim != 2 or j.shape[0] != len(self.weights) or j.shape[1] == 0
                or h.shape != (len(self.weights), j.shape[1], j.shape[1])
                or not np.all(np.isfinite(j)) or not np.all(np.isfinite(h))
                or not np.allclose(h, h.swapaxes(1, 2), atol=1e-12, rtol=1e-10)):
            raise ValueError("finite, consistent modal derivatives and symmetric Hessians required")
        coefficients = np.asarray(result.reaction_coefficients)
        gradient = j.T @ coefficients
        hessian = j.T @ ((self.optical_factor * self.weights)[:, None] * j)
        hessian += np.einsum("m,mij->ij", coefficients, h)
        return result, gradient, hessian

    def site_derivatives(self, coordinates_bohr, charges, frozen, *, state_id, source_id):
        """Analytic Cartesian E/G/H with frozen site charges and fixed cavity.

        For geometry-dependent charges/densities use ``modal_derivatives``
        with their full first/second response instead.
        """
        source = self.site_source(coordinates_bohr, charges, state_id=state_id, source_id=source_id)
        xyz, q = self.conductor._validated_system(coordinates_bohr, charges)
        _, first, second = self.conductor._mode_derivatives(xyz, second=True)
        j = np.einsum("i,ima->mia", q, first).reshape(len(self.weights), -1)
        h = np.zeros((len(self.weights), xyz.size, xyz.size))
        for i in range(len(xyz)):
            h[:, 3*i:3*i+3, 3*i:3*i+3] = q[i] * second[i]
        result, gradient, hessian = self.modal_derivatives(source, frozen, j, h)
        return CPCMSecondOrderResult(result.energy_hartree, gradient, hessian,
                                    "NONEQUILIBRIUM_" + self.conductor.modal_backend)

    def reaction_potential(self, targets_bohr, source, frozen):
        """Potential for a positive unit charge, including the frozen slow field."""
        if np.iscomplexobj(targets_bohr) or not np.all(np.isfinite(targets_bohr)):
            raise ValueError("reaction-potential targets must be finite and real")
        result = self.evaluate(source, frozen)
        return self.conductor.modal_values(targets_bohr) @ np.asarray(result.reaction_coefficients)


__all__ = ["SolvationSource", "FrozenSlowPolarization", "NonequilibriumSolvationResult",
           "NonequilibriumCPCM"]
