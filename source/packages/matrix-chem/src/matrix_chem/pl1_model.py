"""Calibrated PL1 Gaussian residual model.

The CV layer is deliberately not part of this model.  A PL1 rule evaluates
only the residual ``L2 - (L1 + CV)`` for covalent bonds and hydrogen bonds.
Rules are keyed by the smallest useful chemical classes: element pair and
continuous Mayer/synthon bond order for covalent terms, and donor/acceptor
element pair for hydrogen-bond terms.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import exp
from pathlib import Path
from typing import Mapping, Sequence

from .topology.pykko_radii import bond_order_reference_radii
from .topology.vdw_radii import uff_vdw_radius
from .structural_corrections import hbond_angular_factor

PAULING_ELECTRONEGATIVITY = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
PERIOD_BY_Z = {1: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 14: 3, 15: 3, 16: 3, 17: 3, 35: 4, 53: 5}
GROUP_BY_Z = {1: 1, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 14: 14, 15: 15, 16: 16, 17: 17, 35: 17, 53: 17}


def pauling_electronegativity(atomic_number: int) -> float | None:
    return PAULING_ELECTRONEGATIVITY.get(int(atomic_number))


def period_difference(z_left: int, z_right: int) -> float:
    return float(abs(PERIOD_BY_Z.get(int(z_left), 0) - PERIOD_BY_Z.get(int(z_right), 0)))


def group_period_trend(z: int, coefficients: Sequence[float]) -> float:
    """Return an extrapolated atomic parameter from group/period."""
    if len(coefficients) < 3:
        return 0.0
    return float(coefficients[0] + coefficients[1] * GROUP_BY_Z.get(int(z), 0) + coefficients[2] * PERIOD_BY_Z.get(int(z), 0))


def _pair_key(z_left: int, z_right: int) -> str:
    return "-".join(str(z) for z in sorted((int(z_left), int(z_right))))


def covalent_class_key(z_left: int, z_right: int) -> str:
    """Return the compact BL1-compatible chemical class for a pair."""
    pair = tuple(sorted((int(z_left), int(z_right))))
    if 1 in pair:
        return "C-H" if pair == (1, 6) else "X-H"
    return {
        (6, 6): "C-C",
        (6, 7): "C-N",
        (6, 8): "C-O",
        (7, 7): "N/O-N/O",
        (7, 8): "N/O-N/O",
        (8, 8): "N/O-N/O",
    }.get(pair, "heavy-other")


def covalent_gaussian_center(z_left: int, z_right: int, bond_order: float) -> float:
    """Interpolate Pyykko single/double/triple radii at continuous order."""
    left = bond_order_reference_radii(int(z_left), fallback_radius=0.0)
    right = bond_order_reference_radii(int(z_right), fallback_radius=0.0)
    position = max(0.0, min(2.0, float(bond_order) - 1.0))
    lower = min(1, int(position))
    fraction = position - lower
    return float((1.0 - fraction) * (left[lower] + right[lower]) + fraction * (left[lower + 1] + right[lower + 1]))


def covalent_gaussian_width(z_left: int, z_right: int, bond_order: float, fallback: float = 0.057) -> float:
    """Continuous analogue of the legacy BL1 local Gaussian width."""
    left = bond_order_reference_radii(int(z_left), fallback_radius=0.0)
    right = bond_order_reference_radii(int(z_right), fallback_radius=0.0)
    radii = [left[i] + right[i] for i in range(3)]
    spacing = min(abs(radii[1] - radii[0]), abs(radii[2] - radii[1])) / 1.5
    return float(spacing if spacing > 1.0e-8 else fallback)


def vdw_gaussian_center(z_left: int, z_right: int) -> float | None:
    left = uff_vdw_radius(int(z_left))
    right = uff_vdw_radius(int(z_right))
    return None if left is None or right is None else float(left + right)


def hbond_gaussian_center(acceptor_z: int, vdw_scale: float = 0.81) -> float | None:
    """H...A center from vdW radii with the historical PL1 UFF scale."""
    center = vdw_gaussian_center(1, acceptor_z)
    return None if center is None else float(vdw_scale * center)


@dataclass(frozen=True)
class PL1GaussianModel:
    """Serializable residual model for the L1 -> PL1 promotion."""

    covalent_sigma_angstrom: float
    hbond_sigma_angstrom: float
    covalent_amplitudes: Mapping[str, float]
    hbond_amplitudes: Mapping[str, float]
    covalent_feature_coefficients: Sequence[float] | None = None
    zeff_feature_coefficients: Sequence[float] | None = None
    schema: str = "matrix.architect.pl1_gaussian_model.v2"
    atomic_amplitudes: Mapping[str, float] | None = None
    delta_zeff_coefficients: Sequence[float] | None = None
    electronegativity_coefficients: Sequence[float] | None = None
    atomic_trend_coefficients: Sequence[float] | None = None
    linear_order_coefficients: Mapping[str, float] | None = None
    xh_order_intercepts: Mapping[str, float] | None = None
    xh_cv_slopes: Mapping[str, float] | None = None
    xh_cv_intercepts: Mapping[str, float] | None = None
    xh_cv_scale: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.covalent_sigma_angstrom <= 0.0 or self.hbond_sigma_angstrom <= 0.0:
            raise ValueError("PL1 Gaussian widths must be positive")

    def covalent_delta(self, z_left: int, z_right: int, bond_order: float, distance: float, charge_left: float = 0.0, charge_right: float = 0.0, zeff_left: float | None = None, zeff_right: float | None = None) -> float:
        key = _pair_key(z_left, z_right)
        pair = tuple(sorted((int(z_left), int(z_right))))
        class_key = covalent_class_key(z_left, z_right)
        # Reuse the repository's published JCP Conj implementation for the
        # continuous multiple-bond domain.  The fitted atomic model is only
        # used outside this explicitly defined branch.
        if tuple(sorted((int(z_left), int(z_right)))) in {(6, 6), (6, 16)} and float(bond_order) >= 1.5:
            from .accuracy_ladder import core_valence_bond_shift, _conjugation_delta_angstrom
            from .topology.pykko_radii import bond_order_reference_radii
            rl = bond_order_reference_radii(int(z_left), fallback_radius=0.0)
            rr = bond_order_reference_radii(int(z_right), fallback_radius=0.0)
            double_ref = rl[1] + rr[1]
            cv = core_valence_bond_shift(int(z_left), int(z_right), double_ref, weight_threshold=0.0)
            return _conjugation_delta_angstrom(int(z_left), int(z_right), float(distance), cv)
        if self.atomic_amplitudes is not None and self.delta_zeff_coefficients is not None and self.electronegativity_coefficients is not None and zeff_left is not None and zeff_right is not None:
            dz_left, dz_right = float(zeff_left) - float(z_left), float(zeff_right) - float(z_right)
            aleft = self.atomic_amplitudes.get(str(int(z_left)), group_period_trend(z_left, self.atomic_trend_coefficients or ()))
            aright = self.atomic_amplitudes.get(str(int(z_right)), group_period_trend(z_right, self.atomic_trend_coefficients or ()))
            amplitude = 0.5 * (float(aleft) + float(aright))
            amplitude += float(sum(a * b for a, b in zip(self.delta_zeff_coefficients, (abs(dz_left - dz_right), 0.5 * (dz_left + dz_right)))))
            chi_left = pauling_electronegativity(z_left) or 0.0; chi_right = pauling_electronegativity(z_right) or 0.0
            amplitude += float(sum(a * b for a, b in zip(self.electronegativity_coefficients, (abs(chi_left - chi_right), 0.5 * (chi_left + chi_right)))))
        elif self.zeff_feature_coefficients is not None and zeff_left is not None and zeff_right is not None:
            features = (1.0, abs(float(zeff_left) - float(zeff_right)), 0.5 * (float(zeff_left) + float(zeff_right)))
            amplitude = float(sum(a * b for a, b in zip(self.zeff_feature_coefficients, features)))
        elif self.covalent_feature_coefficients is not None:
            chi_left = pauling_electronegativity(z_left) or 0.0
            chi_right = pauling_electronegativity(z_right) or 0.0
            features = (1.0, abs(chi_left - chi_right), period_difference(z_left, z_right))
            amplitude = float(sum(a * b for a, b in zip(self.covalent_feature_coefficients, features)))
        else:
            amplitude = float(self.covalent_amplitudes.get(key, self.covalent_amplitudes.get(class_key, 0.0)))
        center = covalent_gaussian_center(z_left, z_right, bond_order)
        width = covalent_gaussian_width(z_left, z_right, bond_order, self.covalent_sigma_angstrom)
        gaussian = amplitude * exp(-((float(distance) - center) / width) ** 2)
        linear = 0.0
        if self.linear_order_coefficients is not None:
            linear = float(self.linear_order_coefficients.get(key, 0.0)) * float(bond_order)
        if 1 in pair:
            from .structural_corrections import CV_RADIAL_COVALENT_RADII_ANGSTROM, CV_RADIAL_SIGMA_SCALE
            ra = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(pair[0]); rb = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(pair[1])
            if ra is not None and rb is not None:
                r0 = ra + rb
                weight = exp(-((float(distance) - r0) / (CV_RADIAL_SIGMA_SCALE * r0)) ** 2)
                amp0 = float((self.xh_cv_intercepts or {}).get(key, 0.0))
                amp1 = float((self.xh_cv_slopes or {}).get(key, 0.0))
                linear += (amp0 + amp1 * float(bond_order)) * weight
        return gaussian + linear

    def hbond_delta(
        self, donor_z: int, acceptor_z: int, distance: float, angle_radians: float
    ) -> float:
        key = _pair_key(donor_z, acceptor_z)
        amplitude = float(self.hbond_amplitudes.get(key, self.hbond_amplitudes.get("H-bond", 0.0)))
        center = hbond_gaussian_center(acceptor_z)
        if center is None:
            return 0.0
        return amplitude * exp(-((float(distance) - center) / self.hbond_sigma_angstrom) ** 2) * hbond_angular_factor(angle_radians)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "covalent_sigma_angstrom": self.covalent_sigma_angstrom,
            "hbond_sigma_angstrom": self.hbond_sigma_angstrom,
            "covalent_amplitudes": dict(sorted(self.covalent_amplitudes.items())),
            "hbond_amplitudes": dict(sorted(self.hbond_amplitudes.items())),
            "covalent_feature_coefficients": list(self.covalent_feature_coefficients) if self.covalent_feature_coefficients is not None else None,
            "zeff_feature_coefficients": list(self.zeff_feature_coefficients) if self.zeff_feature_coefficients is not None else None,
            "atomic_amplitudes": dict(sorted(self.atomic_amplitudes.items())) if self.atomic_amplitudes is not None else None,
            "delta_zeff_coefficients": list(self.delta_zeff_coefficients) if self.delta_zeff_coefficients is not None else None,
            "electronegativity_coefficients": list(self.electronegativity_coefficients) if self.electronegativity_coefficients is not None else None,
            "atomic_trend_coefficients": list(self.atomic_trend_coefficients) if self.atomic_trend_coefficients is not None else None,
            "linear_order_coefficients": dict(sorted(self.linear_order_coefficients.items())) if self.linear_order_coefficients is not None else None,
            "xh_order_intercepts": dict(sorted(self.xh_order_intercepts.items())) if self.xh_order_intercepts is not None else None,
            "xh_cv_slopes": dict(sorted(self.xh_cv_slopes.items())) if self.xh_cv_slopes is not None else None,
            "xh_cv_intercepts": dict(sorted(self.xh_cv_intercepts.items())) if self.xh_cv_intercepts is not None else None,
            "xh_cv_scale": dict(sorted(self.xh_cv_scale.items())) if self.xh_cv_scale is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PL1GaussianModel":
        return cls(
            float(payload["covalent_sigma_angstrom"]),
            float(payload["hbond_sigma_angstrom"]),
            {str(k): float(v) for k, v in dict(payload.get("covalent_amplitudes", {})).items()},
            {str(k): float(v) for k, v in dict(payload.get("hbond_amplitudes", {})).items()},
            tuple(float(v) for v in payload["covalent_feature_coefficients"]) if payload.get("covalent_feature_coefficients") is not None else None,
            tuple(float(v) for v in payload["zeff_feature_coefficients"]) if payload.get("zeff_feature_coefficients") is not None else None,
            schema=str(payload.get("schema", "matrix.architect.pl1_gaussian_model.v2")),
            atomic_amplitudes={str(k): float(v) for k, v in dict(payload.get("atomic_amplitudes", {})).items()} if payload.get("atomic_amplitudes") is not None else None,
            delta_zeff_coefficients=tuple(float(v) for v in payload["delta_zeff_coefficients"]) if payload.get("delta_zeff_coefficients") is not None else None,
            electronegativity_coefficients=tuple(float(v) for v in payload["electronegativity_coefficients"]) if payload.get("electronegativity_coefficients") is not None else None,
            atomic_trend_coefficients=tuple(float(v) for v in payload["atomic_trend_coefficients"]) if payload.get("atomic_trend_coefficients") is not None else None,
            linear_order_coefficients={str(k): float(v) for k, v in dict(payload.get("linear_order_coefficients", {})).items()} if payload.get("linear_order_coefficients") is not None else None,
            xh_order_intercepts={str(k): float(v) for k, v in dict(payload.get("xh_order_intercepts", {})).items()} if payload.get("xh_order_intercepts") is not None else None,
            xh_cv_slopes={str(k): float(v) for k, v in dict(payload.get("xh_cv_slopes", {})).items()} if payload.get("xh_cv_slopes") is not None else None,
            xh_cv_intercepts={str(k): float(v) for k, v in dict(payload.get("xh_cv_intercepts", {})).items()} if payload.get("xh_cv_intercepts") is not None else None,
            xh_cv_scale={str(k): float(v) for k, v in dict(payload.get("xh_cv_scale", {})).items()} if payload.get("xh_cv_scale") is not None else None,
        )


def load_pl1_gaussian_model(path: str | Path) -> PL1GaussianModel:
    return PL1GaussianModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_pl1_electronegativity_model(
    covalent_observations: Sequence[Mapping[str, float | int]],
    hbond_amplitude: float,
    *,
    covalent_sigma_angstrom: float = 0.08,
    hbond_sigma_angstrom: float = 0.50,
    ridge: float = 1.0e-8,
) -> PL1GaussianModel:
    """Fit one covalent amplitude law from electronegativity only."""
    import numpy as np
    rows = []
    values = []
    for row in covalent_observations:
        chi_left = pauling_electronegativity(int(row["z_left"])) or 0.0
        chi_right = pauling_electronegativity(int(row["z_right"])) or 0.0
        center = covalent_gaussian_center(int(row["z_left"]), int(row["z_right"]), float(row["bond_order"]))
        width = covalent_gaussian_width(int(row["z_left"]), int(row["z_right"]), float(row["bond_order"]), covalent_sigma_angstrom)
        gaussian = exp(-((float(row["distance"]) - center) / width) ** 2)
        rows.append([gaussian, gaussian * abs(chi_left - chi_right), gaussian * period_difference(int(row["z_left"]), int(row["z_right"]))])
        values.append(float(row["residual"]))
    matrix = np.asarray(rows, dtype=float)
    normal = matrix.T @ matrix + float(ridge) * np.eye(3)
    coeff = np.linalg.solve(normal, matrix.T @ np.asarray(values, dtype=float))
    return PL1GaussianModel(covalent_sigma_angstrom, hbond_sigma_angstrom, {}, {"H-bond": float(hbond_amplitude)}, tuple(float(v) for v in coeff), schema="matrix.architect.pl1_gaussian_model.electronegativity.v1")


def fit_pl1_zeff_model(covalent_observations: Sequence[Mapping[str, float | int]], hbond_amplitude: float, *, covalent_sigma_angstrom: float = 0.08, hbond_sigma_angstrom: float = 0.50, ridge: float = 1.0e-8) -> PL1GaussianModel:
    """Fit the compact covalent amplitude law from absolute synthon Zeff."""
    import numpy as np
    rows = []; values = []
    for row in covalent_observations:
        if "zeff_left" not in row or "zeff_right" not in row:
            continue
        center = covalent_gaussian_center(int(row["z_left"]), int(row["z_right"]), float(row["bond_order"]))
        width = covalent_gaussian_width(int(row["z_left"]), int(row["z_right"]), float(row["bond_order"]), covalent_sigma_angstrom)
        gaussian = exp(-((float(row["distance"]) - center) / width) ** 2)
        zl = float(row["zeff_left"]); zr = float(row["zeff_right"])
        rows.append([gaussian, gaussian * abs(zl - zr), gaussian * 0.5 * (zl + zr)])
        values.append(float(row["residual"]))
    matrix = np.asarray(rows, dtype=float); normal = matrix.T @ matrix + float(ridge) * np.eye(3)
    coeff = np.linalg.solve(normal, matrix.T @ np.asarray(values, dtype=float))
    return PL1GaussianModel(covalent_sigma_angstrom, hbond_sigma_angstrom, {}, {"H-bond": float(hbond_amplitude)}, None, tuple(float(v) for v in coeff), schema="matrix.architect.pl1_gaussian_model.zeff.v1")


def fit_pl1_atomic_delta_zeff_model(covalent_observations: Sequence[Mapping[str, float | int]], hbond_amplitude: float, *, covalent_sigma_angstrom: float = 0.08, hbond_sigma_angstrom: float = 0.50, ridge: float = 1.0e-8) -> PL1GaussianModel:
    """Fit atom-wise amplitudes plus a two-partner ΔZ correction."""
    import numpy as np
    rows = []; values = []; elements = sorted({str(int(row["z_left"])) for row in covalent_observations} | {str(int(row["z_right"])) for row in covalent_observations})
    for row in covalent_observations:
        if "zeff_left" not in row or "zeff_right" not in row:
            continue
        z1, z2 = int(row["z_left"]), int(row["z_right"])
        center = covalent_gaussian_center(z1, z2, float(row["bond_order"]))
        width = covalent_gaussian_width(z1, z2, float(row["bond_order"]), covalent_sigma_angstrom)
        gaussian = exp(-((float(row["distance"]) - center) / width) ** 2)
        dz1, dz2 = float(row["zeff_left"]) - z1, float(row["zeff_right"]) - z2
        feat = [0.0] * (len(elements) + 2)
        feat[elements.index(str(z1))] += 0.5 * gaussian
        feat[elements.index(str(z2))] += 0.5 * gaussian
        feat[-2] = gaussian * abs(dz1 - dz2)
        feat[-1] = gaussian * 0.5 * (dz1 + dz2)
        rows.append(feat); values.append(float(row["residual"]))
    matrix = np.asarray(rows, dtype=float); normal = matrix.T @ matrix + float(ridge) * np.eye(matrix.shape[1])
    coeff = np.linalg.solve(normal, matrix.T @ np.asarray(values, dtype=float))
    return PL1GaussianModel(covalent_sigma_angstrom, hbond_sigma_angstrom, {}, {"H-bond": float(hbond_amplitude)}, schema="matrix.architect.pl1_gaussian_model.atomic_delta_zeff.v1", atomic_amplitudes={e: float(c) for e, c in zip(elements, coeff[:-2])}, delta_zeff_coefficients=tuple(float(v) for v in coeff[-2:]))


def fit_pl1_atomic_mixed_model(covalent_observations: Sequence[Mapping[str, float | int]], hbond_amplitude: float, *, covalent_sigma_angstrom: float = 0.08, hbond_sigma_angstrom: float = 0.50, ridge: float = 1.0e-8, robust_iterations: int = 3, huber_threshold_angstrom: float = 0.01) -> PL1GaussianModel:
    """Fit atomic terms plus electronegativity and two-partner ΔZ corrections."""
    import numpy as np
    elements = sorted({str(int(r["z_left"])) for r in covalent_observations} | {str(int(r["z_right"])) for r in covalent_observations})
    rows = []; values = []
    for row in covalent_observations:
        if "zeff_left" not in row or "zeff_right" not in row: continue
        z1, z2 = int(row["z_left"]), int(row["z_right"]); c = covalent_gaussian_center(z1, z2, float(row["bond_order"])); w = covalent_gaussian_width(z1, z2, float(row["bond_order"]), covalent_sigma_angstrom); g = exp(-((float(row["distance"]) - c) / w) ** 2)
        dz1, dz2 = float(row["zeff_left"]) - z1, float(row["zeff_right"]) - z2; chi1 = pauling_electronegativity(z1) or 0.0; chi2 = pauling_electronegativity(z2) or 0.0
        f = [0.0] * (len(elements) + 4); f[elements.index(str(z1))] += 0.5*g; f[elements.index(str(z2))] += 0.5*g; f[-4] = g*abs(dz1-dz2); f[-3] = g*0.5*(dz1+dz2); f[-2] = g*abs(chi1-chi2); f[-1] = g*0.5*(chi1+chi2); rows.append(f); values.append(float(row["residual"]))
    matrix = np.asarray(rows, float); target = np.asarray(values, float); weights = np.ones(len(target), float)
    coeff = np.zeros(matrix.shape[1], float)
    for _ in range(max(1, int(robust_iterations))):
        weighted = matrix * weights[:, None]
        coeff = np.linalg.solve(matrix.T @ weighted + ridge*np.eye(matrix.shape[1]), matrix.T @ (weights * target))
        residuals = target - matrix @ coeff
        scale = np.maximum(np.abs(residuals), 1.0e-12)
        weights = np.minimum(1.0, float(huber_threshold_angstrom) / scale)
    atomic = {e: float(c) for e,c in zip(elements, coeff[:-4])}
    trend_rows = np.asarray([[1.0, GROUP_BY_Z.get(int(e), 0), PERIOD_BY_Z.get(int(e), 0)] for e in elements], float); trend = np.linalg.lstsq(trend_rows, np.asarray([atomic[e] for e in elements]), rcond=None)[0]
    # Add only a homogeneous (zero-intercept) Mayer-order term for pairs not
    # covered by the published C=C/C=S Conj branch.  Fit it to the residual
    # left after the atomic/descriptor model, so it is genuinely additive.
    linear: dict[str, float] = {}
    xh_intercepts: dict[str, float] = {}
    xh_cv_intercepts: dict[str, float] = {}
    xh_cv_slopes: dict[str, float] = {}
    fitted = matrix @ coeff
    for pair in sorted({str(int(r["z_left"])) + "-" + str(int(r["z_right"])) if int(r["z_left"]) <= int(r["z_right"]) else str(int(r["z_right"])) + "-" + str(int(r["z_left"])) for r in covalent_observations}):
        za, zb = (int(v) for v in pair.split("-"))
        if (za, zb) in {(6, 6), (6, 16)}:
            continue
        idx = [i for i, r in enumerate(covalent_observations) if tuple(sorted((int(r["z_left"]), int(r["z_right"])))) == (za, zb)]
        if not idx:
            continue
        orders = np.asarray([float(covalent_observations[i]["bond_order"]) for i in idx], float)
        residual = np.asarray([float(values[i]) for i in idx], float) - fitted[np.asarray(idx, dtype=int)]
        denom = float(np.dot(orders, orders))
        if denom > 1.0e-14:
            if 1 in (za, zb):
                from .structural_corrections import CV_RADIAL_COVALENT_RADII_ANGSTROM, CV_RADIAL_SIGMA_SCALE
                ra = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(za); rb = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(zb)
                if ra is not None and rb is not None:
                    dist = np.asarray([float(covalent_observations[i]["distance"]) for i in idx], float)
                    weight = np.exp(-((dist - (ra + rb)) / (CV_RADIAL_SIGMA_SCALE * (ra + rb))) ** 2)
                    design = np.column_stack((weight, orders * weight))
                    intercept, slope = np.linalg.lstsq(design, residual, rcond=None)[0]
                    xh_cv_intercepts[pair] = float(intercept)
                    xh_cv_slopes[pair] = float(slope)
            else:
                linear[pair] = float(np.dot(orders, residual) / denom)
    return PL1GaussianModel(covalent_sigma_angstrom, hbond_sigma_angstrom, {}, {"H-bond": float(hbond_amplitude)}, schema="matrix.architect.pl1_gaussian_model.atomic_mixed.v1", atomic_amplitudes=atomic, delta_zeff_coefficients=tuple(float(v) for v in coeff[-4:-2]), electronegativity_coefficients=tuple(float(v) for v in coeff[-2:]), atomic_trend_coefficients=tuple(float(v) for v in trend), linear_order_coefficients=linear, xh_order_intercepts=xh_intercepts, xh_cv_intercepts=xh_cv_intercepts, xh_cv_slopes=xh_cv_slopes)


def fit_pl1_gaussian_model(
    covalent_observations: Sequence[Mapping[str, float | int]],
    hbond_observations: Sequence[Mapping[str, float | int]],
    *,
    covalent_sigma_angstrom: float = 0.08,
    hbond_sigma_angstrom: float = 0.50,
    ridge: float = 1.0e-8,
) -> PL1GaussianModel:
    """Fit amplitudes with fixed chemically interpretable Gaussian centers."""
    import numpy as np

    def solve(rows: Sequence[Mapping[str, float | int]], kind: str) -> dict[str, float]:
        if not rows:
            return {}
        keys = []
        features = []
        values = []
        for row in rows:
            if kind == "covalent":
                order = max(1.0, min(3.0, float(row["bond_order"])))
                key = covalent_class_key(int(row["z_left"]), int(row["z_right"]))
                center = covalent_gaussian_center(int(row["z_left"]), int(row["z_right"]), order)
                width = covalent_gaussian_width(int(row["z_left"]), int(row["z_right"]), order, covalent_sigma_angstrom)
            else:
                key = "H-bond"
                center = hbond_gaussian_center(int(row["acceptor_z"]))
                if center is None:
                    continue
                width = hbond_sigma_angstrom
            if key not in keys:
                keys.append(key)
            features.append((key, exp(-((float(row["distance"]) - center) / width) ** 2)))
            values.append(float(row["residual"]))
        if not keys:
            return {}
        matrix = np.zeros((len(features), len(keys)), dtype=float)
        for i, (key, value) in enumerate(features):
            matrix[i, keys.index(key)] = value
        normal = matrix.T @ matrix + float(ridge) * np.eye(len(keys))
        coeff = np.linalg.solve(normal, matrix.T @ np.asarray(values, dtype=float))
        limit = 0.08 if kind == "hbond" else 0.05
        return {key: float(np.clip(value, -limit, limit)) for key, value in zip(keys, coeff)}

    return PL1GaussianModel(
        covalent_sigma_angstrom,
        hbond_sigma_angstrom,
        solve(covalent_observations, "covalent"),
        solve(hbond_observations, "hbond"),
    )


__all__ = [
    "PL1GaussianModel",
    "covalent_gaussian_center",
    "covalent_class_key",
    "covalent_gaussian_width",
    "fit_pl1_gaussian_model",
    "fit_pl1_electronegativity_model",
    "fit_pl1_zeff_model",
    "fit_pl1_atomic_delta_zeff_model",
    "fit_pl1_atomic_mixed_model",
    "hbond_gaussian_center",
    "load_pl1_gaussian_model",
    "vdw_gaussian_center",
]
