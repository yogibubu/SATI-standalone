"""Complementary ONIOM and harmonic boundary repair, DOI 10.1021/acs.jctc.6c00816.

Eqs. 15--23, 31. Explicit postprocessing; no optimizer/backend policy changes.
"""
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from .fragment_oniom import CartesianEvaluation, FragmentOniomEvaluation


@dataclass(frozen=True)
class ParentOniomField:
    """Already assembled parent ONIOM field with declared correction domains.

    Domains label high-minus-low interactions, not just atoms. Excited-state
    identity across partitions must be established by the caller.
    """
    identifier: str
    evaluation: CartesianEvaluation
    coordinates_bohr: np.ndarray
    domains: tuple[str, ...]
    weight: float = 1.0


def assemble_complementary_oniom(low_full: CartesianEvaluation,
                                coordinates_bohr: np.ndarray,
                                fields: Sequence[ParentOniomField], *,
                                required_domains: Sequence[str],
                                coverage_tolerance: float = 1e-10) -> FragmentOniomEvaluation:
    """E_LL + sum a_k(E_ONIOM,k - E_LL), identically for g and H.

    Require unit coverage of each explicitly declared interaction domain.
    Signed inclusion/exclusion weights are allowed. Coverage is declarative:
    it does not prove locality, state consistency or absence of boundary errors.
    All input fields need the same geometry, atom order and Cartesian frame.
    """
    x = np.asarray(coordinates_bohr, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError('parent coordinates must be finite natoms by 3 in bohr')
    if low_full.gradient_hartree_per_bohr.size != x.size:
        raise ValueError('low parent field and geometry dimensions differ')
    if not np.isfinite(coverage_tolerance) or coverage_tolerance <= 0:
        raise ValueError('coverage tolerance must be finite and positive')
    required = tuple(required_domains)
    if not required or len(set(required)) != len(required) or any(not s for s in required):
        raise ValueError('declare distinct nonempty required interaction domains')
    fields = tuple(fields)
    identifiers = tuple(f.identifier for f in fields)
    if not fields or len(set(identifiers)) != len(fields) or any(not s for s in identifiers):
        raise ValueError('complementary fields need distinct nonempty identifiers')
    coverage = dict.fromkeys(required, 0.0)
    for f in fields:
        y = np.asarray(f.coordinates_bohr, dtype=float)
        if y.shape != x.shape or not np.allclose(y, x, rtol=0, atol=1e-10):
            raise ValueError(f'{f.identifier}: parent geometry/frame mismatch')
        if f.evaluation.gradient_hartree_per_bohr.size != x.size:
            raise ValueError(f'{f.identifier}: parent derivative dimension mismatch')
        if not np.isfinite(f.weight) or not np.isfinite(f.evaluation.energy_hartree):
            raise ValueError('weights and energies must be finite')
        if not f.domains or len(set(f.domains)) != len(f.domains):
            raise ValueError('each field must declare distinct correction domains')
        for domain in f.domains:
            if domain not in coverage:
                raise ValueError(f'undeclared interaction domain: {domain}')
            coverage[domain] += f.weight
    failed = {k: v for k, v in coverage.items() if abs(v - 1) > coverage_tolerance}
    if failed:
        raise ValueError(f'high-minus-low interaction coverage is not unity: {failed}')
    energy = low_full.energy_hartree
    gradient = low_full.gradient_hartree_per_bohr.copy()
    hessian = low_full.hessian_hartree_per_bohr2.copy()
    if not np.isfinite(energy):
        raise ValueError('low parent energy must be finite')
    for f in fields:
        e, a = f.evaluation, f.weight
        energy += a * (e.energy_hartree - low_full.energy_hartree)
        gradient += a * (e.gradient_hartree_per_bohr - low_full.gradient_hartree_per_bohr)
        hessian += a * (e.hessian_hartree_per_bohr2 - low_full.hessian_hartree_per_bohr2)
    return FragmentOniomEvaluation(float(energy), gradient, (hessian + hessian.T) / 2, identifiers)


def _symmetric(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size, size) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name}: invalid shape or values')
    if not np.allclose(a, a.T, atol=1e-12, rtol=1e-10):
        raise ValueError(f'{name} must be symmetric')
    return (a + a.T) / 2


@dataclass(frozen=True)
class HarmonicBoundaryCorrection:
    """Auxiliary high-minus-ONIOM internal force block, with explicit provenance.

    Use identical coordinate definitions, order, units and electronic state for
    both auxiliary fields. Stretches in bohr, angles in radians, energy hartree.
    Transferability to parent coordinates is a chemical assumption.
    """
    labels: tuple[str, ...]
    delta_force_constants: np.ndarray
    provenance: str

    @classmethod
    def from_auxiliary(cls, labels, high_force, oniom_force, *, provenance):
        names = tuple(labels)
        return cls(names, _symmetric(high_force, len(names), 'auxiliary high')
                   - _symmetric(oniom_force, len(names), 'auxiliary ONIOM'), provenance)

    def __post_init__(self):
        if not self.labels or len(set(self.labels)) != len(self.labels) or any(not s for s in self.labels):
            raise ValueError('boundary coordinates need distinct nonempty labels')
        if not self.provenance.strip():
            raise ValueError('auxiliary-model provenance is required')
        object.__setattr__(self, 'delta_force_constants',
                           _symmetric(self.delta_force_constants, len(self.labels), 'boundary correction'))


@dataclass(frozen=True)
class CorrectedHarmonicField:
    """Harmonic postprocessing result; deliberately not a PES calculator."""
    force_constants: np.ndarray
    cartesian_hessian_correction: np.ndarray
    reference_coordinates_bohr: np.ndarray
    corrected_labels: tuple[str, ...]
    provenance: str
    harmonic_only: bool = True


def apply_harmonic_boundary_correction(force_constants, coordinate_labels,
                                       b_matrix_per_bohr, reference_coordinates_bohr,
                                       correction: HarmonicBoundaryCorrection) -> CorrectedHarmonicField:
    """Apply an additive calibrated block in an independent internal basis.

    B=ds/dx_bohr, s in bohr/radians. At the declared reference delta_g=0,
    delta_H=B.T delta_F B: there is no missing gradient-times-B' contribution.
    This repairs one harmonic field, not a differentiated global PES. Recompute
    GF modes and transform anharmonic tensors to their new basis afterwards.
    """
    labels = tuple(coordinate_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('parent internal-coordinate labels must be unique')
    force = _symmetric(force_constants, len(labels), 'parent internal force')
    x = np.asarray(reference_coordinates_bohr, dtype=float)
    b = np.asarray(b_matrix_per_bohr, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError('reference must be finite natoms by 3 in bohr')
    if b.shape != (len(labels), x.size) or not np.all(np.isfinite(b)):
        raise ValueError('B matrix does not match the parent chart')
    if np.linalg.matrix_rank(b) != len(labels):
        raise ValueError('boundary transfer requires an independent internal basis')
    positions = {label: i for i, label in enumerate(labels)}
    try:
        indices = [positions[label] for label in correction.labels]
    except KeyError as exc:
        raise ValueError(f'boundary coordinate absent from parent chart: {exc}') from exc
    delta = np.zeros_like(force)
    delta[np.ix_(indices, indices)] = correction.delta_force_constants
    cartesian = b.T @ delta @ b
    return CorrectedHarmonicField(force + delta, (cartesian + cartesian.T) / 2,
                                 x.copy(), correction.labels, correction.provenance)
