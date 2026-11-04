from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

import numpy as np

from .harmonic import CM_PER_HARTREE, solve_wilson_gf


DEFAULT_LARGE_AMPLITUDE_FREQUENCY_CUTOFF_CM = 250.0
DEFAULT_LARGE_AMPLITUDE_FG_COUPLING_TOLERANCE = 0.05
DEFAULT_TORSION_DOUBLE_BOND_ORDER_THRESHOLD = 1.45
DEFAULT_TORSION_PI_INDEX_THRESHOLD = 0.45
DEFAULT_SYNTHON_ZEFF_EQUIVALENCE_TOLERANCE = 0.035
DEFAULT_G_INVERSE_RCOND = 1.0e-12
DEFAULT_HINDERED_ROTOR_BARRIER_CUTOFF_KCAL_MOL = 20.0
CM_PER_KCAL_MOL = 349.755088918
DEFAULT_METRIC_ROLE = "EQUILIBRIUM_REFERENCE_ONLY"
DEFAULT_KINETIC_OPERATOR_STATUS = "REQUIRES_PODOLSKY_GRID_METRIC"
ANHARMONIC_CLASS_LOCAL = "LOCAL_ANHARMONIC"
ANHARMONIC_CLASS_XH_LOCAL = "LOCAL_XH_STRETCH_UNSYMMETRIZED"
ANHARMONIC_CLASS_NORMAL = "NORMAL_MODE_VPT2"
ANHARMONIC_CLASS_PENDING = "PENDING_TOPOLOGY_OR_ANHARMONIC_DATA"
ZEROTH_ORDER_LOCAL = "VSCF_OR_DVR_LOCAL_MODE"
ZEROTH_ORDER_XH_LOCAL = "UNSYMMETRIZED_XH_LOCAL_MODE_VSCF_OR_DVR"
ZEROTH_ORDER_NORMAL = "COUPLED_HARMONIC_NORMAL_MODES"
ZEROTH_ORDER_PENDING = "REQUIRES_TOPOLOGY_OR_SCAN_OR_QFF"
CROSS_COUPLING_LOCAL = "REQUIRE_SMALL_CROSS_SUBSPACE_COUPLING_THEN_VPT2"
CROSS_COUPLING_NORMAL = "FULL_NORMAL_MODE_VPT2"
CROSS_COUPLING_PENDING = "REQUIRES_SUBSPACE_COUPLING_CHECK"
DEFAULT_ANHARMONIC_CLASS = ANHARMONIC_CLASS_PENDING
DEFAULT_ZEROTH_ORDER_MODEL = ZEROTH_ORDER_PENDING
DEFAULT_CROSS_COUPLING_POLICY = CROSS_COUPLING_PENDING
DEFAULT_LARGE_AMPLITUDE_FAMILIES = (
    "local_xh_stretch",
    "torsion",
    "ring_puckering",
    "butterfly",
    "oop",
    "linear",
    "special",
)


@dataclass(frozen=True)
class LargeAmplitudeTopologyContext:
    """Topology/synthon context fixed by preprocessing and consumed by GF only."""

    atomic_numbers: tuple[int, ...] = ()
    bonds: tuple[tuple[int, int], ...] = ()
    bond_orders: dict[tuple[int, int], float] = field(default_factory=dict)
    bond_order_components: dict[tuple[int, int], tuple[float, float, float]] = field(
        default_factory=dict
    )
    synthon_signatures: dict[int, tuple[object, ...]] = field(default_factory=dict)
    synthon_zeff: dict[int, float] = field(default_factory=dict)
    coordinates_angstrom: tuple[tuple[float, float, float], ...] = ()
    bond_order_source: str = ""
    synthon_zeff_tolerance: float = DEFAULT_SYNTHON_ZEFF_EQUIVALENCE_TOLERANCE

    def graph(self) -> dict[int, set[int]]:
        graph: dict[int, set[int]] = {}
        for left, right in self.bonds:
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
        return graph

    def bond_order(self, left: int, right: int) -> float | None:
        key = _pair_key(left, right)
        return self.bond_orders.get(key)

    def bond_indices(self, left: int, right: int) -> tuple[float, float, float] | None:
        key = _pair_key(left, right)
        values = self.bond_order_components.get(key)
        if values is not None:
            return values
        total = self.bond_orders.get(key)
        if total is None:
            return None
        from matrix_chem import bond_order_components

        resolved = bond_order_components(total)
        return resolved.sigma, resolved.pi, resolved.pi_pi

    def total_pi_index(self, left: int, right: int) -> float | None:
        values = self.bond_indices(left, right)
        return None if values is None else float(values[1] + values[2])

    def synthon_signature(self, atom: int) -> tuple[object, ...]:
        signature = self.synthon_signatures.get(atom)
        if signature is not None:
            return signature
        if 1 <= atom <= len(self.atomic_numbers):
            return (self.atomic_numbers[atom - 1],)
        return (atom,)

    def synthon_zeff_value(self, atom: int) -> float | None:
        value = self.synthon_zeff.get(atom)
        if value is None and 1 <= atom <= len(self.atomic_numbers):
            value = float(self.atomic_numbers[atom - 1])
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value


@dataclass(frozen=True)
class AnharmonicModelAssignment:
    family: str
    anharmonic_class: str
    zeroth_order_model: str
    cross_coupling_policy: str
    reason: str


@dataclass(frozen=True)
class LargeAmplitudeCoordinate:
    index: int
    name: str
    irrep: str
    family: str
    label: str
    local_frequency_cm: float | None = None
    active: bool = True
    status: str = "ACTIVE"
    anharmonic_class: str = DEFAULT_ANHARMONIC_CLASS
    zeroth_order_model: str = DEFAULT_ZEROTH_ORDER_MODEL
    cross_coupling_policy: str = DEFAULT_CROSS_COUPLING_POLICY
    anharmonic_reason: str = ""


@dataclass(frozen=True)
class LargeAmplitudeBlock:
    label: str
    family: str
    indices: tuple[int, ...]
    frequencies_cm: tuple[float, ...]
    coordinate_frequencies_cm: tuple[float, ...]
    frequency_model: str
    g_inverse_block: tuple[tuple[float, ...], ...]
    g_inverse_source: str
    metric_role: str
    kinetic_operator_status: str
    anharmonic_class: str
    zeroth_order_model: str
    cross_coupling_policy: str
    anharmonic_reason: str
    max_f_coupling_to_rest: float
    max_g_coupling_to_rest: float
    relative_f_coupling_to_rest: float
    relative_g_coupling_to_rest: float

    @property
    def relative_fg_coupling_to_rest(self) -> float:
        """Dimensionless diagnostic that treats F and G couplings symmetrically."""
        return max(self.relative_f_coupling_to_rest, self.relative_g_coupling_to_rest)


@dataclass(frozen=True)
class LargeAmplitudeModeContribution:
    mode: int
    frequency_cm: float
    ped_percent: float


@dataclass(frozen=True)
class LargeAmplitudeDVRCandidate:
    index: int
    name: str
    irrep: str
    family: str
    status: str
    frequency_cm: float | None
    fg_coupling_to_rest: float
    central_bond: tuple[int, int] | None = None
    periodicity: int | None = None
    minimum_rad: float | None = None
    force_constant_hartree: float | None = None
    fourier_amplitude_cm: float | None = None
    barrier_cm: float | None = None
    barrier_kcal_mol: float | None = None
    rotor_symmetry_number: int | None = None
    rotor_multiplicity: int | None = None
    hindered_rotor_status: str = ""
    hindered_rotor_source: str = ""
    g_inverse_diagonal: float | None = None
    g_inverse_source: str = ""
    reason: str = ""


@dataclass(frozen=True)
class OneTermFourierBootstrap:
    """Local periodic DVR seed fixed by a minimum, curvature and periodicity."""

    force_constant_hartree: float
    periodicity: int
    amplitude_cm: float
    barrier_cm: float


@dataclass(frozen=True)
class HinderedRotorEstimate:
    """Schlegel/Ayala-style one-dimensional rotor metadata from a SONIC torsion."""

    central_bond: tuple[int, int]
    periodicity: int
    symmetry_number: int
    multiplicity: int
    barrier_cm: float
    barrier_kcal_mol: float
    cutoff_kcal_mol: float
    status: str
    source: str = "SONIC_ONE_TERM_FOURIER"


@dataclass(frozen=True)
class GFLargeAmplitudeAnalysis:
    coordinates: tuple[LargeAmplitudeCoordinate, ...]
    blocks: tuple[LargeAmplitudeBlock, ...]
    mode_contributions: tuple[LargeAmplitudeModeContribution, ...]
    dvr_candidates: tuple[LargeAmplitudeDVRCandidate, ...] = ()
    g_inverse: tuple[tuple[float, ...], ...] = ()
    g_inverse_source: str = ""
    families: tuple[str, ...] = DEFAULT_LARGE_AMPLITUDE_FAMILIES
    frequency_cutoff_cm: float | None = DEFAULT_LARGE_AMPLITUDE_FREQUENCY_CUTOFF_CM
    fg_coupling_tolerance: float = DEFAULT_LARGE_AMPLITUDE_FG_COUPLING_TOLERANCE

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinates)

    @property
    def active_coordinate_count(self) -> int:
        return sum(1 for coordinate in self.coordinates if coordinate.active)


def gic_coordinate_family(name: str, label: str) -> str:
    """Return the MATRIX coordinate family encoded in a frozen GIC name/label."""
    text = f"{name} {label}".lower()
    if "local_xh_stretch" in text or "xhst" in text:
        return "local_xh_stretch"
    if any(token in text for token in ("rpck", "qpck", "phip", "pck")):
        return "ring_puckering"
    if any(token in text for token in ("btfl", "butterfly")):
        return "butterfly"
    if any(token in text for token in ("cybe", "cyclic_bend", "ring_bend")):
        return "ring_bend"
    if any(token in text for token in ("fragment", "frag", "centroid", "center", "centre")):
        return "special"
    if re.search(r"\br\s*\(", text) or "str" in text or "stretch" in text or "bond(" in text:
        return "stretch"
    if re.search(r"\ba\s*\(", text) or "bend" in text or "angle(" in text:
        return "bend"
    if re.search(r"\bd\s*\(", text) or "tors" in text or "dih" in text or "dihedral(" in text:
        return "torsion"
    if (
        re.search(r"\bu\s*\(", text)
        or "oupl" in text
        or "oop" in text
        or "improper" in text
        or "out_of_plane(" in text
    ):
        return "oop"
    if re.search(r"\bl\s*\(", text) or "linear_bend" in text or "lin" in text:
        return "linear"
    return ""


def classify_gic_anharmonic_model(
    name: str,
    label: str,
    *,
    topology_context: LargeAmplitudeTopologyContext | None = None,
) -> AnharmonicModelAssignment:
    """Assign the default anharmonic zeroth-order model from chemical GIC type."""
    family = gic_coordinate_family(name, label)
    text = f"{name} {label}"
    if family == "local_xh_stretch":
        return AnharmonicModelAssignment(
            family=family,
            anharmonic_class=ANHARMONIC_CLASS_XH_LOCAL,
            zeroth_order_model=ZEROTH_ORDER_XH_LOCAL,
            cross_coupling_policy=CROSS_COUPLING_LOCAL,
            reason="XH_STRETCH_LOCAL_BASIS_NOT_SYMMETRIZED",
        )
    if family == "oop":
        return _local_assignment(family, "OUT_OF_PLANE")
    if family in {"ring_puckering", "butterfly"}:
        return _ring_torsional_assignment(family, text, topology_context)
    if family == "ring_bend":
        return _local_assignment(family, "RING_MODE")
    if family == "torsion":
        return _torsion_assignment(text, topology_context)
    if family == "special":
        return _local_assignment(family, "PROTECTED_SPECIAL_COORDINATE")
    return AnharmonicModelAssignment(
        family=family,
        anharmonic_class=ANHARMONIC_CLASS_NORMAL,
        zeroth_order_model=ZEROTH_ORDER_NORMAL,
        cross_coupling_policy=CROSS_COUPLING_NORMAL,
        reason="DEFAULT_NORMAL_MODE",
    )


def large_amplitude_analysis_from_gf_matrices(
    *,
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    frequencies_cm: np.ndarray,
    ped: np.ndarray,
    gic_labels: tuple[str, ...],
    gic_names: tuple[str, ...] = (),
    gic_irreps: tuple[str, ...] = (),
    families: tuple[str, ...] = DEFAULT_LARGE_AMPLITUDE_FAMILIES,
    frequency_cutoff_cm: float | None = DEFAULT_LARGE_AMPLITUDE_FREQUENCY_CUTOFF_CM,
    fg_coupling_tolerance: float = DEFAULT_LARGE_AMPLITUDE_FG_COUPLING_TOLERANCE,
    topology_context: LargeAmplitudeTopologyContext | None = None,
    block_frequency_model: str = "full",
    primitive_labels: tuple[str, ...] = (),
    u_matrix: np.ndarray | None = None,
) -> GFLargeAmplitudeAnalysis:
    """Analyze large-amplitude GIC subspaces directly in the non-redundant GF basis."""
    f_mat = np.asarray(force_constants, dtype=float)
    g_mat = np.asarray(g_matrix, dtype=float)
    if f_mat.shape != g_mat.shape or f_mat.ndim != 2 or f_mat.shape[0] != f_mat.shape[1]:
        raise ValueError("large-amplitude analysis needs square F/G matrices of the same shape")
    frequency_model = _normalize_block_frequency_model(block_frequency_model)
    ncoord = f_mat.shape[0]
    g_inverse, g_inverse_source = _invert_g_matrix(g_mat)
    names = _padded(gic_names, ncoord)
    irreps = _padded(gic_irreps, ncoord, fill="UNK")
    labels = _padded(gic_labels, ncoord)
    primitive_sources = _primitive_source_labels(
        primitive_labels=primitive_labels,
        u_matrix=u_matrix,
        ncoord=ncoord,
    )
    selected_families = tuple(dict.fromkeys(family for family in families if family))
    selected = set(selected_families)
    coordinates = []
    by_block: dict[tuple[str, str], list[int]] = {}
    for index in range(ncoord):
        label = labels[index]
        if primitive_sources[index]:
            label = f"{label} {primitive_sources[index]}".strip()
        family = gic_coordinate_family(names[index], label)
        if family not in selected:
            continue
        local_frequency = _local_coordinate_frequency_cm(f_mat, g_mat, index)
        active, status = _coordinate_activity_status(
            local_frequency,
            frequency_cutoff_cm=frequency_cutoff_cm,
        )
        assignment = classify_gic_anharmonic_model(
            names[index],
            labels[index],
            topology_context=topology_context,
        )
        coordinates.append(
            LargeAmplitudeCoordinate(
                index=index + 1,
                name=names[index] or f"GIC{index + 1:03d}",
                irrep=irreps[index] or "UNK",
                family=family,
                label=label,
                local_frequency_cm=local_frequency,
                active=active,
                status=status,
                anharmonic_class=assignment.anharmonic_class,
                zeroth_order_model=assignment.zeroth_order_model,
                cross_coupling_policy=assignment.cross_coupling_policy,
                anharmonic_reason=assignment.reason,
            )
        )
        block_label = _large_amplitude_block_label(
            family,
            names[index],
            labels[index],
            index=index,
            topology_context=topology_context,
        )
        by_block.setdefault((block_label, family), []).append(index)

    blocks = [
        _large_amplitude_block(
            label=label,
            family=family,
            indices=tuple(indices),
            force_constants=f_mat,
            g_matrix=g_mat,
            g_inverse=g_inverse,
            g_inverse_source=g_inverse_source,
            frequency_model=frequency_model,
        )
        for (label, family), indices in by_block.items()
        if indices
    ]
    all_indices = tuple(coordinate.index - 1 for coordinate in coordinates)
    if len(blocks) > 1 and all_indices:
        blocks.append(
            _large_amplitude_block(
                label="all_large_amplitude",
                family="mixed_large_amplitude",
                indices=all_indices,
                force_constants=f_mat,
                g_matrix=g_mat,
                g_inverse=g_inverse,
                g_inverse_source=g_inverse_source,
                frequency_model=frequency_model,
            )
        )

    ped_values = np.asarray(ped, dtype=float)
    mode_contributions: list[LargeAmplitudeModeContribution] = []
    if all_indices and ped_values.ndim == 2 and ped_values.shape[0] == ncoord:
        freqs = np.asarray(frequencies_cm, dtype=float).reshape(-1)
        for mode in range(min(ped_values.shape[1], freqs.size)):
            mode_contributions.append(
                LargeAmplitudeModeContribution(
                    mode=mode + 1,
                    frequency_cm=float(freqs[mode]),
                    ped_percent=float(np.sum(ped_values[list(all_indices), mode])),
                )
            )
    dvr_candidates = tuple(
        _dvr_candidate(
            coordinate,
            force_constants=f_mat,
            g_matrix=g_mat,
            g_inverse=g_inverse,
            g_inverse_source=g_inverse_source,
            topology_context=topology_context,
            fg_coupling_tolerance=fg_coupling_tolerance,
        )
        for coordinate in coordinates
    )
    return GFLargeAmplitudeAnalysis(
        coordinates=tuple(coordinates),
        blocks=tuple(blocks),
        mode_contributions=tuple(mode_contributions),
        dvr_candidates=dvr_candidates,
        g_inverse=tuple(tuple(float(value) for value in row) for row in g_inverse),
        g_inverse_source=g_inverse_source,
        families=selected_families,
        frequency_cutoff_cm=frequency_cutoff_cm,
        fg_coupling_tolerance=float(fg_coupling_tolerance),
    )


def large_amplitude_topology_context_from_arrays(
    *,
    atomic_numbers: np.ndarray,
    coordinates_angstrom: np.ndarray,
    bond_order_overrides: dict[tuple[int, int], float] | None = None,
    bond_order_source: str = "",
) -> LargeAmplitudeTopologyContext:
    """Build the large-amplitude context with the shared MATRIX topology engine."""
    from matrix_chem import build_topology_objects

    numbers = tuple(int(value) for value in np.asarray(atomic_numbers, dtype=int).reshape(-1))
    coords = np.asarray(coordinates_angstrom, dtype=float)
    _continuous, discrete, _ringset, synthons, _aromaticity = build_topology_objects(
        coords,
        numbers,
        bond_order_overrides=bond_order_overrides,
        bond_order_source=bond_order_source or "ORACLE Pauling estimate",
    )
    bonds = tuple((int(i) + 1, int(j) + 1) for i, j in getattr(discrete, "bonds", ()))
    bond_orders: dict[tuple[int, int], float] = {}
    bond_components: dict[tuple[int, int], tuple[float, float, float]] = {}
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            value = float(synthons.bond_order(i, j))
            if value > 0.05:
                key = (i + 1, j + 1)
                bond_orders[key] = value
                components = synthons.bond_order_components(i, j)
                bond_components[key] = (
                    float(components.sigma),
                    float(components.pi),
                    float(components.pi_pi),
                )
    signatures = {
        idx + 1: tuple(synthons.canonical_signature(idx)) for idx in range(len(numbers))
    }
    zeff = {idx + 1: float(synthons.Zeff(idx)) for idx in range(len(numbers))}
    return LargeAmplitudeTopologyContext(
        atomic_numbers=numbers,
        bonds=bonds,
        bond_orders=bond_orders,
        bond_order_components=bond_components,
        synthon_signatures=signatures,
        synthon_zeff=zeff,
        coordinates_angstrom=tuple(tuple(float(x) for x in row) for row in coords),
        bond_order_source=bond_order_source,
    )


def one_term_fourier_bootstrap(
    force_constant_hartree: float,
    periodicity: int,
) -> OneTermFourierBootstrap:
    r"""Return the one-term Fourier potential seed for a periodic coordinate.

    The local model is \(V(q)=A[1-\cos(n(q-q_0))]\).  The harmonic curvature
    at the minimum is \(n^2 A\), so \(A=F_{qq}/n^2\).  The returned barrier is
    the model maximum relative to the minimum, \(2A\), both converted to
    wavenumbers.
    """
    n = int(periodicity)
    if n < 1:
        raise ValueError("periodicity must be a positive integer")
    curvature = float(force_constant_hartree)
    if not math.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("force_constant_hartree must be positive and finite")
    amplitude_cm = curvature * CM_PER_HARTREE / float(n * n)
    return OneTermFourierBootstrap(
        force_constant_hartree=curvature,
        periodicity=n,
        amplitude_cm=amplitude_cm,
        barrier_cm=2.0 * amplitude_cm,
    )


def hindered_rotor_estimate(
    *,
    central_bond: tuple[int, int],
    force_constant_hartree: float,
    periodicity: int,
    symmetry_number: int | None = None,
    multiplicity: int = 1,
    cutoff_kcal_mol: float = DEFAULT_HINDERED_ROTOR_BARRIER_CUTOFF_KCAL_MOL,
) -> HinderedRotorEstimate:
    """Return a one-term hindered-rotor estimate for a SONIC torsion.

    The estimate follows the same data model used by Schlegel/Ayala-type
    hindered-rotor treatments: a rotatable bond, periodicity, symmetry number,
    multiplicity and barrier cutoff.  TRINITY obtains these fields from the
    frozen topology and SONIC metric rather than from redundant internals or
    user-supplied rotor cards.
    """
    n = int(periodicity)
    if n < 1:
        raise ValueError("periodicity must be a positive integer")
    symmetry = int(symmetry_number) if symmetry_number is not None else n
    if symmetry < 1:
        raise ValueError("symmetry_number must be a positive integer")
    mult = int(multiplicity)
    if mult < 1:
        raise ValueError("multiplicity must be a positive integer")
    cutoff = float(cutoff_kcal_mol)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_kcal_mol must be positive and finite")
    bootstrap = one_term_fourier_bootstrap(force_constant_hartree, n)
    barrier_kcal = bootstrap.barrier_cm / CM_PER_KCAL_MOL
    status = (
        "HINDERED_ROTOR_ACTIVE"
        if barrier_kcal <= cutoff
        else "HINDERED_ROTOR_FROZEN_BY_BARRIER"
    )
    return HinderedRotorEstimate(
        central_bond=_pair_key(*central_bond),
        periodicity=n,
        symmetry_number=symmetry,
        multiplicity=mult,
        barrier_cm=bootstrap.barrier_cm,
        barrier_kcal_mol=barrier_kcal,
        cutoff_kcal_mol=cutoff,
        status=status,
    )


def _large_amplitude_block(
    *,
    label: str,
    family: str,
    indices: tuple[int, ...],
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    g_inverse: np.ndarray,
    g_inverse_source: str,
    frequency_model: str,
) -> LargeAmplitudeBlock:
    index = np.asarray(indices, dtype=int)
    f_sub = force_constants[np.ix_(index, index)]
    g_sub = g_matrix[np.ix_(index, index)]
    g_inv_sub = g_inverse[np.ix_(index, index)]
    frequencies = _large_amplitude_block_frequencies(
        f_sub,
        g_sub,
        g_inverse_sub=g_inv_sub,
        frequency_model=frequency_model,
    )
    coordinate_frequencies = tuple(
        float(_signed_frequency_cm(float(f_sub[i, i]) * float(g_sub[i, i])))
        for i in range(len(indices))
    )
    f_coupling, f_relative = _coupling_to_rest(force_constants, indices)
    g_coupling, g_relative = _coupling_to_rest(g_matrix, indices)
    assignment = _block_anharmonic_assignment(family)
    return LargeAmplitudeBlock(
        label=label,
        family=family,
        indices=tuple(int(value) + 1 for value in indices),
        frequencies_cm=tuple(float(value) for value in frequencies),
        coordinate_frequencies_cm=coordinate_frequencies,
        frequency_model=frequency_model,
        g_inverse_block=tuple(tuple(float(value) for value in row) for row in g_inv_sub),
        g_inverse_source=g_inverse_source,
        metric_role=DEFAULT_METRIC_ROLE,
        kinetic_operator_status=DEFAULT_KINETIC_OPERATOR_STATUS,
        anharmonic_class=assignment.anharmonic_class,
        zeroth_order_model=assignment.zeroth_order_model,
        cross_coupling_policy=assignment.cross_coupling_policy,
        anharmonic_reason=assignment.reason,
        max_f_coupling_to_rest=f_coupling,
        max_g_coupling_to_rest=g_coupling,
        relative_f_coupling_to_rest=f_relative,
        relative_g_coupling_to_rest=g_relative,
    )


def _normalize_block_frequency_model(value: str) -> str:
    model = str(value).strip().lower().replace("_", "-")
    aliases = {
        "full": "full",
        "coupled": "full",
        "subspace": "full",
        "projected": "projected",
        "truhlar": "projected",
        "truhlar-projected": "projected",
        "inverse-subblock": "projected",
        "g-inverse-subblock": "projected",
        "diagonal-g": "diagonal-g",
        "diagg": "diagonal-g",
        "g-diagonal": "diagonal-g",
        "separate": "separate",
        "local": "separate",
        "uncoupled": "separate",
    }
    if model not in aliases:
        raise ValueError("block_frequency_model must be full, projected, diagonal-g or separate")
    return aliases[model]


def _large_amplitude_block_frequencies(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    *,
    g_inverse_sub: np.ndarray | None = None,
    frequency_model: str,
) -> np.ndarray:
    if frequency_model == "full":
        return solve_wilson_gf(force_constants, g_matrix, scale_to_cm=True).frequencies_cm
    if frequency_model == "projected":
        if g_inverse_sub is None:
            raise ValueError("projected large-amplitude frequencies need a G inverse subblock")
        projected_g = _projected_subspace_metric(g_inverse_sub)
        return solve_wilson_gf(force_constants, projected_g, scale_to_cm=True).frequencies_cm
    if frequency_model == "diagonal-g":
        return solve_wilson_gf(
            force_constants,
            np.diag(np.diag(g_matrix)),
            scale_to_cm=True,
        ).frequencies_cm
    if frequency_model == "separate":
        values = [
            _signed_frequency_cm(float(force_constants[i, i]) * float(g_matrix[i, i]))
            for i in range(force_constants.shape[0])
        ]
        return np.sort(np.asarray(values, dtype=float))
    raise ValueError("Unsupported large-amplitude block frequency model")


def _projected_subspace_metric(g_inverse_sub: np.ndarray) -> np.ndarray:
    """Return the constrained subspace metric used for projected torsional frequencies."""
    values = np.asarray(g_inverse_sub, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("G inverse subblock must be square")
    symmetric = 0.5 * (values + values.T)
    try:
        projected = np.linalg.inv(symmetric)
    except np.linalg.LinAlgError:
        projected = np.linalg.pinv(symmetric, rcond=DEFAULT_G_INVERSE_RCOND)
    return 0.5 * (projected + projected.T)


def _signed_frequency_cm(eigenvalue: float) -> float:
    scale = 5140.487143715055
    if eigenvalue < 0.0:
        return -float(np.sqrt(abs(eigenvalue)) * scale)
    return float(np.sqrt(eigenvalue) * scale)


def _dvr_candidate(
    coordinate: LargeAmplitudeCoordinate,
    *,
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    g_inverse: np.ndarray,
    g_inverse_source: str,
    topology_context: LargeAmplitudeTopologyContext | None,
    fg_coupling_tolerance: float,
) -> LargeAmplitudeDVRCandidate:
    index0 = coordinate.index - 1
    _f_coupling, f_relative = _coupling_to_rest(force_constants, (index0,))
    _g_coupling, g_relative = _coupling_to_rest(g_matrix, (index0,))
    fg_relative = max(f_relative, g_relative)
    base = {
        "index": coordinate.index,
        "name": coordinate.name,
        "irrep": coordinate.irrep,
        "family": coordinate.family,
        "frequency_cm": coordinate.local_frequency_cm,
        "fg_coupling_to_rest": fg_relative,
        "g_inverse_diagonal": _g_inverse_diagonal(g_inverse, index0),
        "g_inverse_source": g_inverse_source,
    }
    if not coordinate.active:
        return LargeAmplitudeDVRCandidate(
            **base,
            status=coordinate.status,
            reason=coordinate.status,
        )
    if coordinate.family != "torsion":
        return LargeAmplitudeDVRCandidate(
            **base,
            status="ACTIVE_BLOCK_DVR",
            reason="NON_TORSIONAL_LARGE_AMPLITUDE",
        )
    torsions = _torsion_terms(f"{coordinate.name} {coordinate.label}")
    if not torsions:
        return LargeAmplitudeDVRCandidate(
            **base,
            status="ACTIVE_TORSION_NO_PRIMITIVE",
            reason="NO_D_PRIMITIVE_LABEL",
        )
    central_bonds = tuple(dict.fromkeys(_pair_key(term[1], term[2]) for term in torsions))
    if len(central_bonds) != 1:
        return LargeAmplitudeDVRCandidate(
            **base,
            status="ACTIVE_COUPLED_TORSIONS",
            reason="MULTIPLE_CENTRAL_BONDS",
        )
    central_bond = central_bonds[0]
    if topology_context is None:
        return LargeAmplitudeDVRCandidate(
            **base,
            status="PENDING_TOPOLOGY",
            central_bond=central_bond,
            reason="NO_TOPOLOGY_CONTEXT",
        )
    pi_index = topology_context.total_pi_index(*central_bond)
    if pi_index is not None and pi_index >= DEFAULT_TORSION_PI_INDEX_THRESHOLD:
        return LargeAmplitudeDVRCandidate(
            **base,
            status="EXCLUDED_HIGH_BOND_ORDER",
            central_bond=central_bond,
            reason=f"TOTAL_PI_INDEX_GE_{DEFAULT_TORSION_PI_INDEX_THRESHOLD:g}",
        )
    periodicity = _torsion_periodicity(topology_context, central_bond)
    minimum_rad = _torsion_minimum_rad(topology_context, torsions[0])
    force_constant = float(force_constants[index0, index0])
    rotor = None
    if force_constant > 0.0:
        rotor = hindered_rotor_estimate(
            central_bond=central_bond,
            force_constant_hartree=force_constant,
            periodicity=periodicity,
        )
    if fg_relative > float(fg_coupling_tolerance):
        return LargeAmplitudeDVRCandidate(
            **base,
            status="ACTIVE_COUPLED_DVR",
            central_bond=central_bond,
            periodicity=periodicity,
            minimum_rad=minimum_rad,
            force_constant_hartree=force_constant,
            barrier_cm=None if rotor is None else rotor.barrier_cm,
            barrier_kcal_mol=None if rotor is None else rotor.barrier_kcal_mol,
            rotor_symmetry_number=None if rotor is None else rotor.symmetry_number,
            rotor_multiplicity=None if rotor is None else rotor.multiplicity,
            hindered_rotor_status="HINDERED_ROTOR_COUPLED_BLOCK" if rotor is not None else "",
            hindered_rotor_source="" if rotor is None else rotor.source,
            reason="FG_COUPLING_ABOVE_TOLERANCE",
        )
    if force_constant <= 0.0:
        return LargeAmplitudeDVRCandidate(
            **base,
            status="ACTIVE_TORSION_UNSTABLE",
            central_bond=central_bond,
            periodicity=periodicity,
            minimum_rad=minimum_rad,
            force_constant_hartree=force_constant,
            reason="NON_POSITIVE_CURVATURE",
        )
    bootstrap = one_term_fourier_bootstrap(force_constant, periodicity)
    if rotor is None:
        rotor = hindered_rotor_estimate(
            central_bond=central_bond,
            force_constant_hartree=force_constant,
            periodicity=periodicity,
        )
    return LargeAmplitudeDVRCandidate(
        **base,
        status="ACTIVE_TORSION_1D",
        central_bond=central_bond,
        periodicity=periodicity,
        minimum_rad=minimum_rad,
        force_constant_hartree=bootstrap.force_constant_hartree,
        fourier_amplitude_cm=bootstrap.amplitude_cm,
        barrier_cm=bootstrap.barrier_cm,
        barrier_kcal_mol=rotor.barrier_kcal_mol,
        rotor_symmetry_number=rotor.symmetry_number,
        rotor_multiplicity=rotor.multiplicity,
        hindered_rotor_status=rotor.status,
        hindered_rotor_source=rotor.source,
        reason="ONE_TERM_FOURIER",
    )


def _invert_g_matrix(g_matrix: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.asarray(g_matrix, dtype=float)
    symmetric = 0.5 * (values + values.T)
    try:
        inverse = np.linalg.inv(symmetric)
        source = "EQUILIBRIUM_G_INVERSE"
        if not np.all(np.isfinite(inverse)):
            raise np.linalg.LinAlgError("non-finite inverse")
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(symmetric, rcond=DEFAULT_G_INVERSE_RCOND)
        source = "EQUILIBRIUM_G_PSEUDOINVERSE"
    return 0.5 * (inverse + inverse.T), source


def _g_inverse_diagonal(g_inverse: np.ndarray, index: int) -> float | None:
    if index < 0 or index >= g_inverse.shape[0]:
        return None
    value = float(g_inverse[index, index])
    if not math.isfinite(value):
        return None
    return value


def _local_coordinate_frequency_cm(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    index: int,
) -> float | None:
    try:
        gf = solve_wilson_gf(
            np.asarray([[force_constants[index, index]]], dtype=float),
            np.asarray([[g_matrix[index, index]]], dtype=float),
            scale_to_cm=True,
        )
    except ValueError:
        return None
    if gf.frequencies_cm.size == 0:
        return None
    return float(gf.frequencies_cm[0])


def _coordinate_activity_status(
    frequency_cm: float | None,
    *,
    frequency_cutoff_cm: float | None,
) -> tuple[bool, str]:
    if frequency_cm is None:
        return False, "EXCLUDED_INVALID_LOCAL_GF"
    if frequency_cutoff_cm is None:
        return True, "ACTIVE_NO_FREQUENCY_CUTOFF"
    cutoff = float(frequency_cutoff_cm)
    if abs(float(frequency_cm)) <= cutoff:
        return True, "ACTIVE"
    return False, "EXCLUDED_FREQUENCY"


def _coupling_to_rest(matrix: np.ndarray, indices: tuple[int, ...]) -> tuple[float, float]:
    values = np.asarray(matrix, dtype=float)
    selected = np.zeros(values.shape[0], dtype=bool)
    selected[list(indices)] = True
    rest = ~selected
    if not np.any(rest):
        return 0.0, 0.0
    coupling = values[np.ix_(selected, rest)]
    max_abs = float(np.max(np.abs(coupling))) if coupling.size else 0.0
    diag_scale = float(np.max(np.abs(np.diag(values)))) if values.size else 0.0
    return max_abs, max_abs / max(diag_scale, 1.0e-30)


def _torsion_terms(text: str) -> tuple[tuple[int, int, int, int], ...]:
    matches = re.findall(
        r"\bD\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    return tuple(tuple(int(value) for value in match) for match in matches)


def _stretch_terms(text: str) -> tuple[tuple[int, int], ...]:
    matches = re.findall(
        r"\bR\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    return tuple(tuple(int(value) for value in match) for match in matches)


def _large_amplitude_block_label(
    family: str,
    name: str,
    label: str,
    *,
    index: int,
    topology_context: LargeAmplitudeTopologyContext | None,
) -> str:
    if family != "local_xh_stretch":
        return family
    text = f"{name} {label}"
    for left, right in _stretch_terms(text):
        heavy, _hydrogen = _xh_heavy_and_hydrogen(left, right, topology_context)
        if heavy is not None:
            return f"local_xh_stretch_atom{heavy:03d}_{_xh_class_for_heavy_atom(heavy, topology_context)}"
    return f"local_xh_stretch_gic{index + 1:03d}"


def _xh_heavy_and_hydrogen(
    left: int,
    right: int,
    context: LargeAmplitudeTopologyContext | None,
) -> tuple[int | None, int | None]:
    if context is None or not context.atomic_numbers:
        return None, None
    if left < 1 or right < 1:
        return None, None
    if left > len(context.atomic_numbers) or right > len(context.atomic_numbers):
        return None, None
    z_left = int(context.atomic_numbers[left - 1])
    z_right = int(context.atomic_numbers[right - 1])
    if z_left == 1 and z_right != 1:
        return right, left
    if z_right == 1 and z_left != 1:
        return left, right
    return None, None


def _xh_class_for_heavy_atom(
    heavy: int,
    context: LargeAmplitudeTopologyContext | None,
) -> str:
    if context is None:
        return "XH"
    count = 0
    for left, right in context.bonds:
        atom, hydrogen = _xh_heavy_and_hydrogen(left, right, context)
        if atom == heavy and hydrogen is not None:
            count += 1
    if count <= 1:
        return "XH"
    if count == 2:
        return "XH2"
    return "XH3"


def _local_assignment(family: str, reason: str) -> AnharmonicModelAssignment:
    return AnharmonicModelAssignment(
        family=family,
        anharmonic_class=ANHARMONIC_CLASS_LOCAL,
        zeroth_order_model=ZEROTH_ORDER_LOCAL,
        cross_coupling_policy=CROSS_COUPLING_LOCAL,
        reason=reason,
    )


def _torsion_assignment(
    text: str,
    context: LargeAmplitudeTopologyContext | None,
) -> AnharmonicModelAssignment:
    torsions = _torsion_terms(text)
    if not torsions:
        return AnharmonicModelAssignment(
            family="torsion",
            anharmonic_class=ANHARMONIC_CLASS_PENDING,
            zeroth_order_model=ZEROTH_ORDER_PENDING,
            cross_coupling_policy=CROSS_COUPLING_PENDING,
            reason="TORSION_PRIMITIVE_NOT_IDENTIFIED",
        )
    if context is None:
        return AnharmonicModelAssignment(
            family="torsion",
            anharmonic_class=ANHARMONIC_CLASS_PENDING,
            zeroth_order_model=ZEROTH_ORDER_PENDING,
            cross_coupling_policy=CROSS_COUPLING_PENDING,
            reason="TORSION_BOND_ORDER_REQUIRES_TOPOLOGY",
        )
    central_bonds = tuple(dict.fromkeys(_pair_key(term[1], term[2]) for term in torsions))
    pi_indices = [context.total_pi_index(*bond) for bond in central_bonds]
    if pi_indices and all(
        value is not None and value < DEFAULT_TORSION_PI_INDEX_THRESHOLD
        for value in pi_indices
    ):
        return _local_assignment("torsion", "SINGLE_BOND_TORSION")
    if any(
        value is not None and value >= DEFAULT_TORSION_PI_INDEX_THRESHOLD
        for value in pi_indices
    ):
        return AnharmonicModelAssignment(
            family="torsion",
            anharmonic_class=ANHARMONIC_CLASS_NORMAL,
            zeroth_order_model=ZEROTH_ORDER_NORMAL,
            cross_coupling_policy=CROSS_COUPLING_NORMAL,
            reason="HIGH_BOND_ORDER_TORSION",
        )
    return AnharmonicModelAssignment(
        family="torsion",
        anharmonic_class=ANHARMONIC_CLASS_PENDING,
        zeroth_order_model=ZEROTH_ORDER_PENDING,
        cross_coupling_policy=CROSS_COUPLING_PENDING,
        reason="TORSION_BOND_ORDER_UNKNOWN",
    )


def _ring_torsional_assignment(
    family: str,
    text: str,
    context: LargeAmplitudeTopologyContext | None,
    contrast_tolerance: float = 0.50,
) -> AnharmonicModelAssignment:
    torsions = _torsion_terms(text)
    if context is None or not torsions:
        return _local_assignment(family, "RING_MODE")
    central_bonds = tuple(dict.fromkeys(_pair_key(term[1], term[2]) for term in torsions))
    bond_orders = [context.bond_order(*bond) for bond in central_bonds]
    finite = [float(order) for order in bond_orders if order is not None and order > 1.0e-12]
    if (
        len(finite) == len(bond_orders)
        and len(finite) > 1
        and min(finite) > 0.0
        and max(finite) / min(finite) > 1.0 + float(contrast_tolerance)
    ):
        return _local_assignment(family, "RING_MODE_HIGH_BOND_ORDER_STIFFENED")
    return _local_assignment(family, "RING_MODE")


def _block_anharmonic_assignment(family: str) -> AnharmonicModelAssignment:
    if family == "local_xh_stretch":
        return AnharmonicModelAssignment(
            family=family,
            anharmonic_class=ANHARMONIC_CLASS_XH_LOCAL,
            zeroth_order_model=ZEROTH_ORDER_XH_LOCAL,
            cross_coupling_policy=CROSS_COUPLING_LOCAL,
            reason="XH_LOCAL_GEMINAL_BLOCK",
        )
    if family in {"torsion", "ring_puckering", "ring_bend", "butterfly", "oop", "special"}:
        return _local_assignment(family, "CHEMICAL_LOCAL_COORDINATE_FAMILY")
    if family == "mixed_large_amplitude":
        return AnharmonicModelAssignment(
            family=family,
            anharmonic_class=ANHARMONIC_CLASS_PENDING,
            zeroth_order_model=ZEROTH_ORDER_PENDING,
            cross_coupling_policy=CROSS_COUPLING_LOCAL,
            reason="MIXED_LARGE_AMPLITUDE_BLOCK",
        )
    return AnharmonicModelAssignment(
        family=family,
        anharmonic_class=ANHARMONIC_CLASS_NORMAL,
        zeroth_order_model=ZEROTH_ORDER_NORMAL,
        cross_coupling_policy=CROSS_COUPLING_NORMAL,
        reason="DEFAULT_NORMAL_MODE",
    )


def _torsion_periodicity(
    context: LargeAmplitudeTopologyContext,
    central_bond: tuple[int, int],
) -> int:
    left, right = central_bond
    graph = context.graph()
    left_period = _substituent_equivalence_period(
        sorted(atom for atom in graph.get(left, set()) if atom != right),
        context,
    )
    right_period = _substituent_equivalence_period(
        sorted(atom for atom in graph.get(right, set()) if atom != left),
        context,
    )
    return max(1, math.lcm(max(1, left_period), max(1, right_period)))


def _substituent_equivalence_period(
    atoms: list[int],
    context: LargeAmplitudeTopologyContext,
) -> int:
    if not atoms:
        return 1
    clusters: list[list[int]] = []
    for atom in atoms:
        for cluster in clusters:
            if _substituents_equivalent_by_zeff(atom, cluster[0], context):
                cluster.append(atom)
                break
        else:
            clusters.append([atom])
    return max((len(cluster) for cluster in clusters), default=1)


def _substituents_equivalent_by_zeff(
    left: int,
    right: int,
    context: LargeAmplitudeTopologyContext,
) -> bool:
    z_left = _atomic_number(left, context)
    z_right = _atomic_number(right, context)
    if z_left is not None and z_right is not None and z_left != z_right:
        return False
    zeff_left = context.synthon_zeff_value(left)
    zeff_right = context.synthon_zeff_value(right)
    if zeff_left is not None and zeff_right is not None:
        tolerance = max(float(context.synthon_zeff_tolerance), 0.0)
        return abs(zeff_left - zeff_right) <= tolerance
    return context.synthon_signature(left) == context.synthon_signature(right)


def _atomic_number(
    atom: int,
    context: LargeAmplitudeTopologyContext,
) -> int | None:
    if 1 <= atom <= len(context.atomic_numbers):
        return int(context.atomic_numbers[atom - 1])
    signature = context.synthon_signature(atom)
    if signature:
        try:
            return int(signature[0])
        except (TypeError, ValueError):
            return None
    return None


def _torsion_minimum_rad(
    context: LargeAmplitudeTopologyContext,
    atoms: tuple[int, int, int, int],
) -> float | None:
    coords = np.asarray(context.coordinates_angstrom, dtype=float)
    if coords.shape != (len(context.atomic_numbers), 3):
        return None
    if any(atom < 1 or atom > len(context.atomic_numbers) for atom in atoms):
        return None
    p0, p1, p2, p3 = (coords[atom - 1] for atom in atoms)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = float(np.linalg.norm(b1))
    if norm <= 1.0e-12:
        return None
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v_norm = float(np.linalg.norm(v))
    w_norm = float(np.linalg.norm(w))
    if v_norm <= 1.0e-12 or w_norm <= 1.0e-12:
        return None
    v = v / v_norm
    w = w / w_norm
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return math.atan2(y, x)


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (int(left), int(right)) if int(left) < int(right) else (int(right), int(left))


def _padded(values: tuple[str, ...], size: int, *, fill: str = "") -> tuple[str, ...]:
    if len(values) >= size:
        return tuple(values[:size])
    return tuple(values) + tuple(fill for _ in range(size - len(values)))


def _primitive_source_labels(
    *,
    primitive_labels: tuple[str, ...],
    u_matrix: np.ndarray | None,
    ncoord: int,
) -> tuple[str, ...]:
    if not primitive_labels or u_matrix is None:
        return tuple("" for _ in range(ncoord))
    u_values = np.asarray(u_matrix, dtype=float)
    if u_values.ndim != 2 or u_values.shape[1] < ncoord:
        return tuple("" for _ in range(ncoord))
    labels = tuple(str(item) for item in primitive_labels)
    if len(labels) != u_values.shape[0]:
        return tuple("" for _ in range(ncoord))
    sources: list[str] = []
    for column in range(ncoord):
        terms = [
            labels[row]
            for row, coeff in enumerate(u_values[:, column])
            if abs(float(coeff)) > 1.0e-8
        ]
        sources.append(" ".join(terms))
    return tuple(sources)
