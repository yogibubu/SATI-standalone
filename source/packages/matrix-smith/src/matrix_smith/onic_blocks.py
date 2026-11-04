"""Frozen typed-block contracts for composite ONIC coordinate charts.

The records in this module describe ownership, ordering and representation of
coordinate blocks.  They deliberately do not evaluate coordinates: ordinary
natural internals remain owned by the frozen GIC contract, while numerical
realizations are supplied by the dedicated SMITH builders in later gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from matrix_core import read_sectioned_lines, replace_section, section_content

from .payload_codec import (
    BASE64_CANONICAL_JSON_ENCODING,
    decode_canonical_json_lines,
    encode_canonical_json_lines,
    is_payload_subsection_header,
)


ONIC_TYPED_BLOCKS_SCHEMA = "matrix.smith.onic_typed_blocks.v1"
ONIC_TYPED_BLOCKS_SECTION = "ONIC_BLOCKS"
ONIC_BLOCK_KINDS = frozenset({"SUBSTRATE", "MOLECULE_INTERNAL", "RELATIVE_POSE"})
ONIC_BLOCK_REPRESENTATIONS = frozenset(
    {
        "SYMMETRY_ADAPTED_CARTESIAN",
        "INVERSE_DISTANCE_PROJECTOR",
        "NATURAL_INTERNAL",
        "EXPONENTIAL_MAP",
        "PSEUDO_BOND_CONTACT",
    }
)
ONIC_BLOCK_LINEARITIES = frozenset({"MONATOMIC", "LINEAR", "NONLINEAR", "NOT_APPLICABLE"})
ONIC_BLOCK_DERIVATIVE_STATUSES = frozenset({"ANALYTIC_FIRST_ORDER", "DECLARED", "UNAVAILABLE"})
ONIC_BLOCK_SECOND_DERIVATIVE_STATUSES = frozenset(
    {"GENERAL_SPARSE_B_PRIME", "DEFERRED_B_PRIME", "UNAVAILABLE"}
)
ONIC_MATRIX_STORAGE = frozenset(
    {"DENSE", "SPARSE_COO", "IDENTITY", "IMPLICIT_FROM_COEFFICIENTS", "DELEGATED"}
)
ONIC_ORIENTATIONS = frozenset({"TONIC", "CONIC", "SONIC"})
ONIC_GLOBAL_AUDIT_STATUSES = frozenset({"PENDING", "PASS", "FAIL"})

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class OnicBlockContractError(ValueError):
    """Raised when a typed ONIC block contract is incomplete or inconsistent."""


def _identifier(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not _IDENTIFIER_PATTERN.fullmatch(text):
        raise OnicBlockContractError(f"{field} must be a stable non-empty identifier")
    return text


def _normalized_choice(value: str, choices: frozenset[str], *, field: str) -> str:
    text = str(value).strip().upper().replace("-", "_")
    if text not in choices:
        raise OnicBlockContractError(f"unsupported {field}: {value}")
    return text


def _finite(value: float, *, field: str, nonnegative: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise OnicBlockContractError(f"{field} must be {qualifier}")
    return number


def _optional_nonnegative(value: float | None, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field=field, nonnegative=True)


def _sha256(value: str, *, field: str, allow_empty: bool = False) -> str:
    text = str(value).strip().lower()
    if allow_empty and not text:
        return ""
    if not _SHA256_PATTERN.fullmatch(text):
        raise OnicBlockContractError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _coordinate_rows(
    rows: Sequence[Sequence[float]],
    *,
    expected: int | None,
    field: str,
) -> tuple[tuple[float, float, float], ...]:
    normalized: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows):
        values = tuple(float(item) for item in row)
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise OnicBlockContractError(f"{field}[{index}] must be a finite three-vector")
        normalized.append((values[0], values[1], values[2]))
    if expected is not None and len(normalized) != expected:
        raise OnicBlockContractError(f"{field} must contain {expected} coordinate rows")
    return tuple(normalized)


def onic_reference_fingerprint(
    atom_indices_one_based: Sequence[int],
    coordinates_angstrom: Sequence[Sequence[float]],
) -> str:
    """Return the canonical reference fingerprint used by block contracts."""

    atoms = tuple(int(item) for item in atom_indices_one_based)
    rows = _coordinate_rows(
        coordinates_angstrom,
        expected=len(atoms),
        field="reference_coordinates_angstrom",
    )
    payload = {
        "atom_indices_one_based": list(atoms),
        "coordinates_angstrom": [[float(value).hex() for value in row] for row in rows],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OnicSparseMatrixEntry:
    row: int
    column: int
    value: float

    def __post_init__(self) -> None:
        if int(self.row) < 0 or int(self.column) < 0:
            raise OnicBlockContractError("sparse matrix indexes must be zero based and nonnegative")
        object.__setattr__(self, "row", int(self.row))
        object.__setattr__(self, "column", int(self.column))
        object.__setattr__(self, "value", _finite(self.value, field="sparse matrix value"))


@dataclass(frozen=True)
class OnicMatrixRecord:
    """Serializable dense, sparse or delegated linear operator."""

    rows: int
    columns: int
    storage: str
    dense_rows: tuple[tuple[float, ...], ...] = ()
    sparse_entries: tuple[OnicSparseMatrixEntry, ...] = ()
    reference: str = ""

    def __post_init__(self) -> None:
        rows = int(self.rows)
        columns = int(self.columns)
        if rows < 0 or columns < 0:
            raise OnicBlockContractError("matrix dimensions must be nonnegative")
        storage = _normalized_choice(self.storage, ONIC_MATRIX_STORAGE, field="matrix storage")
        dense = tuple(tuple(float(value) for value in row) for row in self.dense_rows)
        sparse = tuple(self.sparse_entries)
        if any(not math.isfinite(value) for row in dense for value in row):
            raise OnicBlockContractError("dense matrix entries must be finite")
        if storage == "DENSE":
            if len(dense) != rows or any(len(row) != columns for row in dense):
                raise OnicBlockContractError("dense matrix payload does not match its shape")
            if sparse or self.reference:
                raise OnicBlockContractError("dense matrix cannot carry sparse or delegated data")
        elif storage == "SPARSE_COO":
            if dense or self.reference:
                raise OnicBlockContractError("sparse matrix cannot carry dense or delegated data")
            keys: set[tuple[int, int]] = set()
            for entry in sparse:
                if entry.row >= rows or entry.column >= columns:
                    raise OnicBlockContractError("sparse matrix entry lies outside its shape")
                key = (entry.row, entry.column)
                if key in keys:
                    raise OnicBlockContractError("sparse matrix entries must have unique indexes")
                keys.add(key)
            sparse = tuple(sorted(sparse, key=lambda item: (item.row, item.column)))
        elif storage == "IDENTITY":
            if rows != columns:
                raise OnicBlockContractError("identity matrix must be square")
            if dense or sparse or self.reference:
                raise OnicBlockContractError("identity matrix cannot carry an explicit payload")
        else:
            if dense or sparse:
                raise OnicBlockContractError(f"{storage} matrix cannot carry explicit entries")
            _identifier(self.reference, field="matrix reference")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "dense_rows", dense)
        object.__setattr__(self, "sparse_entries", sparse)

    def to_dense(self) -> Any:
        """Materialize an explicitly stored matrix for runtime consumers."""

        import numpy as np

        if self.storage == "DENSE":
            return np.asarray(self.dense_rows, dtype=float)
        if self.storage == "SPARSE_COO":
            matrix = np.zeros((self.rows, self.columns), dtype=float)
            for entry in self.sparse_entries:
                matrix[entry.row, entry.column] = entry.value
            return matrix
        if self.storage == "IDENTITY":
            return np.eye(self.rows, dtype=float)
        raise OnicBlockContractError(
            f"matrix storage {self.storage} has no explicit materializable payload"
        )


@dataclass(frozen=True)
class OnicSymmetryOperation:
    label: str
    matrix: OnicMatrixRecord

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label:
            raise OnicBlockContractError("symmetry operation label must be non-empty")
        if self.matrix.rows != self.matrix.columns:
            raise OnicBlockContractError("symmetry representation matrix must be square")
        object.__setattr__(self, "label", label)


@dataclass(frozen=True)
class OnicDegeneracyGroup:
    identifier: str
    irrep: str
    coordinate_identifiers: tuple[str, ...]
    component_gauge: str
    projector: OnicMatrixRecord
    representation_matrices: tuple[OnicSymmetryOperation, ...] = ()

    def __post_init__(self) -> None:
        identifier = _identifier(self.identifier, field="degeneracy-group identifier")
        irrep = str(self.irrep).strip()
        coordinates = tuple(
            _identifier(item, field="degeneracy-group coordinate")
            for item in self.coordinate_identifiers
        )
        if not irrep or not coordinates or len(set(coordinates)) != len(coordinates):
            raise OnicBlockContractError(
                "degeneracy group needs an irrep and unique coordinate identifiers"
            )
        gauge = str(self.component_gauge).strip()
        if not gauge:
            raise OnicBlockContractError("degeneracy group must declare its component gauge")
        operations = tuple(self.representation_matrices)
        for operation in operations:
            if operation.matrix.rows != len(coordinates):
                raise OnicBlockContractError(
                    "symmetry representation dimension must match the degeneracy group"
                )
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "irrep", irrep)
        object.__setattr__(self, "coordinate_identifiers", coordinates)
        object.__setattr__(self, "component_gauge", gauge)
        object.__setattr__(self, "representation_matrices", operations)


@dataclass(frozen=True)
class OnicBlockDiagnostics:
    spectrum: tuple[float, ...] = ()
    condition_number: float | None = None
    projector_symmetry_residual: float | None = None
    projector_idempotency_residual: float | None = None
    row_space_residual: float | None = None
    covariance_residual: float | None = None
    validity_radius: float | None = None
    singularity_threshold: float | None = None
    chirality_policy: str = "NOT_APPLICABLE"
    messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        spectrum = tuple(
            _finite(value, field="diagnostic spectrum", nonnegative=True) for value in self.spectrum
        )
        object.__setattr__(self, "spectrum", spectrum)
        for field in (
            "condition_number",
            "projector_symmetry_residual",
            "projector_idempotency_residual",
            "row_space_residual",
            "covariance_residual",
            "validity_radius",
            "singularity_threshold",
        ):
            object.__setattr__(
                self,
                field,
                _optional_nonnegative(getattr(self, field), field=field),
            )
        chirality = str(self.chirality_policy).strip().upper()
        if not chirality:
            raise OnicBlockContractError("chirality policy must be explicit")
        object.__setattr__(self, "chirality_policy", chirality)
        object.__setattr__(self, "messages", tuple(str(item) for item in self.messages))


@dataclass(frozen=True)
class OnicSiteFrame:
    """Right-handed Cartesian gauge anchored to a declared structural site.

    ``axes_global`` stores the global Cartesian components of the local
    ``x``, ``y`` and ``z`` unit vectors, in that order.
    """

    anchor_atom_indices_one_based: tuple[int, ...]
    origin_angstrom: tuple[float, float, float]
    axes_global: tuple[tuple[float, float, float], ...]
    policy: str
    orientation_sign_policy: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        anchors = tuple(int(item) for item in self.anchor_atom_indices_one_based)
        if any(item < 1 for item in anchors) or len(set(anchors)) != len(anchors):
            raise OnicBlockContractError("site-frame anchors must be unique and one based")
        origin_rows = _coordinate_rows(
            (self.origin_angstrom,),
            expected=1,
            field="site-frame origin",
        )
        axes = _coordinate_rows(self.axes_global, expected=3, field="site-frame axes")
        for left in range(3):
            for right in range(3):
                dot = sum(axes[left][axis] * axes[right][axis] for axis in range(3))
                expected = 1.0 if left == right else 0.0
                if abs(dot - expected) > 1.0e-10:
                    raise OnicBlockContractError("site-frame axes must be orthonormal")
        determinant = (
            axes[0][0] * (axes[1][1] * axes[2][2] - axes[1][2] * axes[2][1])
            - axes[0][1] * (axes[1][0] * axes[2][2] - axes[1][2] * axes[2][0])
            + axes[0][2] * (axes[1][0] * axes[2][1] - axes[1][1] * axes[2][0])
        )
        if abs(determinant - 1.0) > 1.0e-10:
            raise OnicBlockContractError("site-frame axes must form a right-handed frame")
        policy = str(self.policy).strip().upper()
        sign_policy = str(self.orientation_sign_policy).strip().upper()
        if not policy or not sign_policy:
            raise OnicBlockContractError("site-frame gauge and sign policies must be explicit")
        object.__setattr__(self, "anchor_atom_indices_one_based", anchors)
        object.__setattr__(self, "origin_angstrom", origin_rows[0])
        object.__setattr__(self, "axes_global", axes)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "orientation_sign_policy", sign_policy)
        object.__setattr__(self, "provenance", tuple(str(item) for item in self.provenance))


_REPRESENTATIONS_BY_KIND = {
    "SUBSTRATE": frozenset(
        {"SYMMETRY_ADAPTED_CARTESIAN", "INVERSE_DISTANCE_PROJECTOR", "NATURAL_INTERNAL"}
    ),
    "MOLECULE_INTERNAL": frozenset({"NATURAL_INTERNAL"}),
    "RELATIVE_POSE": frozenset({"EXPONENTIAL_MAP", "PSEUDO_BOND_CONTACT"}),
}


@dataclass(frozen=True)
class OnicCoordinateBlock:
    """One owned coordinate block in an ordered composite ONIC chart."""

    identifier: str
    kind: str
    representation: str
    atom_indices_one_based: tuple[int, ...]
    atom_indices_zero_based: tuple[int, ...]
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    reference_fingerprint_sha256: str
    source_family_identifiers: tuple[str, ...]
    source_order: tuple[str, ...]
    coordinate_identifiers: tuple[str, ...]
    target_rank: int
    source_count: int
    nullity: int
    linearity: str
    rank_method: str
    rank_absolute_tolerance: float
    rank_relative_tolerance: float
    coefficient_operator: OnicMatrixRecord
    local_symmetry_provenance: str
    exact_retained_group: str
    irrep_labels: tuple[str, ...]
    degeneracy_groups: tuple[OnicDegeneracyGroup, ...]
    component_gauge: str
    unit: str
    scaling_policy: str
    scale_factors: tuple[float, ...]
    protected: bool
    active: bool
    observable: bool
    analytic_derivative_status: str
    second_derivative_status: str
    diagnostics: OnicBlockDiagnostics
    site_frame: OnicSiteFrame | None = None
    payload_schema: str = ""
    payload_identity_sha256: str = ""
    reference_block_id: str = ""
    moving_block_id: str = ""
    provenance: tuple[str, ...] = ()
    schema: str = ONIC_TYPED_BLOCKS_SCHEMA

    def __post_init__(self) -> None:
        identifier = _identifier(self.identifier, field="block identifier")
        kind = _normalized_choice(self.kind, ONIC_BLOCK_KINDS, field="block kind")
        representation = _normalized_choice(
            self.representation,
            ONIC_BLOCK_REPRESENTATIONS,
            field="block representation",
        )
        if representation not in _REPRESENTATIONS_BY_KIND[kind]:
            raise OnicBlockContractError(f"{representation} is not valid for a {kind} block")
        atoms_one = tuple(int(item) for item in self.atom_indices_one_based)
        atoms_zero = tuple(int(item) for item in self.atom_indices_zero_based)
        if (
            not atoms_one
            or any(item < 1 for item in atoms_one)
            or len(set(atoms_one)) != len(atoms_one)
        ):
            raise OnicBlockContractError("block atom indexes must be unique and one based")
        if atoms_zero != tuple(item - 1 for item in atoms_one):
            raise OnicBlockContractError(
                "zero-based atom indexes must exactly match the serialized one-based indexes"
            )
        coordinates = _coordinate_rows(
            self.reference_coordinates_angstrom,
            expected=len(atoms_one),
            field="reference_coordinates_angstrom",
        )
        expected_fingerprint = onic_reference_fingerprint(atoms_one, coordinates)
        fingerprint = str(self.reference_fingerprint_sha256).strip().lower()
        if (
            fingerprint
            and _sha256(fingerprint, field="reference fingerprint") != expected_fingerprint
        ):
            raise OnicBlockContractError("block reference fingerprint does not match its geometry")
        target_rank = int(self.target_rank)
        source_count = int(self.source_count)
        nullity = int(self.nullity)
        if target_rank < 0 or source_count < target_rank or nullity != source_count - target_rank:
            raise OnicBlockContractError("block rank, source count and nullity are inconsistent")
        sources = tuple(_identifier(item, field="source identifier") for item in self.source_order)
        coordinates_ids = tuple(
            _identifier(item, field="coordinate identifier") for item in self.coordinate_identifiers
        )
        if len(sources) != source_count or len(set(sources)) != len(sources):
            raise OnicBlockContractError("source order must contain one unique id per source")
        if len(coordinates_ids) != target_rank or len(set(coordinates_ids)) != len(coordinates_ids):
            raise OnicBlockContractError("coordinate order must contain one unique id per mode")
        if (
            self.coefficient_operator.rows != target_rank
            or self.coefficient_operator.columns != source_count
        ):
            raise OnicBlockContractError("coefficient operator shape must be rank by source count")
        scale_factors = tuple(float(value) for value in self.scale_factors)
        if len(scale_factors) != target_rank or any(
            not math.isfinite(value) or value <= 0.0 for value in scale_factors
        ):
            raise OnicBlockContractError("scale factors must be positive and match the block rank")
        irreps = tuple(str(item).strip() for item in self.irrep_labels)
        if len(irreps) != target_rank or any(not item for item in irreps):
            raise OnicBlockContractError("irrep labels must match the block rank")
        degeneracy = tuple(self.degeneracy_groups)
        grouped = [item for group in degeneracy for item in group.coordinate_identifiers]
        if len(set(grouped)) != len(grouped) or any(
            item not in coordinates_ids for item in grouped
        ):
            raise OnicBlockContractError(
                "degeneracy groups must be disjoint subsets of block modes"
            )
        derivative = _normalized_choice(
            self.analytic_derivative_status,
            ONIC_BLOCK_DERIVATIVE_STATUSES,
            field="analytic derivative status",
        )
        second_derivative = _normalized_choice(
            self.second_derivative_status,
            ONIC_BLOCK_SECOND_DERIVATIVE_STATUSES,
            field="second derivative status",
        )
        payload_schema = str(self.payload_schema).strip()
        payload_identity = _sha256(
            self.payload_identity_sha256,
            field="payload identity",
            allow_empty=True,
        )
        if (payload_schema and not payload_identity) or (payload_identity and not payload_schema):
            raise OnicBlockContractError("payload schema and identity must be declared together")
        if self.coefficient_operator.storage == "DELEGATED" and not payload_schema:
            raise OnicBlockContractError(
                "delegated coefficient operators require a checksummed payload contract"
            )
        reference_block = str(self.reference_block_id).strip()
        moving_block = str(self.moving_block_id).strip()
        if kind == "RELATIVE_POSE":
            _identifier(reference_block, field="relative-pose reference block")
            _identifier(moving_block, field="relative-pose moving block")
            if reference_block == moving_block:
                raise OnicBlockContractError(
                    "relative-pose blocks need distinct reference and moving blocks"
                )
        elif reference_block or moving_block:
            raise OnicBlockContractError("only relative-pose blocks may reference other blocks")
        source_families = tuple(
            _identifier(item, field="source family") for item in self.source_family_identifiers
        )
        if not source_families or len(set(source_families)) != len(source_families):
            raise OnicBlockContractError("block must declare unique source families")
        for field in (
            "rank_method",
            "local_symmetry_provenance",
            "exact_retained_group",
            "component_gauge",
            "unit",
            "scaling_policy",
        ):
            if not str(getattr(self, field)).strip():
                raise OnicBlockContractError(f"{field} must be explicit")
        if self.schema != ONIC_TYPED_BLOCKS_SCHEMA:
            raise OnicBlockContractError(f"unsupported block schema: {self.schema}")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "atom_indices_one_based", atoms_one)
        object.__setattr__(self, "atom_indices_zero_based", atoms_zero)
        object.__setattr__(self, "reference_coordinates_angstrom", coordinates)
        object.__setattr__(self, "reference_fingerprint_sha256", expected_fingerprint)
        object.__setattr__(self, "source_family_identifiers", source_families)
        object.__setattr__(self, "source_order", sources)
        object.__setattr__(self, "coordinate_identifiers", coordinates_ids)
        object.__setattr__(self, "target_rank", target_rank)
        object.__setattr__(self, "source_count", source_count)
        object.__setattr__(self, "nullity", nullity)
        object.__setattr__(
            self,
            "linearity",
            _normalized_choice(self.linearity, ONIC_BLOCK_LINEARITIES, field="block linearity"),
        )
        object.__setattr__(
            self,
            "rank_absolute_tolerance",
            _finite(self.rank_absolute_tolerance, field="absolute rank tolerance"),
        )
        object.__setattr__(
            self,
            "rank_relative_tolerance",
            _finite(self.rank_relative_tolerance, field="relative rank tolerance"),
        )
        if self.rank_absolute_tolerance <= 0.0 or self.rank_relative_tolerance <= 0.0:
            raise OnicBlockContractError("rank tolerances must be positive")
        object.__setattr__(self, "irrep_labels", irreps)
        object.__setattr__(self, "degeneracy_groups", degeneracy)
        object.__setattr__(self, "scale_factors", scale_factors)
        object.__setattr__(self, "protected", bool(self.protected))
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "observable", bool(self.observable))
        object.__setattr__(self, "analytic_derivative_status", derivative)
        object.__setattr__(self, "second_derivative_status", second_derivative)
        object.__setattr__(self, "payload_schema", payload_schema)
        object.__setattr__(self, "payload_identity_sha256", payload_identity)
        object.__setattr__(self, "reference_block_id", reference_block)
        object.__setattr__(self, "moving_block_id", moving_block)
        object.__setattr__(self, "provenance", tuple(str(item) for item in self.provenance))

        if self.diagnostics.spectrum and len(self.diagnostics.spectrum) != target_rank:
            raise OnicBlockContractError("nonzero diagnostic spectrum must match the block rank")
        if representation in {"SYMMETRY_ADAPTED_CARTESIAN", "INVERSE_DISTANCE_PROJECTOR"}:
            if self.coefficient_operator.storage in {
                "DELEGATED",
                "IMPLICIT_FROM_COEFFICIENTS",
            }:
                raise OnicBlockContractError(
                    f"{representation} requires an explicitly serialized coefficient operator"
                )
        if representation == "SYMMETRY_ADAPTED_CARTESIAN":
            # Frozen legacy records already carry the complete coefficient
            # gauge and remain evaluable without rebuilding it.  Newly built
            # records always serialize the explicit site frame.
            if self.site_frame is not None:
                if len(self.site_frame.anchor_atom_indices_one_based) < 3:
                    raise OnicBlockContractError(
                        "new symmetry-adapted Cartesian site frames need at least three anchors"
                    )
                if not set(self.site_frame.anchor_atom_indices_one_based).issubset(atoms_one):
                    raise OnicBlockContractError(
                        "site-frame anchors must belong to the block subset"
                    )
        if representation == "INVERSE_DISTANCE_PROJECTOR":
            if (
                self.diagnostics.validity_radius is None
                or self.diagnostics.singularity_threshold is None
                or self.diagnostics.condition_number is None
                or self.diagnostics.chirality_policy == "NOT_APPLICABLE"
            ):
                raise OnicBlockContractError(
                    "inverse-distance blocks require conditioning, validity, "
                    "singularity and chirality diagnostics"
                )
            if (
                self.site_frame is None
                or len(self.site_frame.anchor_atom_indices_one_based) < 3
                or not set(self.site_frame.anchor_atom_indices_one_based).issubset(atoms_one)
            ):
                raise OnicBlockContractError(
                    "inverse-distance blocks require a three-anchor subset site frame"
                )
        if representation in {"NATURAL_INTERNAL", "EXPONENTIAL_MAP"} and (
            not payload_schema or not payload_identity
        ):
            raise OnicBlockContractError(
                f"{representation} blocks require a checksummed frozen GIC payload"
            )
        grouped_irreps: dict[str, str] = {}
        for group in degeneracy:
            if group.projector.rows != source_count or group.projector.columns != source_count:
                raise OnicBlockContractError(
                    "degeneracy-subspace projector must act on the complete block source space"
                )
            for coordinate_id in group.coordinate_identifiers:
                grouped_irreps[coordinate_id] = group.irrep
        if set(grouped_irreps) != set(coordinates_ids):
            raise OnicBlockContractError(
                "degeneracy groups must cover every block coordinate exactly once"
            )
        irrep_by_coordinate = dict(zip(coordinates_ids, irreps, strict=True))
        if any(irrep_by_coordinate[item] != irrep for item, irrep in grouped_irreps.items()):
            raise OnicBlockContractError(
                "degeneracy-group irreps contradict the ordered block irrep labels"
            )


@dataclass(frozen=True)
class OnicGlobalAudit:
    status: str
    cartesian_dimension: int
    external_mode_count: int
    target_rank: int
    evaluated_rank: int | None = None
    nullity: int | None = None
    covariance_residual: float | None = None
    condition_number: float | None = None
    rigid_mode_policy: str = "REMOVED_EXACTLY_ONCE"
    messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        status = _normalized_choice(
            self.status,
            ONIC_GLOBAL_AUDIT_STATUSES,
            field="global audit status",
        )
        dimensions = tuple(
            int(value)
            for value in (self.cartesian_dimension, self.external_mode_count, self.target_rank)
        )
        if any(value < 0 for value in dimensions):
            raise OnicBlockContractError("global audit dimensions must be nonnegative")
        evaluated = None if self.evaluated_rank is None else int(self.evaluated_rank)
        nullity = None if self.nullity is None else int(self.nullity)
        if evaluated is not None and (evaluated < 0 or evaluated > self.target_rank):
            raise OnicBlockContractError("evaluated rank is outside the declared target")
        if nullity is not None and nullity < 0:
            raise OnicBlockContractError("global nullity must be nonnegative")
        if status == "PASS" and evaluated != self.target_rank:
            raise OnicBlockContractError("a passing global audit must attain the target rank")
        if status == "PASS" and nullity != dimensions[0] - dimensions[2]:
            raise OnicBlockContractError("a passing global audit must report the Cartesian nullity")
        if dimensions[1] not in {0, 3, 5, 6}:
            raise OnicBlockContractError("external-mode count must be 0, 3, 5, or 6")
        rigid_policy = str(self.rigid_mode_policy).strip().upper()
        if rigid_policy != "REMOVED_EXACTLY_ONCE":
            raise OnicBlockContractError("overall rigid modes must be removed exactly once")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "cartesian_dimension", dimensions[0])
        object.__setattr__(self, "external_mode_count", dimensions[1])
        object.__setattr__(self, "target_rank", dimensions[2])
        object.__setattr__(self, "evaluated_rank", evaluated)
        object.__setattr__(self, "nullity", nullity)
        object.__setattr__(
            self,
            "covariance_residual",
            _optional_nonnegative(self.covariance_residual, field="global covariance residual"),
        )
        object.__setattr__(
            self,
            "condition_number",
            _optional_nonnegative(self.condition_number, field="global condition number"),
        )
        object.__setattr__(self, "rigid_mode_policy", rigid_policy)
        object.__setattr__(self, "messages", tuple(str(item) for item in self.messages))


@dataclass(frozen=True)
class CompositeOnicDefinition:
    """Ordered direct sum of owned coordinate blocks and its global audit."""

    orientation: str
    atom_indices_one_based: tuple[int, ...]
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    reference_fingerprint_sha256: str
    blocks: tuple[OnicCoordinateBlock, ...]
    global_audit: OnicGlobalAudit
    provenance: tuple[str, ...] = ()
    schema: str = ONIC_TYPED_BLOCKS_SCHEMA

    def __post_init__(self) -> None:
        orientation = _normalized_choice(
            self.orientation, ONIC_ORIENTATIONS, field="ONIC orientation"
        )
        atoms = tuple(int(item) for item in self.atom_indices_one_based)
        if not atoms or any(item < 1 for item in atoms) or len(set(atoms)) != len(atoms):
            raise OnicBlockContractError("composite atom indexes must be unique and one based")
        coordinates = _coordinate_rows(
            self.reference_coordinates_angstrom,
            expected=len(atoms),
            field="composite reference coordinates",
        )
        expected_fingerprint = onic_reference_fingerprint(atoms, coordinates)
        fingerprint = str(self.reference_fingerprint_sha256).strip().lower()
        if (
            fingerprint
            and _sha256(fingerprint, field="composite reference fingerprint")
            != expected_fingerprint
        ):
            raise OnicBlockContractError(
                "composite reference fingerprint does not match its geometry"
            )
        blocks = tuple(self.blocks)
        identifiers = tuple(block.identifier for block in blocks)
        if not blocks or len(set(identifiers)) != len(identifiers):
            raise OnicBlockContractError("composite contract needs uniquely identified blocks")
        atom_set = set(atoms)
        for block in blocks:
            if not set(block.atom_indices_one_based).issubset(atom_set):
                raise OnicBlockContractError(
                    f"block {block.identifier} references atoms outside the composite"
                )
            expected_block_reference = tuple(
                coordinates[atoms.index(atom)] for atom in block.atom_indices_one_based
            )
            if block.reference_coordinates_angstrom != expected_block_reference:
                raise OnicBlockContractError(
                    f"block {block.identifier} reference geometry is not in the composite frame"
                )
        owned_sets = [
            (block.identifier, set(block.atom_indices_one_based))
            for block in blocks
            if block.kind in {"SUBSTRATE", "MOLECULE_INTERNAL"}
        ]
        for index, (left_id, left_atoms) in enumerate(owned_sets):
            for right_id, right_atoms in owned_sets[index + 1 :]:
                overlap = sorted(left_atoms.intersection(right_atoms))
                if overlap:
                    raise OnicBlockContractError(
                        f"owned blocks {left_id} and {right_id} overlap on atoms {overlap}"
                    )
        by_id = {block.identifier: block for block in blocks}
        for block in blocks:
            if block.kind != "RELATIVE_POSE":
                continue
            if block.reference_block_id not in by_id or block.moving_block_id not in by_id:
                raise OnicBlockContractError(
                    f"relative-pose block {block.identifier} references unknown blocks"
                )
            if (
                by_id[block.reference_block_id].kind == "RELATIVE_POSE"
                or by_id[block.moving_block_id].kind == "RELATIVE_POSE"
            ):
                raise OnicBlockContractError("relative-pose dependencies must be owned atom blocks")
            dependency_atoms = set(by_id[block.reference_block_id].atom_indices_one_based).union(
                by_id[block.moving_block_id].atom_indices_one_based
            )
            if set(block.atom_indices_one_based) != dependency_atoms:
                raise OnicBlockContractError(
                    f"relative-pose block {block.identifier} atoms must equal its dependency union"
                )
        target_rank = sum(block.target_rank for block in blocks)
        if target_rank != self.global_audit.target_rank:
            raise OnicBlockContractError("sum of block ranks does not match the global target rank")
        if self.global_audit.cartesian_dimension != 3 * len(atoms):
            raise OnicBlockContractError(
                "global Cartesian dimension must equal three times atom count"
            )
        if target_rank != (
            self.global_audit.cartesian_dimension - self.global_audit.external_mode_count
        ):
            raise OnicBlockContractError(
                "composite target rank must remove the declared external modes exactly once"
            )
        if self.schema != ONIC_TYPED_BLOCKS_SCHEMA:
            raise OnicBlockContractError(f"unsupported composite schema: {self.schema}")
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "atom_indices_one_based", atoms)
        object.__setattr__(self, "reference_coordinates_angstrom", coordinates)
        object.__setattr__(self, "reference_fingerprint_sha256", expected_fingerprint)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "provenance", tuple(str(item) for item in self.provenance))


@dataclass(frozen=True)
class OnicBlockRequest:
    """User declaration parsed before a frozen typed block has been built."""

    identifier: str
    kind: str
    representation: str
    atom_indices_one_based: tuple[int, ...] = ()
    site_anchor_atom_indices_one_based: tuple[int, ...] = ()
    reference_block_id: str = ""
    moving_block_id: str = ""
    protected: bool = True
    active: bool = True
    observable: bool = False

    def __post_init__(self) -> None:
        identifier = _identifier(self.identifier, field="block-request identifier")
        kind = _normalized_choice(self.kind, ONIC_BLOCK_KINDS, field="block-request kind")
        representation = _normalized_choice(
            self.representation,
            ONIC_BLOCK_REPRESENTATIONS,
            field="block-request representation",
        )
        if representation not in _REPRESENTATIONS_BY_KIND[kind]:
            raise OnicBlockContractError(f"{representation} is not valid for a {kind} request")
        atoms = tuple(int(item) for item in self.atom_indices_one_based)
        if atoms and (any(item < 1 for item in atoms) or len(set(atoms)) != len(atoms)):
            raise OnicBlockContractError("block-request atom indexes must be unique and one based")
        site_anchors = tuple(int(item) for item in self.site_anchor_atom_indices_one_based)
        if site_anchors and (
            any(item < 1 for item in site_anchors) or len(set(site_anchors)) != len(site_anchors)
        ):
            raise OnicBlockContractError("site-anchor indexes must be unique and one based")
        reference = str(self.reference_block_id).strip()
        moving = str(self.moving_block_id).strip()
        if kind == "RELATIVE_POSE":
            _identifier(reference, field="relative-pose reference block")
            _identifier(moving, field="relative-pose moving block")
            if atoms:
                raise OnicBlockContractError(
                    "relative-pose requests derive atoms from their owned blocks"
                )
        elif not atoms:
            raise OnicBlockContractError("owned block requests must declare atoms")
        elif reference or moving:
            raise OnicBlockContractError("only relative-pose requests may reference blocks")
        if representation in {
            "SYMMETRY_ADAPTED_CARTESIAN",
            "INVERSE_DISTANCE_PROJECTOR",
        }:
            if len(site_anchors) < 3:
                raise OnicBlockContractError(
                    f"{representation} requests require at least three site anchors"
                )
            if not set(site_anchors).issubset(atoms):
                raise OnicBlockContractError("site anchors must belong to the requested block")
        elif site_anchors:
            raise OnicBlockContractError(
                "site anchors are valid only for Cartesian or inverse-distance substrate requests"
            )
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "atom_indices_one_based", atoms)
        object.__setattr__(self, "site_anchor_atom_indices_one_based", site_anchors)
        object.__setattr__(self, "reference_block_id", reference)
        object.__setattr__(self, "moving_block_id", moving)
        object.__setattr__(self, "protected", bool(self.protected))
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "observable", bool(self.observable))


def _matrix_from_dict(payload: Mapping[str, Any]) -> OnicMatrixRecord:
    return OnicMatrixRecord(
        rows=int(payload["rows"]),
        columns=int(payload["columns"]),
        storage=str(payload["storage"]),
        dense_rows=tuple(
            tuple(float(value) for value in row) for row in payload.get("dense_rows", ())
        ),
        sparse_entries=tuple(
            OnicSparseMatrixEntry(
                row=int(entry["row"]),
                column=int(entry["column"]),
                value=float(entry["value"]),
            )
            for entry in payload.get("sparse_entries", ())
        ),
        reference=str(payload.get("reference", "")),
    )


def _degeneracy_group_from_dict(payload: Mapping[str, Any]) -> OnicDegeneracyGroup:
    return OnicDegeneracyGroup(
        identifier=str(payload["identifier"]),
        irrep=str(payload["irrep"]),
        coordinate_identifiers=tuple(str(item) for item in payload["coordinate_identifiers"]),
        component_gauge=str(payload["component_gauge"]),
        projector=_matrix_from_dict(payload["projector"]),
        representation_matrices=tuple(
            OnicSymmetryOperation(
                label=str(operation["label"]),
                matrix=_matrix_from_dict(operation["matrix"]),
            )
            for operation in payload.get("representation_matrices", ())
        ),
    )


def _block_from_dict(payload: Mapping[str, Any]) -> OnicCoordinateBlock:
    diagnostics = payload.get("diagnostics", {})
    return OnicCoordinateBlock(
        identifier=str(payload["identifier"]),
        kind=str(payload["kind"]),
        representation=str(payload["representation"]),
        atom_indices_one_based=tuple(int(item) for item in payload["atom_indices_one_based"]),
        atom_indices_zero_based=tuple(int(item) for item in payload["atom_indices_zero_based"]),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row)
            for row in payload["reference_coordinates_angstrom"]
        ),
        reference_fingerprint_sha256=str(payload.get("reference_fingerprint_sha256", "")),
        source_family_identifiers=tuple(str(item) for item in payload["source_family_identifiers"]),
        source_order=tuple(str(item) for item in payload["source_order"]),
        coordinate_identifiers=tuple(str(item) for item in payload["coordinate_identifiers"]),
        target_rank=int(payload["target_rank"]),
        source_count=int(payload["source_count"]),
        nullity=int(payload["nullity"]),
        linearity=str(payload["linearity"]),
        rank_method=str(payload["rank_method"]),
        rank_absolute_tolerance=float(payload["rank_absolute_tolerance"]),
        rank_relative_tolerance=float(payload["rank_relative_tolerance"]),
        coefficient_operator=_matrix_from_dict(payload["coefficient_operator"]),
        local_symmetry_provenance=str(payload["local_symmetry_provenance"]),
        exact_retained_group=str(payload["exact_retained_group"]),
        irrep_labels=tuple(str(item) for item in payload["irrep_labels"]),
        degeneracy_groups=tuple(
            _degeneracy_group_from_dict(item) for item in payload.get("degeneracy_groups", ())
        ),
        component_gauge=str(payload["component_gauge"]),
        unit=str(payload["unit"]),
        scaling_policy=str(payload["scaling_policy"]),
        scale_factors=tuple(float(item) for item in payload["scale_factors"]),
        protected=bool(payload["protected"]),
        active=bool(payload["active"]),
        observable=bool(payload["observable"]),
        analytic_derivative_status=str(payload["analytic_derivative_status"]),
        second_derivative_status=str(payload["second_derivative_status"]),
        diagnostics=OnicBlockDiagnostics(
            spectrum=tuple(float(item) for item in diagnostics.get("spectrum", ())),
            condition_number=diagnostics.get("condition_number"),
            projector_symmetry_residual=diagnostics.get("projector_symmetry_residual"),
            projector_idempotency_residual=diagnostics.get("projector_idempotency_residual"),
            row_space_residual=diagnostics.get("row_space_residual"),
            covariance_residual=diagnostics.get("covariance_residual"),
            validity_radius=diagnostics.get("validity_radius"),
            singularity_threshold=diagnostics.get("singularity_threshold"),
            chirality_policy=str(diagnostics.get("chirality_policy", "NOT_APPLICABLE")),
            messages=tuple(str(item) for item in diagnostics.get("messages", ())),
        ),
        site_frame=(
            None
            if payload.get("site_frame") is None
            else OnicSiteFrame(
                anchor_atom_indices_one_based=tuple(
                    int(item) for item in payload["site_frame"]["anchor_atom_indices_one_based"]
                ),
                origin_angstrom=tuple(
                    float(item) for item in payload["site_frame"]["origin_angstrom"]
                ),
                axes_global=tuple(
                    tuple(float(value) for value in row)
                    for row in payload["site_frame"]["axes_global"]
                ),
                policy=str(payload["site_frame"]["policy"]),
                orientation_sign_policy=str(payload["site_frame"]["orientation_sign_policy"]),
                provenance=tuple(str(item) for item in payload["site_frame"].get("provenance", ())),
            )
        ),
        payload_schema=str(payload.get("payload_schema", "")),
        payload_identity_sha256=str(payload.get("payload_identity_sha256", "")),
        reference_block_id=str(payload.get("reference_block_id", "")),
        moving_block_id=str(payload.get("moving_block_id", "")),
        provenance=tuple(str(item) for item in payload.get("provenance", ())),
        schema=str(payload.get("schema", ONIC_TYPED_BLOCKS_SCHEMA)),
    )


def composite_onic_definition_to_dict(definition: CompositeOnicDefinition) -> dict[str, Any]:
    """Return the canonical JSON-compatible representation of a contract."""

    # ``dataclasses.asdict`` deliberately preserves tuples.  Normalize through
    # JSON here so callers receive the same array representation that is
    # serialized in enriched XYZ and validated by the bundled JSON Schema.
    return json.loads(json.dumps(asdict(definition), allow_nan=False))


def composite_onic_definition_from_dict(payload: Mapping[str, Any]) -> CompositeOnicDefinition:
    """Validate and reconstruct a composite ONIC definition from JSON data."""

    if not isinstance(payload, Mapping):
        raise OnicBlockContractError("typed ONIC payload must be a JSON object")
    try:
        audit = payload["global_audit"]
        return CompositeOnicDefinition(
            orientation=str(payload["orientation"]),
            atom_indices_one_based=tuple(int(item) for item in payload["atom_indices_one_based"]),
            reference_coordinates_angstrom=tuple(
                tuple(float(value) for value in row)
                for row in payload["reference_coordinates_angstrom"]
            ),
            reference_fingerprint_sha256=str(payload.get("reference_fingerprint_sha256", "")),
            blocks=tuple(_block_from_dict(item) for item in payload["blocks"]),
            global_audit=OnicGlobalAudit(
                status=str(audit["status"]),
                cartesian_dimension=int(audit["cartesian_dimension"]),
                external_mode_count=int(audit["external_mode_count"]),
                target_rank=int(audit["target_rank"]),
                evaluated_rank=(
                    None if audit.get("evaluated_rank") is None else int(audit["evaluated_rank"])
                ),
                nullity=None if audit.get("nullity") is None else int(audit["nullity"]),
                covariance_residual=audit.get("covariance_residual"),
                condition_number=audit.get("condition_number"),
                rigid_mode_policy=str(audit.get("rigid_mode_policy", "REMOVED_EXACTLY_ONCE")),
                messages=tuple(str(item) for item in audit.get("messages", ())),
            ),
            provenance=tuple(str(item) for item in payload.get("provenance", ())),
            schema=str(payload.get("schema", ONIC_TYPED_BLOCKS_SCHEMA)),
        )
    except OnicBlockContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise OnicBlockContractError("typed ONIC payload is incomplete or malformed") from exc


def composite_onic_definition_json(definition: CompositeOnicDefinition) -> str:
    return json.dumps(
        composite_onic_definition_to_dict(definition),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def composite_onic_definition_identity_sha256(definition: CompositeOnicDefinition) -> str:
    return hashlib.sha256(composite_onic_definition_json(definition).encode("utf-8")).hexdigest()


def onic_block_contract_section_lines(definition: CompositeOnicDefinition) -> list[str]:
    """Serialize the complete typed contract as canonical JSON in enriched XYZ."""

    payload = composite_onic_definition_json(definition)
    return [
        f"SCHEMA {ONIC_TYPED_BLOCKS_SCHEMA}",
        "STATUS BUILT",
        f"ENCODING {BASE64_CANONICAL_JSON_ENCODING}",
        f"IDENTITY_SHA256 {composite_onic_definition_identity_sha256(definition)}",
        "[COMPOSITE_JSON]",
        *encode_canonical_json_lines(payload),
    ]


def write_onic_block_contract(path: Path, definition: CompositeOnicDefinition) -> None:
    replace_section(
        Path(path),
        ONIC_TYPED_BLOCKS_SECTION,
        onic_block_contract_section_lines(definition),
    )


def read_onic_block_contract_from_xyzin(
    path: Path,
    *,
    required: bool = True,
) -> CompositeOnicDefinition | None:
    """Read a frozen typed contract without changing legacy GIC-only inputs."""

    section = section_content(read_sectioned_lines(Path(path)), ONIC_TYPED_BLOCKS_SECTION)
    if not section:
        if required:
            raise OnicBlockContractError(f"missing #{ONIC_TYPED_BLOCKS_SECTION} section")
        return None
    if section[0].strip() != f"SCHEMA {ONIC_TYPED_BLOCKS_SCHEMA}":
        raise OnicBlockContractError("unsupported typed ONIC block schema")
    if (_section_value(section, "STATUS") or "").upper() != "BUILT":
        raise OnicBlockContractError("typed ONIC block contract is not built")
    payload_lines = _subsection(section, "COMPOSITE_JSON")
    if not payload_lines:
        raise OnicBlockContractError("typed ONIC contract must contain canonical JSON")
    try:
        serialized = decode_canonical_json_lines(
            payload_lines,
            encoding=_section_value(section, "ENCODING") or "",
        )
        payload = json.loads(serialized)
    except (json.JSONDecodeError, ValueError) as exc:
        raise OnicBlockContractError("typed ONIC contract contains invalid JSON") from exc
    definition = composite_onic_definition_from_dict(payload)
    declared = _section_value(section, "IDENTITY_SHA256")
    actual = composite_onic_definition_identity_sha256(definition)
    if not declared or declared.lower() != actual:
        raise OnicBlockContractError("typed ONIC contract identity checksum does not match")
    return definition


def _bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def _atoms_text(atoms: Sequence[int]) -> str:
    values = tuple(int(item) for item in atoms)
    if not values:
        return "NONE"
    ranges: list[str] = []
    start = prior = values[0]
    for value in values[1:]:
        if value == prior + 1:
            prior = value
            continue
        ranges.append(str(start) if start == prior else f"{start}-{prior}")
        start = prior = value
    ranges.append(str(start) if start == prior else f"{start}-{prior}")
    return ",".join(ranges)


def _parse_atoms(text: str) -> tuple[int, ...]:
    atoms: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token or token.upper() == "NONE":
            continue
        if "-" in token:
            left_text, right_text = token.split("-", 1)
            try:
                left, right = int(left_text), int(right_text)
            except ValueError as exc:
                raise OnicBlockContractError(f"invalid atom range: {token}") from exc
            if left < 1 or right < left:
                raise OnicBlockContractError(f"invalid atom range: {token}")
            atoms.extend(range(left, right + 1))
        else:
            try:
                atoms.append(int(token))
            except ValueError as exc:
                raise OnicBlockContractError(f"invalid atom index: {token}") from exc
    if len(set(atoms)) != len(atoms):
        raise OnicBlockContractError("atom ranges contain duplicates")
    return tuple(atoms)


def _request_line(request: OnicBlockRequest) -> str:
    flags = (
        f" PROTECTED={_bool_text(request.protected)}"
        f" ACTIVE={_bool_text(request.active)}"
        f" OBSERVABLE={_bool_text(request.observable)}"
    )
    if request.kind == "RELATIVE_POSE":
        return (
            f"INTERFACE_POSE ID={request.identifier} REFERENCE={request.reference_block_id} "
            f"MOVING={request.moving_block_id} CHART={request.representation}{flags}"
        )
    return (
        f"BLOCK {request.kind} ID={request.identifier} "
        f"ATOMS={_atoms_text(request.atom_indices_one_based)} "
        f"REPRESENTATION={request.representation}"
        + (
            f" SITE_ANCHORS={_atoms_text(request.site_anchor_atom_indices_one_based)}"
            if request.site_anchor_atom_indices_one_based
            else ""
        )
        + flags
    )


def onic_block_plan_section_lines(
    requests: Sequence[OnicBlockRequest],
    *,
    orientation: str = "SONIC",
) -> list[str]:
    normalized_orientation = _normalized_choice(
        orientation,
        ONIC_ORIENTATIONS,
        field="ONIC orientation",
    )
    records = tuple(requests)
    _validate_block_requests(records)
    return [
        f"SCHEMA {ONIC_TYPED_BLOCKS_SCHEMA}",
        "STATUS PLANNED",
        f"ORIENTATION {normalized_orientation}",
        "INDEXING ATOMS=ONE_BASED",
        "[REQUESTS]",
        *(_request_line(item) for item in records),
    ]


def write_onic_block_plan(
    path: Path,
    requests: Sequence[OnicBlockRequest],
    *,
    orientation: str = "SONIC",
) -> None:
    replace_section(
        Path(path),
        ONIC_TYPED_BLOCKS_SECTION,
        onic_block_plan_section_lines(requests, orientation=orientation),
    )


def read_onic_block_requests_from_xyzin(
    path: Path,
) -> tuple[str, tuple[OnicBlockRequest, ...]]:
    section = section_content(read_sectioned_lines(Path(path)), ONIC_TYPED_BLOCKS_SECTION)
    if not section:
        raise OnicBlockContractError(f"missing #{ONIC_TYPED_BLOCKS_SECTION} section")
    if section[0].strip() != f"SCHEMA {ONIC_TYPED_BLOCKS_SCHEMA}":
        raise OnicBlockContractError("unsupported typed ONIC block schema")
    if (_section_value(section, "STATUS") or "").upper() != "PLANNED":
        raise OnicBlockContractError("typed ONIC block request section is not planned")
    orientation = _normalized_choice(
        _section_value(section, "ORIENTATION") or "SONIC",
        ONIC_ORIENTATIONS,
        field="ONIC orientation",
    )
    requests: list[OnicBlockRequest] = []
    for line in _subsection(section, "REQUESTS"):
        parts = line.split()
        if not parts:
            continue
        keyword = parts[0].upper()
        if keyword == "BLOCK":
            if len(parts) < 2:
                raise OnicBlockContractError(f"invalid block request: {line}")
            kind = parts[1]
            fields = _key_values(parts[2:])
            try:
                requests.append(
                    OnicBlockRequest(
                        identifier=fields["ID"],
                        kind=kind,
                        representation=fields["REPRESENTATION"],
                        atom_indices_one_based=_parse_atoms(fields["ATOMS"]),
                        site_anchor_atom_indices_one_based=(
                            _parse_atoms(fields["SITE_ANCHORS"]) if "SITE_ANCHORS" in fields else ()
                        ),
                        protected=_parse_bool(fields.get("PROTECTED", "TRUE")),
                        active=_parse_bool(fields.get("ACTIVE", "TRUE")),
                        observable=_parse_bool(fields.get("OBSERVABLE", "FALSE")),
                    )
                )
            except KeyError as exc:
                raise OnicBlockContractError(f"incomplete block request: {line}") from exc
        elif keyword == "INTERFACE_POSE":
            fields = _key_values(parts[1:])
            try:
                requests.append(
                    OnicBlockRequest(
                        identifier=fields["ID"],
                        kind="RELATIVE_POSE",
                        representation=fields["CHART"],
                        reference_block_id=fields["REFERENCE"],
                        moving_block_id=fields["MOVING"],
                        protected=_parse_bool(fields.get("PROTECTED", "TRUE")),
                        active=_parse_bool(fields.get("ACTIVE", "TRUE")),
                        observable=_parse_bool(fields.get("OBSERVABLE", "FALSE")),
                    )
                )
            except KeyError as exc:
                raise OnicBlockContractError(f"incomplete interface request: {line}") from exc
        else:
            raise OnicBlockContractError(f"unsupported ONIC request line: {line}")
    records = tuple(requests)
    _validate_block_requests(records)
    return orientation, records


def _validate_block_requests(requests: Sequence[OnicBlockRequest]) -> None:
    records = tuple(requests)
    identifiers = tuple(item.identifier for item in records)
    if not records or len(set(identifiers)) != len(identifiers):
        raise OnicBlockContractError("ONIC block plan needs uniquely identified requests")
    by_id = {item.identifier: item for item in records}
    owned = [item for item in records if item.kind != "RELATIVE_POSE"]
    for index, left in enumerate(owned):
        for right in owned[index + 1 :]:
            overlap = sorted(
                set(left.atom_indices_one_based).intersection(right.atom_indices_one_based)
            )
            if overlap:
                raise OnicBlockContractError(
                    f"owned block requests {left.identifier} and {right.identifier} "
                    f"overlap on atoms {overlap}"
                )
    for request in records:
        if request.kind != "RELATIVE_POSE":
            continue
        if request.reference_block_id not in by_id or request.moving_block_id not in by_id:
            raise OnicBlockContractError(
                f"relative-pose request {request.identifier} references unknown blocks"
            )
        if (
            by_id[request.reference_block_id].kind == "RELATIVE_POSE"
            or by_id[request.moving_block_id].kind == "RELATIVE_POSE"
        ):
            raise OnicBlockContractError(
                "relative-pose requests must reference owned substrate or molecular blocks"
            )


def onic_typed_blocks_schema_path() -> Path:
    return Path(__file__).with_name("schemas") / "onic_typed_blocks_v1.schema.json"


def _section_value(section_lines: Sequence[str], key: str) -> str | None:
    wanted = str(key).upper()
    for line in section_lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].upper() == wanted:
            return parts[1].strip()
    return None


def _subsection(section_lines: Sequence[str], name: str) -> list[str]:
    header = f"[{str(name).upper()}]"
    start: int | None = None
    for index, line in enumerate(section_lines):
        if line.strip().upper() == header:
            start = index + 1
            break
    if start is None:
        return []
    end = len(section_lines)
    for index in range(start, len(section_lines)):
        text = section_lines[index].strip()
        if is_payload_subsection_header(text):
            end = index
            break
    return list(section_lines[start:end])


def _key_values(parts: Sequence[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.upper()] = value
    return fields


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().upper()
    if normalized not in {"TRUE", "FALSE"}:
        raise OnicBlockContractError(f"invalid boolean value: {value}")
    return normalized == "TRUE"


__all__ = [
    "ONIC_BLOCK_DERIVATIVE_STATUSES",
    "ONIC_BLOCK_KINDS",
    "ONIC_BLOCK_LINEARITIES",
    "ONIC_BLOCK_REPRESENTATIONS",
    "ONIC_BLOCK_SECOND_DERIVATIVE_STATUSES",
    "ONIC_MATRIX_STORAGE",
    "ONIC_ORIENTATIONS",
    "ONIC_TYPED_BLOCKS_SCHEMA",
    "ONIC_TYPED_BLOCKS_SECTION",
    "CompositeOnicDefinition",
    "OnicBlockContractError",
    "OnicBlockDiagnostics",
    "OnicBlockRequest",
    "OnicCoordinateBlock",
    "OnicDegeneracyGroup",
    "OnicGlobalAudit",
    "OnicMatrixRecord",
    "OnicSiteFrame",
    "OnicSparseMatrixEntry",
    "OnicSymmetryOperation",
    "composite_onic_definition_from_dict",
    "composite_onic_definition_identity_sha256",
    "composite_onic_definition_json",
    "composite_onic_definition_to_dict",
    "onic_block_contract_section_lines",
    "onic_block_plan_section_lines",
    "onic_reference_fingerprint",
    "onic_typed_blocks_schema_path",
    "read_onic_block_contract_from_xyzin",
    "read_onic_block_requests_from_xyzin",
    "write_onic_block_contract",
    "write_onic_block_plan",
]
