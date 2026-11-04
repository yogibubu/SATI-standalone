"""ORACLE-owned provenance for Cartesian frame changes.

The certificate distinguishes a harmless proper rigid-frame transformation
from atom reordering, reflection, and a genuine change of molecular geometry.
It contains no topology or chemical perception: atoms are compared in their
declared order and Cartesian invariants are evaluated deterministically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content

from .geometry_alignment import kabsch_rotation


ORACLE_GEOMETRY_IDENTITY_SCHEMA = "oracle.geometry_identity.v1"
ORACLE_GEOMETRY_IDENTITY_SECTION = "ORACLE_GEOMETRY_IDENTITY"
ORACLE_GEOMETRY_IDENTITY_OWNER = "ORACLE"

GEOMETRY_IDENTICAL = "IDENTICAL"
GEOMETRY_RIGID_FRAME_ONLY = "RIGID_FRAME_ONLY"
GEOMETRY_ATOM_ORDER_CHANGE = "ATOM_ORDER_CHANGE"
GEOMETRY_REFLECTION = "REFLECTION"
GEOMETRY_TRUE_CHANGE = "TRUE_GEOMETRY_CHANGE"
GEOMETRY_IDENTITY_RELATIONS = frozenset(
    {
        GEOMETRY_IDENTICAL,
        GEOMETRY_RIGID_FRAME_ONLY,
        GEOMETRY_ATOM_ORDER_CHANGE,
        GEOMETRY_REFLECTION,
        GEOMETRY_TRUE_CHANGE,
    }
)
GEOMETRY_CHANGE_NOT_AUTHORIZED = "NONE"
GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION = "ORACLE_SYMMETRY_PROJECTION"
GEOMETRY_CHANGE_AUTHORIZATIONS = frozenset(
    {
        GEOMETRY_CHANGE_NOT_AUTHORIZED,
        GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION,
    }
)
PAIR_DISTANCE_ERROR_METHODS = frozenset(
    {
        "EXACT_ALL_PAIRS",
        "RIGID_ALIGNMENT_CERTIFIED_UPPER_BOUND",
        "UNDEFINED_SHAPE_MISMATCH",
    }
)


class GeometryIdentityError(ValueError):
    """Raised when Cartesian provenance is missing or contradictory."""


@dataclass(frozen=True)
class GeometryIdentityCertificate:
    schema: str
    owner: str
    relation: str
    source_atoms: tuple[str, ...]
    canonical_atoms: tuple[str, ...]
    source_geometry_sha256: str
    canonical_geometry_sha256: str
    cartesian_tolerance_angstrom: float
    rigid_tolerance_angstrom: float
    raw_max_displacement_angstrom: float
    pair_distance_max_error_angstrom: float
    proper_alignment_max_error_angstrom: float
    orthogonal_alignment_max_error_angstrom: float
    orthogonal_determinant: float
    rotation_source_to_canonical: tuple[tuple[float, float, float], ...]
    translation_source_to_canonical_angstrom: tuple[float, float, float]
    geometry_change_authorization: str = GEOMETRY_CHANGE_NOT_AUTHORIZED
    provenance: str = "ORACLE_LINK_CARTESIAN_PROVENANCE@1"
    pair_distance_error_method: str = "EXACT_ALL_PAIRS"


def geometry_array_sha256(
    atoms: Iterable[str], coordinates_angstrom: np.ndarray
) -> str:
    """Return an order-sensitive, platform-stable Cartesian fingerprint."""

    labels = tuple(str(atom) for atom in atoms)
    coordinates = _coordinates(coordinates_angstrom, len(labels), "geometry")
    canonical = np.round(np.asarray(coordinates, dtype="<f8"), decimals=12)
    canonical[canonical == 0.0] = 0.0
    payload = b"\0".join(atom.encode("utf-8") for atom in labels)
    payload += b"\0" + np.ascontiguousarray(canonical, dtype="<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_geometry_identity_certificate(
    source_atoms: Iterable[str],
    source_coordinates_angstrom: np.ndarray,
    canonical_atoms: Iterable[str],
    canonical_coordinates_angstrom: np.ndarray,
    *,
    cartesian_tolerance_angstrom: float = 1.0e-10,
    rigid_tolerance_angstrom: float = 1.0e-6,
    geometry_change_authorization: str = GEOMETRY_CHANGE_NOT_AUTHORIZED,
) -> GeometryIdentityCertificate:
    """Classify the source-to-canonical Cartesian relationship.

    Only an unchanged atom sequence can be classified as identical or as a
    proper rigid-frame change.  Reflections are detected with unconstrained
    orthogonal Procrustes alignment and are never accepted as rotations.
    """

    source_labels = tuple(str(atom) for atom in source_atoms)
    canonical_labels = tuple(str(atom) for atom in canonical_atoms)
    if not np.isfinite(cartesian_tolerance_angstrom) or cartesian_tolerance_angstrom < 0:
        raise GeometryIdentityError("Cartesian identity tolerance must be non-negative")
    if not np.isfinite(rigid_tolerance_angstrom) or rigid_tolerance_angstrom <= 0:
        raise GeometryIdentityError("rigid identity tolerance must be positive")

    source = _coordinates(source_coordinates_angstrom, len(source_labels), "source")
    canonical = _coordinates(
        canonical_coordinates_angstrom, len(canonical_labels), "canonical"
    )
    same_shape = source.shape == canonical.shape
    if same_shape:
        raw_error = float(np.max(np.abs(source - canonical), initial=0.0))
        proper_rotation, proper_error, translation = _proper_alignment(source, canonical)
        orthogonal_rotation, orthogonal_error = _orthogonal_alignment(source, canonical)
        orthogonal_determinant = float(np.linalg.det(orthogonal_rotation))
        if len(source) >= 256:
            proper_aligned = source @ proper_rotation + translation
            proper_bound = 2.0 * float(
                np.max(np.linalg.norm(proper_aligned - canonical, axis=1), initial=0.0)
            )
            orthogonal_translation = (
                np.mean(canonical, axis=0) - np.mean(source, axis=0) @ orthogonal_rotation
            )
            orthogonal_aligned = source @ orthogonal_rotation + orthogonal_translation
            orthogonal_bound = 2.0 * float(
                np.max(np.linalg.norm(orthogonal_aligned - canonical, axis=1), initial=0.0)
            )
            pair_error = min(proper_bound, orthogonal_bound)
            pair_error_method = "RIGID_ALIGNMENT_CERTIFIED_UPPER_BOUND"
        else:
            pair_error = _pair_distance_error(source, canonical)
            pair_error_method = "EXACT_ALL_PAIRS"
    else:
        # ``-1`` is the serialized sentinel for a metric that is undefined
        # because the two Cartesian arrays do not have the same shape.
        raw_error = pair_error = proper_error = orthogonal_error = -1.0
        proper_rotation = np.eye(3, dtype=float)
        translation = np.zeros(3, dtype=float)
        orthogonal_determinant = 1.0
        pair_error_method = "UNDEFINED_SHAPE_MISMATCH"

    if source_labels != canonical_labels:
        relation = (
            GEOMETRY_ATOM_ORDER_CHANGE
            if len(source_labels) == len(canonical_labels)
            and sorted(source_labels) == sorted(canonical_labels)
            else GEOMETRY_TRUE_CHANGE
        )
    elif raw_error <= cartesian_tolerance_angstrom:
        relation = GEOMETRY_IDENTICAL
    elif proper_error <= rigid_tolerance_angstrom:
        relation = GEOMETRY_RIGID_FRAME_ONLY
    elif (
        orthogonal_determinant < 0.0
        and orthogonal_error <= rigid_tolerance_angstrom
    ):
        relation = GEOMETRY_REFLECTION
    else:
        relation = GEOMETRY_TRUE_CHANGE

    certificate = GeometryIdentityCertificate(
        schema=ORACLE_GEOMETRY_IDENTITY_SCHEMA,
        owner=ORACLE_GEOMETRY_IDENTITY_OWNER,
        relation=relation,
        source_atoms=source_labels,
        canonical_atoms=canonical_labels,
        source_geometry_sha256=geometry_array_sha256(source_labels, source),
        canonical_geometry_sha256=geometry_array_sha256(canonical_labels, canonical),
        cartesian_tolerance_angstrom=float(cartesian_tolerance_angstrom),
        rigid_tolerance_angstrom=float(rigid_tolerance_angstrom),
        raw_max_displacement_angstrom=raw_error,
        pair_distance_max_error_angstrom=pair_error,
        proper_alignment_max_error_angstrom=proper_error,
        orthogonal_alignment_max_error_angstrom=orthogonal_error,
        orthogonal_determinant=orthogonal_determinant,
        rotation_source_to_canonical=tuple(
            tuple(float(value) for value in row) for row in proper_rotation
        ),
        translation_source_to_canonical_angstrom=tuple(float(value) for value in translation),
        geometry_change_authorization=str(geometry_change_authorization),
        pair_distance_error_method=pair_error_method,
    )
    validate_geometry_identity_certificate(certificate)
    return certificate


def validate_geometry_identity_certificate(
    certificate: GeometryIdentityCertificate,
    *,
    canonical_atoms: Iterable[str] | None = None,
    canonical_coordinates_angstrom: np.ndarray | None = None,
    allow_true_geometry_change: bool = True,
) -> None:
    """Validate certificate structure and, when supplied, its frozen geometry."""

    if certificate.schema != ORACLE_GEOMETRY_IDENTITY_SCHEMA:
        raise GeometryIdentityError("unsupported ORACLE geometry-identity schema")
    if certificate.owner != ORACLE_GEOMETRY_IDENTITY_OWNER:
        raise GeometryIdentityError("geometry-identity certificate is not ORACLE-owned")
    if certificate.relation not in GEOMETRY_IDENTITY_RELATIONS:
        raise GeometryIdentityError("unknown source-to-canonical geometry relation")
    if certificate.geometry_change_authorization not in GEOMETRY_CHANGE_AUTHORIZATIONS:
        raise GeometryIdentityError("unknown ORACLE geometry-change authorization")
    if certificate.pair_distance_error_method not in PAIR_DISTANCE_ERROR_METHODS:
        raise GeometryIdentityError("unknown pair-distance error method")
    if (
        certificate.geometry_change_authorization
        != GEOMETRY_CHANGE_NOT_AUTHORIZED
        and certificate.relation != GEOMETRY_TRUE_CHANGE
    ):
        raise GeometryIdentityError(
            "geometry-change authorization is valid only for a true geometry change"
        )
    if certificate.relation == GEOMETRY_TRUE_CHANGE and not allow_true_geometry_change:
        raise GeometryIdentityError("ORACLE canonicalization changed the molecular geometry")
    if len(certificate.source_geometry_sha256) != 64 or len(
        certificate.canonical_geometry_sha256
    ) != 64:
        raise GeometryIdentityError("invalid geometry fingerprint")
    rotation = np.asarray(certificate.rotation_source_to_canonical, dtype=float)
    translation = np.asarray(
        certificate.translation_source_to_canonical_angstrom, dtype=float
    )
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise GeometryIdentityError("invalid source-to-canonical rigid transform")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise GeometryIdentityError("non-finite source-to-canonical rigid transform")
    if canonical_atoms is not None or canonical_coordinates_angstrom is not None:
        if canonical_atoms is None or canonical_coordinates_angstrom is None:
            raise GeometryIdentityError("canonical atoms and coordinates must be supplied together")
        labels = tuple(str(atom) for atom in canonical_atoms)
        if labels != certificate.canonical_atoms:
            raise GeometryIdentityError("canonical atom order contradicts ORACLE provenance")
        digest = geometry_array_sha256(labels, canonical_coordinates_angstrom)
        if digest != certificate.canonical_geometry_sha256:
            raise GeometryIdentityError("canonical Cartesian block contradicts ORACLE provenance")


def geometry_identity_payload_sha256(certificate: GeometryIdentityCertificate) -> str:
    payload = _certificate_json(certificate)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_geometry_identity_certificate(
    path: Path, certificate: GeometryIdentityCertificate
) -> None:
    validate_geometry_identity_certificate(certificate)
    payload = _certificate_json(certificate)
    replace_section(
        Path(path),
        ORACLE_GEOMETRY_IDENTITY_SECTION,
        [
            f"SCHEMA {certificate.schema}",
            f"OWNER {certificate.owner}",
            "ENCODING CANONICAL_JSON_UTF8",
            f"PAYLOAD_SHA256 {hashlib.sha256(payload.encode('utf-8')).hexdigest()}",
            "[PAYLOAD]",
            payload,
        ],
    )


def read_geometry_identity_certificate(path: Path) -> GeometryIdentityCertificate:
    content = section_content(
        read_sectioned_lines(Path(path)), ORACLE_GEOMETRY_IDENTITY_SECTION
    )
    if not content:
        raise GeometryIdentityError(f"missing #{ORACLE_GEOMETRY_IDENTITY_SECTION} section")
    metadata: dict[str, str] = {}
    chunks: list[str] = []
    in_payload = False
    for raw in content:
        text = raw.strip()
        if text == "[PAYLOAD]":
            in_payload = True
        elif in_payload:
            chunks.append(text)
        elif text:
            fields = text.split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
    payload_text = "".join(chunks)
    if metadata.get("SCHEMA") != ORACLE_GEOMETRY_IDENTITY_SCHEMA:
        raise GeometryIdentityError("unsupported serialized geometry-identity schema")
    if metadata.get("OWNER") != ORACLE_GEOMETRY_IDENTITY_OWNER:
        raise GeometryIdentityError("serialized geometry identity is not ORACLE-owned")
    if hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != metadata.get(
        "PAYLOAD_SHA256"
    ):
        raise GeometryIdentityError("geometry-identity payload fingerprint mismatch")
    try:
        data = json.loads(payload_text)
        certificate = GeometryIdentityCertificate(
            schema=str(data["schema"]),
            owner=str(data["owner"]),
            relation=str(data["relation"]),
            source_atoms=tuple(str(value) for value in data["source_atoms"]),
            canonical_atoms=tuple(str(value) for value in data["canonical_atoms"]),
            source_geometry_sha256=str(data["source_geometry_sha256"]),
            canonical_geometry_sha256=str(data["canonical_geometry_sha256"]),
            cartesian_tolerance_angstrom=float(data["cartesian_tolerance_angstrom"]),
            rigid_tolerance_angstrom=float(data["rigid_tolerance_angstrom"]),
            raw_max_displacement_angstrom=float(data["raw_max_displacement_angstrom"]),
            pair_distance_max_error_angstrom=float(
                data["pair_distance_max_error_angstrom"]
            ),
            proper_alignment_max_error_angstrom=float(
                data["proper_alignment_max_error_angstrom"]
            ),
            orthogonal_alignment_max_error_angstrom=float(
                data["orthogonal_alignment_max_error_angstrom"]
            ),
            orthogonal_determinant=float(data["orthogonal_determinant"]),
            rotation_source_to_canonical=tuple(
                tuple(float(value) for value in row)
                for row in data["rotation_source_to_canonical"]
            ),
            translation_source_to_canonical_angstrom=tuple(
                float(value) for value in data["translation_source_to_canonical_angstrom"]
            ),
            geometry_change_authorization=str(
                data.get("geometry_change_authorization", GEOMETRY_CHANGE_NOT_AUTHORIZED)
            ),
            provenance=str(data.get("provenance", "")),
            pair_distance_error_method=str(
                data.get("pair_distance_error_method", "EXACT_ALL_PAIRS")
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GeometryIdentityError("invalid geometry-identity payload") from exc
    validate_geometry_identity_certificate(certificate)
    return certificate


def _certificate_json(certificate: GeometryIdentityCertificate) -> str:
    validate_geometry_identity_certificate(certificate)
    return json.dumps(asdict(certificate), sort_keys=True, separators=(",", ":"))


def _coordinates(values: np.ndarray, natoms: int, name: str) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.shape != (natoms, 3):
        raise GeometryIdentityError(
            f"{name} coordinates must have shape ({natoms}, 3), got {coordinates.shape}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise GeometryIdentityError(f"{name} coordinates must be finite")
    return coordinates


def _pair_distance_error(source: np.ndarray, canonical: np.ndarray) -> float:
    source_distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=-1)
    canonical_distances = np.linalg.norm(
        canonical[:, None, :] - canonical[None, :, :], axis=-1
    )
    return float(np.max(np.abs(source_distances - canonical_distances), initial=0.0))


def _proper_alignment(
    source: np.ndarray, canonical: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    if len(source) < 2:
        rotation = np.eye(3, dtype=float)
    else:
        rotation = kabsch_rotation(source, canonical)
    translation = np.mean(canonical, axis=0) - np.mean(source, axis=0) @ rotation
    aligned = source @ rotation + translation
    error = float(np.max(np.abs(aligned - canonical), initial=0.0))
    return rotation, error, translation


def _orthogonal_alignment(
    source: np.ndarray, canonical: np.ndarray
) -> tuple[np.ndarray, float]:
    if len(source) < 2:
        rotation = np.eye(3, dtype=float)
    else:
        source_centered = source - np.mean(source, axis=0)
        canonical_centered = canonical - np.mean(canonical, axis=0)
        u_matrix, _singular, vt_matrix = np.linalg.svd(
            source_centered.T @ canonical_centered
        )
        rotation = u_matrix @ vt_matrix
    translation = np.mean(canonical, axis=0) - np.mean(source, axis=0) @ rotation
    aligned = source @ rotation + translation
    error = float(np.max(np.abs(aligned - canonical), initial=0.0))
    return rotation, error


__all__ = [
    "GEOMETRY_ATOM_ORDER_CHANGE",
    "GEOMETRY_CHANGE_AUTHORIZATIONS",
    "GEOMETRY_CHANGE_NOT_AUTHORIZED",
    "GEOMETRY_CHANGE_ORACLE_SYMMETRY_PROJECTION",
    "GEOMETRY_IDENTICAL",
    "GEOMETRY_IDENTITY_RELATIONS",
    "GEOMETRY_REFLECTION",
    "GEOMETRY_RIGID_FRAME_ONLY",
    "GEOMETRY_TRUE_CHANGE",
    "GeometryIdentityCertificate",
    "GeometryIdentityError",
    "ORACLE_GEOMETRY_IDENTITY_OWNER",
    "ORACLE_GEOMETRY_IDENTITY_SCHEMA",
    "ORACLE_GEOMETRY_IDENTITY_SECTION",
    "build_geometry_identity_certificate",
    "geometry_array_sha256",
    "geometry_identity_payload_sha256",
    "read_geometry_identity_certificate",
    "validate_geometry_identity_certificate",
    "write_geometry_identity_certificate",
]
