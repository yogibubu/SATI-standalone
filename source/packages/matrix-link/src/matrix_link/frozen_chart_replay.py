"""Fail-closed replay of an immutable, externally selected SONIC chart.

This module makes no chemical or coordinate-selection decision.  It only
certifies that a LINK input is byte-for-byte and semantically identical to one
entry in an auditable reference corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from matrix_chem import geometry_array_sha256, read_enriched_xyz
from matrix_smith import (
    read_gic_definition_from_xyzin,
    sonic_definition_identity_sha256,
)


FROZEN_CHART_CORPUS_SCHEMA = "matrix.link.frozen_chart_corpus.v1"
FROZEN_CHART_REPLAY_SCHEMA = "matrix.link.frozen_chart_replay.v1"


class FrozenChartReplayError(RuntimeError):
    """A requested historical chart cannot be certified for exact replay."""


@dataclass(frozen=True)
class FrozenChartReference:
    manifest_path: Path
    manifest_sha256: str
    corpus_id: str
    task_regime: str
    case_id: int
    name: str
    xyzin_path: Path
    xyzin_sha256: str
    sonic_definition_sha256: str
    geometry_sha256: str
    target_rank: int
    gdv_iterations: int
    current_runtime_compatible: bool
    source_campaign: str

    def validate(self, input_xyzin: Path | str) -> "FrozenChartReplayAudit":
        """Certify exact bytes, geometry and SONIC semantics before execution."""

        if not self.current_runtime_compatible:
            raise FrozenChartReplayError(
                f"reference case {self.case_id} is provenance-only under the current runtime"
            )
        target = Path(input_xyzin).expanduser().resolve()
        if not target.is_file():
            raise FrozenChartReplayError(f"frozen chart input does not exist: {target}")
        raw_sha256 = _file_sha256(target)
        if raw_sha256 != self.xyzin_sha256:
            raise FrozenChartReplayError(
                "input bytes differ from the frozen chart reference: "
                f"{raw_sha256} != {self.xyzin_sha256}"
            )
        geometry = read_enriched_xyz(target)
        geometry_sha256 = geometry_array_sha256(
            geometry.atoms,
            geometry.coordinates_angstrom,
        )
        if geometry_sha256 != self.geometry_sha256:
            raise FrozenChartReplayError(
                "ordered Cartesian geometry differs from the frozen reference"
            )
        definition = read_gic_definition_from_xyzin(
            target,
            # This path certifies immutable historical bytes only. Active
            # runtime construction and optimization keep the default complete
            # molecular-irrep gate in every SONIC consumer.
            validate_symmetry_irreps=False,
            allow_legacy_out_of_plane=True,
        )
        definition_sha256 = sonic_definition_identity_sha256(definition)
        if definition_sha256 != self.sonic_definition_sha256:
            raise FrozenChartReplayError(
                "SONIC definition differs from the frozen chart reference"
            )
        if len(definition.gics) != self.target_rank:
            raise FrozenChartReplayError(
                "SONIC rank differs from the frozen chart reference: "
                f"{len(definition.gics)} != {self.target_rank}"
            )
        return FrozenChartReplayAudit(
            corpus_id=self.corpus_id,
            case_id=self.case_id,
            name=self.name,
            task_regime=self.task_regime,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha256,
            input_xyzin=target,
            xyzin_sha256=raw_sha256,
            sonic_definition_sha256=definition_sha256,
            geometry_sha256=geometry_sha256,
            target_rank=self.target_rank,
            gdv_iterations=self.gdv_iterations,
            source_campaign=self.source_campaign,
            lifecycle_policy="DISABLED_IMMUTABLE_REPLAY",
        )


@dataclass(frozen=True)
class FrozenChartReplayAudit:
    corpus_id: str
    case_id: int
    name: str
    task_regime: str
    manifest_path: Path
    manifest_sha256: str
    input_xyzin: Path
    xyzin_sha256: str
    sonic_definition_sha256: str
    geometry_sha256: str
    target_rank: int
    gdv_iterations: int
    source_campaign: str
    lifecycle_policy: str

    def to_json(self) -> dict[str, object]:
        return {
            "schema": FROZEN_CHART_REPLAY_SCHEMA,
            "corpus_id": self.corpus_id,
            "case_id": self.case_id,
            "name": self.name,
            "task_regime": self.task_regime,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "input_xyzin": str(self.input_xyzin),
            "xyzin_sha256": self.xyzin_sha256,
            "sonic_definition_sha256": self.sonic_definition_sha256,
            "geometry_sha256": self.geometry_sha256,
            "target_rank": self.target_rank,
            "gdv_iterations": self.gdv_iterations,
            "source_campaign": self.source_campaign,
            "lifecycle_policy": self.lifecycle_policy,
        }

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def load_frozen_chart_reference(
    manifest_path: Path | str,
    case: int | str,
) -> FrozenChartReference:
    """Load one strictly validated row from a frozen chart corpus."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FrozenChartReplayError(f"frozen chart manifest does not exist: {manifest}")
    raw = manifest.read_bytes()
    header, rows = _parse_manifest(raw)
    selector = str(case).strip()
    matches = [row for row in rows if selector in {row[0], row[1]}]
    if len(matches) != 1:
        raise FrozenChartReplayError(
            f"frozen chart selector {selector!r} matched {len(matches)} cases"
        )
    row = matches[0]
    compatible = _parse_boolean(row[7])
    target = (manifest.parent / "charts" / f"{row[1]}.xyzin").resolve()
    if target.parent != (manifest.parent / "charts").resolve():
        raise FrozenChartReplayError("frozen chart path escapes the corpus directory")
    reference = FrozenChartReference(
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        corpus_id=header["corpus_id"],
        task_regime=header["task_regime"],
        case_id=int(row[0]),
        name=row[1],
        xyzin_path=target,
        xyzin_sha256=_digest(row[2], "xyzin_sha256"),
        sonic_definition_sha256=_digest(row[3], "sonic_definition_sha256"),
        geometry_sha256=_digest(row[4], "geometry_sha256"),
        target_rank=int(row[5]),
        gdv_iterations=int(row[6]),
        current_runtime_compatible=compatible,
        source_campaign=row[8],
    )
    if reference.case_id <= 0 or reference.target_rank <= 0:
        raise FrozenChartReplayError("frozen chart manifest contains a non-positive integer")
    return reference


def _parse_manifest(raw: bytes) -> tuple[dict[str, str], list[list[str]]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise FrozenChartReplayError("frozen chart manifest is not UTF-8") from exc
    header: dict[str, str] = {}
    rows: list[list[str]] = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("# ") and "=" in line:
            key, value = line[2:].split("=", 1)
            header[key.strip()] = value.strip()
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9:
            raise FrozenChartReplayError("frozen chart manifest row must have nine fields")
        rows.append(fields)
    if header.get("schema") != FROZEN_CHART_CORPUS_SCHEMA:
        raise FrozenChartReplayError("unsupported frozen chart corpus schema")
    if not header.get("corpus_id"):
        raise FrozenChartReplayError("frozen chart corpus has no corpus_id")
    if header.get("task_regime") not in {"MINIMUM", "TRANSITION_STATE"}:
        raise FrozenChartReplayError("frozen chart corpus has an invalid task regime")
    identifiers = [row[0] for row in rows]
    names = [row[1] for row in rows]
    if len(set(identifiers)) != len(rows) or len(set(names)) != len(rows):
        raise FrozenChartReplayError("frozen chart corpus contains duplicate cases")
    return header, rows


def _digest(value: str, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(item not in "0123456789abcdef" for item in normalized):
        raise FrozenChartReplayError(f"{field} is not a lowercase SHA-256 digest")
    return normalized


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise FrozenChartReplayError("runtime compatibility must be true or false")
    return normalized == "true"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "FROZEN_CHART_CORPUS_SCHEMA",
    "FROZEN_CHART_REPLAY_SCHEMA",
    "FrozenChartReference",
    "FrozenChartReplayAudit",
    "FrozenChartReplayError",
    "load_frozen_chart_reference",
]
