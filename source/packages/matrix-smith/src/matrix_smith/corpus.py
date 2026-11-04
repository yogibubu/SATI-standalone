from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import TypeVar


GIC_CORPUS_ENV = "ORACLE_GIC_CORPUS"
GEOMETRY_IMPORT_SUFFIXES = (".inp", ".gjf", ".gau", ".com", ".xyz", ".zmat", ".zmt")
T = TypeVar("T")

ROLE_BY_SUFFIX = {
    ".inp": "legacy_gic_input",
    ".gjf": "gaussian_input",
    ".gau": "gaussian_input",
    ".fchk": "gaussian_fchk",
    ".log": "qm_output",
    ".out": "qm_output",
    ".msr": "morpheus_state",
    ".sum": "morpheus_summary",
    ".opt": "morpheus_optimized",
    ".form": "morpheus_formatted",
    ".vlt": "legacy_vlt",
    ".gbs": "basis_set",
    "": "unclassified",
}


@dataclass(frozen=True)
class GICCorpusEntry:
    path: Path
    name: str
    stem: str
    suffix: str
    role: str

    def record(self, *, root: Path | None = None) -> dict[str, str]:
        resolved_root = root.resolve() if root is not None else None
        resolved_path = self.path.resolve()
        if resolved_root is not None:
            try:
                display_path = str(resolved_path.relative_to(resolved_root))
            except ValueError:
                display_path = str(resolved_path)
        else:
            display_path = str(self.path)
        return {
            "name": self.name,
            "path": display_path,
            "stem": self.stem,
            "suffix": self.suffix,
            "role": self.role,
        }


@dataclass(frozen=True)
class GICCorpusSummary:
    root: Path
    entries: tuple[GICCorpusEntry, ...]

    @property
    def total_files(self) -> int:
        return len(self.entries)

    @property
    def suffix_counts(self) -> dict[str, int]:
        return dict(Counter(entry.suffix for entry in self.entries))

    @property
    def role_counts(self) -> dict[str, int]:
        return dict(Counter(entry.role for entry in self.entries))


@dataclass(frozen=True)
class GICCorpusGeometryAuditEntry:
    path: Path
    name: str
    suffix: str
    role: str
    status: str
    source_format: str = ""
    natoms: int | None = None
    charge: int | None = None
    multiplicity: int | None = None
    error_type: str = ""
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def record(self, *, root: Path | None = None) -> dict[str, object]:
        resolved_root = root.resolve() if root is not None else None
        resolved_path = self.path.resolve()
        if resolved_root is not None:
            try:
                display_path = str(resolved_path.relative_to(resolved_root))
            except ValueError:
                display_path = str(resolved_path)
        else:
            display_path = str(self.path)
        return {
            "path": display_path,
            "name": self.name,
            "suffix": self.suffix,
            "role": self.role,
            "status": self.status,
            "source_format": self.source_format,
            "natoms": self.natoms,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass(frozen=True)
class GICCorpusGeometryAudit:
    root: Path
    entries: tuple[GICCorpusGeometryAuditEntry, ...]

    @property
    def total_files(self) -> int:
        return len(self.entries)

    @property
    def passed_files(self) -> int:
        return sum(1 for entry in self.entries if entry.passed)

    @property
    def failed_files(self) -> int:
        return self.total_files - self.passed_files

    @property
    def source_format_counts(self) -> dict[str, int]:
        return dict(Counter(entry.source_format for entry in self.entries if entry.source_format))

    @property
    def error_counts(self) -> dict[str, int]:
        return dict(Counter(entry.error_type for entry in self.entries if entry.error_type))


class GICCorpusError(ValueError):
    """Raised when the GIC regression corpus cannot be discovered."""


def default_gic_corpus_root(repo_root: Path | None = None) -> Path:
    env_root = os.environ.get(GIC_CORPUS_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    if repo_root is None:
        repo_root = Path.cwd()
    return Path(repo_root).resolve() / "tests" / "fixtures" / "test_molecules" / "molecules"


def discover_gic_corpus(
    root: Path,
    *,
    suffixes: tuple[str, ...] | list[str] | None = None,
) -> tuple[GICCorpusEntry, ...]:
    target = Path(root).expanduser().resolve()
    if not target.is_dir():
        raise GICCorpusError(f"GIC regression corpus directory not found: {target}")

    requested = _normalize_suffixes(suffixes)
    entries: list[GICCorpusEntry] = []
    for path in sorted(item for item in target.iterdir() if item.is_file()):
        suffix = path.suffix.lower()
        if requested and suffix not in requested:
            continue
        entries.append(
            GICCorpusEntry(
                path=path,
                name=path.name,
                stem=path.stem,
                suffix=suffix,
                role=ROLE_BY_SUFFIX.get(suffix, "unclassified"),
            )
        )
    return tuple(entries)


def summarize_gic_corpus(
    root: Path,
    *,
    suffixes: tuple[str, ...] | list[str] | None = None,
) -> GICCorpusSummary:
    target = Path(root).expanduser().resolve()
    return GICCorpusSummary(root=target, entries=discover_gic_corpus(target, suffixes=suffixes))


def gic_corpus_records(
    summary: GICCorpusSummary,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    entries = _limited(summary.entries, limit)
    return [entry.record(root=summary.root) for entry in entries]


def format_gic_corpus_summary(summary: GICCorpusSummary) -> list[str]:
    lines = [
        f"ROOT {summary.root}",
        f"TOTAL_FILES {summary.total_files}",
    ]
    for suffix, count in sorted(summary.suffix_counts.items()):
        label = suffix or "<none>"
        lines.append(f"SUFFIX {label} {count}")
    for role, count in sorted(summary.role_counts.items()):
        lines.append(f"ROLE {role} {count}")
    return lines


def format_gic_corpus_paths(
    summary: GICCorpusSummary,
    *,
    limit: int | None = None,
) -> list[str]:
    return [str(entry.path) for entry in _limited(summary.entries, limit)]


def audit_gic_corpus_geometry(
    root: Path,
    *,
    suffixes: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> GICCorpusGeometryAudit:
    target = Path(root).expanduser().resolve()
    requested_suffixes = suffixes if suffixes is not None else GEOMETRY_IMPORT_SUFFIXES
    corpus_entries = _limited(discover_gic_corpus(target, suffixes=requested_suffixes), limit)
    audit_entries = tuple(_audit_geometry_entry(entry) for entry in corpus_entries)
    return GICCorpusGeometryAudit(root=target, entries=audit_entries)


def gic_corpus_geometry_audit_records(
    audit: GICCorpusGeometryAudit,
    *,
    status: str = "all",
    limit: int | None = None,
) -> list[dict[str, object]]:
    entries = _filter_audit_entries(audit.entries, status=status)
    return [entry.record(root=audit.root) for entry in _limited(entries, limit)]


def format_gic_corpus_geometry_audit_summary(audit: GICCorpusGeometryAudit) -> list[str]:
    lines = [
        f"ROOT {audit.root}",
        f"TOTAL_FILES {audit.total_files}",
        f"PASS {audit.passed_files}",
        f"FAIL {audit.failed_files}",
    ]
    for source_format, count in sorted(audit.source_format_counts.items()):
        lines.append(f"SOURCE_FORMAT {source_format} {count}")
    for error_type, count in sorted(audit.error_counts.items()):
        lines.append(f"ERROR_TYPE {error_type} {count}")
    return lines


def format_gic_corpus_geometry_failures(
    audit: GICCorpusGeometryAudit,
    *,
    limit: int | None = None,
) -> list[str]:
    failures = _limited(_filter_audit_entries(audit.entries, status="fail"), limit)
    return [f"FAIL {entry.path} {entry.error_type}: {entry.error}" for entry in failures]


def _audit_geometry_entry(entry: GICCorpusEntry) -> GICCorpusGeometryAuditEntry:
    try:
        from matrix_chem import read_geometry

        geometry = read_geometry(entry.path)
    except Exception as exc:
        return GICCorpusGeometryAuditEntry(
            path=entry.path,
            name=entry.name,
            suffix=entry.suffix,
            role=entry.role,
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    return GICCorpusGeometryAuditEntry(
        path=entry.path,
        name=entry.name,
        suffix=entry.suffix,
        role=entry.role,
        status="PASS",
        source_format=geometry.source_format,
        natoms=geometry.natoms,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
    )


def _filter_audit_entries(
    entries: tuple[GICCorpusGeometryAuditEntry, ...],
    *,
    status: str,
) -> tuple[GICCorpusGeometryAuditEntry, ...]:
    normalized = status.strip().lower()
    if normalized == "all":
        return entries
    if normalized == "pass":
        return tuple(entry for entry in entries if entry.passed)
    if normalized == "fail":
        return tuple(entry for entry in entries if not entry.passed)
    raise GICCorpusError(f"unsupported audit status filter: {status}")


def _limited(entries: tuple[T, ...], limit: int | None) -> tuple[T, ...]:
    if limit is None:
        return entries
    if limit < 0:
        raise GICCorpusError("corpus limit cannot be negative")
    return entries[:limit]


def _normalize_suffixes(suffixes: tuple[str, ...] | list[str] | None) -> set[str]:
    if not suffixes:
        return set()
    normalized: set[str] = set()
    for suffix in suffixes:
        text = suffix.strip().lower()
        if text == "<none>":
            text = ""
        elif text and not text.startswith("."):
            text = f".{text}"
        normalized.add(text)
    return normalized
