from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urljoin
from urllib.request import urlretrieve
import zipfile


LCB25_BASE_URL = "https://www.skies-village.it/webtools/Databases/LCB25/"
LCB25_DATASETS = ("PCS2", "SE", "HPCS2")
LCB25_CACHE_SCHEMA = "oracle.lcb25.cache.v1"


@dataclass(frozen=True)
class LCB25Dataset:
    label: str
    url: str
    archive_name: str


class LCB25CacheError(ValueError):
    """Raised when the managed LCB25 cache is missing or inconsistent."""


def lcb25_dataset_url(label: str) -> str:
    normalized = str(label).strip().upper()
    if normalized not in LCB25_DATASETS:
        raise ValueError(f"unsupported LCB25 dataset {label!r}; expected one of {LCB25_DATASETS}")
    return urljoin(LCB25_BASE_URL, f"{normalized}.zip")


def lcb25_download_plan() -> tuple[LCB25Dataset, ...]:
    return tuple(
        LCB25Dataset(label=label, url=lcb25_dataset_url(label), archive_name=f"{label}.zip")
        for label in LCB25_DATASETS
    )


def download_lcb25_dataset(label: str, target_dir: Path) -> Path:
    """Download one LCB25 ZIP archive to `target_dir`.

    Network use is explicit; tests cover URL planning without downloading.
    """
    dataset = LCB25Dataset(
        label=str(label).strip().upper(),
        url=lcb25_dataset_url(label),
        archive_name=f"{str(label).strip().upper()}.zip",
    )
    outdir = Path(target_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    archive = outdir / dataset.archive_name
    urlretrieve(dataset.url, archive)
    return archive


def sync_lcb25_library(
    target_root: Path,
    *,
    datasets: tuple[str, ...] | list[str] | None = None,
    force: bool = False,
) -> Path:
    """Download, extract and manifest the local ORACLE-managed LCB25 cache."""
    root = Path(target_root)
    archive_dir = root / "archives"
    xyz_root = root / "xyz"
    archive_dir.mkdir(parents=True, exist_ok=True)
    xyz_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for label in _normalize_dataset_labels(datasets):
        archive = archive_dir / f"{label}.zip"
        if force or not archive.exists():
            archive = download_lcb25_dataset(label, archive_dir)

        extracted_dir = xyz_root / label
        existing_xyz = tuple(
            sorted(
                path
                for path in extracted_dir.rglob("*.xyz")
                if path.is_file()
                and _is_lcb25_geometry_path(path)
                and _belongs_to_lcb25_dataset(path, extracted_dir, label)
            )
        )
        if force and extracted_dir.exists():
            shutil.rmtree(extracted_dir)
            existing_xyz = ()
        xyz_files = existing_xyz or tuple(
            path
            for path in extract_lcb25_archive(archive, extracted_dir)
            if _belongs_to_lcb25_dataset(path, extracted_dir, label)
        )
        entries.append(
            {
                "label": label,
                "url": lcb25_dataset_url(label),
                "archive": str(archive.relative_to(root)),
                "archive_sha256": _sha256_file(archive),
                "extracted_dir": str(extracted_dir.relative_to(root)),
                "xyz_count": len(xyz_files),
                "xyz_files": [str(path.relative_to(root)) for path in xyz_files],
            }
        )

    manifest = {
        "schema": LCB25_CACHE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": LCB25_BASE_URL,
        "datasets": entries,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def lcb25_geometry_paths(
    cache_root: Path,
    *,
    dataset: str = "PCS2",
) -> tuple[Path, ...]:
    """Return one dataset from the managed cache in manifest order.

    The cache manifest is the sole authority: callers do not depend on ZIP
    layout, duplicated dataset-directory names, or workstation-specific paths.
    """
    root = Path(cache_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise LCB25CacheError(
            f"LCB25 cache manifest not found: {manifest_path}; "
            "run `matrix lcb25 fetch`"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LCB25CacheError(f"cannot read LCB25 cache manifest: {manifest_path}") from exc
    if manifest.get("schema") != LCB25_CACHE_SCHEMA:
        raise LCB25CacheError(
            f"unsupported LCB25 cache schema {manifest.get('schema')!r}; "
            f"expected {LCB25_CACHE_SCHEMA!r}"
        )
    label = _normalize_dataset_labels((dataset,))[0]
    entries = [
        entry for entry in manifest.get("datasets", ()) if entry.get("label") == label
    ]
    if len(entries) != 1:
        raise LCB25CacheError(
            f"LCB25 dataset {label} is not uniquely represented in {manifest_path}"
        )
    paths = []
    for relative in entries[0].get("xyz_files", ()):
        relative_path = Path(str(relative))
        if not _is_lcb25_geometry_path(relative_path) or not _manifest_path_matches_dataset(
            relative_path, label
        ):
            continue
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents:
            raise LCB25CacheError(
                f"unsafe LCB25 geometry path in manifest: {relative!r}"
            )
        if not candidate.is_file():
            raise LCB25CacheError(f"LCB25 geometry is missing: {candidate}")
        paths.append(candidate)
    listed_usable_count = sum(
        _is_lcb25_geometry_path(Path(str(relative)))
        and _manifest_path_matches_dataset(Path(str(relative)), label)
        for relative in entries[0].get("xyz_files", ())
    )
    if not paths or listed_usable_count != len(paths):
        raise LCB25CacheError(
            f"LCB25 dataset {label} count mismatch: "
            f"manifest_usable={listed_usable_count}, resolved={len(paths)}"
        )
    return tuple(paths)


def resolve_lcb25_geometry(
    cache_root: Path,
    identifier: str,
    *,
    dataset: str = "PCS2",
) -> Path:
    """Resolve one unambiguous LCB25 geometry by name or manifest-relative path."""
    root = Path(cache_root).expanduser().resolve()
    paths = lcb25_geometry_paths(root, dataset=dataset)
    requested = str(identifier).strip()
    if not requested:
        raise LCB25CacheError("LCB25 geometry identifier cannot be empty")
    requested_path = Path(requested)
    direct = (root / requested_path).resolve()
    if root in direct.parents and direct in paths:
        return direct
    filename = requested_path.name
    stem = Path(filename).stem if filename.lower().endswith(".xyz") else filename
    exact = [
        path
        for path in paths
        if path.name == filename or path.stem == requested or path.stem == stem
    ]
    matches = exact or [
        path
        for path in paths
        if path.name.casefold() == filename.casefold()
        or path.stem.casefold() == stem.casefold()
    ]
    if not matches:
        raise LCB25CacheError(
            f"LCB25 {str(dataset).upper()} geometry not found: {identifier!r}"
        )
    if len(matches) != 1:
        relative = ", ".join(str(path.relative_to(root)) for path in matches)
        raise LCB25CacheError(
            f"ambiguous LCB25 geometry {identifier!r}; use one manifest-relative "
            f"path: {relative}"
        )
    return matches[0]


def extract_lcb25_archive(archive: Path, target_dir: Path) -> tuple[Path, ...]:
    """Extract an LCB25 archive and return extracted XYZ files."""
    archive = Path(archive)
    outdir = Path(target_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (outdir / member.filename).resolve()
            if outdir.resolve() not in destination.parents and destination != outdir.resolve():
                raise ValueError(f"unsafe path in LCB25 archive: {member.filename}")
        zf.extractall(outdir)
    return tuple(
        sorted(
            path
            for path in outdir.rglob("*.xyz")
            if path.is_file() and _is_lcb25_geometry_path(path)
        )
    )


def _normalize_dataset_labels(datasets: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    labels = (
        LCB25_DATASETS
        if datasets is None
        else tuple(str(item).strip().upper() for item in datasets)
    )
    for label in labels:
        if label not in LCB25_DATASETS:
            raise ValueError(
                f"unsupported LCB25 dataset {label!r}; expected one of {LCB25_DATASETS}"
            )
    return tuple(labels)


def _is_lcb25_geometry_path(path: Path) -> bool:
    return (
        path.suffix.lower() == ".xyz"
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def _belongs_to_lcb25_dataset(path: Path, dataset_root: Path, label: str) -> bool:
    try:
        relative = path.relative_to(dataset_root)
    except ValueError:
        return False
    return bool(relative.parts and relative.parts[0].casefold() == label.casefold())


def _manifest_path_matches_dataset(path: Path, label: str) -> bool:
    parts = path.parts
    return (
        len(parts) >= 4
        and parts[0].casefold() == "xyz"
        and parts[1].casefold() == label.casefold()
        and parts[2].casefold() == label.casefold()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
