"""Deterministic resolver for the local LCB26 geometry cache."""

from __future__ import annotations

from pathlib import Path


LCB26_DATASETS = ("L2", "PL2", "PCS2", "XTB")
LCB26_GEOMETRY_ONLY_DATASETS = ("L1_GEOMETRY",)


class LCB26CacheError(ValueError):
    """Raised when an LCB26 cache lookup is missing or ambiguous."""


def lcb26_geometry_paths(cache_root: Path, *, dataset: str = "PL2") -> tuple[Path, ...]:
    root = Path(cache_root).expanduser().resolve()
    label = str(dataset).upper()
    if label in LCB26_DATASETS:
        dataset_root = root / "geometries" / label
    elif label in LCB26_GEOMETRY_ONLY_DATASETS:
        dataset_root = root / "l1_geometries" / "records"
    else:
        expected = LCB26_DATASETS + LCB26_GEOMETRY_ONLY_DATASETS
        raise LCB26CacheError(f"unsupported LCB26 dataset {dataset!r}; expected {expected}")
    if not dataset_root.is_dir():
        raise LCB26CacheError(f"LCB26 dataset directory is missing: {dataset_root}")
    return tuple(sorted(path.resolve() for path in dataset_root.rglob("*.xyz")))


def resolve_lcb26_geometry(cache_root: Path, identifier: str, *, dataset: str = "PL2") -> Path:
    root = Path(cache_root).expanduser().resolve()
    paths = lcb26_geometry_paths(root, dataset=dataset)
    requested = str(identifier).strip()
    if not requested:
        raise LCB26CacheError("LCB26 geometry identifier cannot be empty")
    direct = (root / requested).resolve()
    if root in direct.parents and direct in paths:
        return direct
    needle = Path(requested).name
    stem = Path(needle).stem if needle.lower().endswith(".xyz") else needle
    matches = [path for path in paths if path.name == needle or path.stem == stem]
    if not matches:
        matches = [path for path in paths if path.name.casefold() == needle.casefold() or path.stem.casefold() == stem.casefold()]
    if not matches:
        raise LCB26CacheError(f"LCB26 {dataset.upper()} geometry not found: {identifier!r}")
    if len(matches) != 1:
        options = ", ".join(str(path.relative_to(root)) for path in matches)
        raise LCB26CacheError(f"ambiguous LCB26 geometry {identifier!r}; use one path: {options}")
    return matches[0]


__all__ = [
    "LCB26CacheError",
    "LCB26_DATASETS",
    "LCB26_GEOMETRY_ONLY_DATASETS",
    "lcb26_geometry_paths",
    "resolve_lcb26_geometry",
]
