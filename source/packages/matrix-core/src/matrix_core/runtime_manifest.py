"""Canonical manifest of the MATRIX Python runtime installed on one node."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .package_registry import PACKAGE_CAPABILITIES


RUNTIME_MANIFEST_SCHEMA = "matrix.runtime-manifest.v1"
_EXPERIMENTAL_PACKAGES = frozenset(
    {
        "matrix-fragment-ndo",
        "matrix-geomlearn",
        "matrix-pdvci",
    }
)


def build_runtime_manifest() -> dict[str, Any]:
    """Describe installed MATRIX distributions without inspecting source files."""

    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.casefold().startswith("matrix-"):
            packages[name] = distribution.version
    expected_packages = {contract.package for contract in PACKAGE_CAPABILITIES}
    installed_packages = {name.casefold().replace("_", "-") for name in packages}
    missing_packages = sorted(expected_packages - installed_packages)
    experimental_packages = sorted(installed_packages & _EXPERIMENTAL_PACKAGES)
    unexpected_packages = sorted(
        installed_packages - expected_packages - _EXPERIMENTAL_PACKAGES
    )
    versions = sorted(set(packages.values()))
    payload: dict[str, Any] = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_abi": sys.implementation.cache_tag or "",
        "architecture": platform.machine(),
        "operating_system": platform.system(),
        "environment_prefix": sys.prefix,
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].casefold())),
        "experimental_packages": experimental_packages,
        "missing_packages": missing_packages,
        "unexpected_packages": unexpected_packages,
        "versions": versions,
        "qualified": not missing_packages and not unexpected_packages and len(versions) == 1,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_runtime_manifest(destination: Path | str) -> dict[str, Any]:
    """Write a manifest to a path, or print it when destination is ``-``."""

    payload = build_runtime_manifest()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(destination) == "-":
        print(rendered, end="")
        return payload
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    write_runtime_manifest(args.output)
    if str(args.output) != "-":
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
