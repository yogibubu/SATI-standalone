#!/usr/bin/env python3
"""Build a user-selected, self-contained SMITH wheelhouse and example kit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib


PROJECTS = (
    "matrix-core",
    "matrix-chem",
    "matrix-switch",
    "matrix-apoc",
    "matrix-epoc",
    "matrix-fragments",
    "matrix-qm",
    "matrix-gfn2",
    "matrix-rovib",
    "matrix-gaussian",
    "matrix-engines",
    "matrix-link",
    "matrix-pyscf",
    "matrix-xtb",
    "matrix-numerics",
    "matrix-zaff",
    "matrix-smith",
    "matrix-gf",
    "matrix-oracle",
    "matrix-morpheus",
    "matrix-trinity",
    "matrix-sentinel",
    "matrix-cypher",
)
RELEASE_INPUTS = (
    *(f"packages/{name}" for name in PROJECTS),
    "examples/local_pseudosymmetry",
    "examples/smith_release",
    "docs/manuals/gicforge_manual.pdf",
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "tools/build_smith_release.py",
    "tools/verify_smith_install.py",
)


def _staging_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {"build", "dist", "__pycache__", ".ruff_cache"}
        or name.endswith((".egg-info", ".pyc", ".pyo", ".so", ".pyd"))
    }


def _stage_project(project: Path, staging_root: Path) -> Path:
    staged = staging_root / project.name
    shutil.copytree(project, staged, ignore=_staging_ignore)
    return staged


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="User-selected release directory")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build a diagnostic artifact and record modified release inputs",
    )
    args = parser.parse_args()
    dirty_paths = _dirty_release_inputs(root)
    if dirty_paths and not args.allow_dirty:
        raise RuntimeError(
            "SMITH release inputs are not committed; commit them or use --allow-dirty "
            f"for a diagnostic build: {', '.join(dirty_paths)}"
        )

    output = args.output.expanduser().resolve()
    wheelhouse = output / "wheels"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    version = _project_version(root / "packages/matrix-smith")
    missing_dependencies = _missing_internal_dependencies(root, list(PROJECTS))
    if missing_dependencies:
        raise RuntimeError(
            "SMITH release is not closed over mandatory MATRIX dependencies: "
            + ", ".join(missing_dependencies)
        )
    component_versions = {
        name: _project_version(root / "packages" / name) for name in PROJECTS
    }
    with tempfile.TemporaryDirectory(prefix="matrix-smith-release-") as temporary:
        staging_root = Path(temporary)
        for name in PROJECTS:
            project = _stage_project(root / "packages" / name, staging_root)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    str(wheelhouse),
                    str(project),
                ],
                check=True,
            )

    examples = output / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    for source in (
        root / "examples/local_pseudosymmetry/sf6_distorted.xyz",
        root / "examples/local_pseudosymmetry/zrf8_square_antiprismatic.xyz",
        root / "examples/smith_release/formic_acid_water.smith.xyz",
        root / "examples/smith_release/eta3_allyl_palladium.xyz",
    ):
        shutil.copyfile(source, examples / source.name)
    shutil.copyfile(root / "docs/manuals/gicforge_manual.pdf", output / "SMITH_MANUAL.pdf")
    shutil.copyfile(root / "README.md", output / "MATRIX_README.md")
    shutil.copyfile(root / "CITATION.cff", output / "CITATION.cff")
    shutil.copyfile(root / "LICENSE", output / "LICENSE")

    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema": "matrix.smith.release.v1",
        "smith_version": version,
        "matrix_revision": _revision(root),
        "matrix_dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "python": sys.version.split()[0],
        "components": PROJECTS,
        "component_versions": component_versions,
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
        "install": f"python -m pip install --find-links {wheelhouse} matrix-smith=={version}",
    }
    manifest_path = output / "smith-release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


def _project_version(project: Path) -> str:
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    return str(metadata["project"]["version"])


def _missing_internal_dependencies(root: Path, projects: list[str]) -> list[str]:
    """Return mandatory in-tree dependencies absent from the wheelhouse plan."""
    included = set(projects)
    missing: set[str] = set()
    for name in projects:
        metadata = tomllib.loads(
            (root / "packages" / name / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        for requirement in metadata.get("dependencies", ()):
            dependency = _requirement_name(str(requirement))
            if (
                dependency.startswith("matrix-")
                and (root / "packages" / dependency).is_dir()
                and dependency not in included
            ):
                missing.add(f"{name} -> {dependency}")
    return sorted(missing)


def _requirement_name(requirement: str) -> str:
    """Extract and normalize a distribution name without importing packaging."""
    boundary = len(requirement)
    for marker in "[<>=!~; ":
        position = requirement.find(marker)
        if position >= 0:
            boundary = min(boundary, position)
    return requirement[:boundary].strip().lower().replace("_", "-")


def _revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _dirty_release_inputs(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *RELEASE_INPUTS],
        cwd=root,
        text=True,
    )
    return sorted(line[3:].strip() for line in output.splitlines() if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
