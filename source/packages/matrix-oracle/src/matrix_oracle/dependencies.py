"""Optional dependency discovery without importing heavy integrations eagerly."""
from __future__ import annotations
import importlib.util

OPTIONAL_MODULES = {
    "smiles": ("matrix_switch", "matrix_link"),
    "formats": ("matrix_gaussian", "matrix_molpro", "matrix_mrcc", "matrix_orca", "matrix_xtb", "matrix_pyscf"),
    "qt": ("PySide6",),
}

def dependency_status() -> dict[str, object]:
    groups = {name: {module: importlib.util.find_spec(module) is not None for module in modules}
              for name, modules in OPTIONAL_MODULES.items()}
    return {"schema": "matrix.oracle.dependencies.v1", "groups": groups}
