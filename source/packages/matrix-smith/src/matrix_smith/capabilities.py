from __future__ import annotations

import importlib.util
import platform


def smith_capabilities() -> dict[str, object]:
    """Report stable SMITH coordinate and artifact capabilities."""

    modules = ("numpy", "scipy", "matrix_chem", "matrix_qm", "matrix_engines")
    return {
        "schema": "matrix.smith.capabilities.v1",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
        "backends": ("python", "fortran-audit", "gaussian-export"),
        "onic": {
            "core": "ONIC",
            "branches": ("TONIC", "CONIC", "SONIC"),
            "typed_block_representations": (
                "SYMMETRY_ADAPTED_CARTESIAN",
                "INVERSE_DISTANCE_PROJECTOR",
                "NATURAL_INTERNAL",
                "EXPONENTIAL_MAP",
                "PSEUDO_BOND_CONTACT",
            ),
            "self_contained_artifact_schema": "matrix.smith.typed_onic_artifact.v1",
            "general_sparse_b_prime": True,
        },
    }
