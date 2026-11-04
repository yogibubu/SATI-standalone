"""SMILES serialization helpers.

The first contract guarantees lossless source round trips. Canonical graph
serialization will be introduced only with a versioned canonicalization
contract, so atom ordering never changes silently between releases.
"""

from __future__ import annotations

from .model import SwitchMolecularGraph


def write_smiles(graph: SwitchMolecularGraph) -> str:
    return graph.source_smiles


__all__ = ["write_smiles"]
