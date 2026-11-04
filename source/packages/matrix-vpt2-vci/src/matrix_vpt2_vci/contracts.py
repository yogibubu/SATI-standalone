from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from matrix_engines import vpt2_vci_fortran_layout


@dataclass(frozen=True)
class ForceFieldSource:
    """Normalized source descriptor for a Gaussian quartic force field."""

    path: Path
    source_type: str = "gaussian-log"
    checksum: str | None = None


@dataclass(frozen=True)
class DavidsonSettings:
    """Numerical settings for future large VCI diagonalization."""

    n_roots: int = 10
    max_subspace: int = 80
    max_iter: int = 200
    convergence: float = 1.0e-8
    preconditioner: str = "diagonal"

    def validate(self) -> None:
        if self.n_roots < 1:
            raise ValueError("Davidson needs at least one root")
        if self.max_subspace < self.n_roots:
            raise ValueError("Davidson max_subspace must be >= n_roots")
        if self.max_iter < 1:
            raise ValueError("Davidson max_iter must be positive")
        if self.convergence <= 0.0:
            raise ValueError("Davidson convergence must be positive")


@dataclass(frozen=True)
class VCIRequest:
    """High-level VPT2/VCI request before backend-specific input is written."""

    force_field: ForceFieldSource
    max_quanta: int = 6
    basis_energy_cutoff_cm: float | None = None
    davidson: DavidsonSettings = field(default_factory=DavidsonSettings)

    def validate(self) -> None:
        if self.max_quanta < 1:
            raise ValueError("VCI max_quanta must be positive")
        if self.basis_energy_cutoff_cm is not None and self.basis_energy_cutoff_cm <= 0.0:
            raise ValueError("VCI basis energy cutoff must be positive")
        self.davidson.validate()


@dataclass(frozen=True)
class VPT2VCIInventory:
    """Current VPT2/VCI backend status discovered in the ORACLE tree."""

    harmonic_source: Path | None
    active_fortran_sources: tuple[Path, ...]
    davidson_backend: Path | None
    notes: tuple[str, ...]


def inventory_vpt2_vci_backends(repo_root: Path) -> VPT2VCIInventory:
    """Record active VPT2/VCI kernels available in ORACLE."""
    layout = vpt2_vci_fortran_layout(Path(repo_root))
    harmonic_source = layout.source_dir / "gf_core.f"
    sources = tuple(path for path in layout.source_paths if path.exists())
    davidson = layout.source_dir / "davidson_core.f"
    notes = [
        "Harmonic TRINITY is a separate matrix_gf workflow through independent gf_core.f and matrix_gf.harmonic.",
        "VPT2/VCI works on Cartesian normal-mode QFF inputs with independent Davidson support; Gaussian QFF tensor promotion is still being expanded.",
    ]
    return VPT2VCIInventory(
        harmonic_source=harmonic_source if harmonic_source.exists() else None,
        active_fortran_sources=sources,
        davidson_backend=davidson if davidson.exists() else None,
        notes=tuple(notes),
    )
