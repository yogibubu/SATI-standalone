"""Deterministic cost and memory records for symmetry-adapted GFN2 algebra."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .symmetry import SymmetryAdaptedBasis


@dataclass(frozen=True)
class SymmetryBenchmark:
    point_group: str
    ao_dimension: int
    block_labels: tuple[str, ...]
    block_dimensions: tuple[int, ...]
    dense_elements: int
    blocked_elements: int
    storage_reduction: float
    cubic_work_reduction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def symmetry_benchmark(adapter: SymmetryAdaptedBasis) -> SymmetryBenchmark:
    """Report implementation-independent storage and eigensolver work ratios."""

    dimension = int(adapter.transformation.shape[0])
    block_dimensions = adapter.block_dimensions
    dense_elements = dimension * dimension
    blocked_elements = sum(size * size for size in block_dimensions)
    blocked_cubic = sum(size**3 for size in block_dimensions)
    return SymmetryBenchmark(
        point_group=adapter.contract.point_group,
        ao_dimension=dimension,
        block_labels=adapter.block_labels,
        block_dimensions=block_dimensions,
        dense_elements=dense_elements,
        blocked_elements=blocked_elements,
        storage_reduction=dense_elements / blocked_elements,
        cubic_work_reduction=dimension**3 / blocked_cubic,
    )


__all__ = ["SymmetryBenchmark", "symmetry_benchmark"]
