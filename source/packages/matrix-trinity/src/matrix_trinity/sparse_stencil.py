"""Restartable COO records for one-mode VPT2 finite-difference contributions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class SparseModeContributionCache:
    """Persist each completed mode as four-index/value COO tensor records.

    Rank-three records have -1 in the unused fourth index. No numerical
    threshold is applied: the stored entries reconstruct the stencil exactly.
    The caller supplies a namespace identifying its electronic method, chart,
    geometry, and implementation version; the acquisition hashes its harmonic
    field and step into the per-run signature.
    """

    def __init__(self, directory: Path, *, namespace: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)

    def signature(self, *arrays: np.ndarray, step: float | np.ndarray,
                  analytic_metric: bool) -> str:
        digest = hashlib.sha256(self.namespace.encode())
        digest.update(np.asarray(step, dtype=np.float64).tobytes())
        digest.update(np.asarray([float(analytic_metric)], dtype=np.float64).tobytes())
        for array in arrays:
            value = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def _path(self, mode: int) -> Path:
        return self.directory / f"mode_{mode:04d}.npz"

    @staticmethod
    def _encode(array: np.ndarray, mode: int, *, rank: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(array, dtype=np.float64)
        indices = np.argwhere(values != 0.0).astype(np.int32)
        if rank == 3:
            indices = np.column_stack((
                indices,
                np.full(len(indices), mode, dtype=np.int32),
                np.full(len(indices), -1, dtype=np.int32),
            ))
        elif rank == 4:
            indices = np.column_stack((indices, np.full(len(indices), mode, dtype=np.int32)))
        else:
            raise ValueError("unsupported sparse stencil rank")
        return indices, values[tuple(indices[:, :values.ndim].T)] if len(indices) else np.empty(0)

    def store(self, mode: int, signature: str, fields: dict[str, np.ndarray | float]) -> None:
        record: dict[str, np.ndarray] = {"signature": np.asarray(signature)}
        for name in ("f3", "f4", "g1", "g2"):
            array = np.asarray(fields[name], dtype=float)
            indices, values = self._encode(array, mode, rank=3 if array.ndim == 2 else 4)
            if name == "f4" or (name == "g2" and array.ndim == 2):
                indices[:, 3] = mode
            record[f"{name}_indices"] = indices
            record[f"{name}_values"] = values
            record[f"{name}_shape"] = np.asarray(array.shape, dtype=np.int32)
        record["logdet_first"] = np.asarray(fields["logdet_first"], dtype=float)
        record["determinant_second"] = np.asarray(fields["determinant_second"], dtype=float)
        for name in ("gradient_cubic", "gradient_quartic", "energy_gradient_quartic"):
            if fields.get(name) is not None:
                record[name] = np.asarray(fields[name], dtype=float)
        path = self._path(mode)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **record)
        temporary.replace(path)

    def load(self, mode: int, signature: str) -> dict[str, np.ndarray | float] | None:
        path = self._path(mode)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"]) != signature:
                return None
            fields: dict[str, np.ndarray | float] = {}
            for name in ("f3", "f4", "g1", "g2"):
                shape = tuple(int(item) for item in saved[f"{name}_shape"])
                indices = np.asarray(saved[f"{name}_indices"], dtype=int)
                values = np.asarray(saved[f"{name}_values"], dtype=float)
                if indices.shape != (len(values), 4):
                    raise ValueError("invalid sparse stencil record")
                array = np.zeros(shape, dtype=float)
                if len(values):
                    array[tuple(indices[:, :len(shape)].T)] = values
                fields[name] = array
            fields["logdet_first"] = float(saved["logdet_first"])
            fields["determinant_second"] = float(saved["determinant_second"])
            for name in ("gradient_cubic", "gradient_quartic", "energy_gradient_quartic"):
                fields[name] = (np.asarray(saved[name], dtype=float)
                                if name == "gradient_cubic" and name in saved else
                                float(saved[name]) if name in saved else None)
        return fields
