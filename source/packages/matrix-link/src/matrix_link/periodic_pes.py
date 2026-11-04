"""Periodic PES adapter built on LINK's existing scalar realization path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matrix_smith import PeriodicEmbeddingContract, periodic_decode, periodic_embed


@dataclass(frozen=True)
class PeriodicPESAdapter:
    contracts: tuple[PeriodicEmbeddingContract, ...]
    reference_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        contracts = tuple(self.contracts)
        if not contracts:
            raise ValueError("periodic PES adapter requires at least one contract")
        reference = None if self.reference_values is None else np.asarray(self.reference_values, dtype=float).reshape(-1)
        if reference is not None and reference.shape != (len(contracts),):
            raise ValueError("periodic PES reference must match contract count")
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "reference_values", reference)

    @property
    def scalar_count(self) -> int:
        return len(self.contracts)

    @property
    def embedding_count(self) -> int:
        return 2 * self.scalar_count

    def decode(self, embedding: np.ndarray, *, reference: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(embedding, dtype=float).reshape(-1)
        if values.shape != (self.embedding_count,):
            raise ValueError("periodic PES embedding has invalid length")
        prior = self.reference_values if reference is None else np.asarray(reference, dtype=float)
        return np.asarray(
            [
                periodic_decode(
                    values[2 * index : 2 * index + 2], contract,
                    reference=None if prior is None else float(prior[index]),
                )
                for index, contract in enumerate(self.contracts)
            ],
            dtype=float,
        )

    def embed(self, scalar_values: np.ndarray) -> np.ndarray:
        values = np.asarray(scalar_values, dtype=float).reshape(-1)
        if values.shape != (self.scalar_count,):
            raise ValueError("periodic PES scalar values have invalid length")
        return np.concatenate([periodic_embed(float(value), contract) for value, contract in zip(values, self.contracts)])

    def scalar_gradient_to_embedding(
        self,
        scalar_gradient: np.ndarray,
        *,
        scalar_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the minimum-norm tangent gradient compatible with scalar energy."""

        gradient = np.asarray(scalar_gradient, dtype=float).reshape(-1)
        if gradient.shape != (self.scalar_count,):
            raise ValueError("scalar gradient has invalid length")
        values = self.reference_values if scalar_values is None else np.asarray(scalar_values, dtype=float).reshape(-1)
        if values is None or values.shape != (self.scalar_count,):
            raise ValueError("scalar values are required for the embedding gradient")
        result = np.empty(self.embedding_count, dtype=float)
        for index, (value, contract) in enumerate(zip(
            values,
            self.contracts,
        )):
            embedded = periodic_embed(float(value), contract)
            tangent = (2.0 * np.pi / contract.period) * np.asarray(
                (-embedded[1], embedded[0]), dtype=float
            )
            result[2 * index : 2 * index + 2] = gradient[index] * tangent / float(np.dot(tangent, tangent))
        return result

    def embedding_jacobian(self, scalar_values: np.ndarray) -> np.ndarray:
        """Return ``d(embedding)/d(scalar)`` with shape ``(2*n, n)``."""

        values = np.asarray(scalar_values, dtype=float).reshape(-1)
        if values.shape != (self.scalar_count,):
            raise ValueError("scalar values have invalid length")
        jacobian = np.zeros((self.embedding_count, self.scalar_count), dtype=float)
        from matrix_smith import periodic_embedding_derivative

        for index, (value, contract) in enumerate(zip(values, self.contracts)):
            jacobian[2 * index : 2 * index + 2, index] = periodic_embedding_derivative(
                float(value), contract
            )
        return jacobian

    def scalar_hessian_to_embedding(
        self,
        scalar_values: np.ndarray,
        scalar_gradient: np.ndarray,
        scalar_hessian: np.ndarray,
    ) -> np.ndarray:
        """Transform a scalar PES Hessian to the constrained embedding chart."""

        values = np.asarray(scalar_values, dtype=float).reshape(-1)
        gradient = np.asarray(scalar_gradient, dtype=float).reshape(-1)
        hessian = np.asarray(scalar_hessian, dtype=float)
        if values.shape != gradient.shape or values.shape != (self.scalar_count,):
            raise ValueError("scalar PES vectors have invalid lengths")
        if hessian.shape != (self.scalar_count, self.scalar_count):
            raise ValueError("scalar Hessian has invalid shape")
        jacobian = self.embedding_jacobian(values)
        tangent = np.zeros((self.embedding_count, self.embedding_count), dtype=float)
        from matrix_smith import periodic_embed

        for index, (value, contract) in enumerate(zip(values, self.contracts)):
            embedded = periodic_embed(float(value), contract)
            c, s = float(embedded[0]), float(embedded[1])
            inverse_scale = contract.period / (2.0 * np.pi)
            inverse_gradient = inverse_scale * np.asarray((-s, c), dtype=float)
            inverse_hessian = inverse_scale * np.asarray(
                ((2.0 * c * s, s * s - c * c), (s * s - c * c, -2.0 * c * s)),
                dtype=float,
            )
            tangent_block = np.outer(inverse_gradient, inverse_gradient) * float(hessian[index, index])
            curvature_block = float(gradient[index]) * inverse_hessian
            tangent[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = (
                tangent_block + curvature_block
            )
        return jacobian @ hessian @ jacobian.T + tangent
