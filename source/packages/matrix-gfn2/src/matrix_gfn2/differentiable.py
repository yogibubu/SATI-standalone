"""Differentiable MATRIX-GFN2 adapters for fitting and hybrid ML models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calculator import ANGSTROM_TO_BOHR, GFN2Calculator


@dataclass(frozen=True)
class ImplicitVJPResult:
    parameter_cotangent: np.ndarray
    adjoint: np.ndarray
    residual: float


def implicit_fixed_point_vjp(
    state_jacobian: np.ndarray,
    parameter_jacobian: np.ndarray,
    output_cotangent: np.ndarray,
    *,
    regularization: float = 0.0,
) -> ImplicitVJPResult:
    """Differentiate a converged fixed point without unrolling its iterations.

    For ``x = f(x, theta)``, the reverse adjoint solves
    ``(I - df/dx)^T lambda = dL/dx`` and returns
    ``dL/dtheta = lambda^T df/dtheta``.
    """

    state = np.asarray(state_jacobian, dtype=np.float64)
    parameters = np.asarray(parameter_jacobian, dtype=np.float64)
    cotangent = np.asarray(output_cotangent, dtype=np.float64).reshape(-1)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state Jacobian must be square")
    if parameters.ndim != 2 or parameters.shape[0] != state.shape[0]:
        raise ValueError("parameter Jacobian has incompatible state dimension")
    if cotangent.shape != (state.shape[0],):
        raise ValueError("output cotangent has incompatible state dimension")
    system = np.eye(len(state)) - state
    if regularization < 0.0:
        raise ValueError("implicit differentiation regularization must be nonnegative")
    if regularization:
        system = system + float(regularization) * np.eye(len(state))
    adjoint = np.linalg.solve(system.T, cotangent)
    residual = float(np.linalg.norm(system.T @ adjoint - cotangent))
    return ImplicitVJPResult(parameters.T @ adjoint, adjoint, residual)


class DifferentiableGFN2Model:
    """Coordinate-differentiable facade over the autonomous analytic kernel."""

    def __init__(self, calculator: GFN2Calculator) -> None:
        self.calculator = calculator

    def energy_and_gradient(
        self, coordinates_angstrom: np.ndarray
    ) -> tuple[float, np.ndarray]:
        result = self.calculator.evaluate(coordinates_angstrom)
        return result.energy_hartree, result.gradient_hartree_per_bohr.copy()

    def torch_energy(self, coordinates):
        """Return a Torch scalar/vector with the analytic MATRIX backward pass.

        Torch remains optional: importing MATRIX-GFN2 does not import or depend
        on it.  Coordinates may contain one structure or a leading batch axis.
        """

        import torch

        calculator = self.calculator

        class _AnalyticEnergy(torch.autograd.Function):
            @staticmethod
            def forward(ctx, values):
                array = values.detach().cpu().numpy()
                single = array.ndim == 2
                structures = array[None, ...] if single else array
                results = calculator.evaluate_batch(
                    structures,
                    workers=None,
                    retain_electronic_state=False,
                )
                gradients = np.stack(
                    [result.gradient_hartree_per_bohr for result in results]
                ) * ANGSTROM_TO_BOHR
                gradient_tensor = torch.as_tensor(
                    gradients,
                    dtype=values.dtype,
                    device=values.device,
                )
                ctx.single = single
                ctx.save_for_backward(gradient_tensor)
                energies = torch.as_tensor(
                    [result.energy_hartree for result in results],
                    dtype=values.dtype,
                    device=values.device,
                )
                return energies[0] if single else energies

            @staticmethod
            def backward(ctx, energy_cotangent):
                (gradients,) = ctx.saved_tensors
                if ctx.single:
                    return energy_cotangent * gradients[0]
                return energy_cotangent[:, None, None] * gradients

        return _AnalyticEnergy.apply(coordinates)


__all__ = [
    "DifferentiableGFN2Model",
    "ImplicitVJPResult",
    "implicit_fixed_point_vjp",
]
