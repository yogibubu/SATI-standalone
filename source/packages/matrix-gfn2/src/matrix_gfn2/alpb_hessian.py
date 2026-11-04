"""Exact automatic differentiation of the native ALPB energy functional.

Only the second-derivative path uses PyTorch.  Its forward expression is the
same piecewise-smooth GFN2-ALPB functional as :mod:`matrix_gfn2.alpb`; no
finite differences, external xTB calculator, or learned model are involved.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json

import numpy as np

from .alpb import ALPB_ALPHA, ANGSTROM_TO_BOHR, KCAL_TO_HARTREE, P16_ZETA, ALPBParameters


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Analytic ALPB Hessians require PyTorch; install matrix-gfn2[solvation]"
        ) from exc
    return torch


@lru_cache(maxsize=1)
def _angular_grid() -> tuple[np.ndarray, np.ndarray]:
    grid = json.loads(files("matrix_gfn2").joinpath("data/alpb_lebedev_230.json").read_text())
    return np.asarray(grid["xyz"], dtype=float), np.asarray(grid["weights"], dtype=float)


def _surface_neighbors(
    xyz: np.ndarray, numbers: np.ndarray, parameters: ALPBParameters,
) -> tuple[np.ndarray, ...]:
    n = len(numbers)
    surface_radius = parameters.radii_bohr[numbers - 1] + parameters.probe_bohr
    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return tuple(
        np.flatnonzero(
            (distance[i] < surface_radius[i] + surface_radius
             + 0.3 * ANGSTROM_TO_BOHR)
            & (np.arange(n) != i)
        )
        for i in range(n)
    )


def _energy(
    xyz, charges, numbers: np.ndarray, parameters: ALPBParameters,
    surface_neighbors: tuple[np.ndarray, ...] | None = None,
):
    torch = _torch()
    tensor = lambda values: torch.as_tensor(values, dtype=xyz.dtype, device=xyz.device)
    radius = tensor(parameters.radii_bohr[numbers - 1])
    rho = radius * tensor(parameters.descreening[numbers - 1])
    cavity = radius - parameters.born_offset_bohr
    displacement = xyz[:, None, :] - xyz[None, :, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    n = len(numbers)
    diagonal = torch.eye(n, dtype=torch.bool, device=xyz.device)
    pair_distance = torch.where(diagonal, torch.full_like(distance, 1.0e9), distance)
    center = radius[:, None]
    neighbor = rho[None, :]
    ap = pair_distance + neighbor
    am = pair_distance - neighbor
    safe_am = am.clamp_min(1.0e-12)
    outside = (neighbor / (ap * safe_am)
               + 0.5 * torch.log(safe_am / ap) / pair_distance)
    ratio = ap / center
    partial = (1.0 / center - 1.0 / ap
               + 0.5 / pair_distance * (
                   0.5 * am * (1.0 / ap - ratio / center) - torch.log(ratio)
               ))
    pair_psi = torch.where(
        pair_distance >= center + neighbor, outside,
        torch.where(pair_distance + neighbor <= center, 0.0, partial),
    )
    pair_psi = torch.where(
        (pair_distance <= 35.0 * ANGSTROM_TO_BOHR) & ~diagonal,
        pair_psi, 0.0,
    )
    psi = pair_psi.sum(dim=1)
    scaled = 0.5 * cavity * psi
    argument = scaled * (1.0 + scaled * (4.85 * scaled - 0.8))
    born = parameters.born_scale / (1.0 / cavity - torch.tanh(argument) / radius)
    pair_radius = torch.sqrt(born[:, None] * born[None, :])
    p16 = (1.0 + P16_ZETA * distance / (16.0 * pair_radius)) ** -16
    dielectric_factor = ((1.0 / parameters.dielectric - 1.0)
                         / (1.0 + ALPB_ALPHA / parameters.dielectric))
    polar = 0.5 * dielectric_factor * torch.sum(
        charges[:, None] * charges[None, :] / (distance + pair_radius * p16)
    )
    volume = radius**3
    total_volume = volume.sum()
    centered = xyz - (volume[:, None] * xyz).sum(dim=0) / total_volume
    second = torch.einsum("i,ij,ik->jk", volume, centered, centered)
    inertia = ((volume * (centered.square().sum(dim=1)
                + 0.4 * radius.square())).sum() * torch.eye(3, dtype=xyz.dtype)
               - second)
    size = torch.sqrt(torch.det(inertia).pow(1.0 / 3.0) / (0.4 * total_volume))
    polar = polar + (0.5 * dielectric_factor * ALPB_ALPHA
                     / parameters.dielectric * charges.sum().square() / size)

    angular_points, angular_weights = _angular_grid()
    angular = tensor(angular_points)
    weight = tensor(angular_weights)
    surface_radius = radius + parameters.probe_bohr
    smoothing = 0.3 * ANGSTROM_TO_BOHR
    cubic = -1.0 / (4.0 * smoothing**3)
    if surface_neighbors is None:
        surface_neighbors = _surface_neighbors(
            xyz.detach().cpu().numpy(), numbers, parameters,
        )
    areas = []
    for i, r in enumerate(surface_radius):
        def primitive(value):
            return (0.25 / smoothing + 3.0 * cubic
                    * (0.2 * value.square() - 0.5 * value * r + r.square() / 3.0)
                    ) * value.pow(3)
        radial_weight = primitive(r + smoothing) - primitive(r - smoothing)
        neighbors = surface_neighbors[i]
        if len(neighbors) == 0:
            areas.append(radial_weight * weight.sum())
            continue
        points = xyz[i] + r * angular
        separation = torch.linalg.vector_norm(
            points[None, :, :] - xyz[neighbors, None, :], dim=-1,
        )
        reduced = separation - surface_radius[neighbors, None]
        normalized = reduced / smoothing
        switch = torch.where(
            reduced <= -smoothing, 0.0,
            torch.where(reduced >= smoothing, 1.0,
                        0.5 + 0.75 * normalized - 0.25 * normalized.pow(3)),
        )
        exposure = switch.prod(dim=0)
        exposure = torch.where(exposure > 1.0e-6, exposure, 0.0)
        areas.append(radial_weight * torch.dot(weight, exposure))
    area = torch.stack(areas)
    tension = tensor(parameters.surface_tension_hartree_per_bohr2[numbers - 1])
    hbond = (-KCAL_TO_HARTREE
             * tensor(parameters.hydrogen_bond_sqrt_kcal[numbers - 1]).square()
             / (4.0 * np.pi * surface_radius.square()))
    return polar + torch.dot(area, tension + hbond * charges.square()) + parameters.shift_hartree


def alpb_energy_hessian_vector_product_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    charges_electron: np.ndarray,
    parameters: ALPBParameters,
    coordinate_direction_bohr: np.ndarray,
    charge_direction_electron: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact ``(E_RR v + E_Rq w, E_qR v + E_qq w)``.

    PyTorch differentiates the native ALPB expression twice.  The returned
    charge block includes polar and hydrogen-bond contributions; its use in
    CPSCF must not duplicate the already assembled fixed-geometry SCC matrix.
    """
    torch = _torch()
    numbers = np.asarray(atomic_numbers, dtype=int)
    xyz = np.asarray(coordinates_bohr, dtype=float)
    charges = np.asarray(charges_electron, dtype=float)
    direction = np.asarray(coordinate_direction_bohr, dtype=float)
    q_direction = np.asarray(charge_direction_electron, dtype=float)
    if xyz.shape != (len(numbers), 3) or direction.shape != xyz.shape:
        raise ValueError("ALPB coordinate Hessian direction has the wrong shape")
    if charges.shape != (len(numbers),) or q_direction.shape != charges.shape:
        raise ValueError("ALPB charge Hessian direction has the wrong shape")
    n = len(numbers)
    point = torch.as_tensor(np.concatenate((xyz.ravel(), charges)), dtype=torch.float64)
    tangent = torch.as_tensor(np.concatenate((direction.ravel(), q_direction)), dtype=torch.float64)
    def scalar(vector):
        return _energy(vector[:3 * n].reshape(n, 3), vector[3 * n:], numbers, parameters)
    _, product = torch.autograd.functional.hvp(scalar, point, tangent, strict=False)
    result = product.detach().numpy()
    return result[:3 * n].reshape(n, 3), result[3 * n:]


class ALPBHessianOperator:
    """Reuse one differentiated ALPB graph for many exact Hessian actions."""

    def __init__(
        self,
        coordinates_bohr: np.ndarray,
        atomic_numbers: np.ndarray,
        charges_electron: np.ndarray,
        parameters: ALPBParameters,
    ) -> None:
        torch = _torch()
        self.numbers = np.asarray(atomic_numbers, dtype=int)
        xyz = np.asarray(coordinates_bohr, dtype=float)
        charges = np.asarray(charges_electron, dtype=float)
        if xyz.shape != (len(self.numbers), 3) or charges.shape != (len(self.numbers),):
            raise ValueError("ALPB Hessian operator geometry/charge shape mismatch")
        self.natoms = len(self.numbers)
        surface_neighbors = _surface_neighbors(xyz, self.numbers, parameters)
        self.point = torch.tensor(
            np.concatenate((xyz.ravel(), charges)),
            dtype=torch.float64,
            requires_grad=True,
        )
        energy = _energy(
            self.point[:3 * self.natoms].reshape(self.natoms, 3),
            self.point[3 * self.natoms:],
            self.numbers,
            parameters,
            surface_neighbors,
        )
        self.gradient = torch.autograd.grad(
            energy, self.point, create_graph=True,
        )[0]
        self.seed = torch.zeros_like(self.point, requires_grad=True)
        self.second_vjp = torch.autograd.grad(
            self.gradient, self.point, grad_outputs=self.seed,
            create_graph=True, retain_graph=True,
        )[0]

    def matvec(
        self,
        coordinate_direction_bohr: np.ndarray,
        charge_direction_electron: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = _torch()
        direction = np.asarray(coordinate_direction_bohr, dtype=float)
        q_direction = np.asarray(charge_direction_electron, dtype=float)
        if direction.shape != (self.natoms, 3) or q_direction.shape != (self.natoms,):
            raise ValueError("ALPB Hessian operator direction shape mismatch")
        tangent = torch.as_tensor(
            np.concatenate((direction.ravel(), q_direction)), dtype=torch.float64,
        )
        product = torch.autograd.grad(
            self.second_vjp, self.seed, grad_outputs=tangent, retain_graph=True,
        )[0]
        result = product.detach().numpy()
        return result[:3 * self.natoms].reshape(self.natoms, 3), result[3 * self.natoms:]

    def matvec_batch(
        self,
        coordinate_directions_bohr: np.ndarray,
        charge_directions_electron: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a bounded batch of Hessian directions on the shared graph."""
        torch = _torch()
        directions = np.asarray(coordinate_directions_bohr, dtype=float)
        q_directions = np.asarray(charge_directions_electron, dtype=float)
        if (directions.ndim != 3 or directions.shape[1:] != (self.natoms, 3)
                or q_directions.shape != (len(directions), self.natoms)):
            raise ValueError("ALPB Hessian operator batch direction shape mismatch")
        tangent = torch.as_tensor(
            np.concatenate((directions.reshape(len(directions), -1), q_directions), axis=1),
            dtype=torch.float64,
        )
        product = torch.autograd.grad(
            self.second_vjp, self.seed, grad_outputs=tangent,
            retain_graph=True, is_grads_batched=True,
        )[0].detach().numpy()
        return (
            product[:, :3 * self.natoms].reshape(len(directions), self.natoms, 3),
            product[:, 3 * self.natoms:],
        )


def alpb_geometry_hessian_columns_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    charges_electron: np.ndarray,
    parameters: ALPBParameters,
    column_indices: np.ndarray,
) -> np.ndarray:
    """Exact fixed-charge ``E_RR`` columns, sharing one forward-AD batch."""

    torch = _torch()
    xyz = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    charges = np.asarray(charges_electron, dtype=float)
    columns = np.asarray(column_indices, dtype=int)
    if xyz.shape != (len(numbers), 3) or charges.shape != (len(numbers),):
        raise ValueError("ALPB geometry/charge shape mismatch")
    if columns.ndim != 1 or np.any((columns < 0) | (columns >= xyz.size)):
        raise ValueError("ALPB Hessian column indices are invalid")
    if len(columns) == 0:
        return np.empty((xyz.size, 0), dtype=float)
    neighbors = _surface_neighbors(xyz, numbers, parameters)
    point = torch.tensor(xyz, dtype=torch.float64)
    qpoint = torch.tensor(charges, dtype=torch.float64)
    directions = torch.eye(xyz.size, dtype=torch.float64)[columns].reshape(
        len(columns), *xyz.shape,
    )

    def geometry_gradient(positions):
        return torch.func.grad(
            lambda value: _energy(value, qpoint, numbers, parameters, neighbors)
        )(positions)

    products = torch.vmap(
        lambda tangent: torch.func.jvp(
            geometry_gradient, (point,), (tangent,),
        )[1]
    )(directions)
    return products.detach().numpy().reshape(len(columns), -1).T


def alpb_mixed_geometry_charge_hartree(
    coordinates_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    charges_electron: np.ndarray,
    parameters: ALPBParameters,
) -> np.ndarray:
    """Exact ``E_Rq`` using forward charge and reverse geometry AD."""

    torch = _torch()
    xyz = np.asarray(coordinates_bohr, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    charges = np.asarray(charges_electron, dtype=float)
    if xyz.shape != (len(numbers), 3) or charges.shape != (len(numbers),):
        raise ValueError("ALPB mixed derivative geometry/charge shape mismatch")
    neighbors = _surface_neighbors(xyz, numbers, parameters)
    point = torch.tensor(xyz, dtype=torch.float64)
    qpoint = torch.tensor(charges, dtype=torch.float64)

    def geometry_gradient(q):
        return torch.func.grad(
            lambda positions: _energy(positions, q, numbers, parameters, neighbors)
        )(point).reshape(-1)

    return torch.func.jacfwd(geometry_gradient)(qpoint).detach().numpy()
