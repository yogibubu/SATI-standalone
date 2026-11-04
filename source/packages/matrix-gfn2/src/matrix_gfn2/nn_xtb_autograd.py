"""Differentiable native NN-xTB parameter shifts for MATRIX-GFN2."""

from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt

import numpy as np

from .nn_xtb_mace import _block, _pair_paths, _paths, real_cg


@lru_cache(maxsize=None)
def _cg_torch(l1: int, l2: int, out: int):
    import torch

    return torch.as_tensor(np.array(real_cg(l1, l2, out)), dtype=torch.float64)


def _harmonics_torch(unit, l_max: int):
    import torch

    x, y, z = unit.unbind(dim=1)
    parts = [torch.ones_like(x) / sqrt(4.0 * pi)]
    if l_max >= 1:
        c = sqrt(3.0 / (4.0 * pi))
        parts.extend((c * y, c * z, c * x))
    if l_max >= 2:
        c = sqrt(15.0 / pi) / 2.0
        parts.extend((c * x * y, c * y * z,
                      sqrt(5.0 / pi) / 4.0 * (3.0 * z * z - 1.0),
                      c * x * z, c / 2.0 * (x * x - y * y)))
    return torch.stack(parts, dim=1)


def _geometry_torch(coordinates, graph, n_radial: int, cutoff: float,
                    envelope_p: int, l_max: int):
    import torch

    source = torch.as_tensor(graph["source"].copy(), dtype=torch.long)
    target = torch.as_tensor(graph["target"].copy(), dtype=torch.long)
    displacement = coordinates[target] - coordinates[source]
    distance = torch.linalg.vector_norm(displacement, dim=1)
    unit = displacement / distance[:, None]
    scaled = distance / cutoff
    p = envelope_p
    envelope = (1.0 - (p + 1) * (p + 2) / 2 * scaled**p
                + p * (p + 2) * scaled**(p + 1)
                - p * (p + 1) / 2 * scaled**(p + 2))
    indices = torch.arange(1, n_radial + 1, dtype=coordinates.dtype)
    radial = (sqrt(2.0 / cutoff) * envelope[:, None]
              * torch.sin(pi * scaled[:, None] * indices[None, :])
              / distance[:, None])
    return source, target, radial, _harmonics_torch(unit, l_max)


def _mace_layer_torch(h, source, target, radial, harmonics, tensors,
                      layer: int, l_max: int, nu_max: int):
    import torch

    n_atoms, n_sh, channels = h.shape
    n_edges = len(source)
    prefix = f"t{layer}"
    a_feat = torch.zeros_like(h)
    for l1, l2, out in _paths(l_max):
        weight = radial @ tensors[f"{prefix}.radial.{l1}{l2}{out}"]
        coupled = torch.einsum(
            "ea,ebc,oab->eoc", harmonics[:, _block(l1)],
            h[target, _block(l2)], _cg_torch(l1, l2, out),
        )
        full = torch.zeros((n_edges, n_sh, channels), dtype=h.dtype)
        full[:, _block(out)] = coupled * weight[:, None, :]
        a_feat = a_feat + torch.zeros_like(h).index_add(0, source, full)
    message = torch.zeros_like(h)
    for l in range(l_max + 1):
        full = torch.zeros_like(h)
        full[:, _block(l)] = (
            a_feat[:, _block(l)] * tensors[f"{prefix}.msg_gain.nu1.{l}"])
        message = message + full
    pairs = {}
    if nu_max >= 2:
        for la, lb, out in _pair_paths(l_max):
            coupled = torch.einsum(
                "nac,nbc,oab->noc", a_feat[:, _block(la)],
                a_feat[:, _block(lb)], _cg_torch(la, lb, out),
            )
            pairs[(la, lb, out)] = coupled
            full = torch.zeros_like(h)
            full[:, _block(out)] = (
                coupled * tensors[f"{prefix}.msg_gain.nu2.{la}{lb}{out}"])
            message = message + full
    if nu_max >= 3:
        for (la, lb, l12), left in pairs.items():
            for lc in range(l_max + 1):
                for out in range(abs(l12 - lc), min(l12 + lc, l_max) + 1):
                    coupled = torch.einsum(
                        "nac,nbc,oab->noc", left, a_feat[:, _block(lc)],
                        _cg_torch(l12, lc, out),
                    )
                    full = torch.zeros_like(h)
                    full[:, _block(out)] = (
                        coupled * tensors[f"{prefix}.msg_gain.nu3.{la}{lb}{l12}.{lc}{out}"])
                    message = message + full
    updated = []
    z = torch.as_tensor(tensors["_atomic_numbers"].copy(), dtype=torch.long)
    for l in range(l_max + 1):
        updated.append(
            message[:, _block(l)] @ tensors[f"{prefix}.update.U{l}"]
            + h[:, _block(l)] * tensors[f"{prefix}.vgain{l}"][z - 1, None, :]
        )
    return torch.cat(updated, dim=1)


def parameter_shifts_torch(overlay, atomic_numbers, coordinates, *,
                           charge: float = 0.0, unpaired: float = 0.0):
    """Evaluate physical parameter shifts on a differentiable Torch graph."""
    import torch

    if hasattr(overlay, 'torch_parameter_shifts'):
        return overlay.torch_parameter_shifts(
            atomic_numbers, coordinates, charge=charge, unpaired=unpaired)

    numbers = np.asarray(atomic_numbers, dtype=int)
    graph = overlay.prepare_geometry(numbers, coordinates.detach().cpu().numpy())
    header = overlay.info.header
    l_max = int(header["l_max"])
    tensors = {name: torch.as_tensor(value.copy(), dtype=coordinates.dtype)
               for name, value in (overlay.info.tensors or {}).items()}
    tensors["_atomic_numbers"] = numbers
    source, target, radial, harmonics = _geometry_torch(
        coordinates, graph, int(header["n_radial"]), float(header["r_cut"]),
        int(header["envelope_p"]), l_max,
    )
    h = torch.zeros((len(numbers), (l_max + 1)**2, int(header["channels"])),
                    dtype=coordinates.dtype)
    h[:, 0] = tensors["embed"][torch.as_tensor(numbers - 1, dtype=torch.long)]
    for layer in range(int(header["layers"])):
        h = _mace_layer_torch(h, source, target, radial, harmonics, tensors,
                              layer, l_max, int(header["nu_max"]))
    descriptor = h[:, 0, :]
    global_input = torch.cat((
        descriptor.sum(dim=0, keepdim=True),
        coordinates.new_tensor([[charge, unpaired]]),
    ), dim=1)
    bound = header.get("shift_bound", "clamp")
    outputs = {}
    for head in overlay.info.heads:
        x = descriptor if head.kind == "atomwise" else global_input
        for index in (1, 2):
            value = x @ tensors[f"head.{head.name}.w{index}"]
            value = value + tensors[f"head.{head.name}.b{index}"]
            x = torch.nn.functional.silu(value)
        raw = x @ tensors[f"head.{head.name}.w3"]
        raw = raw + tensors[f"head.{head.name}.b3"]
        if bound == "clamp":
            bounded = torch.clamp(1.0e-4 * raw, -0.5, 0.5)
        elif bound == "tanh":
            bounded = 0.5 * torch.tanh(2.0e-4 * raw)
        else:
            raise ValueError(f"unsupported NN-xTB shift bound: {bound}")
        outputs[head.name] = bounded * head.denorm_scale
    return outputs
