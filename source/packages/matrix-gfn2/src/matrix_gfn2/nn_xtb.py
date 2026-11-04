"""Native NN-xTB model contract and ``.gfn2nn`` reader.

This module deliberately contains no external xTB/TBLite call.  It is the
boundary for the native MATRIX implementation: the binary model is validated
here and the eventual MACE encoder/parameter overlay will consume the parsed
metadata.  Keeping this boundary separate makes the plain GFN2 path exactly
backwards compatible while the scientific port is being completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np

from .parameters import EV_PER_HARTREE
from .nn_xtb_mace import (
    _pair_paths, _paths, first_mace_layer, mace_layer_states,
    real_spherical_harmonics,
)


class GFN2NNFormatError(ValueError):
    """Raised when a ``.gfn2nn`` file is malformed or incompatible."""


_MATRIX_SHIFT_FACTORS = {
    # The provisional reference stores the raw xTB table units. MATRIX's
    # parameter objects already contain the converted Hamiltonian units.
    "atom.gam3": 0.1,
    "atom.dpol": 0.01,
    "atom.qpol": 0.01,
    "shell.lev": 1.0 / EV_PER_HARTREE,
    "shell.kcn": 0.1 / EV_PER_HARTREE,
    "shell.poly": 0.01,
    "shell.lpar": 0.1,
    "global.enscale": 0.01,
}


@dataclass(frozen=True)
class GFN2NNHeadSpec:
    name: str
    kind: str
    output_count: int
    denorm_scale: float


@dataclass(frozen=True)
class GFN2NNModelInfo:
    path: str
    version: int
    header: dict[str, str]
    tensor_count: int
    checksum: int
    tensors: dict[str, np.ndarray] | None = None
    heads: tuple[GFN2NNHeadSpec, ...] = ()

    @property
    def architecture(self) -> str:
        return self.header.get("arch", "")


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise GFN2NNFormatError("truncated GFN2NN model")
    return value


def read_gfn2nn_header(path: str | Path) -> GFN2NNModelInfo:
    """Validate a model and return metadata without loading tensor weights.

    Format: ``GFN2NN\\0``, u32 version, u64 header length, newline-separated
    ``key = value`` records, u64 tensor count, tensor records, and a final
    FNV-1a checksum.  Tensor payloads are skipped but their dimensions and
    names are checked so malformed files never reach the calculator.
    """
    model_path = Path(path)
    with model_path.open("rb") as stream:
        if _read_exact(stream, 7) != b"GFN2NN\0":
            raise GFN2NNFormatError("missing GFN2NN magic")
        version = struct.unpack("<I", _read_exact(stream, 4))[0]
        if version != 1:
            raise GFN2NNFormatError(f"unsupported GFN2NN version {version}")
        header_len = struct.unpack("<Q", _read_exact(stream, 8))[0]
        if header_len > 16 * 1024 * 1024:
            raise GFN2NNFormatError("unreasonable GFN2NN header length")
        raw_header = _read_exact(stream, header_len).decode("utf-8")
        header: dict[str, str] = {}
        heads: list[GFN2NNHeadSpec] = []
        for line in raw_header.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key == "head":
                fields = value.split()
                if len(fields) != 4:
                    raise GFN2NNFormatError("invalid GFN2NN head declaration")
                try:
                    count, scale = int(fields[2]), float(fields[3])
                except ValueError as exc:
                    raise GFN2NNFormatError("invalid GFN2NN head dimensions or scale") from exc
                if fields[1] not in {"atomwise", "global"} or count < 1 or not isfinite(scale):
                    raise GFN2NNFormatError("invalid GFN2NN head declaration")
                heads.append(GFN2NNHeadSpec(fields[0], fields[1], count, scale))
            else:
                header[key] = value
        tensor_count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        if tensor_count > 1_000_000:
            raise GFN2NNFormatError("unreasonable GFN2NN tensor count")
        for _ in range(tensor_count):
            name_len = struct.unpack("<H", _read_exact(stream, 2))[0]
            _read_exact(stream, name_len).decode("utf-8")
            rows, cols = struct.unpack("<QQ", _read_exact(stream, 16))
            if rows == 0 or cols == 0 or rows * cols > 2**34:
                raise GFN2NNFormatError("invalid GFN2NN tensor dimensions")
            _read_exact(stream, rows * cols * 8)
        checksum = struct.unpack("<Q", _read_exact(stream, 8))[0]
        if stream.read(1):
            raise GFN2NNFormatError("trailing bytes after GFN2NN checksum")
    payload = model_path.read_bytes()
    expected = 0xCBF29CE484222325
    for byte in payload[:-8]:
        expected ^= byte
        expected = (expected * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    if checksum != expected:
        raise GFN2NNFormatError("GFN2NN checksum mismatch")
    return GFN2NNModelInfo(
        str(model_path), version, header, tensor_count, checksum, heads=tuple(heads)
    )


def load_gfn2nn(path: str | Path) -> GFN2NNModelInfo:
    """Load and validate all model tensors as immutable float64 arrays."""
    info = read_gfn2nn_header(path)
    tensors: dict[str, np.ndarray] = {}
    with Path(path).open("rb") as stream:
        _read_exact(stream, 7 + 4)
        header_len = struct.unpack("<Q", _read_exact(stream, 8))[0]
        _read_exact(stream, header_len)
        count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        for _ in range(count):
            name_len = struct.unpack("<H", _read_exact(stream, 2))[0]
            name = _read_exact(stream, name_len).decode("utf-8")
            rows, cols = struct.unpack("<QQ", _read_exact(stream, 16))
            raw = _read_exact(stream, rows * cols * 8)
            array = np.frombuffer(raw, dtype="<f8").reshape((rows, cols))
            array.setflags(write=False)
            if name in tensors:
                raise GFN2NNFormatError(f"duplicate tensor name {name!r}")
            tensors[name] = array
    return GFN2NNModelInfo(
        info.path, info.version, info.header, info.tensor_count, info.checksum,
        tensors, info.heads,
    )


class NNXTBOverlay:
    """Explicit native overlay contract.

    The implementation is intentionally fail-closed until the MACE encoder,
    Hamiltonian shifts, and analytic first/second derivatives are ported and
    validated against the reference implementation.  Returning plain GFN2
    while silently ignoring a requested NN would be scientifically unsafe.
    """

    def __new__(cls, model_path: str | Path):
        if cls is NNXTBOverlay and Path(model_path).suffix.lower() == '.json':
            from .nn_xtb_official import OfficialNNXTBOverlay
            return object.__new__(OfficialNNXTBOverlay)
        return object.__new__(cls)

    def __init__(self, model_path: str | Path):
        self.info = load_gfn2nn(model_path)
        if self.info.architecture.lower() != "mace":
            raise GFN2NNFormatError(
                f"unsupported native NN-xTB architecture: {self.info.architecture!r}"
            )
        self._validate_mace_schema()

    def _validate_mace_schema(self) -> None:
        """Check the tensor contract before any numerical inference."""
        tensors = self.info.tensors or {}
        required = {"embed"}
        layers = int(self.info.header.get("layers", "0"))
        channels = int(self.info.header.get("channels", "0"))
        elements = int(self.info.header.get("n_elements", "0"))
        radial = int(self.info.header.get("n_radial", "0"))
        l_max = int(self.info.header.get("l_max", "-1"))
        nu_max = int(self.info.header.get("nu_max", "0"))
        if layers < 1 or channels < 1 or elements < 1 or radial < 1:
            raise GFN2NNFormatError("invalid MACE architecture dimensions")
        if l_max < 0 or l_max > 2 or nu_max < 1 or nu_max > 3:
            raise GFN2NNFormatError("unsupported MACE angular or correlation order")
        if tensors.get("embed", np.empty((0, 0))).shape != (elements, channels):
            raise GFN2NNFormatError("MACE embedding shape disagrees with header")
        for layer in range(layers):
            prefix = f"t{layer}"
            for l1, l2, out in _paths(l_max):
                name = f"{prefix}.radial.{l1}{l2}{out}"
                required.add(name)
                if name in tensors and tensors[name].shape != (radial, channels):
                    raise GFN2NNFormatError(f"invalid tensor shape: {name}")
            for l in range(l_max + 1):
                for name, shape in (
                    (f"{prefix}.msg_gain.nu1.{l}", (1, channels)),
                    (f"{prefix}.update.U{l}", (channels, channels)),
                    (f"{prefix}.vgain{l}", (elements, channels)),
                ):
                    required.add(name)
                    if name in tensors and tensors[name].shape != shape:
                        raise GFN2NNFormatError(f"invalid tensor shape: {name}")
            if nu_max >= 2:
                for la, lb, l12 in _pair_paths(l_max):
                    name = f"{prefix}.msg_gain.nu2.{la}{lb}{l12}"
                    required.add(name)
                    if name in tensors and tensors[name].shape != (1, channels):
                        raise GFN2NNFormatError(f"invalid tensor shape: {name}")
                    if nu_max >= 3:
                        for lc in range(l_max + 1):
                            for out in range(abs(l12 - lc), min(l12 + lc, l_max) + 1):
                                name = f"{prefix}.msg_gain.nu3.{la}{lb}{l12}.{lc}{out}"
                                required.add(name)
                                if name in tensors and tensors[name].shape != (1, channels):
                                    raise GFN2NNFormatError(f"invalid tensor shape: {name}")
        missing = sorted(name for name in required if name not in tensors)
        if missing:
            raise GFN2NNFormatError(
                "incomplete MACE tensor schema; missing " + ", ".join(missing)
            )
        if not tensors:
            raise GFN2NNFormatError("empty MACE tensor set")
        hidden = int(self.info.header.get("head_hidden", "0"))
        if hidden < 1 or not self.info.heads:
            raise GFN2NNFormatError("missing MACE head specification")
        names = set()
        for head in self.info.heads:
            if head.name in names:
                raise GFN2NNFormatError(f"duplicate MACE head {head.name!r}")
            names.add(head.name)
            input_dim = channels if head.kind == "atomwise" else channels + 2
            for suffix, shape in (
                ("w1", (input_dim, hidden)), ("b1", (1, hidden)),
                ("w2", (hidden, hidden)), ("b2", (1, hidden)),
                ("w3", (hidden, head.output_count)),
                ("b3", (1, head.output_count)),
            ):
                name = f"head.{head.name}.{suffix}"
                if name not in tensors or tensors[name].shape != shape:
                    raise GFN2NNFormatError(f"missing or malformed head tensor: {name}")

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(sorted((self.info.tensors or {}).keys()))

    def prepare_geometry(self, atomic_numbers, coordinates_bohr):
        """Build the fixed-geometry neighbor graph and radial Bessel features.

        Coordinates are Cartesian bohr.  The returned arrays are immutable and
        use the model's cutoff, radial count and polynomial envelope.  Angular
        equivariants are added by the subsequent MACE layer implementation.
        """
        z = np.array(atomic_numbers, dtype=int, copy=True)
        xyz = np.array(coordinates_bohr, dtype=float, copy=True)
        if z.ndim != 1 or xyz.shape != (z.size, 3):
            raise ValueError("atomic numbers and coordinates have incompatible shapes")
        if np.any(z < 1) or np.any(z > int(self.info.header["n_elements"])):
            raise ValueError("atomic number outside NN-xTB embedding table")
        if not np.all(np.isfinite(xyz)):
            raise ValueError("coordinates must be finite")
        cutoff = float(self.info.header["r_cut"])
        n_radial = int(self.info.header["n_radial"])
        envelope_p = int(self.info.header.get("envelope_p", "6"))
        src, dst, distances, directions = [], [], [], []
        for i in range(z.size):
            delta = xyz - xyz[i]
            norms = np.linalg.norm(delta, axis=1)
            for j in range(z.size):
                if i == j:
                    continue
                if norms[j] <= 1.0e-10:
                    raise ValueError("coincident atoms in NN-xTB geometry")
                if norms[j] >= cutoff:
                    continue
                src.append(i)
                dst.append(j)
                distances.append(float(norms[j]))
                directions.append(delta[j] / norms[j])
        d = np.asarray(distances, dtype=float)
        scaled = d / cutoff
        p = envelope_p
        envelope = (1.0 - (p + 1) * (p + 2) / 2 * scaled**p
                    + p * (p + 2) * scaled**(p + 1)
                    - p * (p + 1) / 2 * scaled**(p + 2))
        n = np.arange(1, n_radial + 1, dtype=float)
        radial = (np.sqrt(2.0 / cutoff) * envelope[:, None]
                  * np.sin(np.pi * scaled[:, None] * n[None, :]) / d[:, None])
        direction = np.asarray(directions, dtype=float).reshape((-1, 3))
        sh = real_spherical_harmonics(direction, int(self.info.header["l_max"]))
        source = np.asarray(src, dtype=int)
        target = np.asarray(dst, dtype=int)
        for array in (z, xyz, source, target, d, direction, radial, sh):
            array.setflags(write=False)
        return {
            "atomic_numbers": z,
            "coordinates_bohr": xyz,
            "source": source,
            "target": target,
            "distance_bohr": d,
            "direction": direction,
            "radial_features": radial,
            "spherical_harmonics": sh,
        }

    def forward_first_layer(self, atomic_numbers, coordinates_bohr) -> np.ndarray:
        """Return the first equivariant layer before the remaining encoder layers."""
        graph = self.prepare_geometry(atomic_numbers, coordinates_bohr)
        return first_mace_layer(
            graph, self.info.tensors or {},
            l_max=int(self.info.header["l_max"]),
            nu_max=int(self.info.header["nu_max"]),
        )

    def forward_layer_states(self, atomic_numbers, coordinates_bohr) -> tuple[np.ndarray, ...]:
        """Return every equivariant layer state for numerical parity checks."""
        graph = self.prepare_geometry(atomic_numbers, coordinates_bohr)
        states = mace_layer_states(
            graph, self.info.tensors or {},
            layers=int(self.info.header["layers"]),
            l_max=int(self.info.header["l_max"]),
            nu_max=int(self.info.header["nu_max"]),
        )
        for state in states:
            state.setflags(write=False)
        return states

    def forward_descriptors(self, atomic_numbers, coordinates_bohr) -> np.ndarray:
        """Return invariant per-atom descriptors from the final MACE layer."""
        return self.forward_layer_states(atomic_numbers, coordinates_bohr)[-1][:, 0, :]

    def forward_head_outputs(
        self, atomic_numbers, coordinates_bohr, *, charge: float = 0.0,
        unpaired: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Evaluate bounded NN head outputs before physical denormalization."""
        descriptors = self.forward_descriptors(atomic_numbers, coordinates_bohr)
        if not isfinite(charge) or not isfinite(unpaired):
            raise ValueError("charge and unpaired electron count must be finite")
        global_input = np.concatenate((
            descriptors.sum(axis=0, keepdims=True),
            np.array([[charge, unpaired]], dtype=float),
        ), axis=1)
        tensors = self.info.tensors or {}
        bound = self.info.header.get("shift_bound", "clamp")
        if bound not in {"clamp", "tanh"}:
            raise GFN2NNFormatError(f"unsupported head bound: {bound}")
        outputs = {}
        for head in self.info.heads:
            x = descriptors if head.kind == "atomwise" else global_input
            for index in (1, 2):
                value = x @ tensors[f"head.{head.name}.w{index}"]
                value += tensors[f"head.{head.name}.b{index}"]
                x = value * np.exp(-np.logaddexp(0.0, -value))
            raw = x @ tensors[f"head.{head.name}.w3"]
            raw += tensors[f"head.{head.name}.b3"]
            if bound == "clamp":
                value = np.clip(1.0e-4 * raw, -0.5, 0.5)
            else:
                value = 0.5 * np.tanh(2.0e-4 * raw)
            if not np.all(np.isfinite(value)):
                raise ArithmeticError(f"nonfinite NN head output: {head.name}")
            value.setflags(write=False)
            outputs[head.name] = value
        return outputs

    def parameter_shifts(self, atomic_numbers, coordinates_bohr, *,
                         charge: float = 0.0, unpaired: float = 0.0):
        """Return per-head physical parameter shifts for this geometry."""
        bounded = self.forward_head_outputs(
            atomic_numbers, coordinates_bohr, charge=charge, unpaired=unpaired,
        )
        scales = {head.name: head.denorm_scale for head in self.info.heads}
        shifts = {name: output * scales[name] for name, output in bounded.items()}
        for shift in shifts.values():
            shift.setflags(write=False)
        return shifts

    def matrix_parameter_shifts(self, atomic_numbers, coordinates_bohr, *,
                                charge: float = 0.0, unpaired: float = 0.0):
        """Convert provisional reference shifts to MATRIX parameter units.

        These arrays are not an energy overlay: absent shell couplings and
        per-atom parameter routing must still be handled by the calculator.
        """
        reference_shifts = self.parameter_shifts(
            atomic_numbers, coordinates_bohr, charge=charge, unpaired=unpaired,
        )
        converted = {
            name: shift * _MATRIX_SHIFT_FACTORS.get(name, 1.0)
            for name, shift in reference_shifts.items()
        }
        for shift in converted.values():
            shift.setflags(write=False)
        return converted

    def parameter_shift_vjp(
        self, atomic_numbers, coordinates_bohr, adjoints, *,
        charge: float = 0.0, unpaired: float = 0.0,
        hessian_direction=None,
    ):
        """Differentiate a weighted sum of physical parameter shifts.

        ``adjoints[name]`` represents the derivative of an external energy
        with respect to that head's physical parameter shift. With a direction
        this also returns the Hessian-vector product of the same *fixed*
        weighted sum. The calculator must supply its own parameter adjoints;
        this function does not assert that its energy path is NN-ready.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("NN-xTB coordinate derivatives require torch") from exc
        from .nn_xtb_autograd import parameter_shifts_torch

        xyz = torch.tensor(np.asarray(coordinates_bohr, dtype=float),
                           dtype=torch.float64, requires_grad=True)
        outputs = parameter_shifts_torch(
            self, atomic_numbers, xyz, charge=charge, unpaired=unpaired,
        )
        scales = {head.name: head.denorm_scale for head in self.info.heads}
        for name, output in outputs.items():
            if (self.info.header.get('shift_bound') != 'none'
                    and torch.max(torch.abs(output.detach() / scales[name])).item() >= 0.5 * (1 - 1e-9)):
                raise ArithmeticError(f"NN-xTB head {name} reached its shift bound")
        extra = set(adjoints) - set(outputs)
        if extra:
            raise ValueError(f"unknown NN-xTB parameter adjoints: {sorted(extra)}")
        scalar = None
        for name, values in adjoints.items():
            weight = torch.as_tensor(values, dtype=torch.float64)
            if weight.shape != outputs[name].shape:
                raise ValueError(f"adjoint for {name} has incompatible shape")
            term = torch.sum(weight * outputs[name])
            scalar = term if scalar is None else scalar + term
        if scalar is None:
            raise ValueError("at least one parameter adjoint is required")
        gradient = torch.autograd.grad(
            scalar, xyz, create_graph=hessian_direction is not None,
        )[0]
        if hessian_direction is None:
            return gradient.detach().numpy()
        direction = torch.as_tensor(hessian_direction, dtype=torch.float64)
        if direction.shape != xyz.shape:
            raise ValueError("hessian direction has incompatible shape")
        hvp = torch.autograd.grad(torch.sum(gradient * direction), xyz)[0]
        return gradient.detach().numpy(), hvp.detach().numpy()

    def parameter_shift_jvp(
        self, atomic_numbers, coordinates_bohr, direction_bohr, *,
        charge: float = 0.0, unpaired: float = 0.0,
    ):
        """Analytic directional derivatives of every physical NN head shift."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("NN-xTB coordinate derivatives require torch") from exc
        from .nn_xtb_autograd import parameter_shifts_torch

        coordinates = np.asarray(coordinates_bohr, dtype=float)
        direction = np.asarray(direction_bohr, dtype=float)
        if coordinates.shape != direction.shape or coordinates.shape != (len(atomic_numbers), 3):
            raise ValueError("NN-xTB parameter JVP direction has incompatible shape")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(direction)):
            raise ValueError("NN-xTB parameter JVP coordinates and direction must be finite")
        xyz = torch.as_tensor(coordinates, dtype=torch.float64)
        tangent = torch.as_tensor(direction, dtype=torch.float64)
        names = tuple(head.name for head in self.info.heads)

        def outputs(values):
            shifts = parameter_shifts_torch(
                self, atomic_numbers, values, charge=charge, unpaired=unpaired,
            )
            return tuple(shifts[name] for name in names)

        _, derivatives = torch.autograd.functional.jvp(
            outputs, xyz, tangent, create_graph=False, strict=False,
        )
        return {
            name: value.detach().numpy()
            for name, value in zip(names, derivatives, strict=True)
        }
