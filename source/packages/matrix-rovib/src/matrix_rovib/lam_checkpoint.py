"""Portable, content-bound LAM workspaces; arrays plus typed JSON, never pickle."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib import import_module
import hashlib, json
from pathlib import Path
import numpy as np

_SCHEMA = "matrix.lock.lam_workspace.v1"
_MODULES = (
    "spin_hyperfine",
    "lam_fit",
    "lam_rotation",
    "lam_k_contraction",
    "lam_fourier",
    "general_lam_operator",
    "molecular_symmetry",
    "rovibrational_symmetry",
)
_EXECUTION = {
    "workspace_checkpoint_path",
    "resume_workspace",
    "_workspace_resume_state",
    "_workspace_resume_global",
    "checkpoint_path",
    "resume_checkpoint",
    "optimizer_state",
    "max_nfev",
    "workers",
    "block_workers",
    "jacobian_workers",
    "dataset_workers",
    "memory_budget_bytes",
    "linearization_path",
    "iteration_trace_path",
    "evaluation_log_path",
    "verbose",
}


def _registry():
    result = {}
    for module in _MODULES:
        imported = import_module("matrix_rovib." + module)
        for name, value in vars(imported).items():
            if (
                isinstance(value, type)
                and is_dataclass(value)
                and value.__module__ == imported.__name__
            ):
                result[value.__module__ + "." + name] = value
    return result


def _digest_array(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def lam_restart_contract(surface, request):
    """Bind physical fields, assignments and numerical spaces, not run budgets."""

    def canonical(value, key=None):
        if isinstance(value, np.ndarray):
            return {"array": _digest_array(value)}
        if isinstance(value, np.generic):
            return canonical(value.item())
        if is_dataclass(value):
            return {f.name: canonical(getattr(value, f.name), f.name) for f in fields(value)}
        if isinstance(value, Mapping):
            return {
                str(k): canonical(v, str(k))
                for k, v in sorted(value.items(), key=lambda p: str(p[0]))
                if str(k) not in _EXECUTION and str(k) != "initial"
            }
        if isinstance(value, (tuple, list)):
            return [canonical(v) for v in value]
        if isinstance(value, complex):
            return [value.real, value.imag]
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported physical contract value: {type(value)}")

    from .lam_rotation import LAM_KINETIC_CONVENTION

    encoded = json.dumps(
        canonical(
            dict(surface=surface, request=request, kinetic_convention=LAM_KINETIC_CONVENTION)
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_lam_workspace(path, payload, *, contract):
    arrays = {}
    objects = {}
    memo = {}
    registry = _registry()

    def encode(value):
        if isinstance(value, np.generic):
            return encode(value.item())
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if np.isfinite(value):
                return value
            return {"float": str(value)}
        if isinstance(value, complex):
            return {"complex": [value.real, value.imag]}
        identity = id(value)
        if identity in memo:
            return {"ref": memo[identity]}
        name = str(len(objects))
        memo[identity] = name
        objects[name] = None
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays are not portable LAM state")
            key = "a" + str(len(arrays))
            arrays[key] = np.ascontiguousarray(value)
            node = {
                "array": key,
                "sha256": _digest_array(value),
                "writeable": value.flags.writeable,
            }
        elif is_dataclass(value):
            kind = type(value).__module__ + "." + type(value).__name__
            if kind not in registry:
                raise TypeError(f"unregistered LAM workspace type: {kind}")
            node = {
                "class": kind,
                "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
            }
        elif isinstance(value, Mapping):
            node = {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
        elif isinstance(value, (tuple, list)):
            node = {"tuple" if isinstance(value, tuple) else "list": [encode(v) for v in value]}
        else:
            raise TypeError(f"unsupported LAM workspace object: {type(value)}")
        objects[name] = node
        return {"ref": name}

    root = encode(payload)
    metadata = dict(schema=_SCHEMA, contract=contract, root=root, objects=objects)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
            **arrays,
        )
    temporary.replace(target)
    return target


def read_lam_workspace(path, *, contract):
    registry = _registry()
    memo = {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schema") != _SCHEMA or metadata.get("contract") != contract:
            raise ValueError("LAM workspace contract mismatch")

        def decode(node):
            if not isinstance(node, dict):
                return node
            if "complex" in node:
                return complex(*node["complex"])
            if "float" in node:
                return float(node["float"])
            key = node["ref"]
            if key in memo:
                return memo[key]
            record = metadata["objects"][key]
            if "array" in record:
                value = archive[record["array"]].copy()
                if _digest_array(value) != record["sha256"]:
                    raise ValueError("LAM workspace array checksum mismatch")
                value.setflags(write=record["writeable"])
            elif "class" in record:
                if record["class"] not in registry:
                    raise ValueError("unregistered LAM workspace class")
                value = registry[record["class"]](
                    **{k: decode(v) for k, v in record["fields"].items()}
                )
            elif "mapping" in record:
                value = {decode(k): decode(v) for k, v in record["mapping"]}
            elif "tuple" in record:
                value = tuple(decode(v) for v in record["tuple"])
            else:
                value = [decode(v) for v in record["list"]]
            memo[key] = value
            return value

        return decode(metadata["root"])


def snapshot_response_workspace(workspace):
    """Store preconditioner data; reconstruct only its action closure on reuse."""
    result = {k: v for k, v in workspace.items() if k != "prepared"}
    result["prepared"] = {
        key: s
        if isinstance(s, dict)
        else dict(
            diagonal=s.diagonal,
            energies=s.energies,
            vectors=s.vectors,
            energy=s.energy,
            cluster=s.cluster,
            counts=s.counts,
            warm=s.warm,
        )
        for key, s in workspace.get("prepared", {}).items()
    }
    return result


def copy_workspace_request(request):
    """Copy JSON inputs without duplicating loaded numerical workspaces."""
    from copy import deepcopy

    def public(value):
        if isinstance(value, Mapping):
            return {
                k: public(v) for k, v in value.items() if not str(k).startswith("_workspace_resume")
            }
        if isinstance(value, list):
            return [public(v) for v in value]
        if isinstance(value, tuple):
            return tuple(public(v) for v in value)
        return value

    return deepcopy(public(request))
