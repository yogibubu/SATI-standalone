"""Full Born–Oppenheimer field transport; observable contractions are separate.

Cartesian derivatives use Hartree/bohr**rank. Normal derivatives use
Hartree/(sqrt(amu)*bohr)**rank. Frequencies retain their signed saddle values.
No mode selection or DVR substitution belongs to this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from matrix_rovib.isotopic_cubic import isotopic_masses_amu
from matrix_rovib.vibin import modes_from_hessian

PROTOCOL_ID = "trinity-isotope-full-field-transform.v1"


@dataclass(frozen=True)
class IsotopicAnharmonicField:
    masses_amu: np.ndarray
    frequencies_cm1: np.ndarray
    modes_mw: np.ndarray
    cartesian_per_normal: np.ndarray
    derivatives: dict[int, np.ndarray]
    coverage: str


def transform_isotope_field(
    symbols,
    coordinates_angstrom,
    cartesian_derivatives: Mapping[int, np.ndarray],
    *,
    substitutions: dict[int, int] | None = None,
    coverage: str,
) -> IsotopicAnharmonicField:
    """Transport every supplied full tensor and retain every vibrational mode.

    F2 defines the isotope modes. F3, F4 and any higher supplied orders are
    transformed on *all* indices, without a semidiagonal restriction. The
    input must have a consistent Cartesian frame and derivative convention.
    ``explicit_zero_model`` declares that unprinted terms were explicitly
    modelled as zero upstream; this qualification is retained in the result.
    An incomplete semidiagonal array cannot be presented as a complete field.
    """
    if coverage not in {"complete_tensor", "explicit_zero_model"}:
        raise ValueError("full tensors require complete_tensor or explicit_zero_model coverage")
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    masses = isotopic_masses_amu(symbols, substitutions)
    dimension = 3 * len(symbols)
    if xyz.shape != (len(symbols), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("finite natoms x 3 coordinates are required")
    tensors = {}
    for rank, values in cartesian_derivatives.items():
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("derivative orders must be positive integers")
        tensor = np.asarray(values, dtype=float)
        if tensor.shape != (dimension,) * rank or not np.all(np.isfinite(tensor)):
            raise ValueError(f"F{rank} must be a finite complete Cartesian rank-{rank} tensor")
        for axis in range(1, rank):
            if not np.allclose(tensor, tensor.swapaxes(0, axis), rtol=1e-8, atol=1e-12):
                raise ValueError(f"F{rank} is not permutation symmetric")
        tensors[rank] = tensor
    if 2 not in tensors or 3 not in tensors:
        raise ValueError("an anharmonic field requires complete F2 and F3")
    frequencies, modes = modes_from_hessian(
        masses, tensors[2][np.tril_indices(dimension)], xyz
    )
    change = modes.reshape((len(frequencies), dimension)).T
    change = change / np.sqrt(np.repeat(masses, 3))[:, None]
    transformed = {}
    for rank, tensor in sorted(tensors.items()):
        result = tensor
        # Replace one Cartesian index at a time; retain its original position.
        for axis in range(rank):
            result = np.tensordot(result, change, axes=(axis, 0))
            result = np.moveaxis(result, -1, axis)
        transformed[rank] = result
    return IsotopicAnharmonicField(
        masses, frequencies, modes, change, transformed, coverage
    )


def transform_isotope_field_request(request_path: str | Path) -> Path:
    """Execute a frozen KeyMaker JSON request against a Cartesian NPZ field.

    NPZ keys: coordinates_angstrom, symbols, F2, F3, optionally F1/F4/...
    Request keys: field, coverage, isotopologues (label -> one-based isotope
    substitutions), output_dir. Paths are relative to the request document.
    """
    request_path = Path(request_path).resolve()
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    required = {"field", "coverage", "isotopologues", "output_dir"}
    if set(request) != required:
        raise ValueError("request must contain exactly: " + ", ".join(sorted(required)))
    field = (request_path.parent / request["field"]).resolve()
    output = (request_path.parent / request["output_dir"]).resolve()
    isotopes = request["isotopologues"]
    if not isinstance(isotopes, dict) or not isotopes:
        raise ValueError("at least one labelled isotopologue is required")
    with np.load(field, allow_pickle=False) as source:
        keys = set(source.files)
        derivative_keys = {key for key in keys if key.startswith("F") and key[1:].isdigit()}
        base_keys={"symbols", "coordinates_angstrom"}
        normal_keys={"normal_modes_mw", "masses_amu", "hessian_cartesian"}
        is_normal=bool(keys & normal_keys)
        expected=derivative_keys | base_keys | (normal_keys if is_normal else set())
        if keys != expected:
            raise ValueError("field archive contains unsupported or missing arrays")
        symbols = tuple(str(value) for value in source["symbols"])
        xyz = source["coordinates_angstrom"].copy()
        derivatives = {int(key[1:]): source[key].copy() for key in derivative_keys}
        if is_normal:
            derivatives=cartesian_derivatives_from_normal_field(
                derivatives,source["normal_modes_mw"],source["masses_amu"],
                coordinates_angstrom=xyz,hessian_cartesian=source["hessian_cartesian"],
            )
    results = []
    for label, changes in isotopes.items():
        if not label or label in {".", ".."} or Path(label).name != label:
            raise ValueError("isotopologue labels must be plain filenames")
        result = transform_isotope_field(
            symbols, xyz, derivatives,
            substitutions={int(atom): int(mass) for atom, mass in changes.items()},
            coverage=request["coverage"],
        )
        results.append((label, result))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for label, result in results:
        path = output / (label + ".npz")
        np.savez_compressed(
            path, symbols=np.asarray(symbols), coordinates_angstrom=xyz,
            masses_amu=result.masses_amu, frequencies_cm1=result.frequencies_cm1,
            modes_mw=result.modes_mw, cartesian_per_normal=result.cartesian_per_normal,
            **{f"F{rank}": values for rank, values in result.derivatives.items()},
        )
        records.append({"label": label, "file": path.name,
                        "sha256": sha256(path.read_bytes()).hexdigest()})
    manifest = output / "isotope_fields.json"
    manifest.write_text(json.dumps({
        "schema": "matrix.trinity.isotope_full_field.v1", "protocol_id": PROTOCOL_ID,
        "source_sha256": sha256(field.read_bytes()).hexdigest(),
        "request_sha256": sha256(request_bytes).hexdigest(),
        "coverage": request["coverage"], "orders": sorted(derivatives),
        "source_representation": "parent_normal_with_rigid_motion_completion" if is_normal else "cartesian",
        "normal_coordinate_units": "sqrt(amu)*bohr", "energy_units": "hartree",
        "mode_selection": "all", "isotopologues": records,
    }, indent=2) + "\n")
    return manifest


def cartesian_derivatives_from_normal_field(
    normal_derivatives: Mapping[int, np.ndarray],
    modes_mw: np.ndarray,
    masses_amu: np.ndarray,
    *,
    coordinates_angstrom: np.ndarray,
    hessian_cartesian: np.ndarray,
) -> dict[int, np.ndarray]:
    """Recover the BO Cartesian field with exact rigid-motion identities.

    A normal field does not supply its mixed rotational derivatives directly.
    They must not be set to zero: rotational invariance fixes rank r terms
    from rank r-1 derivatives and rotation generators. Complete each supplied
    order recursively in the full 3N metric before changing isotope masses.
    Translation contractions vanish. Input normal phases must be consistent.
    The source is a stationary point; derivatives use hartree/bohr**rank.
    """
    from matrix_chem import BOHR_TO_ANGSTROM

    masses = np.asarray(masses_amu, dtype=float)
    modes = np.asarray(modes_mw, dtype=float)
    xyz = np.asarray(coordinates_angstrom, dtype=float) / float(BOHR_TO_ANGSTROM)
    hessian = np.asarray(hessian_cartesian, dtype=float)
    dimension = 3*len(masses)
    if (masses.ndim != 1 or np.any(masses <= 0) or
            modes.ndim != 3 or modes.shape[1:] != (len(masses), 3) or
            xyz.shape != (len(masses),3) or hessian.shape != (dimension,dimension) or
            not all(np.all(np.isfinite(v)) for v in (masses,modes,xyz,hessian))):
        raise ValueError("finite parent field, coordinates, modes and positive masses required")
    n = len(modes)
    if n != dimension-6:
        raise ValueError("a complete nonlinear parent normal-mode set is required")
    flat = modes.reshape((n, dimension))
    if not np.allclose(flat @ flat.T, np.eye(n), rtol=2e-7, atol=2e-7):
        raise ValueError("parent mass-weighted modes must be orthonormal")
    xyz = xyz - np.average(xyz,axis=0,weights=masses)
    inertia = sum(m*(np.dot(x,x)*np.eye(3)-np.outer(x,x)) for m,x in zip(masses,xyz))
    moments,axes = np.linalg.eigh(inertia)
    if np.any(moments <= 1e-10):
        raise ValueError("nonlinear parent geometry required for rigid-motion completion")
    translation=np.broadcast_to(np.eye(3)[:,None,:],(3,len(masses),3))/np.sqrt(sum(masses))
    rotation=np.cross(axes.T[:,None,:],xyz[None,:,:])/np.sqrt(moments)[:,None,None]
    displacements=np.concatenate((modes/np.sqrt(masses)[None,:,None],translation,rotation)).reshape(dimension,dimension)
    mass_coordinates=np.repeat(masses,3)
    if not np.allclose((displacements*mass_coordinates)@displacements.T,
                       np.eye(dimension),rtol=2e-7,atol=2e-7):
        raise ValueError("parent modes must be orthogonal to rigid motions")
    P=displacements.T
    inverse=displacements*mass_coordinates[None,:]
    generators=[]
    for axis in axes.T:
        generator=np.column_stack([np.cross(axis,e) for e in np.eye(3)])
        generators.append(np.kron(np.eye(len(masses)),generator))
    def transform(tensor,change):
        for axis in range(tensor.ndim):
            tensor=np.tensordot(tensor,change,axes=(axis,0))
            tensor=np.moveaxis(tensor,-1,axis)
        return tensor
    result={2:hessian.copy()}
    ranks=sorted(normal_derivatives)
    if not ranks or ranks != list(range(3,max(ranks)+1)):
        raise ValueError("complete consecutive normal derivative orders starting at F3 required")
    for rank in ranks:
        normal=np.asarray(normal_derivatives[rank],dtype=float)
        if normal.shape != (n,)*rank or not np.all(np.isfinite(normal)):
            raise ValueError("complete finite normal derivative tensors required")
        full=np.zeros((dimension,)*rank)
        counts=np.zeros((dimension,)*rank,dtype=int)
        vibrational=(slice(0,n),)*rank
        full[vibrational]=normal;counts[vibrational]=1
        previous=result[rank-1]
        for r,(generator,moment) in enumerate(zip(generators,moments)):
            contraction=np.zeros_like(previous)
            for axis in range(rank-1):
                term=np.tensordot(previous,generator,axes=(axis,0))
                contraction-=np.moveaxis(term,-1,axis)
            projected=transform(contraction/np.sqrt(moment),P)
            # Impose the same Ward identity on every permutation of the
            # contracted rotation index; average duplicate constraints.
            for axis in range(rank):
                index=[slice(None)]*rank;index[axis]=n+3+r
                full[tuple(index)]+=projected
                counts[tuple(index)]+=1
        np.divide(full,counts,out=full,where=counts>0)
        for axis in range(rank):
            index=[slice(None)]*rank;index[axis]=slice(n,n+3)
            full[tuple(index)]=0
        result[rank]=transform(full,inverse)
    return result
