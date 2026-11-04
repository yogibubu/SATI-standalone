"""Second-order isolated-band rotational tensors from a native LAM model."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from .lam_fit import (
    _lam_variable,
    _apply_lam_variables,
    _operator_parameter_values,
    _projected_coupled_factors,
)
from .lam_rotation import solve_lam_basis, _rotational_operators, _lam_contraction_digest
from .lam_measurements import response_factors
from .general_lam_operator import normalize_general_lam_operator_manifest

_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _coefficients(factors, scalar):
    rot = _rotational_operators(1, "Ir")
    operators = (*rot, *(0.5 * (rot[a] @ rot[b] + rot[b] @ rot[a]) for a, b in _PAIRS))
    design = np.column_stack([o.reshape(-1) for o in operators])
    left_right = (*factors, (-scalar, np.eye(3)))
    weights = np.linalg.solve(
        design, np.column_stack([right.reshape(-1) for _, right in left_right])
    )
    return tuple(
        sum(
            (w * left for w, (left, _) in zip(row, left_right, strict=True)),
            start=np.zeros_like(scalar),
        )
        for row in weights
    )


def lam_rotational_observables(surface, request, values=None, *, lam_state=0, workspace=None):
    """Return A/B/C and analytic sensitivities for an isolated LAM band.

    H=H0+La Ja+Qab sym(Ja Jb). The effective tensor includes the
    second-order virtual-band term La (En-H0)^-1 Lb. Higher homogeneous
    rotational orders do not enter this zero-rotation observable definition.
    Degenerate active LAM bands require a matrix effective Hamiltonian and
    are rejected rather than represented by three scalar constants.
    A canonical workspace may reuse a solved basis at the same surface. Smaller
    requested state counts truncate that basis, preserving boundary degeneracy.
    """
    raw = request.get("operator_manifest")
    manifest = None if raw is None else normalize_general_lam_operator_manifest(raw)
    variables = tuple(
        _lam_variable(v, surface, k, {} if manifest is None else manifest.parameters_MHz)
        for k, v in enumerate(request["variables"])
    )
    values = np.array([v["initial"] for v in variables] if values is None else values, float)
    fitted = _apply_lam_variables(surface, variables, values)
    parameters = _operator_parameter_values(manifest, variables, values)
    options = request.get("options", {})
    count = int(request.get("lam_states", 12))
    if workspace is None:
        basis = solve_lam_basis(
            fitted, state_count=count,
            primitive_basis=options.get("primitive_basis", "auto"),
            fourier_cutoffs=options.get("fourier_cutoffs"),
        )
    else:
        expected = None if manifest is None else manifest.digest
        if (workspace.surface_digest != _lam_contraction_digest(fitted)
                or workspace.operator_manifest_digest != expected
                or not workspace.include_geometric_potential):
            raise ValueError("Rotational-observable workspace does not match the surface and operator")
        basis = workspace.basis
        if not 1 <= count <= len(basis.energies_MHz):
            raise ValueError("Requested states exceed the reusable basis")
        tolerance = max(1e-6, 256*np.spacing(max(abs(basis.energies_MHz))))
        while count < len(basis.energies_MHz) and abs(basis.energies_MHz[count]-basis.energies_MHz[count-1]) <= tolerance:
            count += 1
        basis = replace(basis, energies_MHz=basis.energies_MHz[:count],
            vectors=basis.vectors[:, :count],
            spectral_coefficients=None if basis.spectral_coefficients is None else basis.spectral_coefficients[:, :count],
            symmetry_sectors=basis.symmetry_sectors[:count],
            sector_state_indices=basis.sector_state_indices[:count])
    solved = SimpleNamespace(
        basis=basis,
        blocks={0: None, 1: None},
        operator_parameters_MHz=parameters,
        representation="Ir",
    )

    def projected(J):
        return _projected_coupled_factors(
            fitted,
            basis.vectors,
            J,
            basis,
            operator_manifest=manifest,
            operator_parameters_MHz=parameters,
            maximum_rotation_order=2,
        )

    h0 = sum(
        (a * b[0, 0] for a, b in projected(0)),
        start=np.zeros((len(basis.energies_MHz),) * 2, complex),
    )
    energies, w = np.linalg.eigh((h0 + h0.conj().T) / 2)
    if not 0 <= lam_state < len(energies):
        raise ValueError("LAM band index outside the retained basis")
    coefficients = _coefficients(projected(1), h0)
    n = lam_state
    others = np.arange(len(energies)) != n
    gaps = energies[n] - energies[others]
    if np.any(abs(gaps) <= max(1.0e-8, 128 * np.spacing(max(1.0, abs(energies[n]))))):
        raise ValueError("degenerate LAM band cannot be exported as scalar A/B/C")
    linear = coefficients[:3]
    tensor = np.zeros((3, 3))
    v = w[:, n]
    other_vectors = w[:, others]

    def resolvent(z):
        return other_vectors @ ((other_vectors.conj().T @ z) / gaps)

    columns = [operator @ v for operator in linear]
    responses = [resolvent(z) for z in columns]
    for k, (a, b) in enumerate(_PAIRS):
        direct = np.real(np.vdot(v, coefficients[3 + k] @ v)) / (1 if a == b else 2)
        tensor[a, b] = tensor[b, a] = direct + np.real(np.vdot(columns[a], responses[b]))
    constants, axes = np.linalg.eigh(tensor)
    order = np.argsort(constants)[::-1]
    constants = constants[order]
    axes = axes[:, order]
    if np.any(constants <= 0):
        raise ValueError("isolated-band rotational tensor is not positive definite")
    jac = np.zeros((3, len(values)))
    for column in range(len(values)):
        df, _ = response_factors(
            fitted,
            solved,
            variables,
            values,
            column,
            manifest,
            parameters,
            int(options.get("basis_response_states", 48)),
            moving_basis=True,
            maximum_rotation_order=2,
            parameter_reference_surface=surface,
        )
        dh0 = sum((a * b[0, 0] for a, b in df[0]), start=np.zeros_like(h0))
        dcoeff = _coefficients(df[1], dh0)
        de = np.real(np.vdot(v, dh0 @ v))
        dv = resolvent(dh0 @ v)
        dc = [dcoeff[a] @ v + linear[a] @ dv for a in range(3)]
        dr = [
            resolvent(dh0 @ z - de * z)
            - v * np.vdot(dv, z)
            - resolvent(dv) * np.vdot(v, columns[b])
            for b, z in enumerate(responses)
        ]
        dt = np.zeros((3, 3))
        for k, (a, b) in enumerate(_PAIRS):
            direct = np.vdot(v, dcoeff[3 + k] @ v) + 2 * np.real(
                np.vdot(dv, coefficients[3 + k] @ v)
            )
            response = (
                np.vdot(dc[a], responses[b])
                + np.vdot(responses[a], dc[b])
                + np.vdot(columns[a], dr[b])
            )
            dt[a, b] = dt[b, a] = np.real(direct) / (1 if a == b else 2) + np.real(response)
        eigen_response = axes.T @ dt @ axes
        for a in range(3):
            for b in range(a):
                if abs(constants[a] - constants[b]) < 1.0e-8 and (
                    abs(eigen_response[a, b]) > 1.0e-8
                    or abs(eigen_response[a, a] - eigen_response[b, b]) > 1.0e-8
                ):
                    raise ValueError("principal-constant derivative splits a degenerate tensor")
        jac[:, column] = np.diag(eigen_response)
    return dict(
        schema="matrix.lock.lam_rotational_observables.v1",
        definition="isolated_band_second_order_rotational_tensor",
        lam_state=n,
        constants_MHz=constants.tolist(),
        tensor_MHz=tensor.tolist(),
        jacobian=jac.tolist(),
        parameter_paths=[v["path"] for v in variables],
        retained_lam_states=len(energies),
        limitations="isolated nondegenerate band; truncated virtual LAM sum requires numerical qualification",
    )
