"""Sparse handoff of a covariantly assembled local-channel DVR to LOCK.

The matrix already contains overlap transport between neighboring local
channel bases. It is retained as a finite vibrational Hamiltonian, rather
than reduced to an adiabatic scalar surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

from .native_rotor import MHZ_TO_KELVIN, angular_momentum_operators, dipole_transition_matrix
from .rovibrational_symmetry import LAMRotationSymmetry


CM_TO_MHZ = 29979.2458


@dataclass(frozen=True)
class LocalChannelDVR:
    hamiltonian_cm: csr_matrix
    grid_shape: tuple[int, ...]
    channels: int
    provenance: str

    def __post_init__(self) -> None:
        matrix = self.hamiltonian_cm.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        shape = tuple(self.grid_shape)
        if (not shape or any(type(n) is not int or n < 2 for n in shape)
                or type(self.channels) is not int or self.channels < 1):
            raise ValueError("local-channel DVR needs a positive grid and channel count")
        size = int(np.prod(shape)) * self.channels
        if matrix.shape != (size, size) or not np.all(np.isfinite(matrix.data)):
            raise ValueError("local-channel Hamiltonian size or coefficients are invalid")
        antisymmetric = matrix - matrix.getH()
        if antisymmetric.nnz and np.max(np.abs(antisymmetric.data)) > 1e-9:
            raise ValueError("local-channel Hamiltonian is not Hermitian")
        object.__setattr__(self, "hamiltonian_cm", matrix)
        object.__setattr__(self, "grid_shape", shape)

    @property
    def dimension(self) -> int:
        return self.hamiltonian_cm.shape[0]

    @classmethod
    def from_archive(cls, matrix_path: Path | str, manifest_path: Path | str):
        """Read a CSR NPZ plus grid/channel manifest without densifying it."""
        manifest = json.loads(Path(manifest_path).read_text())
        matrix = load_npz(matrix_path)
        shape = tuple(int(n) for n in manifest["explicit_grid_shape"])
        channels = int(manifest["channels"])
        if (matrix.shape[0] != int(manifest["hamiltonian_dimension"])
                or matrix.nnz != int(manifest["hamiltonian_nonzeros"])):
            raise ValueError("channel archive and manifest disagree on matrix size or sparsity")
        return cls(matrix, shape, channels, str(manifest.get("method", manifest_path)))

    def finite_difference_components(
        self, spacings: tuple[float, float], *,
        kinetic_coefficient_cm: float, order: int,
    ) -> "LocalChannelFDComponents":
        """Recover site energies and nearest link overlaps from a 2D FD archive.

        The archive convention must be site-major/channel-minor, with
        off-site blocks ``kinetic_stencil(step) * <phi_i|phi_j>``. This
        factorization is checked; it is not inferred from a fit.
        """
        if len(self.grid_shape) != 2 or order not in {2, 4}:
            raise ValueError("FD component extraction requires a two-dimensional order-2/4 channel grid")
        if (len(spacings) != 2 or any(not np.isfinite(h) or h <= 0 for h in spacings)
                or not np.isfinite(kinetic_coefficient_cm) or kinetic_coefficient_cm <= 0):
            raise ValueError("finite-difference spacings and kinetic coefficient must be positive")
        if np.iscomplexobj(self.hamiltonian_cm.data) and np.max(
            np.abs(self.hamiltonian_cm.data.imag), initial=0.0,
        ) > 1e-10:
            raise ValueError("real FD link extraction cannot discard complex channel phases")
        nq, nu = self.grid_shape
        channels = self.channels
        local = np.empty((nq, nu, channels))
        links_q = np.empty((nq - 1, nu, channels, channels))
        links_u = np.empty((nq, nu - 1, channels, channels))
        step = -kinetic_coefficient_cm * (4 / 3 if order == 4 else 1)
        diagonal = kinetic_coefficient_cm * (5 / 2 if order == 4 else 2)
        center = diagonal * sum(1 / h**2 for h in spacings)

        def block(a, b):
            return self.hamiltonian_cm[
                a * channels:(a + 1) * channels,
                b * channels:(b + 1) * channels,
            ].toarray()

        for i in range(nq):
            for j in range(nu):
                site = i * nu + j
                onsite = block(site, site)
                if np.max(np.abs(onsite - np.diag(np.diag(onsite)))) > 1e-9:
                    raise ValueError("FD source archive has non-diagonal local channel energies")
                local[i, j] = np.diag(onsite).real - center
                if i + 1 < nq:
                    links_q[i, j] = block(site, (i + 1) * nu + j).real / (step / spacings[0]**2)
                if j + 1 < nu:
                    links_u[i, j] = block(site, site + 1).real / (step / spacings[1]**2)
        if max(np.linalg.svd(links_q, compute_uv=False).max(),
               np.linalg.svd(links_u, compute_uv=False).max()) > 1 + 1e-8:
            raise ValueError("recovered channel links cannot be overlaps of orthonormal local spaces")
        return LocalChannelFDComponents(local, links_q, links_u)


@dataclass(frozen=True)
class LocalChannelFDComponents:
    local_energies_cm: np.ndarray
    q_link_overlaps: np.ndarray
    u_link_overlaps: np.ndarray


def coarsen_local_channel_components(
    components: LocalChannelFDComponents,
    stride: int,
) -> LocalChannelFDComponents:
    """Restrict a uniform 2D channel grid while composing link transports.

    The two omitted Dirichlet boundary points are included when deciding the
    retained nodes.  Products of adjacent overlap matrices transport the
    retained channel subspace across removed sites without fitting a new
    molecule-dependent coupling.
    """
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError("coarsening stride must be a positive integer")
    stride = int(stride)
    local = np.asarray(components.local_energies_cm)
    if local.ndim != 3:
        raise ValueError("local channel energies must have (q,u,channel) shape")
    nq, nu, channels = local.shape
    if (nq + 1) % stride or (nu + 1) % stride:
        raise ValueError("coarsening stride must divide both full-grid interval counts")
    q_indices = np.arange(stride, nq + 1, stride) - 1
    u_indices = np.arange(stride, nu + 1, stride) - 1
    restricted = local[np.ix_(q_indices, u_indices, np.arange(channels))]
    qlinks = np.empty((len(q_indices) - 1, len(u_indices), channels, channels), dtype=complex)
    ulinks = np.empty((len(q_indices), len(u_indices) - 1, channels, channels), dtype=complex)
    for i, left in enumerate(q_indices[:-1]):
        right = q_indices[i + 1]
        for j, source_j in enumerate(u_indices):
            overlap = np.eye(channels, dtype=complex)
            for link in range(left, right):
                overlap = overlap @ np.asarray(components.q_link_overlaps[link, source_j])
            qlinks[i, j] = overlap
    for i, source_i in enumerate(q_indices):
        for j, left in enumerate(u_indices[:-1]):
            right = u_indices[j + 1]
            overlap = np.eye(channels, dtype=complex)
            for link in range(left, right):
                overlap = overlap @ np.asarray(components.u_link_overlaps[source_i, link])
            ulinks[i, j] = overlap
    if not any(np.iscomplexobj(value) for value in (
        components.q_link_overlaps, components.u_link_overlaps,
    )):
        qlinks, ulinks = qlinks.real, ulinks.real
    return LocalChannelFDComponents(restricted, qlinks, ulinks)


def refine_scalar_local_channel_components(
    components: LocalChannelFDComponents,
    coordinates: tuple[np.ndarray, np.ndarray],
    factor: int,
) -> tuple[LocalChannelFDComponents, tuple[np.ndarray, np.ndarray]]:
    """Spline-refine one adiabatic channel with geometric overlap transport.

    The interpolation acts on local energies and on the principal-angle
    density ``acos(|S|)/h`` rather than directly on overlaps.  Consequently a
    refined link has the correct quadratic approach to unity for a smooth
    normalized state.  This is intended for independent grid-convergence
    diagnostics after channel-count convergence has already been established.
    """
    local = np.asarray(components.local_energies_cm, dtype=float)
    if local.ndim != 3 or local.shape[-1] != 1:
        raise ValueError("scalar channel refinement requires exactly one retained channel")
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        raise ValueError("refinement factor must be a positive integer")
    factor = int(factor)
    nq, nu, _ = local.shape
    q, u = (np.asarray(values, dtype=float) for values in coordinates)
    if (q.shape != (nq + 2,) or u.shape != (nu + 2,)
            or not np.allclose(np.diff(q), np.diff(q)[0])
            or not np.allclose(np.diff(u), np.diff(u)[0])):
        raise ValueError("scalar refinement requires uniform full grids")
    qfine = np.linspace(q[0], q[-1], (len(q) - 1) * factor + 1)
    ufine = np.linspace(u[0], u[-1], (len(u) - 1) * factor + 1)
    qi, ui = q[1:-1], u[1:-1]
    qfi, ufi = qfine[1:-1], ufine[1:-1]

    def spline(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> RectBivariateSpline:
        return RectBivariateSpline(
            x, y, np.asarray(values, dtype=float),
            kx=min(3, len(x) - 1), ky=min(3, len(y) - 1), s=0,
        )

    energy = spline(qi, ui, local[..., 0])(qfi, ufi, grid=True)[..., None]
    hq, hu = float(q[1] - q[0]), float(u[1] - u[0])
    # A scalar local eigenvector has an arbitrary U(1) phase at every grid
    # point.  Only the principal angle, hence the overlap magnitude, is a
    # gauge-invariant input to a scalar refinement.
    qoverlap = np.abs(np.asarray(components.q_link_overlaps[..., 0, 0]))
    uoverlap = np.abs(np.asarray(components.u_link_overlaps[..., 0, 0]))
    qdensity = np.arccos(np.clip(qoverlap, 0.0, 1.0)) / hq
    udensity = np.arccos(np.clip(uoverlap, 0.0, 1.0)) / hu
    qcenters = 0.5 * (qi[:-1] + qi[1:])
    ucenters = 0.5 * (ui[:-1] + ui[1:])
    qfine_centers = 0.5 * (qfi[:-1] + qfi[1:])
    ufine_centers = 0.5 * (ufi[:-1] + ufi[1:])
    qrate = spline(qcenters, ui, qdensity)(qfine_centers, ufi, grid=True)
    urate = spline(qi, ucenters, udensity)(qfi, ufine_centers, grid=True)
    qlinks = np.cos(np.maximum(qrate, 0.0) * (qfine[1] - qfine[0]))[..., None, None]
    ulinks = np.cos(np.maximum(urate, 0.0) * (ufine[1] - ufine[0]))[..., None, None]
    return LocalChannelFDComponents(energy, qlinks, ulinks), (qfine, ufine)


@dataclass(frozen=True)
class LocalChannelGeometricAccounting:
    """Gauge-covariant bookkeeping for a truncated local-channel basis.

    ``born_huang_MHz`` is the positive retained-space defect generated by
    channel states outside the local contraction.  It is already present in
    the positive edge-flux Hamiltonian and is therefore diagnostic rather
    than an extra potential to be added a second time.
    """

    born_huang_MHz: np.ndarray
    scalar_correction_MHz: np.ndarray
    maximum_link_singular_value: float
    minimum_born_huang_eigenvalue_MHz: float
    nonadiabatic_transport_included: bool = True


def local_channel_geometric_accounting(
    components: LocalChannelFDComponents,
    coordinates: tuple[np.ndarray, np.ndarray],
    F_MHz: np.ndarray,
    *,
    geometric_potential_MHz: np.ndarray | None = None,
    source_geometric_potential_MHz: np.ndarray | None = None,
) -> LocalChannelGeometricAccounting:
    """Audit Born--Huang transport and explicit scalar-potential changes.

    Neighbor overlaps transform as ``U_i^H S_ij U_j``.  Consequently the
    link defects ``I-S S^H`` and ``I-S^H S`` are gauge covariant and measure
    the part of the derivative norm that leaves the retained channel space.
    The edge-flux discretization already contains this positive correction.
    """
    local = np.asarray(components.local_energies_cm, dtype=float)
    if local.ndim != 3 or not np.all(np.isfinite(local)):
        raise ValueError("local channel energies must have (q,u,channel) shape")
    nq, nu, channels = local.shape
    q, u = (np.asarray(grid, dtype=float) for grid in coordinates)
    if (q.shape != (nq + 2,) or u.shape != (nu + 2,)
            or any(not np.all(np.diff(grid) > 0) or not np.allclose(
                np.diff(grid), np.diff(grid)[0]) for grid in (q, u))):
        raise ValueError("channel metric requires uniform grids including Dirichlet boundaries")
    F = np.asarray(F_MHz, dtype=float)
    if (F.shape != (nq + 2, nu + 2, 2, 2) or not np.all(np.isfinite(F))
            or not np.allclose(F, F.swapaxes(-1, -2), atol=1e-8, rtol=0)
            or np.min(np.linalg.eigvalsh(F)) <= 0):
        raise ValueError("the complete 2D metric must cover the channel grid")
    qlink = np.asarray(components.q_link_overlaps)
    ulink = np.asarray(components.u_link_overlaps)
    if (qlink.shape != (nq - 1, nu, channels, channels)
            or ulink.shape != (nq, nu - 1, channels, channels)
            or not np.all(np.isfinite(qlink)) or not np.all(np.isfinite(ulink))):
        raise ValueError("channel links must connect nearest interior sites")
    hq, hu = float(q[1] - q[0]), float(u[1] - u[0])
    qface = 0.5 * (F[:-1, 1:-1, 0, 0] + F[1:, 1:-1, 0, 0]) / hq**2
    uface = 0.5 * (F[1:-1, :-1, 1, 1] + F[1:-1, 1:, 1, 1]) / hu**2
    defect = np.zeros((nq, nu, channels, channels), dtype=complex)
    identity = np.eye(channels)
    maximum_sv = 0.0
    for i in range(nq - 1):
        for j in range(nu):
            overlap = qlink[i, j]
            maximum_sv = max(maximum_sv, float(np.linalg.svd(overlap, compute_uv=False)[0]))
            weight = float(qface[i + 1, j])
            defect[i, j] += weight * (identity - overlap @ overlap.conj().T)
            defect[i + 1, j] += weight * (identity - overlap.conj().T @ overlap)
    for i in range(nq):
        for j in range(nu - 1):
            overlap = ulink[i, j]
            maximum_sv = max(maximum_sv, float(np.linalg.svd(overlap, compute_uv=False)[0]))
            weight = float(uface[i, j + 1])
            defect[i, j] += weight * (identity - overlap @ overlap.conj().T)
            defect[i, j + 1] += weight * (identity - overlap.conj().T @ overlap)
    defect = 0.5 * (defect + defect.conj().swapaxes(-1, -2))
    minimum = float(np.min(np.linalg.eigvalsh(defect)))
    scale = max(1.0, float(np.max(np.abs(defect), initial=0.0)))
    if maximum_sv > 1 + 1e-8 or minimum < -1e-8 * scale:
        raise ValueError("channel overlaps do not define a positive Born-Huang defect")

    def scalar(field: np.ndarray | None, name: str) -> np.ndarray:
        if field is None:
            return np.zeros((nq, nu))
        values = np.asarray(field, dtype=float)
        if values.shape != (nq + 2, nu + 2) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must cover the full metric grid")
        return values[1:-1, 1:-1]

    correction = scalar(geometric_potential_MHz, "geometric scalar") - scalar(
        source_geometric_potential_MHz, "source geometric scalar",
    )
    return LocalChannelGeometricAccounting(
        born_huang_MHz=defect,
        scalar_correction_MHz=correction,
        maximum_link_singular_value=maximum_sv,
        minimum_born_huang_eigenvalue_MHz=minimum,
    )


def reassemble_local_channel_metric(
    components: LocalChannelFDComponents,
    coordinates: tuple[np.ndarray, np.ndarray],
    F_MHz: np.ndarray,
    *,
    geometric_potential_MHz: np.ndarray | None = None,
    source_geometric_potential_MHz: np.ndarray | None = None,
) -> LocalChannelDVR:
    """Build a positive edge-flux multichannel DVR with full 2D metric.

    ``F_MHz`` covers the outer Dirichlet boundary nodes as well as the
    interior channel sites. Axial edge terms use the physical local overlaps.
    The mixed term uses Hermitian covariant first derivatives constructed
    from those same links. No channel is selected or energy refitted.
    """
    local = np.asarray(components.local_energies_cm, dtype=float)
    qlink = np.asarray(components.q_link_overlaps)
    ulink = np.asarray(components.u_link_overlaps)
    if local.ndim != 3 or not np.all(np.isfinite(local)):
        raise ValueError("local channel energies must have (q,u,channel) shape")
    nq, nu, channels = local.shape
    if (qlink.shape != (nq - 1, nu, channels, channels)
            or ulink.shape != (nq, nu - 1, channels, channels)
            or not np.all(np.isfinite(qlink)) or not np.all(np.isfinite(ulink))):
        raise ValueError("channel links must connect nearest interior sites")
    q, u = (np.asarray(grid, dtype=float) for grid in coordinates)
    if (q.shape != (nq + 2,) or u.shape != (nu + 2,)
            or any(not np.all(np.diff(grid) > 0) or not np.allclose(np.diff(grid), np.diff(grid)[0])
                   for grid in (q, u))):
        raise ValueError("channel metric requires uniform grids including Dirichlet boundaries")
    F = np.asarray(F_MHz, dtype=float)
    if (F.shape != (nq + 2, nu + 2, 2, 2) or not np.all(np.isfinite(F))
            or not np.allclose(F, F.swapaxes(-1, -2), atol=1e-8, rtol=0)
            or np.min(np.linalg.eigvalsh(F)) <= 0):
        raise ValueError("the complete 2D metric must be symmetric positive definite")
    accounting = local_channel_geometric_accounting(
        components, coordinates, F,
        geometric_potential_MHz=geometric_potential_MHz,
        source_geometric_potential_MHz=source_geometric_potential_MHz,
    )
    geometric = accounting.scalar_correction_MHz
    hq, hu = float(q[1] - q[0]), float(u[1] - u[0])
    qface = 0.5 * (F[:-1, 1:-1, 0, 0] + F[1:, 1:-1, 0, 0]) / hq**2
    uface = 0.5 * (F[1:-1, :-1, 1, 1] + F[1:-1, 1:, 1, 1]) / hu**2
    diagonal = (local * CM_TO_MHZ + geometric[..., None]
                + qface[:-1, :, None] + qface[1:, :, None]
                + uface[:, :-1, None] + uface[:, 1:, None]).reshape(-1)
    size = nq * nu * channels
    rows, cols, data = [], [], []
    def append_link(a: int, b: int, overlap: np.ndarray, weight: float):
        rr, cc = np.indices((channels, channels))
        left = (a * channels + rr).ravel()
        right = (b * channels + cc).ravel()
        block = overlap.ravel()
        rows.extend((left, right))
        cols.extend((right, left))
        data.extend((-weight * block, -weight * block.conj()))

    for i in range(nq):
        for j in range(nu):
            site = i * nu + j
            if i + 1 < nq:
                append_link(site, (i + 1) * nu + j, qlink[i, j],
                            float(qface[i + 1, j]))
            if j + 1 < nu:
                append_link(site, site + 1, ulink[i, j],
                            float(uface[i, j + 1]))
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    offsite = coo_matrix((np.concatenate(data), (row, col)), shape=(size, size)).tocsr()
    hamiltonian = diags(diagonal, format="csr") + offsite
    if np.any(F[1:-1, 1:-1, 0, 1]):
        p = local_channel_covariant_momenta(components, coordinates)
        mixed = diags(np.repeat(F[1:-1, 1:-1, 0, 1].reshape(-1), channels), format="csr")
        hamiltonian += p[0] @ mixed @ p[1] + p[1] @ mixed @ p[0]
    hamiltonian = ((hamiltonian + hamiltonian.getH()) * (0.5 / CM_TO_MHZ)).tocsr()
    hamiltonian.eliminate_zeros()
    return LocalChannelDVR(
        hamiltonian, (nq, nu), channels,
        "covariant full-metric local-channel reassembly with Born-Huang transport",
    )


def local_channel_covariant_momenta(
    components: LocalChannelFDComponents, coordinates: tuple[np.ndarray, np.ndarray],
) -> tuple[csr_matrix, csr_matrix]:
    """Hermitian centered momenta transported by nearest channel overlaps."""
    nq, nu, channels = np.asarray(components.local_energies_cm).shape
    q, u = (np.asarray(grid, dtype=float) for grid in coordinates)
    if q.shape != (nq + 2,) or u.shape != (nu + 2,):
        raise ValueError("momentum grids must include the two Dirichlet boundaries")
    rows, cols, values = [[], []], [[], []], [[], []]
    for axis, overlaps, spacing in (
        (0, components.q_link_overlaps, float(q[1] - q[0])),
        (1, components.u_link_overlaps, float(u[1] - u[0])),
    ):
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("momentum grid spacing must be positive")
        expected = ((nq - 1, nu) if axis == 0 else (nq, nu - 1)) + (channels, channels)
        if overlaps.shape != expected or not np.all(np.isfinite(overlaps)):
            raise ValueError("channel link shape or coefficients are invalid")
        for i in range(overlaps.shape[0]):
            for j in range(overlaps.shape[1]):
                a, b = i * nu + j, (i + 1) * nu + j if axis == 0 else i * nu + j + 1
                rr, cc = np.indices((channels, channels))
                left = (a * channels + rr).ravel()
                right = (b * channels + cc).ravel()
                block = overlaps[i, j].ravel()
                rows[axis].extend((left, right))
                cols[axis].extend((right, left))
                values[axis].extend((-0.5j * block / spacing,
                                     0.5j * block.conj() / spacing))
    size = nq * nu * channels
    return tuple(coo_matrix(
        (np.concatenate(values[axis]),
         (np.concatenate(rows[axis]), np.concatenate(cols[axis]))),
        shape=(size, size),
    ).tocsr() for axis in (0, 1))


@dataclass(frozen=True)
class LocalChannelDVRResult:
    energies_cm: np.ndarray
    vectors: np.ndarray
    residual_norms_cm: np.ndarray

    @property
    def energies_MHz(self) -> np.ndarray:
        return self.energies_cm * CM_TO_MHZ


def solve_local_channel_dvr(model: LocalChannelDVR, *, state_count: int = 4) -> LocalChannelDVRResult:
    """Find low states of the retained multichannel Hamiltonian."""
    if type(state_count) is not int or not 1 <= state_count < model.dimension - 1:
        raise ValueError("state_count must be smaller than the finite channel dimension minus one")
    matrix = model.hamiltonian_cm
    energies, vectors = eigsh(
        matrix, k=state_count, which="SA", tol=1e-10,
        ncv=min(model.dimension, max(24, 4 * state_count + 4)),
        maxiter=max(1000, 10 * model.dimension),
        v0=np.linspace(1.0, 2.0, model.dimension),
    )
    order = np.argsort(energies)
    energies, vectors = energies[order], vectors[:, order]
    residuals = np.linalg.norm(matrix @ vectors - vectors * energies, axis=0)
    return LocalChannelDVRResult(energies, vectors, residuals)


def contract_local_channel_field(
    model: LocalChannelDVR, result: LocalChannelDVRResult, field: np.ndarray,
) -> np.ndarray:
    """Contract a site-local, channel-independent tensor over retained states.

    The returned leading axes are (left state, right state). This supplies
    rotational and dipole tensor matrix elements once a geometry-grid field
    is available; derivative terms require separate covariant link operators.
    """
    values = np.asarray(field)
    if values.shape[:len(model.grid_shape)] != model.grid_shape or not np.all(np.isfinite(values)):
        raise ValueError("site field must cover the complete interior channel grid")
    vectors = np.asarray(result.vectors)
    if vectors.ndim != 2 or vectors.shape[0] != model.dimension:
        raise ValueError("retained channel vectors disagree with the finite Hamiltonian")
    sites = int(np.prod(model.grid_shape))
    amplitudes = vectors.reshape((sites, model.channels, vectors.shape[1]))
    return np.einsum("scm,s...,scn->mn...", amplitudes.conj(),
                     values.reshape((sites,) + values.shape[len(model.grid_shape):]),
                     amplitudes, optimize=True)


def contract_local_channel_rotation(
    components: LocalChannelFDComponents,
    coordinates: tuple[np.ndarray, np.ndarray],
    result: LocalChannelDVRResult,
    F_MHz: np.ndarray,
    rho: np.ndarray,
    R_MHz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract the general Watson LAM–rotation operators over channel states.

    Returns the coefficients of the linear ``J_a`` and symmetrized quadratic
    ``J_a J_b`` terms. Both derivatives use the same covariant channel links
    as the vibrational Hamiltonian. Coordinates and fields include the outer
    Dirichlet boundary; eigenvectors cover only interior sites.
    """
    local = np.asarray(components.local_energies_cm)
    if local.ndim != 3:
        raise ValueError("local channel energies must have (q,u,channel) shape")
    nq, nu, channels = local.shape
    full = (nq + 2, nu + 2)
    F = np.asarray(F_MHz, dtype=float)
    rho = np.asarray(rho, dtype=float)
    R = np.asarray(R_MHz, dtype=float)
    if (F.shape != full + (2, 2) or rho.shape != full + (3, 2)
            or R.shape != full + (3, 3)
            or not all(np.all(np.isfinite(field)) for field in (F, rho, R))):
        raise ValueError("rotation fields must be finite and cover the full 2D grid")
    U = np.asarray(result.vectors)
    if U.ndim != 2 or U.shape[0] != nq * nu * channels:
        raise ValueError("channel vectors disagree with the interior grid")
    p = local_channel_covariant_momenta(components, coordinates)
    F = F[1:-1, 1:-1].reshape((-1, 2, 2))
    rho = rho[1:-1, 1:-1].reshape((-1, 3, 2))
    R = R[1:-1, 1:-1].reshape((-1, 3, 3))
    pu = tuple(operator @ U for operator in p)
    states = U.shape[1]
    linear = np.empty((3, states, states), dtype=complex)
    quadratic = np.empty((3, 3, states, states), dtype=complex)
    for a in range(3):
        operated = np.zeros_like(U, dtype=complex)
        for mu in range(2):
            for nu in range(2):
                left = np.repeat(F[:, mu, nu] * rho[:, a, nu], channels)
                right = np.repeat(F[:, mu, nu] * rho[:, a, mu], channels)
                operated -= p[mu] @ (left[:, None] * U)
                operated -= right[:, None] * pu[nu]
        linear[a] = U.conj().T @ operated
    for a in range(3):
        for b in range(3):
            coefficient = R[:, a, b] + np.einsum(
                "sm,smn,sn->s", rho[:, a], F, rho[:, b], optimize=True,
            )
            quadratic[a, b] = U.conj().T @ (
                np.repeat(coefficient, channels)[:, None] * U
            )
    if (np.max(np.abs(linear - linear.conj().swapaxes(-1, -2))) > 1e-6
            or np.max(np.abs(quadratic - quadratic.conj().transpose(1, 0, 3, 2))) > 1e-6):
        raise ValueError("contracted local-channel rotational operators are not Hermitian")
    return linear, quadratic


def solve_local_channel_rotational_block(
    result: LocalChannelDVRResult,
    linear_MHz: np.ndarray,
    quadratic_MHz: np.ndarray,
    J: int,
    *,
    representation: str = "Ir",
) -> np.ndarray:
    """Diagonalize one contracted Watson block in an arbitrary rotor basis."""
    return local_channel_rotational_eigensystem(
        result, linear_MHz, quadratic_MHz, J, representation=representation,
    ).energies_MHz


@dataclass(frozen=True)
class LocalChannelRotationalBlock:
    J: int
    energies_MHz: np.ndarray
    vectors: np.ndarray
    irreps: tuple[str | None, ...]


@dataclass(frozen=True)
class LocalChannelSpectralTransition:
    lower_J: int
    lower_index: int
    lower_irrep: str | None
    upper_J: int
    upper_index: int
    upper_irrep: str | None
    frequency_MHz: float
    line_strength_D2: float
    intensity: float


@dataclass(frozen=True)
class LocalChannelLineMatch:
    observation_index: int
    transition_index: int
    observed_frequency_MHz: float
    calculated_frequency_MHz: float
    residual_MHz: float


def local_channel_rotational_eigensystem(
    result: LocalChannelDVRResult,
    linear_MHz: np.ndarray,
    quadratic_MHz: np.ndarray,
    J: int,
    *,
    representation: str = "Ir",
    symmetry: LAMRotationSymmetry | None = None,
) -> LocalChannelRotationalBlock:
    """Diagonalize a contracted block and retain vectors and exact PI irreps."""
    if isinstance(J, bool) or int(J) != J or J < 0:
        raise ValueError("J must be a nonnegative integer")
    energies = np.asarray(result.energies_MHz, dtype=float)
    count = len(energies)
    linear = np.asarray(linear_MHz)
    quadratic = np.asarray(quadratic_MHz)
    if (linear.shape != (3, count, count)
            or quadratic.shape != (3, 3, count, count)
            or not np.all(np.isfinite(linear)) or not np.all(np.isfinite(quadratic))):
        raise ValueError("rotational coefficients disagree with the contracted states")
    rotor = angular_momentum_operators(J, representation)
    dimension = rotor[0].shape[0]
    block = np.kron(np.diag(energies), np.eye(dimension)).astype(complex)
    for a in range(3):
        block += np.kron(linear[a], rotor[a])
        for b in range(3):
            block += np.kron(
                quadratic[a, b], 0.5 * (rotor[a] @ rotor[b] + rotor[b] @ rotor[a]),
            )
    if np.max(np.abs(block - block.conj().T)) > 1e-5:
        raise ValueError("contracted local-channel rotational block is not Hermitian")
    if symmetry is None:
        values, vectors = np.linalg.eigh(block)
        irreps: tuple[str | None, ...] = (None,) * len(values)
    else:
        action = symmetry.contracted_action(result.vectors, J)
        symmetry_blocks = action.diagonalize(block)
        values = np.concatenate([item.energies for item in symmetry_blocks])
        vectors = np.column_stack([item.eigenvectors for item in symmetry_blocks])
        labels = tuple(item.irrep for item in symmetry_blocks for _ in item.energies)
        order = np.argsort(values, kind="stable")
        values, vectors = values[order], vectors[:, order]
        irreps = tuple(labels[int(index)] for index in order)
    return LocalChannelRotationalBlock(int(J), values.real, vectors, irreps)


def local_channel_transitions(
    blocks: tuple[LocalChannelRotationalBlock, ...],
    dipole_D: np.ndarray,
    *,
    representation: str = "Ir",
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
) -> tuple[LocalChannelSpectralTransition, ...]:
    """Generate reduced electric-dipole lines for a multichannel spectrum.

    ``dipole_D[v,v',a]`` is the body-fixed dipole surface contracted over the
    retained channel states.  The rotational factor is evaluated exactly in
    the same ``|J,K>`` convention used to build each Hamiltonian block.
    """
    if not blocks:
        return ()
    count = blocks[0].vectors.shape[0] // (2 * blocks[0].J + 1)
    dipole = np.asarray(dipole_D, dtype=complex)
    if dipole.shape != (count, count, 3) or not np.all(np.isfinite(dipole)):
        raise ValueError("contracted dipole must have (state,state,axis) shape")
    if temperature_K is not None and (not np.isfinite(temperature_K) or temperature_K <= 0):
        raise ValueError("temperature_K must be positive")
    if not np.isfinite(intensity_cut) or intensity_cut < 0:
        raise ValueError("intensity_cut must be finite and nonnegative")
    ground = min(float(np.min(item.energies_MHz)) for item in blocks)
    partition = 1.0
    if temperature_K is not None:
        partition = math.fsum(
            (2 * item.J + 1) * math.exp(
                -(float(energy) - ground) * MHZ_TO_KELVIN / temperature_K
            )
            for item in blocks for energy in item.energies_MHz
        )
    lines = []
    for lower in blocks:
        for upper in blocks:
            if upper.J < lower.J or abs(upper.J - lower.J) > 1 or upper.J + lower.J < 1:
                continue
            strength = np.zeros((len(lower.energies_MHz), len(upper.energies_MHz)))
            for axis, name in enumerate("abc"):
                rotation = dipole_transition_matrix(
                    lower.J, upper.J, name, representation,
                )
                primitive = np.kron(dipole[..., axis], rotation)
                amplitude = lower.vectors.conj().T @ primitive @ upper.vectors
                strength += np.abs(amplitude) ** 2
            for il, elower in enumerate(lower.energies_MHz):
                for iu, eupper in enumerate(upper.energies_MHz):
                    frequency = float(eupper - elower)
                    line_strength = float(strength[il, iu])
                    if frequency <= 0 or line_strength <= 0:
                        continue
                    intensity = line_strength
                    if temperature_K is not None:
                        population = math.exp(
                            -(float(elower) - ground) * MHZ_TO_KELVIN / temperature_K
                        ) / partition
                        stimulated = -math.expm1(-frequency * MHZ_TO_KELVIN / temperature_K)
                        intensity *= population * stimulated
                    if intensity < intensity_cut:
                        continue
                    lines.append(LocalChannelSpectralTransition(
                        lower.J, il, lower.irreps[il], upper.J, iu, upper.irreps[iu],
                        frequency, line_strength, intensity,
                    ))
    return tuple(sorted(lines, key=lambda item: item.frequency_MHz))


def match_local_channel_transitions(
    transitions: tuple[LocalChannelSpectralTransition, ...],
    observations: tuple[dict[str, object], ...],
    *,
    maximum_residual_MHz: float,
) -> tuple[LocalChannelLineMatch, ...]:
    """Find a global one-to-one assignment of measured multichannel lines.

    Optional ``lower_J``, ``upper_J``, ``lower_irrep`` and ``upper_irrep``
    fields are treated as exact constraints.  Frequency matching then uses a
    rectangular Hungarian assignment, avoiding greedy crossings in dense
    tunnelling multiplets.
    """
    if not np.isfinite(maximum_residual_MHz) or maximum_residual_MHz <= 0:
        raise ValueError("maximum_residual_MHz must be positive")
    if not observations:
        return ()
    if not transitions:
        raise ValueError("cannot assign observations without calculated transitions")
    penalty = maximum_residual_MHz * 1.0e6
    cost = np.full((len(observations), len(transitions)), penalty)
    observed = np.empty(len(observations))
    for row, item in enumerate(observations):
        try:
            observed[row] = float(item["frequency_MHz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each observation needs a finite frequency_MHz") from exc
        if not np.isfinite(observed[row]):
            raise ValueError("each observation needs a finite frequency_MHz")
        for column, line in enumerate(transitions):
            if any(
                key in item and item[key] is not None and item[key] != getattr(line, key)
                for key in ("lower_J", "upper_J", "lower_irrep", "upper_irrep")
            ):
                continue
            residual = abs(line.frequency_MHz - observed[row])
            if residual <= maximum_residual_MHz:
                cost[row, column] = residual
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(observations) or np.any(cost[rows, columns] >= penalty):
        missing = sorted(set(range(len(observations))) - {
            int(row) for row, column in zip(rows, columns, strict=True)
            if cost[row, column] < penalty
        })
        raise ValueError(f"no unique calculated assignment for observations {missing}")
    return tuple(
        LocalChannelLineMatch(
            int(row), int(column), float(observed[row]),
            transitions[int(column)].frequency_MHz,
            transitions[int(column)].frequency_MHz - float(observed[row]),
        )
        for row, column in sorted(zip(rows, columns, strict=True))
    )
