"""Resident FP32 closed-shell SCC accelerator with an explicit FP64 boundary.

Geometry preparation and the final finite-temperature SCF/analytic gradient
use the existing calculator. This module supplies a zero-temperature SP2 SCC
predictor on Apple MPS. It never labels the host certification as GPU work.
The equations and packed endpoint operators are shared with multipole.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np


@dataclass
class MPSExecution:
    requested: str = "mps"
    actual: str = "none"
    precision: str = "float32"
    scc_iterations: int = 0
    sp2_products: int = 0
    fock_evaluations: int = 0
    scalar_synchronizations: int = 0
    host_array_uploads: int = 0
    host_array_downloads: int = 0
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0
    resident_bytes: int = 0
    context_reused: bool = False
    buffer_reuses: int = 0
    device_allocations: int = 0  # Retained buffers; excludes operator temporaries.
    incremental_density: bool = False
    operator_pairs: int = 0
    full_operator_pairs: int = 0
    preparation_seconds: float = 0.0
    iteration_seconds: float = 0.0
    fixed_point_residual: float | None = None
    orthogonalizer_residual: float | None = None
    orthogonalizer_kind: str = "provided"
    density_seed_source: str = "gpu"
    gpu_seed_residual: float | None = None
    cpu_seed_residual: float | None = None
    seed_selection_seconds: float = 0.0
    accepted_predictor: bool = False
    fallback_reason: str | None = None
    selection_reason: str | None = None
    final_certification: str = "cpu-fp64-scf-and-analytic-gradient"

    def to_dict(self):
        return asdict(self)


def pairwise_sum(values):
    """Balanced FP32 reduction; avoid a long sequential energy accumulation."""
    import torch
    x = values.reshape(-1)
    if x.numel() == 0:
        return values.new_zeros(())
    while x.numel() > 1:
        if x.numel() % 2:
            x = torch.cat((x, x.new_zeros(1)))
        x = x[::2] + x[1::2]
    return x[0]


def cholesky_orthogonalizer(overlap):
    """Use existing LAPACK Cholesky/triangular solves for X.T S X = I."""
    from scipy.linalg import cholesky, solve_triangular
    factor = cholesky(overlap, lower=True, check_finite=True)
    return solve_triangular(factor.T, np.eye(len(overlap)), lower=False,
                            check_finite=False)


class ResidentSCC:
    """Packed GFN2 SCC/AES/AXC/D4 functional in explicit FP32 tensor algebra.

    ``device='cpu'`` is provided only for equation validation. Production
    acceleration requires MPS and raises if that device is unavailable.
    """

    def upload(self, value, integer=False):
        array = np.asarray(value, dtype=np.int64 if integer else np.float32)
        result = self.torch.tensor(array, device=self.device)
        self.report.host_array_uploads += 1
        self.report.device_allocations += 1
        self.report.uploaded_bytes += array.nbytes
        self.report.resident_bytes += result.numel()*result.element_size()
        return result

    def __init__(self, **geometry):
        self._geometry_buffers = {}
        self._integer_geometry = None
        self._runtime_buffers = {}
        self._previous_device_density = None
        self._certified_host_density = None
        self._certified_host_orbitals = None
        self._geometry_centers = None
        self.eye = None
        self.update(**geometry)

    def runtime_upload(self, name, value):
        """Update an existing workspace without allocating a temporary MPS tensor."""
        array = np.ascontiguousarray(value, dtype=np.float32)
        buffer = self._runtime_buffers.get(name)
        if buffer is None or tuple(buffer.shape) != array.shape:
            buffer = self.upload(array)
            self._runtime_buffers[name] = buffer
        else:
            buffer.copy_(self.torch.from_numpy(array))
            self.report.host_array_uploads += 1
            self.report.uploaded_bytes += array.nbytes
            self.report.resident_bytes += array.nbytes
            self.report.buffer_reuses += 1
        return buffer

    def update(self, *, core, overlap, operators, partition, valence,
                 interaction, anisotropic_parameters, anisotropic_geometry,
                 d4_prepared, device="mps"):
        import torch
        from .multipole import _packed_endpoint_operators

        start = perf_counter()
        if device not in {"mps", "cpu"}:
            raise ValueError("resident SCC device must be mps or cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is unavailable")
        self.torch, self.device = torch, device
        self.report = MPSExecution(actual=f"torch-{device}", context_reused=bool(self._geometry_buffers))
        centers = np.asarray(operators.centers, dtype=float)
        if (self._geometry_centers is not None
                and (centers.shape != self._geometry_centers.shape
                     or np.max(np.abs(centers-self._geometry_centers)) > 0.25)):
            self._previous_device_density = self._certified_host_density = None
            self._certified_host_orbitals = None
        self._geometry_centers = centers.copy()
        previous_off = getattr(self, "off", None)
        self.n = len(core)
        self.natoms = len(valence)

        # Pack the immutable geometry into two uploads rather than allocating
        # one MPS buffer per parameter/operator. Views below share storage.
        upload = lambda value, integer=False: np.asarray(
            value, dtype=np.int64 if integer else np.float32)
        rows, columns = np.tril_indices(self.n)
        # Reuse the exact support of the prepared integrals: no new cutoff or
        # approximation is introduced. Zero operators contribute neither to
        # moments nor to the density derivative of the energy.
        core_pair = np.asarray(core)[rows, columns]
        active = ((operators.pair_overlap != 0)
                  | np.any(operators.pair_dipole != 0, axis=1)
                  | np.any(operators.pair_second != 0, axis=(1, 2))
                  | (core_pair != 0))
        pairs = np.flatnonzero(active)
        self.report.full_operator_pairs = len(rows)
        rows, columns = rows[pairs], columns[pairs]
        self.report.operator_pairs = len(rows)
        self.rows, self.columns = upload(rows, True), upload(columns, True)
        self.off = self.rows != self.columns
        self.atom_l = upload(operators.atom_index[rows], True)
        self.atom_r = upload(operators.atom_index[columns], True)
        shell = np.asarray(partition.orbital_shell)
        self.shell_l, self.shell_r = upload(shell[rows], True), upload(shell[columns], True)
        self.reference_shell = upload(partition.reference_population)
        self.valence = upload(valence)
        self.interaction = upload(interaction)
        self.third = upload(partition.third_order_hartree)
        self.s = upload(operators.pair_overlap[pairs])
        self.core_pair = upload(core_pair[pairs])
        self.overlap = upload(overlap)
        self.core = upload(core)
        dl, _, ql, dr, _, qr = _packed_endpoint_operators(operators, pair_indices=pairs)
        self.dl, self.dr, self.ql, self.qr = map(upload, (dl, dr, ql, qr))
        self.dxc = upload(anisotropic_parameters.dipole_xc)
        self.qxc = upload(anisotropic_parameters.quadrupole_xc)
        geom = anisotropic_geometry
        self.al, self.ar = upload(geom.left, True), upload(geom.right, True)
        self.r = upload(geom.displacement)
        self.r2, self.f3, self.f5 = map(upload, (geom.distance2, geom.f3, geom.f5))
        self.outer = self.r[:, :, None] * self.r[:, None, :]
        self.dw, self.dref, self.dmask = map(upload, (
            d4_prepared.weights, d4_prepared.references, d4_prepared.mask))
        self.deff, self.dhard, self.dqref, self.dfreq = map(upload, (
            d4_prepared.effective, d4_prepared.hardness,
            d4_prepared.reference_charges, d4_prepared.frequency_weights))
        if hasattr(d4_prepared, "rows"):
            di, dj = d4_prepared.rows, d4_prepared.columns
            factor = d4_prepared.pair_factor
        else:
            di, dj = np.tril_indices(self.natoms, -1)
            factor = d4_prepared.pair_factor[di, dj]
        self.di, self.dj = upload(di, True), upload(dj, True)
        self.dfactor = upload(factor)
        for integer in (False, True):
            arrays = [(name, value) for name, value in vars(self).items()
                      if not name.startswith("_") and isinstance(value, np.ndarray)
                      and (value.dtype.kind in "biu") == integer]
            if not arrays:
                continue
            packed = np.concatenate([value.reshape(-1) for _, value in arrays])
            layout = tuple((name, array.shape, array.dtype.str) for name, array in arrays)
            previous = self._geometry_buffers.get(integer)
            if previous is None or previous[0] != layout:
                buffer = self.upload(packed, integer=integer)
                self._geometry_buffers[integer] = (layout, buffer)
            else:
                buffer = previous[1]
                unchanged = integer and self._integer_geometry is not None and np.array_equal(packed, self._integer_geometry)
                if not unchanged:
                    buffer.copy_(torch.from_numpy(packed))
                    self.report.host_array_uploads += 1
                    self.report.uploaded_bytes += packed.nbytes
                self.report.buffer_reuses += 1
                self.report.resident_bytes += packed.nbytes
            if integer:
                if self._integer_geometry is not None and not np.array_equal(packed, self._integer_geometry):
                    self._previous_device_density = self._certified_host_density = None
                    self._certified_host_orbitals = None
                self._integer_geometry = packed.copy()
            offset = 0
            for name, array in arrays:
                view = buffer[offset:offset+array.size].reshape(array.shape)
                if array.dtype == bool:
                    if previous_off is not None and tuple(previous_off.shape) == array.shape:
                        previous_off.copy_(view)
                        view = previous_off
                    else:
                        view = view.bool()
                        self.report.device_allocations += 1
                setattr(self, name, view)
                offset += array.size
        if self.eye is None or self.eye.shape[0] != self.n:
            self.eye = torch.eye(self.n, dtype=torch.float32, device=device)
            self.report.device_allocations += 1
        self.report.resident_bytes += self.eye.numel()*self.eye.element_size()
        self.report.preparation_seconds = perf_counter() - start

    def scalar(self, value):
        self.report.scalar_synchronizations += 1
        return float(value.item())

    def download(self, value):
        self.report.host_array_downloads += 1
        self.report.downloaded_bytes += value.numel() * value.element_size()
        return value.detach().cpu().numpy().astype(np.float64)

    def certify_host_density(self, density, orbitals=None):
        """Anchor GPU changes to the independently converged FP64 density."""
        self._certified_host_density = np.array(density, dtype=float, copy=True)
        self._previous_device_density = self._pending_device_density
        self._certified_host_orbitals = (
            tuple(np.array(value, dtype=float, copy=True) for value in orbitals)
            if orbitals is not None else None
        )

    def moments(self, density):
        p = density[self.rows, self.columns]
        pr = p * self.off
        qs, q = self.reference_shell.clone(), self.valence.clone()
        mu = p.new_zeros((self.natoms, 3))
        theta = p.new_zeros((self.natoms, 3, 3))
        qs.index_add_(0, self.shell_l, -p * self.s)
        qs.index_add_(0, self.shell_r, -pr * self.s)
        q.index_add_(0, self.atom_l, -p * self.s)
        q.index_add_(0, self.atom_r, -pr * self.s)
        mu.index_add_(0, self.atom_l, p[:, None] * self.dl)
        mu.index_add_(0, self.atom_r, pr[:, None] * self.dr)
        theta.index_add_(0, self.atom_l, p[:, None, None] * self.ql)
        theta.index_add_(0, self.atom_r, pr[:, None, None] * self.qr)
        return qs, q, mu, theta

    def fock_energy(self, density):
        """Analytic density derivative of exactly the prepared GFN2 functional."""
        torch = self.torch
        qs, q, mu, theta = self.moments(density)
        v = self.interaction @ qs
        shell_pot = v + self.third * qs.square()
        energy = pairwise_sum(density * self.core)
        energy = energy + pairwise_sum(0.5 * qs * v + self.third * qs**3 / 3.0)
        mp = 2 * self.dxc[:, None] * mu
        tp = 2 * self.qxc[:, None, None] * theta
        qp = torch.zeros_like(q)
        energy = energy + pairwise_sum(self.dxc[:, None] * mu.square())
        energy = energy + pairwise_sum(self.qxc[:, None, None] * theta.square())
        l, r, vec = self.al, self.ar, self.r
        lm = (mu[l] * vec).sum(1)
        rm = (mu[r] * vec).sum(1)
        lt = (theta[l] * self.outer).sum((1, 2))
        rt = (theta[r] * self.outer).sum((1, 2))
        energy = energy + pairwise_sum(
            self.f3 * (-q[l] * rm + q[r] * lm)
            + self.f5 * (q[l] * rt + q[r] * lt - 3 * lm * rm
                        + (mu[l] * mu[r]).sum(1) * self.r2))
        qp.index_add_(0, l, -self.f3 * rm + self.f5 * rt)
        qp.index_add_(0, r, self.f3 * lm + self.f5 * lt)
        mp.index_add_(0, l, (self.f3*q[r])[:, None]*vec + self.f5[:, None]
                      * (-3*vec*rm[:, None] + mu[r]*self.r2[:, None]))
        mp.index_add_(0, r, (-self.f3*q[l])[:, None]*vec + self.f5[:, None]
                      * (-3*vec*lm[:, None] + mu[l]*self.r2[:, None]))
        tp.index_add_(0, l, (self.f5*q[r])[:, None, None]*self.outer)
        tp.index_add_(0, r, (self.f5*q[l])[:, None, None]*self.outer)
        # D4 physical-charge response, with the original charge convention.
        denominator = self.deff + q
        # Invalid domains propagate NaNs and reject the predictor; no clipping
        # is allowed to change the energy model.
        denominator = torch.where(denominator > 0, denominator,
                                  torch.full_like(denominator, float("nan")))
        numerator = self.deff[:, None] + self.dqref
        exp_inner = torch.exp(2*self.dhard[:, None]*(1-numerator/denominator[:, None]))
        scale = torch.exp(3*(1-exp_inner))*self.dmask
        ds = scale*(-3*exp_inner)*(2*self.dhard[:, None]*numerator/denominator[:, None]**2)
        alpha = ((self.dw*scale)[:, :, None]*self.dref).sum(1)
        dalpha = ((self.dw*ds)[:, :, None]*self.dref).sum(1)
        i, j, factor = self.di, self.dj, self.dfactor*(3/np.pi)
        energy = energy - pairwise_sum(
            (alpha[i]*alpha[j]*self.dfreq).sum(1)*factor)
        dpot = torch.zeros_like(q)
        dpot.index_add_(0, i, -(dalpha[i]*alpha[j]*self.dfreq).sum(1)*factor)
        dpot.index_add_(0, j, -(alpha[i]*dalpha[j]*self.dfreq).sum(1)*factor)

        def endpoint(atom, dip, quad):
            return -qp[atom]*self.s + (mp[atom]*dip).sum(1) + (tp[atom]*quad).sum((1, 2))

        fp = self.core_pair - 0.5*self.s*(shell_pot[self.shell_l]
             + shell_pot[self.shell_r] + dpot[self.atom_l] + dpot[self.atom_r])
        fp = fp + 0.5*(endpoint(self.atom_l, self.dl, self.ql)
                       + endpoint(self.atom_r, self.dr, self.qr))
        fock = torch.zeros_like(density)
        fock[self.rows, self.columns] = fp
        fock[self.columns, self.rows] = fp
        self.report.fock_evaluations += 1
        return fock, energy

    def sp2(self, fock, occupied, max_iterations=80):
        """Trace-correcting SP2; branch and convergence decisions stay on device."""
        torch = self.torch
        diag = fock.diagonal()
        radius = fock.abs().sum(1) - diag.abs()
        low, high = (diag-radius).min(), (diag+radius).max()
        p = (high*self.eye - fock)/(high-low)
        # Blocks of four steps reduce synchronizations; convergence freezes
        # the projector on device while the remaining block steps complete.
        done = torch.zeros((), dtype=torch.bool, device=self.device)
        for iteration in range(max_iterations):
            p2 = p @ p
            err = (p2-p).abs().max()
            done = done | ((err < 3e-6) & ((torch.trace(p)-occupied).abs() < 2e-4))
            candidate = torch.where(torch.trace(p) > occupied, p2, 2*p-p2)
            candidate = 0.5*(candidate + candidate.T)
            p = torch.where(done, p, candidate)
            self.report.sp2_products += 1
            if iteration % 4 == 3 and self.scalar(done):
                return p
        if (self.scalar((p@p-p).abs().max()) > 2e-5
                or abs(self.scalar(torch.trace(p))-occupied) > 2e-4):
            raise RuntimeError("FP32 SP2 idempotency did not converge")
        return p

    def run(self, density, orthogonalizer, electrons, *, max_iterations=60,
            tolerance=2e-5):
        """Generate a stationary GPU predictor; host SCF always certifies it."""
        torch = self.torch
        if electrons % 2 or not 0 < electrons//2 < self.n:
            raise ValueError("resident SP2 requires a nontrivial closed shell")
        start = perf_counter()
        x = self.runtime_upload("orthogonalizer", orthogonalizer)
        defect = (x.T @ self.overlap @ x - self.eye).abs().max()
        self.report.orthogonalizer_residual = self.scalar(defect)
        if self.report.orthogonalizer_residual > 3e-4:
            raise RuntimeError("overlap conditioning is unsuitable for FP32")
        p = self.runtime_upload("density", density)
        focks, errors = [], []
        last_residual = float("inf")
        for step in range(max_iterations):
            f, _ = self.fock_energy(p)
            residual_matrix = f @ p @ self.overlap - self.overlap @ p @ f
            errors.append(residual_matrix.reshape(-1))
            focks.append(f)
            focks, errors = focks[-6:], errors[-6:]
            if len(focks) > 1:
                stacked = torch.stack(errors)
                gram = stacked @ stacked.T
                # Scale DIIS before solving; fixed regularization prevents
                # FP32 cancellation in nearly dependent history vectors.
                gram = gram / gram.abs().max().clamp_min(1e-20)
                k = len(focks)
                block = p.new_zeros((k+1, k+1))
                block[:k, :k] = gram + 1e-6*torch.eye(k, device=self.device)
                block[k, :k] = -1
                block[:k, k] = -1
                rhs = p.new_zeros(k+1)
                rhs[k] = -1
                weights = torch.linalg.solve(block, rhs)[:k]
                f = (torch.stack(focks)*weights[:, None, None]).sum(0)
            fo = x.T @ f @ x
            projector = self.sp2(0.5*(fo+fo.T), electrons//2)
            updated = 2*x @ projector @ x.T
            last_residual = self.scalar((updated-p).abs().max())
            self.report.scc_iterations += 1
            if not np.isfinite(last_residual):
                raise RuntimeError("nonfinite FP32 SCC predictor")
            if last_residual < tolerance:
                p = updated
                self.report.accepted_predictor = True
                break
            p = 0.7*updated + 0.3*p
        self.report.fixed_point_residual = last_residual
        self.report.iteration_seconds = perf_counter() - start
        if not self.report.accepted_predictor:
            raise RuntimeError("FP32 SCC predictor iteration limit")
        self._pending_device_density = p
        if (self._previous_device_density is not None
                and self._certified_host_density is not None):
            # Subtract on the device before downloading: retain the low bits
            # of the certified host density instead of replacing them by the
            # absolute FP32 projector on every nearby geometry.
            self.report.incremental_density = True
            return self._certified_host_density + self.download(p-self._previous_device_density)
        return self.download(p)
