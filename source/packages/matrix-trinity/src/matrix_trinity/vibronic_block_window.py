"""Positive block-spectrum proposals for a coupled accepting-state basis.

Reuse resident positive spectral shifts to condition configurations on an
energy window. This selects candidates only: the block spectrum neither bounds
their coupled-Hamiltonian importance nor certifies a molecular rate.
"""
import numpy as np


class PositiveBlockWindowProposal:
    """Backward Gaussian messages on arbitrary finite independent block lines.

    Energies are excitations above each block ground state. Original level
    labels are retained, including when exact-zero weights or unreachable
    high-energy levels are omitted. No FFT or negative-density clipping is used.
    """
    def __init__(self, lines, *, target_cm, sigma_cm, bin_width_cm=.04,
                 gaussian_tail_sigma=12., maximum_workspace_bytes=512*1024**2):
        from matrix_rovib.statistical_convolution import convolve_discrete_measure
        values = (target_cm, sigma_cm, bin_width_cm, gaussian_tail_sigma)
        if (not all(np.isscalar(v) and np.isreal(v) and np.isfinite(v) for v in values)
                or target_cm < 0 or min(sigma_cm, bin_width_cm) <= 0 or gaussian_tail_sigma < 8):
            raise ValueError('nonnegative target, positive Gaussian/grid widths and tail at least eight sigma required')
        self.target = float(target_cm); self.sigma = float(sigma_cm); self.delta = float(bin_width_cm)
        self.maximum_workspace_bytes = maximum_workspace_bytes
        radius = self.sigma*gaussian_tail_sigma; cutoff = self.target+2*radius
        lines = tuple(lines)
        if not lines: raise ValueError('nonempty block spectra required')
        count = int(np.ceil(cutoff/self.delta))+1
        required = 8*count*(len(lines)+5)
        if count >= 9_999_998 or required > maximum_workspace_bytes:
            raise MemoryError('backward block messages exceed explicit grid/workspace budget')
        self.grid = -radius+np.arange(count)*self.delta
        self.lines = []; total_strength = 1.
        for energies, weights in lines:
            e, w = np.asarray(energies), np.asarray(weights)
            if (e.ndim != 1 or not len(e) or w.shape != e.shape
                    or any(np.iscomplexobj(a) or not np.all(np.isfinite(a)) for a in (e, w))
                    or np.any(e < 0) or np.any(w < 0) or not np.any(w > 0)):
                raise ValueError('finite nonnegative block excitation energies and strengths required')
            total_strength *= float(w.sum())
            ids = np.flatnonzero((e <= cutoff) & (w > 0))
            if not len(ids): raise ValueError('no reachable positive block strength')
            self.lines.append((e[ids].astype(float), w[ids].astype(float), ids))
        self.messages = [None]*(len(lines)+1)
        peak = 1/(self.sigma*np.sqrt(2*np.pi))
        self.messages[-1] = peak*np.exp(-.5*(self.grid/self.sigma)**2)
        for i in range(len(lines)-1, -1, -1):
            e, w, _ = self.lines[i]
            self.messages[i] = convolve_discrete_measure(self.messages[i+1], self.delta, e, w,
                                                         algorithm='positive_direct')[:count].copy()
        self.diagnostics = dict(blocks=len(lines), bin_width_cm=self.delta,
            root_window_density_per_cm=float(np.interp(self.target, self.grid, self.messages[0])),
            maximum_line_transport_cm=(len(lines)+1)*self.delta,
            gaussian_tail_density_bound_per_cm=total_strength*peak*np.exp(-.5*gaussian_tail_sigma**2),
            additional_workspace_estimate_bytes=required, positive_convolution=True,
            coupled_basis_convergence_certified=False, full_molecular_rate_converged=False,
            scope='independent-block proposal; coupled importance and basis growth require separate checks')

    def sample(self, uniforms, *, batch_size=128):
        """Sample backward-conditioned prefixes; return original block labels.

        The binned messages are a proposal. Selected configurations can be
        scored subsequently with their exact unbinned block energies/weights.
        """
        u = np.asarray(uniforms)
        if (u.ndim != 2 or u.shape[1] != len(self.lines) or np.iscomplexobj(u)
                or not np.all(np.isfinite(u)) or np.any(u < 0) or np.any(u >= 1)
                or not isinstance(batch_size, int) or not 1 <= batch_size <= 1024):
            raise ValueError('uniform draws in [0,1) and sample batch size between one and 1024 required')
        needed = (self.diagnostics['additional_workspace_estimate_bytes']
                  +32*batch_size*max(len(e) for e, _, _ in self.lines)+16*u.size)
        if needed > self.maximum_workspace_bytes:
            raise MemoryError('conditional block sampling exceeds explicit workspace budget')
        result = np.empty(u.shape, dtype=np.int64)
        for start in range(0, len(u), batch_size):
            stop = min(len(u), start+batch_size); remaining = np.full(stop-start, self.target)
            for i, (e, w, labels) in enumerate(self.lines):
                x = remaining[:, None]-e[None, :]
                conditional = np.interp(x.ravel(), self.grid, self.messages[i+1], left=0., right=0.).reshape(x.shape)
                conditional *= w[None, :]
                cdf = np.cumsum(conditional, axis=1); total = cdf[:, -1]
                if np.any(total <= 0): raise ArithmeticError('positive block proposal lost its conditional mass')
                chosen = np.sum(cdf <= (u[start:stop, i]*total)[:, None], axis=1)
                chosen = np.minimum(chosen, len(e)-1)
                result[start:stop, i] = labels[chosen]; remaining -= e[chosen]
        return result

    def exact_window_contributions(self, configurations):
        """Unnormalized unbinned block weights times Gaussian density.

        Summing this over UNIQUE retained configurations measures their exact
        contribution to the *block* model. It does not certify a coupled basis.
        """
        rows = np.asarray(configurations)
        if (rows.ndim != 2 or rows.shape[1] != len(self.lines) or rows.dtype.kind not in 'iu'
                or len(np.unique(rows, axis=0)) != len(rows)):
            raise ValueError('unique integer configurations in the retained block spectra required')
        energy = np.zeros(len(rows)); log_weight = np.zeros(len(rows))
        for i, (e, w, labels) in enumerate(self.lines):
            pos = np.searchsorted(labels, rows[:, i]); safe = np.minimum(pos, len(labels)-1)
            if np.any(pos >= len(labels)) or np.any(labels[safe] != rows[:, i]):
                raise ValueError('configuration is outside the reachable positive block spectra')
            energy += e[pos]; log_weight += np.log(w[pos])
        return np.exp(log_weight-.5*((energy-self.target)/self.sigma)**2)/(self.sigma*np.sqrt(2*np.pi))
