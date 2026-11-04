"""Reusable, spectrally preconditioned deflated response equations."""

from __future__ import annotations
from inspect import signature
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

_TOLERANCE = "rtol" if "rtol" in signature(gmres).parameters else "tol"


class DeflatedResponseSolver:
    """Solve (H-E)x=b outside an entire degenerate eigenspace.

    Known nonresonant Ritz roots form an exact spectral part of the
    preconditioner. A projected diagonal inverse treats its complement.
    Residual acceptance always uses the original, unpreconditioned equation.
    """

    def __init__(self, action, diagonal, energies, vectors, energy, cluster, *, precondition=True):
        self.action = action
        self.diagonal = np.asarray(diagonal, float)
        self.energies = np.asarray(energies, float)
        self.vectors = np.asarray(vectors, complex)
        self.energy = float(energy)
        self.cluster = np.asarray(cluster, int)
        self.dim = len(self.diagonal)
        self.counts = dict(solves=0, iterations=0, matvecs=0, warm_starts=0)
        if (
            self.vectors.shape != (self.dim, len(self.energies))
            or not len(self.cluster)
            or not np.all(np.isfinite(self.diagonal))
            or not np.all(np.isfinite(self.energies))
        ):
            raise ValueError("invalid deflated response eigenspace")
        self.resonant = self.vectors[:, self.cluster]
        self.scale = max(1.0, abs(self.energy))
        keep = np.ones(len(self.energies), bool)
        keep[self.cluster] = False
        self.gaps = self.energies[keep] - self.energy
        self.inverse_energies = np.zeros(len(self.energies))
        self.inverse_energies[keep] = 1.0 / self.gaps
        if np.any(abs(self.gaps) <= max(1.0e-10, 128 * np.spacing(self.scale))):
            raise ValueError("incomplete degenerate response eigenspace")
        floor = max(
            float(np.min(abs(self.gaps), initial=np.inf)) if len(self.gaps) else 0.0,
            np.sqrt(np.finfo(float).eps) * max(1.0, float(np.max(abs(self.diagonal), initial=0.0))),
        )
        gaps = self.diagonal - self.energy
        self.inverse_diagonal = 1.0 / np.where(
            abs(gaps) >= floor, gaps, np.where(gaps < 0, -floor, floor)
        )
        self.operator = LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=complex)
        self.preconditioner = (
            LinearOperator((self.dim, self.dim), matvec=self.precondition, dtype=complex)
            if precondition
            else None
        )
        self.warm = {}

    def project(self, x):
        return self.resonant @ (self.resonant.conj().T @ x)

    def matvec(self, x):
        self.counts["matvecs"] += 1
        return self.action(x) - self.energy * x + self.scale * self.project(x)

    def precondition(self, x):
        normal = x - self.vectors @ (self.vectors.conj().T @ x)
        normal = self.inverse_diagonal * normal
        result = normal - self.vectors @ (self.vectors.conj().T @ normal)
        if len(self.gaps):
            result += self.vectors @ (self.inverse_energies * (self.vectors.conj().T @ x))
        return result + self.project(x) / self.scale

    def solve(self, rhs, *, key=None, warm_start=None, rtol=1.0e-10, atol=1.0e-11):
        rhs = np.asarray(rhs, complex)
        rhs = rhs - self.project(rhs)
        if rhs.shape != (self.dim,) or not np.all(np.isfinite(rhs)):
            raise ValueError("invalid response right-hand side")
        self.counts["solves"] += 1
        guess = self.warm.get(key) if warm_start is None else warm_start
        if guess is not None:
            guess = np.asarray(guess, complex)
            if guess.shape != rhs.shape:
                guess = None
            else:
                guess = guess - self.project(guess)
                if np.linalg.norm(self.matvec(guess) - rhs) >= np.linalg.norm(rhs):
                    guess = None
                else:
                    self.counts["warm_starts"] += 1

        def iteration(_):
            self.counts["iterations"] += 1

        if np.linalg.norm(rhs) <= atol:
            solution = np.zeros_like(rhs)
            info = 0
        else:
            solution, info = gmres(
                self.operator,
                rhs,
                M=self.preconditioner,
                x0=guess,
                **{_TOLERANCE: rtol},
                atol=atol,
                restart=min(100, self.dim),
                maxiter=100,
                callback=iteration,
                callback_type="legacy",
            )
        solution -= self.project(solution)
        residual = float(np.linalg.norm(self.matvec(solution) - rhs))
        if info or residual > max(atol, rtol * np.linalg.norm(rhs)) * 10:
            raise RuntimeError("analytic Sternheimer response did not converge")
        if key is not None:
            self.warm[key] = solution.copy()
        return solution
