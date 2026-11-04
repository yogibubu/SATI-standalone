"""Vertical singlet TDA with imported canonical orbitals and an Ohno kernel.

No backend two-electron integrals, primitive basis, or xTB orbital shifts are
used. All energies are Hartree, coordinates are bohr, and dipoles are atomic
units. The initial implementation accepts real restricted closed shells.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt, pi
from pathlib import Path

import numpy as np
from matrix_numerics import IterativeEigenResult
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from .electronic_state import _readonly_array
from .sf_tda import HARTREE_TO_EV, grimme_stda_atomic_hardnesses
from .tda_virtual_space import ProjectedTDAVirtualSpace


@dataclass(frozen=True)
class ImportedTDAReference:
    """Backend-neutral AO data; occupations count spatial electrons (0 or 2).

    Coefficients may be rectangular, e.g. after removing AO linear dependencies.
    Every matrix must use the same AO ordering and normalization. ``dipole_ao``
    contains the three components of the electronic dipole operator, not nuclear
    contributions. Missing dipoles leave transition properties unavailable.
    Optional radial_second_moment_ao contains positive r^2 in bohr squared,
    with the same origin as the dipoles, for the default K filter and size diagnostics.
    """

    model_id: str
    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    ao_atom: np.ndarray
    overlap: np.ndarray
    fock: np.ndarray
    coefficients: np.ndarray
    orbital_energies_hartree: np.ndarray
    occupations: np.ndarray
    dipole_ao: np.ndarray | None = None
    radial_second_moment_ao: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("an imported reference requires a nonempty model_id")
        for name in ("atomic_numbers", "ao_atom"):
            raw = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(raw)) or np.any(raw != np.floor(raw)):
                raise ValueError(f"{name} must contain integer indices or atomic numbers")
            object.__setattr__(self, name, _readonly_array(raw, dtype=np.int64))
        for name in ("coordinates_bohr", "overlap", "fock", "coefficients",
                     "orbital_energies_hartree", "occupations"):
            if np.iscomplexobj(getattr(self, name)):
                raise ValueError("imported TDA currently requires real orbitals and matrices")
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        n = len(self.atomic_numbers) if self.atomic_numbers.ndim == 1 else 0
        if not n or self.coordinates_bohr.shape != (n, 3):
            raise ValueError("atomic numbers and coordinates must describe a nonempty molecule")
        if np.any(self.atomic_numbers < 1) or np.any(self.atomic_numbers > 94):
            raise ValueError("the resident hardness table supports atomic numbers 1 through 94")
        if self.overlap.ndim != 2 or self.overlap.shape[0] == 0:
            raise ValueError("overlap must be a nonempty square AO matrix")
        nao = self.overlap.shape[0]
        if self.overlap.shape != (nao, nao) or self.fock.shape != (nao, nao):
            raise ValueError("overlap and Fock must be equally sized square AO matrices")
        if self.coefficients.ndim != 2 or self.coefficients.shape[0] != nao:
            raise ValueError("coefficients must have shape (nao, nmo)")
        nmo = self.coefficients.shape[1]
        if not nmo or self.orbital_energies_hartree.shape != (nmo,) or self.occupations.shape != (nmo,):
            raise ValueError("energies and occupations must have length nmo")
        if self.ao_atom.shape != (nao,) or np.any(self.ao_atom < 0) or np.any(self.ao_atom >= n):
            raise ValueError("ao_atom must assign each AO to a zero-based atom index")
        for matrix in (self.overlap, self.fock):
            if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0):
                raise ValueError("AO overlap and Fock must be symmetric")
        # A semidefinite S is allowed when the backend has removed dependent AOs.
        values = np.linalg.eigvalsh(self.overlap)
        if values[0] < -1e-10 * max(1.0, values[-1]):
            raise ValueError("AO overlap must be positive semidefinite")
        c = self.coefficients
        if not np.allclose(c.T @ self.overlap @ c, np.eye(nmo), atol=2e-7, rtol=0):
            raise ValueError("imported MOs must be orthonormal in the AO overlap metric")
        if not np.allclose(self.fock @ c, (self.overlap @ c) * self.orbital_energies_hartree,
                           atol=2e-7, rtol=2e-7):
            raise ValueError("imported energies and coefficients must solve F C = S C epsilon")
        occupied = np.isclose(self.occupations, 2, atol=1e-8, rtol=0)
        virtual = np.isclose(self.occupations, 0, atol=1e-8, rtol=0)
        if not np.all(occupied | virtual) or not np.any(occupied) or not np.any(virtual):
            raise ValueError("a restricted closed shell with occupied and virtual MOs is required")
        if self.dipole_ao is not None:
            if np.iscomplexobj(self.dipole_ao):
                raise ValueError("dipole AO matrices must be real")
            dipoles = _readonly_array(self.dipole_ao)
            if dipoles.shape != (3, nao, nao) or not np.allclose(
                    dipoles, dipoles.swapaxes(1, 2), atol=1e-10, rtol=0):
                raise ValueError("dipole_ao must contain three symmetric AO matrices")
            object.__setattr__(self, "dipole_ao", dipoles)
        if self.radial_second_moment_ao is not None:
            if np.iscomplexobj(self.radial_second_moment_ao):
                raise ValueError("radial second moments must be real")
            radial = _readonly_array(self.radial_second_moment_ao)
            if radial.shape != (nao, nao) or not np.allclose(radial, radial.T, atol=1e-10, rtol=0):
                raise ValueError("radial_second_moment_ao must be a symmetric AO matrix")
            object.__setattr__(self, "radial_second_moment_ao", radial)

    @classmethod
    def from_npz(cls, path: str | Path) -> ImportedTDAReference:
        """Read named arrays exported by a backend, without pickle or ERIs.

        Archive keys equal the constructor field names; dipole_ao and
        radial_second_moment_ao are optional. model_id is scalar Unicode.
        """
        names = ("atomic_numbers", "coordinates_bohr", "ao_atom", "overlap", "fock",
                 "coefficients", "orbital_energies_hartree", "occupations")
        with np.load(path, allow_pickle=False) as data:
            model = data["model_id"]
            if model.shape != () or model.dtype.kind != "U":
                raise ValueError("NPZ model_id must be a scalar Unicode string")
            return cls(model_id=str(model.item()), **{name: data[name] for name in names},
                       dipole_ao=data["dipole_ao"] if "dipole_ao" in data else None,
                       radial_second_moment_ao=(data["radial_second_moment_ao"]
                           if "radial_second_moment_ao" in data else None))


@dataclass(frozen=True)
class GaussianErfInteraction:
    """Finite Gaussian + erf/R radial interaction, in atomic units.

    t=eta_AB*R and gamma=eta_AB*((1-2*b/sqrt(pi))*exp(-a*t*t)
    +erf(b*t)/t). At zero use its analytic limit eta_AB. This is a scalar
    semiempirical model, not a request for backend two-electron integrals.
    No fitted parameters are silently selected by this class.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if isinstance(self.a, (bool, np.bool_)) or not np.isfinite(self.a) or self.a <= 0:
            raise ValueError("Gaussian exponent a must be finite and positive")
        if (isinstance(self.b, (bool, np.bool_)) or not np.isfinite(self.b)
                or not 0 < self.b <= sqrt(pi) / 2):
            raise ValueError("erf scale b must satisfy 0 < b <= sqrt(pi)/2")

    def evaluate(self, distance_bohr: np.ndarray, hardness_hartree: np.ndarray) -> np.ndarray:
        distance, hardness = np.broadcast_arrays(np.asarray(distance_bohr, dtype=float),
                                                  np.asarray(hardness_hartree, dtype=float))
        if (not np.all(np.isfinite(distance)) or np.any(distance < 0)
                or not np.all(np.isfinite(hardness)) or np.any(hardness <= 0)):
            raise ValueError("radial distances must be nonnegative and hardnesses positive and finite")
        t = hardness * distance
        values = np.fromiter((erf(self.b * float(x)) for x in t.flat),
                             dtype=float, count=t.size).reshape(t.shape)
        screened = np.full_like(t, 2 * self.b / sqrt(pi))
        np.divide(values, t, out=screened, where=t != 0)
        return hardness * ((1 - 2 * self.b / sqrt(pi)) * np.exp(-self.a * t * t) + screened)


@dataclass(frozen=True)
class OhnoTDAParameters:
    """Global exchange fraction and independent Coulomb/exchange profiles.

    Each profile defaults independently to the original exponent-two Ohno
    interaction. An explicit GaussianErfInteraction replaces only the selected
    Coulomb or exchange interaction. eta_AB is the arithmetic mean of resident
    Grimme atomic hardnesses.
    This is a model choice, not a parametrization qualified for every HF/DFT
    reference. It is not the pair of J/K kernels used by std2 -xtb.
    extent_exchange_damping defaults to 0.5 and applies a ground-reference
    spatial filter to K only. Explicit None or zero disables the filter.
    The known uracil Rydberg regression is documented in IMPORTED_TDA.md.
    """

    exact_exchange: float
    coulomb_profile: GaussianErfInteraction | None = None
    exchange_profile: GaussianErfInteraction | None = None
    extent_exchange_damping: float | None = 0.5
    extent_filter_space: str = "full"

    @classmethod
    def from_ground_state_exchange(
        cls, ground_state_exact_exchange: float, *, response_exchange_scale: float = 0.9,
        extent_exchange_damping: float | None = 0.5,
        extent_filter_space: str = "full",
    ) -> "OhnoTDAParameters":
        """Build original radial Ohno J/K with the default spatial K filter.

        The input is the backend's ground-state fraction, not an already
        scaled response fraction. Scaling occurs here exactly once. The
        ordinary constructor instead accepts the final response fraction
        unchanged. This prescription defines vertical energies only.
        """
        if (isinstance(ground_state_exact_exchange, (bool, np.bool_))
                or not np.isfinite(ground_state_exact_exchange)
                or not 0 <= ground_state_exact_exchange <= 1):
            raise ValueError("ground-state exact exchange must be finite and lie in [0, 1]")
        if (isinstance(response_exchange_scale, (bool, np.bool_))
                or not np.isfinite(response_exchange_scale) or response_exchange_scale < 0):
            raise ValueError("response exchange scale must be finite and nonnegative")
        return cls(exact_exchange=float(ground_state_exact_exchange * response_exchange_scale),
                   extent_exchange_damping=extent_exchange_damping,
                   extent_filter_space=extent_filter_space)

    def __post_init__(self) -> None:
        if not np.isfinite(self.exact_exchange) or not 0 <= self.exact_exchange <= 1:
            raise ValueError("exact_exchange must be finite and lie in [0, 1]")
        for profile in (self.coulomb_profile, self.exchange_profile):
            if profile is not None and not isinstance(profile, GaussianErfInteraction):
                raise TypeError("radial profiles must be GaussianErfInteraction instances or None")
        if self.extent_filter_space not in {"full", "retained"}:
            raise ValueError("extent_filter_space must be full or retained (legacy)")
        beta = self.extent_exchange_damping
        if beta is not None and (isinstance(beta, (bool, np.bool_))
                or not np.isfinite(beta) or beta < 0):
            raise ValueError("extent_exchange_damping must be finite and nonnegative or None")


@dataclass(frozen=True)
class ImportedTDASelection:
    """Keep canonical virtuals no higher than HOMO + virtual_window_ev.

    All occupied MOs are retained. An optional maximum_virtuals keeps the lowest
    virtuals within the energy window. The selected space is a full occupied-
    virtual rectangle, with no perturbative CSF selection or diagonal shifts.
    """

    virtual_window_ev: float
    maximum_virtuals: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.virtual_window_ev) or self.virtual_window_ev <= 0:
            raise ValueError("virtual_window_ev must be finite and positive")
        if self.maximum_virtuals is not None and (
            isinstance(self.maximum_virtuals, bool)
            or not isinstance(self.maximum_virtuals, (int, np.integer))
            or self.maximum_virtuals < 1
        ):
            raise ValueError("maximum_virtuals must be a positive integer or None")


@dataclass(frozen=True)
class ImportedTDAResult:
    reference_model_id: str
    parameters: OhnoTDAParameters
    selection: ImportedTDASelection | None
    excitation_energies_hartree: np.ndarray
    amplitudes: np.ndarray
    occupied_orbitals: np.ndarray
    virtual_orbitals: np.ndarray
    retained_virtual_orbitals: np.ndarray
    discarded_virtual_count: int
    residual_norms_hartree: np.ndarray
    converged: bool
    solver_backend: str
    transition_dipoles_au: np.ndarray | None
    oscillator_strengths: np.ndarray | None
    virtual_space: ProjectedTDAVirtualSpace | None = None

    def __post_init__(self) -> None:
        for name in ("excitation_energies_hartree", "amplitudes", "residual_norms_hartree",
                     "transition_dipoles_au", "oscillator_strengths"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        for name in ("occupied_orbitals", "virtual_orbitals", "retained_virtual_orbitals"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name), dtype=np.int64))


class ImportedTDAOperator:
    """Matrix-free A_ia,jb = gap_ia delta + 2(ia|jb) - ax(ij|ab).

    Only atom-pair interactions and orbital-pair atomic charges are stored;
    neither AO ERIs nor the dense excitation matrix are constructed.
    """

    def __init__(self, reference: ImportedTDAReference, *, parameters: OhnoTDAParameters,
                 selection: ImportedTDASelection | None = None,
                 virtual_space: ProjectedTDAVirtualSpace | None = None):
        self.reference = reference
        self.parameters = parameters
        if parameters.extent_exchange_damping is not None and parameters.extent_exchange_damping > 0:
            if reference.dipole_ao is None or reference.radial_second_moment_ao is None:
                raise ValueError(
                    "The active spatial K filter requires dipole_ao and radial_second_moment_ao; "
                    "supply both backend matrices or set extent_exchange_damping=0 for the undamped model")
        self.selection = selection
        self.virtual_space = virtual_space
        energies = reference.orbital_energies_hartree
        occupied = np.flatnonzero(reference.occupations > 1)
        if virtual_space is not None:
            if selection is not None:
                raise ValueError("choose a canonical energy selection or a projected virtual space")
            virtual_space.validate_reference(reference)
            virtual = np.arange(virtual_space.dimension)
            virtual_coefficients = reference.coefficients[:, virtual_space.parent_virtual_orbitals] @ virtual_space.rotation
            virtual_fock = virtual_space.fock_hartree
        else:
            if selection is None:
                raise ValueError("an explicit virtual selection or projected space is required")
            virtual = np.flatnonzero(reference.occupations < 1)
            order = np.argsort(energies[virtual], kind="stable")
            virtual = virtual[order]
            cutoff = np.max(energies[occupied]) + selection.virtual_window_ev / HARTREE_TO_EV
            virtual = virtual[energies[virtual] <= cutoff]
            if selection.maximum_virtuals is not None:
                virtual = virtual[:selection.maximum_virtuals]
            virtual_coefficients = reference.coefficients[:, virtual]
            virtual_fock = np.diag(energies[virtual])
        if not len(virtual):
            raise ValueError("no virtual orbitals survive the requested energy window")
        self.occupied = _readonly_array(occupied, dtype=np.int64)
        self.virtual = _readonly_array(virtual, dtype=np.int64)
        self.virtual_coefficients = _readonly_array(virtual_coefficients)
        self.virtual_fock_hartree = _readonly_array(virtual_fock)
        self._occupied_energies = _readonly_array(energies[occupied])
        self.discarded_virtual_count = int(np.count_nonzero(reference.occupations < 1) - len(virtual))
        self.shape = (len(occupied), len(virtual))
        self.dimension = len(occupied) * len(virtual)
        values, vectors = np.linalg.eigh(reference.overlap)
        square_root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
        lowdin_occupied = square_root @ reference.coefficients[:, occupied]
        lowdin_virtual = square_root @ virtual_coefficients
        natom = len(reference.atomic_numbers)
        qov = np.empty((natom, *self.shape))
        qoo = np.empty((natom, len(occupied), len(occupied)))
        qvv = np.empty((natom, len(virtual), len(virtual)))
        for atom in range(natom):
            mask = reference.ao_atom == atom
            co, cv = lowdin_occupied[mask], lowdin_virtual[mask]
            qov[atom], qoo[atom], qvv[atom] = co.T @ cv, co.T @ co, cv.T @ cv
        hardness = grimme_stda_atomic_hardnesses()[reference.atomic_numbers - 1]
        average = (hardness[:, None] + hardness[None, :]) / 2
        displacement = reference.coordinates_bohr[:, None] - reference.coordinates_bohr[None, :]
        gamma = 1 / np.sqrt(np.sum(displacement**2, axis=2) + average**-2)
        distances = np.sqrt(np.sum(displacement**2, axis=2))
        gamma_j = (gamma if parameters.coulomb_profile is None else
                   parameters.coulomb_profile.evaluate(distances, average))
        gamma_k = (gamma if parameters.exchange_profile is None else
                   parameters.exchange_profile.evaluate(distances, average))
        self.coulomb_interaction_hartree = _readonly_array(gamma_j)
        self.exchange_interaction_hartree = _readonly_array(gamma_k)
        self._qov = _readonly_array(qov.reshape(natom, -1))
        self._gamma_qov = _readonly_array(gamma_j @ self._qov)
        self._qoo = _readonly_array(qoo)
        self._screened_qvv = _readonly_array(np.einsum("AB,Bab->Aab", gamma_k, qvv))
        self.virtual_extent_filter = None
        self.occupied_extent_bound_bohr2 = None
        if parameters.extent_exchange_damping is not None and parameters.extent_exchange_damping > 0:
            from .tda_extent_correction import _virtual_extent_filter
            if parameters.extent_filter_space == "retained":
                filtering, bound = _virtual_extent_filter(
                    reference, reference.coefficients[:, occupied], virtual_coefficients,
                    parameters.extent_exchange_damping)
                self._screened_qvv = _readonly_array(np.array([
                    filtering @ v @ filtering for v in self._screened_qvv]))
            else:
                # Define one full-reference Hamiltonian before any truncation.
                # Contract F_full U at the virtual vertices: U^T F K F U,
                # not (U^T F U)(U^T K U)(U^T F U).
                parents = np.flatnonzero(reference.occupations < 1)
                parent_coefficients = reference.coefficients[:, parents]
                filtering, bound = _virtual_extent_filter(
                    reference, reference.coefficients[:, occupied], parent_coefficients,
                    parameters.extent_exchange_damping)
                if virtual_space is None:
                    parent_positions = {int(a): j for j, a in enumerate(parents)}
                    embedding = filtering[:, [parent_positions[int(a)] for a in virtual]]
                else:
                    embedding = filtering @ virtual_space.rotation
                filtered_virtuals = square_root @ parent_coefficients @ embedding
                filtered_charges = np.array([
                    filtered_virtuals[reference.ao_atom == atom].T
                    @ filtered_virtuals[reference.ao_atom == atom]
                    for atom in range(natom)])
                self._screened_qvv = _readonly_array(
                    np.einsum("AB,Bab->Aab", gamma_k, filtered_charges))
            self.virtual_extent_filter = _readonly_array(filtering)
            self.occupied_extent_bound_bohr2 = bound
        self._gaps = _readonly_array((np.diag(virtual_fock)[None, :] - energies[occupied][:, None]).ravel())
        diagonal_j = np.sum(self._qov * self._gamma_qov, axis=0)
        diagonal_k = np.einsum("Ai,Aa->ia", np.diagonal(qoo, axis1=1, axis2=2),
                               np.diagonal(self._screened_qvv, axis1=1, axis2=2)).ravel()
        self.diagonal_hartree = _readonly_array(
            self._gaps + 2 * diagonal_j - parameters.exact_exchange * diagonal_k)

    def apply(self, vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=float)
        if array.shape != (self.dimension,):
            raise ValueError("TDA vector must have length noccupied * nvirtual")
        return self.apply_block(array[:, None])[:, 0]

    def apply_block(self, vectors: np.ndarray) -> np.ndarray:
        block = np.asarray(vectors, dtype=float)
        if block.ndim != 2 or block.shape[0] != self.dimension or not np.all(np.isfinite(block)):
            raise ValueError("TDA block must be finite with shape (nconfig, nvector)")
        if self.virtual_space is None:
            one_electron = self._gaps[:, None] * block
        else:
            amplitudes = block.reshape(*self.shape, block.shape[1])
            one_electron = (np.einsum("iak,ab->ibk", amplitudes, self.virtual_fock_hartree)
                            - self._occupied_energies[:, None, None] * amplitudes).reshape(block.shape)
        product = one_electron + 2 * self._qov.T @ (self._gamma_qov @ block)
        if self.parameters.exact_exchange:
            amplitudes = block.reshape(*self.shape, block.shape[1])
            exchange = np.zeros_like(amplitudes)
            # Fix contraction order: never let an optimizer form an ia,jb tensor.
            for qoo, screened_qvv in zip(self._qoo, self._screened_qvv):
                left = (qoo @ block.reshape(self.shape[0], -1)).reshape(amplitudes.shape)
                exchange += np.einsum("ibk,bc->ick", left, screened_qvv, optimize=True)
            product -= self.parameters.exact_exchange * exchange.reshape(block.shape)
        return product


def imported_ohno_tda(reference: ImportedTDAReference, *, parameters: OhnoTDAParameters,
                      selection: ImportedTDASelection | None = None,
                      virtual_space: ProjectedTDAVirtualSpace | None = None, roots: int = 5,
                      tolerance_hartree: float = 1e-8, maximum_iterations: int = 100,
                      maximum_subspace: int | None = None, backend: str = "auto") -> ImportedTDAResult:
    """Solve vertical singlet roots; return negative roots without clipping.

    No orbital-energy correction, xTB shift, gradient, or Hessian is applied.
    ``converged`` and residual norms must be checked by the caller.
    """
    operator = ImportedTDAOperator(reference, parameters=parameters, selection=selection,
                                   virtual_space=virtual_space)
    if isinstance(roots, bool) or not isinstance(roots, (int, np.integer)) or roots < 1:
        raise ValueError("roots must be a positive integer")
    nroots = min(roots, operator.dimension)
    if backend not in {"auto", "scipy-arpack", "dense"}:
        raise ValueError("imported Ohno TDA backend must be auto, scipy-arpack, or dense")
    if not np.isfinite(tolerance_hartree) or tolerance_hartree <= 0:
        raise ValueError("tolerance_hartree must be finite and positive")
    if not isinstance(maximum_iterations, (int, np.integer)) or maximum_iterations < 1:
        raise ValueError("maximum_iterations must be a positive integer")
    if maximum_subspace is not None and (
        not isinstance(maximum_subspace, (int, np.integer)) or maximum_subspace <= nroots
    ):
        raise ValueError("maximum_subspace must be an integer greater than the root count")
    if backend == "dense" or nroots == operator.dimension:
        values, vectors = np.linalg.eigh(operator.apply_block(np.eye(operator.dimension)))
        values, vectors = values[:nroots], vectors[:, :nroots]
        solver_backend = "numpy-dense"
    else:
        # SciPy is already a MATRIX-numerics dependency. Lanczos avoids the
        # symmetry-sector trapping and loss of orthogonality observed with
        # the shared Python Davidson on this new class of imported references.
        linear = LinearOperator((operator.dimension, operator.dimension),
                                matvec=operator.apply, matmat=operator.apply_block, dtype=float)
        ncv = min(operator.dimension, maximum_subspace or max(4 * nroots + 8, 40))
        try:
            values, vectors = eigsh(
                linear, k=nroots, which="SA", ncv=ncv, tol=tolerance_hartree * .1,
                maxiter=maximum_iterations,
                v0=np.random.default_rng(0).standard_normal(operator.dimension))
        except ArpackNoConvergence as exc:
            raise RuntimeError("imported Ohno TDA Lanczos did not converge; enlarge the subspace or iteration limit") from exc
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        solver_backend = "scipy-arpack"
    residuals = np.linalg.norm(operator.apply_block(vectors) - vectors * values, axis=0)
    orthogonal = np.allclose(vectors.T @ vectors, np.eye(nroots), atol=1e-8, rtol=0)
    solved = IterativeEigenResult(values, vectors.T, residuals, 0,
                                 bool(orthogonal and np.all(residuals <= tolerance_hartree)), solver_backend)
    dipoles = strengths = None
    if reference.dipole_ao is not None:
        co = reference.coefficients[:, operator.occupied]
        cv = operator.virtual_coefficients
        mu_ia = np.einsum("mi,dmn,na->dia", co, reference.dipole_ao, cv)
        dipoles = np.sqrt(2) * solved.eigenvectors @ mu_ia.reshape(3, -1).T
        # Oscillator strengths describe absorption only for positive roots.
        strengths = (2 / 3) * np.maximum(solved.eigenvalues, 0) * np.sum(dipoles**2, axis=1)
    return ImportedTDAResult(
        reference.model_id, parameters, selection, solved.eigenvalues, solved.eigenvectors,
        np.repeat(operator.occupied, len(operator.virtual)),
        np.tile(operator.virtual, len(operator.occupied)), operator.virtual,
        operator.discarded_virtual_count, solved.residual_norms, solved.converged,
        solved.backend, dipoles, strengths, virtual_space)
