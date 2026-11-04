"""Two-surface VCI in optimized common modes, with residual-quadratic dressing.

The coordinate optimizer, PDVCI product integrals, native-potential projector,
variational solver and resonance/PT2 owners are reused. No Duschinsky matrix
is used by the overlap contraction. Constants, translations, kinetic mixing
and property transport remain explicit. No electronic acquisition is performed.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.sparse import csr_matrix

from matrix_gf.harmonic import CM_PER_HARTREE, HESSIAN_EIGENVALUE_TO_CM
from matrix_pdvci.basis import harmonic_mode
from matrix_pdvci.operators import ProductOperator, ProductTerm
from .common_quadratic_modes import CommonQuadraticFrame, common_quadratic_coefficients
from .vibronic_sparse import SparseVibrationalHamiltonian, solve_sparse_banded_variational


@dataclass(frozen=True)
class CommonVCIHamiltonian:
    """H = reference + quadratic; reference already includes anharmonicity."""
    reference: SparseVibrationalHamiltonian
    quadratic_cm: object
    reference_id: str
    diagnostics: dict

    def __post_init__(self):
        if not isinstance(self.reference, SparseVibrationalHamiltonian) or not self.reference_id:
            raise ValueError('identified finite common reference Hamiltonian required')
        checked = SparseVibrationalHamiltonian(self.reference.basis,
            self.reference.frequencies_cm, csr_matrix(self.quadratic_cm), self.diagnostics)
        object.__setattr__(self, 'quadratic_cm', checked.matrix_cm)

    def full(self):
        return SparseVibrationalHamiltonian(self.reference.basis, self.reference.frequencies_cm,
            self.reference.matrix_cm + self.quadratic_cm, self.reference.diagnostics)


@dataclass(frozen=True)
class CommonVCIProperty:
    """Complex transition property with an explicit shared-basis identity."""
    reference_id: str
    real: ProductOperator
    imaginary: ProductOperator

    def sparse_matrix(self, basis):
        return self.real.sparse_matrix(basis) + 1j * self.imaginary.sparse_matrix(basis)


@dataclass(frozen=True)
class CommonVCIStates:
    basis: np.ndarray
    energies_cm: np.ndarray
    wavefunction_orders: tuple[np.ndarray, ...]
    reference_id: str
    diagnostics: dict

    def __post_init__(self):
        basis = np.asarray(self.basis)
        energy = np.asarray(self.energies_cm, float)
        waves = tuple(np.asarray(w, complex) for w in self.wavefunction_orders)
        if (basis.ndim != 2 or basis.dtype.kind not in 'iu' or not len(basis)
                or np.any(basis < 0) or len(set(map(tuple, basis))) != len(basis)):
            raise ValueError('distinct integer common-basis rows required')
        if (energy.ndim != 1 or not len(energy) or np.any(np.diff(energy) < 0)
                or not np.all(np.isfinite(energy)) or len(waves) != 3
                or not self.reference_id
                or any(w.shape != (len(basis), len(energy)) or not np.all(np.isfinite(w)) for w in waves)):
            raise ValueError('ordered finite energies and three compatible wavefunction orders required')
        for name, value in (('basis', basis), ('energies_cm', energy)):
            value = value.copy(); value.setflags(write=False)
            object.__setattr__(self, name, value)
        frozen = tuple(w.copy() for w in waves)
        for w in frozen:
            w.setflags(write=False)
        object.__setattr__(self, 'wavefunction_orders', frozen)


class CommonModeVCI:
    """One modal basis shared by every state of a CommonQuadraticFrame.

    Optional custom modal functions must store integrals in dimensionless
    q_i=x_i/length_i. Additional metric products can come from the existing
    PDVCI curvilinear owner. Constant local metrics alone do not certify a
    complete curvilinear Hamiltonian; that scope is recorded for every model.
    """
    def __init__(self, frame, reference_frequencies_cm, *, modal_sizes=None, modes=None):
        if not isinstance(frame, CommonQuadraticFrame):
            raise TypeError('optimized CommonQuadraticFrame required')
        self.frame = frame
        self.omega = np.array(reference_frequencies_cm, float, copy=True)
        self.omega.setflags(write=False)
        common_quadratic_coefficients(frame.states[0], self.omega)
        if (modal_sizes is None) == (modes is None):
            raise ValueError('supply saved modal functions or explicit harmonic modal sizes')
        if modes is None:
            sizes = tuple(modal_sizes)
            if len(sizes) != len(self.omega):
                raise ValueError('one modal size per common coordinate required')
            modes = tuple(harmonic_mode(float(w), label=f'common_{i}', active_size=n,
                virtual_size=n) for i, (w, n) in enumerate(zip(self.omega, sizes)))
        self.modes = tuple(modes)
        if len(self.modes) != len(self.omega):
            raise ValueError('one modal basis per common coordinate required')
        digest = hashlib.sha256(frame.states[0].coordinate_identity.encode())
        digest.update(self.omega.astype('<f8').tobytes())
        for mode in self.modes:
            for key, value in sorted(mode.operators.items()):
                digest.update(key.encode()); digest.update(np.asarray(value, dtype='<c16').tobytes())
        self.reference_id = digest.hexdigest()
        self.length = common_quadratic_coefficients(frame.states[0], self.omega)[-1]

    def _basis(self, basis):
        rows = np.asarray(basis)
        if (rows.ndim != 2 or rows.shape[1] != len(self.modes) or not len(rows)
                or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0)
                or np.any(rows >= [m.dimension for m in self.modes])
                or len(set(map(tuple, rows))) != len(rows)):
            raise ValueError('distinct bounded integer product occupations required')
        return tuple(map(tuple, rows))

    def original_coordinates(self, common_dimensionless):
        q = np.asarray(common_dimensionless, float)
        if q.ndim != 2 or q.shape[1] != len(self.omega) or not np.all(np.isfinite(q)):
            raise ValueError('finite common dimensionless coordinate rows required')
        return self.frame.origin + (q * self.length) @ self.frame.transform.T

    def quadratic_terms(self, state_index):
        state = self.frame.states[state_index]
        kq, kp, linear, constant, _ = common_quadratic_coefficients(state, self.omega)
        diagonal = [ProductTerm(constant, ())]
        mixed = []
        for i in range(len(self.omega)):
            diagonal.extend([ProductTerm(linear[i], ((i, 'q'),)),
                ProductTerm(.5*kq[i,i], ((i, 'q_q'),)),
                ProductTerm(.5*kp[i,i], ((i, 'p_p'),))])
            for j in range(i+1, len(self.omega)):
                mixed.extend([ProductTerm(kq[i,j], ((i, 'q'), (j, 'q'))),
                              ProductTerm(kp[i,j], ((i, 'p'), (j, 'p')))])
        return tuple(t for t in diagonal if t.coefficient != 0), tuple(t for t in mixed if t.coefficient != 0)

    def _container(self, basis, reference, quadratic, **scope):
        basis = np.asarray(basis, dtype=np.int64)
        diagnostics = dict(schema='matrix.trinity.common_mode_vci.v1',
            maximum_total_quanta=int(basis.sum(axis=1).max()),
            maximum_harmonic_energy_cm=float((basis @ self.omega).max()),
            coordinate_identity=self.frame.states[0].coordinate_identity,
            common_basis_id=self.reference_id, Duschinsky_in_overlap=False,
            quadratic_potential_and_kinetic_mixing_retained=True,
            finite_basis_convergence_certified=False, new_electronic_evaluations=0, **scope)
        reference, quadratic = csr_matrix(reference), csr_matrix(quadratic)
        # PDVCI includes p as complex even when the complete Hamiltonian is
        # exactly real. Preserve real arithmetic without pruning any entry.
        if not np.any(reference.data.imag):
            reference = reference.real
        if not np.any(quadratic.data.imag):
            quadratic = quadratic.real
        h = SparseVibrationalHamiltonian(basis, self.omega, reference, diagnostics)
        return CommonVCIHamiltonian(h, quadratic, self.reference_id, diagnostics)

    def product_hamiltonian(self, state_index, basis, *, anharmonic_terms=(),
                            kinetic_correction_terms=(), correction_coordinate_identity=None):
        """Use existing PDVCI products; supplied anharmonic terms exclude H2.

        Native/nonpolynomial modal integrals may be supplied through saved
        ContractedModeBasis operators. No Taylor reconstruction is required.
        Product coefficients are in cm-1, in this exact common coordinate gauge.
        """
        basis = self._basis(basis)
        anharmonic_terms = tuple(anharmonic_terms)
        kinetic_correction_terms = tuple(kinetic_correction_terms)
        if (anharmonic_terms or kinetic_correction_terms) and correction_coordinate_identity != self.frame.states[0].coordinate_identity:
            raise ValueError('corrections must identify the optimized common coordinate frame')
        diagonal, mixed = self.quadratic_terms(state_index)
        reference = ProductOperator(self.modes, diagonal + tuple(anharmonic_terms)
                                    + tuple(kinetic_correction_terms)).sparse_matrix(basis)
        quadratic = ProductOperator(self.modes, mixed).sparse_matrix(basis)
        return self._container(basis, reference, quadratic,
            potential_representation='resident PDVCI sum of saved modal products',
            kinetic_model='constant state metric plus supplied correction products',
            supplied_kinetic_correction_terms=len(kinetic_correction_terms))

    def native_hamiltonian(self, state_index, potential_hartree, *, potential_identifier,
                           reference_energy_hartree, kinetic_correction_terms=(),
                           correction_coordinate_identity=None, **projection):
        """Project a complete fitted potential via the resident native projector.

        Callback accepts original-chart coordinate rows. It must be an already
        accepted analytic PES, never an electronic backend. Only harmonic HO
        modal bases are supported by this existing projector. Sparse metric
        corrections are added exactly in the declared finite product space.
        """
        from .native_pes_hamiltonian import build_native_modal_hamiltonian
        if any(m.basis_kind != 'harmonic' for m in self.modes):
            raise ValueError('direct native projection requires shared harmonic modal functions')
        if not np.isfinite(reference_energy_hartree):
            raise ValueError('finite potential-origin energy required')
        kinetic_correction_terms = tuple(kinetic_correction_terms)
        if kinetic_correction_terms and correction_coordinate_identity != self.frame.states[0].coordinate_identity:
            raise ValueError('corrections must identify the optimized common coordinate frame')
        def values(q):
            v = np.asarray(potential_hartree(self.original_coordinates(q)))
            if v.shape != (len(q),) or np.iscomplexobj(v) or not np.all(np.isfinite(v)):
                raise ValueError('native potential must return finite real Hartree energies per row')
            return (v - reference_energy_hartree) * CM_PER_HARTREE
        projected = build_native_modal_hamiltonian(self.omega, values,
            potential_identifier=potential_identifier, **projection)
        if np.any(projected.basis.max(axis=0) >= [m.dimension for m in self.modes]):
            raise ValueError('native projection exceeds saved common modal integrals')
        basis = self._basis(projected.basis)
        _, kp, _, _, _ = common_quadratic_coefficients(self.frame.states[state_index], self.omega)
        kinetic = []
        for i, w in enumerate(self.omega):
            kinetic.append(ProductTerm(.5*(kp[i,i]-w), ((i,'p_p'),)))
            for j in range(i+1,len(self.omega)):
                kinetic.append(ProductTerm(kp[i,j], ((i,'p'),(j,'p'))))
        _, mixed = self.quadratic_terms(state_index)
        full = projected.matrix_cm + ProductOperator(self.modes, tuple(kinetic)
                + tuple(kinetic_correction_terms)).sparse_matrix(basis)
        quadratic = ProductOperator(self.modes, mixed).sparse_matrix(basis)
        return self._container(projected.basis, full-quadratic, quadratic,
            potential_representation='complete native analytic potential; no polynomial reexpansion',
            native_projection=projected.diagnostics,
            reference_energy_hartree=float(reference_energy_hartree),
            kinetic_model='constant state metric plus supplied correction products',
            supplied_kinetic_correction_terms=len(kinetic_correction_terms))

    def linear_properties(self, constants, gradients_original, reference_original, *,
                          momentum_coefficients_original=None):
        """Transport constant + position + momentum operators consistently.

        Momentum is p_s=-i d/ds in the original chart and p_q=-i d/dq
        in the common dimensionless chart. Coefficients multiply p on its
        left; callers supply the complete operator (including any scalar
        derivative-coupling term). No derivative-coupling completion is inferred.
        """
        c = np.atleast_1d(np.asarray(constants, complex))
        g = np.atleast_2d(np.asarray(gradients_original, complex))
        r = np.asarray(reference_original, float)
        n = len(self.omega)
        momentum = (np.zeros_like(g) if momentum_coefficients_original is None
                    else np.atleast_2d(np.asarray(momentum_coefficients_original, complex)))
        if momentum.shape != g.shape or not np.all(np.isfinite(momentum)):
            raise ValueError('aligned finite original-chart momentum coefficients required')
        if g.shape != (len(c),n) or r.shape != (n,) or not all(np.all(np.isfinite(v)) for v in (c,g,r)):
            raise ValueError('aligned finite original-chart property constants/gradients required')
        centers = c + g @ (self.frame.origin-r)
        gradients = (g @ self.frame.transform) * self.length
        momentum = momentum @ np.linalg.inv(self.frame.transform*self.length).T
        def operator(value, row, momentum_row):
            terms = ((ProductTerm(value, ()),) + tuple(
                ProductTerm(a, ((i, 'q'),)) for i, a in enumerate(row) if a != 0)
                + tuple(ProductTerm(a, ((i, 'p'),)) for i, a in enumerate(momentum_row) if a != 0))
            real = ProductOperator(self.modes, tuple(ProductTerm(float(t.coefficient.real), t.factors) for t in terms))
            imag = ProductOperator(self.modes, tuple(ProductTerm(float(t.coefficient.imag), t.factors) for t in terms))
            return CommonVCIProperty(self.reference_id, real, imag)
        return tuple(operator(value, row, p) for value, row, p in zip(centers, gradients, momentum))


def solve_common_mode_vci(hamiltonian, *, levels, quadratic_treatment='variational',
                          maximum_reference_states=512, selection=None, **solver):
    """Solve Href+K exactly, or dress anharmonic-reference VCI states in K.

    PT2 acts only on the residual quadratic operator, not the anharmonic
    reference. Hybrid treatment uses existing exact resonance polyads and
    external PT2; pure PT2 rejects a selected resonance. All intermediate
    states belong to the explicit supplied finite basis.
    """
    if not isinstance(hamiltonian, CommonVCIHamiltonian):
        raise TypeError('common-mode Hamiltonian required')
    size = len(hamiltonian.reference.basis)
    if isinstance(levels,bool) or not isinstance(levels,int) or not 0 < levels <= size:
        raise ValueError('bounded positive level count required')
    if quadratic_treatment == 'variational':
        result = solve_sparse_banded_variational(hamiltonian.full(),level_count=levels,**solver)
        waves = (result.eigenvectors, np.zeros_like(result.eigenvectors), np.zeros_like(result.eigenvectors))
        return CommonVCIStates(result.basis,result.energies_cm,waves,hamiltonian.reference_id,
            dict(result.diagnostics,quadratic_treatment='variational'))
    if quadratic_treatment not in ('pt2','hybrid'):
        raise ValueError('quadratic treatment must be variational, pt2 or hybrid')
    if (isinstance(maximum_reference_states, bool) or not isinstance(maximum_reference_states, int)
            or maximum_reference_states < 1):
        raise ValueError('positive integer reference-state budget required')
    if size > maximum_reference_states:
        raise MemoryError('quadratic dressing reference exceeds explicit spectral budget')
    from .local_quadratic import build_local_quadratic_model
    from .vibronic_common_basis import selected_quadratic_dressing
    reference = solve_sparse_banded_variational(hamiltonian.reference,level_count=size,**solver)
    u = reference.eigenvectors
    k = u.conj().T @ (hamiltonian.quadratic_cm @ u)
    if np.max(abs(k.imag),initial=0) > 1e-10:
        raise ValueError('resident quadratic PT2 polyads require a real time-reversal-invariant representation')
    k = k.real
    energies = reference.energies_cm + np.diag(k)
    edges = {(i,j):float(k[i,j]) for i in range(size) for j in range(i+1,size) if k[i,j] != 0}
    model = build_local_quadratic_model(energies, {}, edges, **dict(selection or {}))
    if quadratic_treatment == 'pt2' and model.polyads:
        raise ValueError('residual quadratic resonance requires hybrid or variational treatment')
    dressed = selected_quadratic_dressing(model,maximum_states=maximum_reference_states)
    take = np.argsort(dressed['energies_cm'])[:levels]
    waves = tuple(u @ w[:,take] for w in dressed['wavefunction_orders'])
    return CommonVCIStates(reference.basis,np.asarray(dressed['energies_cm'])[take],waves,
        hamiltonian.reference_id,dict(hamiltonian.diagnostics,quadratic_treatment=quadratic_treatment,
        reference_VCI_states=size,polyad_sizes=[len(p) for p in model.polyads],
        diagonal_quadratic_expectation_in_reference=True,
        external_quadratic_order=2,variational_residuals_not_PT_error_bounds=True))


def common_vci_transition_orders(initial, final, property_operator=None):
    """Shared-basis amplitudes and strict order-two strengths on both states."""
    if initial.reference_id != final.reference_id:
        raise ValueError('both electronic states must use the exact same modal functions and coordinate frame')
    left = final.wavefunction_orders; right = initial.wavefunction_orders
    if property_operator is None:
        lookup = {tuple(row):i for i,row in enumerate(initial.basis)}
        source = csr_matrix(([1. for row in final.basis if tuple(row) in lookup],
            ([i for i,row in enumerate(final.basis) if tuple(row) in lookup],
             [lookup[tuple(row)] for row in final.basis if tuple(row) in lookup])),
             shape=(len(final.basis),len(initial.basis)))
    else:
        if not isinstance(property_operator, CommonVCIProperty) or property_operator.reference_id != initial.reference_id:
            raise ValueError('transition property must use the exact common modal frame')
        # Project once over the union; different finite state ceilings remain valid.
        union = tuple(dict.fromkeys([*map(tuple,initial.basis),*map(tuple,final.basis)]))
        lookup = {row:i for i,row in enumerate(union)}
        matrix = property_operator.sparse_matrix(union)
        source = matrix[[lookup[tuple(row)] for row in final.basis],:][:,[lookup[tuple(row)] for row in initial.basis]]
    amplitudes = tuple(sum((left[i].conj().T @ (source @ right[k-i]) for i in range(k+1)),
                            start=np.zeros((left[0].shape[1],right[0].shape[1]),complex)) for k in range(3))
    strength_orders = tuple(sum((amplitudes[i]*amplitudes[k-i].conj()).real for i in range(k+1)) for k in range(3))
    return dict(amplitude_orders=amplitudes,strength_orders=strength_orders,
        strengths=sum(strength_orders),completed_strengths=abs(sum(amplitudes))**2,
        initial_energies_cm=initial.energies_cm,final_energies_cm=final.energies_cm,
        Duschinsky_in_overlap=False,external_quadratic_order=2)
