"""Partial number-state contractions using the existing Raman TD algebra.

Only the indices to eliminate and their number weights are generalized here.
All uncontracted Gaussian variables remain available to the shared Hermite
recurrence, including active-mode/initial-state correlations.
"""
import numpy as np


class GaussianCoherentBathPackets:
    """Bargmann packets for the same additive native/harmonic bath model.

    For standard complex Gaussian z, K_ab(t) = E[v_a(exp(+i wt)z)* v_b(z)].
    This is an exact resolution identity, not a claim that finitely many
    supplied integration nodes converge. No bath number cutoff is introduced.
    """
    def __init__(self, model, initial, basis, active_final_modes, *, kind='emission',
                 initial_level=0, generator=None, embedding=None,
                 maximum_integrals=500000, maximum_workspace_bytes=256*1024**2):
        from .vibronic_harmonic import orient_harmonic_model, harmonic_gaussian_generator
        from .vibronic_variational import FiniteVibrationalStates
        from .gaussian_source_polynomial import GaussianSourceAmplitude
        from .quantum_dynamics import _positive_integer
        oriented, _ = orient_harmonic_model(model, kind)
        n = len(oriented.final_frequencies_cm)
        basis, active = np.asarray(basis), np.asarray(active_final_modes)
        if active.size == 0:
            active = active.astype(np.int64)
        if (not isinstance(initial, FiniteVibrationalStates)
                or not np.array_equal(initial.frequencies_cm, oriented.initial_frequencies_cm)
                or isinstance(initial_level, (bool, np.bool_))
                or not isinstance(initial_level, (int, np.integer))
                or not 0 <= initial_level < len(initial.energies_cm)
                or active.ndim != 1 or active.dtype.kind not in 'iu'
                or len(np.unique(active)) != len(active) or np.any(active < 0) or np.any(active >= n)
                or basis.ndim != 2 or basis.dtype.kind not in 'iu' or not len(basis)
                or basis.shape[1] != len(active) or np.any(basis < 0)
                or len(np.unique(basis, axis=0)) != len(basis)):
            raise ValueError('matching finite initial state, unique active modes and HO basis required')
        _positive_integer(maximum_integrals, 'maximum_integrals')
        _positive_integer(maximum_workspace_bytes, 'maximum_workspace_bytes')
        initial_modes = np.flatnonzero(initial.basis.max(axis=0))
        bath = np.setdiff1d(np.arange(n), active)
        kept = np.r_[initial_modes, n+active]
        selected = np.r_[kept, n+bath]
        g = harmonic_gaussian_generator(oriented) if generator is None else generator
        if np.shape(g['matrix']) != (2*n, 2*n) or np.shape(g['linear']) != (2*n,):
            raise ValueError('Gaussian generator dimensions differ from oriented model')
        self.matrix = np.asarray(g['matrix'])[np.ix_(kept, kept)].copy()
        self.linear = np.asarray(g['linear'])[kept].copy()
        self.cross = np.asarray(g['matrix'])[np.ix_(kept, n+bath)].copy()
        self.bath_matrix = np.asarray(g['matrix'])[np.ix_(n+bath, n+bath)].copy()
        self.bath_linear = np.asarray(g['linear'])[n+bath].copy()
        self.log_ground = float(g['log_ground'])
        # Check the same finite, symmetric and contractive Gaussian contract.
        gaussian_partial_number_trace(np.asarray(g['matrix'])[np.ix_(selected, selected)],
            np.asarray(g['linear'])[selected], self.log_ground,
            np.arange(len(kept), len(selected)), np.ones(len(bath)))
        if embedding is not None:
            embedding = np.asarray(embedding)
            if (embedding.ndim != 2 or embedding.shape[0] != len(basis)
                    or not 0 < embedding.shape[1] <= len(basis)
                    or not np.all(np.isfinite(embedding))
                    or not np.allclose(embedding.conj().T@embedding, np.eye(embedding.shape[1]),
                                       atol=1e-11, rtol=1e-11)):
                raise ValueError('orthonormal finite active embedding required')
            embedding = embedding.copy()
            embedding.setflags(write=False)
        self.amplitude = GaussianSourceAmplitude(initial.basis[:, initial_modes],
            initial.eigenvectors[:, initial_level], basis, maximum_integrals=maximum_integrals,
            maximum_workspace_bytes=maximum_workspace_bytes)
        self.embedding = embedding
        self.embedding_adjoint = None if embedding is None else embedding.conj().T.copy()
        if self.embedding_adjoint is not None:
            self.embedding_adjoint.setflags(write=False)
        self.size = len(basis) if embedding is None else embedding.shape[1]
        self.bath_frequencies_cm = oriented.final_frequencies_cm[bath].copy()
        self.initial_energy_cm = float(initial.energies_cm[initial_level]-oriented.electronic_gap_cm)
        self.bath_zero_point_cm = float(.5*self.bath_frequencies_cm.sum())
        self.provenance = dict(model.provenance)
        self.numeric_storage_bytes = (self.amplitude.numeric_storage_bytes
            +sum(a.nbytes for a in (self.matrix, self.linear, self.cross,
                self.bath_matrix, self.bath_linear, self.bath_frequencies_cm))
            +(0 if embedding is None else embedding.nbytes+self.embedding_adjoint.nbytes))
        self.evaluation_workspace_bytes = self.amplitude.evaluation_workspace_bytes
        self.maximum_workspace_bytes = maximum_workspace_bytes
        if self.numeric_storage_bytes+self.evaluation_workspace_bytes > maximum_workspace_bytes:
            raise MemoryError('coherent bath packets exceed numeric workspace budget')
        for array in (self.matrix, self.linear, self.cross, self.bath_matrix,
                      self.bath_linear, self.bath_frequencies_cm):
            array.setflags(write=False)
        self.diagnostics = dict(self.amplitude.diagnostics, initial_states=len(initial.basis),
            active_ho_states=len(basis), spectator_modes=bath.tolist(), active_modes=active.tolist(),
            spectator_number_cutoff_used=False, active_spectator_factorization_assumed=False,
            active_embedding_used=embedding is not None, finite_active_space_convergence_certified=False)

    def evaluate(self, node, time_fs=0.):
        """Return v(exp(+i wt)z); Gaussian node weights are owned by the caller."""
        node = np.asarray(node, complex)
        if node.ndim != 1:
            raise ValueError('one finite coherent coordinate per bath mode required')
        return self.evaluate_nodes(node[None, :], time_fs)[:, 0]

    def evaluate_nodes(self, nodes, time_fs=0.):
        """Reuse a single matrix product for a batch of embedded packets."""
        return self.evaluate_grid(nodes, np.array([time_fs]))[0]

    def evaluate_grid(self, nodes, times_fs):
        """Batch times and nodes before the shared spectral projection."""
        from .quantum_dynamics import PHASE_PER_CM_FS
        nodes = np.asarray(nodes, complex)
        times = np.asarray(times_fs, float)
        if (nodes.ndim != 2 or nodes.shape[1] != len(self.bath_frequencies_cm)
                or not len(nodes) or not np.all(np.isfinite(nodes))
                or times.ndim != 1 or not len(times) or not np.all(np.isfinite(times))):
            raise ValueError('one finite coherent coordinate per bath mode and finite times required')
        entries = len(times)*len(nodes)*(self.diagnostics['active_ho_states']+3*self.size
                  +3*len(self.bath_frequencies_cm)+2*len(self.linear)+2)
        if self.numeric_storage_bytes+self.evaluation_workspace_bytes+16*entries > self.maximum_workspace_bytes:
            raise MemoryError('coherent packet time/node block exceeds numeric workspace budget')
        z = (nodes[None,:,:]*np.exp(1j*times[:,None,None]
             *self.bath_frequencies_cm[None,None,:]*PHASE_PER_CM_FS)).reshape(
                 len(times)*len(nodes),len(self.bath_frequencies_cm))
        constants = np.exp(self.log_ground+.5*np.einsum('ij,jk,ik->i',z,self.bath_matrix,z,optimize=True)
                           +z@self.bath_linear)
        linear = self.linear[None,:]+z@self.cross.T
        value = self.amplitude.evaluate_batch(self.matrix,linear,constants).T
        self.diagnostics.update(self.amplitude.diagnostics)
        if self.embedding is not None:
            if np.isrealobj(self.embedding_adjoint):
                value = self.embedding_adjoint@value.real+1j*(self.embedding_adjoint@value.imag)
            else:
                value = self.embedding_adjoint@value
        return value.reshape(self.size,len(times),len(nodes)).transpose(1,0,2)


class GaussianBathKernel:
    """Analytic harmonic bath trace for a finite, coherent initial state.

    The retained active basis is in final HO coordinates, in the supplied mode
    order. K_ab(t) = sum_s v_as* v_bs exp(-i E_s t) excludes bath zero-point
    energy and electronic couplings. Thus its contraction with a propagator is
    sum_ab K_ab U_ab, without another conjugation or transpose.

    No bath occupations are enumerated and no initial coefficients are pruned.
    Small requests use the exact joint Hermite reference. When its pair budget
    is exceeded, the source-first polynomial backend is used if its own
    polynomial and FFT grids fit. Both keep every initial coefficient.
    An orthonormal embedding can map contracted active states into the HO basis.
    It does not remove the cost of constructing the HO kernel.
    """
    def __init__(self, model, initial, basis, active_final_modes, *, kind='emission',
                 initial_level=0, generator=None, embedding=None,
                 maximum_integrals=500000, maximum_workspace_bytes=256*1024**2):
        from .vibronic_harmonic import orient_harmonic_model, harmonic_gaussian_generator
        from .vibronic_variational import FiniteVibrationalStates
        from .gaussian_hermite import hermite_recurrence_plan
        from .quantum_dynamics import _positive_integer
        oriented, _ = orient_harmonic_model(model, kind)
        n = len(oriented.final_frequencies_cm)
        basis, active = np.asarray(basis), np.asarray(active_final_modes)
        if active.size == 0:
            active = active.astype(np.int64)
        if (not isinstance(initial, FiniteVibrationalStates)
                or not np.array_equal(initial.frequencies_cm, oriented.initial_frequencies_cm)
                or isinstance(initial_level, (bool, np.bool_))
                or not isinstance(initial_level, (int, np.integer))
                or not 0 <= initial_level < len(initial.energies_cm)
                or active.ndim != 1 or active.dtype.kind not in 'iu'
                or len(np.unique(active)) != len(active) or np.any(active < 0) or np.any(active >= n)
                or basis.ndim != 2 or basis.dtype.kind not in 'iu' or not len(basis)
                or basis.shape[1] != len(active) or np.any(basis < 0)
                or len(np.unique(basis, axis=0)) != len(basis)):
            raise ValueError('matching finite initial state, unique active modes and HO basis required')
        _positive_integer(maximum_integrals, 'maximum_integrals')
        _positive_integer(maximum_workspace_bytes, 'maximum_workspace_bytes')
        ni, na = len(initial.basis), len(basis)
        pairs = (ni*na)**2
        initial_modes = np.flatnonzero(initial.basis.max(axis=0))
        dimension = 2*(len(initial_modes)+len(active))
        largest_order = (max(sum(map(int, row)) for row in initial.basis)
                         +max(sum(map(int, row)) for row in basis))
        if 2*largest_order+1 > maximum_integrals:
            raise MemoryError(f'analytic bath needs {pairs} initial/active pairs; '
                              'lowering chain exceeds Hermite plan budget')
        # Bound numeric plan/output arrays, including a maximally connected
        # recurrence. Python graph-building overhead is separately constrained
        # by maximum_integrals and the caller's process address-space limit.
        source_first = None
        if pairs > maximum_integrals:
            from .gaussian_source_polynomial import GaussianSourcePolynomial
            try:
                source_first = GaussianSourcePolynomial(initial.basis[:, initial_modes],
                    initial.eigenvectors[:, initial_level], basis,
                    maximum_integrals=maximum_integrals, maximum_workspace_bytes=maximum_workspace_bytes)
            except MemoryError as error:
                raise MemoryError(f'analytic bath needs {pairs} initial/active pairs; '
                                  f'{error}; no states discarded') from error
        estimate = (source_first.evaluation_workspace_bytes if source_first is not None else
                    pairs*(8*dimension+64)+maximum_integrals*(32+24*dimension)+128*na*na)
        if estimate > maximum_workspace_bytes:
            raise MemoryError('analytic bath Hermite plan exceeds numeric workspace budget')
        if embedding is not None:
            embedding = np.asarray(embedding, complex)
            if (embedding.ndim != 2 or embedding.shape[0] != na or not embedding.shape[1]
                    or embedding.shape[1] > na or not np.all(np.isfinite(embedding))
                    or not np.allclose(embedding.conj().T@embedding, np.eye(embedding.shape[1]),
                                       atol=1e-11, rtol=1e-11)):
                raise ValueError('orthonormal finite active embedding required')
            if estimate+3*embedding.nbytes > maximum_workspace_bytes:
                raise MemoryError('active embedding exceeds numeric workspace budget')
            embedding = embedding.copy()
            embedding.setflags(write=False)
        g = harmonic_gaussian_generator(oriented) if generator is None else generator
        if np.shape(g['matrix']) != (2*n, 2*n) or np.shape(g['linear']) != (2*n,):
            raise ValueError('Gaussian generator dimensions differ from oriented model')
        spectators = np.setdiff1d(np.arange(n), active)
        selected = np.r_[initial_modes, n+active, n+spectators]
        self.matrix = np.asarray(g['matrix'])[np.ix_(selected, selected)].copy()
        self.linear = np.asarray(g['linear'])[selected].copy()
        self.log_ground = float(g['log_ground'])
        self.contracted = np.arange(len(initial_modes)+len(active), len(selected))
        self.frequencies_cm = oriented.final_frequencies_cm[spectators].copy()
        # Validate the same Gaussian contract before building an expensive plan.
        gaussian_partial_number_trace(self.matrix, self.linear, self.log_ground,
                                      self.contracted, np.ones(len(spectators)))
        if source_first is None:
            rows = np.column_stack([np.repeat(initial.basis[:, initial_modes], na, axis=0),
                                    np.tile(basis, (ni, 1))])
            words = np.column_stack([np.repeat(rows, len(rows), axis=0), np.tile(rows, (len(rows), 1))])
            self.plan = hermite_recurrence_plan(words, 0, maximum_integrals)
        else:
            self.plan = source_first.plan
        self.source_first = source_first
        self.coefficients = initial.eigenvectors[:, initial_level].copy()
        for array in (self.matrix, self.linear, self.coefficients, self.frequencies_cm):
            array.setflags(write=False)
        self.embedding = embedding
        self.initial_states, self.ho_states = ni, na
        self.size = na if embedding is None else embedding.shape[1]
        self.initial_energy_cm = float(initial.energies_cm[initial_level]-oriented.electronic_gap_cm)
        self.bath_zero_point_cm = float(.5*self.frequencies_cm.sum())
        self.provenance = dict(model.provenance)
        self.numeric_storage_bytes = ((sum(a.nbytes for a in self.plan) if source_first is None else
                                     source_first.numeric_storage_bytes)+self.matrix.nbytes
            +self.linear.nbytes+self.coefficients.nbytes
            +(0 if embedding is None else embedding.nbytes))
        self.diagnostics = dict(initial_states=ni, active_ho_states=na,
            contracted_active_states=self.size, spectator_modes=spectators.tolist(),
            active_modes=active.tolist(), hermite_recurrence_entries=len(self.plan[0]),
            initial_active_pair_count=pairs, numeric_storage_bytes=self.numeric_storage_bytes,
            initial_active_pair_plan_constructed=source_first is None,
            contraction_backend='joint Hermite' if source_first is None else 'source-first Wick/Horner/FFT',
            numeric_workspace_estimate_bytes=estimate,
            spectator_occupation_sum_analytic=True, spectator_sampling_used=False,
            spectator_occupation_cutoff_used=False, initial_coefficients_pruned=False,
            active_spectator_factorization_assumed=False,
            gaussian_generator_supplied_by_caller=generator is not None,
            additional_gaussian_threshold_applied=False,
            active_embedding_used=embedding is not None,
            finite_active_space_convergence_certified=False)
        self.evaluation_workspace_bytes = (48*pairs+16*len(self.plan[0])+64*na*na
            if source_first is None else source_first.evaluation_workspace_bytes)
        if source_first is not None:
            self.diagnostics.update(source_first.diagnostics)

    def evaluate(self, time_fs):
        """Return one coupled kernel; no time-by-active-by-active table is kept."""
        from .quantum_dynamics import PHASE_PER_CM_FS
        from .gaussian_hermite import hermite_coefficients
        if not np.isscalar(time_fs) or not np.isfinite(time_fs) or time_fs < 0:
            raise ValueError('finite nonnegative time required')
        traced = gaussian_partial_number_trace(self.matrix, self.linear, self.log_ground,
            self.contracted, np.exp(-1j*self.frequencies_cm*PHASE_PER_CM_FS*time_fs))
        if self.source_first is None:
            joint = hermite_coefficients(traced['matrix'], traced['linear'], traced['constant'], self.plan)
            joint = joint.reshape(self.initial_states, self.ho_states, self.initial_states, self.ho_states)
            kernel = np.einsum('i,iajb,j->ab', self.coefficients.conj(), joint, self.coefficients,
                               optimize=True)
        else:
            kernel = self.source_first.evaluate(traced['matrix'], traced['linear'], traced['constant'])
            self.diagnostics.update(self.source_first.diagnostics)
        if self.embedding is not None:
            kernel = self.embedding.T@kernel@self.embedding.conj()
        if not np.all(np.isfinite(kernel)):
            raise FloatingPointError('analytic bath kernel exceeds numerical range')
        return kernel


def gaussian_partial_number_trace(matrix, linear, log_ground, indices, weights):
    """Sum G_left* G_right times product(weights**occupation), analytically.

    ``matrix`` and ``linear`` define the complete normalized Bargmann generator.
    The returned retained-variable order is [conjugate bra, unconjugated ket].
    Number weights are dimensionless, of modulus at most one. Real-time weights
    exp(-i*omega*tau) exclude the oscillator zero-point phase, which belongs to
    the caller's Hamiltonian energy convention.

    The covariance/response arrays expose the same polynomial-insertion
    contract consumed by Raman HT; no property operator is inferred here.
    """
    m, l = np.asarray(matrix), np.asarray(linear)
    indices, z = np.asarray(indices), np.asarray(weights, complex)
    if indices.size == 0:
        indices = indices.astype(np.int64)
    if (m.ndim != 2 or m.shape[0] != m.shape[1] or l.shape != (len(m),)
            or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(m))
            or z.shape != indices.shape or not np.all(np.isfinite(z))
            or np.any(abs(z) > 1+1e-12) or not np.all(np.isfinite(m))
            or not np.all(np.isfinite(l)) or not np.isfinite(log_ground)
            or not np.allclose(m, m.T, rtol=1e-12, atol=1e-12)):
        raise ValueError('finite symmetric generator, unique indices and bounded number weights required')
    retained = np.setdiff1d(np.arange(len(m)), indices)
    nr, nc = len(retained), len(indices)
    aa = m[np.ix_(retained, retained)]
    ab = m[np.ix_(retained, indices)]
    bb = m[np.ix_(indices, indices)]
    ra, rb = l[retained], l[indices]
    contraction = np.zeros((2*nc, 2*nc), complex)
    contraction[:nc, nc:] = np.diag(z)
    contraction[nc:, :nc] = np.diag(z)
    precision = np.zeros_like(contraction)
    precision[:nc, :nc] = bb.conj()
    precision[nc:, nc:] = bb
    covariance = np.linalg.solve(np.eye(2*nc)-contraction@precision, contraction)
    eigen = np.linalg.eigvals((z[:, None]*bb.conj()*z[None, :])@bb)
    if len(eigen) and (np.any(abs(eigen) > 1+1e-7) or np.min(abs(1-eigen)) < 1e-12):
        raise FloatingPointError('ill-conditioned Gaussian number contraction')
    response = np.zeros((2*nc, 2*nr), complex)
    response[:nc, :nr] = ab.conj().T
    response[nc:, nr:] = ab.T
    mean = covariance@np.r_[rb.conj(), rb]
    reduced_linear = np.r_[ra.conj(), ra]+response.T@mean
    reduced_matrix = np.zeros((2*nr, 2*nr), complex)
    reduced_matrix[:nr, :nr] = aa.conj()
    reduced_matrix[nr:, nr:] = aa
    reduced_matrix += response.T@covariance@response
    constant = np.exp(2*float(log_ground)-.5*np.log1p(-eigen).sum()
                      +.5*np.r_[rb.conj(), rb]@mean)
    if (not np.all(np.isfinite(reduced_matrix)) or not np.all(np.isfinite(reduced_linear))
            or not np.isfinite(constant)):
        raise FloatingPointError('Gaussian contraction overflow or nonfinite result')
    return dict(matrix=reduced_matrix, linear=reduced_linear, constant=constant,
        retained_indices=retained, contracted_indices=indices.copy(),
        covariance=covariance, mean=mean, response=response,
        diagnostics=dict(contracted_mode_count=nc, retained_variable_count=nr,
            final_occupation_sum_analytic=True, number_cutoff_used=False,
            active_spectator_factorization_assumed=False,
            determinant_branch='Raman TD log1p eigenvalue continuation'))


def gaussian_vacuum_trace_batch(matrix, linear, log_ground, number_weights, *,
                               chunk_size=128, maximum_workspace_bytes=256*1024**2):
    """Batched zero-retained-variable limit of gaussian_partial_number_trace.

    Reuses its normalized Bargmann/Mehler convention and analytic unit-disk
    determinant branch. This specialization eliminates the doubled covariance
    when only the vacuum-source scalar correlation is required.
    """
    return gaussian_vacuum_operator_trace_batch(matrix, linear, log_ground,
        number_weights, np.empty(0), np.empty((0, len(linear))),
        chunk_size=chunk_size, maximum_workspace_bytes=maximum_workspace_bytes)['vacuum_trace']


def gaussian_vacuum_operator_trace_batch(matrix, linear, log_ground, number_weights,
        operator_constants, operator_linear, *, chunk_size=128,
        maximum_workspace_bytes=256*1024**2):
    """Contract linear Bargmann insertions with the shared vacuum trace.

    Each inserted amplitude is ``(c_a + L_a z) G(z)``. The returned kernel
    has order time, unconjugated operator a, conjugated operator b. Signed
    interference follows by contracting this bank, without finite differences
    or a number cutoff. An empty bank recovers the scalar vacuum owner.
    """
    m, r = np.asarray(matrix), np.asarray(linear)
    constants = np.asarray(operator_constants, complex)
    insertion = np.asarray(operator_linear, complex)
    if (constants.ndim != 1 or insertion.shape != (len(constants), len(r))
            or not np.all(np.isfinite(constants)) or not np.all(np.isfinite(insertion))):
        raise ValueError('finite constants and one linear insertion row per operator required')
    z = np.asarray(number_weights, complex)
    if z.ndim != 2 or not len(z):
        raise ValueError('nonempty rows of number weights required')
    # Existing owner validates the full Gaussian and first weight row.
    gaussian_partial_number_trace(m, r, log_ground, np.arange(len(m)), z[0])
    if (z.shape[1] != len(m) or not np.all(np.isfinite(z)) or np.any(abs(z)>1+1e-12)
            or isinstance(chunk_size,bool) or not isinstance(chunk_size,int) or chunk_size<1
            or isinstance(maximum_workspace_bytes,bool) or not isinstance(maximum_workspace_bytes,int)
            or maximum_workspace_bytes<1):
        raise ValueError('bounded finite number weights and positive chunk/workspace budgets required')
    capacity=maximum_workspace_bytes//(16*(10*len(m)**2+12*len(m)
        +4*len(m)*len(constants)+3*len(constants)**2))
    if capacity<1:
        raise MemoryError('vacuum Gaussian trace exceeds declared workspace')
    chunk_size=min(chunk_size,capacity)
    result=np.empty(len(z),complex)
    kernels=np.empty((len(z),len(constants),len(constants)),complex)
    for first in range(0,len(z),chunk_size):
        weights=z[first:first+chunk_size]
        product=(weights[:,:,None]*m.conj()[None]*weights[:,None,:])@m
        eigen=np.linalg.eigvals(product)
        if np.any(abs(eigen)>1+1e-7) or np.min(abs(1-eigen))<1e-12:
            raise FloatingPointError('ill-conditioned Gaussian vacuum trace')
        rhs=weights*r.conj()+weights*np.einsum('ij,tj->ti',m.conj(),weights*r)
        right_hand=np.concatenate([rhs[:,:,None],
            weights[:,:,None]*insertion.conj().T[None]],axis=2)
        solved=np.linalg.solve(np.eye(len(m))[None]-product,right_hand)
        left=solved[:,:,0]
        right=weights*r+weights*np.einsum('ij,tj->ti',m,left)
        exponent=2*log_ground-.5*np.log1p(-eigen).sum(axis=1)
        exponent+=.5*np.einsum('i,ti->t',r,left)+.5*np.einsum('i,ti->t',r.conj(),right)
        result[first:first+len(weights)]=np.exp(exponent)
        polynomial_left=constants[None]+np.einsum('ad,td->ta',insertion,left)
        polynomial_right=constants.conj()[None]+np.einsum('ad,td->ta',insertion.conj(),right)
        kernels[first:first+len(weights)]=result[first:first+len(weights),None,None]*(
            polynomial_left[:,:,None]*polynomial_right[:,None,:]
            +np.einsum('ad,tdb->tab',insertion,solved[:,:,1:]))
    return dict(vacuum_trace=result,operator_kernel=kernels)
