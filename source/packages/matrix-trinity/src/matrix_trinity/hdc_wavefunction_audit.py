"""Same-Hamiltonian wavefunction diagnostics and residual-controlled Ritz refinement."""
import numpy as np
from scipy.linalg import eigh, svd, qr
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


def assess_hdc_energy_frame(hamiltonian, vectors, prescribed_energies_cm):
    """Quantify the energetic deformation attached to an orthonormal frame.

    Numerical Ritz residuals, prescribed-energy residuals, and changes in
    pair spacings are independent outputs. The latter are invariant to an
    additive energy reference. Exactly zero Rayleigh gaps carry an explicit
    mask instead of an inferred coupling ratio. All states use the supplied
    ordering; no frequency matching or molecular threshold is introduced.
    """
    h = np.asarray(hamiltonian)
    u = np.asarray(vectors, complex)
    e = np.asarray(prescribed_energies_cm, float)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or u.ndim != 2
            or u.shape[0] != h.shape[0] or not u.shape[1]
            or e.shape != (u.shape[1],)
            or not all(np.all(np.isfinite(a)) for a in (h, u, e))
            or not np.allclose(h, h.conj().T, atol=1e-9, rtol=1e-12)
            or not np.allclose(u.conj().T @ u, np.eye(u.shape[1]), atol=1e-9)):
        raise ValueError("finite common Hermitian Hamiltonian, orthonormal frame and energy list required")
    acted = h @ u
    projected = u.conj().T @ acted
    rayleigh = np.diag(projected).real
    residuals = np.linalg.norm(acted - u * rayleigh, axis=0)
    shifts = e - rayleigh
    native_gaps = rayleigh[:, None] - rayleigh[None, :]
    model_gaps = e[:, None] - e[None, :]
    resolved = native_gaps != 0
    ratios = np.divide(model_gaps, native_gaps,
                       out=np.zeros_like(native_gaps), where=resolved)
    centered = np.diag(e - e.mean()) - (projected - np.eye(len(e)) * rayleigh.mean())
    return dict(
        schema="matrix.trinity.hdc_energy_frame_assessment.v1",
        mechanical_rayleigh_energies_cm=rayleigh,
        mechanical_residual_norms_cm=residuals,
        prescribed_energy_residual_norms_cm=np.linalg.norm(acted - u * e, axis=0),
        residual_identity_rhs_cm=np.hypot(residuals, shifts),
        prescribed_minus_rayleigh_cm=shifts,
        excitation_corrections_cm=shifts - shifts[0],
        pair_gap_changes_cm=model_gaps - native_gaps,
        pair_gap_ratios=ratios,
        pair_gap_ratio_defined=resolved,
        pair_order_preserved=(model_gaps * native_gaps >= 0),
        centered_projected_deformation_frobenius_cm=float(np.linalg.norm(centered)),
        partial_frame=u.shape[1] < len(h),
    )


def audit_hdc_wavefunctions(hamiltonian, vectors, reference_vectors, *, operators=()):
    """Independent overlap, invariant subspace, residual and amplitude checks.

    Reference vectors must use exactly the same basis/metric/operator. Energy
    differences alone cannot qualify a wavefunction reconstruction.
    """
    h = np.asarray(hamiltonian)
    u = np.asarray(vectors, complex)
    ref = np.asarray(reference_vectors, complex)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or u.ndim != 2 or ref.ndim != 2
            or u.shape[0] != h.shape[0] or ref.shape[0] != h.shape[0]
            or ref.shape[1] < u.shape[1] or not len(u.T)
            or not all(np.all(np.isfinite(a)) for a in (h,u,ref))
            or not np.allclose(h,h.conj().T,atol=1e-9,rtol=1e-12)
            or not np.allclose(u.conj().T@u,np.eye(u.shape[1]),atol=1e-9)
            or not np.allclose(ref.conj().T@ref,np.eye(ref.shape[1]),atol=1e-9)):
        raise ValueError("common finite Hermitian Hamiltonian and orthonormal wavefunctions required")
    overlap = u.conj().T@ref
    _, ids = linear_sum_assignment(-abs(overlap)**2)
    target = ref[:,ids]
    pivots = np.diag(u.conj().T@target)
    target = target * np.conj(np.divide(pivots,abs(pivots),out=np.ones_like(pivots),where=abs(pivots)>0))
    hu = h@u
    rayleigh = np.sum(u.conj()*hu,axis=0).real
    residual = np.linalg.norm(hu-u*rayleigh,axis=0)
    props = []
    for op in operators:
        op = np.asarray(op)
        if op.shape[:2] != h.shape or not np.all(np.isfinite(op)):
            raise ValueError("operator must use the same common basis")
        a = np.einsum('im,ij...,jn->mn...',u.conj(),op,u,optimize=True)
        b = np.einsum('im,ij...,jn->mn...',target.conj(),op,target,optimize=True)
        props.append(dict(maximum_amplitude_error=float(np.max(abs(a-b))),
                          maximum_strength_error=float(np.max(abs(abs(a)**2-abs(b)**2)))))
    return dict(reference_assignment=ids, squared_overlaps=abs(overlap[np.arange(len(ids)),ids])**2,
        minimum_squared_subspace_overlap=float(np.min(svd(u.conj().T@target,compute_uv=False))**2),
        physical_residual_norms_cm=residual,rayleigh_energies_cm=rayleigh,
        property_comparisons=props,energy_prescription_used_as_accuracy_test=False)


def refine_hdc_wavefunctions(hamiltonian, vectors, prescribed_cm, *,
                             residual_tolerance_cm=1e-5, maximum_iterations=32,
                             maximum_subspace=512, workers=1):
    """Targeted block-Krylov/Ritz refinement without changing prescribed energies.

    Residuals enlarge one collective subspace; no molecule-specific resonance
    list or fitted mixing angle. The physical Ritz values are diagnostics,
    never substituted for the HDC polynomial energies. A finite local space
    remains a declared model; no global confinement is inferred.
    """
    u = np.asarray(vectors, complex)
    e = np.asarray(prescribed_cm,float)
    if (u.ndim != 2 or e.shape != (u.shape[1],) or not len(e)
            or not np.all(np.isfinite(u)) or not np.all(np.isfinite(e))
            or not np.allclose(u.conj().T@u,np.eye(len(e)),atol=1e-9)
            or not np.isfinite(residual_tolerance_cm) or residual_tolerance_cm <= 0
            or maximum_iterations < 1 or maximum_subspace < len(e)):
        raise ValueError("orthonormal targets, prescribed energies and valid refinement budgets required")
    n,k = u.shape
    if getattr(hamiltonian,'shape',None) != (n,n):
        raise ValueError("Hamiltonian and target basis dimensions differ")
    if isinstance(hamiltonian,np.ndarray):
        if not np.all(np.isfinite(hamiltonian)) or not np.allclose(hamiltonian,hamiltonian.conj().T,atol=1e-9,rtol=1e-12):
            raise ValueError("physical Hamiltonian must be finite and Hermitian")
    basis = qr(u, mode='economic', check_finite=False)[0]
    followed = u.copy()
    history = []
    with threadpool_limits(limits=workers,user_api='blas'):
        acted = hamiltonian@basis
        for iteration in range(maximum_iterations):
            small = basis.conj().T@acted
            values, rotation = eigh((small+small.conj().T)/2,check_finite=False)
            candidates = basis@rotation
            _, take = linear_sum_assignment(-abs(followed.conj().T@candidates)**2)
            followed = candidates[:,take]
            ritz = values[take]
            residual = acted@rotation[:,take]-followed*ritz
            norms = np.linalg.norm(residual,axis=0)
            history.append(dict(iteration=iteration,subspace_size=basis.shape[1],maximum_residual_cm=float(max(norms))))
            if max(norms) <= residual_tolerance_cm:
                break
            if iteration+1 == maximum_iterations or basis.shape[1] >= min(n,maximum_subspace):
                break
            for _ in range(2):
                residual -= basis@(basis.conj().T@residual)
            add, singular, _ = svd(residual,full_matrices=False,check_finite=False)
            keep = singular > max(n,k)*np.finfo(float).eps*max(1.,float(max(singular)))
            add = add[:,keep][:,:min(n,maximum_subspace)-basis.shape[1]]
            if not add.shape[1]:
                break
            # SVD directions associated with small singular values can regain
            # components in the existing space. Reorthogonalize the entire
            # expanded basis before the next ordinary Rayleigh--Ritz solve.
            basis = qr(np.column_stack((basis,add)), mode='economic', check_finite=False)[0]
            acted = hamiltonian@basis
    return dict(vectors=followed,energies_cm=e.copy(),physical_ritz_values_cm=ritz,
        residual_norms_cm=norms,converged=bool(max(norms)<=residual_tolerance_cm),
        iterations=history,prescribed_energies_modified=False,
        partial_frame=True,finite_model_only=True)
