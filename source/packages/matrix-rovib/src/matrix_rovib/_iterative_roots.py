"""Lowest-root checks for complex Hermitian matrix-free Ritz subspaces."""
from __future__ import annotations
import numpy as np


def complete_lowest_ritz_subspace(apply, values, vectors, *, spectral_bound, seed=314159, tolerance=2.e-12):
    """Find lower omitted roots by an orthogonal-complement Lanczos solve.

    Small residuals certify individual eigenpairs, but cannot detect a missed
    member of a degenerate eigenspace. Lift the selected Ritz span above an
    operator norm bound and solve for the lowest root in its complement. If it
    precedes the selected cutoff, include it and repeat Rayleigh--Ritz. The
    retained dimension stays fixed; no degeneracy multiplicity is assumed.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    values = np.asarray(values, dtype=float)
    vectors, _ = np.linalg.qr(np.asarray(vectors, dtype=complex), mode="reduced")
    roots = len(values); dimension = len(vectors)
    rng = np.random.default_rng(seed)
    shift = 2 * max(1., float(spectral_bound), float(np.max(np.abs(values), initial=0.)))
    for _ in range(dimension):
        products = apply(vectors)
        projected = vectors.conj().T @ products
        values, rotation = np.linalg.eigh(.5*(projected+projected.conj().T))
        vectors = vectors @ rotation; products = products @ rotation
        residuals = np.linalg.norm(products-vectors*values,axis=0)
        if roots >= dimension:
            return values, vectors
        def complementary(values_in):
            array=np.asarray(values_in,dtype=complex); vector=array.ndim==1
            columns=array.reshape(dimension,-1)
            selected=vectors@(vectors.conj().T@columns)
            orthogonal=columns-selected
            acted=apply(orthogonal)
            result=acted-vectors@(vectors.conj().T@acted)+shift*selected
            return result[:,0] if vector else result
        operator=LinearOperator((dimension,dimension),matvec=complementary,matmat=complementary,dtype=np.complex128)
        start=rng.normal(size=dimension)+1j*rng.normal(size=dimension)
        start-=vectors@(vectors.conj().T@start)
        start/=np.linalg.norm(start)
        omitted, trial=eigsh(operator,k=1,which='SA',v0=start,tol=tolerance,maxiter=max(1000,20*dimension))
        trial=trial[:,0]; trial-=vectors@(vectors.conj().T@trial); trial/=np.linalg.norm(trial)
        trial_product=apply(trial)
        omitted_value=float(np.vdot(trial,trial_product).real)
        uncertainty=max(1.e-8,4*float(np.max(residuals,initial=0.)),
                        4*float(np.linalg.norm(trial_product-omitted_value*trial)),
                        256*float(np.spacing(float(np.max(np.abs(values),initial=0.)))))
        if omitted_value >= values[-1]-uncertainty:
            return values, vectors
        enlarged=np.column_stack((vectors,trial))
        acted=np.column_stack((products,trial_product))
        small=enlarged.conj().T@acted
        _, rotation=np.linalg.eigh(.5*(small+small.conj().T))
        vectors=enlarged@rotation[:,:roots]
    raise RuntimeError('lowest-root completion did not stabilize')
