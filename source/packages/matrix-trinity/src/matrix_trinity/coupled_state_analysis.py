"""Population diagnostics for normalized active-coordinate/Fock eigenstates.

The supplied product basis must be orthonormal. Modal populations and
participation counts depend on that basis; reduced-state purity is invariant
to separate unitary changes within each complete retained factor.
"""
import numpy as np


def coupled_product_state_diagnostics(coefficients, occupations):
    """Analyze columns C[active, Fock word, state] without renormalizing them."""
    c=np.asarray(coefficients);words=np.asarray(occupations)
    if (c.ndim!=3 or words.ndim!=2 or c.shape[1]!=len(words) or not c.shape[0]
            or not c.shape[2] or not np.all(np.isfinite(c)) or not np.all(np.isfinite(words))
            or np.any(words<0) or np.any(words!=np.rint(words))):
        raise ValueError('Finite product coefficients and nonnegative integer Fock words required')
    if len(set(map(tuple,words)))!=len(words):raise ValueError('Unique Fock words required')
    norm=np.sum(abs(c)**2,axis=(0,1))
    if not np.allclose(norm,1,atol=1e-8,rtol=0):raise ValueError('Normalized states required; no rescaling is performed')
    rows=[]
    for k in range(c.shape[2]):
        wave=c[:,:,k];p=abs(wave)**2;active=p.sum(axis=1);bath=p.sum(axis=0)
        rho=wave@wave.conj().T;eigen=np.linalg.eigvalsh(rho)
        if min(eigen)<-1e-10:raise ValueError('Reduced density is not positive')
        zero=np.flatnonzero(np.all(words==0,axis=1))
        order=np.argsort(p.ravel())[::-1][:5]
        leading=[dict(active_modal_index=int(i//len(words)),
            occupation=words[i%len(words)].astype(int).tolist(),probability=float(p.ravel()[i])) for i in order]
        rows.append(dict(norm=float(norm[k]),active_modal_probabilities=active.tolist(),
            mean_bath_occupations=(bath@words).tolist(),
            mean_total_bath_quanta=float(bath@words.sum(axis=1)),
            bath_vacuum_probability=float(bath[zero].sum()),
            active_bath_purity=float(np.trace(rho@rho).real),
            dominant_Schmidt_probability=float(max(eigen)),
            joint_product_participation=float(1/np.sum(p*p)),leading_products=leading))
    return dict(states=rows,normalization_changed=False,
        basis_dependent_quantities=['modal probabilities','Fock occupations','product participation'],
        scope='Nuclear-state diagnostics, not an optical assignment or a completeness certificate')
