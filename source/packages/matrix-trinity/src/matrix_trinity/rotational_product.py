"""Full vibrational-product rotation blocks without a band contraction."""
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
from .coupled_bath_galerkin import _TensorSumOperator
from scipy.sparse.linalg import LinearOperator


def unfactored_rotation_operator(hamiltonian,linear,quadratic,rotation,*,fixed_quadratic=None):
    """Full finite-J product for matrix-free nuclear and rotational vertices.

    No vibrational eigenstate contraction is made. Nuclear indices precede
    rotational indices. All operators must act on the same nuclear space.
    """
    axes=tuple(np.asarray(v,complex) for v in rotation);n=hamiltonian.shape[0]
    if (len(axes)!=3 or axes[0].ndim!=2 or axes[0].shape[0]!=axes[0].shape[1]
            or any(v.shape!=axes[0].shape or not np.allclose(v,v.conj().T) for v in axes)
            or len(linear)!=3 or len(quadratic)!=3
            or any(v.shape!=(n,n) for v in linear)
            or any(len(row)!=3 or any(v.shape!=(n,n) for v in row) for row in quadratic)):
        raise ValueError('Matching complete nuclear operators and three Hermitian rotation matrices required')
    nr=len(axes[0]);terms=[(linear[a],axes[a]) for a in range(3)]
    terms.extend((quadratic[a][b],(axes[a]@axes[b]+axes[b]@axes[a])/2) for a in range(3) for b in range(3))
    terms=[(left,right) for left,right in terms if np.any(right)]
    fixed=np.zeros((3,3)) if fixed_quadratic is None else np.asarray(fixed_quadratic,float)
    if fixed.shape!=(3,3) or not np.all(np.isfinite(fixed)) or not np.allclose(fixed,fixed.T):
        raise ValueError('Finite symmetric reference tensor required')
    rotor=sum(fixed[a,b]*(axes[a]@axes[b]+axes[b]@axes[a])/2 for a in range(3) for b in range(3))
    def action(vector):
        x=np.asarray(vector).reshape(n,nr)
        y=hamiltonian@x+x@rotor.T
        for left,right in terms:y+=(left@x)@right.T
        return y.ravel()
    result=LinearOperator((n*nr,n*nr),matvec=action,dtype=complex)
    if hasattr(hamiltonian,'diagonal_values') and all(hasattr(v,'diagonal_values') for v,_ in terms):
        result.diagonal_values=(np.repeat(hamiltonian.diagonal_values,nr)+np.tile(rotor.diagonal(),n)
            +sum(np.outer(v.diagonal_values,r.diagonal()).ravel() for v,r in terms))
    return result


def product_rotation_operator(coupled, rotation, *, fixed_quadratic=None):
    """Retain every J and sym(JJ) vertex in the original Galerkin space.

    Ordering is active coordinate, oscillator state, rotational state, as in
    LOCK's contracted LAM--rotation product. Coefficients retain input units.
    """
    rotation = tuple(np.asarray(r, complex) for r in rotation)
    if (len(rotation) != 3 or rotation[0].ndim != 2
            or rotation[0].shape[0] != rotation[0].shape[1]
            or any(r.shape != rotation[0].shape or not np.all(np.isfinite(r))
                   or not np.allclose(r, r.conj().T) for r in rotation)):
        raise ValueError('Three compatible Hermitian rotational operators required')
    h = coupled.hamiltonian_cm
    nr = len(rotation[0])
    if isinstance(h, _TensorSumOperator):
        terms = [(a, kron(b, eye(nr), format='csr')) for a, b in h.terms]
    elif (isinstance(h, np.ndarray) and coupled.bath_dimension == 1
            and h.shape == (coupled.active_dimension, coupled.active_dimension)
            and np.all(np.isfinite(h)) and np.allclose(h, h.conj().T)):
        # An active-only weak contraction already owns the complete nuclear
        # matrix. Retain it in its original basis instead of diagonalizing
        # every nuclear state or constructing a dense rotational product.
        terms = [(h, eye(nr, format='csr'))]
    else:
        raise ValueError('Factorized or Hermitian active-only Galerkin Hamiltonian required')
    for axis in range(3):
        terms.extend((a, kron(b, csr_matrix(rotation[axis]), format='csr'))
                     for a, b in coupled.linear_terms[axis])
        for other in range(3):
            sym = (rotation[axis]@rotation[other]+rotation[other]@rotation[axis])/2
            terms.extend((a, kron(b, csr_matrix(sym), format='csr'))
                         for a, b in coupled.quadratic_terms[axis][other])
    if fixed_quadratic is not None:
        fixed = np.asarray(fixed_quadratic, float)
        if fixed.shape != (3, 3) or not np.all(np.isfinite(fixed)) or not np.allclose(fixed, fixed.T):
            raise ValueError('Finite symmetric fixed rotational tensor required')
        rotor = sum((fixed[a, b]*(rotation[a]@rotation[b]+rotation[b]@rotation[a])/2
                     for a in range(3) for b in range(3)), start=np.zeros((nr, nr), complex))
        terms.append((np.eye(coupled.active_dimension),
                      kron(eye(coupled.bath_dimension), csr_matrix(rotor), format='csr')))
    return _TensorSumOperator(terms, coupled.active_dimension, coupled.bath_dimension*nr)
