"""Finite-box spectral controls for directly sampled Cartesian modal potentials."""
import numpy as np
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import LinearOperator, eigsh


def solve_modal_sine_grid(frequencies_cm, potential_cm, *, half_width,
                          levels=4, tolerance=1e-10):
    """Return low levels and grid probabilities in an explicit Dirichlet box.

    Coordinates are dimensionless Cartesian normal modes; the kinetic metric
    is constant and diagonal. This finite-subspace calculation is not a full
    molecular convergence certificate. Potential values include the quadratic
    part and must be sampled at the returned interior sine-grid coordinates.
    """
    w = np.asarray(frequencies_cm, float)
    v = np.asarray(potential_cm, float)
    if (w.ndim != 1 or not len(w) or np.any(w <= 0)
            or v.ndim != len(w) or len(set(v.shape)) != 1
            or min(v.shape) < 3 or not np.all(np.isfinite(w))
            or not np.all(np.isfinite(v)) or not np.isfinite(half_width)
            or half_width <= 0 or not isinstance(levels, int)
            or not 1 <= levels < v.size-1 or tolerance <= 0):
        raise ValueError('positive frequencies and finite equal-axis modal grid required')
    n = v.shape[0]
    axis = -half_width + 2*half_width/(n+1)*np.arange(1, n+1)
    kinetic = np.zeros(v.shape)
    for i, frequency in enumerate(w):
        shape = [1]*len(w); shape[i] = n
        kinetic += (frequency/2*(np.pi*np.arange(1, n+1)/(2*half_width))**2).reshape(shape)
    def action(vector):
        grid = vector.reshape(v.shape)
        return (idstn(kinetic*dstn(grid, type=1, norm='ortho'),
                       type=1, norm='ortho') + v*grid).ravel()
    operator = LinearOperator((v.size, v.size), matvec=action, dtype=float)
    # A fixed nonsymmetric seed includes both parity sectors.
    guess = np.random.default_rng(17931).normal(size=v.size)
    energies, vectors = eigsh(operator, k=levels, which='SA', v0=guess,
                              tol=tolerance, maxiter=20000)
    order = np.argsort(energies); energies = energies[order]; vectors = vectors[:, order]
    coordinates = np.array(np.meshgrid(*([axis]*len(w)), indexing='ij')).reshape(len(w), -1).T
    density = vectors**2
    outer = np.any(abs(coordinates) >= .8*half_width, axis=1)
    return dict(energies_cm=energies, eigenvectors=vectors, coordinates=coordinates,
                outer_box_probability=density[outer].sum(axis=0),
                residuals_cm=np.array([np.linalg.norm(action(vectors[:, i])-energies[i]*vectors[:, i])
                                       for i in range(levels)]),
                coordinate_second_moments=(coordinates**2).T@density)
