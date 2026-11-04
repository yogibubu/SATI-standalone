"""Local quartic states in an explicit Dirichlet box, projected onto HO states.

The finite box defines the local problem. No change is made to the polynomial;
box, grid and HO-projection convergence are separate required controls. These
states are not global bound states or computed resonance lifetimes.
"""
from math import comb

import numpy as np
from scipy.fft import dstn, idstn
from scipy.sparse.linalg import LinearOperator, eigsh
from threadpoolctl import threadpool_limits

from .hdc_hamiltonian import _shell_basis
from .vibronic_variational import FiniteVibrationalStates
from .vibronic_vpt2 import SAMVPT2Field, _positive_integer


def solve_local_quartic_ground_state(
    field, *, half_width=5.0, grid_points=39, maximum_total_quanta=16,
    maximum_grid_points=2_000_000, maximum_ho_states=8192,
    maximum_projection_loss=1e-6, eigensolver_tolerance=1e-11, workers=1,
):
    """Solve the unchanged local quartic with the common direct-PES solver."""
    if not isinstance(field, SAMVPT2Field):
        raise TypeError('SAMVPT2Field required')
    if np.any(field.metric_first_cm) or np.any(field.metric_second_cm):
        raise ValueError('local box solver requires potential-only fields')
    def potential(coordinates):
        return (0.5*np.sum(field.frequencies_cm*coordinates**2, axis=1)
            + np.einsum('ijk,si,sj,sk->s', field.cubic_cm, coordinates, coordinates, coordinates)/6
            + np.einsum('ijkl,si,sj,sk,sl->s', field.quartic_cm,
                        coordinates, coordinates, coordinates, coordinates)/24
            + field.constant_shift_cm)
    result=solve_local_potential_ground_state(field.frequencies_cm,potential,
        half_width=half_width,grid_points=grid_points,maximum_total_quanta=maximum_total_quanta,
        maximum_grid_points=maximum_grid_points,maximum_ho_states=maximum_ho_states,
        maximum_projection_loss=maximum_projection_loss,
        eigensolver_tolerance=eigensolver_tolerance,workers=workers,
        potential_identifier='supplied unchanged quartic field')
    diagnostics=dict(result.diagnostics,
        schema='matrix.sati.local_quartic_ground_state.v1',
        hamiltonian='H0+V3+V4 unchanged inside the declared box',
        quartic_coverage=field.quartic_coverage)
    return FiniteVibrationalStates(result.basis,result.energies_cm,
        result.eigenvectors,result.frequencies_cm,diagnostics)


def solve_local_potential_ground_state(
    frequencies_cm, potential_evaluator, *, half_width=5.0, grid_points=39, maximum_total_quanta=16,
    maximum_grid_points=2_000_000, maximum_ho_states=8192,
    maximum_projection_loss=1e-6, eigensolver_tolerance=1e-11, workers=1,
    potential_identifier="direct native potential",
):
    """Solve a directly evaluated PES in a constant-diagonal modal metric.

    The callback accepts a (point, mode) array of dimensionless coordinates
    and returns the COMPLETE potential in cm^-1 for every point, including
    its quadratic term. No Taylor expansion or polynomial fit is performed.
    Frequencies define the fixed Cartesian-modal kinetic operator and HO
    projection units. Nonlinear internal-coordinate paths require their own
    metric-aware kinetic operator and must not be passed as Cartesian modes.
    The Dirichlet box remains explicit and needs independent convergence.
    """
    frequencies_cm = np.asarray(frequencies_cm, dtype=float)
    if (frequencies_cm.ndim != 1 or not len(frequencies_cm)
            or not np.all(np.isfinite(frequencies_cm)) or np.any(frequencies_cm <= 0)
            or not callable(potential_evaluator) or not isinstance(potential_identifier, str)
            or not potential_identifier.strip()):
        raise ValueError('positive modal frequencies and an identified potential callback required')
    for name, value in (('grid_points', grid_points), ('maximum_total_quanta', maximum_total_quanta),
                        ('maximum_grid_points', maximum_grid_points), ('maximum_ho_states', maximum_ho_states),
                        ('workers', workers)):
        _positive_integer(value, name)
    if grid_points < 3:
        raise ValueError('at least three interior grid points per coordinate required')
    if (not np.isfinite(half_width) or half_width <= 0
            or not np.isfinite(maximum_projection_loss) or not 0 < maximum_projection_loss < 1
            or not np.isfinite(eigensolver_tolerance) or eigensolver_tolerance <= 0):
        raise ValueError('positive finite box width, eigensolver tolerance and projection-loss bound required')
    n = len(frequencies_cm)
    size = grid_points**n
    ho_size = comb(n+maximum_total_quanta, n)
    if (size > maximum_grid_points or ho_size > maximum_ho_states
            or (maximum_total_quanta+1)**n > maximum_grid_points):
        raise MemoryError('local grid or HO projection exceeds its declared state budget')
    step = 2*half_width/(grid_points+1)
    axis = -half_width+step*np.arange(1, grid_points+1)
    mesh = np.meshgrid(*([axis]*n), indexing='ij')
    coordinates = np.array(mesh).reshape(n, -1).T
    potential = np.asarray(potential_evaluator(coordinates))
    if np.iscomplexobj(potential):
        raise ValueError('direct potential must be real')
    potential = np.asarray(potential, dtype=float)
    if potential.shape != (size,) or not np.all(np.isfinite(potential)):
        raise ValueError('direct potential must return one finite cm^-1 energy per point')
    shape = (grid_points,)*n
    kinetic = np.zeros(shape)
    for i, frequency in enumerate(frequencies_cm):
        index_shape = [1]*n
        index_shape[i] = grid_points
        kinetic += (frequency/2*(np.pi*np.arange(1, grid_points+1)/(2*half_width))**2).reshape(index_shape)
    def action(vector):
        values = vector.reshape(shape)
        return (idstn(kinetic*dstn(values, type=1, norm='ortho'), type=1, norm='ortho').ravel()
                +potential*vector)
    operator = LinearOperator((size, size), matvec=action, dtype=float)
    guess = np.exp(-np.sum(coordinates**2, axis=1)/2)
    guess /= np.linalg.norm(guess)
    with threadpool_limits(limits=workers, user_api='blas'):
        energy, vectors = eigsh(operator, k=1, which='SA', v0=guess,
                               tol=eigensolver_tolerance, maxiter=10000)
    vector = vectors[:, 0]
    if vector @ guess < 0:
        vector = -vector
    residual = float(np.linalg.norm(action(vector)-energy[0]*vector))
    density = vector**2
    boundary = np.any(abs(coordinates) >= .8*half_width, axis=1)
    functions = np.zeros((maximum_total_quanta+1, grid_points))
    functions[0] = np.pi**(-.25)*np.exp(-axis**2/2)
    functions[1] = np.sqrt(2)*axis*functions[0]
    for k in range(2, len(functions)):
        functions[k] = np.sqrt(2/k)*axis*functions[k-1]-np.sqrt((k-1)/k)*functions[k-2]
    coefficients = vector.reshape(shape)
    for dimension in range(n):
        coefficients = np.moveaxis(np.tensordot(functions*np.sqrt(step), coefficients,
                                               axes=(1, dimension)), 0, dimension)
    basis = np.asarray(_shell_basis(n, maximum_total_quanta), np.int64)
    projected = coefficients[tuple(basis.T)]
    retained_norm = float(projected @ projected)
    if not 1-maximum_projection_loss <= retained_norm <= 1+maximum_projection_loss:
        raise ValueError(f'local HO projection norm {retained_norm:.12g} exceeds its loss/aliasing bound')
    projected /= np.sqrt(retained_norm)
    diagnostics = dict(schema='matrix.trinity.local_direct_potential_ground_state.v1',
        method='Dirichlet sine spectral box with HO projection',
        hamiltonian='constant diagonal modal kinetic plus directly evaluated potential',
        potential_identifier=potential_identifier,
        polynomial_reexpansion_used=False,
        coordinate_metric='constant diagonal Cartesian-modal subspace',
        potential_modified=False, hdc_completion_used=False,
        global_bound_state_claimed=False, local_state_convergence_certified=False,
        half_width_dimensionless=float(half_width), grid_points_per_axis=grid_points,
        grid_step_dimensionless=float(step), grid_size=size,
        finite_box_energy_cm=float(energy[0]), grid_residual_norm_cm=residual,
        outer_box_probability=float(density[boundary].sum()),
        coordinate_means=coordinates.T @ density,
        coordinate_second_moments=(coordinates**2).T @ density,
        maximum_total_quanta=maximum_total_quanta, basis_size=ho_size,
        ho_projection_retained_norm=retained_norm,
        outer_ho_shell_probability=float(np.sum(projected[basis.sum(axis=1)>maximum_total_quanta-4]**2)),
        quartic_coverage=None)
    from matrix_core.owner_evidence import emit_owner_validation
    emit_owner_validation('SATI', 'LOCAL_MODAL_STATE_SOLVE', passed=True,
        metrics={'grid_residual_norm': residual, 'projection_norm': retained_norm,
                 'energy': float(energy[0]), 'outer_box_probability': diagnostics['outer_box_probability']},
        units={'grid_residual_norm': 'cm-1', 'projection_norm': 'dimensionless',
               'energy': 'cm-1', 'outer_box_probability': 'dimensionless'},
        thresholds={'maximum_projection_loss': maximum_projection_loss},
        context={'scope': 'One local solve; independent box/grid convergence not certified'})
    return FiniteVibrationalStates(basis, energy, projected[:, None], frequencies_cm, diagnostics)


def truncate_local_ground_state_projection(state, maximum_total_quanta, *, maximum_projection_loss=1e-6):
    """Reduce a saved local ground-state HO projection and report its norm loss."""
    if not isinstance(state, FiniteVibrationalStates) or len(state.energies_cm) != 1:
        raise ValueError('one finite local ground state required')
    _positive_integer(maximum_total_quanta, 'maximum_total_quanta')
    if (not np.isfinite(maximum_projection_loss) or not 0 < maximum_projection_loss < 1
            or maximum_total_quanta > state.diagnostics['maximum_total_quanta']):
        raise ValueError('valid loss bound and a shell cutoff no larger than the saved projection required')
    keep = state.basis.sum(axis=1) <= maximum_total_quanta
    coefficients = state.eigenvectors[keep, 0]
    retained = float(np.vdot(coefficients, coefficients).real)
    grid_norm = retained*state.diagnostics['ho_projection_retained_norm']
    if not 1-maximum_projection_loss <= grid_norm <= 1+maximum_projection_loss:
        raise ValueError('reduced local HO projection exceeds its declared loss bound')
    coefficients = coefficients/np.sqrt(retained)
    info = dict(state.diagnostics, source_projection_cutoff=state.diagnostics['maximum_total_quanta'],
                maximum_total_quanta=maximum_total_quanta, basis_size=int(keep.sum()),
                ho_projection_retained_norm=grid_norm,
                relative_projection_retained_norm=retained,
                outer_ho_shell_probability=float(np.sum(abs(coefficients[
                    state.basis[keep].sum(axis=1)>maximum_total_quanta-4])**2)))
    return FiniteVibrationalStates(state.basis[keep], state.energies_cm,
                                  coefficients[:, None], state.frequencies_cm, info)


__all__ = ['solve_local_quartic_ground_state', 'solve_local_potential_ground_state',
           'truncate_local_ground_state_projection']
