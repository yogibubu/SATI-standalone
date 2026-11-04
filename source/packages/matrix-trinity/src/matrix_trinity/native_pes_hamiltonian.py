"""Finite HO Hamiltonians from direct native potential quadrature.

No force constants are reconstructed. The callback supplies the full potential
in dimensionless Cartesian modal coordinates, including the quadratic term.
Quadrature and basis convergence remain separate from potential acceptance.
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import roots_hermite

from .vibronic_sparse import SparseVibrationalHamiltonian



@dataclass(frozen=True)
class NativeModalAction:
    """Same finite Galerkin operator, retaining no dense Hamiltonian matrix."""
    basis: np.ndarray
    frequencies_cm: np.ndarray
    operator_cm: object
    diagnostics: dict

def build_native_modal_hamiltonian(frequencies_cm, potential_evaluator, *,
        potential_identifier, maximum_total_quanta, maximum_harmonic_energy_cm,
        quadrature_points, maximum_states=4096, maximum_grid_points=1000000,
        maximum_workspace_bytes=8*1024**3, batch_size=128, progress=None,
        projection_method="dense", action_only=False):
    """Galerkin projection of an unchanged potential using Gauss--Hermite nodes.

    The exact HO Hamiltonian is used as an algebraic reference; its quadratic
    potential is subtracted at *every quadrature node* before projection.
    This does not attach a quadratic tail or change the supplied potential.
    The finite matrix is dense in general and is returned in the common CSR
    container so existing spectral filters and Duschinsky overlap consumers
    can use it. No coefficient or matrix-element pruning is performed.
    With ``projection_method='tensor', action_only=True``, return a
    NativeModalAction using the same finite operator without materializing H.
    """
    if not isinstance(action_only, bool) or (action_only and projection_method != 'tensor'):
        raise ValueError('action-only native projection requires the tensor method')
    if projection_method not in ('dense', 'tensor'):
        raise ValueError('native projection method must be dense or tensor')
    frequencies = np.asarray(frequencies_cm, float)
    if (frequencies.ndim != 1 or not len(frequencies)
            or np.any(frequencies <= 0) or not np.all(np.isfinite(frequencies))
            or not callable(potential_evaluator) or not potential_identifier):
        raise ValueError('positive frequencies and an identified complete potential required')
    controls = (maximum_total_quanta,quadrature_points,maximum_states,
                maximum_grid_points,maximum_workspace_bytes,batch_size)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in controls):
        raise ValueError('positive integer basis, quadrature and memory budgets required')
    if not np.isfinite(maximum_harmonic_energy_cm) or maximum_harmonic_energy_cm<=0:
        raise ValueError('positive finite harmonic excitation bound required')
    rows = []
    def visit(prefix, energy):
        i = len(prefix)
        if i == len(frequencies):
            rows.append(tuple(prefix))
            if len(rows)>maximum_states:
                raise MemoryError('direct native Hamiltonian exceeds the declared basis budget')
            return
        limit = min(maximum_total_quanta-sum(prefix),
                    int(np.floor((maximum_harmonic_energy_cm-energy)/frequencies[i]+1e-12)))
        for occupation in range(limit+1):
            visit(prefix+[occupation],energy+occupation*frequencies[i])
    visit([],0.)
    rows.sort(key=lambda row:(sum(row),row))
    basis = np.asarray(rows,dtype=np.int64)
    maximum_powers = basis.max(axis=0)
    if quadrature_points <= int(maximum_powers.max())+1:
        raise ValueError('quadrature must represent every retained HO function and its quadratic reference')
    size = quadrature_points**len(frequencies)
    if size>maximum_grid_points:
        raise MemoryError('direct native quadrature exceeds the declared point budget')
    workspace = (0 if action_only else 64*len(rows)**2) + 32*batch_size*len(rows) + 16*size*len(frequencies)
    if workspace>maximum_workspace_bytes:
        raise MemoryError('direct native matrix exceeds the declared workspace budget')
    axis, weights = roots_hermite(quadrature_points)
    polynomials = np.empty((int(maximum_powers.max())+1,quadrature_points))
    polynomials[0] = np.pi**(-.25)
    if len(polynomials)>1:
        polynomials[1] = np.sqrt(2)*axis*polynomials[0]
    for n in range(2,len(polynomials)):
        polynomials[n] = (np.sqrt(2/n)*axis*polynomials[n-1]
                          -np.sqrt((n-1)/n)*polynomials[n-2])
    functions = polynomials*np.sqrt(weights)
    gram_error = float(np.max(abs(functions@functions.T-np.eye(len(functions)))))
    if gram_error>1e-10:
        raise ArithmeticError('Gauss--Hermite basis orthogonality failed')
    diagonal = (basis+.5)@frequencies
    matrix = None if action_only else np.diag(diagonal)
    values = np.empty(size)
    started = time.monotonic()
    for start in range(0,size,batch_size):
        take = np.arange(start,min(size,start+batch_size))
        indices = np.array(np.unravel_index(take,(quadrature_points,)*len(frequencies))).T
        q = axis[indices]
        sampled = np.asarray(potential_evaluator(q))
        if np.iscomplexobj(sampled) or sampled.shape!=(len(take),) or not np.all(np.isfinite(sampled)):
            raise ValueError('complete potential must return one real finite cm-1 energy per point')
        values[take] = sampled
        residual = sampled-.5*np.sum(frequencies*q*q,axis=1)
        if projection_method == 'dense':
            projection = np.ones((len(take),len(basis)))
            for mode in range(len(frequencies)):
                projection *= functions[basis[:,mode][None,:],indices[:,mode,None]]
            matrix += projection.T@(residual[:,None]*projection)
        if progress is not None:
            progress(dict(acquired_points=int(take[-1])+1,total_points=size,
                          elapsed_seconds=time.monotonic()-started))
    if projection_method == 'tensor':
        from scipy.sparse import diags
        from .native_tensor_operator import NativeTensorQuadratureOperator
        # Same nodes, weights and every residual value as the dense projector.
        # Reuse modal tensor transforms instead of outer products per node.
        indices = np.array(np.unravel_index(np.arange(size),
                           (quadrature_points,)*len(frequencies))).T
        residual = values - .5*np.sum(frequencies*axis[indices]**2,axis=1)
        operator = NativeTensorQuadratureOperator(basis,
            tuple(functions[:power+1] for power in maximum_powers), residual,
            reference=diags(diagonal), maximum_workspace_bytes=maximum_workspace_bytes,
            maximum_batch_size=min(batch_size,32))
        if not action_only:
            matrix = operator.matmat(np.eye(len(basis)))
    if not action_only:
        matrix = .5*(matrix+matrix.T)
    diagnostics = dict(schema='matrix.trinity.direct_native_ho_hamiltonian.v1',
        hamiltonian='HO kinetic plus directly quadrature-projected complete native potential',
        potential_identifier=str(potential_identifier),
        maximum_total_quanta=maximum_total_quanta,
        maximum_harmonic_energy_cm=maximum_harmonic_energy_cm,
        quadrature_points_per_axis=quadrature_points,quadrature_point_count=size,
        basis_size=len(basis),workspace_guard_bytes=workspace,
        quadrature_orthogonality_error=gram_error,
        minimum_sampled_potential_cm=float(values.min()),
        maximum_sampled_potential_cm=float(values.max()),
        polynomial_reexpansion_used=False,potential_modified=False,
        coefficient_pruning_used=False,quadrature_convergence_certified=False,
        infinite_basis_convergence_certified=False,
        projection_method=projection_method,dense_hamiltonian_constructed=not action_only,
        acquisition_seconds=time.monotonic()-started)
    if action_only:
        return NativeModalAction(basis,frequencies,operator,diagnostics)
    return SparseVibrationalHamiltonian(basis,frequencies,csr_matrix(matrix),diagnostics)
