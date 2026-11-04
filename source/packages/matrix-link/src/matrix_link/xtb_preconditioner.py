"""Bounded modal acceleration of the ORACLE Cartesian xTB model.

Ritz vectors supplement, rather than replace, the full projected tangent.
No dense Cartesian Hessian or complete normal-mode basis is constructed.
"""
from .sycart_matrix_free import davidson_inverse_preconditioner


def xtb_davidson_preconditioner(source, runtime, *, initial_vectors=None):
    project = runtime.project
    action = lambda v: project(source.cartesian_action(project(v)))
    inverse = davidson_inverse_preconditioner(action, project,
        dimension=runtime.cartesian_dimension, active_rank=runtime.active_rank,
        roots=min(12, runtime.active_rank), maximum_iterations=64,
        maximum_subspace=96, initial_vectors=initial_vectors,
        baseline_curvature=0.5, krylov_maximum_iterations=64)
    audit = dict(source.audit, curvature_source='ORACLE_XTB_ANC_LOCAL_CARTESIAN',
        conditioning_variant=source.audit['model'],
        hessian_action='P_H_XTB_ANC_P', dense_matrix_materialized=False,
        initial_hessian_uses_gradients=False,
        cartesian_dimension=runtime.cartesian_dimension, active_rank=runtime.active_rank,
        davidson_requested_roots=min(12, runtime.active_rank),
        davidson_retained_roots=int(inverse.eigenvalues.size),
        davidson_iterations=inverse.iterations, davidson_maximum_iterations=64,
        davidson_maximum_subspace=96, davidson_converged=inverse.converged,
        davidson_eigenvalues=inverse.eigenvalues.tolist(),
        baseline_inverse=inverse.baseline_inverse,
        retained_complement='ALL_PROJECTED_CARTESIAN_DIRECTIONS',
        inverse_contract='BOUNDED_PRECONDITIONER_NOT_EXACT_INVERSE')
    return inverse, audit
