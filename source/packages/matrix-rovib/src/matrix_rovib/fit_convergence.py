"""Local statistical stationarity for uncertainty-whitened least squares.

The diagnostic measures the remaining linearized correction in observable
space. It does not infer convergence from a small change in RMS, nor certify
model adequacy or a global minimum.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import norm
from scipy.optimize import OptimizeResult, lsq_linear


def assess_statistical_convergence(
    residual, jacobian, values, lower, upper, *, data_count,
    joint_tolerance=0.1, line_tolerance=0.01, rank_rcond=1.e-10,
    data_correction_transform=None,
):
    """Assess a bounded Gauss--Newton correction in whitened residual units.

    Experimental rows precede any priors/predicates. The joint tolerance is
    the norm of the predicted change of *all* whitened residuals; the line
    tolerance bounds each predicted experimental displacement in sigma units.
    Both tolerances are numerical accuracy budgets, not p-value thresholds.
    Column equilibration makes the calculation invariant to parameter units.
    Rank deficiency prevents certification, even if the fitted subspace is
    stationary. The bounds are imposed on the linearized problem itself.
    """
    r = np.asarray(residual, dtype=float)
    j = np.asarray(jacobian, dtype=float)
    x = np.asarray(values, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if (r.ndim != 1 or x.ndim != 1 or j.shape != (len(r), len(x))
            or lo.shape != x.shape or hi.shape != x.shape or not len(x)):
        raise ValueError("inconsistent convergence residual, Jacobian or bounds")
    if (not np.all(np.isfinite(r)) or not np.all(np.isfinite(j))
            or not np.all(np.isfinite(x)) or np.any(np.isnan(lo))
            or np.any(np.isnan(hi)) or np.any(lo >= hi)
            or np.any(x < lo) or np.any(x > hi)):
        raise ValueError("convergence requires finite inputs and feasible bounds")
    if (isinstance(data_count, bool) or int(data_count) != data_count
            or not 0 < data_count <= len(r)):
        raise ValueError("data_count must identify the leading experimental rows")
    if (not np.isfinite(joint_tolerance) or joint_tolerance <= 0
            or not np.isfinite(line_tolerance) or line_tolerance <= 0
            or not np.isfinite(rank_rcond) or not 0 < rank_rcond < 1):
        raise ValueError("invalid statistical convergence tolerance")
    column_norms = np.asarray([norm(j[:, k]) for k in range(len(x))])
    column_norms[column_norms == 0.] = 1.
    a = j / column_norms
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    keep = singular > (singular[0] * rank_rcond if len(singular) else 0.)
    rank = int(np.count_nonzero(keep))
    z = -(vh[keep].T @ ((u[:, keep].T @ r) / singular[keep]))
    step_lower = (lo - x) * column_norms
    step_upper = (hi - x) * column_norms
    bounded = bool(np.any(z < step_lower) or np.any(z > step_upper))
    linear_success = True
    if bounded:
        linear = lsq_linear(a, -r, bounds=(step_lower, step_upper),
                            method="bvls", tol=1.e-12, max_iter=1000)
        z = linear.x
        linear_success = bool(linear.success)
    change = a @ z
    joint = float(norm(change))
    data_change=change[:int(data_count)]
    if data_correction_transform is not None:
        transform=np.asarray(data_correction_transform,dtype=float)
        if transform.shape!=(int(data_count),int(data_count)) or not np.all(np.isfinite(transform)):
            raise ValueError("invalid standardized data correction transform")
        data_change=transform@data_change
    maximum = float(np.max(np.abs(data_change)))
    converged = bool(linear_success and rank == len(x)
                     and joint <= joint_tolerance and maximum <= line_tolerance)
    classification = ("converged" if converged else "linear_solver_failed"
                      if not linear_success else "rank_deficient"
                      if rank < len(x) else "not_stationary")
    return {
        "criterion": "bounded_whitened_gauss_newton",
        "classification": classification,
        "converged": converged,
        "joint_tolerance": float(joint_tolerance),
        "line_tolerance_sigma": float(line_tolerance),
        "joint_correction_norm": joint,
        "maximum_data_correction_sigma": maximum,
        "predicted_chi_square_reduction": float(max(0., r @ r - (r + change) @ (r + change))),
        "rank": rank,
        "parameter_count": len(x),
        "data_count": int(data_count),
        "augmented_row_count": len(r),
        "bounds_applied": bounded,
        "linear_solver_success": linear_success,
        "parameter_step": (z / column_norms).tolist(),
        "column_norms": column_norms.tolist(),
        "singular_values_equilibrated": singular.tolist(),
        "scope": "local linearized stationarity, not model adequacy or a global minimum",
    }


def bounded_gauss_newton(fun, jac, values, lower, upper, *, data_count,
                         max_nfev=100, rank_rcond=1.e-10):
    """Column-equilibrated bounded GN with an Armijo backtracking search.

    The SVD step follows correlated parameter combinations directly, rather
    than clipping each coefficient to a box-shaped numerical trust region.
    Physical parameter bounds remain in the linear subproblem. No small-RMS
    or small-step success condition is imposed: the caller's checked Jacobian
    supplies the statistical stopping condition. Budget and failed line search
    return unsuccessful results. The routine is for a linear residual loss.
    """
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")
    x = np.asarray(values, dtype=float).copy()
    r = np.asarray(fun(x), dtype=float)
    evaluations, jacobians = 1, 0
    status, message = 0, "The maximum number of function evaluations is exceeded."
    while True:
        j = np.asarray(jac(x), dtype=float)
        jacobians += 1
        if evaluations >= max_nfev:
            break
        diagnostic = assess_statistical_convergence(
            r, j, x, lower, upper, data_count=data_count, rank_rcond=rank_rcond,
        )
        step = np.asarray(diagnostic["parameter_step"])
        slope = float(r @ (j @ step))
        if slope >= 0. or not diagnostic["linear_solver_success"]:
            status, message = -3, "No descending feasible linearized step; statistical stationarity is not certified."
            break
        alpha, accepted = 1., False
        while evaluations < max_nfev:
            candidate = np.clip(x + alpha * step, lower, upper)
            if np.array_equal(candidate, x):
                break
            trial = np.asarray(fun(candidate), dtype=float)
            evaluations += 1
            if .5 * float(trial @ trial) <= .5 * float(r @ r) + 1.e-4 * alpha * slope:
                x, r, accepted = candidate, trial, True
                break
            alpha *= .5
        if not accepted:
            if evaluations < max_nfev:
                status, message = -3, "Line search stalled before statistical stationarity."
            break
    return OptimizeResult(
        x=x, success=False, status=status, message=message,
        nfev=evaluations, njev=jacobians, cost=.5 * float(r @ r),
        optimality=float(norm(j.T @ r, ord=np.inf)),
    )


def statistical_trust_region(fun, jac, values, lower, upper, scales, *,
                             max_nfev=100, initial_radius=1., initial_damping=0., trace=None,
                             trace_callback=None, state_callback=None):
    """Adaptive LM using the shared MORPHEUS More--Hebden SVD kernel.

    Scales define the dimensionless parameter metric. The caller's checked
    Jacobian alone certifies statistical stationarity. Rejected trials reuse
    the same linearization, with a radius controlled by actual/predicted cost
    decrease. Parameter bounds are imposed in the linear subproblem.
    """
    from matrix_numerics.trust_region import bounded_svd_trust_region_lm_step
    x = np.asarray(values,dtype=float).copy()
    scales = np.asarray(scales,dtype=float)
    lo,hi = np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if (max_nfev < 1 or not np.isfinite(initial_radius) or initial_radius <= 0
            or not np.isfinite(initial_damping) or initial_damping < 0
            or scales.shape != x.shape or np.any(scales <= 0) or not np.all(np.isfinite(scales))):
        raise ValueError('invalid trust-region budget, radius or parameter scales')
    r = np.asarray(fun(x),dtype=float); evaluations,jacobians=1,0
    radius,shift = float(initial_radius),float(initial_damping)
    def save_state():
        if state_callback is not None:
            state_callback(dict(values=x.tolist(), scales=scales.tolist(),
                                trust_radius=radius, damping=shift))
    save_state()
    history = [] if trace is None else trace
    status,message=0,'The maximum number of function evaluations is exceeded.'
    while True:
        j = np.asarray(jac(x),dtype=float);jacobians+=1
        a = j*scales
        if evaluations >= max_nfev:break
        accepted = False
        while evaluations < max_nfev:
            proposed=bounded_svd_trust_region_lm_step(a,-r,shift,radius,(lo-x)/scales,(hi-x)/scales)
            step=proposed.step;length=float(np.linalg.norm(step))
            candidate=np.clip(x+scales*step,lo,hi)
            # Use the actual representable, feasible increment in the model.
            actual_step=candidate-x
            before=.5*float(r@r)
            predicted=-float(r@(j@actual_step))-.5*float((j@actual_step)@(j@actual_step))
            if predicted <= 0. or np.array_equal(candidate,x):
                status,message=-3,'Trust region stalled before statistical stationarity.'
                break
            trial=np.asarray(fun(candidate),dtype=float);evaluations+=1
            after=.5*float(trial@trial);actual=before-after;ratio=actual/predicted
            accepted=bool(actual > 0 and ratio > 1.e-4)
            previous_radius=radius
            if ratio < .25:radius=.25*min(radius,length)
            elif ratio > .75 and length >= .8*radius:radius*=2.
            shift=proposed.shift
            record=dict(evaluation=evaluations,accepted=accepted,objective_before=before,
                        objective_trial=after,predicted_reduction=predicted,actual_reduction=actual,
                        ratio=ratio,step_norm=length,trust_radius=previous_radius,
                        next_trust_radius=radius,damping=shift,linear_solver=proposed.solver)
            history.append(record)
            if trace_callback is not None:trace_callback(history)
            if accepted:
                x,r=candidate,trial
            save_state()
            if accepted:
                break
            if radius <= np.finfo(float).eps*max(1.,float(np.linalg.norm(x/scales))):
                status,message=-3,'Trust radius exhausted before statistical stationarity.'
                break
        if not accepted:break
    return OptimizeResult(x=x,success=False,status=status,message=message,nfev=evaluations,
                          njev=jacobians,cost=.5*float(r@r),
                          optimality=float(norm(j.T@r,ord=np.inf)))
