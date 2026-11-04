"""Periodic energy/force least squares using Architect's Fourier surface.

Wavevectors and derivative weights are explicit inputs; no validation point
participates in fitting and no electronic acquisition is performed.
"""
import numpy as np
from scipy.linalg import null_space
from .multiminimum import MultiMinimumFourierSurface, _fourier_constraint_row


def fit_fourier_energy_gradient(phases_radian, energies_hartree, gradients_hartree,
                                wavevectors, *, derivative_weight_radian=1., anchor_index=None):
    """Fit finite E/G observations without constructing higher derivatives.

    Multiplying gradient rows by the supplied radian weight makes the residual
    objective explicit. An optional observation index retains its E/G exactly.
    Full column rank is required. The returned surface
    retains Architect's existing periodic E/G/H evaluator and serialization.
    """
    q = np.asarray(phases_radian, float)
    e = np.asarray(energies_hartree, float)
    g = np.asarray(gradients_hartree, float)
    waves = tuple(tuple(v for v in row) for row in wavevectors)
    if (q.ndim != 2 or not q.size or e.shape != (len(q),) or g.shape != q.shape
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(e))
            or not np.all(np.isfinite(g)) or not np.isfinite(derivative_weight_radian)
            or derivative_weight_radian <= 0):
        raise ValueError('Finite compatible E/G arrays and positive derivative weight required')
    if any(any(isinstance(v, bool) or int(v) != v for v in wave) for wave in waves):
        raise ValueError('Integer periodic wavevectors required')
    template = MultiMinimumFourierSurface(waves, 0., (0.,)*len(waves), (0.,)*len(waves))
    if template.dimension != q.shape[1]:
        raise ValueError('Wavevector dimension differs from coordinate dimension')
    design, targets = [], []
    for x, energy, gradient in zip(q, e, g):
        design.append(_fourier_constraint_row(x, waves, ())); targets.append(energy)
        for axis in range(q.shape[1]):
            design.append(derivative_weight_radian*_fourier_constraint_row(x, waves, (axis,)))
            targets.append(derivative_weight_radian*gradient[axis])
    design = np.asarray(design); targets = np.asarray(targets)
    coefficients, residual, rank, singular = np.linalg.lstsq(design, targets, rcond=1e-12)
    if rank != design.shape[1]:
        raise ValueError(f'Rank-deficient E/G Fourier fit: {rank}/{design.shape[1]}')
    constraint_residual = 0.
    if anchor_index is not None:
        if isinstance(anchor_index, bool) or not isinstance(anchor_index, (int,np.integer)) or not 0<=anchor_index<len(q):
            raise ValueError('Anchor must identify one supplied observation')
        block=slice(anchor_index*(q.shape[1]+1),(anchor_index+1)*(q.shape[1]+1))
        constraint=design[block];observed=targets[block]
        particular=np.linalg.lstsq(constraint,observed,rcond=1e-12)[0]
        free=null_space(constraint)
        coefficients=particular+free@np.linalg.lstsq(design@free,targets-design@particular,rcond=1e-12)[0]
        constraint_residual=float(np.max(abs(constraint@coefficients-observed)))
        if constraint_residual>1e-10*max(1.,np.max(abs(observed))):
            raise ValueError('Supplied E/G anchor cannot be retained')
    n = len(waves)
    surface = MultiMinimumFourierSurface(waves, float(coefficients[0]),
        tuple(coefficients[1:n+1]), tuple(coefficients[n+1:]))
    audit = dict(samples=len(q), parameters=len(coefficients), rank=int(rank),
        condition=float(singular[0]/singular[-1]), derivative_weight_radian=float(derivative_weight_radian),
        anchor_index=None if anchor_index is None else int(anchor_index), maximum_weighted_anchor_residual_hartree=constraint_residual,
        weighted_residual_norm_hartree=float(np.linalg.norm(design@coefficients-targets)))
    return surface, audit


def fit_regularized_fourier_energy_gradient(phases_radian, energies_hartree,
        gradients_hartree, wavevectors, *, derivative_weight_radian=1.,
        regularization=1e-6, anchor_index=0, even_origin_radian=None,
        lower_bound_points=None, lower_bound_values=None):
    """Smooth periodic E/G regression with exact reference E/G constraints.

    Penalize the squared Laplacian of the Fourier field. An optional inversion
    center restricts the field to cosine terms about that center. Regularization
    permits more basis functions than observations. Optional finite interpolation
    geometries impose lower-energy inequalities without acquiring new data.
    No external controls are consulted.
    """
    q=np.asarray(phases_radian,float);e=np.asarray(energies_hartree,float)
    g=np.asarray(gradients_hartree,float);waves=tuple(tuple(v) for v in wavevectors)
    if any(any(isinstance(v,bool) or not np.isfinite(v) or int(v)!=v for v in row) for row in waves):
        raise ValueError('Integer periodic wavevectors required')
    # Reuse the existing owner's validation of geometry and wavevector data.
    template=MultiMinimumFourierSurface(waves,0.,(0.,)*len(waves),(0.,)*len(waves))
    if (q.ndim!=2 or not len(q) or q.shape[1]!=template.dimension or e.shape!=(len(q),)
            or g.shape!=q.shape or not all(np.all(np.isfinite(v)) for v in (q,e,g))
            or not np.isfinite(regularization) or regularization<=0
            or not np.isfinite(derivative_weight_radian) or derivative_weight_radian<=0
            or isinstance(anchor_index,bool) or not isinstance(anchor_index,(int,np.integer))
            or not 0<=anchor_index<len(q)):
        raise ValueError('Finite E/G, positive smoothing and a valid anchor required')
    n=len(waves);transform=np.eye(1+2*n)
    if even_origin_radian is not None:
        origin=np.asarray(even_origin_radian,float)
        if origin.shape!=(q.shape[1],) or not np.all(np.isfinite(origin)):
            raise ValueError('A finite inversion center in each coordinate required')
        phase=np.asarray(waves)@origin
        transform=np.zeros((1+2*n,1+n));transform[0,0]=1
        transform[1:n+1,1:]=np.diag(np.cos(phase))
        transform[n+1:,1:]=np.diag(np.sin(phase))
    rows=[];values=[]
    for x,y,dy in zip(q,e,g):
        rows.append(_fourier_constraint_row(x,waves,()));values.append(y)
        for j in range(q.shape[1]):
            rows.append(derivative_weight_radian*_fourier_constraint_row(x,waves,(j,)))
            values.append(derivative_weight_radian*dy[j])
    A=np.asarray(rows)@transform;y=np.asarray(values)
    block=slice(anchor_index*(q.shape[1]+1),(anchor_index+1)*(q.shape[1]+1))
    C=A[block];cy=y[block];particular=np.linalg.lstsq(C,cy,rcond=1e-12)[0]
    if np.max(abs(C@particular-cy))>1e-10*max(1.,np.max(abs(cy))):
        raise ValueError('Inversion symmetry conflicts with supplied anchor')
    free=null_space(C)
    penalty=np.r_[0.,np.sum(np.asarray(waves,dtype=float)**2,axis=1),np.sum(np.asarray(waves,dtype=float)**2,axis=1)]
    L=penalty[:,None]*transform*np.sqrt(regularization)
    B=np.vstack([A@free,L@free]);rhs=np.r_[y-A@particular,-L@particular]
    coeff=particular+free@np.linalg.lstsq(B,rhs,rcond=1e-12)[0]
    bound_audit={}
    if lower_bound_points is not None:
        from scipy.optimize import minimize,LinearConstraint
        points=np.asarray(lower_bound_points,float);bounds=np.asarray(lower_bound_values,float)
        if (points.ndim!=2 or points.shape[1]!=q.shape[1] or not len(points)
                or bounds.shape not in ((),(len(points),))
                or not np.all(np.isfinite(points)) or not np.all(np.isfinite(bounds))):
            raise ValueError('Finite lower-bound geometries and scalar/per-point values required')
        bounds=np.broadcast_to(bounds,(len(points),))
        rows=np.array([_fourier_constraint_row(x,waves,()) for x in points])@transform
        scale=max(float(np.max(abs(y))),float(np.max(abs(bounds))),1e-12)
        matrix=rows@free;lower=(bounds-rows@particular)/scale
        z=free.T@(coeff-particular)/scale
        normal=B.T@B;linear=B.T@(rhs/scale)
        constrained=False;iterations=0
        if np.min(matrix@z-lower)<-1e-10:
            if free.shape[1]==0:
                raise ValueError('Shape bounds conflict with the exact reference constraints')
            constrained=True
            result=minimize(lambda v:.5*v@normal@v-linear@v,z,
                jac=lambda v:normal@v-linear,method='SLSQP',
                constraints=[LinearConstraint(matrix,lower,np.inf)],
                options=dict(ftol=1e-11,maxiter=500))
            z=result.x;iterations=int(result.nit)
            if not result.success or np.min(matrix@z-lower)<-1e-8:
                raise ValueError('Periodic shape constraints failed: '+result.message)
            coeff=particular+scale*free@z
        bound_audit=dict(shape_constraints=len(points),constrained_optimization=constrained,
            shape_iterations=iterations,maximum_shape_violation=float(max(0.,-np.min(matrix@z-lower)*scale)))
    elif lower_bound_values is not None:
        raise ValueError('Lower bound values require explicit points')
    full=transform@coeff
    field=MultiMinimumFourierSurface(waves,float(full[0]),tuple(full[1:n+1]),tuple(full[n+1:]))
    return field,dict(**bound_audit,samples=len(q),parameters=len(coeff),regularization=float(regularization),
        derivative_weight_radian=float(derivative_weight_radian),anchor_index=int(anchor_index),
        maximum_weighted_anchor_residual_hartree=float(np.max(abs(C@coeff-cy))),
        weighted_residual_norm_hartree=float(np.linalg.norm(A@coeff-y)),
        even_origin_radian=None if even_origin_radian is None else origin.tolist())
