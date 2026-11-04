"""Reference-anchored periodic Hermite surfaces with a positive quartic form."""
from dataclasses import dataclass
from itertools import product
from math import comb, factorial
import numpy as np
from .periodic_hermite import PeriodicHermiteSurface, angular_features
from .damped_periodic import periodic_envelope
from .periodic_fields import PeriodicVectorField


def _gram_features(points, powers, scales, order):
    order = tuple(order); n = len(powers); tri = np.tril_indices(n)
    result = np.zeros((len(points), len(tri[0])))
    for left in product(*[range(k+1) for k in order]):
        right = tuple(k-j for k, j in zip(order, left))
        weight = np.prod([comb(k, j) for k, j in zip(order, left)])
        a = angular_features(points, powers, scales, left)
        b = angular_features(points, powers, scales, right)
        result += weight*a[:, tri[0]]*b[:, tri[1]]
    result *= np.where(tri[0] == tri[1], 1., 2.)
    return result


@dataclass(frozen=True)
class PositivePeriodicSurface:
    baseline: PeriodicHermiteSurface
    powers: np.ndarray
    scales: np.ndarray
    gram: np.ndarray
    positive_softening_rates: np.ndarray | None = None
    quartic_anchor: PeriodicVectorField | None = None

    def __post_init__(self):
        for name in ('powers', 'scales', 'gram'):
            value = np.array(getattr(self, name), dtype=int if name == 'powers' else float, copy=True)
            if not np.all(np.isfinite(value)): raise ValueError('Finite model required')
            value.flags.writeable = False; object.__setattr__(self, name, value)
        d = len(self.scales); n = len(self.powers)
        if self.positive_softening_rates is not None:
            rates=np.array(self.positive_softening_rates,float,copy=True)
            if rates.shape!=(d,) or np.any(rates<0) or not np.all(np.isfinite(rates)):
                raise ValueError('Finite nonnegative softening rates required')
            rates.flags.writeable=False;object.__setattr__(self,'positive_softening_rates',rates)
        degree = self.powers.sum(axis=1)
        if self.quartic_anchor is not None and (np.any(degree<3) or self.quartic_anchor.coefficients.shape[1]!=1):
            raise ValueError('A hard quartic anchor requires a residual beginning at degree six')
        if (self.powers.ndim != 2 or self.powers.shape[1] != d or np.any(self.powers < 0)
                or np.any((degree < 2) | (degree > 4)) or np.any(self.scales <= 0)
                or self.gram.shape != (n, n) or not np.allclose(self.gram, self.gram.T, atol=1e-10)
                or np.any(abs(self.gram[(degree[:, None]-degree[None, :]) % 2 != 0]) > 1e-9)
                or np.linalg.eigvalsh(self.gram).min() < -1e-9):
            raise ValueError('Quadratic periodic features and a positive semidefinite Gram matrix required')

    def softened_baseline(self,points,order,*,allow_extrapolation=False):
        result=self.baseline.evaluate(points,order,allow_extrapolation=allow_extrapolation)
        if self.positive_softening_rates is not None:
            rates=np.asarray(self.positive_softening_rates,float)
            eig,U=np.linalg.eigh(self.baseline.reference_hessian);positive=(U*np.maximum(eig,0.))@U.T
            d=len(rates);sheet=PeriodicHermiteSurface(positive,np.empty((0,d),int),np.empty(0),np.ones(d))
            lengths=np.full(d,np.inf);mask=rates>0;lengths[mask]=1/np.sqrt(rates[mask])
            for left in product(*[range(k+1) for k in order]):
                right=tuple(k-j for k,j in zip(order,left));factor=np.prod([comb(k,j) for k,j in zip(order,left)])
                envelope=periodic_envelope(points,lengths,1,left)
                if not sum(left):envelope=envelope-1.
                result+=factor*envelope*sheet.baseline(points,right)
        if self.quartic_anchor is not None:result+=self.quartic_anchor.evaluate(points,order)[:,0]
        return result

    def evaluate(self, points, derivative=None, *, allow_extrapolation=False):
        q = np.asarray(points, float)
        if q.ndim != 2 or q.shape[1] != len(self.scales): raise ValueError('Coordinate matrix required')
        order = (0,)*q.shape[1] if derivative is None else tuple(derivative)
        tri = np.tril_indices(len(self.powers))
        return self.softened_baseline(q, order, allow_extrapolation=allow_extrapolation) + _gram_features(q, self.powers, self.scales, order)@self.gram[tri]

    def to_dict(self):
        return dict(schema='matrix.dvr.positive_periodic_surface.v1', baseline=self.baseline.to_dict(),
                    powers=self.powers.tolist(), scales=self.scales.tolist(), gram=self.gram.tolist(),
                    reference_hessian_exact=True, positive_quartic=self.quartic_anchor is None,
                    quartic_anchor=None if self.quartic_anchor is None else self.quartic_anchor.to_dict(),
                    reference_fourth_exact=self.quartic_anchor is not None,positive_residual=True,
                    positive_softening_rates=None if self.positive_softening_rates is None else np.asarray(self.positive_softening_rates).tolist(),
                    qualification='Algebraic positivity is not outside-source electronic validation')

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.positive_periodic_surface.v1': raise ValueError('Unsupported model schema')
        return cls(PeriodicHermiteSurface.from_dict(data['baseline']), np.array(data['powers']), np.array(data['scales']), np.array(data['gram']),None if data.get('positive_softening_rates') is None else np.array(data['positive_softening_rates']),None if data.get('quartic_anchor') is None else PeriodicVectorField.from_dict(data['quartic_anchor']))


def fit_positive_periodic_surface(coordinates, observations, reference_hessian, *, scales,
                                  orders, derivative_weights, relative_ridge=1e-8, max_iterations=50000,
                                  include_self_sextic=True, include_self_octic=False,positive_softening_rates=None,reference_fourth=None,anchor_tail_lengths=None):
    """Fit a PSD quadratic-feature Gram matrix with exact reference E/G/H.

    The periodic baseline splits the reference Hessian into positive and
    negative parts. Its negative part has an explicit finite lower bound;
    the fitted quartic form cannot introduce additional negative wells.
    """
    q = np.asarray(coordinates, float); y = np.asarray(observations, float)
    H = np.asarray(reference_hessian, float); scale = np.asarray(scales, float); d = len(scale)
    orders = tuple(tuple(o) for o in orders); weights = np.asarray(derivative_weights, float)
    if (q.ndim != 2 or q.shape[1] != d or y.shape != (len(orders), len(q))
            or weights.shape != (len(orders),) or np.any(weights <= 0) or not 0 < relative_ridge < 1
            or H.shape != (d, d) or not np.allclose(H, H.T)):
        raise ValueError('Compatible finite Hermite observations and symmetric Hessian required')
    eigen, U = np.linalg.eigh(H); Hplus = (U*np.maximum(eigen, 0.))@U.T
    baseline = PeriodicHermiteSurface(H, 4*np.eye(d, dtype=int), np.diag(Hplus-H)/8., np.ones(d), np.max(abs(q), axis=0))
    anchor=None
    if reference_fourth is not None:
        fourth=np.asarray(reference_fourth,float);lengths=np.asarray(anchor_tail_lengths,float)
        if fourth.shape!=(d,)*4 or not np.all(np.isfinite(fourth)) or lengths.shape!=(d,) or np.any(lengths<=0) or not np.all(np.isfinite(lengths)) or positive_softening_rates is not None:
            raise ValueError('Complete finite fourth tensor and positive anchor tail lengths required')
        quartics=np.array([p for p in product(range(5),repeat=d) if sum(p)==4],int);coefficients=[]
        for power in quartics:
            axes=tuple(np.repeat(np.arange(d),power));leading=np.prod([factorial(int(i)) for i in power])
            coefficients.append((fourth[axes]-baseline.evaluate(np.zeros((1,d)),power,allow_extrapolation=True)[0])/leading)
        anchor=PeriodicVectorField(quartics,np.array(coefficients)[:,None],np.ones(d),lengths,2,False)
        powers=np.array([p for p in product(range(4),repeat=d) if sum(p)==3],int)
        if include_self_octic:powers=np.vstack([powers,4*np.eye(d,dtype=int)])
    else:
        powers = np.array([p for p in product(range(3), repeat=d) if sum(p) == 2], int)
        if include_self_sextic: powers = np.vstack([powers, 3*np.eye(d, dtype=int)])
        if include_self_octic: powers = np.vstack([powers, 4*np.eye(d, dtype=int)])
    n = len(powers); full_tri = np.tril_indices(n)
    parity = powers.sum(axis=1) % 2
    keep = parity[full_tri[0]] == parity[full_tri[1]]
    tri = (full_tri[0][keep], full_tri[1][keep])
    A = np.concatenate([_gram_features(q, powers, scale, o)[:, keep]*w for o, w in zip(orders, weights)])
    template=PositivePeriodicSurface(baseline,powers,scale,np.zeros((n,n)),positive_softening_rates,anchor)
    rhs = np.concatenate([(y[k]-template.softened_baseline(q, o, allow_extrapolation=True))*weights[k] for k, o in enumerate(orders)])
    norm = max(float(np.linalg.norm(A, 2)), 1e-30); A /= norm; rhs /= norm
    # Orthonormal symmetric coordinates make Euclidean PSD projection the
    # Frobenius projection. This is a convex quadratic problem on PSD blocks,
    # rather than a nonlinear factor optimization with rotational null spaces.
    packing = np.where(tri[0] == tri[1], 1., np.sqrt(2.))
    A /= packing[None, :]
    normal = A.T@A+relative_ridge*np.eye(A.shape[1]); target = A.T@rhs
    def matrix(v):
        Q = np.zeros((n, n)); Q[tri] = v/packing; Q[(tri[1], tri[0])] = v/packing
        return Q
    def project(v):
        Q = matrix(v); result = np.zeros_like(Q)
        for sign in (0, 1):
            ids = np.flatnonzero(parity == sign)
            if len(ids):
                ev, vec = np.linalg.eigh(Q[np.ix_(ids, ids)])
                result[np.ix_(ids, ids)] = (vec*np.maximum(ev, 0.))@vec.T
        return result[tri]*packing
    lipschitz = float(np.linalg.eigvalsh(normal).max())
    current = project(np.linalg.solve(normal, target)); extrapolated = current.copy(); momentum = 1.
    value = .5*current@normal@current-target@current; success = False
    for iteration in range(max_iterations):
        proposed = project(extrapolated-(normal@extrapolated-target)/lipschitz)
        new_value = .5*proposed@normal@proposed-target@proposed
        if new_value > value+1e-14:
            extrapolated = current.copy(); momentum = 1.
            proposed = project(current-(normal@current-target)/lipschitz)
            new_value = .5*proposed@normal@proposed-target@proposed
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        extrapolated = proposed+(momentum-1)/next_momentum*(proposed-current)
        current = proposed; momentum = next_momentum; value = new_value
        if iteration % 25 == 0:
            gradient_mapping = lipschitz*(current-project(current-(normal@current-target)/lipschitz))
            if np.max(abs(gradient_mapping)) < 1e-9*max(1., float(np.max(abs(target)))):
                success = True; break
    newton_iterations=0
    if not success:
        # Refine the convex projection KKT equation. Its generalized
        # derivative follows the divided differences of max(eigenvalue,0).
        # This resolves nearly active PSD boundaries without relaxing the
        # reported optimality tolerance or adding an eigenvalue floor.
        K=np.eye(len(current))-normal/lipschitz
        for newton_iterations in range(1,201):
            trial=current-(normal@current-target)/lipschitz
            residual=current-project(trial)
            gradient_mapping=lipschitz*residual
            if np.max(abs(gradient_mapping))<1e-9*max(1.,float(np.max(abs(target)))):
                success=True;break
            W=matrix(trial);blocks=[]
            for sign in (0,1):
                ids=np.flatnonzero(parity==sign)
                if not len(ids):continue
                ev,U=np.linalg.eigh(W[np.ix_(ids,ids)]);positive=np.maximum(ev,0.)
                denominator=ev[:,None]-ev[None,:];numerator=positive[:,None]-positive[None,:]
                tolerance=1e-12*max(1.,float(abs(ev).max()))
                omega=np.divide(numerator,denominator,out=np.zeros_like(denominator),where=abs(denominator)>tolerance)
                repeated=abs(denominator)<=tolerance
                mean=(ev[:,None]+ev[None,:])/2
                omega[repeated]=np.where(mean[repeated]>tolerance,1.,np.where(mean[repeated]<-tolerance,0.,.5))
                blocks.append((ids,U,omega))
            def derivative(direction):
                E=matrix(direction);result=np.zeros_like(E)
                for ids,U,omega in blocks:
                    result[np.ix_(ids,ids)]=U@(omega*(U.T@E[np.ix_(ids,ids)]@U))@U.T
                return result[tri]*packing
            jacobian=np.eye(len(current))-np.column_stack([derivative(K[:,j]) for j in range(len(current))])
            direction=np.linalg.lstsq(jacobian,-residual,rcond=1e-12)[0];old_norm=np.linalg.norm(residual)
            accepted=False
            for backtrack in range(25):
                step=2.**(-backtrack);candidate=project(current+step*direction)
                candidate_residual=candidate-project(candidate-(normal@candidate-target)/lipschitz)
                if np.linalg.norm(candidate_residual)<old_norm*(1-1e-4*step):
                    current=candidate;accepted=True;break
            if not accepted:current=project(current-(normal@current-target)/lipschitz)
        gradient_mapping=lipschitz*(current-project(current-(normal@current-target)/lipschitz))
        success=bool(np.max(abs(gradient_mapping))<1e-9*max(1.,float(np.max(abs(target)))))
    Q = matrix(current)
    model = PositivePeriodicSurface(baseline, powers, scale, Q,positive_softening_rates,anchor)
    report = dict(optimizer_success=success, iterations=iteration+1,
                  optimizer='Convex PSD projected gradient with monotone acceleration and semismooth Newton refinement',
                  newton_iterations=newton_iterations,
                  projected_gradient_residual=float(np.max(abs(gradient_mapping))),
                  minimum_gram_eigenvalue=float(np.linalg.eigvalsh(Q).min()),
                  global_energy_lower_bound_cm=float(.5*d*min(float(eigen.min()), 0.)-(0. if anchor is None else 16*np.sum(abs(anchor.coefficients)))),
                  reference_fourth_exact=anchor is not None,
                  relative_ridge=relative_ridge, include_self_sextic=include_self_sextic,
                  include_self_octic=include_self_octic,
                  reflection_parity_blocks_exact=True, reference_E_G_H_exact=True)
    return model, report
