"""Effective rotational exchange terms, separate from a structural centroid.

These operators may represent measured component differences. They neither
modify an electronic field nor turn a calibrated splitting into a prediction.
The two component labels carry no implicit permutation/spin assignment.
"""
import numpy as np
from .native_rotor import angular_momentum_operators


def exchange_component_tensors(centroid_MHz, difference_MHz):
    """Return A/B tensors with the specified arithmetic mean and A-minus-B."""
    mean=np.asarray(centroid_MHz,float);delta=np.asarray(difference_MHz,float)
    if (mean.shape!=(3,3) or delta.shape!=(3,3)
            or not all(np.all(np.isfinite(x)) and np.allclose(x,x.T,atol=1e-12) for x in (mean,delta))):
        raise ValueError('Finite symmetric centroid and exchange tensors required')
    components=np.array([mean+delta/2,mean-delta/2])
    if min(np.linalg.eigvalsh(x).min() for x in components)<=0:
        raise ValueError('Both effective rotational tensors must remain positive')
    return components


def lowest_branch_exchange_progression(centroid_MHz,difference_MHz,*,max_J=5):
    """Diagonalize both full asymmetric-rotor operators, then difference lines.

    The lowest branch is the near-prolate Ka=0 progression. Band origins
    cancel within each transition and are deliberately not inferred here.
    Hyperfine structure and centrifugal distortion are outside this operator.
    """
    tensors=exchange_component_tensors(centroid_MHz,difference_MHz)
    if int(max_J)!=max_J or max_J<1:raise ValueError('Positive integer max_J required')
    energies=np.zeros((2,max_J+1))
    for J in range(1,max_J+1):
        axes=angular_momentum_operators(J,'Ir')
        for s,tensor in enumerate(tensors):
            h=sum(tensor[a,b]*(axes[a]@axes[b]+axes[b]@axes[a])/2 for a in range(3) for b in range(3))
            energies[s,J]=np.linalg.eigvalsh(h)[0]
    lines=np.diff(energies,axis=1)
    return [dict(J_upper=J,component_A_MHz=float(lines[0,J-1]),component_B_MHz=float(lines[1,J-1]),
                 A_minus_B_MHz=float(lines[0,J-1]-lines[1,J-1])) for J in range(1,max_J+1)]


def fit_effective_exchange_progression(centroid_MHz,J_upper,delta_frequency_MHz,*,training_mask=None,initial_BC_MHz=(1.,.5)):
    """Calibrate two exchange constants on declared transition differences.

    No geometry, average rotational constant, band origin or A difference is
    fitted. Reserved transitions are evaluated only after the calibration.
    Returned residuals are not a physical parameter covariance.
    """
    from scipy.optimize import least_squares
    js=np.asarray(J_upper,int);target=np.asarray(delta_frequency_MHz,float)
    train=np.ones(len(js),bool) if training_mask is None else np.asarray(training_mask,bool)
    if (js.ndim!=1 or target.shape!=js.shape or train.shape!=js.shape or train.sum()<2
            or len(np.unique(js[train]))<2 or np.any(js<1) or not np.all(np.isfinite(target))):
        raise ValueError('At least two distinct calibration J values and finite transition targets required')
    def evaluate(bc):
        rows=lowest_branch_exchange_progression(centroid_MHz,np.diag([0.,*bc]),max_J=int(max(js)))
        return np.array([rows[j-1]['A_minus_B_MHz'] for j in js])
    fit=least_squares(lambda bc:(evaluate(bc)-target)[train],initial_BC_MHz,
                      diff_step=1e-3,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    if not fit.success:raise RuntimeError('Effective exchange calibration did not converge')
    prediction=evaluate(fit.x);error=prediction-target
    return dict(B_difference_MHz=float(fit.x[0]),C_difference_MHz=float(fit.x[1]),
                predicted_splitting_MHz=prediction.tolist(),residual_MHz=error.tolist(),
                training_mask=train.tolist(),training_RMS_MHz=float(np.sqrt(np.mean(error[train]**2))),
                reserved_max_error_MHz=float(max(abs(error[~train]))) if np.any(~train) else None,
                independent_ab_initio_prediction=False)


__all__=['exchange_component_tensors','lowest_branch_exchange_progression','fit_effective_exchange_progression']
