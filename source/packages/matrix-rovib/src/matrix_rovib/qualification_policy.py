"""Observable accuracy policies shared by exploratory and precision LOCK work."""
from __future__ import annotations

import numpy as np


def observable_qualification_policy(scales_MHz,*,purpose,numerical_fraction=.2,
        relative_accuracy=None,absolute_accuracy_MHz=None,experimental_sigma_MHz=None):
    """Build a policy consumed directly by ``qualify_lam_accuracy``.

    Exploration targets max(absolute_accuracy, relative_accuracy*abs(scale));
    refinement reserves a declared fraction of each measurement's uncertainty.
    Scales describe requested observables, never an absolute electronic ZPE.
    This declares a numerical convergence target, not attained QM accuracy.
    """
    if not np.isrealobj(scales_MHz):raise ValueError('real observable scales required')
    scales=np.asarray(scales_MHz,float)
    if scales.ndim!=1 or not len(scales) or not np.all(np.isfinite(scales)):
        raise ValueError('a finite vector of observable scales is required')
    if not np.isfinite(numerical_fraction) or not 0<numerical_fraction<1:
        raise ValueError('numerical_fraction must lie in (0,1)')
    if purpose=='explore':
        if experimental_sigma_MHz is not None:
            raise ValueError('exploration uses a declared accuracy target, not measurement uncertainties')
        if relative_accuracy is None and absolute_accuracy_MHz is None:
            raise ValueError('declare an exploratory relative or absolute accuracy')
        relative=0. if relative_accuracy is None else float(relative_accuracy)
        if not np.isfinite(relative) or relative<0:raise ValueError('nonnegative finite relative accuracy required')
        absolute=np.broadcast_to(np.asarray(0. if absolute_accuracy_MHz is None else absolute_accuracy_MHz,float),scales.shape)
        if not np.all(np.isfinite(absolute)) or np.any(absolute<0):
            raise ValueError('nonnegative finite absolute accuracy required')
        target=np.maximum(absolute,relative*abs(scales))
        if np.any(target<=0):raise ValueError('zero-scale observables require a positive absolute accuracy')
        return dict(purpose=purpose,observable_budget_MHz=(target*numerical_fraction).tolist())
    if purpose=='refine':
        if relative_accuracy is not None or absolute_accuracy_MHz is not None or experimental_sigma_MHz is None:
            raise ValueError('refinement requires measurement uncertainties instead of exploratory targets')
        sigma=np.broadcast_to(np.asarray(experimental_sigma_MHz,float),scales.shape)
        if not np.all(np.isfinite(sigma)) or np.any(sigma<=0):raise ValueError('positive finite measurement uncertainties required')
        return dict(purpose=purpose,fraction_of_experimental_sigma=float(numerical_fraction),
            frequency_tolerance_MHz=float(np.min(sigma)*numerical_fraction))
    raise ValueError('purpose must be explore or refine')


__all__=['observable_qualification_policy']
