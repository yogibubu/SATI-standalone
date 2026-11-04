"""Experimental identifiability and explicit conditional parameter selection."""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def select_spectral_parameters(experimental_jacobian, parameter_names, scales, *,
                               augmented_jacobian=None, rcond=1.e-10,
                               minimum_singular_value=0.0):
    """Select independent columns and report the physical observable contrasts.

    Jacobians must already be covariance-whitened; validation rows must not
    occur in either matrix. Scales are physical parameter increments, not
    automatically inferred units. A selected column set is a *conditional*
    parameterization: fixing the other columns does not make the individual
    selected constants experimentally identifiable.
    """
    names = list(map(str, parameter_names))
    scale = np.asarray(scales, float)
    data = np.asarray(experimental_jacobian, float)
    aug = data if augmented_jacobian is None else np.asarray(augmented_jacobian, float)
    if (not names or len(set(names)) != len(names) or scale.shape != (len(names),)
        or np.any(scale <= 0) or not np.all(np.isfinite(scale))
        or not np.isfinite(rcond) or not 0 < rcond < 1
        or not np.isfinite(minimum_singular_value) or minimum_singular_value < 0
        or any(m.ndim != 2 or m.shape[1] != len(names) or not np.all(np.isfinite(m)) for m in (data, aug))):
        raise ValueError("invalid scaled, whitened parameter-selection problem")

    def analyze(matrix):
        a = matrix * scale
        # full_matrices only when needed to complete a short right-null basis.
        _, singular, vh = np.linalg.svd(a, full_matrices=a.shape[0] < a.shape[1])
        cutoff = max(minimum_singular_value, rcond * float(singular[0]) if len(singular) else 0.0)
        rank = int(np.count_nonzero(singular > cutoff))
        observed, null = vh[:rank], vh[rank:]
        _, _, pivots = qr(a, mode="economic", pivoting=True)
        selected = list(map(int, pivots[:rank]))
        # QR chooses a representative set; enforce the same actual rank
        # criterion on that set rather than trusting its diagonal alone.
        while selected and np.linalg.svd(a[:, selected], compute_uv=False)[-1] <= cutoff:
            selected.pop()
        return dict(rank=rank, singular_values=singular.tolist(), cutoff=cutoff,
            observable_scaled_directions=observed.tolist(),
            null_scaled_directions=null.tolist(),
            observable_projector=(observed.T @ observed).tolist(),
            individual_identifiability=np.sum(observed**2, axis=0).tolist(),
            selected_indices=selected,
            conditional_selected_parameters=[names[k] for k in selected],
            conditional_fixed_parameters=[n for k, n in enumerate(names) if k not in selected])

    experiment, augmented = analyze(data), analyze(aug)
    return dict(schema="matrix.lock.parameter_selection.v1", parameter_names=names,
        parameter_scales=scale.tolist(), experimental=experiment, augmented=augmented,
        rank_gain_from_predicates=augmented["rank"] - experiment["rank"],
        information_sources=[dict(name=n, experimental_score=experiment["individual_identifiability"][k],
            augmented_score=augmented["individual_identifiability"][k],
            status="experiment-determined" if experiment["individual_identifiability"][k] >= 1-1.e-8
            else "predicate-supported" if augmented["individual_identifiability"][k] >= 1-1.e-8
            else "unresolved-combination") for k,n in enumerate(names)],
        scope="local identifiability; representative columns require fixed complementary values; nonlinear refit and independent prediction remain necessary")
