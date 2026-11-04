"""Identity-based marginal covariance selection for changing measurement sets."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def remap_assignment_covariance(covariance, source_labels, target_labels):
    """Select/reorder a marginal; never condition on discarded measurements."""
    from matrix_numerics.least_squares import CovarianceWhitening
    source=[str(s) for s in source_labels]; target=[str(s) for s in target_labels]
    matrix=np.asarray(covariance,float)
    if (len(set(source))!=len(source) or len(set(target))!=len(target)
            or any(not s for s in source+target) or matrix.shape!=(len(source),len(source))):
        raise ValueError('covariance needs unique measurement IDs and a matching square matrix')
    CovarianceWhitening(np.sqrt(np.maximum(np.diag(matrix),0.)),matrix)
    lookup={name:i for i,name in enumerate(source)}
    unknown=set(target)-lookup.keys()
    if unknown:
        raise ValueError(f'new measurement covariance is not declared for IDs: {sorted(unknown)}')
    indices=[lookup[name] for name in target]
    selected=matrix[np.ix_(indices,indices)].copy()
    return selected,dict(source_labels=source,target_labels=target,source_indices=indices,
        excluded_labels=[s for s in source if s not in set(target)],
        convention='marginal submatrix; all retained cross-correlations preserved; no conditioning or rescaling')


def assignment_fit_covariance(fit_request, assignment_report, observations):
    """Infer covariance row IDs from explicit metadata, source rows, or peaks.

    Measurement covariances are indexed by observed peaks, not unresolved
    components. Local QM covariance blocks are preserved unchanged.
    """
    request=deepcopy(fit_request)
    if request.get('covariance_MHz2') is None:
        return request,dict(convention='original independent measurement uncertainties')
    labels=request.get('covariance_labels')
    if labels is None:
        source=request.get('observations') or [r['peak'] for r in assignment_report['rows']]
        labels=[r.get('label',r.get('id')) for r in source]
        if any(label is None for label in labels):
            raise ValueError('source measurement labels are required for covariance remapping')
    matrix,audit=remap_assignment_covariance(request['covariance_MHz2'],labels,
                                            [r['label'] for r in observations])
    request['covariance_MHz2']=matrix.tolist()
    request['covariance_labels']=[r['label'] for r in observations]
    return request,audit


__all__=['remap_assignment_covariance','assignment_fit_covariance']
