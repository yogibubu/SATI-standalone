"""Declared error budgets and finite-look decisions for vibronic validation."""
import numpy as np
from scipy.stats import t
import hashlib
import json
from pathlib import Path


COMPONENTS=('basis','quadrature','spectral_filter','sampling','spectator_domain','mode_hierarchy')


def validate_convergence_protocol(protocol):
    budgets=protocol['budgets_s_1']
    if (set(budgets)!=set(COMPONENTS) or any(not np.isfinite(v) or v<=0 for v in budgets.values())
            or not np.isclose(sum(budgets.values()),protocol['total_absolute_target_s_1'],rtol=0,atol=1e-9)
            or not 0<protocol['family_alpha']<1):
        raise ValueError('Positive partition of the declared rate-error target required')
    for key in ('maximum_comparisons','maximum_looks','replicas','successive_passing_enlargements'):
        value=protocol[key]
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Positive integer protocol controls required')
    sizes=protocol['samples_per_replica']
    if (protocol['replicas']<2 or len(sizes)!=protocol['maximum_looks']
            or any(isinstance(v,bool) or not isinstance(v,int) or v<2 for v in sizes)
            or any(a>=b for a,b in zip(sizes,sizes[1:]))):
        raise ValueError('Increasing finite production-look schedule required')
    if (not np.isfinite(protocol['rate_scale_s_1']) or protocol['rate_scale_s_1']<=0
            or protocol['minimum_absolute_change_ess']<=0
            or not 0<protocol['maximum_absolute_change_fraction']<=1):
        raise ValueError('Positive reporting scale and valid importance diagnostics required')
    return protocol


def reserve_convergence_comparison(path,identifier,protocol):
    """Count independent comparisons before sampling; restarts keep their slot."""
    import fcntl
    validate_convergence_protocol(protocol)
    if not isinstance(identifier,str) or not identifier:raise ValueError('Comparison identity required')
    path=Path(path);digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text()) if path.exists() else dict(protocol_sha256=digest,comparisons=[])
        if data['protocol_sha256']!=digest:raise ValueError('Validation protocol changed after registration')
        if identifier not in data['comparisons']:
            if len(data['comparisons'])>=protocol['maximum_comparisons']:
                raise RuntimeError('Declared simultaneous-comparison budget exhausted')
            data['comparisons'].append(identifier)
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)
        return data['comparisons'].index(identifier)+1


def assess_paired_convergence(mean_s_1,standard_error_s_1,*,protocol,component,
        replicas,independent_validation,absolute_change_ess,maximum_change_fraction):
    """An estimated simultaneous interval, never a distribution-free certificate."""
    validate_convergence_protocol(protocol)
    if (component not in COMPONENTS or not np.isfinite(mean_s_1)
            or not np.isfinite(standard_error_s_1) or standard_error_s_1<0
            or replicas<2):raise ValueError('Finite paired estimate and replica count required')
    alpha=protocol['family_alpha']/(protocol['maximum_comparisons']*protocol['maximum_looks'])
    factor=float(t.ppf(1-alpha/2,replicas-1))
    halfwidth=factor*standard_error_s_1;upper=abs(mean_s_1)+halfwidth
    lower=max(0.,abs(mean_s_1)-halfwidth);budget=protocol['budgets_s_1'][component]
    reliable=(replicas>=protocol['replicas'] and np.isfinite(absolute_change_ess)
        and absolute_change_ess>=protocol['minimum_absolute_change_ess']
        and np.isfinite(maximum_change_fraction)
        and 0<=maximum_change_fraction<=protocol['maximum_absolute_change_fraction'])
    decision=('pilot_only' if not independent_validation else 'sampling_unresolved' if not reliable
              else 'enlarge_model' if lower>budget else 'local_pass' if upper<=budget
              else 'sampling_unresolved')
    return dict(decision=decision,absolute_change_upper_s_1=upper,
        absolute_change_lower_s_1=lower,estimated_interval_halfwidth_s_1=halfwidth,
        critical_factor=factor,component_budget_s_1=budget,
        interval_is_distribution_free=False,local_pass_is_global_convergence=False)


def aggregate_convergence_status(protocol,evidence):
    """Require aggregate components: marginal mode tests cannot close the budget."""
    validate_convergence_protocol(protocol);missing=[];failed=[];certified=True;total=0.
    for component in COMPONENTS:
        item=evidence.get(component)
        if (not item or not item.get('aggregate') or not item.get('independent_validation')
                or (item.get('bound_kind')!='analytic' and
                    item.get('successive_passes',0)<protocol['successive_passing_enlargements'])):
            missing.append(component);certified=False;continue
        bound=item.get('error_upper_s_1',float('inf'))
        if not np.isfinite(bound) or bound<0 or bound>protocol['budgets_s_1'][component]:
            failed.append(component);certified=False
        else:total+=bound
        certified=certified and item.get('bound_kind') in ('analytic','finite_sample_validated')
    observed=not missing and not failed
    return dict(numerical_convergence_observed=observed,
        global_error_certified=bool(observed and certified),missing_components=missing,
        failed_components=failed,aggregate_error_indicator_s_1=total if observed else None)


__all__=['validate_convergence_protocol','assess_paired_convergence','aggregate_convergence_status']
