"""Frequency-budget selection of physical n-mode coupling blocks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping, Sequence
import numpy as np

CM_TO_MHZ=29979.2458


class NModeStabilityError(ValueError):
    """A polynomial contraction failed its confinement preflight."""


@dataclass(frozen=True)
class NModeCouplingBlock:
    """An atomic PES/kinetic/property block, including its complete symmetry orbit."""
    name: str
    active_pairs: tuple = ()
    contracted_pairs: tuple = ()
    prerequisites: tuple[str,...] = ()
    estimated_cost: float = 1.
    field_increments: Mapping | None = None

    def __post_init__(self):
        if (not self.name.strip() or not (self.active_pairs or self.contracted_pairs or self.field_increments)
            or not np.isfinite(self.estimated_cost) or self.estimated_cost <= 0):
            raise ValueError('coupling blocks require a name, physical terms and positive declared cost')


def build_nmode_lam_surface(base_surface, specification: Mapping,
                            blocks: Sequence[NModeCouplingBlock]):
    """Contract supplied TRINITY/GF pure n-mode increments and build a LOCK surface.

    Axis zero is the path; subsequent confined axes are the active coordinates
    of the existing GF VSCF adapter, sharing its active grid. The base potential
    must include the harmonic transverse ZPE, so the VSCF minus harmonic-ZPE
    correction is added exactly once. Kinetic and tensor fields come from the
    supplied SMITH/ORACLE surface, optionally averaged using explicit property
    expansions in the same converged transverse VSCF states.
    """
    from matrix_gf.reduced_anharmonic import contract_nmode_vscf_to_active_surface
    if specification.get('potential_reference') != 'harmonic-transverse-zpe':
        raise ValueError('declare a base potential including harmonic transverse ZPE')
    if dict(base_surface.metadata or {}).get('nmode_contraction') is not None:
        raise ValueError('do not apply the n-mode correction twice')
    dimension=int(specification['active_dimension'])
    if base_surface.dimension != dimension+1 or any(base_surface.periodic[1:]):
        raise ValueError('VSCF surface bridge requires a path plus confined active coordinates')
    grid=np.asarray(specification['active_grid'],float)
    if any(not np.array_equal(q,grid) for q in base_surface.coordinates[1:]):
        raise ValueError('active LOCK coordinates must equal the declared VSCF grid')
    from matrix_gf.potential_stability import audit_nmode_polynomial
    stability=audit_nmode_polynomial(specification['one_mode_coefficients_cm'],
        specification['one_mode_powers'],specification['frequencies_cm'],
        [term for block in blocks for term in block.active_pairs],
        [term for block in blocks for term in block.contracted_pairs],
        active_dimension=dimension,active_grid=grid)
    if not stability['certified']:
        raise NModeStabilityError('polynomial n-mode confinement is not certified: '
            f"{stability['failures'][0]}; use a qualified finite-support sampled potential; "
            'increasing the oscillator basis or dropping blocks to match experiment is not a repair')
    controls={k:specification[k] for k in ('maximum_quanta','convergence_tolerance',
        'maximum_iterations','density_mixing') if k in specification}
    from .vscf_tensors import prepare_vscf_tensor_fields, contract_vscf_tensor_fields
    fields,references,words=prepare_vscf_tensor_fields(base_surface,specification)
    result=contract_nmode_vscf_to_active_surface(
        specification['one_mode_coefficients_cm'],specification['one_mode_powers'],
        np.asarray(specification['frequencies_cm'],float),
        [term for block in blocks for term in block.active_pairs],
        [term for block in blocks for term in block.contracted_pairs],
        active_dimension=dimension,active_grid=grid,effective_degree=1,
        _return_active_samples=True,property_powers=words,**controls)
    _,_,scalar,_,diagnostics,points,samples=result[:7]
    expected=np.stack(np.meshgrid(*base_surface.coordinates[1:],indexing='ij'),axis=-1).reshape(-1,dimension)
    if samples.shape != (base_surface.grid_shape[0],len(expected)) or not np.array_equal(points,expected):
        raise ValueError('VSCF samples do not match the LOCK path/active grid')
    updates=contract_vscf_tensor_fields(base_surface,fields,references,points,result[7]) if fields else {}
    correction=(samples+scalar[:,None]).reshape(base_surface.grid_shape)*CM_TO_MHZ
    metadata=dict(base_surface.metadata or {})
    metadata['nmode_contraction']=dict(schema='matrix.lock.nmode_surface.v1',
        selected_blocks=[b.name for b in blocks],maximum_vscf_iterations=int(np.max(diagnostics[:,1])),
        stability_preflight=stability,
        sampled_surface=True,potential_reference='harmonic-transverse-zpe',
        kinetic_tensors='complete quadratic form averaged before Schur reconstruction'
            if 'quadratic_kinetic_MHz' in fields else 'supplied SMITH/ORACLE fields retained',
        contracted_tensor_fields=sorted(fields),
        tensor_scope='coordinate-multiplication expectations in the converged product VSCF state')
    surface = replace(base_surface,potential_MHz=np.asarray(base_surface.potential_MHz)+correction,metadata=metadata,**updates)
    return apply_physical_field_blocks(surface, blocks)


def apply_physical_field_blocks(surface, blocks):
    """Add reference-frame physical increments, summing the complete (p,J) form.

    A block may contain PES, geometric scalar, quadratic kinetics, dipole and
    spin tensors together. F/rho/R are reconstructed ONCE after all increments;
    independent averaging/interpolation of these three fields is forbidden.
    Sampled fields share the existing grid, coordinate units and frame. Blocks
    must contain complete symmetry orbits from their physical producer.
    """
    blocks = tuple(blocks)
    if not any(b.field_increments for b in blocks):
        return surface
    d = surface.dimension
    f, rho, r = map(np.asarray, (surface.F_MHz, surface.rho, surface.R_MHz))
    joint = np.zeros(surface.grid_shape+(d+3, d+3))
    mixed = -f@rho.swapaxes(-1, -2)
    joint[..., :d, :d] = f
    joint[..., :d, d:] = mixed
    joint[..., d:, :d] = mixed.swapaxes(-1, -2)
    joint[..., d:, d:] = r+rho@f@rho.swapaxes(-1, -2)
    updates = {}
    spin = dict(surface.spin_tensor_surfaces_MHz or {})
    kinetic = False
    changed = []
    for block in blocks:
        for name, raw in (block.field_increments or {}).items():
            if name == 'quadratic_kinetic_MHz':
                shape = joint.shape
            elif name in {'potential_MHz', 'geometric_potential_MHz'}:
                shape = surface.grid_shape
            elif name == 'dipole_au':
                if surface.dipole_au is None:
                    raise ValueError('incremental dipoles require a declared reference field, possibly zero')
                shape = surface.grid_shape+(3,)
            elif name.startswith('spin_tensor_surfaces_MHz.'):
                key = name.split('.', 1)[1]
                if key not in spin:
                    raise ValueError('incremental spin tensors require a declared reference field')
                shape = np.shape(spin[key])
            else:
                raise ValueError(f'unsupported physical field increment: {name}')
            if not np.isrealobj(raw):
                raise ValueError('reference-frame sampled field increments must be real')
            value = np.asarray(raw, float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f'incompatible finite physical field increment: {name}')
            changed.append(name)
            if name == 'quadratic_kinetic_MHz':
                if not np.allclose(value, value.swapaxes(-1, -2), atol=1e-12, rtol=1e-12):
                    raise ValueError('quadratic kinetic increments must be symmetric')
                joint += value
                kinetic = True
            elif name.startswith('spin_tensor_surfaces_MHz.'):
                spin[key] = np.asarray(spin[key])+value
            else:
                base = updates.get(name, getattr(surface, name))
                updates[name] = value.copy() if base is None else np.asarray(base)+value
    if kinetic:
        f, mixed = joint[..., :d, :d], joint[..., :d, d:]
        if np.any(np.linalg.eigvalsh(f) <= 0):
            raise ValueError('joint kinetic form is not positive definite')
        rho = -np.linalg.solve(f, mixed).swapaxes(-1, -2)
        r = joint[..., d:, d:]-rho@f@rho.swapaxes(-1, -2)
        r = .5*(r+r.swapaxes(-1, -2))
        if np.any(np.linalg.eigvalsh(r) <= 0):
            raise ValueError('joint kinetic Schur complement is not positive definite')
        updates.update(F_MHz=f, rho=rho, R_MHz=r)
    if any(n.startswith('spin_tensor_surfaces_MHz.') for n in changed):
        updates['spin_tensor_surfaces_MHz'] = spin
    metadata = dict(surface.metadata or {})
    if changed:
        metadata['physical_field_blocks'] = dict(fields=sorted(set(changed)),
            blocks=[b.name for b in blocks if b.field_increments],
            convention='additive physical increments in a common reference frame; complete quadratic kinetics')
    return replace(surface, metadata=metadata, **updates)


@dataclass(frozen=True)
class AdaptiveCouplingResult:
    surface: object
    report: Mapping
    request: Mapping | None = None


def select_nmode_couplings(base_surface, request, specification, blocks, *,
                            frequency_budget_MHz, mandatory_names=(), maximum_evaluations=64,
                            numerical_policy=None, observable_budgets=None):
    """Greedily select atomic coupling blocks against the full supplied model.

    Selection is based only on native observable changes, declared costs and
    prerequisite closure. No experimental residual enters the decision.
    Full-reference comparison catches combinations invisible to single-term
    screening. Certification applies to the requested frequencies and supplied
    blocks, not unknown couplings, numerical basis accuracy or QM accuracy.
    """
    from .lam_qualification import _evaluate
    if request.get('variables') or request.get('predicates'):
        raise ValueError('coupling selection requires fixed physical coefficients')
    blocks=tuple(blocks);byname={b.name:b for b in blocks}
    if (not blocks or len(byname)!=len(blocks) or set(mandatory_names)-set(byname)
        or isinstance(maximum_evaluations,bool) or not isinstance(maximum_evaluations,int)
        or maximum_evaluations < 2):
        raise ValueError('invalid coupling registry or evaluation budget')
    for b in blocks:
        if set(b.prerequisites)-set(byname):raise ValueError('unknown coupling prerequisite')
    def closure(names,trail=()):
        result=set()
        for name in names:
            if name in trail:raise ValueError('cyclic coupling prerequisites')
            result.add(name);result.update(closure(byname[name].prerequisites,trail+(name,)))
        return frozenset(result)
    for b in blocks:closure((b.name,))
    budget=np.broadcast_to(np.asarray(frequency_budget_MHz,float),(len(request.get('observations',())),)).copy()
    if not len(budget) or not np.all(np.isfinite(budget)) or np.any(budget<=0):
        raise ValueError('positive finite frequency budgets required')
    from copy import deepcopy
    from .nmode_accuracy import evaluate_nmode_accuracy
    from .lam_accuracy import _carry_assignments
    policy=deepcopy(numerical_policy)
    if policy is not None:
        fraction=float(policy.get('numerical_fraction',.25))
        if not np.isfinite(fraction) or not 0<fraction<1:
            raise ValueError('numerical_fraction must lie strictly between zero and one')
        numeric_budget=budget*fraction/2
    requested_properties = dict(observable_budgets or {})
    for name, raw in requested_properties.items():
        if name != 'intensity' or not np.all(np.isfinite(raw)) or np.any(np.asarray(raw) <= 0):
            raise ValueError('positive finite native intensity budgets required')
    numeric_properties = {name:np.asarray(raw)*fraction/2 for name,raw in requested_properties.items()} if policy is not None else {}
    all_names=frozenset(byname);cache={};history=[];numerics={}
    def evaluate(names,reference=None):
        if names in cache:return cache[names]
        start=perf_counter()
        chosen=[b for b in blocks if b.name in names]
        if policy is None:
            surface=build_nmode_lam_surface(base_surface,specification,chosen)
            value=_evaluate(surface,request,reference=reference)
            numerics[names]=dict(converged=False,envelope_MHz=np.zeros(len(budget)).tolist())
        else:
            surface,value,numerics[names]=evaluate_nmode_accuracy(
                base_surface,request,specification,chosen,policy,numeric_budget,reference=reference,
                observable_budgets=numeric_properties)
        cache[names]=(surface,value,perf_counter()-start)
        return cache[names]
    full_surface,full,_=evaluate(all_names)
    property_budgets = {}
    for name, raw in (observable_budgets or {}).items():
        if name != 'intensity' or full.get(name) is None or not np.all(np.isfinite(full[name])):
            raise ValueError('joint observable selection currently accepts native intensity with an explicit dipole')
        bound = np.broadcast_to(np.asarray(raw, float), np.shape(full[name])).copy()
        if not bound.size or not np.all(np.isfinite(bound)) or np.any(bound <= 0):
            raise ValueError('positive finite absolute observable budgets required')
        property_budgets[name] = bound
    reference=(full['solved'],full_surface,full['context'])
    def envelope(value):
        names=next(names for names,(_,entry,_) in cache.items() if entry is value)
        pieces=[numerics[n]['envelope_MHz'] for n in (names,all_names)]
        return sum(np.full(len(budget),np.inf) if piece is None else np.asarray(piece) for piece in pieces)
    def property_envelope(value, name):
        if policy is None:
            return np.zeros_like(property_budgets[name])
        names=next(names for names,(_,entry,_) in cache.items() if entry is value)
        pieces=[numerics[n].get('observable_envelopes',{}).get(name) for n in (names,all_names)]
        return sum(np.full_like(property_budgets[name],np.inf) if piece is None else np.asarray(piece)
                   for piece in pieces)
    def error(value):
        ratios = [float(np.max((np.abs(value['frequency']-full['frequency'])+envelope(value))/budget,initial=0.))]
        for name, bound in property_budgets.items():
            observed = value.get(name)
            if observed is None or np.shape(observed) != bound.shape or not np.all(np.isfinite(observed)):
                raise ValueError('native property observable missing or nonfinite in a candidate')
            ratios.append(float(np.max((np.abs(observed-full[name])+property_envelope(value,name))/bound, initial=0.)))
        return max(ratios)
    selected=closure(mandatory_names)
    surface,current,_=evaluate(selected,reference)
    while error(current)>1 and selected!=all_names and (policy is None or numerics[all_names]['converged']):
        trials=[]
        for b in blocks:
            target=closure((*selected,b.name))
            if target==selected:continue
            if target not in cache and len(cache)>=maximum_evaluations:continue
            candidate_surface,candidate,seconds=evaluate(target,reference)
            cost=sum(byname[n].estimated_cost for n in target-selected)
            score=((error(current)-error(candidate)) if np.isfinite(error(current)) else -error(candidate))/cost
            trials.append((score,error(candidate),tuple(sorted(target)),
                target,candidate_surface,candidate,seconds))
        if not trials:break
        chosen=min(trials,key=lambda row:(-row[0],row[1],row[2]))
        selected,surface,current=chosen[3:6]
        history.append(dict(selected_blocks=sorted(selected),maximum_error_ratio=error(current),
            measured_evaluation_seconds=chosen[6],score=float(chosen[0]) if np.isfinite(chosen[0]) else None))
    difference=current['frequency']-full['frequency']
    coupled_modes=sorted({int(term[0]) for b in blocks if b.name in selected for term in b.active_pairs}
        | {int(mode) for b in blocks if b.name in selected for term in b.contracted_pairs for mode in term[:2]})
    def finite_scalar(value):return float(value) if np.isfinite(value) else None
    def finite_vector(values):return [finite_scalar(x) for x in values]
    for item in history:item['maximum_error_ratio']=finite_scalar(item['maximum_error_ratio'])
    probes=[dict(blocks=sorted(names),maximum_error_ratio=finite_scalar(error(value)),seconds=seconds,
        declared_cost=sum(byname[name].estimated_cost for name in names))
        for names,(_,value,seconds) in sorted(cache.items(),key=lambda row:tuple(sorted(row[0])))]
    return AdaptiveCouplingResult(surface,dict(schema='matrix.lock.adaptive_nmode_couplings.v1',
        qualified=error(current)<=1 and (policy is None or
            (numerics[selected]['converged'] and numerics[all_names]['converged'])),selected_blocks=sorted(selected),omitted_blocks=sorted(all_names-selected),
        mandatory_blocks=sorted(closure(mandatory_names)),frequency_budget_MHz=budget.tolist(),
        coupled_mode_indices=coupled_modes,evaluated_models=probes,
        selection_rule='largest full-reference observable-error reduction per declared added cost; deterministic name tie break',
        maximum_error_ratio=finite_scalar(error(current)),selected_minus_full_MHz=difference.tolist(),
        evaluation_count=len(cache),maximum_evaluations=maximum_evaluations,history=history,
        full_frequencies_MHz=full['frequency'].tolist(),selected_frequencies_MHz=current['frequency'].tolist(),
        basis_convergence_certified=bool(policy is not None and numerics[selected]['converged'] and numerics[all_names]['converged']),
        numerical_policy=policy,numerical_checks=[dict(blocks=sorted(names),**report) for names,report in numerics.items()],
        numerical_envelope_MHz=finite_vector(envelope(current)),
        combined_error_estimate_MHz=finite_vector(abs(difference)+envelope(current)),
        experimental_data_used_for_selection=False,
        observable_comparisons={name:dict(budget=bound.tolist(),
            full=np.asarray(full[name]).tolist(),selected=np.asarray(current[name]).tolist(),
            selected_minus_full=(current[name]-full[name]).tolist(),
            numerical_envelope=finite_vector(property_envelope(current,name)),
            combined_error_estimate=finite_vector(abs(current[name]-full[name])+property_envelope(current,name))) for name,bound in property_budgets.items()},
        property_basis_convergence_certified=bool(property_budgets and policy is not None
            and all(numerics[n]['converged'] and numerics[n].get('properties_converged',False)
                    for n in (selected,all_names))),
        basis_convergence_scope='frequencies and requested intensity observables on declared numerical axes',
        scope='joint compression of supplied PES, kinetic and property blocks; no missing-surface or predictive-accuracy claim'),
        _carry_assignments(
            deepcopy(current['context'].get('_nmode_numerical_request',request)),current))


__all__=['NModeCouplingBlock','AdaptiveCouplingResult','build_nmode_lam_surface','select_nmode_couplings',
         'apply_physical_field_blocks']
