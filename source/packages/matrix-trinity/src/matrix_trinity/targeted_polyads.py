"""Target-directed refinement of finite dressed vibrational subspaces.

Selection uses coherent external corrections to requested eigenvectors, not
unweighted ratios for every basis state. The returned convergence concerns
that explicitly supplied external wave-operator estimate and changes on
actual expansions; it is not a full-VCI residual certificate.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix

from .curvilinear_vpt2 import (
    _canonical_cubic_terms, _canonical_quartic_terms,
    _compiled_operator_couplings, _compiled_operator_matrix_element,
    _modalize_sonic_field, _operator_terms_by_changed_modes, _resonance_kind,
)


def _encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {'complex': [value.real, value.imag]}
    raise TypeError(type(value).__name__)


def _decode(value):
    return complex(*value['complex']) if set(value) == {'complex'} else value


def _merged(components):
    merged = []
    for component in components:
        component = set(component)
        while True:
            overlaps = [group for group in merged if group & component]
            if not overlaps:
                break
            for group in overlaps:
                component.update(group)
                merged.remove(group)
        merged.append(component)
    return sorted(merged, key=lambda group: min(group))


class CubicDressedWorkspace:
    """Same-field H0+V3+V4 in P; symmetric external cubic PT2 dressing.

    Quartics enter each internal Hamiltonian in their supplied coverage.
    The external selection estimate here is cubic: first-order V3 wave
    corrections in algebraic 1<->2 exchange channels. External quartic effects are not certified by this adapter.
    """

    def __init__(self, field, *, maximum_cached_couplings=2000000, allow_reduced_quartic_model=False):
        if field.quartic_coverage != "complete" and not allow_reduced_quartic_model:
            raise ValueError("internal closure requires complete quartics or explicit reduced-polynomial consent")
        self.pseudopotential_cm=float(field.pseudopotential_cm)
        if (isinstance(maximum_cached_couplings, bool)
                or not isinstance(maximum_cached_couplings, int)
                or maximum_cached_couplings < 1):
            raise ValueError('positive integer coupling-cache budget required')
        self.frequencies, _, f3, f4, g1, g2 = _modalize_sonic_field(field)
        self.cubic_terms = _canonical_cubic_terms(f3, g1, 0.)
        self.quartic_terms = _canonical_quartic_terms(f4, g2)
        self.quartic_index = _operator_terms_by_changed_modes(self.quartic_terms)
        self.cache = {}
        self.cached_count = 0
        self.maximum_cached_couplings = maximum_cached_couplings
        self.components = OrderedDict()
        self.component_cache_hits = 0
        self.component_assemblies = 0

    def cubic(self, state):
        if state not in self.cache:
            values = {q:v for q,v in _compiled_operator_couplings(state, self.cubic_terms).items()
                      if abs(v) > 1e-12}
            if self.cached_count+len(values) > self.maximum_cached_couplings:
                raise ValueError('cubic coupling-cache budget exceeded')
            self.cache[state] = values
            self.cached_count += len(values)
        return self.cache[state]

    def matrix(self, states):
        return self.component(states)['matrix'].copy()

    def component(self, states):
        states=tuple(map(tuple,states))
        if states in self.components:
            self.component_cache_hits+=1
            self.components.move_to_end(states)
            return self.components[states]
        self.component_assemblies+=1
        energies = np.asarray(states) @ self.frequencies
        indices = {state:i for i,state in enumerate(states)}
        matrix = np.diag(energies.astype(complex))
        external = {}
        rows, cols, values, gaps = [], [], [], []
        for i,state in enumerate(states):
            for target,value in self.cubic(state).items():
                if target in indices:
                    matrix[indices[target],i] += value
                else:
                    gap = energies[i]-np.dot(self.frequencies,target)
                    if abs(gap) < 1e-6:
                        raise ValueError('unresolved external cubic degeneracy; enlarge initial component')
                    rows.append(i)
                    cols.append(external.setdefault(target,len(external)))
                    values.append(value.conjugate())
                    gaps.append(gap)
            for j in range(i+1):
                value = _compiled_operator_matrix_element(state, states[j], self.quartic_terms,
                                                          self.quartic_index)
                matrix[j,i] += value
                if i != j:
                    matrix[i,j] += value.conjugate()
        shape = (len(states),len(external))
        coupling = coo_matrix((values,(rows,cols)),shape=shape).tocsr()
        resolvent = coo_matrix((np.asarray(values)/gaps,(rows,cols)),shape=shape).tocsr()
        sigma = (resolvent @ coupling.conjugate().T).toarray()
        result=dict(matrix=matrix+(sigma+sigma.conjugate().T)/2+self.pseudopotential_cm*np.eye(len(states)),
                    external_states=tuple(external), resolvent=resolvent)
        self.components[states]=result
        if len(self.components)>128:
            self.components.popitem(last=False)
        return result

    def lifted_wavefunctions(self, states, columns, _eigenvalues):
        """Canonical lift [I - Omega^dagger Omega/2; Omega] C.

        Unlike scalar renormalization, the P correction retains off-diagonal
        overlap terms. Its Gram matrix is I+O(V3^4). Q second-order property
        dressing belongs to the existing spectroscopy contraction, not this
        first-order mechanical selection estimate.
        """
        columns=np.asarray(columns,complex)
        component=self.component(states)
        resolvent=component['resolvent']
        external=resolvent.conjugate().T @ columns
        internal=columns-.5*(resolvent @ external)
        result={tuple(state):row for state,row in zip(states,internal)}
        result.update({state:row for state,row in zip(component['external_states'],external)
                       if np.any(row)})
        return result

    def balanced_expansion(self, pending, models, proposals, targets):
        """Transport each selected cubic exchange to every compatible target.

        The dominant coherent source contribution identifies an operator
        occupation change, not a molecule or a fitted transition. Exact
        source-field matrix elements decide whether its translated partners
        exist. This balances spectator families without unweighted closure
        over the whole graph.
        """
        pending=[set(group) for group in pending]
        owner={tuple(target):i for i,model in enumerate(models) for target in model['targets']}
        families=[]
        for _,external,component_index in proposals:
            model=models[component_index]
            states=tuple(map(tuple,model['states']))
            vectors=np.asarray(model['eigenvectors'])[:,np.asarray(model['target_roots'])]
            source_weights=np.linalg.norm(vectors,axis=1)
            candidates=[]
            for i,state in enumerate(states):
                if _resonance_kind(state,external)!='1-2':
                    continue
                value=self.cubic(state).get(external,0j)
                gap=float(np.dot(self.frequencies,np.subtract(state,external)))
                if abs(value)>1e-12 and abs(gap)>1e-6:
                    candidates.append((abs(value/gap)*source_weights[i],state))
            if not candidates:
                continue
            _,source=max(candidates,key=lambda row:(row[0],row[1]))
            delta=tuple(q-p for p,q in zip(source,external))
            added=[]
            for target in targets:
                partner=tuple(p+d for p,d in zip(target,delta))
                if min(partner)<0 or partner==target:
                    continue
                if abs(self.cubic(tuple(target)).get(partner,0j))<=1e-12:
                    continue
                group=pending[owner[tuple(target)]]
                if partner not in group:
                    group.add(partner)
                    added.append(partner)
            families.append(dict(source=source,partner=external,delta=delta,translated_additions=added))
        return pending,families

    def external_corrections(self, states, columns, eigenvalues):
        """Complete cubic Q Omega^(1) P C for dressed-vector comparisons."""
        return self._external(states,columns,eigenvalues,fermi_only=False)

    def fermi_corrections(self, states, columns, eigenvalues):
        """Only cubic 1<->2 occupation exchanges can have resonance poles.

        Pure 0<->1 and 0<->3 changes have positive sums of frequencies in
        their denominators at a vibrational minimum. They stay in external
        PT2 dressing, irrespective of their amplitude; no molecular label or
        fitted energy window enters this algebraic channel classification.
        """
        return self._external(states,columns,eigenvalues,fermi_only=True)

    def _external(self, states, columns, _eigenvalues, *, fermi_only):
        members = set(states)
        energies = np.asarray(states) @ self.frequencies
        corrections = {}
        for i,state in enumerate(states):
            for target,value in self.cubic(state).items():
                if target in members or (fermi_only and _resonance_kind(state,target) != '1-2'):
                    continue
                gap = energies[i]-np.dot(self.frequencies,target)
                if abs(gap) < 1e-6:
                    raise ValueError('unresolved external cubic wave-operator denominator')
                if target not in corrections:
                    corrections[target] = np.zeros(columns.shape[1],complex)
                corrections[target] += value*columns[i]/gap
        return corrections


class CanonicalGVPT2Workspace:
    """Restrictions of one occupation-indexed GVPT2 operator, independent of P.

    External estimates include every retained Fermi/DD edge. The contact lift
    is the fixed first-order nonresonant frame; it supplies a common
    physical reference for tracking, while spectroscopy retains its PT2 rules.
    """

    def __init__(self, field, *, operator_options, allow_reduced_quartic_model=False, operator_workspace=None):
        from .gvpt2_operator import GVPT2EffectiveOperator
        if field.quartic_coverage != 'complete' and not allow_reduced_quartic_model:
            raise ValueError('internal closure requires complete quartics or explicit reduced-polynomial consent')
        self.operator=GVPT2EffectiveOperator(field, workspace=operator_workspace, **operator_options)
        self.reduced=allow_reduced_quartic_model
        self.components=OrderedDict()
        self.component_cache_hits=0
        self.component_assemblies=0
        self.edges={}

    @property
    def cached_count(self):
        return sum(map(len,self.operator.cubic_cache.values()))

    def coupling(self,a,b):
        return self.operator.coupling(a,b,reduced_polynomial=self.reduced)

    def matrix(self,states):
        states=tuple(map(tuple,states))
        matrix=np.zeros((len(states),len(states)),complex)
        for i,a in enumerate(states):
            matrix[i,i]=self.operator.diagonal(a)
            for j,b in enumerate(states[:i]):
                value=self.coupling(a,b)
                matrix[i,j]=value
                matrix[j,i]=np.conjugate(value)
        return matrix

    def lifted_wavefunctions(self,states,columns,_eigenvalues):
        states=tuple(map(tuple,states))
        if states not in self.components:
            self.component_assemblies+=1
            rows={s:np.eye(len(states),dtype=complex)[i] for i,s in enumerate(states)}
            for i,a in enumerate(states):
                for q,value in self.operator.cubic(a).items():
                    if self.operator.resonant_fermi(a,q,value):
                        continue
                    gap=self.operator.harmonic(a)-self.operator.harmonic(q)
                    if abs(gap)<=self.operator.denominator_tolerance:
                        if abs(value)>self.operator.zero:
                            raise ValueError('unresolved nonresonant contact denominator')
                        continue
                    if q not in rows:
                        rows[q]=np.zeros(len(states),complex)
                    rows[q][i]+=value/gap
            keys=tuple(rows)
            frame=np.asarray(list(rows.values()))
            # Keep this map independent of P. Polar orthogonalization of its
            # selected columns would rotate old functions on every expansion.
            # The engine uses the full Gram metric and normalizes each column.
            self.components[states]=(keys,frame)
            if len(self.components)>32:
                self.components.popitem(last=False)
        else:
            self.component_cache_hits+=1
        keys,frame=self.components[states]
        return dict(zip(keys,frame@columns))

    def external_corrections(self,states,columns,eigenvalues):
        members=set(states)
        coupling={}
        for i,a in enumerate(states):
            if a not in self.edges:
                self.edges[a]={q:self.coupling(q,a) for q in self.operator.neighbors(a)}
                self.edges[a]={q:v for q,v in self.edges[a].items() if abs(v)>0}
            for q,value in self.edges[a].items():
                if q in members:
                    continue
                if q not in coupling:
                    coupling[q]=np.zeros(columns.shape[1],complex)
                coupling[q]+=value*columns[i]
        corrections={}
        for q,value in coupling.items():
            gaps=np.asarray(eigenvalues)-self.operator.diagonal(q)
            safe=np.where(abs(gaps)>self.operator.denominator_tolerance,gaps,
                          np.where(gaps<0,-1.,1.)*self.operator.denominator_tolerance)
            corrections[q]=value/safe
        return corrections

    def balanced_expansion(self,pending,models,proposals,targets):
        """Translate whole source/partner paths between compatible target families."""
        pending=[set(group) for group in pending]
        owner={tuple(t):i for i,m in enumerate(models) for t in m['targets']}
        families=[]
        for _,q,index in proposals:
            model=models[index]
            states=tuple(map(tuple,model['states']))
            vectors=np.asarray(model['eigenvectors'])[:,np.asarray(model['target_roots'])]
            weights=[abs(self.coupling(q,a))*np.linalg.norm(vectors[i]) for i,a in enumerate(states)]
            if not max(weights,default=0):
                continue
            source=states[int(np.argmax(weights))]
            added=[]
            for origin in model['targets']:
                for target in targets:
                    shift=np.subtract(target,origin)
                    parent=tuple(np.asarray(source)+shift)
                    partner=tuple(np.asarray(q)+shift)
                    group=pending[owner[tuple(target)]]
                    if min(parent)<0 or min(partner)<0 or parent not in group or partner in group:
                        continue
                    if abs(self.coupling(partner,parent))>0:
                        group.add(partner)
                        added.append(partner)
            families.append(dict(source=source,partner=q,translated_additions=added))
        return pending,families


def refine_targeted_polyads(components, targets, build_matrix, external_corrections, *,
        energy_tolerance_cm=0.01, wavefunction_tolerance=0.01,
        external_amplitude_tolerance=0.1, maximum_states=512, maximum_quanta=6,
        maximum_iterations=12, batch_size=4, checkpoint_path=None, identity=None,
        initial_tracked_states=None, energy_combinations=None, dressed_corrections=None,
        admissible_state=None, component_completion=None, dressed_wavefunctions=None,
        expansion_completion=None):
    """Refine requested states; stop honestly at convergence or resource guards.

    ``external_corrections(states, target_columns, target_energies)`` returns
    occupation -> one complex estimated correction per requested target.
    Arbitrary operator adapters can supply this callback. Estimates must
    cover omitted states outside quantum limits too: limits are not masks.
    ``batch_size`` limits additions per component, so requested branches do
    not starve behind unrelated targets. Wavefunction changes include the
    supplied external correction and are compared after normalization. Optional
    ``energy_combinations`` rows define linear observables of target energies.
    Changes are measured on genuine support expansions, never by solving an
    unchanged matrix twice. Roots follow overlaps with the preceding vectors.
    Checkpoints preserve accepted support and tracked columns for resumption.
    """
    for value in (energy_tolerance_cm,wavefunction_tolerance,external_amplitude_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('positive finite refinement tolerances required')
    for value in (maximum_states,maximum_quanta,maximum_iterations,batch_size):
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError('positive integer refinement budgets required')
    dressing = external_corrections if dressed_corrections is None else dressed_corrections
    allowed = (lambda state: True) if admissible_state is None else admissible_state
    targets = tuple(tuple(state) for state in targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('distinct target occupations required')
    n = len(targets[0])
    def valid(state):
        return (len(state)==n and all(not isinstance(v,bool) and isinstance(v,(int,np.integer))
                                      and v>=0 for v in state))
    complete = (lambda group: set(group)) if component_completion is None else component_completion
    groups = _merged([set(map(tuple,complete(set(map(tuple,group))))) for group in components])
    if (not groups or any(not valid(s) for g in groups for s in g)
            or any(not valid(s) for s in targets)
            or not set(targets).issubset(set().union(*groups))
            or any(not (set(targets)&g) for g in groups)):
        raise ValueError('valid target-containing initial components required')
    if len(set().union(*groups)) > maximum_states or any(sum(s)>maximum_quanta or not allowed(s) for g in groups for s in g):
        raise ValueError('initial support exceeds refinement budget')
    combinations = np.eye(len(targets)) if energy_combinations is None else np.asarray(energy_combinations,float)
    if (combinations.ndim != 2 or combinations.shape[1] != len(targets)
            or len(combinations) == 0 or not np.all(np.isfinite(combinations))
            or np.any(np.sum(abs(combinations),axis=1) == 0)):
        raise ValueError('finite nonzero energy-combination rows required')
    controls = dict(energy_tolerance_cm=energy_tolerance_cm, wavefunction_tolerance=wavefunction_tolerance,
        external_amplitude_tolerance=external_amplitude_tolerance, maximum_states=maximum_states,
        maximum_quanta=maximum_quanta, maximum_iterations=maximum_iterations,batch_size=batch_size)
    signature = dict(profile='targeted-external-wave-operator-v1', identity=identity,
        components=[sorted(g) for g in groups], targets=targets, controls=controls,
        energy_combinations=combinations, initial_tracked_states=initial_tracked_states,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # JSON normalization makes occupation tuples compare identically after reload.
    signature = json.loads(json.dumps(signature,default=_encode))
    previous = {} if initial_tracked_states is None else {tuple(row['target']):row for row in initial_tracked_states}
    if previous and set(previous) != set(targets):
        raise ValueError('initial tracked columns must cover precisely the requested targets')
    history, models = [], []
    resumed = False
    repeat_support = False
    path = None if checkpoint_path is None else Path(checkpoint_path)
    if path is not None and path.exists():
        saved = json.loads(path.read_text(),object_hook=_decode)
        # Initial tracked columns can be complex; compare JSON-normalized
        # signatures without losing their decoded numerical result arrays.
        saved['signature']=json.loads(json.dumps(saved['signature'],default=_encode))
        increased_budget=False
        if saved['signature'] != signature:
            old_signature=json.loads(json.dumps(saved['signature']))
            new_signature=json.loads(json.dumps(signature))
            resource_keys=('maximum_states','maximum_quanta','maximum_iterations')
            monotone=all(new_signature['controls'][key]>=old_signature['controls'][key] for key in resource_keys)
            for key in resource_keys:
                old_signature['controls'].pop(key)
                new_signature['controls'].pop(key)
            if old_signature!=new_signature or not monotone:
                raise ValueError('targeted checkpoint physical input, implementation or tolerances differ')
            increased_budget=True
        if saved['finished'] and (saved['result']['converged'] or not increased_budget):
            saved['result']['resumed'] = True
            return saved['result']
        if saved['finished']:
            groups=[set(map(tuple,model['states'])) for model in saved['result']['models']]
            history=saved['result']['history']
            previous={tuple(row['target']):row for row in saved['result']['tracked_states']}
            # The first resumed solve repeats accepted support to construct its
            # next proposal. Its zero increment cannot certify convergence.
            repeat_support=True
        else:
            groups = [set(map(tuple,g)) for g in saved['pending_components']]
            history = saved['history']
            previous = {tuple(row['target']):row for row in saved['previous']}
        resumed = True

    def save(result, pending=None):
        if path is None:
            return
        payload = dict(signature=signature,finished=pending is None,result=result,
            pending_components=None if pending is None else [sorted(g) for g in pending],
            previous=list(previous.values()),history=history)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(payload,default=_encode,allow_nan=False)+'\n')
        temporary.replace(path)

    reason = 'iteration_budget'
    converged = False
    for iteration in range(len(history)+1,maximum_iterations+1):
        models, current, candidates = [], {}, []
        energy_changes, vector_changes, scores, norms = [], [], [], []
        vector_change_by_target={}
        for group_index,group in enumerate(groups):
            states = tuple(sorted(group,key=lambda s:(sum(s),s)))
            local_targets = tuple(t for t in targets if t in group)
            matrix = np.asarray(build_matrix(states),complex)
            if (matrix.shape != (len(states),len(states)) or not np.all(np.isfinite(matrix))
                    or np.max(abs(matrix-matrix.conj().T)) > 1e-8):
                raise ValueError('finite Hermitian selected Hamiltonian required')
            energies,vectors = np.linalg.eigh(matrix)
            references = np.zeros((len(states),len(local_targets)),complex)
            positions = {s:i for i,s in enumerate(states)}
            lifted_basis=None
            root_norms=np.ones(len(states))
            if dressed_wavefunctions is not None:
                lifted_basis=dressed_wavefunctions(states,np.eye(len(states)),matrix.diagonal().real)
                rows=np.asarray(list(lifted_basis.values()))
                gram=rows.conjugate().T@rows
                root_norms=np.sqrt(np.real(np.sum(vectors.conjugate()*(gram@vectors),axis=0)))
            old_physical={}
            for j,target in enumerate(local_targets):
                if target in previous:
                    old = previous[target]
                    if lifted_basis is None:
                        for state,value in zip(old['states'],old['column']):
                            references[positions[tuple(state)],j] = value
                    else:
                        wave=dressed_wavefunctions(tuple(map(tuple,old['states'])),
                            np.asarray(old['column'])[:,None],np.array([old['energy_cm']]))
                        old_physical[target]=wave
                        old_norm=np.sqrt(sum(abs(value[0])**2 for value in wave.values()))
                        for state,value in wave.items():
                            if state in lifted_basis:
                                references[:,j]+=lifted_basis[state].conjugate()*value[0]/old_norm
                elif lifted_basis is None:
                    references[positions[target],j] = 1.
                else:
                    references[:,j]=lifted_basis[target].conjugate()
            rr,cc = linear_sum_assignment(-abs(references.conj().T@vectors)**2/root_norms[None,:]**2)
            roots = cc[np.argsort(rr)]
            columns = vectors[:,roots].copy()
            for j,target in enumerate(local_targets):
                overlap = np.vdot(references[:,j],columns[:,j])
                if abs(overlap):
                    columns[:,j] *= overlap.conjugate()/abs(overlap)
                current[target] = dict(target=target,states=states,column=columns[:,j],energy_cm=float(energies[roots[j]]))
                if target in previous:
                    energy_changes.append(abs(energies[roots[j]]-previous[target]['energy_cm']))
            corrections = external_corrections(states,columns,energies[roots])
            dressed = corrections if dressed_corrections is None else dressing(states,columns,energies[roots])
            norm = np.zeros(len(local_targets))
            for state,values in corrections.items():
                state = tuple(state)
                values = np.asarray(values,complex)
                if not valid(state) or state in group or values.shape != (len(local_targets),) or not np.all(np.isfinite(values)):
                    raise ValueError('invalid external target corrections')
                # Row norm is invariant to unitary rotations of a complete
                # requested target subspace, unlike a largest-column score.
                score = float(np.linalg.norm(values))
                norm += abs(values)**2
                if score > 1e-14:
                    candidates.append((score,state,group_index))
            for j,target in enumerate(local_targets):
                if target not in previous:
                    continue
                old=previous[target]
                old_states=tuple(map(tuple,old['states']))
                old_column=np.asarray(old['column'],complex)
                if dressed_wavefunctions is None:
                    old_corrections=dressing(old_states,old_column[:,None],np.array([old['energy_cm']]))
                    old_wave=dict(zip(old_states,old_column))
                    old_wave.update({state:np.asarray(value)[0] for state,value in old_corrections.items()})
                    new_wave=dict(zip(states,columns[:,j]))
                    new_wave.update({state:np.asarray(value)[j] for state,value in dressed.items()})
                else:
                    old_wave={state:values[0] for state,values in old_physical[target].items()}
                    lifted=dressed_wavefunctions(states,columns[:,j:j+1],energies[roots[j:j+1]])
                    new_wave={state:values[0] for state,values in lifted.items()}
                old_norm=sum(abs(value)**2 for value in old_wave.values())
                new_norm=sum(abs(value)**2 for value in new_wave.values())
                overlap=sum(value.conjugate()*new_wave.get(state,0j) for state,value in old_wave.items())
                change=float(np.sqrt(max(0.,2.-2.*min(1.,abs(overlap)/np.sqrt(old_norm*new_norm)))))
                vector_changes.append(change)
                vector_change_by_target[target]=change
            norms.extend(np.sqrt(norm).tolist())
            scores.append(max((score for score,_,gi in candidates if gi==group_index),default=0.))
            models.append(dict(states=states,hamiltonian_cm=matrix,eigenvalues_cm=energies,
                               eigenvectors=vectors,targets=local_targets,target_roots=roots))
        maximum_score = max(scores,default=0.)
        de = (float(np.max(abs(combinations @ np.array([current[t]['energy_cm']-previous[t]['energy_cm']
                                                       for t in targets])))) if previous and history and not repeat_support else None)
        dv = max(vector_changes,default=None) if history and not repeat_support else None
        observable_changes=None
        if previous and history and not repeat_support:
            observable_changes=(combinations@np.array([current[t]['energy_cm']-previous[t]['energy_cm'] for t in targets])).tolist()
        history.append(dict(iteration=iteration,states=sum(map(len,groups)),component_sizes=list(map(len,groups)),
            energy_change_cm=de,energy_observable_changes_cm=observable_changes,
            wavefunction_changes_by_target=[vector_change_by_target.get(t) for t in targets],maximum_raw_level_change_cm=max(energy_changes,default=None),wavefunction_change=dv,
            maximum_external_amplitude=maximum_score,
            maximum_external_estimated_norm=max(norms,default=0.)))
        previous = current
        repeat_support=False
        if maximum_score <= external_amplitude_tolerance and ((de is not None and de<=energy_tolerance_cm
                and dv<=wavefunction_tolerance) or not candidates):
            converged=True
            reason='target_criteria_satisfied' if candidates else 'no_external_corrections'
            break
        violating_domain = [s for score,s,_ in candidates if score>external_amplitude_tolerance and not allowed(s)]
        if violating_domain:
            reason='physical_domain_with_significant_target_correction'
            break
        violating_boundary = [s for score,s,_ in candidates if score>external_amplitude_tolerance and sum(s)>maximum_quanta]
        if violating_boundary:
            reason='quantum_boundary_with_significant_target_correction'
            break
        eligible=sorted((entry for entry in candidates if sum(entry[1])<=maximum_quanta and allowed(entry[1])),key=lambda x:(-x[0],x[1],x[2]))
        if not eligible:
            reason='quantum_domain_exhausted_before_expansion_check'
            break
        pending=[set(group) for group in groups]
        counts={}
        chosen=[]
        for score,state,group_index in eligible:
            if counts.get(group_index,0)>=batch_size:
                continue
            pending[group_index].add(state)
            chosen.append((score,state,group_index))
            counts[group_index]=counts.get(group_index,0)+1
        if expansion_completion is not None:
            pending,expansion_info=expansion_completion(pending,models,chosen,targets)
            history[-1]['operator_family_expansion']=expansion_info
        pending=_merged([set(map(tuple,complete(group))) for group in pending])
        if any(sum(state)>maximum_quanta or not allowed(state) for group in pending for state in group):
            reason='component_completion_crosses_domain_boundary'
            break
        if len(set().union(*pending))>maximum_states:
            reason='state_budget'
            break
        if iteration==maximum_iterations:
            break
        save(dict(converged=False,reason='pending_expansion'),pending)
        groups=pending
    result=dict(schema='matrix.trinity.targeted-polyads.v1',converged=converged,reason=reason,
        resumed=resumed,controls=controls,history=history,models=models,tracked_states=list(previous.values()),
        scope='Convergence of supplied external wave-operator estimate and actual target energy/vector increments; not a full-VCI error bound.')
    save(result)
    return result


class TargetedPolyadConvergenceError(ValueError):
    """No qualified spectrum is returned when a declared refinement fails."""

    def __init__(self, result):
        self.result=result
        last=result['history'][-1] if result['history'] else {}
        super().__init__(f"target-directed polyad refinement not converged: {result['reason']}; "
                         f"last accepted diagnostics: {last}")


def refine_sonic_gvpt2_polyads(field, reference, *, target_states, require_convergence=True,
        allow_reduced_quartic_model=False, maximum_mode_quanta=None,
        polyad_excluded_modes=(), return_state=False, balance_target_families=True,
        warm_start_checkpoint=None, operator_options=None, seed_components=None,
        _operator_workspace=None, **controls):
    """Production adapter for native GVPT2 polyads and their property columns.

    The public fixed/adaptive solvers share the canonical GVPT2 contractions.
    The perturbative partition is fixed before support expansion. Fermi and
    DD external edges are estimated coherently for the requested columns.
    Ground is tracked to preserve a common origin. Reduced quartics require
    explicit consent. Historical references must supply their operator profile.
    """
    from dataclasses import replace
    from .curvilinear_vpt2 import _gvpt2_report

    targets=tuple(dict.fromkeys(tuple(state) for state in target_states))
    n=len(reference.frequencies_cm)
    ground=(0,)*n
    if ground not in targets:
        if controls.get('energy_combinations') is not None:
            combinations=np.asarray(controls['energy_combinations'],float)
            controls['energy_combinations']=np.column_stack((combinations,np.zeros(len(combinations))))
        targets=(*targets,ground)
    if not set(targets).issubset(reference.states):
        raise ValueError('refinement targets must belong to the supplied GVPT2 reference')
    components=[{reference.states[i] for i in block['indices']} for block in reference.blocks
                if any(reference.states[i] in targets for i in block['indices'])]
    grouped=set().union(*components) if components else set()
    components.extend({state} for state in targets if state not in grouped)
    if seed_components is not None:
        acquired=[set(map(tuple,group)) for group in seed_components]
        if any(not group or not group.issubset(reference.states) for group in acquired):
            raise ValueError('acquired seed occupations must belong to the supplied reference')
        components=_merged([*components,*acquired])
    excluded=tuple(polyad_excluded_modes)
    if (len(set(excluded))!=len(excluded) or any(not isinstance(mode,int) or not 0<=mode<n for mode in excluded)):
        raise ValueError('valid distinct excluded-mode indices required')
    if maximum_mode_quanta is not None and (isinstance(maximum_mode_quanta,bool)
            or not isinstance(maximum_mode_quanta,int) or maximum_mode_quanta<1):
        raise ValueError('positive per-mode occupation guard required')
    def admissible(state):
        return ((maximum_mode_quanta is None or max(state)<=maximum_mode_quanta)
                and not any(state[mode] for mode in excluded))
    digest=hashlib.sha256()
    for value in (field.harmonic_sonic,field.cubic_sonic,field.quartic_sonic,
                  field.metric_sonic,field.metric_first_sonic,field.metric_second_sonic):
        array=np.ascontiguousarray(value,dtype='<f8')
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(str((field.pseudopotential_cm,field.quartic_coverage)).encode())
    if field.symmetry_actions is not None:
        digest.update(np.ascontiguousarray(field.symmetry_actions,dtype='<f8').tobytes())
        digest.update(str(field.symmetry_tolerance).encode())
        digest.update(hashlib.sha256(Path(__file__).with_name('field_symmetry.py').read_bytes()).digest())
    profile=reference.diagnostics.get('effective_operator_profile')
    if operator_options is not None:
        if profile is not None and json.dumps(profile,sort_keys=True)!=json.dumps(operator_options,sort_keys=True):
            raise ValueError('operator options differ from supplied GVPT2 reference')
        profile=dict(operator_options)
    if profile is None:
        raise ValueError('historical GVPT2 reference requires explicit operator_options')
    profile=dict(profile)
    if tuple(profile.get('polyad_excluded_modes',()))!=excluded:
        raise ValueError('operator profile and refinement excluded modes differ')
    source_identity=dict(operator_profile=profile,
        operator_sha256=hashlib.sha256(Path(__file__).with_name('gvpt2_operator.py').read_bytes()).hexdigest(),
        field_sha256=digest.hexdigest(),
        reference_sha256=hashlib.sha256(json.dumps(dict(states=reference.states,
            energies=reference.energies_cm,blocks=reference.blocks),default=_encode).encode()).hexdigest(),
        curvilinear_solver_sha256=hashlib.sha256(Path(__file__).with_name('curvilinear_vpt2.py').read_bytes()).hexdigest(),
        maximum_mode_quanta=maximum_mode_quanta,excluded_modes=excluded,
        allow_reduced_quartic_model=allow_reduced_quartic_model,balance_target_families=balance_target_families)
    if warm_start_checkpoint is not None:
        path=Path(warm_start_checkpoint)
        saved=json.loads(path.read_text(),object_hook=_decode)
        previous_identity=saved["signature"]["identity"]
        for key in ("field_sha256","reference_sha256","maximum_mode_quanta","excluded_modes","allow_reduced_quartic_model"):
            old_value=json.loads(json.dumps(previous_identity.get(key),default=_encode))
            new_value=json.loads(json.dumps(source_identity.get(key),default=_encode))
            if old_value!=new_value:
                raise ValueError("warm-start checkpoint physical reference or domain differs")
        state=saved["result"] if saved["finished"] else dict(models=[dict(states=g) for g in saved["pending_components"]],tracked_states=saved["previous"])
        components=[set(map(tuple,model["states"])) for model in state["models"]]
        # Different operators can share acquired occupations, never convergence
        # history or eigenvectors. Reassemble the new fixed operator on this P.
        if json.dumps(previous_identity.get('operator_profile'),sort_keys=True)==json.dumps(profile,sort_keys=True):
            controls["initial_tracked_states"]=state["tracked_states"]
        source_identity["warm_start_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    def complete(group):
        group=set(group)
        for block in reference.blocks:
            members={reference.states[i] for i in block['indices']}
            if group & members:
                group.update(members)
        return group
    workspace=CanonicalGVPT2Workspace(field,operator_options=profile,allow_reduced_quartic_model=allow_reduced_quartic_model,
        operator_workspace=_operator_workspace)
    refinement=refine_targeted_polyads(components,targets,workspace.matrix,workspace.external_corrections,
        dressed_corrections=workspace.external_corrections,admissible_state=admissible,
        identity=source_identity,component_completion=complete,
        dressed_wavefunctions=workspace.lifted_wavefunctions,
        expansion_completion=workspace.balanced_expansion if balance_target_families else None,**controls)
    refinement["workspace_cache"]=dict(component_assemblies=workspace.component_assemblies,component_cache_hits=workspace.component_cache_hits,cubic_couplings=workspace.cached_count)
    if not refinement['converged'] and require_convergence:
        raise TargetedPolyadConvergenceError(refinement)
    extras={tuple(s) for model in refinement['models'] for s in model['states']}-set(reference.states)
    states=(*reference.states,*sorted(extras,key=lambda s:(sum(s),s)))
    indices={state:i for i,state in enumerate(states)}
    tracked={tuple(row['target']):row for row in refinement['tracked_states']}
    absolute_ground=tracked[ground]['energy_cm']
    origin_change=absolute_ground-(reference.zero_point_cm-np.sum(reference.frequencies_cm)/2)
    energies=np.zeros(len(states))
    diagonal=np.zeros(len(states))
    energies[:len(reference.states)]=reference.energies_cm-origin_change
    diagonal[:len(reference.states)]=reference.deperturbed_cm-origin_change
    replaced_states={tuple(s) for model in refinement['models'] for s in model['states']}
    blocks=[dict(block,eigenvalues_cm=np.asarray(block['eigenvalues_cm'])-origin_change)
            for block in reference.blocks if not any(reference.states[i] in replaced_states for i in block['indices'])]
    for model in refinement['models']:
        local_states=tuple(map(tuple,model['states']))
        local_indices={s:i for i,s in enumerate(local_states)}
        vectors=np.asarray(model['eigenvectors'],complex)
        roots=np.asarray(model['eigenvalues_cm'])-absolute_ground
        assigned=np.full(len(local_states),-1,int)
        for target,root in zip(model['targets'],model['target_roots']):
            assigned[local_indices[tuple(target)]]=root
        remaining=np.flatnonzero(assigned<0)
        available=np.array(sorted(set(range(len(assigned)))-set(assigned[assigned>=0])))
        if len(remaining):
            rr,cc=linear_sum_assignment(-abs(vectors[np.ix_(remaining,available)])**2)
            assigned[remaining[rr]]=available[cc]
        members=tuple(indices[s] for s in local_states)
        matrix=np.asarray(model['hamiltonian_cm'],complex)
        energies[list(members)]=roots[assigned]
        diagonal[list(members)]=matrix.diagonal().real-absolute_ground
        if len(members)>1:
            blocks.append(dict(indices=members,hamiltonian_cm=matrix,eigenvalues_cm=roots,
                eigenvectors=vectors,basis_to_root=assigned,
                direct_resonance_states=tuple((left,right) for i,left in enumerate(local_states)
                                             for j,right in enumerate(local_states[i+1:],i+1) if abs(matrix[i,j])>1e-12)))
    if abs(energies[indices[ground]])>1e-8:
        raise RuntimeError('targeted polyad ground origin mismatch')
    # Keep the public resonance ledger consistent with the actual matrices.
    retained={}
    for block in blocks:
        if not any(states[i] in replaced_states for i in block['indices']):
            continue
        matrix=np.asarray(block['hamiltonian_cm'])
        for position,i in enumerate(block['indices']):
            for other,j in enumerate(block['indices'][position+1:],position+1):
                value=matrix[position,other]
                if abs(value)>1e-12:
                    retained[(min(i,j),max(i,j))]=float(value.real)
    candidates=[]
    seen=set()
    closed_pairs={tuple(sorted((i,j))) for block in blocks
        if any(states[i] in replaced_states for i in block['indices'])
        for position,i in enumerate(block['indices']) for j in block['indices'][position+1:]}
    for item in reference.resonances:
        item=dict(item)
        key=tuple(sorted((int(item['i']),int(item['j']))))
        if key in retained:
            item.update(initial_decision=item['decision'],initial_coupling_cm=item['coupling_cm'],
                        coupling_cm=retained[key],decision='internal closure')
            seen.add(key)
        elif key in closed_pairs:
            item.update(initial_decision=item['decision'],coupling_cm=0.,decision='zero in closed block')
        candidates.append(item)
    for (i,j),value in sorted(retained.items()):
        if (i,j) in seen:
            continue
        candidates.append(dict(i=i,j=j,kind=_resonance_kind(states[i],states[j]) or 'internal',
            gap_cm=float(abs(np.dot(reference.frequencies_cm,np.subtract(states[i],states[j])))),
            coupling_cm=value,decision='internal closure'))
    diagnostics=dict(reference.diagnostics,effective_operator_profile=profile,
        targeted_polyad_refinement={k:v for k,v in refinement.items() if k not in ('models','tracked_states')},
        targeted_polyad_operator_coverage=dict(internal_quartics=field.quartic_coverage,
            reduced_polynomial_explicit=allow_reduced_quartic_model,
            external_selection='fixed GVPT2 Fermi and DD operator edges',
            operator_profile=profile,completion='all available DD edges in the declared window'),
        initial_candidate_pair_count=reference.diagnostics.get('candidate_pair_count',0),
        initial_accepted_pair_count=reference.diagnostics.get('accepted_pair_count',0),
        candidate_pair_count=len(candidates),
        accepted_pair_count=sum(item['decision'] in ('accepted','internal closure') for item in candidates),
        resonant_block_count=len(blocks),largest_polyad_dimension=max(map(lambda b:len(b['indices']),blocks),default=1))
    if diagnostics.get('adaptive_observable_tolerance') is not None:
        diagnostics['initial_reference_adaptive_observable_tolerance']=diagnostics['adaptive_observable_tolerance']
        diagnostics['adaptive_observable_tolerance']=None
    report=_gvpt2_report(states,reference.frequencies_cm,diagonal,energies,candidates,blocks,diagnostics)
    report+='\nTarget-directed fixed GVPT2 polyad closure: '+str(refinement['converged'])+' ('+refinement['reason']+')\n'
    result=replace(reference,states=states,deperturbed_cm=diagonal,energies_cm=energies,
        fundamentals_cm=np.array([energies[indices[tuple(int(i==j) for i in range(n))]] for j in range(n)]),
        zero_point_cm=float(absolute_ground+np.sum(reference.frequencies_cm)/2),
        blocks=tuple(blocks),resonances=tuple(candidates),diagnostics=diagnostics,report=report)
    return (result,refinement) if return_state else result
