"""Mode blocks of the full perturbative effective Hamiltonian.

The vacuum of omitted blocks is contracted before discarding connected
interblock terms. This differs physically from masking the raw Taylor field:
nonresonant virtual paths through every mode dress each local Hamiltonian.
No molecular constants or fitted shifts enter this construction.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .curvilinear_vpt2 import SonicGVPT2Result, _modalize_sonic_field, _maximum_overlap_assignment, _candidate_state_pairs
from .gvpt2_operator import GVPT2EffectiveOperator
from .mode_blocks import VibrationalModePartition, _column, _integer


@dataclass(frozen=True)
class VacuumDressedModeBlocks:
    partition: VibrationalModePartition
    results: tuple
    coordinate_id: str
    zero_point_cm: float
    diagnostics: dict

    def configurations(self, values):
        values = np.asarray(values)
        if (values.ndim != 2 or values.shape[1] != len(self.results) or values.dtype.kind not in 'iu'
                or np.any(values < 0) or any(np.any(values[:,b] >= len(r.states)) for b,r in enumerate(self.results))):
            raise ValueError('valid local assigned-level indices required')
        return values

    def energies(self, values):
        values = self.configurations(values)
        return sum((r.energies_cm[values[:,b]] for b,r in enumerate(self.results)),np.zeros(len(values)))

    def bare_column(self, configuration, *, maximum_terms=100000):
        maximum_terms = _integer(maximum_terms,'maximum_terms',1)
        configuration = self.configurations(np.asarray(configuration)[None])[0]
        local = [_column(r,int(i)) for r,i in zip(self.results,configuration)]
        count = int(np.prod([len(ids) for ids,_ in local],dtype=object))
        if count > maximum_terms:
            raise MemoryError('product-column support budget exceeded')
        occupations = np.zeros((count,self.partition.mode_count),int)
        coefficients = np.ones(count,complex)
        for row, positions in enumerate(product(*(range(len(ids)) for ids,_ in local))):
            for block,result,(ids,weights),position in zip(self.partition.blocks,self.results,local,positions):
                occupations[row,list(block)] = result.states[ids[position]]
                coefficients[row] *= weights[position]
        return occupations, coefficients

    def moments(self, provider, initial, final, *, coordinate_id, maximum_bare_pairs=100000):
        """Apply a full effective property, including its mixed-block terms.

        The provider must use the same full-field perturbative transformation
        and modal frame. No raw electrical or mechanical derivative is masked.
        Only selected product columns are expanded; no product basis is built.
        """
        if coordinate_id != self.coordinate_id:
            raise ValueError('property and mechanics must share their modal coordinate_id')
        initial,final = self.configurations(initial),self.configurations(final)
        if initial.shape != final.shape:
            raise ValueError('paired configurations required')
        maximum_bare_pairs = _integer(maximum_bare_pairs,'maximum_bare_pairs',1)
        output=[]
        for n,f in zip(initial,final):
            ni,ci=self.bare_column(n,maximum_terms=maximum_bare_pairs)
            nf,cf=self.bare_column(f,maximum_terms=maximum_bare_pairs)
            if len(ni)*len(nf)>maximum_bare_pairs:
                raise MemoryError('coherent product-pair budget exceeded')
            left=np.repeat(ni,len(nf),axis=0);right=np.tile(nf,(len(ni),1))
            # Native cold remainders use the less excited endpoint as source.
            reverse=np.array([(sum(b),tuple(b))<(sum(a),tuple(a)) for a,b in zip(left,right)])
            source,target=left.copy(),right.copy()
            source[reverse],target[reverse]=right[reverse],left[reverse]
            values=provider.moments(source,target)
            values[reverse]=values[reverse].conj()
            weights=np.repeat(ci,len(nf))*np.tile(cf.conj(),len(ni))
            output.append(np.einsum('p,prc->rc',weights,values))
        return np.asarray(output)


def solve_vacuum_dressed_mode_blocks(field, partition, *, coordinate_id,
        candidate_states_per_block, max_states_per_block=30000, solver_options=None,
        maximum_matrix_bytes=256*1024**2, maximum_cached_states=128, workspace=None):
    """Contract the complete GVPT2 operator over external harmonic vacua.

    H_B(n,m)=<n,0_external|H_eff|m,0_external>-E_vac delta_nm.
    The approximation is H_eff ~= E_vac + sum_B H_B. Connected occupation-
    dependent interblock energies and resonance edges must be audited/merged
    on the intended domain. Diagonal tiers discard local off-diagonal terms;
    harmonic tiers keep the dressed one-quantum gap and an analytic ladder.
    An optional workspace reuses the same field and screening profile across
    partitions; it must not be reused after mutating the field.
    This does not claim the same model as raw ``solve_mode_blocks`` projection.
    """
    if not isinstance(partition,VibrationalModePartition):
        raise TypeError('VibrationalModePartition required')
    if not isinstance(coordinate_id,str) or not coordinate_id.strip():
        raise ValueError('identified common modal coordinates required')
    max_states_per_block=_integer(max_states_per_block,'max_states_per_block',2)
    maximum_matrix_bytes=_integer(maximum_matrix_bytes,'maximum_matrix_bytes',1)
    maximum_cached_states=_integer(maximum_cached_states,'maximum_cached_states',1)
    bases=tuple(tuple(map(tuple,basis)) for basis in candidate_states_per_block)
    if len(bases)!=len(partition.blocks):
        raise ValueError('one candidate basis per mechanical block required')
    workspace = {} if workspace is None else workspace
    if workspace.get("mode_blocks_field", field) is not field:
        raise ValueError("mode-block workspace belongs to a different field")
    workspace["mode_blocks_field"] = field
    if "modalized" not in workspace:
        workspace["modalized"] = _modalize_sonic_field(field)
    modal=workspace["modalized"]
    if len(modal[0])!=partition.mode_count:
        raise ValueError('partition and field dimensions differ')
    if field.symmetry_actions is not None:
        from matrix_gf.harmonic import CM_PER_HARTREE
        from matrix_qm import HESSIAN_EIGENVALUE_TO_CM
        gamma=np.sqrt(modal[0]*CM_PER_HARTREE)/HESSIAN_EIGENVALUE_TO_CM
        displacement=modal[1]/gamma[None]
        actions=tuple(np.linalg.solve(displacement,a@displacement) for a in field.symmetry_actions)
        partition.validate_symmetry(actions,tolerance=field.symmetry_tolerance)
    op=GVPT2EffectiveOperator(field,workspace=workspace,**dict(solver_options or {}))
    if op.policy not in {'gdv','none','fermi','fermi_auto'}:
        raise ValueError('effective blocks currently require GDV or Fermi/none screening')
    vacuum=(0,)*partition.mode_count
    origin=op.diagonal(vacuum)
    results=[]
    candidate_pair_count=0
    for block,tier,basis in zip(partition.blocks,partition.tiers,bases):
        n=len(block)
        required={(0,)*n,*map(tuple,np.eye(n,dtype=int))}
        if (len(basis)>max_states_per_block or len(set(basis))!=len(basis) or not required.issubset(basis)
                or any(len(s)!=n or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<0 for v in s) for s in basis)):
            raise ValueError('invalid or incomplete local effective-operator basis')
        if len(basis)**2*16>maximum_matrix_bytes:
            raise MemoryError('local effective Hamiltonian exceeds matrix-storage budget')
        embedded=[]
        for state in basis:
            value=np.zeros(partition.mode_count,int);value[list(block)]=state
            embedded.append(tuple(value))
        for state in embedded:
            op.cubic(state)
            while len(op.cubic_cache)>maximum_cached_states:
                del op.cubic_cache[next(iter(op.cubic_cache))]
        diagonal=np.array([op.diagonal(s)-origin for s in embedded])
        if tier=='harmonic':
            if n!=1:
                raise ValueError('harmonic tier must be a singleton')
            gap=diagonal[basis.index((1,))]
            diagonal=np.array([s[0]*gap for s in basis])
        h=np.diag(diagonal.astype(complex))
        if tier=='coupled':
            kinds = frozenset({'1-2'} if op.policy in {'none','fermi','fermi_auto'}
                              else {'1-1','1-2','2-2','1-3'})
            for low,high,kind in _candidate_state_pairs(basis,max(map(sum,basis)),kinds=kinds):
                i,j=high,low
                candidate_pair_count+=1
                h[i,j]=op.coupling(embedded[i],embedded[j])
                if h[i,j] and kind in {'1-1','2-2','1-3'}:
                    screening=abs(h[i,j])*(np.sqrt(8) if kind=='2-2' and ((2 in embedded[i])!=(2 in embedded[j])) else 1.)
                    mixing=abs(h[i,j])/max(abs(diagonal[i]-diagonal[j]),op.denominator_tolerance)
                    if not (screening>=10. or kind=='1-1' and mixing>=.3):
                        h[i,j]=0.
                h[j,i]=h[i,j].conjugate()
        count,labels=connected_components(csr_matrix(abs(h-np.diag(np.diag(h)))>0),directed=False)
        energies=diagonal.copy();blocks=[]
        for label in range(count):
            indices=np.flatnonzero(labels==label)
            if len(indices)==1:continue
            # The complete solver closes weak 1:1 contacts between fundamental
            # roots already joined by accepted resonances. Keep the same rule.
            for offset,i in enumerate(indices):
                for j in indices[offset+1:]:
                    a,b=embedded[i],embedded[j]
                    if (h[i,j] or sum(a)!=1 or sum(b)!=1
                            or abs(op.harmonic(a)-op.harmonic(b))>op.dd_window):
                        continue
                    coupling=op.dd_coupling(a,b,'1-1')
                    if abs(coupling)>=op.zero:
                        h[i,j]=coupling;h[j,i]=np.conjugate(coupling)
            local=h[np.ix_(indices,indices)]
            eigenvalues,eigenvectors=np.linalg.eigh(local)
            assignment=np.asarray(_maximum_overlap_assignment(abs(eigenvectors)**2),int)
            energies[indices]=eigenvalues[assignment]
            blocks.append(dict(indices=indices,hamiltonian_cm=local,eigenvalues_cm=eigenvalues,
                eigenvectors=eigenvectors,basis_to_root=assignment))
        units=[basis.index(tuple(s)) for s in np.eye(n,dtype=int)]
        results.append(SonicGVPT2Result(modal[0][list(block)],basis,diagonal,energies,
            energies[units],(),tuple(blocks),0.,'Full-field vacuum-dressed local GVPT2 operator'))
    diagnostics=dict(operator_projection='Full effective Hamiltonian first; external harmonic vacuum contraction second',
        candidate_pair_count=candidate_pair_count,
        full_virtual_modes_retained=partition.mode_count,raw_derivative_masking=False,
        omitted_terms='Connected interblock terms of the effective Hamiltonian',
        local_state_counts=[len(r.states) for r in results],full_product_basis_constructed=False,
        property_policy='Full effective property required, including mixed mechanical and electrical terms')
    return VacuumDressedModeBlocks(partition,tuple(results),coordinate_id,float(np.sum(modal[0])/2+origin),diagnostics)
