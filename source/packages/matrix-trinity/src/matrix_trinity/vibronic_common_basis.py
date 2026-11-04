"""FGR in one orthonormal reference basis: coupled, independent, harmonic blocks.

Reuse positive discrete convolution for the separable rate. The coordinate and
kinetic owners provide states in the SAME chart and measure. No Duschinsky map
is inferred from separately computed normal-mode states.
"""
from dataclasses import dataclass
import numpy as np

from .vibronic_variational import FiniteVibrationalStates
from .vibronic_rates import GOLDEN_RULE_FACTOR


@dataclass(frozen=True)
class CommonCoordinateBlock:
    modes: tuple
    initial: FiniteVibrationalStates
    final: FiniteVibrationalStates
    reference_id: str
    tier: str

    def __post_init__(self):
        modes=tuple(self.modes)
        if (not modes or len(set(modes))!=len(modes)
                or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or i<0 for i in modes)
                or not isinstance(self.initial,FiniteVibrationalStates)
                or not isinstance(self.final,FiniteVibrationalStates)
                or len(self.initial.frequencies_cm)!=len(modes)
                or not np.array_equal(self.initial.frequencies_cm,self.final.frequencies_cm)
                or not isinstance(self.reference_id,str) or not self.reference_id
                or self.tier not in ('coupled','diagonal','harmonic')
                or (self.tier!='coupled' and len(modes)!=1)):
            raise ValueError('matching common orthonormal bases, identified modes and a valid tier required')
        object.__setattr__(self,'modes',modes)

    def overlaps(self,initial_levels,final_levels):
        """Contract only requested columns on exact common HO row intersections."""
        left=np.asarray(initial_levels);right=np.asarray(final_levels)
        for ids,state in ((left,self.initial),(right,self.final)):
            if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0)
                    or np.any(ids>=len(state.energies_cm))):
                raise ValueError('valid integer initial/final eigenlevel labels required')
        lookup={tuple(row):i for i,row in enumerate(self.initial.basis)}
        source=np.zeros((len(self.final.basis),len(left)),complex)
        for row,word in enumerate(self.final.basis):
            position=lookup.get(tuple(word))
            if position is not None:source[row]=self.initial.eigenvectors[position,left]
        final=self.final.eigenvectors[:,right]
        return (final.T@source.conj()).conj()


def _partition(blocks):
    blocks=tuple(blocks)
    if not blocks or any(not isinstance(b,CommonCoordinateBlock) for b in blocks):
        raise TypeError('nonempty common-coordinate block sequence required')
    if len({b.reference_id for b in blocks})!=1:
        raise ValueError('all blocks must use the same identified chart and orthonormal measure')
    modes=[m for b in blocks for m in b.modes]
    if len(set(modes))!=len(modes) or sorted(modes)!=list(range(len(modes))):
        raise ValueError('coupled/diagonal/harmonic blocks must partition every mode exactly once')
    return blocks


def common_product_overlaps(blocks,initial_configurations,final_configurations):
    """Selected product-state overlaps, including complete 1D wavefunctions."""
    blocks=_partition(blocks);initial=np.asarray(initial_configurations);final=np.asarray(final_configurations)
    if any(a.ndim!=2 or a.shape[1]!=len(blocks) or a.dtype.kind not in 'iu' or np.any(a<0)
           for a in (initial,final)):
        raise ValueError('one nonnegative eigenlevel index per block required')
    overlap=np.ones((len(final),len(initial)),complex)
    for column,block in enumerate(blocks):
        il,ii=np.unique(initial[:,column],return_inverse=True)
        fl,fi=np.unique(final[:,column],return_inverse=True)
        overlap*=block.overlaps(il,fl)[fi[:,None],ii[None,:]]
    return overlap


def factorized_common_coordinate_rate(blocks,*,initial_levels,electronic_gap_cm,
        coupling_hartree,gaussian_sigma_cm,bin_width_cm,tail_tolerance_s_1=1.,
        maximum_workspace_bytes=2*1024**3):
    """Finite-block Condon FGR by existing positive discrete convolution.

    Electronic gap is E(final reference)-E(initial reference). Each block's
    absolute energies include its own potential-origin convention and ZPE.
    Linear deposition moves each line by at most one grid step per block.
    Gaussian interval envelopes bound that transport error a posteriori.
    Positive high-energy truncation is bounded separately. Bounds exclude
    floating-point roundoff, eigenstate errors and states outside input blocks.
    """
    from matrix_rovib.statistical_convolution import convolve_discrete_measure
    blocks=_partition(blocks);levels=np.asarray(initial_levels);coupling=np.asarray(coupling_hartree,complex)
    sigma,delta=float(gaussian_sigma_cm),float(bin_width_cm)
    if (levels.shape!=(len(blocks),) or levels.dtype.kind not in 'iu' or np.any(levels<0)
            or coupling.ndim!=1 or not len(coupling) or not np.all(np.isfinite(coupling))
            or not all(np.isfinite(x) for x in (electronic_gap_cm,sigma,delta,tail_tolerance_s_1))
            or min(sigma,delta,tail_tolerance_s_1)<=0
            or isinstance(maximum_workspace_bytes,bool) or not isinstance(maximum_workspace_bytes,int)
            or maximum_workspace_bytes<1):
        raise ValueError('finite initial levels, gap, coupling and positive grid/error/workspace controls required')
    lines=[];initial_energy=0.;ground=0.;strength=1.
    for block,level in zip(blocks,levels):
        overlap=block.overlaps(np.array([level]),np.arange(len(block.final.energies_cm)))[:,0]
        weights=abs(overlap)**2;energies=block.final.energies_cm-block.final.energies_cm[0]
        lines.append((energies,weights));strength*=float(weights.sum())
        initial_energy+=block.initial.energies_cm[level];ground+=block.final.energies_cm[0]
    factor=GOLDEN_RULE_FACTOR*float(np.sum(abs(coupling)**2));peak=1/(sigma*np.sqrt(2*np.pi))
    target=float(initial_energy-ground-electronic_gap_cm);transport=len(blocks)*delta
    radius=sigma*np.sqrt(2*max(0.,np.log(max(factor*strength*peak/tail_tolerance_s_1,1.))))
    cutoff=max(0.,target+radius+transport);count=int(np.floor(cutoff/delta))+1
    required=8*(10*count+sum(len(e) for e,w in lines))
    if count>=9_999_998 or required>maximum_workspace_bytes:
        raise MemoryError('positive common-coordinate convolution exceeds grid/workspace budget')
    if factor==0 or strength==0:
        return dict(rate_s_1=0.,lower_rate_s_1=0.,upper_rate_s_1=0.,absolute_error_bound_s_1=0.,
                    diagnostics=dict(zero_coupling_or_finite_source_strength=True))
    # Deposit the largest discrete block first, while the input measure has one bin.
    lines.sort(key=lambda pair:len(pair[0]),reverse=True)
    measure=np.ones(1)
    for energies,weights in lines:
        take=(energies<=cutoff)&(weights>0)
        if not np.any(take):measure=np.zeros(1);break
        measure=convolve_discrete_measure(measure,delta,energies[take],weights[take],algorithm='positive_direct')[:count]
    distance=abs(np.arange(len(measure))*delta-target)
    value=factor*float(measure@(peak*np.exp(-.5*(distance/sigma)**2)))
    lower=factor*float(measure@(peak*np.exp(-.5*((distance+transport)/sigma)**2)))
    upper=factor*float(measure@(peak*np.exp(-.5*(np.maximum(distance-transport,0)/sigma)**2)))
    tail=factor*strength*peak*np.exp(-.5*(radius/sigma)**2)
    upper+=tail
    return dict(schema='matrix.trinity.common_coordinate_factorized_rate.v1',rate_s_1=value,
        lower_rate_s_1=lower,upper_rate_s_1=upper,absolute_error_bound_s_1=max(value-lower,upper-value),
        diagnostics=dict(reference_id=blocks[0].reference_id,tiers=[b.tier for b in blocks],
            mode_groups=[list(b.modes) for b in blocks],block_count=len(blocks),
            initial_energy_cm=float(initial_energy),final_ground_energy_cm=float(ground),
            available_excitation_cm=target,finite_source_strength=strength,
            maximum_line_transport_cm=transport,bin_width_cm=delta,retained_bins=len(measure),
            energy_cutoff_cm=cutoff,high_energy_rate_bound_s_1=tail,
            additional_workspace_estimate_bytes=required,statistical_sampling_used=False,
            full_product_basis_constructed=False,duschinsky_rotation_used=False,
            anharmonic_overlap_wavefunctions_included=True,residual_couplings_included=False,
            bounds_include_roundoff=False,full_molecular_convergence_certified=False))


def selected_quadratic_dressing(model,*,maximum_states=512):
    """Normalized external-K wavefunctions through order two, with exact polyads.

    Existing local_quadratic owns polyads and their symmetric K^2 self-energy.
    This adapter supplies the compatible anti-Hermitian external transformation
    for overlaps. Diagonal first-order contributions belong in the supplied
    reference energies. Induced second-order interblock resonances are rejected.
    """
    from .local_quadratic import LocalQuadraticModel
    if not isinstance(model,LocalQuadraticModel):raise TypeError('existing LocalQuadraticModel required')
    n=len(model.reference_energies_cm)
    if n>maximum_states:raise MemoryError('selected quadratic overlap block exceeds state budget')
    reference=np.column_stack([model.reference_vector(i,variational=True) for i in range(n)])
    external=np.zeros((n,n))
    for i,j,value in model.external_k_edges:external[i,j]=external[j,i]=value
    external=reference.T@external@reference
    labels=np.arange(n)
    for group,members in enumerate(model.polyads):labels[list(members)]=n+group
    outside=labels[:,None]!=labels[None,:]
    gaps=model.variational_energies_cm[None,:]-model.variational_energies_cm[:,None]
    unsafe=outside&(abs(gaps)<=model.denominator_tolerance_cm)
    if np.any(abs(external[unsafe])>model.numerical_zero_cm):
        raise ValueError('unresolved external quadratic resonance; merge the selected polyads')
    safe=outside&~unsafe;s1=np.zeros((n,n));s1[safe]=external[safe]/gaps[safe]
    commutator=.5*(external@s1-s1@external)
    if np.any(abs(commutator[unsafe])>model.numerical_zero_cm):
        raise ValueError('induced second-order resonance; enlarge the variational polyad')
    s2=np.zeros((n,n));s2[safe]=commutator[safe]/gaps[safe]
    operative=model.operative_spectrum();rotation=reference.copy()
    for members,vectors in zip(model.polyads,operative.eigenvectors):rotation[np.ix_(members,members)]=vectors
    local_rotation=reference.T@rotation
    vectors=(rotation,reference@s1@local_rotation,reference@(s2+.5*s1@s1)@local_rotation)
    return dict(energies_cm=operative.energies_cm,wavefunction_orders=vectors,
        diagnostics=dict(exact_intra_polyad=True,external_quadratic_order=2,
            eigenvectors_normalized_through_second_order=True,first_order_diagonal_terms_owned_by_reference=True,
            selected_basis_size=n,full_product_space_convergence_certified=False))


def quadratic_overlap_orders(initial_model,final_model,reference_overlaps,*,maximum_states=512):
    """Common-basis overlap amplitudes through external quadratic order two."""
    initial=selected_quadratic_dressing(initial_model,maximum_states=maximum_states)
    final=selected_quadratic_dressing(final_model,maximum_states=maximum_states)
    overlap=np.asarray(reference_overlaps,complex)
    if (overlap.shape!=(len(final['energies_cm']),len(initial['energies_cm']))
            or not np.all(np.isfinite(overlap))):raise ValueError('matching finite reference overlaps required')
    left,right=final['wavefunction_orders'],initial['wavefunction_orders']
    orders=tuple(sum((left[i].conj().T@overlap@right[k-i] for i in range(k+1)),
                     start=np.zeros_like(overlap)) for k in range(3))
    return dict(initial_energies_cm=initial['energies_cm'],final_energies_cm=final['energies_cm'],
        overlap_amplitude_orders=orders,diagnostics=dict(external_quadratic_order=2,
            intra_polyad_mixing_included=True,unresolved_induced_resonances_rejected=True))
