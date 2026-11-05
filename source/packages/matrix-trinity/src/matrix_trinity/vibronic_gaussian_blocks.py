"""Zero-temperature harmonic TI factored into final-mode Gaussian blocks.

The block line lists retain state labels. Their products are summed with the
existing discrete-measure convolution, without enumerating every full state.
Small discarded interblock elements and property leakage are explicit inputs;
comparison with the original analytic TD is required before molecular use.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components
from .vibronic_harmonic import (HarmonicVibronicModel, harmonic_gaussian_generator,
    orient_harmonic_model,harmonic_strength_sum_rule)


def zero_temperature_gaussian_blocks(model, *, kind='absorption', matrix_tolerance=0.,
                                     operator_relative_tolerance=0., channel_weights=None):
    """Prepare pure Gaussian blocks with identical final-mode wavefunctions.

    This construction applies to the initial vibrational ground state only.
    Each incoherent operator channel must belong to one block; small property
    components may be discarded only under the explicit relative tolerance.
    Same-channel interference within the retained block is exact. General
    cross-block coherent operators are rejected, not treated as independent.
    """
    if not np.isfinite(matrix_tolerance+operator_relative_tolerance) or min(matrix_tolerance,operator_relative_tolerance)<0:
        raise ValueError('Finite nonnegative factorization tolerances required')
    oriented,direction=orient_harmonic_model(model,kind)
    g=harmonic_gaussian_generator(oriented);n=len(oriented.final_frequencies_cm)
    a=g['matrix'][n:,n:];b=g['linear'][n:]
    weights=np.ones(3) if channel_weights is None else np.asarray(channel_weights,float)
    if weights.shape!=(3,) or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError('Three explicit nonnegative channel populations required')
    h=g['ht_linear'][:,n:]*np.sqrt(weights[:,None])
    c=(oriented.transition_dipole_au+g['ht_constant'])*np.sqrt(weights)
    count,labels=connected_components(abs(a)>matrix_tolerance,directed=False)
    groups=[np.flatnonzero(labels==k) for k in range(count)]
    fc=[];reconstruction=[]
    for ids in groups:
        aa=a[np.ix_(ids,ids)];bb=b[ids];eye=np.eye(len(ids))
        precision=np.linalg.solve(eye+aa,eye-aa);precision=(precision+precision.T)/2
        eigen,vectors=np.linalg.eigh(precision)
        if np.any(eigen<=0):raise ValueError('Positive Gaussian block precision required')
        inverse_root=(vectors/np.sqrt(eigen))@vectors.T
        w=oriented.final_frequencies_cm[ids];root=np.sqrt(w)
        mapping=inverse_root*root[None,:]/root[:,None]
        shift=np.linalg.solve(precision,(eye+precision)@bb)/np.sqrt(2)
        block=HarmonicVibronicModel(w,w,mapping,shift,0.,[1.,0.,0.],np.zeros((3,len(ids))))
        check=harmonic_gaussian_generator(block)
        error=max(float(np.max(abs(check['matrix'][len(ids):,len(ids):]-aa),initial=0)),
            float(np.max(abs(check['linear'][len(ids):]-bb),initial=0)))
        if error>1e-9:raise FloatingPointError('Gaussian block reconstruction failed')
        fc.append(block);reconstruction.append(error)
    operators=[];assignments=[];leakage=[]
    for channel in range(3):
        norms=np.array([np.linalg.norm(h[channel,ids]) for ids in groups]);chosen=int(np.argmax(norms))
        outside=h[channel].copy();outside[groups[chosen]]=0
        leak=float(np.linalg.norm(outside));scale=max(float(np.linalg.norm(h[channel])),float(abs(c[channel])))
        if leak>operator_relative_tolerance*scale:
            raise ValueError('Coherent operator spans several Gaussian blocks; a one-block factorization is insufficient')
        ids=groups[chosen];base=fc[chosen];k=len(ids)
        jj=base.mode_map;ww=base.initial_frequencies_cm
        precision=np.diag(ww)+(jj.T*ww)@jj
        response=np.linalg.solve(precision,np.sqrt(2)*jj.T*np.sqrt(ww))
        insertion=np.sqrt(ww)[:,None]*response
        gradient=np.zeros((3,k),complex);gradient[channel]=np.linalg.solve(insertion.T,h[channel,ids])
        temporary=HarmonicVibronicModel(ww,ww,jj,base.final_shift_dimensionless,0.,np.zeros(3),gradient)
        ht0=harmonic_gaussian_generator(temporary)['ht_constant'];constant=-ht0;constant[channel]+=c[channel]
        operators.append(HarmonicVibronicModel(ww,ww,jj,base.final_shift_dimensionless,0.,constant,gradient))
        assignments.append(chosen);leakage.append(leak)
    return dict(fc_models=fc,operator_models=operators,channel_blocks=assignments,
        full_channel_strengths=[harmonic_strength_sum_rule(v,0.)['total'] for v in operators],
        final_mode_groups=groups,origin_cm=float(direction*oriented.origin_cm),direction=direction,
        diagnostics=dict(matrix_tolerance=matrix_tolerance,
            discarded_interblock_matrix_norm=float(np.linalg.norm(a*(labels[:,None]!=labels[None,:]))),
            operator_relative_tolerance=operator_relative_tolerance,discarded_channel_insertion_norm=leakage,
            block_reconstruction_errors=reconstruction,
            constant_and_linear_interference_retained=True,temperature_K=0.,
            original_model_TD_comparison_required=True))


def convolve_gaussian_block_lines(prepared, fc_lines, operator_lines, *, gaussian_sigma_cm,
                                  bin_width_cm, profile_axis_cm, excitation_cutoff_cm):
    """Sum all finite block products and bound deposited-line energy transport.

    Input line tables contain excitation energies and unrenormalized strengths.
    No product-state list is built. FC and operator truncations remain explicit;
    an independent basis/cutoff check and original-model TD comparison are needed.
    FFT roundoff is excluded from the analytic transport envelope.
    """
    from matrix_rovib.statistical_convolution import convolve_discrete_measure
    from .vibronic_rates import GOLDEN_RULE_FACTOR
    sigma,delta,cutoff=map(float,(gaussian_sigma_cm,bin_width_cm,excitation_cutoff_cm))
    axis=np.asarray(profile_axis_cm,float)
    if (min(sigma,delta,cutoff)<=0 or not np.isfinite(sigma+delta+cutoff)
            or axis.ndim!=1 or len(axis)<2 or np.any(np.diff(axis)<=0)
            or not axis[0]<=0<=axis[-1] or not np.all(np.isfinite(axis))):
        raise ValueError('Positive finite widths/cutoff and finite target axis required')
    count=int(np.ceil(cutoff/delta))+1
    if count>=9999998:raise MemoryError('Block convolution grid exceeds owner budget')
    if len(fc_lines)!=len(prepared['fc_models']) or len(operator_lines)!=3:
        raise ValueError('Complete FC block lists and three operator lists required')
    for energies,weights in [*fc_lines,*operator_lines]:
        e=np.asarray(energies);w=np.asarray(weights)
        if (e.ndim!=1 or w.shape!=e.shape or not np.all(np.isfinite(e))
                or not np.all(np.isfinite(w)) or np.any(e< -1e-9) or np.any(w<0)):
            raise ValueError('Nonnegative finite block excitations and strengths required')
    profile=np.zeros(len(axis));lower=profile.copy();upper=profile.copy();channels=[];finite_strength=[]
    omitted_strength=0.;tail=np.zeros(len(axis))
    transport=len(fc_lines)*delta;peak=1/(sigma*np.sqrt(2*np.pi))
    for channel,active in enumerate(prepared['channel_blocks']):
        lists=[operator_lines[channel] if j==active else lines for j,lines in enumerate(fc_lines)]
        source_strength=float(np.prod([np.sum(w) for e,w in lists]))
        full_strength=prepared.get('full_channel_strengths',[None]*3)[channel]
        if full_strength is not None:omitted_strength+=max(0.,full_strength-source_strength)
        tail+=source_strength*peak*np.exp(-.5*(np.maximum(0.,cutoff-prepared['direction']*(axis-prepared['origin_cm']))/sigma)**2)
        measure=np.ones(1)
        for energies,weights in sorted(lists,key=lambda pair:len(pair[0]),reverse=True):
            energies=np.asarray(energies);weights=np.asarray(weights)
            take=(energies<=cutoff)&(weights>0)
            if not np.any(take):measure=np.zeros(1);break
            measure=convolve_discrete_measure(measure,delta,energies[take],weights[take],algorithm='fft')[:count]
        energy=prepared['origin_cm']+prepared['direction']*np.arange(len(measure))*delta
        values=np.zeros(len(axis));lo=values.copy();hi=values.copy()
        for start in range(0,len(axis),8):
            distance=abs(energy[:,None]-axis[None,start:start+8])
            values[start:start+8]=measure@(peak*np.exp(-.5*(distance/sigma)**2))
            lo[start:start+8]=measure@(peak*np.exp(-.5*((distance+transport)/sigma)**2))
            hi[start:start+8]=measure@(peak*np.exp(-.5*(np.maximum(distance-transport,0)/sigma)**2))
        profile+=values;lower+=lo;upper+=hi;channels.append(values);finite_strength.append(source_strength)
    return dict(profile_axis_cm=axis,density=profile,lower_transport_density=lower,
        upper_transport_density=upper,channel_densities=np.asarray(channels),
        upper_density_including_basis_and_cutoff=upper+peak*omitted_strength+tail,
        basis_omission_density_upper_bound=float(peak*omitted_strength),high_energy_density_upper_bound=tail,
        zero_energy_rate_s_1=float(GOLDEN_RULE_FACTOR*np.interp(0.,axis,profile)),
        diagnostics=dict(bin_width_cm=delta,maximum_line_transport_cm=transport,
            finite_channel_strength=finite_strength,excitation_cutoff_cm=cutoff,
            full_block_source_norms_available='full_channel_strengths' in prepared,
            product_states_explicitly_enumerated=False,FFT_roundoff_included_in_bounds=False,
            additional_basis_and_TD_checks_required=True))


def gaussian_block_line_strengths(prepared, fc_columns, operator_columns, final_quanta):
    """Recover unbinned strengths for explicit full-mode state assignments.

    Columns are (integer final occupations, physical positive strengths).
    Missing block levels give a named unavailable channel, never a fabricated
    amplitude. Original final-mode indices are retained across the partition.
    """
    groups=prepared['final_mode_groups'];n=sum(len(v) for v in groups)
    q=np.asarray(final_quanta)
    if (q.ndim!=2 or q.shape[1]!=n or not np.issubdtype(q.dtype,np.integer) or np.any(q<0)
            or not np.array_equal(np.sort(np.concatenate(groups)),np.arange(n))
            or len(fc_columns)!=len(groups) or len(operator_columns)!=3):
        raise ValueError('Integer full-mode assignments and complete disjoint block tables required')
    def lookup(columns,ids):
        states,strengths=map(np.asarray,columns)
        if (states.ndim!=2 or states.shape[1]!=len(ids) or strengths.shape!=(len(states),)
                or not np.issubdtype(states.dtype,np.integer) or np.any(states<0)
                or np.any(states>32767) or np.any(q[:,ids]>32767)
                or not np.all(np.isfinite(strengths)) or np.any(strengths<0)):
            raise ValueError('Finite positive block weights and bounded integer occupations required')
        dtype=np.dtype('V'+str(2*len(ids)))
        keys=np.ascontiguousarray(states,dtype='<i2').view(dtype).ravel()
        queries=np.ascontiguousarray(q[:,ids],dtype='<i2').view(dtype).ravel()
        order=np.argsort(keys);ordered=keys[order]
        if np.any(ordered[1:]==ordered[:-1]):raise ValueError('Duplicate block states are not independent transitions')
        positions=np.searchsorted(ordered,queries);found=positions<len(ordered)
        found[found]=ordered[positions[found]]==queries[found]
        values=np.zeros(len(q));values[found]=strengths[order[positions[found]]]
        return values,found
    fc=[lookup(v,ids) for v,ids in zip(fc_columns,groups)]
    channel_values=[];available=[]
    for channel,active in enumerate(prepared['channel_blocks']):
        values,found=lookup(operator_columns[channel],groups[active])
        for j,(weight,matched) in enumerate(fc):
            if j!=active:values*=weight;found&=matched
        channel_values.append(values);available.append(found)
    return dict(strengths=np.sum(channel_values,axis=0),channel_strengths=np.asarray(channel_values),
        channel_states_available=np.asarray(available),final_quanta=q,
        final_mode_indices_preserved=True,profile_binning_used=False)
