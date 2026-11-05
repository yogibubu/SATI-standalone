"""Rate targeting, shared column reductions and coherent SOC selection."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import voigt_profile
from matrix_trinity.soc_linear import coherent_soc_pairs, rank_soc_promoting_modes
from matrix_trinity.vibronic_targeted import SpectralSelectionTarget
from matrix_trinity import vibronic_rates
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel


def target(point=1000.):
    axis=np.linspace(-2000.,4000.,3001)
    y=voigt_profile(axis-1000.,50.,0.)
    return SpectralSelectionTarget([[500.,1500.]],axis,y[None],y,
        gaussian_sigma_cm=50.,rate_at_cm=point)


def test_rate_point_stopping_and_exact_density():
    t=target();assert not t.converged(.99)
    t.add(np.array([1000.]),np.array([.995]),np.array([[.995]]))
    assert t.converged(.99)
    report=t.report(np.array([1000.]),np.array([[.995]]),[1.])
    assert report['rate_target']['recovery_fraction'][0]==pytest.approx(.995,abs=1e-12)
    assert report['rate_target']['rigorous_bound'] is False
    with pytest.raises(ValueError):target(2000.)
    with pytest.raises(ValueError):t.add(np.array([1000.]),np.array([1.]),np.array([[-1.]]))


def test_rate_bias_matches_gaussian_energy_shell():
    t=target();m=SimpleNamespace(initial_frequencies_cm=np.array([1000.]),
        final_frequencies_cm=np.array([1000.]),origin_cm=0.)
    g=dict(matrix=np.zeros((2,2)),linear=np.array([0.,np.sqrt(2.)]))
    bias=t.automatic_bias(m,g,0.,1)
    assert bias==pytest.approx(np.log(2)/1000.,abs=1e-6)
    assert t.rate_ranking_mean_cm==pytest.approx(1000.,abs=1.)
    m.origin_cm=500.
    t.rate_at_cm=-1500.
    bias=t._rate_bias(m,g,np.array([0.,1.]),np.array([-1000.,1000.]),-1,0.,.003)
    assert bias==pytest.approx(np.log(2)/1000.,abs=1e-6)
    assert t.rate_ranking_mean_cm==pytest.approx(-1500.,abs=1.)


def endpoint(value,phase=1.,score=.99):
    return dict(soc=np.asarray(value)/phase,spin_frame='reference',
        singlet_tracking=dict(phase=phase,score=score,source='APOC fixture'),
        triplet_tracking=dict(phase=1.,score=1.,source='APOC fixture'))


def test_coherent_same_channel_modes_and_independent_phases():
    pairs=[]
    for mode,value in enumerate(([1.,2j,0.],[3.,-1j,0.])):
        value=np.asarray(value)
        pairs.append(dict(mode_index=mode,step=.1,
            plus=endpoint(.1*value,1j),minus=endpoint(-.1*value,-1.)))
    result=coherent_soc_pairs(np.zeros(3),pairs,mode_count=2)
    np.testing.assert_allclose(result['gradient_dimensionless'],[[1.,3.],[2j,-1j],[0.,0.]])
    pairs[0]['plus']['singlet_tracking']['score']=.1
    with pytest.raises(ValueError):coherent_soc_pairs(np.zeros(3),pairs,mode_count=2)


def test_coherent_derivative_transports_both_ket_phases():
    reference=np.array([.2,.3j,0.]);derivative=np.array([1j,2.,3.])
    endpoints=[]
    for sign,sp,tp in [(1,1j,-1),(-1,-1,-1j)]:
        entry=endpoint(reference+sign*.1*derivative)
        entry['soc']=(reference+sign*.1*derivative)/(np.conj(tp)*sp)
        entry['singlet_tracking']['phase']=sp
        entry['triplet_tracking']['phase']=tp
        endpoints.append(entry)
    pair=dict(mode_index=0,step=.1,plus=endpoints[0],minus=endpoints[1])
    result=coherent_soc_pairs(reference,[pair],mode_count=1)
    np.testing.assert_allclose(result['gradient_dimensionless'][:,0],derivative)
    assert result['pairs'][0]['even_residual_hartree']<1e-14
    pair['plus']['spin_frame']='untracked'
    with pytest.raises(ValueError):coherent_soc_pairs(reference,[pair],mode_count=1)


def test_response_ranking_keeps_interference_and_rejects_indefinite_response():
    g=np.array([[1.,-.9,.001],[0.,0.,0.],[0.,0.,0.]])
    r=np.ones((3,3));result=rank_soc_promoting_modes(g,r,relative_rate_tolerance=.03)
    assert result['selected_modes']==[0,1]
    for row in result['history']:
        a=np.zeros_like(g);a[:,result['selected_modes'][:result['history'].index(row)+1]]=g[:,result['selected_modes'][:result['history'].index(row)+1]]
        retained=float(np.einsum('aj,jk,ak->',a.conj(),r,a).real)
        assert abs(result['total_response']-retained)<=row['omission_absolute_bound']+1e-12
    with pytest.raises(ValueError):rank_soc_promoting_modes(g,np.diag([1.,-1.,1.]))


def test_orthogonal_promoters_do_not_acquire_artificial_interference():
    result=rank_soc_promoting_modes(np.eye(3),np.diag([115.,103.,.1]))
    assert result['selected_modes']==[0,1]
    assert result['history'][-1]['omission_absolute_bound']==pytest.approx(.1)


def test_nonradiative_rate_reduces_columns_without_line_records(monkeypatch):
    m=HarmonicVibronicModel([1000.],[1000.],[[1.]],[0.],0.,[0.,0.,0.],np.zeros((3,1)))
    energies=np.array([-10.,0.,10.]);strengths=np.array([1.,2.,1.])*1e-10
    def selected(*args,**kwargs):
        assert kwargs['line_format']=='columns'
        return dict(line_format='columns',line_columns=dict(wavenumber_cm=energies,weighted_strength_au2=strengths),
            diagnostics=dict(strength_sum_rule=dict(total=strengths.sum()),reported_strength_au2=strengths.sum()))
    monkeypatch.setattr(vibronic_rates,'harmonic_vibronic_ti',selected)
    monkeypatch.setattr(vibronic_rates,'_harmonic_ti_lines',lambda *a:pytest.fail('record materialization'))
    actual=vibronic_rates.harmonic_nonradiative_rate(m,[0.,0.,0.],method='TI')
    assert actual['rate_s_1']==pytest.approx(vibronic_rates.GOLDEN_RULE_FACTOR*np.dot(strengths,voigt_profile(energies,50.,0.)))


def test_energy_filter_retains_resonant_lines_and_can_resume_unfiltered(tmp_path):
    from matrix_trinity.vibronic_ti import harmonic_vibronic_ti
    model=HarmonicVibronicModel([1000.],[1000.],[[1.]],[1.],-1000.,[1.,0.,0.],np.zeros((3,1)))
    common=dict(selection='targeted',selection_windows_cm=[[-100.,100.]],
        selection_reference_controls=dict(time_step_fs=.5,time_points=8193,gaussian_sigma_cm=50.),
        selection_energy_bias_per_cm=0.,maximum_final_quanta=8,maximum_class=1,
        maximum_integrals=100,maximum_states=20,maximum_transitions=20,
        line_format='columns',target_fraction=.99,selection_batch_size=1)
    checkpoint=tmp_path/'ti.npz'
    first=harmonic_vibronic_ti(model,checkpoint_path=checkpoint,**{**common,'maximum_states':1})
    filtered=harmonic_vibronic_ti(model,checkpoint_path=checkpoint,resume=True,
        selection_energy_range_cm=[-400.,400.],**common)
    all_lines=harmonic_vibronic_ti(model,**common)
    f=filtered['line_columns'];a=all_lines['line_columns']
    np.testing.assert_allclose(f['weighted_strength_au2'][f['wavenumber_cm']==0.],
        a['weighted_strength_au2'][a['wavenumber_cm']==0.])
    assert filtered['diagnostics']['energy_filter_density_upper_bound']>=0
    assert filtered['diagnostics']['checkpoint_resumed']
    with pytest.raises(ValueError,match='cannot widen'):
        harmonic_vibronic_ti(model,checkpoint_path=checkpoint,resume=True,**common)


def test_energy_filter_separates_proposals_from_evaluated_transition_budget():
    from matrix_trinity.vibronic_ti import harmonic_vibronic_ti
    model=HarmonicVibronicModel([1000.],[1000.],[[1.]],[1.],-1000.,[1.,0.,0.],np.zeros((3,1)))
    result=harmonic_vibronic_ti(model,selection='targeted',selection_windows_cm=[[-100.,100.]],
        selection_reference_controls=dict(time_step_fs=.5,time_points=8193,gaussian_sigma_cm=50.),
        selection_energy_bias_per_cm=0.,maximum_final_quanta=8,maximum_class=1,
        maximum_integrals=100,maximum_states=20,maximum_transitions=1,
        selection_energy_range_cm=[-400.,400.],line_format='columns',selection_batch_size=1)
    assert result['diagnostics']['evaluated_transitions']==1
    assert result['diagnostics']['visited_proposal_states']>1
    assert result['diagnostics']['energy_filtered_proposal_states']>0


def test_filtered_checkpoint_pending_rows_respect_resumed_transition_limit(tmp_path):
    from matrix_trinity.vibronic_ti import harmonic_vibronic_ti
    model=HarmonicVibronicModel([1000.],[1000.],[[1.]],[1.],0.,[1.,0.,0.],np.zeros((3,1)))
    common=dict(selection='targeted',selection_windows_cm=[[-500.,3500.]],
        selection_reference_controls=dict(time_step_fs=.5,time_points=8193,gaussian_sigma_cm=50.),
        selection_energy_bias_per_cm=0.,maximum_final_quanta=8,maximum_class=1,
        maximum_states=20,selection_energy_range_cm=[-5000.,8000.],line_format='columns',
        selection_batch_size=4,checkpoint_path=tmp_path/'pending.npz',target_fraction=1.)
    first=harmonic_vibronic_ti(model,maximum_integrals=2,maximum_transitions=20,**common)
    assert first['diagnostics']['pending_transitions']>=2
    resumed=harmonic_vibronic_ti(model,maximum_integrals=100,maximum_transitions=3,resume=True,**common)
    assert resumed['diagnostics']['evaluated_transitions']==3
    assert resumed['diagnostics']['pending_transitions']>=1
