"""Final-mode Gaussian blocks preserve ground-state transition amplitudes."""
import numpy as np
import pytest
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel,PHASE_PER_CM_FS
from matrix_trinity.vibronic_td import harmonic_vibronic_correlation
from matrix_trinity.vibronic_gaussian_blocks import zero_temperature_gaussian_blocks,convolve_gaussian_block_lines


@pytest.mark.parametrize('kind',['absorption','emission'])
def test_block_reconstruction_keeps_complex_operator_interference(kind):
    m=HarmonicVibronicModel([800.,1200.],[900.,1100.],np.diag([.9,1.1]),[.3,-.2],-10000.,
        [.1j,.2,0.],np.array([[.3j,0.],[0.,.4],[0.,0.]]))
    p=zero_temperature_gaussian_blocks(m,kind=kind,matrix_tolerance=1e-14,operator_relative_tolerance=1e-12)
    t=np.linspace(0,100.,15)
    expected=harmonic_vibronic_correlation(m,t,kind=kind)['correlation']
    fc=[harmonic_vibronic_correlation(b,t)['correlation'] for b in p['fc_models']]
    actual=np.zeros(len(t),complex)
    for channel,active in enumerate(p['channel_blocks']):
        v=harmonic_vibronic_correlation(p['operator_models'][channel],t)['correlation']
        for j,g in enumerate(fc):
            if j!=active:v=v*g
        actual+=v
    if p['direction']<0:actual=actual.conj()
    actual*=np.exp(-1j*p['origin_cm']*PHASE_PER_CM_FS*t)
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-12)


def test_same_channel_cross_block_operator_requires_coherent_extension():
    m=HarmonicVibronicModel([1000.,1500.],[1000.,1500.],np.eye(2),[.1,.2],0.,[0.,0.,0.],[[.1,.2],[0.,0.],[0.,0.]])
    with pytest.raises(ValueError,match='spans several'):
        zero_temperature_gaussian_blocks(m,matrix_tolerance=1e-14)


def test_finite_block_convolution_contains_all_products():
    from scipy.special import voigt_profile
    prepared=dict(fc_models=[None,None],channel_blocks=[0,1,0],origin_cm=-1000.,direction=1)
    fc=[(np.array([0.,1000.]),np.array([.8,.2])),(np.array([0.,2000.]),np.array([.9,.1]))]
    ops=[(np.array([1000.]),np.array([2.])),(np.array([2000.]),np.array([3.])),(np.array([0.]),np.array([0.]))]
    axis=np.array([-1000.,0.,1000.,2000.])
    result=convolve_gaussian_block_lines(prepared,fc,ops,gaussian_sigma_cm=50.,bin_width_cm=10.,profile_axis_cm=axis,excitation_cutoff_cm=5000.)
    expected=sum(w*voigt_profile(axis-e,50.,0.) for e,w in [(0.,1.8),(2000.,.2),(1000.,2.4),(2000.,.6)])
    np.testing.assert_allclose(result['density'],expected,atol=1e-16,rtol=1e-12)
    assert np.all(result['lower_transport_density']<=expected+1e-16)
    assert np.all(result['upper_transport_density']>=expected-1e-16)


def test_full_state_assignment_strengths_are_unbinned():
    from matrix_trinity.vibronic_gaussian_blocks import gaussian_block_line_strengths
    prepared=dict(final_mode_groups=[np.array([0]),np.array([1])],channel_blocks=[0,1,0])
    fc=[(np.array([[0],[1]]),np.array([.8,.2])),(np.array([[0],[1]]),np.array([.9,.1]))]
    op=[(np.array([[1]]),np.array([2.])),(np.array([[1]]),np.array([3.])),(np.array([[0]]),np.array([0.]))]
    q=np.array([[1,0],[0,1],[1,1]])
    result=gaussian_block_line_strengths(prepared,fc,op,q)
    np.testing.assert_allclose(result['strengths'],[1.8,2.4,.8])
    assert result['profile_binning_used'] is False
