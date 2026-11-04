import numpy as np
import pytest
from matrix_trinity.spectral_tail import anchor_mode_spectra,spectral_matched_tail_factors,matched_diagonal_quartic
from matrix_trinity.hdc_polynomial import HDCPolynomialEnergy
from matrix_trinity.statistical_partition import matched_occupation_partition


def test_low_anchors_and_forward_spectral_gaps():
    raw=np.array([[0.,110.,240.,390.,560.,750.]])
    a=anchor_mode_spectra(raw,np.array([100.]),np.array([5.]))
    np.testing.assert_allclose(a['levels_cm'][0,:3],[0.,100.,210.])
    np.testing.assert_allclose(np.diff(a['levels_cm'][0])[2:],np.diff(raw[0])[2:]-10.)
    assert not a['perturbative_absolute_ZPE_replaced']


def test_harmonic_limit_reproduces_resident_cones():
    nu=np.array([100.,200.]);m=HDCPolynomialEnergy(10.,nu,np.zeros((2,2)))
    beta=np.array([.002,.005]);matched=matched_occupation_partition(m,inverse_temperatures=beta,max_total_quanta=2)
    levels=nu[:,None]*np.arange(200)
    a=spectral_matched_tail_factors(m,matched,levels,inverse_temperatures=beta)
    expected=[]
    for row in matched['tail_frequencies_cm']:
        gaps=row[np.isfinite(row)]
        expected.append(np.expm1(-np.log(-np.expm1(-beta[:,None]*gaps)).sum(axis=1)))
    np.testing.assert_allclose(a['residual_products'],expected,rtol=1e-12)


def test_bound_requires_more_levels_instead_of_dropping_states():
    m=HDCPolynomialEnergy(0.,np.array([100.]),np.zeros((1,1)));beta=np.array([.002])
    p=matched_occupation_partition(m,inverse_temperatures=beta,max_total_quanta=2)
    with pytest.raises(ValueError,match='remainder'):
        spectral_matched_tail_factors(m,p,np.array([100.*np.arange(10)]),inverse_temperatures=beta)


def test_nonmonotone_anchored_spectrum_is_rejected():
    with pytest.raises(ValueError,match='monotone'):
        anchor_mode_spectra(np.array([[0.,200.,210.,211.,212.]]),np.array([50.]),np.array([1.]))


def test_no_global_single_well_claim_for_unstable_quartics():
    from matrix_dvr.regularized_modes import regularized_transverse_potential
    for cubic,quartic in ((1.,-1.),(100.,1.)):
        with pytest.raises(ValueError,match='quartic'):
            regularized_transverse_potential(np.array([0.]),10.,cubic,quartic,model='quartic-single-well',wall_height_cm=0.)


def test_quartic_dvr_against_independent_harmonic_basis():
    from scipy.linalg import eigh
    from matrix_dvr.regularized_modes import regularized_mode_levels
    w,c,q=200.,30.,80.;n=150
    ladder=np.diag(np.sqrt(np.arange(1,n)),1);Q=(ladder+ladder.T)/np.sqrt(2)
    H=np.diag(w*(np.arange(n)+.5))+c*(Q@Q@Q)/6+q*(Q@Q@Q@Q)/24
    ref=eigh(H,subset_by_index=(0,10),eigvals_only=True);ref-=ref[0]
    a=regularized_mode_levels(w,c,q,max_quanta=10,grid_points=320,half_width=12.,model='quartic-single-well',wall_height_cm=0.)
    np.testing.assert_allclose(a['excitation_levels_cm'],ref,rtol=0,atol=1e-6)


def test_morse_dvr_against_analytic_bound_spectrum():
    from matrix_dvr.regularized_modes import regularized_mode_levels
    w,alpha=1000.,.2
    result=regularized_mode_levels(w,-3*w*alpha,7*w*alpha**2,max_quanta=8,
        grid_points=480,half_width=24.,model='handy-morse',wall_height_cm=0.)
    n=np.arange(9);ref=w*n-w*alpha**2*n*(n+1)/2
    np.testing.assert_allclose(result['excitation_levels_cm'],ref,rtol=0,atol=.02)


def test_curvilinear_metric_against_independent_harmonic_basis():
    from scipy.linalg import eigh
    from matrix_dvr.regularized_modes import regularized_mode_levels
    w,c,q,g1,g2=200.,30.,80.,15.,12.;n=170
    ladder=np.diag(np.sqrt(np.arange(1,n)),1);Q=(ladder+ladder.T)/np.sqrt(2)
    P=1j*(ladder.T-ladder)/np.sqrt(2)
    H=np.diag(w*(np.arange(n)+.5))+c*(Q@Q@Q)/6+q*(Q@Q@Q@Q)/24+g1*(P@Q@P)/2+g2*(P@Q@Q@P)/4
    ref=eigh(H,subset_by_index=(0,10),eigvals_only=True);ref-=ref[0]
    a=regularized_mode_levels(w,c,q,max_quanta=10,grid_points=360,half_width=12.,model='quartic-single-well',
        wall_height_cm=0.,metric_first_cm=g1,metric_second_cm=g2)
    np.testing.assert_allclose(a['excitation_levels_cm'],ref,rtol=0,atol=1e-6)


def test_nonpositive_quadratic_metric_is_rejected():
    from matrix_dvr.regularized_modes import regularized_mode_levels
    with pytest.raises(ValueError,match='metric'):
        regularized_mode_levels(100.,0.,10.,max_quanta=4,grid_points=80,model='quartic-single-well',
            wall_height_cm=0.,metric_first_cm=100.,metric_second_cm=1.)


def test_protected_metric_preserves_native_jet_and_positive_bound():
    from matrix_dvr.regularized_modes import completed_diagonal_metric
    q=np.linspace(-30.,30.,6001);w,a,b=1000.,-40.,-20.
    values,info=completed_diagonal_metric(q,w,a,b,damping_radius=3.)
    assert values.min()>=info['positive_lower_bound_cm']>0
    h=1e-3;y,_=completed_diagonal_metric(np.array([-h,0.,h]),w,a,b,damping_radius=3.)
    assert y[1]==w
    np.testing.assert_allclose((y[2]-y[0])/(2*h),a,rtol=0,atol=1e-7)
    np.testing.assert_allclose((y[2]+y[0]-2*y[1])/h**2,b,rtol=0,atol=1e-6)


def test_coupled_spectral_cones_against_full_two_mode_sum():
    from matrix_trinity.mixed_statistical import occupation_shells
    nu=np.array([100.,170.]);x=np.array([[3.,8.],[8.,2.]])
    m=HDCPolynomialEnergy(0.,nu,x);beta=np.array([.005,.01]);n=np.arange(100)
    L=nu[:,None]*n+np.diag(x)[:,None]*n*(n-1)
    matched=matched_occupation_partition(m,inverse_temperatures=beta,max_total_quanta=2)
    a=spectral_matched_tail_factors(m,matched,L,inverse_temperatures=beta,coupled_pair=(0,1))
    states=np.array(list(occupation_shells(2,2)));finite=np.exp(-beta[:,None]*m.energies(states)).sum(axis=1)
    anchors=m.energies(matched['tail_anchor_states'])
    total=finite+np.sum(np.exp(-beta[None,:]*anchors[:,None])*a['residual_products'],axis=0)
    energies=L[0,:,None]+L[1,None,:]+x[0,1]*n[:,None]*n[None,:]
    exact=np.exp(-beta[:,None]*energies.ravel()).sum(axis=1)
    np.testing.assert_allclose(total,exact,rtol=1e-10)


def test_effective_quartic_matches_full_hdc_anharmonicity_with_metric():
    from matrix_trinity.vibronic_vpt2 import SAMVPT2Field
    from matrix_trinity.hdc_polynomial import sam_hdc_polynomial_energy
    w=np.array([200.]);c=np.array([30.]);q=np.array([80.]);g1=np.array([15.]);g2=np.array([12.]);target=np.array([3.])
    a=matched_diagonal_quartic(w,c,q,g1,g2,target)
    field=SAMVPT2Field(w,c.reshape(1,1,1),a['effective_quartic_cm'].reshape(1,1,1,1),
        g1.reshape(1,1,1),g2.reshape(1,1,1,1),0.,'complete','test')
    m=sam_hdc_polynomial_energy(field)
    np.testing.assert_allclose(np.diag(m.x_matrix_cm),target,atol=1e-10)
    assert not a['acquired_force_field_changed']


def test_shared_completed_energy_provider_matches_joint_spectrum():
    from matrix_trinity.spectral_tail import spectral_matched_energies
    from matrix_trinity.mixed_statistical import occupation_shells
    nu=np.array([100.,170.]);x=np.array([[3.,8.],[8.,2.]])
    m=HDCPolynomialEnergy(40.,nu,x);k=np.arange(20);L=nu[:,None]*k+np.diag(x)[:,None]*k*(k-1)
    p=matched_occupation_partition(m,inverse_temperatures=[.005],max_total_quanta=2)
    states=np.array(list(occupation_shells(2,8)))
    actual=spectral_matched_energies(m,states,p,L,coupled_pair=(0,1))
    np.testing.assert_allclose(actual,m.energies(states),rtol=0,atol=1e-10)


def test_empty_spectral_replacement_preserves_native_cones():
    from matrix_trinity.spectral_tail import spectral_matched_energies
    from matrix_trinity.statistical_partition import matched_occupation_energies
    from matrix_trinity.mixed_statistical import occupation_shells
    m=HDCPolynomialEnergy(20.,np.array([150.,220.]),np.array([[3.,7.],[7.,6.]]))
    beta=np.array([.005,.01]);p=matched_occupation_partition(m,inverse_temperatures=beta,max_total_quanta=2)
    n=np.arange(100);L=m.fundamentals_cm[:,None]*n+10*n*(n-1)
    r=spectral_matched_tail_factors(m,p,L,inverse_temperatures=beta,replacement_modes=[])
    expected=np.array([np.expm1(-np.log(-np.expm1(-beta[:,None]*g[np.isfinite(g)])).sum(axis=1))
                       for g in p['tail_frequencies_cm']])
    np.testing.assert_allclose(r['residual_products'],expected,rtol=1e-13)
    states=np.array(list(occupation_shells(2,9)))
    np.testing.assert_allclose(spectral_matched_energies(m,states,p,L,replacement_modes=[]),
                              matched_occupation_energies(m,states,p),atol=1e-10)
    assert r['maximum_relative_table_completion']==0.


def test_partial_spectral_sum_matches_explicit_completed_energies():
    from matrix_trinity.spectral_tail import spectral_matched_energies
    m=HDCPolynomialEnergy(20.,np.array([150.,220.]),np.array([[3.,7.],[7.,6.]]))
    beta=np.array([.005,.01]);p=matched_occupation_partition(m,inverse_temperatures=beta,max_total_quanta=2)
    n=np.arange(100);L=m.fundamentals_cm[:,None]*n+np.array([3.,40.])[:,None]*n*(n-1)
    a=spectral_matched_tail_factors(m,p,L,inverse_temperatures=beta,replacement_modes=[0])
    states=np.array([(i,j) for i in range(100) for j in range(100)])
    levels=spectral_matched_energies(m,states,p,L,replacement_modes=[0])
    low=states.sum(axis=1)<=2
    finite=np.exp(-beta[:,None]*levels[low]).sum(axis=1)
    anchors=m.energies(p['tail_anchor_states'])
    completed=finite+(np.exp(-beta[None,:]*anchors[:,None])*a['residual_products']).sum(axis=0)
    exact=np.exp(-beta[:,None]*levels).sum(axis=1)
    np.testing.assert_allclose(completed,exact,rtol=1e-11)
    # The unselected mode's arbitrary variational curvature has no effect.
    other=L.copy();other[1]=220*n
    np.testing.assert_allclose(spectral_matched_tail_factors(m,p,other,inverse_temperatures=beta,
        replacement_modes=[0])['residual_products'],a['residual_products'],rtol=1e-13)


@pytest.mark.parametrize('selected',[[0,0],[-1],[2],[True]])
def test_invalid_partial_mode_selection_is_rejected(selected):
    m=HDCPolynomialEnergy(0.,np.array([100.,200.]),np.zeros((2,2)))
    p=matched_occupation_partition(m,inverse_temperatures=[.01],max_total_quanta=1)
    with pytest.raises(ValueError,match='replacement mode'):
        spectral_matched_tail_factors(m,p,m.fundamentals_cm[:,None]*np.arange(100),
            inverse_temperatures=[.01],replacement_modes=selected)
