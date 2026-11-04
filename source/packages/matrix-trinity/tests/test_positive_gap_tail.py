import numpy as np
import pytest
from matrix_trinity.positive_gap_tail import positive_gap_ladder, relaxed_matched_tail_factors


def test_harmonic_limit_and_bound():
    beta=np.array([.002,.005])
    for form in ('exponential','gaussian'):
        a=positive_gap_ladder(120.,120.,inverse_temperatures=beta,form=form)
        np.testing.assert_allclose(a['partition'],1/(-np.expm1(-beta*120.)),rtol=1e-11)
        assert np.all(a['omitted_partition_upper_bound']>=0)


def test_positive_levels_and_exact_boundary():
    for g,w in ((30.,300.),(300.,30.)):
        for form in ('exponential','gaussian'):
            a=positive_gap_ladder(g,w,inverse_temperatures=np.array([.002]),form=form)
            gaps=np.diff(a['levels_cm'])
            assert gaps[0]==g
            assert gaps.min()>=min(g,w)-1e-9
            assert gaps.max()<=max(g,w)+1e-9


def test_resident_disjoint_cones_harmonic_limit():
    from matrix_trinity.hdc_polynomial import HDCPolynomialEnergy
    from matrix_trinity.statistical_partition import matched_occupation_partition
    model=HDCPolynomialEnergy(10.,np.array([100.,200.]),np.zeros((2,2)))
    beta=np.array([.002,.005])
    matched=matched_occupation_partition(model,inverse_temperatures=beta,max_total_quanta=2)
    a=relaxed_matched_tail_factors(matched,model.fundamentals_cm,inverse_temperatures=beta,form='gaussian')
    expected=[]
    for row in matched['tail_frequencies_cm']:
        gaps=row[np.isfinite(row)]
        expected.append(np.expm1(-np.log(-np.expm1(-beta[:,None]*gaps)).sum(axis=1)))
    np.testing.assert_allclose(a['residual_products'],expected,rtol=1e-10)
    assert a['finite_anchors_preserved']


def test_invalid_or_unbounded_ladders_are_rejected():
    for gap in (0.,-1.,np.nan):
        with pytest.raises(ValueError):
            positive_gap_ladder(gap,100.,inverse_temperatures=np.array([.002]),form='gaussian')
    with pytest.raises(ValueError):
        positive_gap_ladder(1.,1.,inverse_temperatures=np.array([.002]),form='gaussian',maximum_levels=10)


@pytest.mark.parametrize('form',['exponential','gaussian'])
def test_positive_shared_spectral_convolution(form):
    from scipy.integrate import simpson
    from matrix_rovib.statistical_convolution import convolve_discrete_measure
    beta=np.array([.002,.005]);step=.1;energy=np.arange(0.,20000.+step/2,step)
    # A positive barrier kernel with a strict lower threshold; two independent
    # nonharmonic ladders are convolved by the resident rovibrational owner.
    curve=np.where(energy>500.,1-np.exp(-np.maximum(energy-500.,0.)/300.),0.)
    source=simpson(curve[:,None]*np.exp(-energy[:,None]*beta),x=energy,axis=0)
    product=np.ones(len(beta));combined=curve
    for g,w in ((120.,180.),(240.,200.)):
        ladder=positive_gap_ladder(g,w,inverse_temperatures=beta,form=form)
        product*=ladder['partition']
        combined=convolve_discrete_measure(combined,step,ladder['levels_cm'],np.ones(len(ladder['levels_cm'])),algorithm='positive_direct')
    axis=np.arange(len(combined))*step
    micro=simpson(combined[:,None]*np.exp(-axis[:,None]*beta),x=axis,axis=0)
    np.testing.assert_allclose(micro,source*product,rtol=2e-5)
    assert np.all(combined>=0)
