import numpy as np
import pytest
from matrix_trinity.fixed_action_spectral import (FixedActionSpectralCorrection,
    occupation_coefficients,perturbative_action_anchors)
from matrix_trinity.hdc_polynomial import HDCPolynomialEnergy

LABELS=np.array([[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]])

def test_complete_anchor_interpolation():
    model=HDCPolynomialEnergy(20.,[500.,700.],[[2.,3.],[3.,4.]])
    z,n,x=occupation_coefficients(LABELS,model.energies(LABELS))
    assert z==20.;np.testing.assert_allclose(n,model.fundamentals_cm)
    np.testing.assert_allclose(x,model.x_matrix_cm)

def test_subtraction_identity_all_actions_and_external_couplings():
    model=HDCPolynomialEnergy(100.,[500.,700.,900.],[[2.,3.,4.],[3.,5.,6.],[4.,6.,7.]])
    c=np.arange(18.).reshape(3,6)
    correction=FixedActionSpectralCorrection(LABELS,c,c)
    for action in (-2.,0.,.7):
        result=correction.apply(model,[0,2],action)
        assert result.zero_point_cm==model.zero_point_cm
        np.testing.assert_array_equal(result.fundamentals_cm,model.fundamentals_cm)
        np.testing.assert_array_equal(result.x_matrix_cm,model.x_matrix_cm)

def test_ground_counted_once_and_action_dependence_preserved():
    model=HDCPolynomialEnergy(100.,[500.,700.,900.],np.ones((3,3)))
    pt=np.zeros((3,6));v=np.zeros_like(pt)
    v[0]=10.+LABELS@[2.,3.];v[1]=5.+LABELS@[7.,8.];v[2]=2.
    result=FixedActionSpectralCorrection(LABELS,v,pt).apply(model,[0,2],.5)
    assert result.zero_point_cm==113.
    np.testing.assert_allclose(result.fundamentals_cm,[505.5,700.,907.])
    np.testing.assert_allclose(result.x_matrix_cm,model.x_matrix_cm)

def test_harmonic_action_exact():
    w=np.array([500.,730.]);b=np.array([20.,30.])
    c=perturbative_action_anchors(w,np.zeros((2,)*3),np.zeros((2,)*4),b,LABELS)
    np.testing.assert_allclose(c[0],(LABELS+.5)@w)
    np.testing.assert_allclose(c[1],(LABELS+.5)@b)
    np.testing.assert_array_equal(c[2],0.)

def test_cubic_ground_analytic_and_action_series():
    # V3=f q^3/6: E0=w/2-11 f^2/(288 w).
    labels=np.array([[0],[1],[2]]);w=500.;b=20.;f=30.
    c=perturbative_action_anchors([w],np.array([[[f]]]),np.zeros((1,)*4),[b],labels)
    value=11*f*f/288
    np.testing.assert_allclose(c[:,0],[w/2-value/w,b/2+value*b/w**2,-value*b*b/w**3])

@pytest.mark.parametrize('labels',[LABELS[:-1],np.vstack([LABELS,LABELS[0]])])
def test_incomplete_manifold_rejected(labels):
    with pytest.raises(ValueError):occupation_coefficients(labels,np.zeros(len(labels)))

def test_resonance_rejected():
    f=np.ones((2,)*3)
    with pytest.raises(ValueError,match='resonant'):
        perturbative_action_anchors([500.,1000.],f,np.zeros((2,)*4),[20.,30.],LABELS)
