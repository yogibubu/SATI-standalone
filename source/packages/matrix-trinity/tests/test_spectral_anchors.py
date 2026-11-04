import itertools
import numpy as np
import pytest

from matrix_trinity.hdc_polynomial import HDCPolynomialEnergy
from matrix_trinity.spectral_anchors import variational_spectral_anchors,replace_spectral_anchor_block


def inputs(modes=2):
    states=np.array([p for p in itertools.product(range(3),repeat=modes) if sum(p)<=2],int)
    nu=np.arange(modes)*170.+500.;x=np.full((modes,modes),4.);np.fill_diagonal(x,12.)
    model=HDCPolynomialEnergy(0.,nu,x)
    order=np.argsort(model.energies(states));levels=model.energies(states)[order]+81.
    overlap=np.zeros((len(states),len(states)));overlap[order,np.arange(len(states))]=1.
    return states,levels,overlap,model


@pytest.mark.parametrize('modes',[1,2,3])
def test_full_anchor_manifold(modes):
    states,levels,overlap,model=inputs(modes)
    result=variational_spectral_anchors(levels,states,overlap,source_identifier='acquired-field',maximum_level_change_cm=.01)
    assert np.allclose(result['model'].energies(states),model.energies(states))
    assert not result['absolute_ground_energy_replaced']
    assert not result['diagonalization_per_occupation']


def test_unconverged_anchors_rejected():
    states,levels,overlap,_=inputs()
    with pytest.raises(ValueError,match='converged'):
        variational_spectral_anchors(levels,states,overlap,source_identifier='field',maximum_level_change_cm=.2)


def test_ambiguous_anchors_rejected():
    states,levels,overlap,_=inputs()
    with pytest.raises(ValueError,match='unambiguously'):
        variational_spectral_anchors(levels,states,overlap*.49,source_identifier='field',maximum_level_change_cm=.01)


def test_incomplete_manifold_rejected():
    states,levels,overlap,_=inputs()
    with pytest.raises(ValueError,match='complete'):
        variational_spectral_anchors(levels,states[:-1],overlap[:-1],source_identifier='field',maximum_level_change_cm=.01)


def test_non_normalized_overlaps_rejected():
    states,levels,overlap,_=inputs()
    with pytest.raises(ValueError,match='normalized'):
        variational_spectral_anchors(levels,states,np.ones_like(overlap),source_identifier='field',maximum_level_change_cm=.01)


def test_exact_resonance_assignment_rejected():
    states,levels,overlap,_=inputs()
    with pytest.raises(ValueError,match='unambiguously'):
        variational_spectral_anchors(levels,states,overlap*.5,source_identifier='field',maximum_level_change_cm=.01)


def test_selected_block_keeps_bath_and_vacuum():
    original=HDCPolynomialEnergy(342.,np.array([400.,500.,600.]),
        np.array([[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]),coordinate_id='native')
    states,levels,overlap,_=inputs()
    anchor=variational_spectral_anchors(levels,states,overlap,source_identifier='field',maximum_level_change_cm=.01)['model']
    subtraction=HDCPolynomialEnergy(0.,original.fundamentals_cm[[2,0]],
        original.x_matrix_cm[np.ix_([2,0],[2,0])])
    revised=replace_spectral_anchor_block(original,[2,0],anchor,perturbative_block=subtraction)
    assert revised.zero_point_cm==342.
    assert revised.fundamentals_cm[1]==original.fundamentals_cm[1]
    assert np.array_equal(revised.x_matrix_cm[1],original.x_matrix_cm[1])
    assert np.array_equal(revised.x_matrix_cm[np.ix_([2,0],[2,0])],anchor.x_matrix_cm)
    assert revised.coordinate_id=='native'


def test_subspace_replacement_requires_subtraction():
    original=HDCPolynomialEnergy(0.,np.array([300.,400.,500.]),np.zeros((3,3)))
    anchor=HDCPolynomialEnergy(0.,np.array([350.,450.]),np.zeros((2,2)))
    with pytest.raises(ValueError,match='subtraction'):
        replace_spectral_anchor_block(original,[0,1],anchor)


def test_virtual_bath_correction_is_retained():
    full=HDCPolynomialEnergy(90.,np.array([510.,620.,700.]),np.eye(3)*5.)
    perturbative=HDCPolynomialEnergy(0.,np.array([500.,600.]),np.eye(2)*4.)
    variational=HDCPolynomialEnergy(0.,np.array([450.,550.]),np.eye(2)*2.)
    revised=replace_spectral_anchor_block(full,[0,1],variational,perturbative_block=perturbative)
    np.testing.assert_allclose(revised.fundamentals_cm,[460.,570.,700.])
    np.testing.assert_allclose(np.diag(revised.x_matrix_cm),[3.,3.,5.])
    assert revised.zero_point_cm==90.
