import numpy as np
import pytest
from matrix_trinity.hdc_polynomial import HDCPolynomialEnergy
from matrix_trinity.statistical_partition import matched_occupation_partition
from matrix_trinity.spectral_tail import spectral_matched_tail_factors,spectral_matched_energies
from matrix_trinity.joint_spectral_tail import joint_spectral_tail_factors,joint_matched_energies,anchor_pair_spectrum,joint_spectral_energies


def test_separable_joint_limit_preserves_resident_completion():
    m=HDCPolynomialEnergy(20.,np.array([150.,250.,310.]),np.zeros((3,3)));beta=np.array([.015,.025])
    matched=matched_occupation_partition(m,inverse_temperatures=beta,max_total_quanta=2)
    L=m.fundamentals_cm[:,None]*np.arange(100);table=L[1,:12,None]+L[2,None,:12]
    old=spectral_matched_tail_factors(m,matched,L,inverse_temperatures=beta)
    new=joint_spectral_tail_factors(m,matched,L,inverse_temperatures=beta,pair=(1,2),pair_levels_cm=table)
    np.testing.assert_allclose(new['residual_products'],old['residual_products'],rtol=1e-12,atol=1e-14)
    n=np.array([(a,b,c) for a in range(5) for b in range(15) for c in range(15)])
    np.testing.assert_allclose(joint_matched_energies(m,n,matched,L,pair=(1,2),pair_levels_cm=table),spectral_matched_energies(m,n,matched,L),atol=1e-10)


def test_joint_counting_matches_shared_energy_provider_including_borders():
    m=HDCPolynomialEnergy(20.,np.array([150.,250.,310.]),np.array([[0.,2.,3.],[2.,0.,4.],[3.,4.,0.]]));beta=np.array([.015,.025])
    matched=matched_occupation_partition(m,inverse_temperatures=beta,max_total_quanta=2)
    L=m.fundamentals_cm[:,None]*np.arange(100);i,j=np.indices((6,6));table=250*i+310*j+4*i*j+.2*(i*j)**2
    out=joint_spectral_tail_factors(m,matched,L,inverse_temperatures=beta,pair=(1,2),pair_levels_cm=table)
    n=np.array([(a,b,c) for a in range(14) for b in range(14) for c in range(14)])
    energy=joint_matched_energies(m,n,matched,L,pair=(1,2),pair_levels_cm=table)
    finite=n.sum(axis=1)<=2;explicit=np.exp(-beta[:,None]*energy[finite]).sum(axis=1)
    anchor=m.energies(matched['tail_anchor_states']);total=explicit+(np.exp(-beta[None,:]*anchor[:,None])*out['residual_products']).sum(axis=0)
    np.testing.assert_allclose(total,np.exp(-beta[:,None]*energy).sum(axis=1),rtol=1e-12)
    np.testing.assert_allclose(energy[finite],m.energies(n[finite]),atol=1e-12)


def test_common_local_pair_coupling_removed_once():
    i,j=np.indices((8,8));a=200*np.arange(8);b=300*np.arange(8);L=np.array([a,b])
    higher=.001*i*(i-1)*(i-2)*j
    raw=a[:,None]+b[None,:]+10*i+12*j+7*i*(i-1)/2+4*j*(j-1)/2+5*i*j+higher
    result=anchor_pair_spectrum(raw,(a,b),L,3.)
    np.testing.assert_allclose(result['levels_cm'],a[:,None]+b[None,:]+3*i*j+higher,atol=1e-10)


@pytest.mark.parametrize('axis',[0,1])
def test_nonpositive_joint_gaps_rejected(axis):
    i,j=np.indices((5,5));table=(100*i+200*j).astype(float);table[2:]-=150
    if axis:table=table.T
    with pytest.raises(ValueError,match='positive forward'):
        joint_spectral_energies(table,np.array([[0,0]]))


def test_conditional_nonpositive_joint_gaps_rejected_by_both_consumers():
    nu=np.full(3,1000.);x=np.zeros((3,3));x[0,1]=x[1,0]=-100.
    model=HDCPolynomialEnergy(0.,nu,x);matched=matched_occupation_partition(model,inverse_temperatures=[.01],max_total_quanta=2)
    L=nu[:,None]*np.arange(100);i,j=np.indices((6,6));table=100.*i+1000.*j
    with pytest.raises(ValueError,match='conditional joint'):
        joint_spectral_tail_factors(model,matched,L,inverse_temperatures=[.01],pair=(1,2),pair_levels_cm=table)
    with pytest.raises(ValueError,match='conditional joint'):
        joint_matched_energies(model,np.array([[2,1,0]]),matched,L,pair=(1,2),pair_levels_cm=table)
