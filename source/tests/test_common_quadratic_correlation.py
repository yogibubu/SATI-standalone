import numpy as np
import pytest
from matrix_trinity import common_quadratic_correlation,retain_common_quadratic_pairs
from matrix_trinity.common_quadratic_modes import joint_quadratic_modes
from matrix_trinity.vibronic_harmonic import ESALCHarmonicState,harmonic_vibronic_from_esalc
from matrix_trinity.vibronic_td import harmonic_vibronic_correlation
from matrix_trinity.gaussian_partial_trace import gaussian_partial_number_trace,gaussian_vacuum_trace_batch
from matrix_gf.harmonic import HESSIAN_EIGENVALUE_TO_CM


def test_direct_common_quadratic_matches_standard_harmonic():
    omega=np.array([430.,700.,1150.])
    f=np.diag((omega/HESSIAN_EIGENVALUE_TO_CM)**2)
    a=ESALCHarmonicState(f,np.eye(3),np.array([.2,-.3,.1]),'shared')
    r=np.array([[1.,.1,0.],[.1,1.2,.04],[0.,.04,.9]])
    b=ESALCHarmonicState(r@f@r,np.diag([1.1,.9,1.02]),np.array([-.1,.15,0.]),'shared')
    frame=joint_quadratic_modes([a,b]);times=np.arange(257)*.25
    direct=common_quadratic_correlation(*frame.states,omega,times,electronic_gap_cm=12000.)
    standard=harmonic_vibronic_from_esalc(*frame.states,electronic_gap_cm=12000.,transition_dipole_au=[1.,0.,0.])
    ref=harmonic_vibronic_correlation(standard,times,operator='FC',workers=1)['correlation']
    assert direct['correlation'] == pytest.approx(ref,abs=1e-11)
    assert not direct['diagnostics']['intersurface_mode_map_constructed']
    diagonal=retain_common_quadratic_pairs(frame.states,[])
    check=common_quadratic_correlation(*diagonal,omega,times,electronic_gap_cm=12000.)
    assert check['diagnostics']['connected_quadratic_block_sizes']==[1,1,1]


def test_batch_number_trace_reuses_scalar_branch():
    m=np.array([[.1,.03],[.03,-.15]]);r=np.array([.2,-.1]);log=-.1
    z=np.exp(-1j*np.arange(7)[:,None]*np.array([.2,.3]))
    got=gaussian_vacuum_trace_batch(m,r,log,z,chunk_size=3)
    expected=[gaussian_partial_number_trace(m,r,log,np.arange(2),row)['constant'] for row in z]
    assert got==pytest.approx(expected,abs=1e-13)


def test_state_specific_screen_and_dressing_full_limit():
    from matrix_trinity import screen_common_quadratic_pairs,common_quadratic_dressed_correlation
    w=np.array([500.,510.]);f=np.diag((w/HESSIAN_EIGENVALUE_TO_CM)**2)
    f[0,1]=f[1,0]=.01*np.sqrt(f[0,0]*f[1,1])
    a=ESALCHarmonicState(f,np.eye(2),np.zeros(2),'same')
    cold=screen_common_quadratic_pairs(a,w,excitation_ceiling_cm=0.)
    hot=screen_common_quadratic_pairs(a,w,excitation_ceiling_cm=10000.)
    assert len(cold['variational_pairs'])<len(hot['variational_pairs'])
    times=np.arange(129)*.5
    result=common_quadratic_dressed_correlation(a,a,w,times,
        variational_pairs_initial=[(0,1)],variational_pairs_final=[(0,1)],electronic_gap_cm=0.)
    assert result['correlation']==pytest.approx(np.ones(len(times)),abs=1e-11)


def test_residual_response_has_third_order_error_in_coupling():
    from matrix_trinity import common_quadratic_dressed_correlation
    w=np.array([500.,810.]);base=np.diag((w/HESSIAN_EIGENVALUE_TO_CM)**2)
    a=ESALCHarmonicState(base,np.eye(2),np.array([.1,-.06]),'response')
    errors=[]
    for scale in (.04,.02):
        f=base.copy();f[0,1]=f[1,0]=scale*np.sqrt(f[0,0]*f[1,1])
        b=ESALCHarmonicState(f,np.eye(2),np.array([-.04,.03]),'response')
        times=np.arange(129)*.25
        exact=common_quadratic_correlation(a,b,w,times,electronic_gap_cm=10000.)
        dressed=common_quadratic_dressed_correlation(a,b,w,times,variational_pairs_initial=[],
            variational_pairs_final=[],electronic_gap_cm=10000.,parameter_step=1/32)
        errors.append(max(abs(exact['correlation']-dressed['correlation'])))
        assert abs(dressed['correlation'][0]-1)<1e-10
    assert errors[0]/errors[1]>6
