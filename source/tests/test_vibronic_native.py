"""Standalone native backend against the general Python scientific owners.

Set MATRIX_VIBRONIC_EXECUTABLE to an explicitly built remote executable.
This test never compiles a binary or silently enables a new runtime backend.
"""
import csv
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel
from matrix_trinity.vibronic_native import write_native_harmonic_input,execute_prepared_native_harmonic
from matrix_trinity.vibronic_td import harmonic_vibronic_correlation
from matrix_trinity.vibronic_ti import harmonic_vibronic_ti


@pytest.fixture
def executable():
    path=os.environ.get('MATRIX_VIBRONIC_EXECUTABLE')
    if not path:pytest.skip('Explicitly built native executable required')
    assert Path(path).is_file()
    return path


def model(n):
    rng=np.random.default_rng(842+n)
    rotation=np.linalg.qr(rng.normal(size=(n,n)))[0]
    rotation=rotation@np.diag(np.linspace(.98,1.02,n))
    return HarmonicVibronicModel(np.linspace(500.,1800.,n),np.linspace(550.,1900.,n),
        rotation,np.linspace(-.1,.3,n),25000.,rng.normal(size=3)+.2j*rng.normal(size=3),
        .03*(rng.normal(size=(3,n))+.2j*rng.normal(size=(3,n))),
        transition_momentum_dimensionless_au=.02*(rng.normal(size=(3,n))+.3j*rng.normal(size=(3,n))))


@pytest.mark.parametrize('n',[1,3,7,27])
def test_dynamic_ti_complex_operator_and_assignments(executable,tmp_path,n):
    m=model(n);temperature=0. if n==27 else 300.
    controls=dict(temperature_K=temperature,maximum_initial_quanta=1,
        maximum_final_quanta=2,maximum_class=n,maximum_transitions=20000,
        maximum_integrals=100000,minimum_strength_au2=0.)
    inp=tmp_path/'model.dat';write_native_harmonic_input(m,inp,method='TI',**controls)
    result=execute_prepared_native_harmonic(inp,tmp_path/'native',executable=executable)
    reference=harmonic_vibronic_ti(m,operator='FCHT',maximum_states=10000,**controls)
    with Path(result['artifacts']['lines']).open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
    native={tuple(int(x) for x in (r['initial_quanta']+r['final_quanta']).split(',') if x!=''):r for r in rows}
    assert len(native)==len(reference['lines'])
    for row in reference['lines']:
        observed=native[tuple(row['initial_quanta']+row['final_quanta'])]
        for key in ('wavenumber_cm','weighted_strength_au2','FC_overlap','FC_strength_au2',
                    'HT_strength_au2','interference_au2','strength_au2'):
            np.testing.assert_allclose(float(observed[key]),row[key],rtol=3e-11,atol=3e-13)


@pytest.mark.parametrize('n,operator',[(1,'FC'),(3,'HT'),(7,'FCHT')])
def test_dynamic_td_thermal_complex_operator(executable,tmp_path,n,operator):
    m=model(n);inp=tmp_path/'model.dat'
    write_native_harmonic_input(m,inp,method='TD',operator=operator,temperature_K=300.,
        gaussian_sigma_cm=500.,time_step_fs=1.,time_points=513,fft_padding_factor=1,
        lower_cm=20000.,upper_cm=30000.)
    result=execute_prepared_native_harmonic(inp,tmp_path/'native',executable=executable)
    trace=np.loadtxt(result['artifacts']['correlation'],skiprows=1)
    reference=harmonic_vibronic_correlation(m,trace[:,0],temperature_K=300.,operator=operator,
        include_origin=False,workers=1)
    np.testing.assert_allclose(trace[:,1]+1j*trace[:,2],reference['correlation'],rtol=3e-11,atol=3e-13)
    assert abs(result['fraction']-1)<3e-12


def test_native_cache_budget_is_enforced(executable,tmp_path):
    inp=tmp_path/'model.dat'
    write_native_harmonic_input(model(7),inp,method='TI',maximum_final_quanta=3,maximum_integrals=2)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        execute_prepared_native_harmonic(inp,tmp_path/'native',executable=executable)
    assert 'cache budget' in failure.value.stderr


@pytest.mark.parametrize('n,channels,kind,method,width',[
    (3,45,'absorption','TD',(100.,40.)),
    (7,6,'emission','TD',(300.,0.)),
    (7,9,'absorption','TI',(90.,30.)),
    (3,6,'emission','TI',(0.,70.)),
    (12,6,'absorption','TI',(90.,30.)),
])
def test_general_signed_operator_bank(executable,tmp_path,n,channels,kind,method,width):
    from matrix_trinity.vibronic_native import write_native_operator_input
    from matrix_trinity.vibronic_harmonic import PHASE_PER_CM_FS
    from matrix_trinity.vibronic_observables import harmonic_operator_spectrum
    from scipy.special import voigt_profile
    m=model(n);rng=np.random.default_rng(981+n+channels)
    c=rng.normal(size=channels)+.2j*rng.normal(size=channels)
    h=.01*(rng.normal(size=(channels,n))+.3j*rng.normal(size=(channels,n)))
    raw=rng.normal(size=(3,channels,channels))+1j*rng.normal(size=(3,channels,channels))
    forms=(raw+raw.conj().swapaxes(1,2))/2
    controls=dict(temperature_K=300.,maximum_initial_quanta=1,maximum_final_quanta=2,
        maximum_class=n,maximum_transitions=20000,maximum_integrals=100000)
    inp=tmp_path/'bank.dat'
    write_native_operator_input(m,inp,c,forms,gradients=h,method=method,kind=kind,
        gaussian_sigma_cm=width[0],lorentz_hwhm_cm=width[1],time_step_fs=1.,time_points=2049,
        fft_padding_factor=1,lower_cm=20000.,upper_cm=30000.,grain_cm=5.,**controls)
    result=execute_prepared_native_harmonic(inp,tmp_path/'bank',executable=executable)
    reference=harmonic_operator_spectrum(m,c,forms,gradients=h,method=method,kind=kind,
        time_step_fs=1.,time_points=2049,gaussian_sigma_cm=width[0],lorentz_hwhm_cm=width[1],
        workers=1,**controls)
    np.testing.assert_allclose(result['sum_rules'],reference['sum_rules'],rtol=2e-11,atol=2e-12)
    if method=='TD':
        trace=np.loadtxt(result['artifacts']['correlation'],skiprows=1)
        carrier=np.exp(1j*reference['origin_cm']*trace[:,0]*PHASE_PER_CM_FS)
        observed=(trace[:,1::2]+1j*trace[:,2::2]).T
        np.testing.assert_allclose(observed,reference['correlations']*carrier,rtol=2e-10,atol=2e-10)
    else:
        with Path(result['artifacts']['lines']).open() as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
        native={tuple(int(x) for x in (r['initial_quanta']+r['final_quanta']).split(',') if x!=''):r for r in rows}
        for k in range(len(reference['wavenumbers_cm'])):
            key=tuple(np.r_[reference['initial_quanta'][k],reference['final_quanta'][k]])
            np.testing.assert_allclose([float(native[key][f'strength_{f}']) for f in range(3)],reference['strengths'][:,k],rtol=2e-10,atol=2e-12)
        spectrum=np.loadtxt(result['artifacts']['spectrum'],skiprows=1)
        expected=reference['strengths']@voigt_profile(spectrum[:,0][None,:]-reference['wavenumbers_cm'][:,None],*width)
        np.testing.assert_allclose(spectrum[:,1:].T,expected,rtol=2e-10,atol=2e-12)


@pytest.mark.parametrize('spectroscopy',['ECD','CPL','TPA','TPCD','MCD'])
@pytest.mark.parametrize('method',['TI','TD'])
def test_public_spectroscopy_uses_native_bank(executable,spectroscopy,method):
    from matrix_trinity.vibronic_optical import harmonic_circular_spectrum,harmonic_two_photon_spectrum,harmonic_mcd_spectrum
    rng=np.random.default_rng(617);m=model(2)
    controls=dict(method=method,temperature_K=200.,maximum_initial_quanta=1,
        maximum_final_quanta=2,maximum_class=2,time_step_fs=1.,time_points=513,
        gaussian_sigma_cm=300.,lorentz_hwhm_cm=20.,workers=1)
    if spectroscopy in ('ECD','CPL'):
        function=harmonic_circular_spectrum
        args=(m,m.transition_dipole_au,.01j*m.transition_dipole_au)
        options=dict(spectroscopy=spectroscopy,electric_gradient=m.transition_gradient_dimensionless_au,
            magnetic_gradient=.01j*m.transition_gradient_dimensionless_au)
    elif spectroscopy in ('TPA','TPCD'):
        function=harmonic_two_photon_spectrum
        tensor=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3));tensor+=tensor.T
        args=(m,tensor);options=dict(spectroscopy=spectroscopy)
        if spectroscopy=='TPCD':
            q=rng.normal(size=(3,3,3));q+=q.swapaxes(1,2);q-=np.einsum('i,jk->ijk',np.trace(q,axis1=1,axis2=2)/3,np.eye(3))
            options.update(magnetic_tensor_over_c=.01j*rng.normal(size=(3,3)),quadrupole_tensor=q,
                photon_wavenumber_cm=12000.,gauge='length',tensor_origin_bohr=[0.,0.,0.])
    else:
        function=harmonic_mcd_spectrum
        args=(m,m.transition_dipole_au,.02*rng.normal(size=(3,3)))
        options=dict(transition_zeeman_cm_per_T=[.1,.2,.3],lower_zeeman_cm_per_T=[.02,-.03,.01])
    reference=function(*args,**options,**controls)
    actual=function(*args,**options,**controls,backend='cpp',native_executable=executable)
    np.testing.assert_allclose(actual['sum_rules'],reference['sum_rules'],rtol=2e-10,atol=3e-12)
    if method=='TI':
        old={tuple(np.r_[i,f]):k for k,(i,f) in enumerate(zip(reference['initial_quanta'],reference['final_quanta']))}
        order=[old[tuple(np.r_[i,f])] for i,f in zip(actual['initial_quanta'],actual['final_quanta'])]
        np.testing.assert_allclose(actual['strengths'],reference['strengths'][:,order],rtol=2e-10,atol=3e-12)
    else:
        np.testing.assert_allclose(actual['wavenumbers_cm'],reference['wavenumbers_cm'],rtol=0,atol=3e-10)
        np.testing.assert_allclose(actual['densities_per_cm'],reference['densities_per_cm'],rtol=2e-9,atol=3e-12)
        if spectroscopy=='MCD':
            np.testing.assert_allclose(actual['MCD_density_per_T_per_cm'],reference['MCD_density_per_T_per_cm'],rtol=2e-8,atol=3e-12)


@pytest.mark.parametrize('temperature',[0.,30000.])
def test_logdet_certificate_and_spectral_branch(executable,tmp_path,temperature):
    m=model(7);path=tmp_path/'branch.dat'
    write_native_harmonic_input(m,path,method='TD',temperature_K=temperature,
        gaussian_sigma_cm=500.,time_step_fs=1.,time_points=513,fft_padding_factor=1,
        lower_cm=20000.,upper_cm=30000.)
    result=execute_prepared_native_harmonic(path,tmp_path/'branch',executable=executable)
    assert result['certified_LU_logdet']==(temperature==0.)
    data=np.loadtxt(result['artifacts']['correlation'],skiprows=1)
    reference=harmonic_vibronic_correlation(m,data[:,0],temperature_K=temperature,include_origin=False,workers=1)
    np.testing.assert_allclose(data[:,1]+1j*data[:,2],reference['correlation'],rtol=2e-10,atol=3e-11)


def test_native_short_window_preserves_diagnostics(executable):
    from matrix_trinity.vibronic_observables import harmonic_operator_spectrum
    m=model(2);arguments=(m,m.transition_dipole_au,[np.eye(3)])
    controls=dict(gradients=m.transition_gradient_dimensionless_au,method='TD',time_points=129,workers=1)
    reference=harmonic_operator_spectrum(*arguments,**controls)
    actual=harmonic_operator_spectrum(*arguments,**controls,backend='cpp',native_executable=executable)
    np.testing.assert_allclose(actual['correlations'],reference['correlations'],rtol=2e-11,atol=2e-12)
    np.testing.assert_allclose(actual['diagnostics']['time_window_tail_relative_bank_norm'],
        reference['diagnostics']['time_window_tail_relative_bank_norm'],rtol=2e-11,atol=2e-12)


@pytest.mark.parametrize('method',['TI','TD'])
@pytest.mark.parametrize('operator',['FC','HT','FCHT'])
@pytest.mark.parametrize('surfaces',[1,2])
def test_native_coherent_raman_routes(executable,method,operator,surfaces):
    from dataclasses import replace
    from matrix_trinity.vibronic_raman import harmonic_resonance_raman
    first=replace(model(2),transition_momentum_dimensionless_au=np.zeros((3,2)),
        provenance={'initial_normal_basis_sha256':'native-raman-qualification'})
    models=[first]
    if surfaces==2:
        models.append(replace(first,final_frequencies_cm=first.final_frequencies_cm*1.07,
            electronic_gap_cm=first.electronic_gap_cm+1300.,transition_dipole_au=first.transition_dipole_au*.7j,
            transition_gradient_dimensionless_au=first.transition_gradient_dimensionless_au*.7j))
    controls=dict(method=method,operator=operator,temperature_K=180.,maximum_initial_quanta=1,
        maximum_final_quanta=2,maximum_intermediate_quanta=8,maximum_class=2,
        gamma_cm=1000.,time_step_fs=.5,time_points=257,workers=1)
    reference=harmonic_resonance_raman(models,[24000.,25500.,27700.],**controls)
    actual=harmonic_resonance_raman(models,[24000.,25500.,27700.],**controls,
        backend='cpp',native_executable=executable)
    np.testing.assert_allclose(actual['polarizability_au'],reference['polarizability_au'],rtol=3e-10,atol=3e-9)
    for key in ('isotropic_90deg_parallel_m2_sr','isotropic_90deg_perpendicular_m2_sr','isotropic_90deg_total_m2_sr'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=5e-10,atol=3e-40)
    key='omitted_polarizability_frobenius_bound_au' if method=='TI' else 'lifetime_tail_polarizability_frobenius_bound_au'
    np.testing.assert_allclose(actual['diagnostics'][key],reference['diagnostics'][key],rtol=5e-7,atol=3e-10)


@pytest.fixture
def native_library(executable):
    from matrix_trinity.vibronic_native_kernels import NativeKernels
    path=Path(str(executable)+'.kernels.so')
    if not path.exists():pytest.skip('Explicitly built native shared kernels required')
    return NativeKernels(path)


def test_native_gaussian_preparation_for_operator_bank(native_library):
    from matrix_trinity.vibronic_observables import _bank
    m=model(7);c=m.transition_dipole_au;h=m.transition_gradient_dimensionless_au
    w=np.array([np.eye(3),[[0.,-.5j,0],[.5j,0.,0],[0.,0.,0.]]])
    _,_,_,_,_,g,constant,linear=_bank(m,c,h,w,'absorption')
    actual,native_c,native_h,rules,norm,q=native_library.prepare_bank(m,c,h,w,300.)
    for key in ('matrix','linear','log_ground'):np.testing.assert_allclose(actual[key],g[key],rtol=3e-12,atol=3e-14)
    np.testing.assert_allclose(native_c,constant,rtol=3e-12,atol=3e-14)
    np.testing.assert_allclose(native_h,linear,rtol=3e-12,atol=3e-14)


@pytest.mark.parametrize('kind',['pure','mixed','fft'])
def test_native_gaussian_pair_proposals(native_library,kind):
    from matrix_trinity.vibronic_selection import _gaussian_number_marginal,gaussian_pair_marginal
    vx=np.array([[1.2,.12],[.12,.8]]);vp=.25*np.linalg.inv(vx) if kind=='pure' else np.array([[.9,-.03],[-.03,1.1]])
    mean=np.array([.4,-.2]);caps=(32,40) if kind=='fft' else (7,6)
    expected=gaussian_pair_marginal(vx,vp,mean,*caps)
    actual=native_library.marginal2(vx,vp,mean,*caps)
    np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=3e-12)
    np.testing.assert_allclose(native_library.marginal1(vx[0,0],vp[0,0],mean[0],64),
        _gaussian_number_marginal(vx[0,0],vp[0,0],mean[0],64),rtol=3e-12,atol=3e-14)


@pytest.mark.parametrize('n',[1,3,7,27])
@pytest.mark.parametrize('budget',[30,100000])
@pytest.mark.parametrize('recurrence_mode',['direct','levels'])
def test_native_packed_cache_preserves_checkpoint_arrays(native_library,n,budget,recurrence_mode):
    if recurrence_mode=='levels':native_library=type(native_library)(native_library.path,recurrence_mode=recurrence_mode)
    from matrix_trinity.vibronic_harmonic import harmonic_gaussian_generator
    from matrix_trinity.vibronic_selection import PackedHermiteCache
    m=model(n);g=harmonic_gaussian_generator(m);rng=np.random.default_rng(201+n)
    states=np.zeros((80,2*n),np.int64)
    for row in range(80):
        for k in rng.integers(0,2*n,size=row%7):states[row,k]+=1
    reference=PackedHermiteCache(g,g['ht_linear'],budget,8)
    actual=PackedHermiteCache(g,g['ht_linear'],budget,8,native_library=native_library)
    ids,exhausted=reference.evaluate(states);native_ids,native_exhausted=actual.evaluate(states)
    assert exhausted==native_exhausted and reference.count==actual.count
    assert reference.edge_count==actual.edge_count
    np.testing.assert_array_equal(native_ids,ids)
    np.testing.assert_array_equal(actual.keys[:actual.count],reference.keys[:reference.count])
    np.testing.assert_allclose(actual.values[:actual.count],reference.values[:reference.count],rtol=2e-13,atol=3e-15)


@pytest.mark.parametrize('maximum_quanta',[1,3,15,255,65535,2**32-1])
@pytest.mark.parametrize('budget',[30,20000])
@pytest.mark.parametrize('recurrence_mode',['direct','levels'])
def test_native_sparse_recurrence_rehash_and_key_widths(native_library,maximum_quanta,budget,recurrence_mode):
    if recurrence_mode=='levels':native_library=type(native_library)(native_library.path,recurrence_mode=recurrence_mode)
    from matrix_trinity.vibronic_selection import PackedHermiteCache
    rng=np.random.default_rng(431);dimension=34
    matrix=np.zeros((dimension,dimension))
    for k in range(0,dimension,2):
        matrix[k:k+2,k:k+2]=[[.12,-.07],[-.07,.08]]
    linear=np.zeros(dimension);linear[::3]=.13
    gaussian=dict(matrix=matrix,linear=linear,log_ground=-.2)
    insertion=np.ones((3,dimension))
    reference=PackedHermiteCache(gaussian,insertion,budget,maximum_quanta)
    actual=PackedHermiteCache(gaussian,insertion,budget,maximum_quanta,native_library=native_library)
    states=np.zeros((64,dimension),np.int64)
    for row in range(64):
        for k in rng.choice(dimension,size=row%7,replace=False):states[row,k]=min(maximum_quanta,1+row%3)
    for batch in (states[:32],states[32:],states[:8]):
        expected,exhausted=reference.evaluate(batch);observed,native_exhausted=actual.evaluate(batch)
        assert exhausted==native_exhausted
        assert (actual.count,actual.edge_count)==(reference.count,reference.edge_count)
        np.testing.assert_array_equal(observed,expected)
        np.testing.assert_array_equal(actual.keys[:actual.count],reference.keys[:reference.count])
        np.testing.assert_array_equal(actual.values[:actual.count],reference.values[:reference.count])
        # Rebuilding checkpoint slot indices must preserve every future lookup.
        native_library.rehash(actual)


@pytest.mark.parametrize("n,temperature,ties",[(1,0.,False),(3,300.,False),(7,300.,False),(3,300.,True)])
def test_native_tree_keeps_existing_heap_order(native_library,n,temperature,ties):
    from matrix_trinity.vibronic_selection import ConditionalOccupationTree
    from matrix_trinity.vibronic_checkpoint import _heap_arrays,_restore_heap
    first=ConditionalOccupationTree(model(n),temperature,12,1024,qi=4,qf=8,maximum_class=3)
    if ties:
        costs=np.broadcast_to(np.arange(13,dtype=float),first.costs.shape)
        first.costs=np.where(np.isfinite(first.costs),costs,np.inf)
    second=ConditionalOccupationTree.__new__(ConditionalOccupationTree)
    for name in ('costs','values','parents','original','ancestors','axes','ranks'):
        setattr(second,name,getattr(first,name).copy())
    for name in ('allocated','visited','budget','modes','qi','qf','maximum_class'):
        setattr(second,name,getattr(first,name))
    second.heap=_restore_heap(*_heap_arrays(first.heap));second.enable_native(native_library)
    for _ in range(4):
        np.testing.assert_array_equal(second.batch(64),first.batch(64))
        assert second.allocated==first.allocated and second.visited==first.visited


@pytest.mark.parametrize('first_backend,second_backend',[('python','cpp'),('cpp','python')])
def test_adaptive_checkpoint_moves_between_backends(executable,native_library,tmp_path,first_backend,second_backend):
    from matrix_trinity.vibronic_observables import harmonic_operator_spectrum
    m=model(3);c=m.transition_dipole_au;h=m.transition_gradient_dimensionless_au
    forms=np.array([np.eye(3),[[0,-.5j,0],[.5j,0,0],[0,0,0]]]);path=tmp_path/'resume.npz'
    controls=dict(gradients=h,method='TI',selection='adaptive',temperature_K=300.,
        maximum_initial_quanta=4,maximum_final_quanta=8,maximum_class=3,
        maximum_integrals=20000,selection_maximum_per_mode_quanta=12,
        selection_batch_size=4,target_fraction=.999999999,workers=1)
    start=harmonic_operator_spectrum(m,c,forms,maximum_states=8,maximum_transitions=8,
        checkpoint_path=path,backend=first_backend,native_executable=executable,**controls)
    result=harmonic_operator_spectrum(m,c,forms,maximum_states=256,maximum_transitions=256,
        checkpoint_path=path,resume=True,backend=second_backend,native_executable=executable,**controls)
    full=harmonic_operator_spectrum(m,c,forms,maximum_states=256,maximum_transitions=256,
        backend=first_backend,native_executable=executable,**controls)
    assert result['diagnostics']['checkpoint_resumed']
    assert len(result['initial_quanta'])>=len(start['initial_quanta'])
    np.testing.assert_array_equal(result['initial_quanta'],full['initial_quanta'])
    np.testing.assert_array_equal(result['final_quanta'],full['final_quanta'])
    np.testing.assert_allclose(result['strengths'],full['strengths'],rtol=3e-10,atol=3e-13)
    with np.load(path,allow_pickle=False) as archive:
        import json
        assert json.loads(str(archive['metadata']))['schema']=='matrix.sati.adaptive_checkpoint.v1'


def test_native_adaptive_operator_proposal(executable,native_library):
    from matrix_trinity.vibronic_observables import harmonic_operator_spectrum
    m=model(2);controls=dict(gradients=m.transition_gradient_dimensionless_au,method='TI',selection='adaptive',
        selection_proposal='operator_c2',temperature_K=300.,maximum_initial_quanta=3,maximum_final_quanta=6,
        maximum_class=2,maximum_states=512,maximum_transitions=512,maximum_integrals=10000,
        selection_maximum_per_mode_quanta=8,selection_batch_size=16,target_fraction=.99,workers=1)
    reference=harmonic_operator_spectrum(m,m.transition_dipole_au,[np.eye(3)],**controls)
    actual=harmonic_operator_spectrum(m,m.transition_dipole_au,[np.eye(3)],**controls,
        backend='cpp',native_executable=executable)
    np.testing.assert_allclose(actual['strengths'].sum(),reference['strengths'].sum(),rtol=2e-10,atol=3e-13)


@pytest.mark.parametrize('kind',['absorption','emission'])
@pytest.mark.parametrize('operator',['FC','HT','FCHT'])
def test_public_dipole_ti_td_correlation(executable,kind,operator):
    from matrix_trinity.vibronic_td import harmonic_vibronic_td
    m=model(3);native=dict(backend='cpp',native_executable=executable)
    controls=dict(kind=kind,operator=operator,temperature_K=300.,maximum_initial_quanta=1,
        maximum_final_quanta=3,maximum_class=3,workers=1)
    reference=harmonic_vibronic_ti(m,**controls);actual=harmonic_vibronic_ti(m,**controls,**native)
    assert actual['schema']==reference['schema']
    assert len(actual['lines'])==len(reference['lines'])
    for a,b in zip(actual['lines'],reference['lines']):
        assert a['initial_quanta']==b['initial_quanta'];assert a['final_quanta']==b['final_quanta']
        for key in ('weighted_strength_au2','FC_overlap','HT_strength_au2','interference_au2'):
            np.testing.assert_allclose(a[key],b[key],rtol=3e-11,atol=3e-13)
    times=np.array([-.5,0.,0.,.7,3.,19.])
    for origin in (False,True):
        controls=dict(kind=kind,operator=operator,temperature_K=300.,include_origin=origin,workers=1)
        ref=harmonic_vibronic_correlation(m,times,**controls)
        out=harmonic_vibronic_correlation(m,times,**controls,**native)
        for key in ref['components']:
            np.testing.assert_allclose(out['components'][key],ref['components'][key],rtol=3e-11,atol=3e-13)
    controls=dict(kind=kind,operator=operator,temperature_K=300.,workers=1,time_points=257,
        time_step_fs=.7,gaussian_sigma_cm=300.,lorentz_hwhm_cm=70.)
    ref=harmonic_vibronic_td(m,**controls);out=harmonic_vibronic_td(m,**controls,**native)
    np.testing.assert_allclose(out['wavenumbers_cm'],ref['wavenumbers_cm'],atol=1e-10)
    np.testing.assert_allclose(out['density_au2_per_cm'],ref['density_au2_per_cm'],rtol=2e-10,atol=3e-14)
    np.testing.assert_allclose(out['correlation'],ref['correlation'],rtol=3e-11,atol=3e-13)


@pytest.mark.parametrize('first_cpp',[False,True])
def test_public_dipole_adaptive_checkpoint(executable,tmp_path,first_cpp):
    m=model(3);native=dict(backend='cpp',native_executable=executable)
    controls=dict(selection='adaptive',temperature_K=300.,maximum_initial_quanta=3,
        maximum_final_quanta=5,maximum_class=3,selection_maximum_per_mode_quanta=5,
        maximum_integrals=20000,selection_batch_size=16,line_format='columns',target_fraction=1.,workers=1)
    checkpoint=tmp_path/'dipole.npz'
    harmonic_vibronic_ti(m,maximum_states=32,maximum_transitions=32,checkpoint_path=checkpoint,
        **controls,**(native if first_cpp else {}))
    result=harmonic_vibronic_ti(m,maximum_states=128,maximum_transitions=128,checkpoint_path=checkpoint,
        resume=True,**controls,**({} if first_cpp else native))
    reference=harmonic_vibronic_ti(m,maximum_states=128,maximum_transitions=128,**controls)
    assert result['diagnostics']['checkpoint_resumed']
    for key in reference['line_columns']:
        np.testing.assert_allclose(result['line_columns'][key],reference['line_columns'][key],rtol=3e-10,atol=3e-12)


@pytest.mark.parametrize('width',[(150.,0.),(0.,100.),(120.,70.)])
@pytest.mark.parametrize('method',['TI','TD'])
def test_public_native_rates(executable,width,method):
    from matrix_trinity.vibronic_rates import harmonic_nonradiative_rate,harmonic_radiative_emission_rate
    from dataclasses import replace
    m=replace(model(3),electronic_gap_cm=2000.)
    native=dict(backend='cpp',native_executable=executable)
    controls=dict(temperature_K=300.,workers=1)
    if method=='TI':controls.update(maximum_initial_quanta=2,maximum_final_quanta=5,maximum_class=3)
    else:controls.update(time_points=513,time_step_fs=.7)
    kwargs=dict(gradient_hartree=m.transition_gradient_dimensionless_au*.01,
        momentum_hartree=m.transition_momentum_dimensionless_au*.01,
        channel_weights=[1.,.5,0.],method=method,gaussian_sigma_cm=width[0],lorentz_hwhm_cm=width[1],**controls)
    ref=harmonic_nonradiative_rate(m,m.transition_dipole_au*.001,**kwargs)
    out=harmonic_nonradiative_rate(m,m.transition_dipole_au*.001,**kwargs,**native)
    np.testing.assert_allclose(out['rate_s_1'],ref['rate_s_1'],rtol=3e-10)
    if method=='TI':
        ref=harmonic_radiative_emission_rate(m,**controls)
        out=harmonic_radiative_emission_rate(m,**controls,**native)
        np.testing.assert_allclose(out['rate_s_1'],ref['rate_s_1'],rtol=3e-11)


@pytest.mark.parametrize('n',[1,3,7,27])
def test_native_variational_overlap_grid(executable,n):
    from matrix_trinity.vibronic_variational import harmonic_overlap_matrix
    m=model(n);initial=np.zeros((3,n),np.int64);initial[1,0]=1;initial[2,-1]=2
    final=initial.copy();final[1,-1]=1
    ref,_=harmonic_overlap_matrix(m,initial,final)
    out,_=harmonic_overlap_matrix(m,initial,final,backend='cpp',native_executable=executable)
    np.testing.assert_allclose(out,ref,rtol=3e-11,atol=3e-13)


@pytest.mark.parametrize('operator',['FC','HT','FCHT'])
def test_native_vpt2_transition_contractions(executable,operator):
    from matrix_trinity.vibronic_vpt2 import SAMVPT2Field,vpt2_vibronic_ti
    m=model(1)
    a=SAMVPT2Field(m.initial_frequencies_cm,np.ones((1,)*3)*6.,np.ones((1,)*4)*2.)
    b=SAMVPT2Field(m.final_frequencies_cm,np.ones((1,)*3)*5.,np.ones((1,)*4)*3.)
    controls=dict(final_states=[[0],[1],[2],[3]],initial_states=[[0],[1]],initial_populations=[.8,.2],operator=operator,workers=1)
    ref=vpt2_vibronic_ti(m,a,b,**controls)
    out=vpt2_vibronic_ti(m,a,b,**controls,backend='cpp',native_executable=executable)
    for key in ('amplitude_orders_au','strength_orders_au2','strengths_au2','wavenumbers_cm'):
        np.testing.assert_allclose(out[key],ref[key],rtol=3e-10,atol=3e-13)


@pytest.mark.parametrize('n',[1,3,27])
def test_public_native_occupation_order(executable,n):
    from matrix_trinity.vibronic_ti import harmonic_vibronic_occupations
    expected=harmonic_vibronic_occupations(n,3,min(n,3))
    actual=harmonic_vibronic_occupations(n,3,min(n,3),backend='cpp',native_executable=executable)
    assert actual==expected


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('compression',[0,1,6])
def test_streamed_numpy_archive_and_checkpoint_digest(tmp_path,axis,compression):
    from matrix_trinity.vibronic_storage import ArrayBlocks,save_vibronic_columns
    from matrix_trinity.vibronic_checkpoint import _digest
    shapes=[(3,4),(2,4)] if axis==0 else [(3,4),(3,2)]
    parts=[np.arange(np.prod(shape)).reshape(shape).astype(np.float64) for shape in shapes]
    blocks=ArrayBlocks(parts,axis=axis);joined=np.concatenate(parts,axis=axis)
    assert _digest(dict(x=blocks))==_digest(dict(x=joined))
    path=tmp_path/'columns.npz';save_vibronic_columns(path,dict(x=blocks,scalar=np.asarray('metadata')),compression_level=compression)
    with np.load(path,allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['x'],joined)
        assert str(archive['scalar'])=='metadata'


@pytest.mark.parametrize('kind,temperature,operator',[('absorption',0.,'FC'),('emission',300.,'HT'),('absorption',300.,'FCHT')])
def test_fused_ti_streamed_lines_and_resume(executable,tmp_path,kind,temperature,operator):
    m=model(3);common=dict(selection='adaptive',kind=kind,temperature_K=temperature,operator=operator,
        maximum_initial_quanta=3,maximum_final_quanta=5,maximum_class=3,
        selection_maximum_per_mode_quanta=5,selection_batch_size=17,maximum_integrals=10000,
        line_format='columns',target_fraction=1.,workers=1)
    native=dict(backend='cpp',native_executable=executable)
    checkpoint=tmp_path/'checkpoint.npz'
    first=harmonic_vibronic_ti(m,maximum_states=34,maximum_transitions=34,checkpoint_path=checkpoint,
        line_storage_path=tmp_path/'lines',**common,**native)
    result=harmonic_vibronic_ti(m,maximum_states=136,maximum_transitions=136,checkpoint_path=checkpoint,
        resume=True,line_storage_path=tmp_path/'lines',**common,**native)
    reference=harmonic_vibronic_ti(m,maximum_states=136,maximum_transitions=136,**common)
    for key in reference['line_columns']:
        np.testing.assert_allclose(result['line_columns'][key],reference['line_columns'][key],rtol=3e-10,atol=3e-10)
    assert isinstance(result['line_columns']['wavenumber_cm'],np.memmap)
    assert not result['line_columns']['wavenumber_cm'].flags.writeable
    assert result['diagnostics']['checkpoint_resumed']
    assert result['diagnostics']['timing_seconds']['selection']>=0
    assert not list((tmp_path/'lines').glob('.line-blocks-*'))


def test_preparation_reuses_geometry_and_finite_overlaps(executable):
    from dataclasses import replace
    from matrix_trinity import NativePreparationCache
    cache=NativePreparationCache();m=model(3)
    native=dict(backend='cpp',native_executable=executable)
    controls=dict(temperature_K=300.,maximum_initial_quanta=1,maximum_final_quanta=3,maximum_class=3,workers=1)
    harmonic_vibronic_ti(m,preparation_cache=cache,**controls,**native)
    changed=replace(m,transition_dipole_au=m.transition_dipole_au*(1+.2j),
        transition_gradient_dimensionless_au=m.transition_gradient_dimensionless_au*(1-.3j))
    actual=harmonic_vibronic_ti(changed,preparation_cache=cache,**controls,**native)
    expected=harmonic_vibronic_ti(changed,**controls,**native)
    np.testing.assert_allclose([x['weighted_strength_au2'] for x in actual['lines']],
        [x['weighted_strength_au2'] for x in expected['lines']],rtol=3e-11,atol=3e-13)
    assert cache.snapshot()['hits']['geometry']>=1
    assert cache.snapshot()['hits']['overlaps']>=1
    harmonic_vibronic_ti(replace(m,mode_map=m.mode_map*1.001),preparation_cache=cache,**controls,**native)
    assert cache.snapshot()['misses']['geometry']==2
    cache.clear();assert cache.snapshot()['bytes']==0


def test_preparation_reuses_td_when_only_widths_change(executable):
    from dataclasses import replace
    from matrix_trinity import NativePreparationCache
    from matrix_trinity.vibronic_td import harmonic_vibronic_td
    cache=NativePreparationCache();m=model(3)
    controls=dict(backend='cpp',native_executable=executable,time_points=129,time_step_fs=.7,temperature_K=300.,workers=1)
    first=harmonic_vibronic_td(m,gaussian_sigma_cm=200.,preparation_cache=cache,**controls)
    first['correlation'][:]=0.  # Published arrays must not mutate the private cache.
    actual=harmonic_vibronic_td(m,gaussian_sigma_cm=400.,lorentz_hwhm_cm=30.,preparation_cache=cache,**controls)
    expected=harmonic_vibronic_td(m,gaussian_sigma_cm=400.,lorentz_hwhm_cm=30.,**controls)
    np.testing.assert_allclose(actual['density_au2_per_cm'],expected['density_au2_per_cm'],rtol=3e-11,atol=3e-13)
    assert cache.snapshot()['hits']['correlation']==1
    harmonic_vibronic_td(replace(m,transition_dipole_au=m.transition_dipole_au*2),preparation_cache=cache,**controls)
    harmonic_vibronic_td(m,preparation_cache=cache,**{**controls,'temperature_K':301.})
    harmonic_vibronic_td(m,preparation_cache=cache,**{**controls,'time_step_fs':.8})
    assert cache.snapshot()['misses']['correlation']==4


def test_preparation_limits_and_serializes_requests(executable):
    from concurrent.futures import ThreadPoolExecutor
    from matrix_trinity import NativePreparationCache
    cache=NativePreparationCache(maximum_bytes=1,maximum_entries=1)
    controls=dict(backend='cpp',native_executable=executable,maximum_final_quanta=2,workers=1)
    harmonic_vibronic_ti(model(1),preparation_cache=cache,**controls)
    assert cache.snapshot()['entries']==0
    cache=NativePreparationCache()
    def call(_):return harmonic_vibronic_ti(model(3),preparation_cache=cache,**controls)['diagnostics']['computed_strength_au2']
    with ThreadPoolExecutor(max_workers=2) as pool:values=list(pool.map(call,range(2)))
    assert values[0]==values[1]
    assert cache.snapshot()['hits']['geometry']==1


def test_fused_pending_restart_after_cache_budget(executable,tmp_path):
    native=dict(backend='cpp',native_executable=executable)
    controls=dict(selection='adaptive',maximum_states=128,maximum_transitions=128,
        line_format='columns',target_fraction=1.,selection_batch_size=17,
        maximum_initial_quanta=3,maximum_final_quanta=5,maximum_class=3,selection_maximum_per_mode_quanta=5,workers=1)
    path=tmp_path/'pending.npz'
    first=harmonic_vibronic_ti(model(3),maximum_integrals=1,checkpoint_path=path,**controls,**native)
    assert first['diagnostics']['cache_budget_exhausted']
    assert first['diagnostics']['pending_transitions']>0
    actual=harmonic_vibronic_ti(model(3),maximum_integrals=10000,checkpoint_path=path,resume=True,**controls,**native)
    expected=harmonic_vibronic_ti(model(3),maximum_integrals=10000,**controls,**native)
    for key in expected['line_columns']:
        np.testing.assert_allclose(actual['line_columns'][key],expected['line_columns'][key],rtol=3e-10,atol=3e-10)


def test_column_export_streams_strided_views(tmp_path):
    from matrix_trinity.vibronic_storage import save_vibronic_columns
    view=np.arange(600,dtype=np.float64).reshape(30,20)[:,::3]
    assert not view.flags.c_contiguous
    path=tmp_path/'strided.npz';save_vibronic_columns(path,dict(view=view))
    with np.load(path,allow_pickle=False) as archive:np.testing.assert_array_equal(archive['view'],view)


@pytest.mark.parametrize('backend',['python','cpp'])
def test_finite_column_output_matches_transition_records(executable,backend):
    controls=dict(maximum_initial_quanta=1,maximum_final_quanta=3,maximum_class=3,temperature_K=300.,workers=1)
    native=dict(backend=backend,native_executable=executable)
    reference=harmonic_vibronic_ti(model(3),**controls,**native)
    actual=harmonic_vibronic_ti(model(3),line_format='columns',**controls,**native)
    assert not actual['lines']
    for key,values in actual['line_columns'].items():
        np.testing.assert_allclose(values,[row[key] for row in reference['lines']],rtol=3e-11,atol=3e-13)


def test_native_rate_consumes_column_output(executable):
    from matrix_trinity.vibronic_rates import harmonic_radiative_emission_rate,harmonic_nonradiative_rate
    m=model(3);controls=dict(backend='cpp',native_executable=executable,maximum_final_quanta=3,workers=1)
    a=harmonic_radiative_emission_rate(m,**controls)
    b=harmonic_radiative_emission_rate(m,line_format='columns',**controls)
    np.testing.assert_allclose(a['rate_s_1'],b['rate_s_1'],rtol=3e-11)
    a=harmonic_nonradiative_rate(m,m.transition_dipole_au*.001,method='TI',**controls)
    b=harmonic_nonradiative_rate(m,m.transition_dipole_au*.001,method='TI',line_format='columns',**controls)
    np.testing.assert_allclose(a['rate_s_1'],b['rate_s_1'],rtol=3e-11)


def test_native_zero_coupling_adaptive_columns(executable):
    from matrix_trinity.vibronic_rates import harmonic_nonradiative_rate
    result=harmonic_nonradiative_rate(model(3),np.zeros(3),method='TI',line_format='columns',selection='adaptive',
        backend='cpp',native_executable=executable,maximum_states=128,maximum_transitions=128,workers=1)
    assert result['rate_s_1']==0.


@pytest.mark.parametrize('workers',[1,2,4])
def test_degree_levels_persistent_workspace_and_parallel_sums(native_library,workers):
    from itertools import combinations
    from matrix_trinity.vibronic_harmonic import harmonic_gaussian_generator
    from matrix_trinity.vibronic_selection import PackedHermiteCache
    native_library=type(native_library)(native_library.path,recurrence_mode='levels',recurrence_workers=workers)
    g=harmonic_gaussian_generator(model(8));states=np.zeros((560,16),np.int64)
    for row,indices in enumerate(combinations(range(16),3)):states[row,list(indices)]=1
    reference=PackedHermiteCache(g,g['ht_linear'],10000,3)
    actual=PackedHermiteCache(g,g['ht_linear'],10000,3,native_library=native_library)
    expected,exhausted=reference.evaluate(states);observed,native_exhausted=actual.evaluate(states)
    assert not exhausted and not native_exhausted
    np.testing.assert_array_equal(observed,expected)
    np.testing.assert_array_equal(actual.keys[:actual.count],reference.keys[:reference.count])
    np.testing.assert_array_equal(actual.values[:actual.count],reference.values[:reference.count])
    first=native_library.workspace_stats(actual);handle=actual._native_workspace.handle
    actual.evaluate(states[:2]);second=native_library.workspace_stats(actual)
    assert actual._native_workspace.handle==handle and second['batches']==first['batches']+1
    assert second['buffer_growths']==first['buffer_growths']
    assert second['level_nodes']==actual.count-1 and second['tag_rejections']>0
    assert second['workers']==workers
    assert second['parallel_levels']>0 if workers>1 else second['parallel_levels']==0
    extra=np.zeros((1,16),np.int64);extra[0,:6]=1
    extra_expected,_=reference.evaluate(extra);extra_observed,_=actual.evaluate(extra)
    np.testing.assert_array_equal(extra_observed,extra_expected)
    np.testing.assert_array_equal(actual.values[:actual.count],reference.values[:reference.count])
    assert native_library.workspace_stats(actual)['buffer_growths']==second['buffer_growths']+1
    # New backing allocations trigger a metadata rebuild without stale pointers.
    actual.keys=actual.keys.copy();actual.values=actual.values.copy();actual.slots=actual.slots.copy()
    np.testing.assert_array_equal(actual.evaluate(states[:2])[0],expected[:2])
    native_library.rehash(actual);assert native_library.workspace_stats(actual) is None
    np.testing.assert_array_equal(actual.evaluate(states[:2])[0],expected[:2])
    assert native_library.workspace_stats(actual)['batches']==1
    actual._native_workspace.close();actual._native_workspace.close()


@pytest.mark.parametrize('first_mode,second_mode',[('direct','levels'),('levels','direct')])
def test_adaptive_level_checkpoint_interchange(executable,tmp_path,first_mode,second_mode):
    m=model(3);checkpoint=tmp_path/'levels.npz'
    controls=dict(operator='FCHT',selection='adaptive',backend='cpp',native_executable=executable,
        temperature_K=300.,maximum_initial_quanta=4,maximum_final_quanta=8,maximum_class=3,
        maximum_integrals=10000,selection_maximum_per_mode_quanta=12,selection_batch_size=16,
        target_fraction=.999999999,line_format='columns')
    harmonic_vibronic_ti(m,maximum_states=32,maximum_transitions=32,recurrence_mode=first_mode,
        checkpoint_path=checkpoint,**controls)
    resumed=harmonic_vibronic_ti(m,maximum_states=256,maximum_transitions=256,recurrence_mode=second_mode,
        checkpoint_path=checkpoint,resume=True,**controls)
    full=harmonic_vibronic_ti(m,maximum_states=256,maximum_transitions=256,recurrence_mode='direct',**controls)
    for key in full['line_columns']:
        np.testing.assert_array_equal(resumed['line_columns'][key],full['line_columns'][key])
    assert resumed['diagnostics']['recurrence_workspace']['mode']==second_mode


@pytest.mark.parametrize('recurrence_mode',['direct','levels'])
def test_hash_tag_collisions_still_compare_exact_keys(native_library,recurrence_mode):
    from matrix_trinity.vibronic_selection import PackedHermiteCache
    native_library=type(native_library)(native_library.path,recurrence_mode=recurrence_mode)
    gaussian=dict(matrix=np.zeros((16,16)),linear=np.linspace(.1,.4,16),log_ground=-.2)
    insertion=np.ones((3,16));reference=PackedHermiteCache(gaussian,insertion,128,1)
    actual=PackedHermiteCache(gaussian,insertion,128,1,native_library=native_library)
    seen={};mask=(1<<64)-1;pair=None
    for word in range(1,65536):
        mixed=(word+0x9e3779b97f4a7c15)&mask
        mixed=((mixed^(mixed>>30))*0xbf58476d1ce4e5b9)&mask
        mixed=((mixed^(mixed>>27))*0x94d049bb133111eb)&mask;mixed^=mixed>>31
        hashed=((1469598103934665603^mixed)*1099511628211)&mask
        signature=(hashed&(len(actual.slots)-1),hashed>>56)
        if signature in seen:pair=(seen[signature],word);break
        seen[signature]=word
    assert pair is not None
    states=np.array([[(word>>k)&1 for k in range(16)] for word in (*pair,*pair)],np.int64)
    expected,exhausted=reference.evaluate(states);observed,native_exhausted=actual.evaluate(states)
    assert not exhausted and not native_exhausted
    assert expected[0,0]!=expected[1,0]
    np.testing.assert_array_equal(observed,expected)
    np.testing.assert_array_equal(actual.keys[:actual.count],reference.keys[:reference.count])
    np.testing.assert_array_equal(actual.values[:actual.count],reference.values[:reference.count])


def test_persistent_index_switches_storage_width_without_changing_ids(native_library):
    from matrix_trinity.vibronic_selection import PackedHermiteCache
    gaussian=dict(matrix=np.array([[.1,.02],[.02,.12]]),linear=np.array([.2,.3]),log_ground=-.1)
    insertion=np.ones((3,2));reference=PackedHermiteCache(gaussian,insertion,128,3)
    actual=PackedHermiteCache(gaussian,insertion,128,3,native_library=native_library)
    states=np.array([[1,2],[3,1],[2,2]],np.int64)
    expected,_=reference.evaluate(states)
    # Reserve virtual capacity; only these few recurrence nodes are touched.
    # The small exact slot table suffices for this explicitly bounded test.
    keys=np.empty((2**24,1),np.uint64);values=np.empty(2**24)
    keys[0]=actual.keys[0];values[0]=actual.values[0]
    actual.keys=keys;actual.values=values
    np.testing.assert_array_equal(actual.evaluate(states)[0],expected)
    np.testing.assert_array_equal(actual.values[:actual.count],reference.values[:reference.count])
    assert native_library.workspace_stats(actual)['index_bytes']==len(actual.slots)*8
    actual.keys=actual.keys[:128].copy();actual.values=actual.values[:128].copy()
    np.testing.assert_array_equal(actual.evaluate(states)[0],expected)
    assert native_library.workspace_stats(actual)['index_bytes']==len(actual.slots)*4


def _checkpoint_fixture():
    from types import SimpleNamespace
    from matrix_trinity.vibronic_native_kernels import SavedNativeHeap
    tree=SimpleNamespace(costs=np.arange(4.),values=np.arange(4,dtype=np.int64),
        parents=np.array([-1,0]),original=np.arange(2),ancestors=np.zeros(16,np.int64),
        axes=np.zeros(16,np.int64),ranks=np.zeros(16,np.int64),allocated=2,visited=2,
        modes=1,qi=2,qf=2,maximum_class=1,
        heap=SavedNativeHeap(np.array([1.]),np.array([1],np.int64),np.array([-1],np.int64)))
    cache=SimpleNamespace(keys=np.arange(16,dtype=np.uint64).reshape(-1,1),values=np.arange(16,dtype=float),count=3,edge_count=4)
    return tree,cache


@pytest.mark.parametrize('append_only',[False,True])
def test_async_checkpoint_immutable_snapshot_and_bounded_queue(tmp_path,monkeypatch,append_only):
    import threading,json
    from matrix_trinity.vibronic_checkpoint import AdaptiveCheckpoint,_digest
    from matrix_trinity.vibronic_storage import ArrayBlocks
    tree,cache=_checkpoint_fixture();path=tmp_path/'async.npz'
    checkpoint=AdaptiveCheckpoint(path,False,{}, {},asynchronous=True)
    entered=threading.Event();release=threading.Event();original=checkpoint._write
    def delayed(arrays,metadata):
        entered.set();assert release.wait(15);return original(arrays,metadata)
    monkeypatch.setattr(checkpoint,'_write',delayed)
    components=np.array([1.,2.,3.]);column=np.arange(6.).reshape(3,2)
    try:
        assert checkpoint.save(tree,cache,dict(components=components,column_0=ArrayBlocks([column])),append_only=append_only)
        assert entered.wait(15)
        assert not checkpoint.save(tree,cache,dict(components=components))
        components[:]=99.;tree.visited=12
        if append_only:
            cache.values[3:]=99.;tree.axes[3:]=99
        else:
            cache.values[:]=99.;tree.costs[:]=99.;column[:]=99.
    finally:release.set();checkpoint.close()
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        arrays={key:archive[key] for key in archive.files if key!='metadata'}
    assert metadata['sha256']==_digest(arrays) and metadata['tree']['visited']==2
    np.testing.assert_array_equal(arrays['cache_values'],np.arange(3.))
    np.testing.assert_array_equal(arrays['payload_components'],[1.,2.,3.])
    np.testing.assert_array_equal(arrays['payload_column_0'],np.arange(6.).reshape(3,2))
    assert checkpoint.statistics['maximum_pending']==1 and checkpoint.statistics['skipped_busy']==1


def test_async_checkpoint_failure_preserves_complete_archive(tmp_path,monkeypatch):
    import hashlib
    import matrix_trinity.vibronic_checkpoint as module
    tree,cache=_checkpoint_fixture();path=tmp_path/'atomic.npz'
    checkpoint=module.AdaptiveCheckpoint(path,False,{}, {});checkpoint.save(tree,cache,{});checkpoint.close()
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint=module.AdaptiveCheckpoint(path,True,{}, {},asynchronous=True)
    def fail(*args,**kwargs):raise IOError('injected writer failure')
    monkeypatch.setattr(module,'write_npz_blocks',fail)
    checkpoint.save(tree,cache,{})
    with pytest.raises(IOError,match='injected'):checkpoint.flush()
    checkpoint.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('disk',[False,True])
@pytest.mark.parametrize('mode',['direct','levels'])
def test_public_async_checkpoints_match_synchronous_and_resume(executable,tmp_path,disk,mode):
    m=model(3);results=[]
    for asynchronous in (False,True):
        root=tmp_path/str(asynchronous);root.mkdir();path=root/'checkpoint.npz'
        controls=dict(operator='FCHT',selection='adaptive',backend='cpp',native_executable=executable,
            recurrence_mode=mode,checkpoint_async=asynchronous,selection_batch_size=17,
            selection_maximum_per_mode_quanta=8,maximum_integrals=10000,target_fraction=1.,
            line_format='columns',checkpoint_interval_seconds=0.,checkpoint_interval_batches=1)
        first=harmonic_vibronic_ti(m,maximum_states=68,maximum_transitions=68,checkpoint_path=path,
            line_storage_path=root/'first' if disk else None,**controls)
        resumed=harmonic_vibronic_ti(m,maximum_states=136,maximum_transitions=136,checkpoint_path=path,resume=True,
            line_storage_path=root/'resumed' if disk else None,**controls)
        assert resumed['diagnostics']['checkpoint_writer']['scheduled']==resumed['diagnostics']['checkpoint_writer']['completed']
        results.append(resumed)
    for field in results[0]['line_columns']:
        np.testing.assert_array_equal(results[0]['line_columns'][field],results[1]['line_columns'][field])


@pytest.mark.parametrize('mode,expected',[('direct',4096),('levels',8192)])
def test_automatic_native_batch_is_bounded_by_budget(executable,mode,expected):
    result=harmonic_vibronic_ti(model(1),selection='adaptive',backend='cpp',native_executable=executable,
        recurrence_mode=mode,maximum_states=5,maximum_transitions=5,maximum_integrals=1000)
    assert result['diagnostics']['selection_batch_size']==min(5,expected)


@pytest.mark.parametrize('temperature',[0.,298.15,30000.])
@pytest.mark.parametrize('pole_grid',[False,True])
@pytest.mark.parametrize('kind',['absorption','emission'])
def test_thermal_resolvent_branch_and_nonorthogonal_modes(executable,tmp_path,temperature,pole_grid,kind):
    from matrix_trinity.vibronic_native import write_native_operator_input
    from matrix_trinity.vibronic_observables import harmonic_operator_spectrum
    from matrix_trinity.vibronic_harmonic import PHASE_PER_CM_FS
    m=model(12);c=m.transition_dipole_au;h=m.transition_gradient_dimensionless_au
    forms=np.array([np.eye(3),np.diag([1.,-1.,.3])]);path=tmp_path/'resolvent.dat'
    dt=2*np.pi/(4*PHASE_PER_CM_FS*m.final_frequencies_cm[0]) if pole_grid else 1.
    write_native_operator_input(m,path,c,forms,gradients=h,method='TD',kind=kind,temperature_K=temperature,
        gaussian_sigma_cm=500.,time_step_fs=dt,time_points=513,fft_padding_factor=1,
        lower_cm=m.origin_cm-400.,upper_cm=m.origin_cm+400.)
    result=execute_prepared_native_harmonic(path,tmp_path/'resolvent',executable=executable)
    reference=harmonic_operator_spectrum(m,c,forms,gradients=h,method='TD',kind=kind,temperature_K=temperature,
        gaussian_sigma_cm=500.,time_step_fs=dt,time_points=513,workers=1)
    trace=np.loadtxt(result['artifacts']['correlation'],skiprows=1)
    carrier=np.exp(1j*reference['origin_cm']*trace[:,0]*PHASE_PER_CM_FS)
    np.testing.assert_allclose((trace[:,1::2]+1j*trace[:,2::2]).T,
        reference['correlations']*carrier,rtol=2e-10,atol=2e-11)
    counts=result['correlation_evaluations']
    assert sum(counts.values())==513
    assert (counts['symmetric']>0)==(temperature>0)
    assert counts['legacy']==(513 if temperature==0 else 0)
    assert (counts['reduced']>0)==(temperature>0)
    assert result['certified_LU_logdet']==(temperature<30000.)
