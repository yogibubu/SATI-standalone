"""Qualify profile-controlled stopping, shared-cache rankings and assignments."""
import argparse
import json
from pathlib import Path
import socket
import tempfile
import numpy as np
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel
from matrix_trinity.vibronic_targeted import SpectralSelectionTarget
from matrix_trinity.vibronic_observables import harmonic_operator_spectrum
from matrix_trinity.vibronic_ti import harmonic_vibronic_ti
from matrix_trinity.vibronic_optical import harmonic_circular_spectrum
from matrix_trinity.vibronic_workflow import vibronic_json_value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('executable',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if socket.getfqdn()!='oracle.sns.it':raise RuntimeError('Verified Oracle required')
    from matrix_trinity.vibronic_window_selection import _admit
    packed_states=np.zeros((3,150),np.int64);packed_states[1,-1]=1
    indices,count=_admit(packed_states,np.empty((8,19),np.uint64),np.full(16,-1,np.int64),0,8)
    packed_ok=count==2 and np.array_equal(indices,[0,1])
    x=np.linspace(28000.,33000.,4096,endpoint=False);sigma=30.
    def gauss(center):return np.exp(-.5*((x-center)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    bank=.999*gauss(30000)+.001*gauss(31000);signed=-.0005*gauss(31000)
    target=SpectralSelectionTarget([[29500.,31500.]],x,np.array([bank,signed]),bank,gaussian_sigma_cm=sigma)
    target.add(np.array([30000.]),np.array([.999]),np.array([[.999],[0.]]))
    report=dict(multiword_deduplication=bool(packed_ok),area_alone_does_not_stop=bool(target.computed_bank[0]/target.reference_bank[0]>.99 and not target.converged(.99)))
    target.add(np.array([31000.]),np.array([.001]),np.array([[.001],[-.0005]]))
    report['signed_profile_then_converges']=target.converged(.99)
    angle=.23;j=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    model=HarmonicVibronicModel(np.array([240.,810.]),np.array([270.,760.]),j,np.array([.8,-.5]),
        30000.,np.array([.1,.2,.3]),np.array([[.01,.02],[.02,-.01],[.01,.01]]))
    origin=model.origin_cm;windows=[[origin-100,origin+500],[origin+1100,origin+2400]]
    c=np.array([.1,.2,.001j]);h=np.array([[.01,.02],[.02,-.01],[.001j,.002j]])
    forms=np.array([np.eye(3),[[0,0,-.5j],[0,0,0],[.5j,0,0]]])
    controls=dict(selection='targeted',selection_windows_cm=windows,target_fraction=.99,
        backend='cpp',native_executable=a.executable,workers=1,maximum_states=30000,maximum_transitions=30000,
        maximum_integrals=300000,maximum_initial_quanta=8,maximum_final_quanta=10,
        selection_maximum_per_mode_quanta=10,selection_batch_size=4,
        gaussian_sigma_cm=75.,time_step_fs=.25,time_points=4097,selection_assignment_limit=5)
    bounded_result=harmonic_operator_spectrum(model,c,forms,gradients=h,temperature_K=298.15,**controls)
    bounded_diag=bounded_result['diagnostics'];bounded_target=bounded_diag['targeted_selection']
    report['finite_domain_profile_failure_rejected']=bool(np.all(bounded_target['bank_recovery_fraction']>=.99)
        and not bounded_diag['converged'] and np.any(~bounded_target['profile_criteria_met']))
    controls.update(maximum_states=100000,maximum_transitions=100000,maximum_final_quanta=16,
        selection_maximum_per_mode_quanta=16)
    result=harmonic_operator_spectrum(model,c,forms,gradients=h,temperature_K=298.15,**controls)
    diag=result['diagnostics'];stats=diag['window_ranking_statistics'];states=np.column_stack([result['initial_quanta'],result['final_quanta']])
    report['multiwindow_converged']=bool(diag['converged'])
    report['distinct_biases']=bool(len(np.unique(diag['selection_energy_bias_per_cm']))==2)
    report['deduplicated']=bool(stats['duplicate_candidates_removed']>0 and len(np.unique(states,axis=0))==len(states))
    report['both_rankings_used']=all(value>0 for value in stats['per_window_proposal_visits'])
    report['shared_cache']=stats['shared_integral_cache']
    records=result['assignment_report']['windows']
    report['all_line_summaries']=all(np.allclose(w['strength_from_centers_inside']+w['strength_from_external_tails'],
        diag['targeted_selection']['computed_observable_areas'][i],rtol=2e-12,atol=1e-17) for i,w in enumerate(records))
    report['components_sum']=all(np.allclose(np.sum(list(row['components'].values()),axis=0),row['line_strengths'],
        rtol=2e-12,atol=1e-19) for w in records for row in w['assignments'])
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'report.json'
        fc_controls={k:v for k,v in controls.items() if k not in ['gaussian_sigma_cm','time_step_fs','time_points']}
        fc=harmonic_vibronic_ti(model,operator='FCHT',temperature_K=298.15,line_format='columns',
            selection_reference_controls=dict(gaussian_sigma_cm=75.,time_step_fs=.25,time_points=4097),
            selection_assignment_report_path=path,**fc_controls)
        report['FC_HT_multiwindow']=bool(fc['diagnostics']['converged'] and path.exists()
            and json.loads(path.read_text())['window_summaries_include_all_selected_lines'])
        report['FC_components_sum']=all(np.allclose(np.sum(list(row['components'].values()),axis=0),row['line_strengths'],
            rtol=2e-12,atol=1e-19) for w in fc['assignment_report']['windows'] for row in w['assignments'])
        optical_path=Path(directory)/'ecd.json'
        ecd=harmonic_circular_spectrum(model,model.transition_dipole_au,1j*np.array([.001,.002,.003]),
            electric_gradient=model.transition_gradient_dimensionless_au,temperature_K=298.15,
            selection_assignment_report_path=optical_path,**controls)
        saved=json.loads(optical_path.read_text())
        report['optical_report_labels']=bool(saved['observable_names']==list(ecd['observable_names'])
            and all('initial_state_label' in row for w in saved['windows'] for row in w['assignments']))
        displaced=HarmonicVibronicModel(model.initial_frequencies_cm,model.final_frequencies_cm,model.mode_map,
            np.array([4.,4.]),30000.,model.transition_dipole_au,model.transition_gradient_dimensionless_au)
        empty=harmonic_operator_spectrum(displaced,c,forms,gradients=h,
            **(controls|dict(maximum_states=1,maximum_transitions=1,maximum_initial_quanta=0,maximum_final_quanta=0)))
        report['empty_budget_report']=bool(not empty['diagnostics']['converged']
            and empty['diagnostics']['evaluated_transitions']==0
            and all(w['listed_transition_count']==0 for w in empty['assignment_report']['windows']))
        checkpoint=Path(directory)/'checkpoint'
        bounded=controls|dict(maximum_states=3000,maximum_transitions=3000,maximum_initial_quanta=3,
            maximum_final_quanta=4,selection_maximum_per_mode_quanta=4,target_fraction=1.,
            selection_profile_peak_tolerance=0.,selection_profile_L1_tolerance=0.)
        harmonic_operator_spectrum(model,c,forms,gradients=h,temperature_K=298.15,checkpoint_path=checkpoint,
            **(bounded|dict(maximum_states=16,maximum_transitions=16)))
        resumed=harmonic_operator_spectrum(model,c,forms,gradients=h,temperature_K=298.15,
            checkpoint_path=checkpoint,resume=True,**bounded)
        direct=harmonic_operator_spectrum(model,c,forms,gradients=h,temperature_K=298.15,**bounded)
        def sorted_lines(r):
            states=np.column_stack([r['initial_quanta'],r['final_quanta']]);order=np.lexsort(states.T[::-1])
            return states[order],r['strengths'][:,order]
        x,s=sorted_lines(resumed);y,t=sorted_lines(direct)
        report['forest_restart']=bool(resumed['diagnostics']['checkpoint_resumed'] and np.array_equal(x,y)
            and np.allclose(s,t,rtol=3e-13,atol=1e-18))
        report['strict_budget_remains_unconverged']=not resumed['diagnostics']['converged']
    report['diagnostics']=diag
    checks=[v for k,v in report.items() if k!='diagnostics'];report['passed']=all(checks)
    a.output.write_text(json.dumps(vibronic_json_value(report),indent=2)+'\n');print(json.dumps(vibronic_json_value(report),indent=2),flush=True)
    if not report['passed']:raise RuntimeError('Targeted TI extensions failed qualification')


if __name__=='__main__':main()
