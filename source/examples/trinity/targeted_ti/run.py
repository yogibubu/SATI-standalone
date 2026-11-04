"""Portable two-window FC/HT ECD example; uses the installed SATI packages only."""
import argparse
import json
from pathlib import Path
import numpy as np
from matrix_trinity.vibronic_harmonic import HarmonicVibronicModel
from matrix_trinity.vibronic_optical import harmonic_circular_spectrum
from matrix_trinity.vibronic_workflow import _complex_input, vibronic_json_value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,default=Path(__file__).with_name('request.json'))
    parser.add_argument('--output',type=Path,default=Path('targeted-ti-result'))
    parser.add_argument('--executable',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.request.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    m=config['model'];model=HarmonicVibronicModel(np.array(m['initial_frequencies_cm']),
        np.array(m['final_frequencies_cm']),np.array(m['mode_map']),np.array(m['final_shift_dimensionless']),
        m['electronic_gap_cm'],np.array(m['electric_dipole_au']),np.array(m['electric_gradient']))
    electric=model.transition_dipole_au;magnetic=_complex_input(m['magnetic_dipole_over_c_au'])
    operators=dict(electric_gradient=model.transition_gradient_dimensionless_au)
    common=config['spectrum']|dict(backend='cpp',native_executable=args.executable.resolve(),workers=1)
    td=harmonic_circular_spectrum(model,electric,magnetic,method='TD',**operators,**common)
    checkpoint=out/'checkpoint.npz';selection=config['selection']
    ti=harmonic_circular_spectrum(model,electric,magnetic,method='TI',**operators,**common,**selection,
        checkpoint_path=checkpoint,resume=checkpoint.exists(),
        selection_assignment_report_path=out/'assignments.json')
    diagnostics=ti['diagnostics'];target=diagnostics['targeted_selection'];report=ti['assignment_report']
    columns=np.column_stack([ti['initial_quanta'],ti['final_quanta']])
    checks=dict(all_windows_converged=bool(diagnostics['converged']),
        all_observable_profiles_converged=bool(np.all(target['profile_criteria_met'])),
        bank_recovery=bool(np.all(target['bank_recovery_fraction']>=selection['target_fraction'])),
        unique_physical_transitions=bool(len(np.unique(columns,axis=0))==len(columns)),
        exported_assignment_report=(out/'assignments.json').is_file(),
        signed_FC_HT_components=all(np.allclose(np.sum(list(row['components'].values()),axis=0),
            row['line_strengths'],rtol=2e-12,atol=1e-18) for w in report['windows'] for row in w['assignments']),
        shared_integral_cache=diagnostics['window_ranking_statistics']['shared_integral_cache'])
    np.savez_compressed(out/'profiles.npz',TD_axis_cm=td['wavenumbers_cm'],TD_densities=td['densities_per_cm'],
        window_axis_cm=target['profile_wavenumbers_cm'],window_TI_profiles=target['ti_profiles'],
        window_TD_profiles=target['reference_profiles'])
    np.savez_compressed(out/'transitions.npz',wavenumbers_cm=ti['wavenumbers_cm'],strengths=ti['strengths'],
        initial_quanta=ti['initial_quanta'],final_quanta=ti['final_quanta'])
    result=dict(schema='sati.standalone.targeted_example.v1',passed=all(checks.values()),checks=checks,
        observable_names=ti['observable_names'],diagnostics=diagnostics,
        interpretation='TI supplies individually assigned transitions; TD verifies their broadened window profiles. No line strengths are rescaled.',
        input=str(args.request),output_files=['assignments.json','profiles.npz','transitions.npz','checkpoint.npz'])
    (out/'result.json').write_text(json.dumps(vibronic_json_value(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(vibronic_json_value(dict(passed=result['passed'],checks=checks,
        evaluated_transitions=diagnostics['evaluated_transitions'],recovery=target['bank_recovery_fraction'])),indent=2))
    if not result['passed']:raise RuntimeError('Window qualification failed; inspect result.json')


if __name__=='__main__':main()
