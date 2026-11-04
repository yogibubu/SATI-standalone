"""Run the native two-state ESALC example and compare TI and analytic TD."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import voigt_profile
from matrix_trinity import run_esalc_vibronic_workflow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,default=Path(__file__).with_name('request.json'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();config=json.loads(args.request.read_text())
    result=run_esalc_vibronic_workflow(config)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    ti=result['time_independent'];td=result['time_dependent']
    if ti is not None and td is not None:
        x=np.array(td['wavenumbers_cm']);density=np.array(td['density_au2_per_cm']);broadened=np.zeros(len(x))
        for line in ti['lines']:
            broadened+=line['weighted_strength_au2']*voigt_profile(x-line['wavenumber_cm'],
                config['td'].get('gaussian_sigma_cm',50.),config['td'].get('lorentz_hwhm_cm',0.))
        difference=float(np.sum(abs(density-broadened))*(x[1]-x[0]))
        summary=dict(status=result['status'],initial_frequencies_cm=result['model']['initial_frequencies_cm'],
            final_frequencies_cm=result['model']['final_frequencies_cm'],
            coordinate_map_orthogonality_error=result['model']['provenance']['coordinate_map_orthogonality_error'],
            TI_line_count=len(ti['lines']),TI_strength_fraction=ti['diagnostics']['captured_fraction'],
            TD_zero_time_error=td['diagnostics']['zero_time_error'],
            TD_integrated_strength=td['diagnostics']['integrated_strength_au2'],
            TI_TD_correlation_error=result['comparison']['maximum_correlation_error'],
            broadened_spectral_L1_difference=difference,TD_time_window_tail=td['diagnostics']['time_window_tail_ratio'],
            scope='illustrative common-ESALC harmonic fields, not an experimental molecule or GDV differential equivalence claim')
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if args.plot:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fig,axes=plt.subplots(2,1,figsize=(7,5),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
            axes[0].plot(x,broadened,label='TI progression, Gaussian convolution',lw=1.8)
            axes[0].plot(x,density,'--',label='Analytic thermal TD',lw=1.2)
            origin=td['origin_cm'];axes[0].set_xlim(origin-2000,origin+7000)
            axes[0].set_ylabel('Dipole-strength density / a.u.² per cm⁻¹');axes[0].legend()
            axes[0].set_title('Two electronic states in a common ESALC chart · native Wilson GF')
            axes[1].plot(x,density-broadened);axes[1].set_ylabel('TD − TI');axes[1].set_xlabel('Wavenumber / cm⁻¹')
            fig.savefig(args.out/'ti_td_comparison.png',dpi=160);fig.savefig(args.out/'ti_td_comparison.pdf');plt.close(fig)
        print(json.dumps(summary,indent=2))
    else:print(result['status'])


if __name__=='__main__':main()
