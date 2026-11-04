# TI state assignments verified by TD in two windows

This finite-temperature two-mode model is a small reproducibility example,
not an electronic-structure prediction. Its input defines both harmonic fields,
Duschinsky mixing, displacement, electric/magnetic moments, electric HT
coefficients, temperature, broadening and finite TI budgets. All numbers are
supplied locally in `request.json`; no electronic backend or parent checkout
is accessed.

After installing the standalone and explicitly building its native executable:

```
env/bin/python source/examples/trinity/targeted_ti/run.py \
  --executable bin/sati-harmonic --output results/targeted-ti
```

`result.json` must report `passed: true`. `assignments.json` gives full initial
and final occupations, signed physical FC/HT/interference strengths, broadened
window contributions and inside/tail totals. `transitions.npz` retains every
selected physical line. `profiles.npz` supplies the independent analytic TD
profile and the sampled TI/TD comparison in each requested window. Running the
same command again resumes the checkpoint and verifies the existing line list.
Neither transition strengths nor the displayed profiles are rescaled.

Both windows must recover at least 99% of the positive operator-bank reference;
every observable must also meet 2% relative peak and 1% integrated absolute
profile-error goals. Distinct occupation rankings share one exact integral
cache and a deduplicated physical transition list. Global completeness remains
separately reported. Run the supplied `qualify_extensions.py` driver for the
additional signed-cancellation, restart, empty-budget and shared-cache checks.
