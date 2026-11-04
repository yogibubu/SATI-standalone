# Two-dimensional water PES example

This is the first complete chemical LINK-SENTINEL v1 example. LINK constructs
an unsymmetrized SONIC model for water and exposes two active variables: one
O-H stretch and the H-O-H bend. The deterministic SENTINEL mock requests a 2x2
batch. LINK realizes all four Cartesian structures concurrently and evaluates
E, Cartesian G and Cartesian H on an inexpensive positive quadratic surface.

From the MATRIX repository root:

```bash
python packages/matrix-link/examples/link_sentinel/water_2d/run_example.py \
  --run-dir /tmp/matrix-water-link-sentinel
```

The calculation is intentionally cheap and analytically reproducible. It tests
the protocol and coordinate machinery; it is not presented as an accurate
water potential.

## Derivative-free parallel xTB Monte Carlo

The recommended baseline uses energy-only xTB, independently adapted stretch
and bend moves, two temperatures and replica exchange:

```bash
python packages/matrix-link/examples/link_sentinel/water_2d/run_mc_xtb.py \
  --run-dir /tmp/matrix-water-mc-xtb \
  --batch-workers 2
```

The checked configuration is `mc_xtb.json`. To use another coordinate contract,
copy that file, replace `bounds` and the `proposal_blocks[].variables` labels,
then pass it with `--config`. `batch-workers` controls concurrent xTB jobs;
each job is restricted to one OpenMP/MKL thread to avoid oversubscription.
The final output reports local acceptance per block and replica-swap
acceptance. Production calculations should use a longer warm-up and discard
its points before analysis.

## Force-biased xTB smoke test

The same active variables can drive a short MALA trajectory using analytic
GFN2-xTB energies and Cartesian gradients projected by LINK:

```bash
python packages/matrix-link/examples/link_sentinel/water_2d/run_mala_xtb.py \
  --run-dir /tmp/matrix-water-mala-xtb \
  --batch-workers 2
```

The checked configuration is `mala_xtb.json`. The example requests energy and
gradient for every walker and enables LINK's fast C1 independent-candidate
policy. A completed run retains every xTB input/output, LINK exchange and
restartable SENTINEL RNG checkpoint below the selected run directory.
