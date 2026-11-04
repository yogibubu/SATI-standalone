# matrix-trinity

TRINITY builds harmonic and anharmonic vibrational fields and prepares the
contracts consumed by the MATRIX rovibrational solvers. Primitive coordinate
definitions are supplied by SMITH; Wilson GF transformations are supplied by
`matrix-gf`.

Optional sampling backends are installed with `matrix-trinity[sampling]` or
`matrix-trinity[gpu]`. Keymaker owns workflows that combine TRINITY with LINK,
MORPHEUS and LOCK; those applications no longer enlarge the TRINITY runtime.

## Harmonic electronic vibronic spectra

Native common-ESALC/Wilson-GF FC, HT and FCHT spectra are available through
`matrix trinity vibronic`, also exposed as `matrix lock vibronic`.
Both sum-over-states TI and independent analytic thermal TD calculations are
implemented. See [the method and input guide](../../docs/TRINITY_vibronic_harmonic.md)
and [the example](../../examples/trinity/esalc_vibronic/request.json).

SATI (**Symmetry-Adapted Transition Intensities**) is the electronic vibronic
frontend: `matrix sati vibronic` and `matrix sati raman` reuse these native
services. Coherent KHD resonance Raman supports independent TI/TD paths and
polarized cross sections; see [the Raman guide](../../docs/SATI_resonance_raman.md).
The [published-constant iodine example](../../docs/reports/sati_iodine_raman_20261020/README.md)
compares TI, TD and an independent overlap quadrature without solver-specific
molecular adjustments.

## Repeated anharmonic Golden Rule sampling

`vibronic_spectral_cache` prepares a reusable complete-lowest eigenwindow of a
finite sparse Hamiltonian. `SparseSpectatorObservable(prepared_spectrum=...)`
then projects only eigenvectors close to each spectator-dependent target.
The omitted Gaussian tails are bounded per unit wavepacket norm, using the
same total error budget as the positive Chebyshev filter. Targets outside
the covered window fall back to that filter. Force constants, finite HO
basis, Duschinsky map, linewidth and sampling law are unchanged.

The spectrum is tied to the exact Hamiltonian fingerprint. Its acquisition
checks every eigenpair residual and the full orthogonality Gram matrix;
uncompressed array files support read-only memory mapping and checksum
validation. The acquisition cost is paid once and must be included when
comparing total campaign cost. Roundoff, force-field, active-space and
finite-basis errors remain distinct from the analytic Gaussian-tail bound.

`scripts/prepare_gaussian_spectrum.py` builds such a window from any portable
quartic field with explicit memory, host, spectral-range and worker budgets.
It can also reuse a recorded complete-lowest subset from the general
variational solver, after checking it against the supplied Hamiltonian.

Preparation also supports `--method mkl-dense` and `--method sliced` with an
explicit `--mkl-library` path. They use an existing MKL runtime; no Python MKL
package is required. The sliced method retains the sparse matrix, partitions
the requested energy range using unperturbed PARDISO inertia counts, and
recovers each interval with ARPACK shifted inverse iteration. Every interval
must contain its predicted number of eigenpairs. The assembled result must
also pass residual and full orthogonality checks before publication.

The process budget defaults to two native workers per independent window,
up to 16 processes within the requested total worker budget. Each process
reuses its symbolic factorization; a failed numerical reuse is retried with
a fresh analysis. Counts and completed eigenwindows are checkpointed against
the exact Hamiltonian and numerical controls, so a restart preserves work.
Changing process counts does not change the Hamiltonian or retained window.
The default dense method remains available for complex matrices and systems
without an explicitly configured native runtime.

PARDISO parameter meanings, including inertia and perturbed pivots, follow the
[Intel reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/pardiso-iparm-parameter.html).

### Active-space polynomial audit

`vibronic_active_audit.audit_active_polynomial(field, active_modes)` describes
which canonical cubic/quartic polynomial terms are retained, couple the
active and spectator coordinates, or involve spectators only. Its explicit
coefficient reporting tolerance does not prune the field. Modal coefficient
norms and an omitted-mode ranking are descriptive in the supplied
dimensionless frame; they are not energy or rate error bounds and never
certify active-space convergence.

`quartic_descent_witness(field)` searches explicitly normalized rays. A
negative quartic leading coefficient is a direct witness that the supplied
bare polynomial is unbounded below; a failed search never certifies
confinement. Radial stationary barriers are directional polynomial values,
not minimum-energy-path barriers or molecular dissociation predictions.
`harmonic_basis_count_enclosure` estimates the basis cost without enumerating
product states: upward/downward frequency rounding bounds integer partition
counts, with an explicit count cap and no overflow-based exactness claim.

### Existing Gaussian-field extensions

`vibronic_gaussian_fields` builds positive-source normal QFF blocks and
transports their entire declared polynomial into a supplied harmonic frame.
The source and target geometries and mode gauges are explicit. A separately
authorized rank-deficient anharmonic map permits harmonic-only collective
directions; all target quadratic frequencies remain unchanged. This option
does not acquire new derivatives or convert an imaginary source frequency
into a positive source frequency. `normal_field_as_sonic` retains the field's
actual frequencies in the potential-only VPT2/GVPT2 calculation; a later
frequency-shift proxy and a transported variational Hamiltonian are distinct.

`gaussian_saddle_field_in_harmonic_frame` extends the same tangent transport
to all derivatives of a verified real signed-mode Gaussian QFF, including
its unstable source coordinate. Source absolute frequencies define real
coordinate units only; the output quadratic model uses positive target
frequencies. This does not turn an unstable B3LYP harmonic field into a stable
B3LYP field. Gaussian `Im.Val.` phases and real derivative signs must already
be verified by the signed-QFF adapter. Source omissions and original
derivative geometry remain explicit.

The general `vibronic_active_audit.restrict_modal_field` operation retains a
coordinate subset of a potential-only modal field, setting the omitted
coordinates to zero and preserving every retained mixed tensor element.
`retain_anharmonic_modes` instead keeps the full harmonic frame and switches off
V3/V4 terms involving omitted modes, allowing paired inclusion calculations on
an identical finite basis. Neither operation performs a vacuum contraction or
claims rate convergence. The confinement guard now recognizes an exactly
separable certified anharmonic block plus independent positive harmonic
directions; it does not accept uncaught negative quartic rays.

### Direct native-PES local states

`native_pes_variational` samples the complete resident ARCHITECT/ZAFF energy
through `NativeCartesianModalPotential` and `export_native_modal_grid`, then
`run_native_pes_variational_protocol` qualifies an initial local state without
re-expanding the field into a quartic. The shared
`solve_local_potential_ground_state` uses the same matrix-free Dirichlet
kinetic/projection engine as the existing quartic wrapper. The native callback
must supply the full potential, including its quadratic term and all declared
mixed/nonbonded contributions; frequencies supply kinetic units only.

Installed KEYMAKER protocol: `trinity-native-pes-variational.v1`.
Command: `matrix-keymaker --native-pes-variational request.json`.
See `protocols/keymaker/trinity-native-pes-variational.v1/README.md` and the
analytic native-Morse control under `examples/trinity/native_pes_variational`.
The v1 executable scope is a fixed Cartesian-modal initial state; high-energy
final spectra and total vibronic rates remain separate qualified stages.

### Quantum dynamics and radiative cooling

TRINITY now includes operator-action Chebyshev propagation, dipole-correlation
spectra and reproducible event-driven radiative cascades. Compiled ladder
operators and factorized second-order contact properties avoid constructing a
complete transition-pair inventory. See [quantum dynamics](QUANTUM_DYNAMICS.md)
for units, input examples, effective-operator requirements and qualification.

### Analytic harmonic bath with native active dynamics

`GaussianBathKernel` in `gaussian_partial_trace` reuses the shared Raman
Gaussian number trace and Hermite recurrence. It preserves the coherent finite
initial state and active/spectator Duschinsky correlations while summing every
harmonic spectator occupation analytically. `analytic_bath_nonradiative_rate`
in `vibronic_rates` contracts that kernel with the existing `QuantumDynamics`
engine and uses the common unweighted Golden Rule Fourier transform.

```python
from matrix_trinity.gaussian_partial_trace import GaussianBathKernel
from matrix_trinity.vibronic_rates import analytic_bath_nonradiative_rate

kernel = GaussianBathKernel(model, initial, active_ho_basis, active_modes,
                           kind="emission", generator=existing_generator)
result = analytic_bath_nonradiative_rate(
    dynamics, kernel, soc_hartree, times_fs,
    source_coordinate_id=dynamics.coordinate_id, gaussian_sigma_cm=sigma_cm)
```

The model Hamiltonian is the supplied native active operator plus independent
harmonic spectators; the *source* is not factorized. Constant incoherent
electronic coupling channels are supported. An explicit orthonormal
`embedding` maps contracted active states into the declared HO basis, including
complex phases. Raw DVR modal vectors require a separately qualified overlap
map into this HO basis. For an existing diagonal dynamics representation only
time-by-level phases are stored, rather than time-by-level-by-level matrices.

Small requests use the joint Hermite reference. If its initial/active pair
count exceeds the existing entry budget, the kernel first contracts the
finite initial source using Wick derivatives, multivariate Horner substitution
and a linear FFT convolution. Horner visits only the exact total-degree
support. Every source coefficient is retained; restriction to the requested
active occupation box is algebraically exact. This polynomial represents the
initial wavefunction, not a Taylor replacement of the native potential.

The fallback uses the same entry and workspace limits and adds no physical
cutoff. Its active coefficient box can still grow exponentially with the
number of active modes or their maximum occupations. Oversized calculations
are rejected before allocation. The reported FFT roundoff indicator is not
a certified error bound. Analytic bath summation does not establish active
basis or coupled-mode convergence. Qualification is archived in
`docs/reports/analytic_native_bath_20261024/` and
`docs/reports/source_first_bath_20261025/`.

For a large active space, `GaussianCoherentBathPackets` uses the same source
Wick/Horner contraction with the shared Gaussian lowering closure and a sparse
single-packet convolution. It constructs neither active pairs nor an active
Cartesian coefficient box. `coherent_bath_correlation` in `vibronic_td_rates`
propagates these vectors with `QuantumDynamics` and reuses the common FGR
transform through `nonradiative_rate_from_correlation`. Time/node blocks share
one embedding projection. The optional embedding remains an explicitly
declared active projection and needs independent convergence checks.

The caller supplies coherent nodes and nonnegative weights for the standard
complex Gaussian measure, `pi**(-nb) * exp(-abs(z)**2) dRe(z) dIm(z)`.
For Monte Carlo, use `z=(x+1j*y)/sqrt(2)` with independent standard normal
`x,y` and weights `1/sample_count`. The engine does not normalize supplied
weights. It returns unweighted `node_correlations` so the existing independent
replica analysis can be applied outside the propagator. No number-state bath
cutoff is introduced, but finite integration remains a statistical or
quadrature approximation. The full trace has an analytic resolution identity;
a finite node set is not an analytic bath sum. The rate adapter deliberately
reports no certified Gaussian time-tail bound for these finite coherent sums.
See `docs/reports/coherent_native_bath_20261025/` for qualification and limits.

For a supplied diagonal active spectral representation,
`vibronic_rates.coherent_bath_filtered_rate` is the preferred coherent rate
adapter. It applies the square root of the Gaussian energy filter to each
packet before taking its norm squared. Individual node estimates are then
nonnegative, avoiding cancellation of a direct finite-node bilinear rate.
Both time branches are retained for complex packets. The supplied uniform
positive-time grid must have `4*m+1` points; twice the step and three quarters
of the time window reuse the same packet evaluations as empirical controls.
It returns individual node rates for independent replica analysis. Finite
sampling, time quadrature, active-space and omitted spectral errors are not
silently certified by positivity or by these temporal checks.

### Anharmonic FC/linear-HT observable banks

`vibronic_vpt2_observables.vpt2_operator_spectrum` reuses the native two-surface
SAM contraction in blocks of three operator components. Arbitrary Hermitian
forms retain signed FC/HT interference and strict mechanical order-two
strengths. Completed squares are separate diagnostics. The current bank uses
absorption orientation and explicit nonresonant target states; it does not
supply an anharmonic completeness certificate.

`vibronic_variational_observables.project_sine_grid_states` applies the existing
tensor Hermite projection to saved finite-box eigenvector columns. It returns
raw coefficients, retained norms and a Gram defect, without normalization or
orthogonalization. `finite_basis_operator_spectrum` contracts supplied finite
wavefunctions through the existing full-Duschinsky Gaussian overlap matrix.
Initial occupations are extended for coordinate ladder action, preserving
contributions beyond the supplied top shell. Box size, grid and projection
cutoff remain independent controls; the potential is never regularized here.
