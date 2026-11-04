# Shared time dependent nonradiative rates

The public service is
`matrix_trinity.vibronic_rates.time_dependent_nonradiative_rate`.
It reuses `QuantumDynamics.correlate`, the common Gaussian Fourier transform,
and the existing FGR conversion factor. The supplied source is
`V_fi |i>` in Hartree, expressed in the final Hamiltonian's coordinates.
Several source columns represent incoherent coupling channels. Initial and
final energies must share an origin. The rate uses the unweighted strength
density at the exact zero-energy FFT bin; photon frequency factors are excluded.

## Ownership and reuse

| Component | Responsibility |
| --- | --- |
| ARCHITECT | Existing native ZAFF potential and its provenance |
| SMITH/GF | Existing coordinate definitions and transformations |
| TRINITY `QuantumDynamics` | Existing finite-operator propagation and correlations |
| TRINITY shared Gaussian algebra | Raman Mehler contraction and Hermite recurrences |
| TRINITY `vibronic_rates` | Coupling units, energy conservation and rate conversion |
| SATI | Existing vibronic application interface |
| KeyMaker | Existing workflow orchestration |
| SENTINEL | Existing statistical error controls where sampling is used |

No additional propagator, electronic-structure keyword, CLI keyword or workflow
protocol is introduced. `gaussian_hermite` holds the recurrence previously
implemented inside `vibronic_raman`; the Raman private aliases remain available.
`gaussian_partial_trace` generalizes the same Raman Gaussian elimination to a
selected set of harmonic number variables. It preserves correlations between
retained and eliminated coordinates, including Duschinsky mixing.

## Qualified capabilities and remaining molecular integration

* A supplied native finite Hamiltonian and doorway can be propagated directly
  with the existing engine. The adapter checks their coordinate identifiers.
* `finite_initial_harmonic_correlation` sums all final harmonic occupations
  analytically for a finite, potentially anharmonic initial state. It retains
  complex initial coefficients and the supplied two-surface Gaussian generator.
  This is explicitly a harmonic-final reference.
* A partial spectator trace returns a correlated Gaussian kernel, with the same
  covariance and polynomial-insertion contract as Raman. Independent tests
  compare it with explicit occupation sums.

The mixed native/harmonic model is connected through the existing conditional
doorways: `SparseSpectatorObservable.conditional_doorways(states)` reuses its
Gaussian contractions and caches. `spectator_doorway_correlation` computes

```
C(t) = exp(i E_i t) sum_s w_s exp(-i E_s t) <v_s|exp(-i H_A t)|v_s>.
```

The same `v_s = <s|V|i>` depends on the complete Duschinsky map. It cannot be
replaced by a common active packet and an independent bath probability.
Spectator energies include their zero-point energy; `H_A` contains only active
energies. The provider subtracts the oriented electronic gap from `E_i`.
Constant SOC channels with identical spatial dependence are combined by their
squared norms exactly, including complex couplings.

`prepared_spectator_correlation` reuses existing native eigenwindows or exact
disjoint spectral blocks. Packet projection is performed once; QuantumDynamics
uses analytic phases for an explicitly diagonal DIA Hamiltonian, avoiding a
Chebyshev expansion in an already diagonal representation. No new
diagonalization or global block eigenvector matrix is needed. Missing levels
have a separate nonnegative strength bound. Its Gaussian rate bound reuses
the existing spectral-window tail profile and excludes eigensolver errors and
roundoff. An incomplete eigenwindow is never labelled complete.

`reduce_spectators=False` retains one correlation column per spectator, and
`nonradiative_rate_from_correlation` returns `rates_by_channel_s_1`. These
values can feed the existing statistical sampler. Weights are never
renormalized. Sampling, support selection and omitted spectator-domain errors
remain the responsibility of the existing convergence protocol.

The analytic Gaussian partial trace is independently used to qualify this
conditional construction on a coupled native/harmonic test. Efficient direct
contraction of that analytic kernel with the full molecular propagator,
without a selected or sampled spectator support, remains unimplemented. No
dense active-pair kernel is built for the full molecular basis and no native
all-mode rate convergence is claimed.

Hot-band populations belong to the existing initial-state/ensemble provider;
this adapter accepts its individual doorways. Radiative cascade populations
remain in `SecularRadiativeDynamics`. Neither is reimplemented here.

## Controls

The rate reports a Gaussian finite-time tail bound and, when supplied by the
propagator, a propagation error bound. These exclude roundoff and do not certify
time-grid aliasing, source/basis completeness or potential-model convergence.
Time step, time window, operator space and initial ensemble need separate
convergence checks. Negative zero-energy densities cause an error rather than
being clipped.

Numerical qualification runs on Oracle with 128 GiB per process. The laptop
contains the same general source code and is used for editing and versioning.
