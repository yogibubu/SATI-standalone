# Quantum dynamics in TRINITY

The dynamics engine accepts Hamiltonian and property **actions on vectors**.
It supports sparse matrices, `scipy.sparse.linalg.LinearOperator` objects and
compiled normal-ordered operators in occupation coordinates. The representation
can therefore be shared by vibrational, vibronic and rovibrational applications.

## Three operations

- `QuantumDynamics.propagate`: Chebyshev propagation of one packet or a batch.
- `QuantumDynamics.correlate`: spectral correlations without storing the packet
  at every time. The supplied spectral enclosure determines a uniform Bessel
  tail bound. Work and memory budgets raise explicit errors.
- `SecularRadiativeDynamics.sample`: event-driven spontaneous-emission cascades
  in a supplied energy eigenbasis. Dipoles are applied only for visited states.
  A bounded shared source cache avoids repeated evaluations. Independent random
  streams make results reproducible across worker counts. Reported uncertainties
  are trajectory standard errors.

Hamiltonian and level energies use cm⁻¹, propagation uses fs, radiative waiting
uses s and dipoles use atomic units. Photon rates contain the Einstein
frequency-cubed factor. A frequency-fourth weighting would describe emitted
energy per unit time instead of photon probability per unit time.

## Effective properties and a common reference

`contact_dynamics_property((O0, O1, O2), S1, S2)` evaluates

```
Oeff0 = O0
Oeff1 = O1 + [O0,S1]
Oeff2 = O2 + [O1,S1] + [O0,S2] + 1/2 [[O0,S1],S1]
```

by composition of actions. It retains interference through
`sum_dynamics_operators(orders)`. Generators are anti-Hermitian, properties
Hermitian. Mechanical and electrical derivatives can be retained separately.
When external intermediate states are required, supply all factors on the
augmented space and pass an embedding of the retained space: the engine projects
**after** the compositions. The provider owns the perturbative partition,
resonance masks and completeness of that augmented space.

`gvpt2_quantum_dynamics(result, effective_bare_dipoles_au, coordinate_id=...)`
reuses existing assigned GVPT2 blocks to rotate the supplied effective properties
into the energy basis. The block transformations do not form a global square
basis map. This adapter requires effective dipoles consistent with that GVPT2
model; bare Taylor derivatives alone are insufficient to specify them.

`LadderDynamicsOperator(polynomial, basis)` supplies a compiled finite-basis
operator action from the existing `LadderPolynomial`. Exact ladder factors,
complex phases and adjoints are retained. It constructs no transition-pair
inventory and introduces no magnitude threshold. The finite projection is
explicit and must not be inserted between contact-transform factors when their
intermediates lie outside that space.

## Example API

```python
from matrix_trinity.quantum_dynamics import (
    gvpt2_quantum_dynamics, stationary_emission_correlation,
    correlation_emission_spectrum,
)

model = gvpt2_quantum_dynamics(
    gvpt2_result, effective_dipoles_in_bare_basis,
    coordinate_id="my-consistent-modal-reference",
)
correlation = stationary_emission_correlation(
    gvpt2_result.energies_cm, source_index,
    model["dipoles_eigenbasis_au"], times_fs,
    coordinate_id="my-consistent-modal-reference",
)
spectrum = correlation_emission_spectrum(correlation, gaussian_sigma_cm=10.)
cascade = model["radiative"].sample(
    initial_population, trajectories=1000,
    frequency_edges_cm=frequency_edges, seed=73, workers=4,
)
```

The spectrum example starts in one eigenstate. The cascade accepts an incoherent
initial population. Its secular emission model resolves final states, retains
hot bands, overtones and combination transitions, and assumes no instantaneous
thermalization after emission. Dark states and photons outside the requested
frequency range are reported. The quantum propagator also accepts coherent
packets; coherent open-system evolution requires a corresponding dissipative
model beyond this secular cascade interface.

## Prepared-input command

```sh
python -m matrix_trinity.quantum_dynamics \
  --input prepared.npz --output dynamics.npz \
  --time-step-fs 1 --time-points 4097 --gaussian-sigma-cm 10 \
  --trajectories 1000 --seed 73 --workers 4
```

Required NPZ arrays: `energies_cm`, `initial_population`,
`frequency_edges_cm`, scalar strings `coordinate_id` and `basis_representation`.
Property input can be either `dipoles_au` of shape `(3,N,N)` from an existing
reference, or compiled normal-ordered data: `basis`, integer `creation` and
`annihilation` arrays of shape `(terms,modes)`, and `dipole_coefficients_au`
of shape `(3,terms)`. Repeated monomials are summed.

With `basis_representation="energy_eigenbasis"`, properties act in the energy
basis. With `"harmonic_occupation"`, also provide the assigned basis map as CSR
arrays `basis_map_data`, `basis_map_indices`, `basis_map_indptr`. That map rotates
the properties; it does not recompute the eigenstates. The command submits no
QC calculations.

The command spectrum uses the most populated initial eigenstate, while its
cascade uses the full input population. Time-window tail, negative spectral
area, zero-time strength, Fourier grid and trajectory uncertainties are saved.
A broadened photon-rate spectrum differs slightly from rates evaluated at exact
transition frequencies; both quantities are identified separately.

## Qualification and molecular application

Tests compare packet propagation with independent matrix exponentiation,
contact actions with explicit matrix algebra, compiled ladder actions with the
occupation-transition algebra, spectra with resolved Gaussian lines, and
sampled cascade times with analytical rate sums. Parallel and serial trajectories
use identical random streams.

`benchmarks/gvpt2_rates/qualify_quantum_dynamics.py` checks a previously acquired
molecular rate column against TD spectral reconstruction. This qualifies the
propagation and transform on molecular data. Production application to a new
force field additionally requires its native effective-dipole action to pass a
resolved-reference comparison; importing an existing rate column does not
supply or qualify that action.

## Method references

- Tal-Ezer and Kosloff, *J. Chem. Phys.* **81**, 3967 (1984),
  [Chebyshev wavepacket propagation](https://doi.org/10.1063/1.448136).
- Dalibard, Castin and Mølmer, *Phys. Rev. Lett.* **68**, 580 (1992),
  [stochastic wave-function treatment of emission](https://doi.org/10.1103/PhysRevLett.68.580).

The implemented cascade is the energy-resolved secular population limit;
the general coherent wave-function jump formalism is a methodological reference.

### Link to the hybrid rate model

`radiative.rate_reference(occupations, sources, maximum_pairs=...)` exports the
selected complete source columns as the existing `GVPT2RateReference` object.
It can be passed to the current algebraic/hybrid rate fitters. Each source is
computed through dipole-vector actions; individual moments are not acquired.
Unselected sources remain unmeasured. A zero computed rate is not interpreted
as a symmetry prohibition.

`maximum_source_evaluations` bounds the total number of operator evaluations,
including repeated evaluations after cache eviction. The default is 4096 and
can be changed explicitly. Exceeding it raises rather than reporting an
unfinished simulation as a converged cascade.

### Native reduced contact fields

`reduced_field_dynamics_properties(prepared_field, derivatives, augmented_basis,
coordinate_id=..., retained_rows=...)` connects the existing
`ReducedTransitionField` generators and modal D1/D2/D3 derivatives directly to
compiled ladder actions and factorized contact properties. It avoids expansion
of the full effective property and retains coherent perturbative orders.
`include_electrical=False` selects the linear dipole surface with the same
mechanical generators. Both choices therefore use the same nuclear dynamics.

The global-generator adapter is checked against independently expanded reduced
coefficients on a curvilinear field. Its generator builder covers the
nonresonant domain. Resonant calculations use the native masked construction
below with the direct exclusions of the supplied GVPT2 result.

### Native resonant effective dipole

```python
model = gvpt2_quantum_dynamics(
    gvpt2_result, field=force_field, derivatives=dipole_derivatives,
    coordinate_id="my-consistent-modal-reference",
    native_controls={
        "maximum_work": 100_000_000,
        "operators": {"cached_columns": 32, "maximum_input_sources": 128},
    },
)
```

The derivatives D1, D2 and D3 use the same modal reference as the force field.
`NativeResonantProperty.from_gvpt2` imports its direct resonance exclusions;
connected polyad membership alone does not remove external denominators.
It evaluates the normalized wave-operator expression through second order,
including the internal normalization, the pole-first finite remainder of
rectilinear resonant fundamentals, and mechanical/electrical interference.
The three orders remain available through `provider.moments(initial, final)`.
Setting D2 and D3 to zero selects mechanical corrections alone, without changing
the Hamiltonian, energies or resonance space. Other vector and tensor properties
can use the same native provider with their own derivatives.

For the direct entrywise resonance mask `M`, cubic Hamiltonian `C`, quartic
Hamiltonian `H2`, and harmonic gaps `d_mn = E_n - E_m`, the construction is

```
D = M * C
A = (1-M) * C/d
B = (1-M) * (H2 + C A - A D)/d
N = M * (A† A)
Oeff0 = P0
Oeff1 = P1 + A† P0 + P0 A
Oeff2 = P2 + A† P1 + P1 A + B† P0 + P0 B
           + A† P0 A - (N P0 + P0 N)/2
```

Here `*` denotes entrywise masking; juxtaposition denotes operator composition.
Excluded entries are removed before division. The selected-moment provider's
existing denominator tolerance and finite-remainder convention are retained.

Native scalar tensor convolutions are shared across property components.
They retain every allowed tensor shift, including external intermediates
outside the retained eigenstate support. There is no added virtual excitation
cutoff. Full integer shifts verify lookup hashes, so lookup collisions cannot
change the result and the number of modes is not limited by a machine word.
The Hermitian action uses the same less-excited-endpoint convention as the
acquired bare-moment reference. Assigned GVPT2 eigenvectors then rotate these
complex amplitudes coherently. Probabilities are never used to infer phases.

Only occupied input columns are evaluated. Components share a bounded column
cache and the accumulated property packet. The native batch kernel releases
Python's interpreter lock, allowing independent packet requests from cascade
workers to execute concurrently. Work and packet-memory limits raise explicit
errors without changing the perturbative expressions. Recursive kernels compile
once per process; they are not persisted as recursive machine-code addresses.

`benchmarks/gvpt2_rates/qualify_native_resonant_property.py` compares separate
complex orders with archived molecular moments and compares coherently mixed
M/M+E rates for a complete selected polyad. This is a bounded qualification of
the native property, distinct from convergence of a new full cascade or a
whole-calculation performance benchmark.

```sh
python benchmarks/gvpt2_rates/qualify_native_resonant_property.py \
  --case PATH_TO_FIELD_AND_DIPOLE --checkpoint PATH_TO_POLYADS_AND_BARE_MOMENTS \
  --output qualification --pairs 12
```

### Restarting from acquired probabilities

`SecularRadiativeDynamics.from_rate_columns(energies_cm, columns,
coordinate_id=...)` reuses complete downward source columns in s⁻¹. Each
entry is `source: (target_indices, rates)`. An explicitly empty column is a
known terminal or dark state. A missing column remains unknown: sampling
stops at its first visit and reports `unresolved_first_hit_probability` and
its sampling error. The emitted spectrum and arrival time then describe the
observed prefix; they must not be interpreted as a fully resolved trajectory.
These probabilities do not determine transition phases or coherent dynamics.

The restart check is reproducible with:

```sh
python benchmarks/gvpt2_rates/restart_radiative_dynamics.py \
  --checkpoint PATH_TO_ACQUIRED_Q2_RESULTS --trajectories 20000
```

It checks all lower endpoints, reuses the initial population and energies,
and compares jump sampling with independent acyclic probability propagation.
No new transition moments are acquired. This check does not qualify the
native resonant effective-dipole action needed to extend the acquisition.
