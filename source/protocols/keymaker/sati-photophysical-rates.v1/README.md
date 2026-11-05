# Photophysical rate protocols

Five installed KEYMAKER protocols cover fluorescence, phosphorescence, internal
conversion, intersystem crossing/triplet decay and photodissociation. The
entry points are `sati.rates.<process>` under `prediction_assignment`.
Their executable preparation owner is `matrix_trinity.photophysical_protocols`.
Preparation selects providers and returns a plan; it does not run a QM job.

| Process | Essential electronic data | Preferred provider, subject to actual capability evidence |
| --- | --- | --- |
| Fluorescence | Electric dipole; its derivatives for HT | Gaussian, then ORCA/TURBOMOLE |
| Phosphorescence | Complex scalar SOC matrix, optical dipoles, state tracking; derivatives of the **mixed optical dipole** for HT | ORCA, then certified alternative |
| Internal conversion | Complete nonadiabatic nuclear operator, including the declared momentum/divergence convention | TURBOMOLE or ORCA only when the required complete operator is certified |
| Intersystem crossing / triplet decay | Direct SOC, and SOC derivatives for linear or symmetry-activated coupling | ORCA, then certified alternative |
| Photodissociation | Dissociative potential, asymptote, optical operator, state tracking | Provider qualified for the selected electronic states; also requires a certified outgoing-channel solver |

Gaussian analytic Hessians are preferred for nuclear fields when supported by
the selected method. A preferred backend is a ranking, not a declaration that
all its methods or versions implement a property. Separate providers may supply
fields and operators only after matching state identities, frames and units.
Do not infer a nonradiative coupling from a phosphorescence transition dipole.
A bare NAC vector does not certify a complete nonadiabatic operator.

Each request has schema `matrix.trinity.photophysical_rate_request.v1`, process,
method, basis, state_model, nuclear_model, operator_order and rate_methods.
`backend_profiles` must contain installed name/version/executable and explicit
`capability_records`: feature, certified, method, basis, state_model, evidence.
Availability or a program name alone is insufficient. Measured cost estimates
can rank eligible providers; incomplete providers are rejected before ranking.
Missing data raises `MissingPhotophysicalCapability` with a structured report
and `submitted_jobs=0`. No alternate method, constant operator, zero derivative
or numerical derivative is inserted to bypass the stop.

The finite acquisition plan and exact electronic keywords require explicit
user authorization. Existing fields and checkpoints take priority. The nuclear
hierarchy is harmonic, DVR with harmonic bath, VPT2 or VCI; existing SATI owners
perform TI and TD. Frozen analytic DVR potentials and their fixed acquisition
budgets remain mandatory. Rates require the same Hamiltonians, operator, gap,
temperature, channel populations and broadening in both representations.
For radiationless rates an energy-conservation model is explicit. For
photodissociation a cross section becomes a rate only with a photon flux.
The preflight stops if the outgoing-channel solver has no supporting evidence.

Current anthracene acquisition uses seven already authorized geometries, one
job at a time, eight processes and 32 GB on MAC-STUDIO. This is an application
budget; none of the five general protocols assumes a molecule, host name,
number of modes or electronic-state count.

`theory_by_role` may explicitly provide a different method/basis/state_model
for nuclear fields and electronic operators (for example a composite nuclear
field and a less expensive SOC derivative). This is never inferred as a
fallback. Capability checks use the declared theory of each role. The planner
also rejects requested nuclear TI/TD adapters that are not implemented; it does
not relabel a harmonic calculation as VPT2, DVR or VCI. In particular the
current strict VPT2 nonradiative owner is TI only, and the common-VCI radiative
owner is TI only. Completing an unsupported adapter is a separate code task.

The KEYMAKER command-line entry point accepts
`--photophysical-rate-request request.json --photophysical-rate-plan plan.json`.
It resolves the registered process protocol, invokes the scientific preflight,
and returns exit status 2 with the missing-data report if any required capability
is absent. A successful preflight returns status 0 and submits no jobs; exact
job inputs and authorization remain subsequent stages.

### Energy-specific TI and coherent promoting modes

`harmonic_vibronic_ti(..., selection='targeted', selection_rate_at_cm=0.,
selection_windows_cm=[[-150.,150.]], ...)` selects a positive density at the
energy-conservation point. It uses the same-model TD density to stop and a
damped Gaussian energy-moment match to guide the existing operator C2 tree.
The requested fraction concerns that point; window profiles and global
omitted-strength bounds are still reported separately. TD time-grid convergence
must be checked independently. Physical line strengths are never reweighted.
An explicit energy bias keeps the existing ranking and checkpoint reusable;
changing the automatic bias requires a distinct selection checkpoint.

Nonradiative rates consume numeric line columns directly, including C++ rate
reductions. All SOC components retain the existing shared Hermite recurrence;
no component-specific cache or millions of Python line dictionaries are needed.

`matrix_trinity.soc_linear.coherent_soc_pairs` accepts independent singlet and
triplet phase/overlap evidence from APOC for each displacement. It preserves
signs and interference between multiple modes in one spin channel, and rejects
ambiguous tracking or an untransported spin frame. Missing phase information
cannot be reconstructed from a symmetry-zero SOC value. The magnitude-only
orthogonal-channel path remains available for that restricted case.

`rank_soc_promoting_modes` uses an explicitly supplied, same-model positive
nuclear response matrix and includes coherent interference in a
Cauchy--Schwarz omission bound. It does not certify modes lacking electronic
derivatives or nuclear responses, and does not acquire additional QM points.

### Converged TI assignments with a TD rate reference

For targeted TI, `selection_energy_range_cm` can restrict completed candidate
states before exact recurrence evaluation. It never prunes their descendants.
The requested windows must lie strictly inside the interval. The global source
norm and the maximal line-shape tail provide an upper bound on the excluded
window density. Candidate and evaluated-transition budgets are independent in
this mode. Existing unfiltered checkpoints can resume with a narrower interval;
a previously filtered domain cannot be widened silently. Numeric line columns
can remain on disk while checkpoint replay releases redundant in-memory copies.

For a zero-temperature source, `zero_temperature_gaussian_blocks` identifies
independent blocks of the final-mode ground-state Gaussian. Each incoherent
channel must belong to one block; significant cross-block coherent properties
are rejected. Small discarded terms have explicit tolerances and diagnostics.
The original analytic TD must verify the factorization before molecular use.
Each block is then solved with the existing adaptive TI and native recurrence.
`convolve_gaussian_block_lines` sums all finite block-state products using the
existing discrete-measure convolution, preserving positive strengths. Paired
energy bins control interpolation; source-norm deficits and line-transport
intervals are reported separately. This applies to the source ground state,
not automatically to a thermal population.

`gaussian_block_line_strengths` reconstructs the unbinned strength of any
available full-mode assignment and retains its original final-mode indices.
Block line tables preserve all included occupations, energies and strengths;
the full Cartesian product need not be stored. TD supplies the rate, while
TI supplies numerically converged state assignments and a spectrum check.

### Complete frozen-reference IC operators

A derivative-coupling vector may be combined with an explicitly supplied scalar
Born--Huang coefficient to define a frozen-reference operator. The general
`matrix_trinity.nonadiabatic_operator.frozen_born_huang_operator` transports both
coefficients to dimensionless source coordinates and retains the adjoint block.
It never assigns a missing scalar to zero. A zero reference scalar can be
certified with `symmetry_forbidden_scalar` when supplied electronic characters,
operator symmetry and reference invariance forbid its matrix element. For
example an inversion-even kinetic scalar vanishes between g/u states at an
inversion-symmetric reference. The evidence must identify those electronic
states and verify the isotope-resolved reference symmetry.

This certifies the declared frozen-reference approximation, not a complete
coordinate-dependent NAC field. The approximation, local nuclear measure,
source/target convention and evidence enter the capability record. Nonzero
scalar terms or coordinate-dependent fields require their own electronic data.
At zero temperature a harmonic source satisfies p_j|0> = i q_j|0>; the resulting
position-source representation may reuse existing Gaussian-block TI services
only after checking its correlation against the original momentum operator.
