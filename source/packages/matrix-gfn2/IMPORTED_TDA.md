# Imported HF/DFT reference with a semiempirical TDA kernel

This separate Python API computes vertical singlet excitations from a real,
canonical, restricted closed-shell reference. The reference can come from HF
or DFT with any AO basis. No external electronic-structure package is imported
or installed by this API. Backend execution and conversion of backend file
formats remain the responsibility of the caller.

## Backend contract

Construct `ImportedTDAReference` from these arrays, or use
`ImportedTDAReference.from_npz(path)` with the same names as archive keys:

| Field | Convention |
| --- | --- |
| `model_id` | Nonempty provenance string; scalar Unicode in NPZ |
| `atomic_numbers` | `(natom,)`, integers 1 through 94 |
| `coordinates_bohr` | `(natom, 3)` |
| `ao_atom` | `(nao,)`, zero-based atom indices |
| `overlap`, `fock` | `(nao, nao)`, real symmetric; Fock in Hartree |
| `coefficients` | `(nao, nmo)`, canonical MO columns |
| `orbital_energies_hartree` | `(nmo,)` |
| `occupations` | `(nmo,)`, spatial occupations 0 or 2 |
| `dipole_ao` | `(3, nao, nao)` electronic dipoles; required for the selected spatial K filter and transition properties |
| `radial_second_moment_ao` | `(nao, nao)` matrix of `x^2+y^2+z^2`, in bohr squared; required for the default K filter; same origin as dipoles |

All AO quantities must have identical ordering and normalization. The constructor
checks `C.T @ S @ C = I` and `F @ C = (S @ C) * epsilon`. Rectangular MO
coefficients and positive semidefinite overlap are supported when a backend has
removed dependent AO combinations. The selected default uses `extent_exchange_damping=0.5`, so electronic dipoles
and the r^2 matrix are required. Explicit zero disables this filter and allows
older exports without r^2 data. No primitive Gaussian basis is needed.
Inputs are copied into read-only arrays. Open shells, complex orbitals and
fractional occupations are not supported in this initial version.

## Selection and response

`ImportedTDASelection(virtual_window_ev=..., maximum_virtuals=None)` retains all
occupied MOs and virtual MOs satisfying

    epsilon_virtual <= epsilon_HOMO + virtual_window_ev / HARTREE_TO_EV.

The window is measured from the HOMO, not from the LUMO. Optional
`maximum_virtuals` keeps the lowest-energy virtuals within that window.
Original MO indices are preserved in the result; virtuals are ordered by energy.
The caller should establish convergence by increasing the window and any cap.

In the retained canonical space the operator is

    A_ia,jb = (epsilon_a - epsilon_i) delta_ij delta_ab
              + 2 (ia|jb)_SE - exact_exchange (ij|ab)_SE,
    (pq|rs)_SE = sum_AB q_A(pq) gamma_AB q_B(rs),
    q_A(pq) = sum_{mu on A} (S^(1/2) C)_{mu,p} (S^(1/2) C)_{mu,q},
    gamma_AB = [R_AB^2 + eta_AB^(-2)]^(-1/2),
    eta_AB = (eta_A + eta_B) / 2.

Atomic hardnesses come from the existing resident Grimme sTDA table. Both
Coulomb and exchange use this same exponent-two Klopman--Ohno interaction;
`exact_exchange` multiplies exchange explicitly, including its on-site term.
This is a defined starting model, not a validated parametrization for arbitrary
HF/DFT orbitals, and is not the distinct J/K parametrization of `std2 -xtb`.
The DFT exchange-correlation response kernel is not evaluated.

The Fock matrix is used to certify the reference. Its canonical MO blocks are
represented by the imported eigenvalues; it is not added again to the gaps.
No AO or MO two-electron integrals are requested from the backend. No range
separation, xTB diagonal shift, a posteriori correction, perturbative CSF
selection, gradient or Hessian is applied by the response solver. The optional
extent correction described below is a separate operation. Existing calculators
and their derivative paths are unchanged.

The implementation stores orbital-pair atomic charges and the atom-pair Ohno
matrix, and applies the response operator without assembling the full TDA
matrix. It uses matrix-free Lanczos through SciPy, which is already a
MATRIX-numerics dependency. A deterministic starting vector has support across
symmetry sectors. Returned eigenvectors are checked for orthonormality and
their residuals are recomputed with the full operator. The shared Python
Davidson was not reliable for this new reference class in qualification;
the existing solvers and calculators are unchanged. Orbital-pair charge
storage still scales with `natom * (nocc^2 + nvirt^2 + nocc*nvirt)`.

## Python usage

```python
from matrix_gfn2 import (
    ImportedTDAReference, ImportedTDASelection, OhnoTDAParameters,
    imported_ohno_tda,
)

reference = ImportedTDAReference.from_npz("backend_reference.npz")
result = imported_ohno_tda(
    reference,
    parameters=OhnoTDAParameters.from_ground_state_exchange(0.25),
    selection=ImportedTDASelection(virtual_window_ev=30.0),
    roots=5,
)
```

The example uses a backend ground-state fraction of 0.25 and the selected
response scale of 0.9, giving a final response fraction of 0.225. The virtual
window is illustrative and must be converged. Record the backend method and
basis in `model_id`.

## Selected effective response exchange

The selected imported-reference model uses original radial Ohno interactions
for J and K, `a_response = 0.9 * a_ground`, and the spatial K filter with
`extent_exchange_damping=0.5`. Use
`OhnoTDAParameters.from_ground_state_exchange(a_ground)` to apply this
prescription once. It yields 0.18 for B3LYP (ground fraction 0.20) and 0.225
for PBE0 (0.25). The optional `response_exchange_scale` argument explicitly
overrides 0.9 for comparisons. This constructor selects no radial profile.

`OhnoTDAParameters(exact_exchange=a_response)` remains the low-level interface
for an **already effective** fraction and applies no scaling. Thus
`from_ground_state_exchange(0.25)` and `OhnoTDAParameters(0.225)` specify
the same operator. Do not feed an already scaled fraction to the ground-state
constructor. Solvers and virtual-space controllers use `parameters.exact_exchange`
directly, without applying a second factor.

For the earlier **undamped** reference (`extent_exchange_damping=0`),
unchanged B3LYP/PBE0 aug-cc-pVDZ Cartesian references and lambda=0.9 lower
the mean QUEST TBE error from 0.5273 to 0.4727 eV on 30 calibration states
and from 0.5143 to 0.4466 eV on 34 assigned external states. Eight of the
42 external target assignments remain unresolved. These are reused reference
sets, not a fresh blind validation. Two originally accurate external states
regress by more than 0.10 eV: cyclopropenone/PBE0 A2 n-to-pi* (absolute error
0.0066 to 0.2245 eV) and nitrobenzene/PBE0 A2 n-to-pi* (0.0024 to 0.2222 eV).
The mean improvement is not a guarantee for individual states or arbitrary
HF/DFT methods and bases.

Jointly lowering the SR coefficient while adding a 100% LR tail did not
improve over constant exchange on the sampled grid. No range separation is
adopted. Protocols and state-level results are in
`docs/reports/tda_long_range_exchange_20261022/README.md` and
`docs/reports/tda_joint_range_exchange_20261022/README.md`.

This is the imported-reference vertical response only. It changes no original
GFN2/TDA-xTB calculator, posthoc correction, gradient or Hessian route. The
imported API does not supply nuclear derivatives for this model.

Check `result.converged` and `result.residual_norms_hartree`. Amplitudes have
shape `(nroot, nconfig)`; `occupied_orbitals` and `virtual_orbitals` identify
each configuration. `retained_virtual_orbitals` and `discarded_virtual_count`
record the truncation. Requested roots are capped at the configuration count.
Negative excitation energies are returned unchanged.

When dipoles are supplied, normalized closed-shell singlet transition moments
are `sqrt(2) * sum_ia X_ia <i|mu|a>`. Oscillator strengths use
`(2/3) * omega * |transition_dipole|^2` for positive roots and zero for
nonpositive roots. Both property fields are `None` without AO dipoles.
`backend="auto"` selects `scipy-arpack`; `backend="dense"` explicitly assembles
the response matrix for small diagnostic calculations. Requesting every root
also uses the dense route. `maximum_subspace` controls Lanczos subspace size;
`maximum_iterations` bounds Lanczos update iterations. Failure to converge in
ARPACK raises an error. Numerical checks and the initial HF/DFT comparison are
run by `tools/qualification/qualify_imported_ohno_tda.py`; its PySCF and std2
requirements apply only to qualification, not production.

## Independent Coulomb and exchange radial profiles

`OhnoTDAParameters` accepts independent `coulomb_profile` and
`exchange_profile` arguments. Each defaults to `None`, preserving the original
Ohno interaction for that channel. In the singlet matrix,

\[
A_{ia,jb}=\Delta\epsilon_{ia}\delta_{ij}\delta_{ab}
+2\sum_{AB}q_A^{ia}\gamma^J_{AB}q_B^{jb}
-a_x\sum_{AB}q_A^{ij}\gamma^K_{AB}q_B^{ab}.
\]

The two contractions have different orbital indices; selecting independent
profiles does not change that distinction or the explicit exchange fraction.
For either selected channel `GaussianErfInteraction(a, b)` implements

\[
\gamma^X_{AB}=\eta_{AB}\left[
\left(1-\frac{2b_X}{\sqrt\pi}\right)e^{-a_Xt^2}
+\frac{\operatorname{erf}(b_Xt)}{t}\right],\qquad
t=\eta_{AB}R_{AB},\quad \eta_{AB}=(\eta_A+\eta_B)/2.
\]

The zero-distance value is evaluated analytically as `eta_AB`. Positive
`a` and `0 < b <= sqrt(pi)/2` give a positive nonincreasing radial function
with a `1/R` asymptote. This is a semiempirical scalar interaction; it does
not ask the backend for ERIs or implement range-separated ab-initio exchange.
The Coulomb interaction is still contracted with atomic transition monopoles:
if they vanish, no radial profile can supply the missing multipoles.

**The original Ohno interaction remains the operational choice for both J
and K.** The external validation does not justify adopting the new K profile.
No fitted preset is installed. The following is an experimental construction
for reproducing the K-only comparison:

```python
from matrix_gfn2 import GaussianErfInteraction, OhnoTDAParameters

parameters = OhnoTDAParameters(0.20, exchange_profile=
    GaussianErfInteraction(a=30.0, b=0.6667222473322341))
# coulomb_profile is None: J is the original Ohno interaction.
```

The legacy constructor still selects original Ohno J/K. Fixing `a=30` removes a parameter to
which these spectra are insensitive: changing it to 10 or 100 changes matched
training energies by at most 0.000066 eV. This does not establish a unique
physical value. The optional Coulomb-profile API was used for a separate
diagnostic comparison; its fitted profile is not adopted.

On 34 reliably assigned external states out of 42, the mean absolute QUEST
TBE error is 0.518 eV versus 0.514 eV for the original interaction and
0.515 eV for std2 on the same orbitals. The experimental K-only variant does
not pass the predeclared MAE-improvement gate and is rejected for adoption.
Eight assignments remain unresolved.
The virtual-window study used the diagnostic J/K variant, not the experimental
K-only variant: its 100 eV agreement must not be transferred to K-only without
checking. Validate a reduced space or use the full-space fallback.

Full protocol, state assignments, guards, basis checks and numerical contracts
are in `docs/reports/tda_jk_finalization_20261022/README.md`. These are vertical
spectral variants; no geometry derivatives or original GFN2 calculator behavior
are changed.

## Projected virtual valence spaces

`build_projected_tda_virtual_space` accepts two additional **one-electron**
arrays from the backend: the cross-overlap `cross_overlap` between the original
AO basis and a compact atomic basis, and that compact basis' `seed_overlap`.
Both must describe the same geometry and atom ordering as the reference.
Production does not require PySCF, MINAO, or Gaussian primitive data.

For the complete parent virtual coefficient matrix `Cv`, the projection is

    B = Cv.T @ cross_overlap @ seed_overlap^(-1/2).

The leading left singular vectors form `U`, with `U.T @ U = I`. The retained
AO coefficients are `Cv @ U`. The number `virtual_count` is explicit. For a
complete minimal atomic basis including core functions, a conventional valence
count is `nminimal - noccupied`; this is not inferred from tiny nonzero singular
values of projected atomic core/occupied functions. Singular-value and metric
rank thresholds are controlled by `relative_cutoff` (default 1e-8).

Optional `augment_virtual_window_ev` takes the orthonormal union with the low
canonical virtuals selected relative to the HOMO. It does not replace the
valence vectors or count the same direction twice.

```python
from matrix_gfn2 import build_projected_tda_virtual_space

space = build_projected_tda_virtual_space(
    reference, cross_overlap, seed_overlap,
    virtual_count=nminimal - noccupied,
    augment_virtual_window_ev=15.0,
)
result = imported_ohno_tda(
    reference, parameters=OhnoTDAParameters(exact_exchange=0.2),
    virtual_space=space, roots=12,
)
```

The numbers in the example are choices to qualify, not recommended defaults.
`selection` and `virtual_space` are mutually exclusive. Reference MOs remain
canonical and are not modified. Compressed virtuals generally are not eigenvectors
of the full Fock. The operator uses their **entire projected Fock block**:

    Fv = U.T @ diag(epsilon_virtual) @ U,
    A_ia,jb = delta_ij Fv_ab - delta_ab epsilon_i delta_ij + K_SE_ia,jb.

The same AO Löwdin partition and Ohno interaction are retained. This reduced
operator equals the Galerkin projection `(Iocc x U).T @ A_full @ (Iocc x U)`;
its construction and application do not require constructing `A_full`.
Rotating the reduced virtual basis transforms the Fock and kernel together.

For projected results, `virtual_orbitals` and `retained_virtual_orbitals` are
**compressed column labels**, not original MO indices. `occupied_orbitals`
retain their original MO indices. `result.virtual_space` records the parent
indices, rotation, full projected Fock and projection diagnostics. Use
`space.expand_amplitudes(result.amplitudes, noccupied)` to return amplitudes of
shape `(nroot, noccupied, nparent_virtual)` in the original canonical basis.
This mapping supports comparison of states, intensities and root ordering.
A fingerprint rejects applying a projected space to a different reference.

The MINAO qualification is run with
`tools/qualification/qualify_projected_tda_virtuals.py` against the previous
imported-reference data. It includes pure valence spaces, canonical enrichment,
an energy-only selection at equal virtual counts, basis covariance, Galerkin
identity and complete-space recovery. A compact valence space is not assumed
to retain diffuse states or to converge all valence excitation energies.

## Automatic canonical virtual convergence

`converge_imported_ohno_tda` enlarges the canonical virtual window without
changing the reference, Ohno kernel or exchange fraction. It compares states
in the shared canonical configuration basis, using phase-insensitive normalized
amplitude overlaps and a one-to-one maximum-overlap assignment. It does not
follow a root merely because it retains the same energy rank.

```python
from matrix_gfn2 import (
    TDAVirtualConvergencePolicy, converge_imported_ohno_tda,
)

policy = TDAVirtualConvergencePolicy(
    energy_tolerance_ev=0.01,
    strength_absolute_tolerance=1e-4,
    strength_relative_tolerance=0.01,
)
outcome = converge_imported_ohno_tda(
    reference, parameters=OhnoTDAParameters(exact_exchange=0.2),
    roots=6, policy=policy,
)
# Check outcome.converged and outcome.reason before using the convergence claim.
energies = outcome.excitation_energies_hartree
strengths = outcome.oscillator_strengths
```

In this default mode the requested set is the lowest six roots. When a new
state enters that set, the target set is updated and the stability counter
restarts. Returned energies are ordered by current energy rank.

For a particular identified state, supply **one-based root numbers from the
first window that can be solved**, and set `roots` to their count:

```python
outcome = converge_imported_ohno_tda(
    reference, parameters=OhnoTDAParameters(exact_exchange=0.2),
    roots=1, target_roots=(8,), policy=policy,
)
```

This follows the initially selected eighth root even when its energy rank
changes. The eighth root is an example, not a universal state assignment.
Fixed-state output preserves the input target order. Unresolved matches do not
replace a target with a nearby root; the last reliable anchor is retained.

The default window schedule starts at 40 eV, then uses 80, 160 and 320 eV
above the HOMO, with the complete supplied space as a final fallback.
Only enlargements adding virtuals provide stability evidence. Defaults require
all tracked states to satisfy:

- energy changes at most 0.01 eV;
- oscillator-strength changes at most `1e-4 + 0.01 * max(abs(f_old), abs(f_new))`;
- overlap at least 0.9 and a margin of at least 0.1 over competing assignments;
- two consecutive stable enlargements, followed by a confirmation window at
  twice the current window (bounded by the allowed maximum).

The absolute intensity tolerance handles dark states without dividing by
near-zero strengths. Ambiguous assignments reset stability and are not accepted
as converged individual states. Near-degenerate blocks are checked using
principal overlaps before individual-root diagnostics. If a complete block
is selected and resolved, convergence compares its sorted energies and summed
oscillator strength; it does not certify individual root identities within
the block. A partially selected unresolved block cannot be accepted on
subspace evidence alone; uniquely identified component roots still use their
individual energy and strength criteria. The default
clustering gap is 0.05 eV and the minimum principal overlap/margin are 0.9/0.1;
these are numerical tracking controls, not energy-model parameters.
The initial candidate pool includes six extra
roots and can grow up to 64 (`candidate_buffer`, `candidate_root_limit`).

By default `full_space_fallback=True` adds a final window containing all
virtuals supplied by the backend. If reached with resolved identities and
converged eigenvectors, there is no remaining virtual truncation within that
supplied reference. A convergence claim from window stability is an estimate,
not a mathematical bound on the uncomputed tail. Setting the fallback to false
can return `window_schedule_exhausted` without convergence.

An optional `maximum_virtuals` is a strict resource budget. If the next window
exceeds it, the controller returns `virtual_budget_exhausted`, not success;
it does not silently truncate the requested window to the budget. AO dipoles
are required by default. For explicit energy-only convergence set
`check_oscillator_strengths=False`.

`outcome.candidate_result` contains the last candidate spectrum, which may have
more roots than requested. `outcome.target_roots` maps the requested states to
that spectrum. The selected energy and strength properties expose only these
states. `outcome.history` contains windows, actual dimensions, candidate counts,
source/matched/current root numbers, overlaps, margins, energy/strength changes
and stop decisions. Changes refer to `source_roots -> matched_roots`, including
when a different state enters the current lowest-root set.

The fixed-reference controller does not implement geometry optimization or
derivatives. Its molecular and synthetic decision checks are reproduced with
`tools/qualification/qualify_tda_virtual_convergence.py`.

## Orbital size and optional posteriori extent correction

The optional `radial_second_moment_ao` field adds one monoelectronic AO matrix,
`<mu|x^2+y^2+z^2|nu>`. Its origin must match `dipole_ao`, whose sign is the
**electronic** dipole `-r`. Older NPZ exports without this field remain valid
when `extent_exchange_damping=0` is explicitly selected. No Gaussian basis, model-potential matrix,
two-electron integral, or external electronic-structure library is needed by
these helpers.

`imported_orbital_extents(reference)` gives each canonical MO centroid,
variance and radius in bohr:

    sigma_p = sqrt(<r^2>_p - |<r>_p|^2).

`imported_tda_extents(reference, result)` uses the full hole `X @ X.T` and
particle `X.T @ X` density matrices, including their off-diagonal elements.
It returns hole/particle radii, `sigma_particle/sigma_hole`, and
`max(1/sigma_hole - 1/sigma_particle, 0)`. Spatial size is a diagnostic, not an
automatic Rydberg assignment. State radii are invariant to coordinate origin,
rigid rotations, MO phases and consistent virtual-space rotations; single MO
radii may change under rotations among different orbitals. Canonical and
projected virtual-space results are supported.

The optional one-coefficient **empirical** energy model is, in atomic units,

    delta_omega = scale * a_response
                  * max(1/sigma_hole - 1/sigma_particle, 0).

The coefficient is explicit. No fitted value is selected automatically and
this is not part of the Grimme parametrization. The correction is applied
only to caller-selected zero-based root indices (an empty list is allowed):

```python
from matrix_gfn2 import (
    imported_orbital_extents, imported_tda_extents,
    TDAExtentCorrectionParameters, correct_imported_tda_extent_energies,
)

sizes = imported_tda_extents(reference, result)
# The separate posteriori alternative requires a response computed with
# extent_exchange_damping=0. The default K filter must be disabled first.
# Explicit diagnostic coefficient; caller supplies independently assigned roots.
post = correct_imported_tda_extent_energies(
    reference, result,
    parameters=TDAExtentCorrectionParameters(scale=0.5243166806164612),
    state_indices=rydberg_root_indices,
)
energies = post.corrected_excitation_energies_hartree
```

All returned energy arrays retain original root IDs and ordering. Unselected
energies are unchanged. The original reference and TDA result, eigenvectors,
oscillator strengths, Hamiltonian and nuclear derivatives are not modified.
Corrected energies need not remain in increasing numerical order. Strengths
remain properties of the original response, not a newly solved Hamiltonian.
This helper does not supply a geometry-optimization or Hessian correction.

The illustrative coefficient above is calibrated by minimax error on 16
previously inspected PBE0/aug-cc-pVTZ Rydberg states of pyridine, pyrrole and
imidazole. Across 21 assigned Rydberg targets of six molecules, the maximum
error changes from 1.2164 to 0.6994 eV (MAE 0.6157 to 0.2579 eV). Leaving an
entire molecule out gives a maximum predicted error of 0.8512 eV. Five control
states were excluded from coefficient fitting: their maximum worsens from
0.4846 to 0.6994 eV, with uracil increasing from 0.2240 to 0.6994 eV. The
coefficient is therefore experimental and opt-in, not a universal default for
arbitrary molecules, functionals or bases. No blind external validation is
claimed. Full records and numerical invariance checks are in
`docs/reports/tda_extent_correction_20261022/`.

## Default extent correction inside the two-electron exchange model

Use `extent_exchange_damping=beta` to put the spatial correction directly
inside K, before solving the response:

```python
parameters = OhnoTDAParameters.from_ground_state_exchange(
    0.25, extent_exchange_damping=0.5,
)
result = imported_ohno_tda(reference, parameters=parameters, selection=selection)
```

The default is beta=0.5 in both the direct constructor and
`from_ground_state_exchange`. The explicit argument in the example records
that choice. It is distinct from the posteriori coefficient. Explicit None
or zero preserves the undamped operator and requires no r^2 data. A positive beta needs
`dipole_ao` and `radial_second_moment_ao`. The filter is constructed solely
from the imported ground reference, without an excited-state label or energy.

Let R0 be the mean occupied orbital centroid. In the default
`extent_filter_space="full"` convention, diagonalize the complete supplied
parent virtual matrix
Qv of `(r-R0)^2`; q0 is the largest eigenvalue of the analogous occupied
matrix. In that basis the filter is

    F(q) = [1 + beta * max(q/q0 - 1, 0)]^(-1/4),
    K_new = (I_occ tensor F) K_Ohno (I_occ tensor F).

In the full space the screened virtual matrices become
`F @ screened_qvv[A] @ F`. For a canonical selection or orthonormal
subspace U, the actual retained contraction is `U.T @ F @ Q[A] @ F @ U`.
The implementation forms filtered coefficients `C_full @ F @ U` and their
atomic populations, avoiding a full four-index response tensor. J and Fock
are projected from the same complete Hamiltonian. Truncation is therefore a
Galerkin projection; `(U.T @ F @ U) @ (U.T @ Q[A] @ U) @ (U.T @ F @ U)`
would not be equivalent. This is a symmetric congruence acting only on K.
The matrix function is covariant under virtual rotations. A common ground
center provides a linear size operator; a state-dependent intrinsic radius
would not define a fixed linear response matrix. Two virtual vertices give
an asymptotic diagonal K attenuation proportional to 1/radius. Original
Ohno Coulomb, occupied orbitals, Fock and orbital gaps remain fixed. The
effective exchange fraction is scaled by 0.9 once, as before.

This route recomputes eigenvectors and strengths, so it differs from the
separate posteriori energy helper. Root identity is tracked by overlaps in
qualification. The diagonal and matrix-vector products use identical filtered
K matrices. Both canonical and projected virtual spaces are supported. The
operator exposes `virtual_extent_filter` (in the complete parent virtual
basis for the default convention) and `occupied_extent_bound_bohr2` when
enabled. `extent_filter_space="retained"` reproduces the earlier convention
that recalculated F after selection; it is retained for archived comparisons.
Both conventions agree in the complete virtual space. Full-reference filter
construction requires dense parent-virtual preprocessing; the imported route
does not inherit the native localized calculator's linear-scaling claim. No primitive basis, model-potential integrals or backend ERIs
are used; no external dependency is added. No imported-reference gradient
or Hessian is supplied. Combining this filter with the separate posteriori
extent correction is rejected to avoid an undefined double correction.

On the same 21 assigned PBE0/AVTZ Rydberg states of six previously inspected
molecules, maximum errors are 1.2164, 0.9088, 0.7685 and 0.7489 eV at beta
0, .25, .5 and 1. For the same 17 assigned valence states they are 0.5350,
0.5551, 0.5683 and 0.5914 eV. At beta=.5, uracil worsens from 0.2240 to
0.5956 eV. These results motivate the option but do not define a universal
beta for arbitrary references. We adopt beta=0.5 as the default model
choice after this comparison; the grid was not fitted, and no blind external
validation is claimed. `docs/reports/tda_extent_kernel_20261022/`
contains the state-level errors, coverage, invariance checks and API validation.

### Known limitation: uracil

**The selected default worsens the assigned uracil Rydberg vertical energy:**
with PBE0/aug-cc-pVTZ orbitals its signed error against QUEST increases from
+0.2240 eV without the filter to **+0.5956 eV at beta=0.5**. This exception is
retained explicitly; the reduction of the overall maximum error is not a
claim of uniform improvement. Set `extent_exchange_damping=0` to reproduce
the undamped model. The selected default applies to the imported-reference
vertical response; original GFN2/TDA-xTB calculators and their analytic
derivatives are unchanged.

### Default-filter diagnostics and basis convergence

The fixed-parameter Oracle study in
`docs/reports/tda_extent_stability_20261022/` separates the native derivative
model from the imported vertical response and follows states by overlap.
For the uracil AVTZ target, filtering changes the exchange contribution from
-1.536 to -1.174 eV; the gap expectation changes by only +0.009 eV.
The state overlap is 0.9868, so its +0.224 to +0.596 eV error regression
is retained as a kernel limitation.

Across the inspected pyridine, pyrrole and uracil bases, a 40 eV virtual
window gives a maximum tracked Rydberg shift of 0.0219 eV relative to the
full-space response; 80 eV reduces it to 0.00468 eV. These are sample
results, not new automatic window defaults or intensity-convergence bounds.
An extra even-tempered diffuse-shell uracil control changes the default
target by -0.1132 eV, despite near agreement between aug-cc-pVDZ and
aug-cc-pVTZ. Seven dependent AO directions are removed by the backend.
Additional lower diffuse roots appear, so root order and a fixed
variance-ratio ordinal cannot replace overlap tracking. Exact basis shells,
all comparisons and the remaining assignment/basis limitations are archived
in the report. No new energy parameter is introduced.

## Explicit state and subspace continuity

`track_imported_tda_states(source_reference, source_result,
candidate_reference, candidate_result, source_state_indices=indices,
cross_overlap_ao=metric, policy=TDAStateTrackingPolicy(...))` accepts canonical
or projected virtual spaces. All indices in this interface are **zero-based**.
Different references require an explicit cross-AO overlap matrix; model names
alone do not establish a common orbital basis. Results must contain converged,
orthonormal state vectors. No result or Hamiltonian is mutated.

The maximum-amplitude assignment seeds correspondences of energy clusters.
Singular values of their signed overlap matrices and two-sided external-span
competitors determine whether the whole block is resolved. Additional nearby
candidate states make a non-closed block unresolved. Within an accepted block,
a score combining amplitude overlap, normalized Hilbert--Schmidt hole and
particle density overlaps, and sorted NTO-weight overlap refines the mapping.
Density and NTO resemblance cannot override failed individual amplitude/margin
criteria. The NTO score compares occupation spectra; directional information
is supplied by the density overlaps.

`matches` explicitly separates `individually_resolved` from each block's
`resolved` flag in `subspaces`. A resolved span with unresolved component
roots is returned as such; candidate eigenvectors and energies are never
rotated. A controller step records the complete tracking result in
`step.tracking`. When a block converges, its individual strength changes in
the history remain diagnostic; the acceptance decision uses the block sum.
These APIs concern fixed-geometry response/basis continuity and do not replace
LINK/APOC's geometry-dependent electronic-state tracking.

### Qualified projection and adaptive outcomes

`docs/reports/tda_projection_adaptive_20261022/` records the fixed-parameter
Oracle comparison for uracil, pyridine and pyrrole. Complete-space energies
agree with the previous model within 7.18e-13 eV; canonical and noncanonical
Galerkin projections agree within 5.33e-14 hartree. The complete parent filter
is the new default; historical selection-dependent windows explicitly use
`extent_filter_space="retained"`.

All six default adaptive runs (lowest 5 and 12 states) completed via the
full-space fallback. This qualifies conservative stopping decisions, **not**
a cost reduction at the default energy/intensity tolerances. The controller
did not silently relax a criterion to accept a shorter window.

In the extra-diffuse uracil control the best scalar correspondence has overlap
0.899557 and is unresolved at the default 0.9 threshold. Its neighboring
block is nonclosed. Explicit 0.7 scalar acceptance is reported separately
without accepting the whole block or modifying the energies. Hole and NTO
occupation scores near unity do not compensate for changed particle density.
The rotated-orbital and controlled near-degenerate-state diagnostics are
archived alongside all outcomes.
