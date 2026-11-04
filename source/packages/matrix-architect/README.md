# matrix-architect

The authoritative scientific construction workflow is
[`docs/architecture/ARCHITECT_ZAFF_FORCE_FIELD_PROTOCOL.md`](../../docs/architecture/ARCHITECT_ZAFF_FORCE_FIELD_PROTOCOL.md).
It defines the ORACLE geometry fallback chain, CM5/Mayer/synthon provenance,
ZAFF0/ZAFF1/ZAFF2/ZAFF3 levels and the primitive-to-SONIC boundary.
The document is normative: every build and enrichment job MUST follow it (or
record an explicit machine-readable skip/block); raw CM5/Mayer data are
preserved and H-bond/X-bond depuration is deferred to use time.
The concise scientific description is available in
[`docs/science/architect_zaff_protocol.tex`](../../docs/science/architect_zaff_protocol.tex).

ARCHITECT is the single construction boundary for the internal **GFN-FF** and
**ZAFF** force fields in MATRIX. It assigns topology-dependent terms, checks
complete family coverage, binds the model to the certified ORACLE topology,
and serializes one immutable artifact. The independent `matrix-zaff` package
contains the shared sparse numerical kernels; it does not select or compile a
second model. KeyMaker selects the calculation and LINK owns the persistent
evaluation session.

For GFN-FF, incomplete term-by-term records are validation artifacts only.
Production construction must cover coordination, EEQ, bonded and non-bonded
repulsion, angles, ordinary and special torsions, D3, H/X bonds and bonded ATM;
`calculator_ready` is then sealed by ARCHITECT. `libxtb/GFN-FF` is an
independent parity oracle and is never an automatic runtime fallback.

ZAFF means **Zoomable Asymptotically-Correct Force Field**:
the resolution can grow from a compact transferable model to a molecule-specific
one while every level remains chemically readable and anchored to QM geometry,
gradient and Hessian observables.
ORACLE stops at perception, synthons, atom classes and the canonical PIC/B
source; SMITH supplies optional SONIC blocks; ARCHITECT owns model authoring.
Production energy, Cartesian gradient and Cartesian Hessian evaluation belongs
to the resident ZAFF backend.

## Initial-geometry providers

The frozen MATRIX initialization protocol couples the calculator to topology
quality. With a certified graph, ARCHITECT compiles the internal GFN-FF model
and LINK performs the inexpensive Cartesian preoptimization. Without a
certified graph, KeyMaker forbids force-field construction and uses the
autonomous resident MATRIX-GFN2 calculator until ORACLE can freeze a topology. SWITCH
remains only the deterministic SMILES-to-Cartesian seeder. The historical UFF
and external-xTB routes remain explicit reproducibility references; neither is
an automatic production fallback.

```bash
architect uff-compile molecule.xyzin --output molecule.uff.json
architect uff-point molecule.uff.json point.xyz --result point.json \
  --properties energy,gradient,hessian
```

Transferable seed construction is implemented in
`matrix_architect.seed_compiler`. The historical
`matrix_architect.seed_runtime` import remains available only as a compatibility
facade: every evaluator and persistent runtime object it exposes is owned by
`matrix_zaff.seed_runtime`.

The first compiled ZAFF contract contains an exact analytic Cartesian harmonic
E/G/H kernel plus a versioned local-SONIC anharmonic acquisition/parameter
model.  The harmonic kernel remains the stable backend boundary used by LINK
and SENTINEL while nonlinear terms are activated behind the same API.

## ZAFF force-field library

ARCHITECT writes a versioned, portable library consumed from the same resident
manifest used by `matrix-zaff`; ARCHITECT does not ship a second copy.  The
manifest separates compiled fields from immutable monomer references.  A
monomer reference contains one rigid geometry, canonical APOC intrinsic CM5
charges, the complete Mayer matrix, and one intrinsic atomic synthon per atom.
LCB25/LCB26 geometries can therefore be enriched progressively without
changing their geometry identity, and every MATRIX tool reads the same
electronic/covalent state.

A compiled-field entry
is accepted only when atom order, ORACLE covalent topology, charge and
multiplicity match exactly and the stored field passes its SHA-256 and schema
checks. Selection is deterministic by priority. Runtime consumers use the
read-only resolver in `matrix-zaff`. If no valid entry exists they select
`gfn-ff`; they never compile a field implicitly. Building a transferable or
molecule-specific field is an explicit ARCHITECT operation. Requested and
realized backends, the molecular fingerprint and the selection reason are
written into point-result provenance.

```bash
architect library create ./zaff-library
architect library add water.zaff.json --library ./zaff-library --id water-neutral
architect library list --library ./zaff-library
architect library resolve water.xyz --library ./zaff-library
matrix link optimize water.xyzin --run-dir run \
  --backend zaff --zaff-library ./zaff-library
```

`MATRIX_ZAFF_LIBRARY` supplies the default manifest or directory.  Resident
monomer references do not imply that a compiled field exists; a molecule
without a validated field still produces the explicit GFN-FF fallback.
`--force-field` remains an explicit, provenance-recorded
override for reproducibility of older workflows.

The registry also accepts fields generated from the transferable diagonal
library. MATRIX ships a resident, hash-pinned official AmberClassic
`gaff2.dat`, so normal compilation is local and requires no Amber installation
or download:

```bash
architect library seed molecule.xyz \
  --library ./zaff-library --output molecule-seed.zaff.json
```

An explicit GAFF file remains available as a reproducibility override. CM5
charges may likewise be supplied only together with the matching complete
Mayer bond-order file; otherwise ORACLE generates the whole paired level from
electronegativity charges and Pauling bond orders.

Stretches use the MM3-to-Morse synthon catalog; GAFF bends and torsions use
ZAFF's existing bond-order radial factors; and GAFF 12-6 non-bonded records are
converted to the Exp-PE modified-Morse polynomial after continuous
coordination/pi/charge synthon interpolation. The same values are initial
parameters and regularized priors when a complete molecule-specific field is
fitted. D4 and three-body repulsion are deliberately absent. The selected
paired charge level is the reference for two explicit response contracts.
`VARIATIONAL_QEQ_SQE` is the general constrained polarizable model.
`LOCAL_HBOND_CLOSED_FORM` consumes ORACLE's continuous hydrogen-bond state and
fitted donor/acceptor synthon responses without an iterative solve. It
separates fragment-neutral polarization from charge transfer and provides
sparse analytic first and second charge derivatives. Paired CM5 states or the
variational model itself can train this faster surrogate; in a finite box the
teacher and student must use the same versioned spherical/ellipsoidal boundary
correction. The variational route remains the fallback outside the calibrated
local domain. Both routes must pass analytic E/G/H and size-consistency gates.
The complete contract is
in `docs/specifications/zaff_parameter_library_v1.md`.

```bash
architect build molecule.xyzin --output field.zaff.json --workers 0
architect inspect field.zaff.json
architect parameters field.zaff.json --output field.zaff.txt
architect zoom field.zaff.json
architect validate field.zaff.json --json
architect evaluate field.zaff.json point.xyz --property energy \
  --property gradient --property hessian --zoom-level H3 --output point.json
```

`architect build --coordinate-chart primitive` freezes the primitive PIC
chart; `--coordinate-chart sonic` freezes the SMITH SONIC chart; `adaptive`
compares both. Bond stretches remain individual uncombined primitives in all
three cases. SONIC transformations act only on the other coordinate blocks
and consume SMITH CHARM coordinates directly for ring puckering.

A completed local-SONIC model is evaluated through the same public command by
supplying the frozen enriched geometry that defines its coordinate chart:

```bash
architect evaluate field-complete.zaff.json point.xyz \
  --xyzin molecule.xyzin --property energy --property gradient \
  --property hessian --output point.json
architect validate field-complete.zaff.json --xyzin molecule.xyzin --json
```

`--xyzin` selects validation against the analytic local-SONIC anharmonic E/G/H
path implemented by `matrix-zaff`. A harmonic
`--zoom-level` cannot be combined with it because the completed derivative
tensors belong to one frozen SONIC basis. Omitting `--xyzin` retains the stable
Cartesian harmonic path and its H0--H3 selection. Derivative validation uses
`1e-4 bohr` for the harmonic kernel and `5e-5 bohr` for the more strongly curved
physical SONIC kernel unless `--step-bohr` is supplied explicitly.

The `parameters` report is the stable human-readable view of the final model.
It lists atomic charges, nonbonded scales, every bonded equilibrium parameter
after the mandatory stationarity refit, torsional periodicities and the four
ZAFF zoom levels. It also prints, immediately after the SONIC description,
every SMITH periodic-coordinate record: reference value in degrees, domain,
periodicity, symmetry number, priority atom, barrier seed and validation
status. Ring endocyclic bends are deliberately absent from this table. Native
RPck components remain the analytic coordinates; derived ring phases are
reported and scanned but are not substituted into the local Hessian chart.
The JSON remains the machine-readable source of truth.

## Periodic coordinates and ring puckering

ARCHITECT copies the immutable SMITH periodic-coordinate contract into
`matrix.zaff.anharmonic.v1`. A torsion or improper produces a six-point
reference-centred prescan plan containing its periodicity, molecular symmetry,
candidate Fourier harmonics, barrier seed and escalation policy. The seed is
an acquisition prior, never a fitted barrier; promotion requires the scan or
Hessian validation recorded by the source contract.

The dimension-first contract assigns an N-membered ring `N-3` native RPck
components, one amplitude and `N-4` phases. CP Cartesian components are the
canonical descriptors, balanced triangular flaps are the SMITH/LINK
optimization chart, and newly constructed ZAFF fields use CHARM as their
smooth native ring-surface chart. Five-membered rings therefore use
two components and one large-amplitude pseudorotation phase. Six-membered
rings retain three native components and derive amplitude, azimuth and polar
angle. Rings with `N >= 7` are represented generically by an
`(N-3)`-dimensional puckering vector and real hyperspherical harmonics on
`S(N-4)`. Thus seven- and eight-membered rings use `S3` and `S4`,
respectively; molecule-specific truncations require their own provider-chart
rank and E/G/H validation. The versioned
`matrix.zaff.ring_dimension_contract.v1` record serializes this rule. For the priority-chair convention,
the polar value is exactly 0 degrees when the highest-priority atom is out of
plane and its opposite atom has the chair displacement. At a chair pole the
azimuth is a gauge and is serialized as zero with an explicit gauge status.
The scan generator consequently emits only one point at each pole.

The production force-field form follows the compact TEAM+ construction tested
by Charvati and Sun (2023). For every relevant bond or angle `x` and ring
torsion `phi`, ARCHITECT can compile
`(x-x0) sum[n=1..3] K_n cos(n phi)`. The three coefficients copy the local
mixed quadratic, cubic and quartic derivatives supplied by L0, while the
runtime removes the reference quadratic contribution already present in the
SONIC Hessian. This gives periodic bond--torsion and angle--torsion coupling
without introducing a stand-alone high-degree puckering polynomial.

For six-membered rings the preferred TEAM+ angular variables are the two
derived CHARM phases, not balanced flaps or the six individual endocyclic
dihedrals:
`PhiP1=atan2(RPck2,RPck1)` and
`PhiP2=atan2(sqrt(RPck1^2+RPck2^2),chair_sign*RPck3)`. ARCHITECT evaluates
these couplings through the equivalent normalized native-RPck expressions.
Azimuthal order `n` is represented by
`sin(PhiP2)^n cos(n PhiP1)` and its sine partner, so the energy, gradient and
Hessian remain analytic when `PhiP1` becomes a gauge at either chair pole.
The compact fit uses mixed valence--RPck Hessian blocks and reports its design
rank; a rank-deficient landmark set is never silently promoted.

`ZaffRingPuckeringSurface` provides the common executable five-/six-ring
continuation. Its production basis contains inverse-radial Fourier/spherical
harmonics plus positive `Q^2 exp(beta Q^2)` confinement, never a standalone
polynomial. Reference E/G/H are removed analytically. The molecule-independent
`tools/zaff/zaff_promote_team_plus_ring.py` fits the surface, extracts ORACLE typing,
enumerates every screened TEAM+ subset and promotes only subsets passing the
complete Cartesian E/G/H gate at the declared QM level.

For a heterocycle, coordinate periodicity and potential symmetry are separate
metadata. Oxane retains topological azimuthal periodicity three but molecular
symmetry number one; its surface therefore spans the full azimuth and may not
be forced into a threefold energy form. The current benchmark uses saved
MP2/cc-pVTZ Cartesian landmarks reprojected onto CHARM.

## LCB parameter transfer

`transfer_zaff_parameter` combines whole-environment Gaussian similarity with
local synthon agreement to select and weight LCB donors. It returns the value,
weighted uncertainty, effective donor count and complete provenance. Exact
duplicate donor occurrences are collapsed and a bounded median/Huber pass
downweights isolated parameter outliers after the Gaussian selection; this
does not change compatibility tiers or add a machine-learning model. Every
transferred force-field parameter is marked as requiring validation against
the target Hessian. `leave_one_out_parameter_transfer` forbids self-donation
and provides the corresponding validation experiment for an existing LCB
molecule. Saccharin is the first LCB26 coverage-gap application; phenol is the
tracked LCB25 leave-one-out geometry reconstruction.

Every new Hessian-backed build first reproduces the complete QM harmonic field after analytic
Gaussian-electrostatic and van der Waals subtraction.  The default H3 field
retains every quadratic coupling required for exact reconstruction; H0--H2 are
explicit harmonic truncation diagnostics rather than production ZAFF zoom
levels.  ZAFF1 retains fitted physical diagonal terms and only geminal X-H
stretch couplings. ZAFF2 restores the complete harmonic matrix. With no
displaced data, its bonds use fitted `r0` and corrected-QM `k2` together with
the ZAFF library `De` to define Morse continuations without changing the
harmonic Hessian. ZAFF3 replaces a stretch by a fitted SPF expansion when
higher derivatives are available; its default closure coefficient imposes
`sum(c_n) = De`, so the ZAFF dissociation asymptote is retained.

Every new build also embeds `matrix.zaff.anharmonic.v1`.  Its coordinate records
freeze the human-readable SONIC identifier, primitive expansion, reference
value, unit, finite-difference step and intended functional form.  Acquisition
uses one reference gradient and the four centered points `-2h,-h,+h,+2h` for
every SONIC.  Points are generated with LINK's finite soft predictor and
constrained hard corrector, with the full damped solver as a small-hard-step
fallback.  Independent points are submitted concurrently.

The local-SONIC Wilson matrix is audited before either the harmonic or
anharmonic compilation.  If an exact local pseudosymmetry makes a canonical
SALC first-order parallel to a native inversion coordinate (the exact-C3
ammonia geometry is the regression case), ARCHITECT keeps the out-of-plane
coordinate and regenerates only the complementary local combinations with the
rank-revealing SMITH reduction.  The fallback and singular-value margin are
serialized; global molecular symmetrization remains disabled.

The force-field coordinate catalog may be redundant. ORACLE automatically
promotes an acidic hydrogen to a transferable bicoordinated center when a
nearby acceptor enters the covalent transfer range. Both donor--H and
acceptor--H distances are retained and SMITH adds `r(D,H)-r(H,A)` as an
explicit proton-transfer coordinate. The promoted covalent domain excludes
the ordinary hydrogen-bond term. Unpromoted hydrogen bonds retain their C2
radial switch and angular envelope. SMITH derives a separate nonredundant
chart only for displacements, normal modes and GF analysis; it does not delete
terms from the ZAFF field.

An external Gaussian, ORCA or other driver can complete the model through the
JSON request/result boundary:

```bash
architect acquire-anharmonic molecule.xyzin field.zaff.json \
  --output field-complete.zaff.json --run-directory qff-run --workers 0 \
  --command "python qm_driver.py {request} {response}"
```

Each request follows `matrix.zaff.anharmonic.point_request.v1` and asks for
energy and Cartesian gradient.  The response follows
`matrix.zaff.anharmonic.point_result.v1`.  For a nonplanar signed out-of-plane
coordinate the driver receives one additional
`CONSTRAINED_OPTIMIZATION_THEN_ENERGY_GRADIENT` request with that local SONIC
fixed at zero.  The relaxed planar energy is the physical barrier used with
the observed minimum and curvature to determine the even sextic double well;
it is not a fitted pseudo-observation.

The five-point gradient stencil gives the full internal quadratic check,
all semidiagonal cubics `Phi[i,j,j]`, and the diagonal quartics; both matrices
are retained in the serialized model rather than discarded after constructing
the one-dimensional terms.  Primitive stretches compile to a
Simons--Parr--Finlan inverse-distance series matching the QM second, third and
fourth derivatives by default (`--stretch-model morse` and `inverse_power`
remain explicit alternatives),
primitive nonlinear bends to a three-harmonic cosine series, and signed
out-of-plane coordinates to an even polynomial in `sin(chi)`.  Collective
local SALCs retain an explicit local-SONIC anharmonic record rather than being
misinterpreted as a single bond length or valence angle.

The harmonic matrix obtained from the gradient stencil is reciprocalized
explicitly. The directed estimates `d g_i / d q_j` and `d g_j / d q_i` are
retained as `quadratic_matrix_directed`, while `quadratic_matrix` stores their
pairwise mean. The serialized reciprocity audit reports the RMS and maximum
antisymmetric differences, so numerical inconsistency remains visible.

The completed derivative block also exposes a local-SONIC physical/Taylor E/G
model executed by ZAFF. TRINITY can sample this backend at paired dimensionless Cartesian normal-mode
displacements and fit a standard full-cubic/semi-diagonal-quartic QFF without
forming derivatives of the Wilson matrix. ARCHITECT remains the constructor
and validator; ownership of resident evaluation belongs to ZAFF, while the
normal modes, QFF projection and VPT2/VCI remain with TRINITY.

ZAFF zooming acts above the invariant complete harmonic field. `ZAFF_BASE`
uses the cubic and quartic derivatives generated automatically by the physical
diagonal Morse, inverse-distance, cosine and Fourier functions.
`ZAFF_REFINED_SELECTED` adds higher diagonal terms and selected switched
extra-diagonal terms. `ZAFF_REFINED_FULL` retains the complete acquired
cubic/quartic support. The harmonic matrix is identical at all three levels;
the Taylor-only truncations are diagnostics and are not ZAFF production levels.
The TRINITY fit record includes its component amplitude, training and holdout
seeds, exact `+Q/-Q` pairing, radial sampling interval, SVD convention and
energy/gradient weights. This makes point selection reproducible rather than
an implicit fitting choice.

Semi-diagonal normal-mode quartics are screened by a separate, chemically
local contract.  In particular, an X--H stretching/H--X--Y bending
`K_iijj` is proposed only when the PED identifies both characters and traces
them to the same X--H bond.  The Gaussian normal QFF can validate and rank the
proposal, but is never used to make it.  The report also gives `K_iijj/8`, the
direct first-order fundamental contribution in dimensionless normal
coordinates.  Large torsion--stretch quartics are excluded from this
polynomial rule and delegated to the existing large-amplitude scan protocol.
The standalone audit is reproducible with `tools/zaff/analyze_zaff_iijj.py`.

The production acquisition starts from a still smaller hypothesis: for an
XH2 center it tests only the local totally symmetric stretch
`(r_XH1+r_XH2)/sqrt(2)` against the native H--X--H bend.  Four simultaneous
`(+/-s,+/-b)` energy-gradient points determine the residual `K_ssbb`; no
displaced Hessian is calculated.  The eight gradient components also resolve
the four cubic nuisance derivatives and the two odd cross-quartic diagnostics.
Only `K_ssbb` is eligible for promotion, and only when its smallest gradient
signal exceeds the post-fit residual by a factor of five.  Thus the decision
is made before and independently of any reference normal QFF.  The executable
workflow is `tools/zaff/zaff_symmetric_stretch_bend.py`; its JSON contract is
`matrix.zaff.two_mode_quartic.v1`.

The current harmonic ARCHITECT route fits one cross curvature for every pair
of X–H stretches on the same heavy atom. Its general ZAFF term is

`E_ij = K_ij (r_i-r_i0)(r_j-r_j0) exp[-(alpha_i(r_i-r_i0))^4] exp[-(alpha_j(r_j-r_j0))^4]`.

Only the Hessian fit supplies `K_ij`; without an accepted fit it is zero.
The existing seed Morse exponents supply `alpha_i` and `alpha_j` as damping
scales. The mixed curvature at the reference is exactly `K_ij`, while the
term tends to zero if either X–H distance diverges. The fourth-power switch
leaves the local expansion through fifth order unchanged. The fitted field
serializes the coefficients and evaluates the nonlinear correction with
analytic energy, gradient and Hessian, subtracting its Cartesian quadratic
Taylor term to avoid counting the fitted Hessian twice.

The corresponding physical ZAFF functional form is now available as
`ZaffStretchStretchAnglePotential`.  It uses centered, unit-slope Morse
coordinates

`S_i(r) = [1-exp(-alpha_i (r-r_i0))]/alpha_i`

and the coupling

`S_i S_j [K_ij + A(cos(theta)-cos(theta0)) + B(cos(theta)^2-cos(theta0)^2)]`.

`K_ij` therefore remains the exact harmonic stretch--stretch coupling, while
`A` and `B` copy independently `Phi_ijtheta` and `K_ijthetatheta`.  Because
both radial coordinates vanish at the reference, the coupling introduces no
spurious reference energy, gradient, or pure-angle curvature.  Scalar linear
angles are rejected and remain assigned to the two-component bend contract.
The complete parameters and copied derivatives have a JSON round trip.
`compile_zaff_stretch_stretch_angle_term` obtains them directly from the
multilevel tensors, and `attach_zaff_coupled_terms` embeds the record in the
ZAFF force-field JSON.  The local-SONIC ZAFF path evaluates only the anharmonic
remainder because `K_ij` is already present in the base quadratic matrix; the
serialized audit therefore makes double counting explicitly false.

## Multilevel same-geometry derivative ladder

`matrix.zaff.multilevel_derivatives.v1` implements three valid same-geometry
contracts: `L0+L1`, `L1+L2`, and `L0+L1+L2`. The default stencil is only
`0,+/-h` for each SONIC:

- L1 always supplies energies and Cartesian gradients;
- L0, when selected, supplies analytic Cartesian Hessians and their gradients;
- L2, when selected, supplies energies and one reference gradient.

An optional extended L2 stencil adds energy-only `+/-2h` points. It is needed
only for a diagonal quartic at L2. The L1 diagonal quartic already follows from
the three L1 energies and three L1 gradients through a Hermite formula.

The L0 gradient is required even though it is not used to refit the harmonic
field: away from equilibrium it contracts the sparse analytic Wilson-matrix
derivative to remove `sum_i g_i B'_i` before the Cartesian Hessian is inverted.
With `L0+L1`, the compiler copies the L0 harmonic/full-cubic/semi-diagonal-
quartic backbone and replaces harmonic diagonals, `Phi_i,j,j`, and diagonal
quartics by L1 values. With `L1+L2`, it constructs the complete L1 harmonic
matrix and semidiagonal cubics and promotes diagonal cubics to L2; unavailable
all-distinct cubics and off-diagonal semidiagonal quartics remain explicitly
absent. With all three levels both rules apply. No source derivative is fitted.

Nonstationary reference geometries are handled explicitly.  L1 gradients are
detrended by `g_L1(0)`, and L2 energies by
`E_L2(0)+g_L2(0).q`.  Thus an L0 geometry activates both corrections, an L1
geometry only the L2 correction, and an L2 geometry only the L1 correction.
`build_zaff_multilevel_requests`, `compile_zaff_multilevel_derivatives`, and
`cartesian_hessian_to_internal` expose the portable request, compiler, and
analytic chain-rule layers.

SURVIBTM-inspired optional surface bases are exposed through
`ZaffSpecialBasis`: Taylor, Simons--Parr--Finlan, bounded Ogilvie, complete or
half-angle Fourier, and associated Legendre functions in either `x` or
`cos(x)`. Legendre/Fourier products support compact orientational surfaces.
They complement rather than replace the default ZAFF Morse and Exp-PE forms.
The serialized contract is `matrix.zaff.special_basis.v1`.

## Weighted and exactly constrained surface fitting

`ArchitectLinearFitProblem` and `fit_architect_linear_surface` provide the
common numerical layer for energy and property surfaces. Observation weights
are a recorded product of a base array and named components, for example
`point`, `property`, and `level`. Coefficients copied from QM derivatives are
fixed before fitting. Linear equality constraints are then eliminated through
an SVD null-space parameterization, so they are exact rather than high-weight
pseudo-observations.

The rank-revealing, column-scaled SVD reports singular values and cutoff,
effective rank, identifiable-subspace and full condition numbers, named null
combinations, leverage, standardized weighted residuals, parameter
identifiability, and absolute as well as reduced-chi-square covariance and
correlation matrices. Zero-weight points remain visible in the output. The
portable schemas are `matrix.architect.linear_fit.problem.v1` and
`matrix.architect.linear_fit.result.v1`.

```bash
architect fit-surface surface-fit.json --output surface-result.json
```

## Property surfaces

ARCHITECT uses the same basis, weighting, exact-constraint and rank-diagnostic
contract for scalar, vector and tensor observables. A property problem declares
one common analytic basis and any number of named components (for example
`x,y,z` for a dipole or `xx,xy,xz,yy,yz,zz` for a symmetric polarizability).
Each component has its own observations, units, weights, fixed coefficients and
symmetry constraints. Qualified labels such as `x::q1` additionally permit
exact relations between components.

The multi-component problem is solved as one block system. Consequently its
SVD report detects both deficient basis functions within a component and
unidentifiable cross-component combinations. Zero-weight observations remain
in the residual audit. The serialized result contains the coefficient matrix,
predictions, residuals, covariance/correlation matrices and complete linear-fit
provenance. Analytic values and first/second coordinate derivatives are
obtained by contracting the coefficients with basis values, Jacobians and
Hessians.

```bash
architect fit-property-surface dipole-problem.json \
  --output dipole-surface.json
```

The schemas are `matrix.architect.property_surface.problem.v1` and
`matrix.architect.property_surface.result.v1`.

The H2O integration benchmark uses the equilibrium dipole and Cartesian dipole
derivatives in a Gaussian FCHK file. TRINITY transforms them to mass-normalized
normal coordinates, promotes symmetry-forbidden numerical noise to exact zero
constraints, fits the ARCHITECT surface, and contracts it to harmonic IR
intensities. Its independent Gaussian references are reproduced within
`1e-6 km mol-1`:

```bash
matrix trinity ir-from-fchk tests/fixtures/gf/h2o.fchk \
  --output h2o.ir.json --surface-output h2o.dipole-surface.json
```

```bash
matrix vpt2-vci --zaff-force-field field-complete.zaff.json \
  --zaff-xyzin molecule.xyzin --run-dir runs/zaff-qff --zaff-fit-only
```

At build time ARCHITECT constructs harmonic Pareto families from the same QM
Hessian and nonbonded subtraction: redundant ORACLE PICs, nonredundant SMITH
SONICs and, when SMITH supplies ring-special rows, a local hybrid. SONICs are
built with local SALCs only; global molecular symmetry is deliberately disabled
to avoid excessive delocalization. The compiler
compares independent diagonal scalars plus retained couplings at equal
projected-Cartesian Hessian eigenvalue error. It therefore chooses PIC,
local-SONIC or hybrid from measured economy, not from a fixed coordinate
preference.

The four ordered harmonic compression tiers are quantitative: `H0`, `H1` and `H2` are the smallest
PIC, local-SONIC or hybrid models reaching 5%, 2% and 1% spectral error,
respectively;
`H3` is the smaller complete harmonic ceiling. A basis transition is allowed
when the equal-error audit justifies it. Every level stores its parameter and
coupling counts, eigenvalue errors, sign mismatches and equilibrium-gradient
residual.

`architect validate` probes a nonstationary geometry and checks the analytic
gradient against centered energy differences and the analytic Hessian against
centered gradient differences. The resulting
`matrix.architect.derivative_validation.v1` record is stored in every frozen
benchmark. The force-field JSON also embeds
`matrix.architect.runtime_contract.v1`: atom order, units, selectable zoom
level, E/G/H properties and batch support are therefore explicit rather than
being conventions private to LINK.

ORACLE and SMITH always provide native out-of-plane coordinates at
tricoordinate centers. Gaussian-style improper dihedrals and the expansion of
composite torsions are export-only translations applied when a G16 `.gjf` file
is requested; they never enter the ARCHITECT fit.

## Experimental local ring hybrid

The hybrid is intentionally local. For each ring it retains SMITH's `RDef` and
`RPck` Fourier blocks, removes only the ORACLE PIC rows represented by those
blocks, and keeps all other native PICs. In particular all stretches, non-ring
valence angles and exocyclic torsions remain available. If the special block
does not close the vibrational rank, covered PICs are restored one at a time in
canonical order and the restoration is serialized. A ring-special coordinate
is never discarded in favor of exocyclic angles.

The hybrid enters the equal-error audit only when a ring-special block exists;
acyclic systems therefore retain the original PIC versus local-SONIC contract.
The model records the special rows, replaced PIC labels, retained PIC count,
rank and any restoration. This is the initial harmonic hybrid; nonlinear
assembly remains behind the unchanged resident ZAFF E/G/H boundary.

Coupling scores are evaluated in parallel. `--workers 0` selects up to 32
available CPUs; an explicit positive count is useful on a compute server. BLAS
thread counts should be kept at one when many ARCHITECT workers run concurrently
to avoid oversubscription.

LINK can use the same model instead of a QM program:

```bash
matrix link driver-run molecule.xyzin --run-dir run \
  --driver-command "python sentinel.py {request} {response}" \
  --backend zaff --force-field field.zaff.json
```

The runtime field is loaded by `matrix-zaff`; ARCHITECT is not imported during
point evaluation.

The complete glycine build/inspect/zoom example is documented in
`benchmarks/zaff/README.md`.

## Frozen nonlinear ZAFF contract

The transferable prior is deliberately more physical than a diagonal
harmonic field:

- Electrostatics uses one indivisible observable level. APOC CM5 is used only
  together with the complete Mayer matrix.
  `architect apoc SOURCE --xyzin molecule.xyzin` runs APOC explicitly;
  `architect build molecule.xyzin --apoc-source SOURCE` runs it as the first
  build step. If either member is missing, ORACLE generates both
  electronegativity charges and Pauling bond orders and ZAFF consumes that
  complete fallback; cross-level mixing is forbidden. A different explicit
  definition must likewise provide and label both members.
- UFF pair scales are transformed to the Exp-PE modified Morse radial form.
- Bond stretches use the complete 180-entry MM3 Morse catalog of Shannon
  *et al.*; continuous ORACLE synthons interpolate missing environments using
  separate first-pi and second-pi (pi-pi) indices.
- If no QM Hessian is available, the interpolated Morse prior is used
  directly.  If a Hessian is available, its depth is retained while the local
  equilibrium displacement and exponent are refitted to the target gradient
  and curvature.
- A valence-angle prior is multiplied by the normalized continuous bond-order
  product `b_ij(r_ij) b_jk(r_jk)`.  Nonlinear primitive angles use a smooth
  three-harmonic cosine series that copies the QM quadratic and cubic
  derivatives exactly; linear angles remain a two-component bending problem.
- Native signed out-of-plane coordinates use an even polynomial in
  `sin(chi)`.  It represents either a planar single well from `k2,k4` or a
  symmetric double well from the nonplanar minimum, its curvature and a
  separately calculated planar barrier.
- A torsional Fourier series is multiplied by all three bond orders,
  `b_ij(r_ij) b_jk(r_jk) b_kl(r_kl)`.  Breaking any one of the three bonds
  therefore removes the torsional energy, gradient and Hessian continuously.

ORACLE perception is rerun for every geometry.  MM3 interpolation uses all
element-compatible prototypes, so the weights and resulting parameters follow
the continuous synthons without a discrete nearest-type transition.  The
reported atom type is a diagnostic label, not a switch in the energy.
Triple-bond prototypes use `(PI_INDEX, PI_PI_INDEX)=(1,1)`, whereas double and
aromatic prototypes use `(1,0)`; ARCHITECT never reconstructs these indices
from a private integer bond class.

The force field is split into a topology-independent collective field and
topology-dependent local corrections. Gaussian-charge electrostatics and the
rationally switched Exp-PE term are evaluated for every pair; a discrete graph
never deletes or rescales a 1-2, 1-3 or 1-4 contribution. Stretching, bending,
torsional and out-of-plane terms are fitted to the residual energy derivatives.
The distance-only switch and its first two derivatives are analytic.

Topology is used only for a serialized diagnostic: the compiler checks that
the radial non-bonded switch is nearly off through both 1-2 and geminal 1-3
distances, then reports 1-4 behavior. A strained structure with no radial gap is
flagged for explicit bending or torsional scans instead of silently changing
the collective energy rule.

The bond-order factors are not ad hoc switching functions: they are the smooth
strong/weak Pauling blend used by ORACLE, with the PL1 covalent-radius state
frozen and the value normalized to one at the reference geometry.  The
analytic kernels expose energy, internal gradient and internal Hessian.
Completed fields use the physical Morse, inverse-power/SPF, cosine-angle and
out-of-plane diagonal kernels through the public `--xyzin` validation path and
the resident ZAFF backend. Coordinates
that still require a torsional scan or another unresolved global model retain
their explicit local Taylor fallback; the execution record reports both counts.

The production harmonic policy retains the exact residual-QM Hessian.  Every
diagonal and every required off-diagonal term is immutable.  ARCHITECT orders
off-diagonal terms using the VPT2-like mixing index
`|Kij|/Deltaij` and relative second-order shift
`|Kij|^2/(Deltaij sqrt(|Kii Kjj|))`, with a declared near-degeneracy gap floor.
This ordering defines sparse harmonic diagnostics, while H3 retains the
complete support and reproduces the Hessian by construction. Production ZAFF
zooming starts from this invariant H3 quadratic field and changes only the
anharmonic content.

`--coupling-policy exact` is the default.  The `hybrid` and `refitted` policies
remain explicit diagnostic truncation controls.  Chemical relations are recorded after selection, and
significant torsion--torsion residuals are promoted to two-dimensional Fourier
scans.  SONIC terms are labelled as stretch--stretch, stretch--bend,
out-of-plane, torsional, ring or fragment-pose combinations so a retained term
can be judged chemically instead of remaining an anonymous matrix element.

## Non-bonded radial transformations

ARCHITECT can replace an existing UFF 12-6 pair by the Yang--Sun--Deng
Exp-PE modified Morse form without changing its well depth, minimum distance
or radial Hessian at the minimum. The UFF mixing rules remain the source of
the pair parameters; only the radial functional form changes.

```python
from matrix_architect import uff_pair_to_exppe

# Carbon--oxygen pair; energy in hartree and distance in bohr.
potential = uff_pair_to_exppe(6, 8)
value = potential.derivatives(potential.r_min)
```

General Mie `m-n` wells and arbitrary wells specified by `epsilon`, `r_min`,
`k2` and optional `k3` use `mie_minimum_derivatives`, `exppe_from_minimum`,
`morse_from_minimum` and `double_exponential_from_minimum`. The Exp-PE mapping
preserves `k2`; the double exponential preserves both `k2` and `k3` when two
positive real exponents exist. `morse_from_zero_crossing` implements the
historical Abraham--Stolevik minimum/zero-crossing prescription.

## Nonperiodic CPCM and mean-field confinement

ZAFF can attach one spherical or triaxial-ellipsoidal CPCM reaction field and
one mean-field `UvdW` correction to the same confinement geometry. The latter
uses homothetic ellipsoidal shells, a species-resolved Morse baseline, signed
Gaussian shell corrections and a compact analytic C2 switch. Energy,
Cartesian gradient and Hessian-vector products are analytic; the frozen field
also runs in the vectorized rigid-torsional SENTINEL screen.

`VdwWarmupCalibrator` consumes blocks of accepted Cartesian geometries from
the SENTINEL warm-up. It measures exact shell number densities, applies a
diminishing iterative-Boltzmann update, and selects the Morse--Gaussian basis
by held-out ridge validation. Its checkpoint is restartable. Calling
`freeze()` is an irreversible state transition: only the resulting
`FROZEN_FOR_PRODUCTION` record may be attached with
`attach_calibrated_vdw_confinement`; subsequent density adaptation fails
closed.

The resident TIP3P-FB reference is available without OpenMM or an AMBER data
installation:

```python
from matrix_architect import (
    tip3p_fb_parameters,
    tip3p_fb_uvdw_confinement,
)

water = tip3p_fb_parameters()  # rigid geometry, charges and O--O LJ data
field = tip3p_fb_uvdw_confinement(center_bohr, semiaxes_bohr, rotation)
```

The record also retains the published degree-10 NPBC UvdW polynomial.  ZAFF
uses its audited compact-C2 Morse--Gaussian translation on the oxygen
reference site; source and translated profiles can be compared with
`tip3p_fb_source_uvdw_kj_per_mol`.

```python
from matrix_architect import (
    VdwWarmupCalibrator,
    VdwWarmupConfig,
    attach_calibrated_vdw_confinement,
)

calibrator = VdwWarmupCalibrator(
    center_bohr,
    semiaxes_bohr,
    rotation,
    solvent_site_types,
    VdwWarmupConfig(
        target_number_density_bohr3={"O": rho_oxygen, "H": rho_hydrogen},
        shell_edges_rho=shell_edges,
        temperature_kelvin=298.15,
    ),
)
for accepted_geometry_block in sentinel_warmup_blocks:
    provisional = calibrator.observe_block(accepted_geometry_block)
    # Use provisional.provisional_confinement in the next warm-up block.
production_model = attach_calibrated_vdw_confinement(
    zaff_model, calibrator.freeze()
)
```

## Persistent large-system nonbonded services

All MATRIX callers can use `DynamicMolecularFMM` and
`DynamicCenterPairList` through the public `matrix_architect` API.  The
molecular multipole tree supports local energy differences and analytic target
gradients; accepted moves update one root-to-leaf path and rebuild
deterministically after a leaf crossing or full state restore.  The center
pair list tracks accepted cell crossings and returns a conservative candidate
set for switched short-range interactions.

Accelerated applications should use the shared fail-closed validation
functions rather than duplicating benchmark logic:

```python
from matrix_architect import (
    audit_accelerated_reference,
    audit_incremental_state,
)

speed_and_accuracy = audit_accelerated_reference(
    accelerated_batch,
    direct_batch,
    maximum_error=2.0e-3,
    minimum_speedup=1.25,
    unit="kcal/mol",
)
consistency = audit_incremental_state(
    incremental_moments,
    exactly_rebuilt_moments,
    maximum_relative_error=5.0e-12,
    resynchronize=restore_exact_moments,
)
```

The acceleration audit warms both callbacks, checks shapes and finite values,
uses median timings, and raises before production when either its accuracy or
speed gate fails.  The consistency audit can validate and replace any
incrementally maintained numerical state.  The explicit-solvent reference
driver uses these same public functions for water-pair, end-to-end proposal,
and periodic CPCM-moment checks.


## Joint native fitting at minima, transition states and path geometries

`build_native_reference_seed` prepares the original inverse-power ZAFF functions
from physical radial catalogue priors and a frozen primitive set. It needs no
reference Hessian and therefore applies equally to a minimum or a saddle.
The nonbonded evaluator and its population provenance are retained unchanged.
An unfitted seed is not a qualified potential.

`NativeReferencePoint` declares each point's role (`minimum`, `transition_state`
or `path`), available observations, source hash and coordinate realization.
`fit_native_reference_data` fits these references together to one
`NativePicField`. A single minimum is the same procedure with one reference.
Negative saddle curvatures remain negative; Hessians are never made positive.
Missing derivative entries are omitted. The coordinate realization includes
both first and second geometry derivatives, including the gradient term in
nonstationary Hessian transformations. Cubic and quartic constraints differentiate
the native model columns, without acquiring new reference energies.

All fitted products retain the original native functions and support-bond
switches. Intermediate energies and derivatives are then evaluated directly by
the resident field, without calling the source potential. Calibration residuals,
source hashes, coefficient basis, design and portable field are saved together.
Passing calibration does not replace independent validation of withheld
higher derivatives, spectra or rates. Use a fresh output directory per frozen
fit; retain existing designs and molecular acquisition checkpoints.


## Bond formation and cleavage in reactive valence fields

`ValenceConnectivity` declares each reactant/product connectivity and builds
its own native primitive set. A newly formed bond therefore introduces its
angles and torsions as well as its radial term. The transition state constrains
the adiabatic potential; it is not itself a reactant/product connectivity.

`ValenceExchangeCoupling` provides the off-diagonal energy, Cartesian gradient
and Hessian for resident multistate EVB evaluation. For two valence states,

    H_ab = A_ab * product(w_b(R_b), b in symmetric_difference(B_a, B_b))
    w_b(R) = expit(-alpha_b * (R - Rc_b))

Both forming and cleaving bonds enter the product. Bond occupancy increases
under compression, and the exchange vanishes when either changing bond is
separated. Reversing the state labels gives the same adiabatic eigenvalues.
The coupling also vanishes when every changing bond is separated.
Reference distances, widths and amplitudes are explicit model parameters;
the API supplies no universal values or fitted molecular accuracy claim.
They must be determined and checked together with both diagonal ZAFF fields.
This mechanism is separately tested from calibration of molecular potentials
and rates; passing the coupling tests does not qualify OH + CH4.
