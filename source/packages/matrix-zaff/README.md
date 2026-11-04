# matrix-zaff

`matrix-zaff` is the resident runtime for compiled ZAFF force fields.  It is
the canonical implementation of the **Zoomable Asymptotically-Correct Force
Field** contract.

Historical schema identifiers are accepted only at the read boundary through
`matrix_zaff.compatibility.normalize_legacy_zaff_payload`; every writer emits
`matrix.zaff.*`, and there is no legacy package, import namespace or duplicate
runtime.
It is deliberately separate from ARCHITECT:

- ARCHITECT constructs, fits, validates and publishes fields;
- ZAFF loads immutable compiled artifacts and evaluates them;
- MATRIX tools consume ZAFF through the common potential-backend contract.

The runtime implements the compiled Cartesian harmonic and zoom-level field
through three distinct paths: energy, energy plus analytic gradient, and
energy plus analytic gradient plus analytic Hessian. It also owns the analytic
radial and bonded forms, Gaussian electrostatics, persistent pair lists and
FMM operators, charge response, induced-dipole polarization, ellipsoidal
confinement, CPCM and the joint variational cavity.

Transferable diagonal seed artifacts use the same backend. Their energy-only
entry point avoids Cartesian coordinate transformations and all force
construction; E+G reuses persistent pair-list/FMM state, and E+G+H provides
the analytic Hessian required for stationary-point and frequency work.

Runtime lookup is read-only. A missing library entry is reported to the caller
as a GFN-FF fallback; ZAFF never invokes ARCHITECT implicitly. The optional
`fmm` extra enables the large-system Laplace backend.

`matrix_zaff.seed_runtime` is the single implementation of transferable seed
E, E+G and E+G+H execution. `matrix_zaff.library` is likewise the single
read-only resolver and publishes `matrix.zaff.force_field_resolution.v2` with
the canonical backend names `zaff` and `gfn-ff`. ARCHITECT consumes those
objects directly and adds only authoring operations.

The same resident manifest also exposes immutable monomer references through
`list_zaff_monomers` and `resolve_zaff_monomer`.  Each reference stores the
rigid geometry together with one canonical APOC CM5 vector, the full Mayer
matrix, and intrinsic atomic synthons. The definitive electronic reference is
PBE0/def2-TZVP plus APOC and may be supplied by any validated QM code. PBE0
has one functional definition across codes, while def2-TZVP is the adopted
cost--accuracy compromise. All MATRIX consumers therefore use identical
stored arrays.

Rigid construction can also translate a resolved, frozen GFN-FF non-bonded
atom pair with `FrozenGFNFFNonbondedPair`.  The translation preserves the
source well depth, minimum distance and curvature in the ZAFF Exp-PE form.
The source GFN-FF model damps its attractive dispersion contribution while
retaining short-range repulsion. Its damping is not carried into ZAFF Exp-PE.
It intentionally requires atom-pair parameters resolved after GFN-FF topology,
charge and coordination preparation: the underlying force field does not
define an equivalent element-only van der Waals table. Electrostatics and the
explicit hydrogen/halogen-bond terms remain separate objectives.

Rigid construction exposes `uff` and `gfnff` as explicit parameter-source
alternatives. GFN-FF charge and coordination metadata are frozen separately
for each covalent fragment. Only declared covalent edges contribute to the
coordination used in this translation: other fragments and intramolecular
non-covalent contacts are excluded. Both sources are compiled to the same
undamped runtime Exp-PE table, and the conversion preserves the source well
depth, minimum and curvature.

The definitive ZAFF model does not retain GFN-FF coordination-number typing.
It discretizes the complete MATRIX atomic synthon with versioned component
thresholds; intrinsic CM5 charge is already one component and is never applied
again as a separate parameter correction. If the available CM5 vector comes
from a hydrogen-bonded reference, the resident polarization/charge-transfer
table first removes that response. The resulting intrinsic charge fixes the
synthon type for the whole run. Geometry-dependent H-bond response changes
only electrostatics and cannot retype the atom or change its Exp-PE pair.

Catalog thresholds, ordered prototypes, pair matrices, schema and version are
serialized as one artifact. A complete UFF-derived matrix is available as the
universal prior; ARCHITECT may replace it with a calibrated synthon-pair
matrix. Directional H-bond/XB libraries are independently versioned. They can
either be obtained by fitting the residual left after CM5 electrostatics and
isotropic Exp-PE at identical geometries, or be compiled directly from the
resident ZAFF directional parameter set (`compile_zaff_directional_exppe_contact`),
which preserves the prescribed radial minimum and curvature in the ZAFF
Morse-polynomial form. Both routes expose explicit CM5 and synthon modulation;
missing typed residuals fail closed. The shared construction batch reports
electrostatic, isotropic and directional energies separately and its total
includes all three.

Non-bonded, H-bond and halogen-bond radial factors all use the same undamped
Exp-PE evaluator and squared-distance lookup bank. Directional terms use
normalized dot products and small integer powers; no inverse trigonometric
function is evaluated in the construction loop.

`ExpPEPotential` denotes the analytic form. `DampedExpPEPotential` remains
as a compatibility name for the same undamped form; its epsilon and distance
are the realized well depth and minimum. Immutable type-pair alpha matrices compile an
`ExpPELookupPlan` once, so repeated batches avoid runtime type discovery. The
resolved matrices are accepted by the NumPy and Torch accelerator paths, while
the analytic evaluator remains the numerical reference. Pair lists retain the
short-range schedule and point-charge construction retains the Laplace FMM
route.

The permanent hierarchy is stored in `data/zaff_levels.json`: ZAFF0 is the
rigid exploration projection; ZAFF1 is the fitted physical diagonal field with
only dissociating geminal X-H stretch couplings; ZAFF2 restores the complete
ARCHITECT harmonic field and uses Morse stretches with dissociation energies
from the ZAFF library; ZAFF3 adds fitted inverse-distance, cosine and Fourier
terms and validated couplings through fourth order. These native variables
encode dissociation, bounded angles and periodicity, so the fourth-order field
needs substantially fewer mixed couplings than a QFF in Cartesian normal
modes. Ordinary hydrogen bonds use a C2 radial switch and angular envelope;
an ORACLE-promoted transferable hydrogen instead belongs to a bicoordinated
covalent domain and its ordinary hydrogen-bond term is excluded. GFN-FF elemental, charge, coordination, bond
order and ring rules initialize the seed; validated ring terms remain part of
the primitive field and SMITH supplies their SONIC view. The legacy aliases
Z0/Z1/Z2/Z3, ZAFF-fast, ZAFF-flex and ZAFF-full remain accepted aliases.
Those coefficients are compiled onto immutable four-synthon keys, so runtime
evaluation never reapplies separate charge or coordination corrections.

## Planar explicit--continuum interfaces

`matrix_zaff.interfacial_pcm` implements the first INTERPHASES contract for
surface and interface studies. Every explicit molecular charge site lies
above a frozen plane in the dielectric-one region; the residual substrate
below the plane has a scalar dielectric constant greater than or equal to
one. The continuum contribution is the exact planar image Green function for
the normalized Gaussian charge densities used throughout ZAFF. Every
explicit--image pair uses `erf(beta_ij r_ij)/r_ij`; a point-image `1/r`
surrogate is forbidden. The model includes the regular vacuum limit, zero
contrast at dielectric one, and the conductor limit.

The record schema is `matrix.zaff.interfacial_pcm.v2`. Homogeneous CPCM and
interfacial PCM are alternative reaction fields and cannot be attached to the
same seed model. Energy, gradient, Hessian--vector products and directional
response operators share the analytic Gaussian-erf kernel. The blocked CPU
path is the reference implementation. Until a Gaussian-capable accelerated
image kernel is available, FMM requests fall back explicitly to that reference
path and report the fallback; backend selection cannot reintroduce point-image
physics. The full
Cartesian Hessian remains an analytic, vectorized dense assembly because its
output itself contains \(O(N^2)\) elements. The compression tolerance,
crossover and direct block size are serialized in the model record through
`PlanarSpectralCompression`. The same symmetric kernel participates
variationally in QEq/SQE response. An explicit exclusion gap prevents
molecular sites from crossing or overlapping the continuum dividing surface.
Multiple HVP directions share one charge expansion and use FMM multi-density
dipoles; the portable C++ backend also evaluates all requested directions in
one native call. Fixed-geometry charge-response solves compile an immutable
cluster hierarchy. Large operators use a symmetric H-matrix with exact near
blocks and tolerance-controlled low-rank far blocks, while smaller operators
retain the exact dense matrix. Conformer populations use a memory-bounded
vectorized geometry batch. Energy/gradient, one HVP, batched HVP and
charge-response each have an independently calibrated crossover.
Per-call telemetry records the
selected and actual backend, elapsed time, requested accuracy, audit error and
fallback. The global ZAFF Hessian builder accumulates the interfacial columns
through batched analytic HVPs. `calibrate_planar_spectral_crossover` measures
and fingerprints the four local direct/FMM crossovers; calibration records are
accepted only on the matching architecture; those crossover records are
advisory while the exact Gaussian fallback is active. `physical_diagnostics` audits
reciprocity, kernel sign, interface clearance, and translations parallel to
the plane.

The NumPy/Python implementation remains the scientific reference. When the
optional portable C++ extension can be built, ZAFF selects it automatically
for sufficiently large direct Gaussian-electrostatic workloads. Selection is
reproducible with `MATRIX_NUMERICAL_BACKEND=python|compiled|auto`; an
unavailable extension falls back to Python unless
`MATRIX_NUMERICAL_BACKEND_STRICT=1` is set. The native pilot has distinct
energy, energy-plus-gradient and analytic Hessian-vector paths. The
architecture-local comparison is available through
`python tools/benchmarks/benchmark_zaff_native.py`. Bond-order-damped bends and torsions
use the same native Cartesian E, E+G and analytic Hessian-vector contract;
`tools/benchmarks/benchmark_zaff_local_terms.py` reports the Python/native crossover.

Transferable all-pair QEq/SQE fields are stored as an O(N) channel generator.
The exact variational flows are eliminated in site space and pair quantities
are regenerated in bounded chunks, so neither the artifact nor the runtime
materializes N(N-1)/2 channel records. Explicit arbitrary channel sets remain
available as the parity reference. Use
`tools/benchmarks/benchmark_charge_response_storage.py` for the storage audit.

## Organic-solvent library

`matrix_zaff.solvent_library` provides the four organic GLOB solvents and the
published SPC-water fit from
Mancini *et al.*, *Chem. Phys. Lett.* **625** (2015) 186-192:
chloroform, carbon tetrachloride, methanol, acetonitrile, and SPC water. The
published boundary polynomials are retained as auditable source data and refitted to
ZAFF's compact-C2 Morse plus signed-Gaussian form. The resulting frozen
spherical or ellipsoidal confinement supplies separate energy-only,
energy-gradient, and energy-gradient-Hessian paths for genetic algorithms,
Monte Carlo, and molecular dynamics.

The molecular recipes use MATRIX's resident GAFF2 catalogue and molecule-
specific charges; SPC retains the fixed charges of the published model. The
boundary acts on each molecular center of mass and propagates analytic
derivatives to all atomic sites. TIP3P-FB remains the default resident water
model, while `spc_water` is available for exact reproduction of the 2015 GLOB
fit.
