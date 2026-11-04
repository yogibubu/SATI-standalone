# MATRIX-GFN2

`matrix-gfn2` is an independent implementation of the published GFN2-xTB
energy model. It contains no third-party semiempirical source code and has no runtime
dependency on either project.

The scientific method is due to Bannwarth, Ehlert, and Grimme, *J. Chem.
Theory Comput.* **2019**, 15, 1652-1671, DOI 10.1021/acs.jctc.8b01176. The
implementation is derived from the article and Supporting Information. Method
attribution is retained independently of implementation ownership.

## KEYMAKER FAST_XTB observables

When KEYMAKER selects resident FAST_XTB, it returns the energy, analytic
gradient, variational GFN2 SCC charges, and bond orders from the same converged
`GFN2Calculator.evaluate` result. KEYMAKER does not run APOC to replace those
charges with CM5 or to recalculate the bond orders. The resident GFN2 code
evaluates its native bond orders with a Mayer AO expression; this describes
the calculation inside GFN2, not an additional APOC analysis. The result JSON
keeps `charges_electron` and `bond_orders` under GFN2 provenance and does not
label the charges as CM5.

## SCF convergence

Closed-shell calculations use the TBLite GFN2 stopping thresholds by default:
an energy change below `1e-6` Hartree and an RMS change below `2e-5` in the
combined shell-charge, atomic-dipole, and six-component quadrupole vector,
with a maximum of 250 cycles. Dense and sparse SCF evaluate the same
quantities and limits. `accuracy` scales the default tolerances; callers can
override them with `scf-energy-tolerance`, `scf-mixer-tolerance`, and
`max-iter`. Open-shell calculations retain the density criterion because the
TBLite multipole mixer criterion here is closed-shell only.

## Qualification status

This package is under construction and is not yet a production calculator.
Implemented and unit-qualified components are:

- Cartesian contracted-Gaussian overlap, dipole, and second-moment integrals;
- native cell-list AO integral generation that evaluates only domain pairs and
  returns values plus centre derivatives in pair-packed form;
- environment-dependent extended-Hueckel Hamiltonian and classical repulsion;
- spatial orbital domains and error-reporting sparse truncation;
- shell-wise monopole SCC, finite-temperature occupations, and generalized
  Roothaan-Hall solution;
- cumulative atomic multipoles, anisotropic electrostatics/XC, and their Fock
  contribution;
- self-consistent two-body D4, its charge derivative/Fock term, and ATM energy.
- exact overlap/repulsion/frozen-D4 radial derivative kernels;
- sparse local matrices, local natural orbitals, fingerprinted restart, and SP2
  purification with a small-gap fallback;
- a fail-closed, machine-readable production qualification manifest and corpus
  parity evaluator.
- complete fitted GFN2 Tables S49-S50 for H-Rn (86 elements, 237 shells),
  checksum-locked and loaded without an external runtime;
- the stationary Lagrangian gradient contraction including Pulay, analytic CN
  Jacobians, explicit SCC kernel forces, and D4 coordination derivatives.
- analytic nuclear derivatives of overlap, dipole, and Cartesian second-moment
  integrals, plus the normalized real-spherical 6d-to-5d transformation.
- an integrated dense ISCF in which shell monopoles, CAMM multipoles, and
  charge-dependent D4 contribute to one Fock matrix and converge together;
- a complete autonomous energy and stationary analytic gradient, independently
  checked against numerical derivatives of the converged total energy;
- an executable 18-case corpus runner and a machine-readable parity report;
- standalone xTB-oracle parity for energies, energy-consistent gradients,
  charges, and explicitly reported Wiberg bond orders across all 18 cases.
- domain-restricted sparse SP2 certified against dense purification from 32 to
  256 orbitals with constant nonzero storage per orbital;
- autonomous Cartesian BFGS optimization, numerical Hessians, rigid-motion
  projection, and harmonic frequencies qualified against frozen xTB workflows.
- self-consistent Mulliken-monopole point-charge embedding with analytic forces
  on both QM atoms and external sites, qualified against xTB hardness-99 point
  charges and finite differences.
- variational SCC atomic charges exposed directly for solvent electrostatics,
  embedding and other xTB-level observables. CM5 is deliberately not evaluated
  by MATRIX-GFN2 and remains a higher-level QM observable;
- one cached overlap orthogonalizer per SCC cycle, process-parallel numerical
  Hessian columns, and automatic SciPy/NumPy/CuPy-CUDA/Torch-CUDA symmetric
  diagonalization selection in mandatory float64 precision.
- a production ISCF density-solver selector: diagonalization for fractional,
  open-shell, small-gap, medium, or geometrically compact problems; explicitly
  requested dense SP2 and symmetry-block SP2 for qualified gapped closed
  shells; and domain-sparse SP2 for the large localized regime. Automatic
  routing uses both AO dimension and measured localized-domain fill, so an
  efficient dense eigensolver remains selected while it is faster; all
  alternatives retain fail-closed diagonalization fallback;
- Newton--Schulz construction of the overlap inverse square root, checked by
  its `||I-X S X||` residual and automatically backed by spectral FP64
  orthogonalization if it cannot meet tolerance;
- native domain-sparse Newton--Schulz orthogonalization using symmetric sparse
  products and propagated truncation-error bounds, with no dense intermediate;
- lazily materialized Cartesian-to-real-spherical maps and a native pair-packed
  congruence for overlap, multipoles, and their centre derivatives;
- pair-packed extended-Hueckel, monopole, CAMM, anisotropic and D4 Fock
  assembly, followed by native sparse `X F X`, SP2 and `X P X` density builds;
- a matrix-free SCC shell operator: Laplace FMM evaluates the untruncated
  Coulomb tail and a symmetric dual tree evaluates the complete softened
  `gamma-1/r` correction. Energy, shell potential and analytic atomic forces
  come from the same functional, every near pair is exact, the accepted
  far-field clusters carry a conservative energy-error bound, and the force
  sum is translationally invariant;
- bounded sparse fixed-point DIIS that keeps the best historical residuals on
  the localized AO graph rather than retaining dense Fock/error matrices;
- certified incremental sparse SP2 based on density-matrix perturbation theory:
  the complete polynomial path and branch sequence are reused only for a truly
  local Fock correction, while spectral containment, canonical trace,
  idempotency and Fock-projector stationarity force a complete rebuild whenever
  the update leaves its validity or performance domain;
- lazy sparse density and Mayer bond-order observables on the SP2 route. Their
  NumPy compatibility boundary materializes a dense array only when an
  external consumer explicitly requests one;
- sparse Pulay matrices and direct pair-derivative gradient contractions;
- pair-packed two-body D4 preparation/evaluation with linear pair storage at
  fixed cutoff and an explicit conservative bound on the omitted long-range
  energy; an uncertified cutoff is never reported as exact;
- native neighbor-graph ATM triangle enumeration with sparse energy, analytic
  geometry/CN gradients, and an explicit omitted-triple bound;
- consumer-GPU SP2 products retaining the density diagonal and accumulation in
  FP64 while evaluating the cubic off-diagonal product in FP32, plus
  incremental AO/orthogonal-basis transforms with periodic FP64 resets;
- orbital-free construction of the energy-weighted density for analytic Pulay
  forces, asynchronous structure batches, and unique-pair integral assembly;
- an optional Torch coordinate-autograd facade backed by the analytic MATRIX
  gradient, plus a constant-memory implicit fixed-point VJP primitive for
  parameter fitting and hybrid ML models.

The automatic large-system route is end-to-end localized: it constructs
pair-packed Cartesian and spherical AO integrals, sparse overlap/core/Fock,
native sparse Newton--Schulz and SP2 densities, matrix-free SCC, sparse DIIS,
and lazy final density/bond orders. The stationary-gradient connector consumes
pair-packed overlap, dipole and second-moment derivatives directly, including
extended-Hueckel CN response and CAMM charge/dipole/quadrupole response; it does
not allocate an AO square or a nuclear-coordinate by AO-by-AO derivative tensor.
Explicit `sparse-sp2` requests fail closed for open shells, frozen-symmetry
sessions or point-charge embedding rather than silently entering
the dense compatibility route.

The Apple MPS backend is intentionally rejected only for diagonalization,
because it does not provide the float64 eigensolver required by the scientific
tolerances. On Apple Silicon it is used for the cubic FP32 SP2 products and
incremental congruence transforms; diagonal terms, matrix accumulation,
residuals, traces, and scientific decisions remain FP64 on the CPU. CUDA
execution provides the same split through CuPy or Torch when suitable hardware
is present. Automatic MPS selection starts at 2048 orbitals, the measured
crossover on the qualified M3 Ultra; ``MATRIX_MPS_MIN_ORBITALS`` can override
that threshold and ``MATRIX_SP2_PRODUCT_BACKEND=torch-mps`` can force it for
qualification.

Accelerated congruence transforms retain their immutable transformation on the
device between SCC iterations. Dense SP2 avoids a redundant FP64 matrix square
on every purification step: a cheap mixed-precision indicator triggers periodic
and near-convergence FP64 certificates, then switches to FP64 polishing below
the dimension-scaled noise floor, with one mandatory final certificate.
Sparse SP2 likewise reuses its current domain product and includes discarded
matrix norm in the convergence error. Accelerator failures are visible and
cause one fail-closed restart on the CPU; qualification calls never silently
claim a backend that did not execute.

All scientific production gates in the package manifest now pass. LINK uses
the resident `matrix-gfn2` session directly after checking this manifest;
there is no transitional or external semiempirical runtime fallback.

## Resident Apple FP32 SCC predictor

`GFN2Calculator(atoms, device="mps", density_solver="diagonalization")`
enables an additional resident MPS phase for gapped, closed-shell dense SCC.
The prepared packed multipole operators, Fock matrices, density, DIIS history,
and SP2 products remain in FP32 on the GPU during this phase. The calculator
retains the geometry buffers, integer pair indices, overlap workspace, and
density workspace between calls. The first geometry uses two packed geometry
uploads; later compatible geometries update the existing floating buffer and
reuse integer data. Including the overlap transform and initial density,
this reduces uploads from four to three, with no new retained buffer
allocations. Only one density array is downloaded.
Scalar convergence checks synchronize explicitly and are counted separately.
The SCC/AES/AXC and charge-dependent D4 Fock use the same analytical density
derivatives as the host implementation, without an autograd tape.
Endpoint operators retain only pairs with at least one nonzero prepared
overlap, dipole, second moment, or core-Hamiltonian entry. This is exact-zero
compression of the existing integral support, not an additional spatial or
numerical cutoff. The Fock matrix is zero-initialized outside that support.
The report exposes retained and full pair counts. A changed support invalidates
the density anchor and reallocates buffers when their shapes change.

For recurrent direct-generalized calls, the predictor's overlap transform
uses existing LAPACK Cholesky and triangular solves rather than a separate
overlap eigensolve. The GPU uses the general congruences `X.T @ S @ X`,
`X.T @ F @ X`, and `X @ P @ X.T`; it does not assume symmetric `X`.
The overlap residual and final canonical FP64 certification remain mandatory.

The initial orbital/overlap preparation, final finite-temperature FP64 SCF,
and analytical gradient run on the CPU. A nearby GPU density is propagated
as an increment relative to the previous GPU solution and anchored to the
last certified FP64 density. This preserves the host density's low bits.
Large displacements and changed integer pair graphs invalidate that anchor.
Thus `device="mps"` describes a
hybrid calculation, not an entirely FP32 energy/gradient implementation.
Balanced FP32 energy reductions, scaled regularized DIIS, overlap residuals,
and SP2 trace/idempotency checks control numerical noise. Unsupported states,
device errors, and a failed host refinement are reported and retried on CPU.
For sufficiently large gapped systems with integer Fermi occupations, the
host refinement reuses the existing matrix-free GMRES solver to solve a
preconditioned occupied/virtual Sylvester equation. Cholesky normalization
maintains the occupied metric. Spectral-gap, stationarity, and metric checks
reject unsuitable intermediate projectors. A canonical eigensolve and the
original SCF energy and mixer tests independently certify the final state.
`MATRIX_GFN2_MPS_REFINE_OCCUPIED` defaults to `auto`. The automatic policy
uses packaged complete-call qualifications, matching the CPU model, NumPy
and Torch versions, one-thread execution, and the neutral gas-phase
electronic conditions of the benchmark. Occupied refinement requires at
least a 10% advantage at both adjacent qualified sizes. There is no
extrapolation: missing, malformed, numerically unqualified or mismatched
calibrations select canonical refinement. The qualified M3 Ultra cases
therefore select canonical refinement automatically. No online trial SCFs
or extra diagonalizations are introduced by selection.
`0`/`canonical` forces full-spectrum host refinement; `1`/`occupied` requests
the guarded occupied algorithm when eligible. A canonical decision bypasses
both SP2 density construction and occupied refinement. The policy decision,
reason and calibration identifier are included in the accelerator report.

For compatible recurrent calls the calculator compares the GPU proposal and
the existing extrapolated host restart using their FP64 Fock commutator and
electron-count residuals. It selects the lower-residual seed and restores the
matching native observables before SCF. This selection is not a convergence
certificate: the original finite-temperature SCF tests still apply.
The context also retains the certified canonical orbitals. Existing Cholesky
and triangular solves transport them into the new overlap metric, with a
bounded metric change and explicit orthonormality check. A complete orbital
frame can be transported with finite-temperature weights; integer-projector
eligibility is checked using the current SCF temperature, not old occupation
weights. Failed transport or inadmissible current occupations return to
canonical diagonalization.

Occupied refinement constructs only the three independent MO Fock blocks,
caches overlap-times-reference-orbitals and the gap preconditioner, and
reuses Fock/metric products through the occupied rotations. Virtual-column
normalization is delayed until a second Newton step is actually needed.
The canonical reference and its cached products are replaced together.
The switches `MATRIX_GFN2_MPS_SELECT_DENSITY=0`,
`MATRIX_GFN2_MPS_TRANSPORT_ORBITALS=0`, and
`MATRIX_GFN2_MPS_REUSE_BLOCKS=0` independently disable these three mechanisms
for qualification; all three are included in the result-cache key.

`result.telemetry.accelerator_execution` distinguishes the requested device
from actual execution and records Fock calls, SP2 products, scalar
synchronizations, array uploads/downloads, resident input bytes and timings.
It also records context reuse, retained-buffer allocations (excluding
temporary tensors inside operators), occupied-refinement steps, and canonical
solves during host polishing. Batch clones isolate their mutable GPU context.
The report identifies the selected density seed, both FP64 seed residuals,
comparison time, orbital transport, and refinement-block reuse. Transport
acceptance does not guarantee a saved eigensolve: later residual and spectral
checks can still require a canonical solution.
`MATRIX_GFN2_DEVICE=cpu|mps|auto` provides the same selection to callers that
do not pass `device`. The default remains CPU. Automatic selection requires
an explicit `mps_min_orbitals` (or `MATRIX_GFN2_MPS_MIN_ORBITALS`) until a
total-call crossover is qualified for this complete hybrid route. This is
separate from the existing mixed-product SP2 crossover calibration.

The resource-bounded qualification is reproducible with:

```sh
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/gfn2_mps/qualify.py --waters 1 4 16 32 64 128 256 \
    --molecules --repeats 3 --output benchmarks/gfn2_mps/m3_pro_qualification.json
```

The report includes displaced geometries (avoiding cached results), matched
CPU/GPU electronic thresholds, full elapsed times, molecular force parity,
and a short velocity-Verlet NVE comparison. The M3 Pro measurements retain
CPU automatic selection: the new predictor has not yet improved complete
energy/gradient call times in the measured range.
Add `--compare-fresh` to interleave a fresh-context/full-spectrum control
on the same hardware. This comparison includes preparation, synchronization,
transfers, and the final host certification in both GPU routes.

The occupied-refinement comparison is
`benchmarks/gfn2_mps/m3_ultra_fp64_refinement_qualification.json` (three interleaved
repetitions, one host thread). At 384 AOs the complete persistent call takes
106.0 ms versus 102.9 ms for the fresh-context/canonical-refinement control
and 77.1 ms for CPU. At 384 AOs recurrent refinement uses one canonical solve;
at 1536 AOs conservative checks still require two. At 1536 AOs the corresponding
times are 2.400, 2.080, and 1.734 s. The canonical control is faster on this
larger case; occupied refinement is not a universally favorable strategy.
No CPU/GPU crossover is qualified. Maximum final CPU/hybrid differences are
1.41e-11 Hartree in energy and 4.61e-8 Hartree/bohr in gradient; both short
NVE checks pass. All recurrent predictors are accepted; cold tiopronin and
some cold clusters exercise the documented CPU retry. An additional 2304-AO
case was interrupted during the cold CPU SCF to bound the resource cost;
it is not included in the qualified range and has no reported speedup.

`benchmarks/gfn2_mps/profile_ground_state.py` records separate phase timings
and a diagnostic Python profile on displaced geometries. Profiled calls are
separate from the timed repetitions. `qualify.py --resume` can reuse completed
sizes after an interruption, but rejects mismatches in scientific source
hashes, native binaries, hardware, electronic thresholds, repeats, or routes.

For reproducible accelerator qualification, run
``benchmarks/gfn2_symmetry/run_benchmark.py --backend torch-mps`` on the
M-series host.  MPS diagonalization is float32-only, so the benchmark records
its platform-specific parity envelope (energy ``2e-5`` Hartree and gradient
``5e-7`` Hartree/bohr) separately from the strict FP64 CPU/CUDA envelope.
The production solver still keeps residual checks, accumulation, gradients,
SCC charges, and bond orders in FP64.  The same benchmark exercises the bounded
symmetry cache, block-SP2 retry path, degeneracy alignment, asymmetric
embedding fallback, and serialized telemetry schema ``v1.1``.

Symmetry execution is configurable without changing the scientific contract:
``GFN2Calculator`` accepts ``symmetry_cache_capacity`` and
``symmetry_small_system_host_threshold``. The cache key includes the frozen
ORACLE digest, geometry, and overlap matrix, so a reused adapter cannot carry
an overlap from a different geometry. Small MPS eigensolves are routed to host
BLAS by policy while symmetry adaptation, orbital assignment, density
expansion, SCC charges, bond orders, and gradient projection remain active.

# MATRIX-GFN2 and native NN-xTB

## Official Code Ocean weights

The official `model_unified.pt` from capsule `1479726`, version 1, has a
different architecture and parameter convention from the provisional
`.gfn2nn` model. Export it once with `tools/build/export_nnxtb_codeocean.py`, using
the capsule's Python environment and `config_unified.yaml`. The exporter
loads the checkpoint strictly and emits a JSON manifest and a sibling
TorchScript **parameter encoder**, with SHA-256 provenance. It does not
export or invoke the reference electronic calculator.

```python
from matrix_gfn2 import GFN2Calculator

calc = GFN2Calculator(
    ["O", "H", "H"], nn_enabled=True,
    nn_model_path="matrix-nnxtb-official.json",
)
result = calc.evaluate(coordinates_angstrom)
_, response = calc.evaluate_independent_hessian(coordinates_angstrom)
hessian = response.hessian_hartree_per_bohr2
```

Runtime requires NumPy, the native MATRIX kernels and the `ml`/Torch extra;
it does **not** require MACE, e3nn, dxtb, libcint, tblite or an xTB executable.
Plain `GFN2Calculator(atoms)` remains unchanged and does not load NN weights.
Native GFN2-ALPB is selected with `GFN2Calculator(atoms,
alpb_solvent="methanol")`; `ALPBParameters.available_gfn2_solvents()` lists the
25 supported solvents. The SCC reaction field, cavity/dispersion, hydrogen
bonding, and reference shift are evaluated by MATRIX. Analytic gradients use
closed-form derivatives; analytic Hessians differentiate the native ALPB
energy functional twice with the optional `solvation`/Torch extra and include
geometry–charge blocks in the existing CPSCF response. Neither route invokes
tblite or xTB at runtime. The ALPB Hessian has been checked against fresh,
independent displaced-gradient calculations for water in methanol; the
matrix-free and explicit Hessian operators agree.
For a full parallel Hessian, each process evaluates a bounded batch of exact
fixed-charge ALPB geometry-Hessian columns with a shared forward-mode graph.
The mixed geometry--charge block is reused for the shell-charge response;
the matrix-free operator retains its reverse-mode graph for arbitrary
directions. Each process uses one PyTorch intra-op thread to avoid CPU
oversubscription. On the 149-atom chlorophyll--(methanol)2 geometry, 16
MAC-STUDIO processes took 24.5 s for the ALPB Hessian, compared with 29.0 s
for the previous per-column route; the maximum matrix-element difference was
`1.92e-14 Eh/bohr²`.
For a solvent-relaxed structure, LINK passes the same ALPB field through
every native energy and gradient evaluation:

```sh
matrix link optimize molecule.xyz --run-dir run-alpb \
  --coordinate-kind cartesian --backend matrix-gfn2 \
  --method MATRIX-GFN2 --charge 0 --multiplicity 1 \
  --alpb-solvent methanol --no-geometry-correction
```

The gas-phase-trained geometry correction and gas-only compiled optimization
route are bypassed in this mode; the run record stores the solvent identity.
For a plain GFN2 full analytic Hessian, `calc.evaluate_independent_hessian(
coordinates_angstrom, workers=8)` distributes independent Cartesian columns
across processes. Each process owns its SCF and matrix-free CPSCF state; worker
count is capped by the available-memory estimate. The NN Hessian remains
analytic but currently uses one process, so `workers>1` is rejected with NN.
The native AO second-derivative kernel skips orbital pairs whose centers do
not move in a directional displacement; this leaves dense directions
unchanged and accelerates Cartesian Hessian columns.
The directional anisotropic Fock derivative is contracted in the native
kernel by forward differentiation of the resident analytic first-derivative
expression; the Python expression remains the source-only fallback.
The sparse D4 Axilrod–Teller–Muto three-body Hessian-vector contraction also
uses a native analytic directional kernel, retaining the Python expression as
the fallback. Its geometry and coordination-number contributions are
contracted without finite-difference displacements.
The directional cumulative-atomic-multipole Fock contraction projects the
quadrupole potentials before forming Cartesian products, avoiding temporary
pair-by-3-by-3 geometry tensors. The density-response symmetry check retains
its absolute tolerance with a direct maximum-residual reduction.
The matrix-free CPSCF solver uses a single Arnoldi orthogonalization pass
when its true residual certifies convergence; otherwise it retries with full
reorthogonalization. The response tolerance is unchanged.
For a closed-shell state with occupations within `1e-11` of integers, the
CPSCF solver removes occupation roundoff and forms only the occupied–virtual
rotation block. Other accepted occupation patterns retain the general orbital
response path. On the 149-atom chlorophyll–(MeOH)2 benchmark, two 12-process
analytic Hessians took 28.9 and 29.0 s, versus 33.8–34.1 s before this CPSCF
change; the maximum Hessian-element difference was `1.37e-10 Eh/bohr²`.
Within a process, separately constructed official-NN calculators reuse the same
verified, read-only TorchScript encoder. Its cache key includes the file identity
and expected SHA-256; a changed encoder is verified again and rejected if it no
longer matches the manifest. Each calculator retains its own geometry-dependent
parameter cache.

The official profile reproduces the archived inference semantics:

- The 29 atomic and 24 global outputs are not clamped to the provisional
  ±0.5 bound. Their physical units, reference eV/Hartree conversion and
  per-shell indexing are preserved.
- `shell.h0exp` changes the H0 Slater prefactor, including learned `wexp`.
  The official libcint route uses **fixed baseline Gaussian integrals**;
  moving the basis too would give a different energy surface. The existing
  `shell.exp` moving-basis/Pulay route remains available for the provisional
  model, including its analytic mixed and second derivatives.
- Learned `klight`, `s6` and `s10` enter repulsion and D4, including the C10
  term, parameter adjoints, mixed derivatives and CPSCF sources. ATM uses
  neutral reference charges, as in the archived official code.
- AES uses its unshifted coordination radii, while H0 uses learned radii.
  This is mapped through the existing CN/vcn interface with analytic first
  and second chain rules. Reference ReLU mappings are retained; singular
  native SCF parameters are rejected, not replaced by provisional floors.
- `kpol` and the predicted `ss/pp/dd/sd/pd` outputs are inactive in the
  archived reference Hamiltonian. They are not silently activated here.

Electronic structure, multipoles, SCF/CPSCF, geometry derivatives and the
complete Hessian assembly remain native MATRIX. Torch differentiates only
the neural parameter encoder and its parameter mapping. There is no finite
difference or external electronic-code fallback in the analytic route.
For the official weights, the analytic Hessian reuses one neural forward
graph for the full weighted parameter-curvature matrix; it no longer builds
a separate network graph for every Cartesian curvature direction.
The molecular NN restrictions described below still apply. Native GFN2
parameter coverage is H–Rn; network element labels alone do not extend it.

Qualification is in `tests/test_matrix_gfn2_official_nn.py`. Set
`MATRIX_OFFICIAL_NNXTB_MODEL` to the exported manifest; independent encoder
and electronic fixtures are read from its directory. The tests cover water,
formaldehyde, H2S, NH4+, tiopronin and a disconnected water dimer, including
neural first/second derivatives, all stationary head-response blocks and
complete energy/gradient/Hessian comparisons. An optional
`MATRIX_OFFICIAL_NNXTB_REPORT` path records the electronic comparison errors.
The reference fixtures use the audited capsule implementation, including
the corrected AES self-pair mask and converged differentiable SCF.

## Provisional model and common calculator API

`GFN2Calculator` keeps plain MATRIX-GFN2 as the default. Native neural
descriptors and parameter shifts can be inspected separately:

```python
from matrix_gfn2 import GFN2Calculator, NNXTBOverlay

plain = GFN2Calculator(["O", "H", "H"])
overlay = NNXTBOverlay("model.gfn2nn")
shifts = overlay.parameter_shifts([8, 1, 1], coordinates_bohr)
matrix_shifts = overlay.matrix_parameter_shifts([8, 1, 1], coordinates_bohr)
```

The model reader validates the binary format, tensor dimensions and FNV-1a
checksum. `GFN2Calculator(..., nn_enabled=True, nn_model_path=...)` selects
the NN Hamiltonian; omitting both arguments retains plain MATRIX-GFN2. The
NN route uses no external TBLite or xTB executable. For now it is a
correctness-first dense molecular route: `evaluate(...)` returns energy,
the fixed-parameter analytic gradient plus the NN parameter-response VJP,
SCC charges and bond orders;
`evaluate_independent_hessian(...)`, `numerical_hessian(...)`,
`optimize_geometry(...)` and `harmonic_analysis(...)` work on the NN surface.
The NN gradient uses analytic coordinate derivatives and stationary-SCC
adjoints for every active molecular head, including basis-exponent overlap,
CAMM and Pulay terms; unsupported heads fail explicitly rather than falling
back to converged-SCC differences. Install the `ml`
extra for the Torch NN VJP. Passing `gradient_step_bohr` to `evaluate(...)`
explicitly requests the independent full-energy coordinate-difference audit.
`evaluate_nn_hessian_analytic(...)` assembles the complete analytic molecular
NN Hessian, including fixed-parameter `E_RR`, both geometry/parameter cross
terms, `J.T E_pp J`, and neural curvature `sum(E_p * p_RR)`. H0, SCC, AES,
D4 pair/ATM, repulsion, moving basis exponents and CN-dependent `atom.rcov`
all enter the stationary CPSCF response and explicit second derivatives.
No displaced geometry or parameter SCF calculation is used in this route.
Both the NN chain-rule matrix and the resident fixed-parameter block expose
their pre-symmetrization reciprocity errors and are checked independently.
The current implementation materializes a dense Cartesian matrix and requires
integer final orbital occupations, as does the resident ground-state CPSCF.

`evaluate_independent_hessian(...)` uses this analytic route with NN enabled,
and the existing analytic GFN2 Hessian when NN is disabled. Both return a
result object with `hessian_hartree_per_bohr2`. The compatibility entry point
`evaluate_nn_hessian_mixed_response(...)` and the NN branches of
`numerical_hessian(...)`, `sycart_numerical_hessian(...)` and
`hessian_vector_product(...)` also use the analytic NN matrix. Their legacy
step/stencil arguments do not select numerical NN derivatives.
The explicitly named `evaluate_nn_hessian_mixed_response_audit(...)` retains
the displaced-SCC decomposition for independent diagnostics; it is not used
by any of these production entry points. The `chain_rule_error` compatibility
property of an analytic result measures block-sum rounding only; numerical
validation is performed separately by the tests.
`evaluate_nn_fixed_parameter_hessian(...)` separately exposes the analytic
overlaid-CPSCF `E_RR` block; it is **not** the Hessian of the NN energy surface
because geometry-dependent parameter shifts are held fixed.
Embedding, frozen symmetry and orbital/projector retention are not supported
by this dense NN route and fail explicitly.

```python
calc = GFN2Calculator(
    ["O", "H", "H"], nn_enabled=True, nn_model_path="model.gfn2nn",
)
result = calc.evaluate(coordinates_angstrom)
result, hessian = calc.evaluate_independent_hessian(coordinates_angstrom)
cartesian_hessian = hessian.hessian_hartree_per_bohr2
```

`load_gfn2nn()` loads all tensors as immutable NumPy arrays.
`NNXTBOverlay.forward_layer_states(numbers, coordinates_bohr)` evaluates every
fixed-geometry MACE layer and exposes their states for layerwise comparison.
`forward_descriptors(...)` returns the final invariant descriptor per atom.
The layer equations include real spherical harmonics, Clebsch-Gordan products,
correlations through order three, and residual channel updates. Translation,
rotation and atom-permutation consistency have been tested. Direct descriptor
parity with the provisional Rust reference was measured on water, ethane and
tiopronin: maximum absolute descriptor differences were respectively
`3.4e-16`, `3.4e-16` and `4.5e-16`; the 33 bounded heads agreed within
`8.4e-17`. The water output is retained in
`tests/fixtures/nn_xtb_water_reference.tsv`; set `MATRIX_NNXTB_TEST_MODEL` to
the matching `.gfn2nn` file to run the parity gate.

`NNXTBOverlay.parameter_shifts(...)` gives the 33 physical parameter shifts
for an atom list and Cartesian bohr geometry. With the optional `ml` extra,
`parameter_shift_vjp(..., adjoints=...)` gives the exact coordinate derivative
of a weighted sum of those shifts, and optionally its Hessian-vector product
for fixed adjoints. This is the neural chain derivative, not a complete
NN-GFN2 energy gradient. `matrix_parameter_shifts(...)` converts raw reference
head units into MATRIX units; `build_nn_parameter_state(...)` in
`matrix_gfn2.nn_xtb_parameters` applies them to all per-atom, shell and global
parameters, retaining absent-shell masks. Water parameter values match the
provisional reference within `2e-12`. H0, repulsion and dense SCC kernels can
accept the learned global scalings and have finite-difference derivative tests.
The unfused `evaluate_nn_energy_development(...)` route uses these parameters
in a converged SCC calculation, including AES, D4 pair and charge-dependent
ATM energy/potential. Its energy errors against the provisional reference are
`4.1e-12 Eh` (water), `4.3e-12 Eh` (ethane) and `2.3e-10 Eh` (tiopronin).
`evaluate_nn_derivatives_development(...)` recomputes the complete energy at
displaced geometries for numerical gradients and Hessians. On water, these
match the provisional reference within `1e-7 Eh/bohr` and `1e-6 Eh/bohr²`.
These diagnostics underpin the public correctness-first route. The public
gradient contracts analytic parameter-response adjoints with the analytic NN VJP; on
tiopronin it agrees with the independent full-energy coordinate difference
within `1.85e-8 Eh/bohr`. Complete analytic Hessian checks are in
`tests/test_matrix_gfn2_nn_analytic_hessian.py`: head responses against
reconverged stationary adjoints, full matrices against gradient differences,
reciprocity, translations, rotations, and guards against displaced-SCF or
numerical-derivative fallback. Reference fixtures are in `tests/fixtures/nn_xtb_water_*`.
