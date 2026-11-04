# matrix-oracle

Public Python API and command-line interface for ORACLE molecular perception.
The scientific implementation is provided by `matrix-chem`; optional GUI and
format adapters are loaded only when requested. ORACLE freezes both molecular
perception and the ordered redundant primitive/Wilson-B source subsequently
consumed by SMITH to construct SONIC coordinates.

Robust perception under declared numerical or physical thermal uncertainty,
temporal hysteresis and the immutable exploration-to-exploitation handoff are
specified in `docs/architecture/ORACLE_PERCEPTION_ROBUSTNESS.md`. Production
SMITH accepts only complete v2 ORACLE semantics and never reconstructs missing
chemistry.

```bash
oracle doctor
oracle examples ./oracle-examples
oracle analyze ./oracle-examples/benzene.xyz -o benzene.xyzin \
  --report benzene.oracle.json --human-report benzene.oracle.txt
oracle prepare-initial 'NCC(=O)O' -o glycine.xyz \
  --lcb26-root data/lcb26 --json
```

The enriched file contains `#PRIMITIVES` with the primitive definitions,
reference values, B-matrix rank and deterministic fingerprints. The same
contract is available from Python:

```python
from matrix_oracle import read_primitive_contract, primitive_b_matrix
from matrix_chem import read_enriched_xyz

state = read_enriched_xyz("benzene.xyzin")
contract = read_primitive_contract("benzene.xyzin")
B = primitive_b_matrix(contract.primitives, state.coordinates_angstrom)
```

ORACLE emits one synthon model, one atomic charge and one chemical bond order.
Complete QM observations select the paired CM5/Mayer level; without both
complete vectors, ORACLE uses the paired electronegativity/Pauling level. A
state therefore never mixes availability levels, atoms or bonds. The smooth
connectivity weight used to perceive the graph is not a second bond order.
The CV posterior correction is likewise unique: the radius-aware period-line
Gaussian calibration is fixed and no CLI, GUI, or Python option selects a
second calibration. Descriptor radii and UFF non-bonded radii have separate,
explicit functions tied to their physical roles rather than a runtime scheme
selector.

The population ladder is separate and explicit.  L0 normally means a
fixed-geometry PBE0/def2-TZVP density, using the associated def2 ECP for heavy
atoms, followed by APOC CM5/Mayer analysis.  L1 means MP2; its cardinal orbital
basis family is intentionally not yet frozen.  Both preserve the complete
electronic state, one-particle density, natural orbitals and natural
occupations.  Either natural-orbital space can seed a later local-correlation
calculation.  At L0 that calculation must still construct pair information;
L1 instead retains the MP2 amplitudes and pair densities needed to form PNOs
directly.  L1 is consequently the natural rung for cardinal-basis
extrapolations and related systematic convergence studies, whereas L0 remains
the standard economical population reference.  Overlapping fragment results
may be synthesized only when every atom and perceived bond is covered and
every fragment uses the same declared level.

This new population notation does not silently re-label the existing
geometry-refinement calibration: the resident L1-to-PL1 correction was fitted
to the legacy DPCS3 reference and must be revalidated before application to a
new MP2 geometry level.

ORACLE exposes two separate geometry models. The initial-structure model
constructs a baseline from LCB26/L2 fragments and does not consume L1−L2
correction deltas. In `INITIAL-L1` mode it supplies a starting geometry for a
new QM L1 optimization; the L2 fragments are only structural priors. The
replacement PL1 geometry model is deliberately separate from population
analysis. LCB26 retains the stable L0/L2 CM5 and Mayer vectors, while a paired
L1 geometry store (rDSD-PBEP86-D3/D4 with the validated 3F12 basis) contributes
only atom-mapped primitive-coordinate deltas, `Δ(L1−L2)`, with L2 as the
reference. For a declared-L1 input ORACLE applies the calibrated CV/VAL/HBond
corrections to the L1 geometry to produce PL1; L2 is not used as the
operational geometry. No L1 CM5 charges or Mayer orders are needed for this
correction.

The correction layers are not conflated: the existing `CV` term is
method-independent and remains frozen. The calibrated posterior in
`data/lcb26/pl1_model.json` fits only the residual
`L2-(L1+CV)`. Its covalent Gaussian centres use covalent radii and continuous
Mayer/synthon bond order. The reduced posterior has seven covalent chemical
classes and one shared H-bond amplitude; sulfur is handled by the period-aware fallback; its hydrogen-bond centre uses scaled
vdW radii.

The active `data/lcb26/pl1_model.json` uses atom-wise amplitudes plus
two-partner corrections based on electronegativity difference and
`ΔZ = Zeff - Z`; CM5 charges are not used. Unknown elements use a fitted
group/period trend, while simpler fits remain available for auditing.
The paired residual fit uses bounded robust reweighting so isolated bad pairs
do not dominate the atomic parameters.
The legacy `BL1_CONJUGATION` and `HBond` path remains the compatibility
default, while `refine_l1_geometry(..., pl1_model_path=...)` activates the
refitted model without changing the file or tool contract.

ORACLE also exposes a reusable, noniterative hydrogen-bond charge-response
contract. Its continuous D--H...A perception drives two explicit channels:
fragment-neutral polarization (stored separately for donor and acceptor) and
interfragment charge transfer. In particular the donor X--H redistribution is
not represented by a net-transfer parameter. Each donor/acceptor synthon rule
can be fitted either from paired CM5 calculations without/with one bridge or
from the variational QEq/SQE model used as a teacher. Runtime evaluation
returns the two polarization vectors and transfer vector separately, together
with sparse analytic Cartesian first and second derivatives. The same
operation therefore covers solvent--solvent, solute--solvent and
intramolecular hydrogen bonds without introducing 1--2/1--3/1--4 energy
classes.

The resident paired-CM5 library averages 320 fixed-geometry
PBE0/def2-TZVP dimers into all twenty combinations of four donor classes
(O--H, N--H, S--H and P--H) and five acceptor classes (O, N, S, P and
carbonyl O).  Each class contains sixteen chemically audited substitution
environments.  Polarization and charge transfer remain separate physical
contributions, and the runtime response is confined to the active donor X--H
bond and the local acceptor environment.

The unified LCB26 reference index is directly queryable from ORACLE through
`matrix_oracle.query_lcb26`. It selects geometries and immutable CM5/Mayer
synthon observations by identifier, name/alias, constitutional SMILES,
formula, element counts, dataset, atom count and ORACLE ring count. ORACLE
perception remains the authority for rings; ARCHITECT consumes the selected
reference to compile ZAFF-fast and ZAFF0, so the responsibilities do not
overlap.

The single versioned table is distributed by `matrix-chem` as
`data/hbond_charge_response_l0.json`.  The shared
`prepare_cm5_hydrogen_bond_charge_model` API supplies a reusable skinned pair
list for optimization, SENTINEL exploration and simulation, while
`tune_cm5_hydrogen_bond_charges` provides the equivalent one-shot operation.
ORACLE consumes these common contracts for perception and retains its
full electronic-training provenance.

No independent radial parameters are fitted. The runtime amplitude is the
ratio of a generalized Pauling H...A bond order at the current and reference
distances. Its natural decay length is the resident sum of H and acceptor UFF
van der Waals radii; Mayer H...A orders from the L0 dimers are stored as the
electronic reference and quality control. Contacts outside ORACLE's existing
continuous-perception domain receive exactly the isolated-fragment charges.

CM5 values can be declared intrinsic or treated as observations at their
stored reference geometry.  In the latter case, the shared service subtracts
the tabulated response at the reference strengths to recover the intrinsic
no-contact vector. At runtime it adds the same vectors at the current
strengths. Equivalently, every charge is the reference CM5 value plus
`(current_strength - reference_strength) * unit_response`. This reproduces the
CM5 input exactly, reaches the intrinsic vector at dissociation, and prevents
double counting. The default `auto` policy applies this inversion to supplied
CM5 data and treats electronegativity fallback charges as already intrinsic.

Water uses the published closed-form missing-coordination/charge-transfer
model (Barone et al., ACS Omega 2022, eqs 18--20), with integer contact counts
replaced by the shared continuous hydrogen-bond strength. No new water fit is
performed. The isolated endpoint is CM5, the fully four-coordinate endpoint
is exactly TIP3P-FB, and the signed per-contact transfer is the published
0.10 e. In the symmetric pentamer, two donations and two acceptances cancel
the transfer and leave the central water exactly neutral and TIP3P-FB.
The resident paired-CM5 table is used for the other donor/acceptor classes.
Spherical and ellipsoidal exposure restores only missing donor and lone-pair
capacities. In a finite box the identical versioned boundary correction is
mandatory for both the variational teacher and the closed-form student, so no
box-size artifact is absorbed into synthon parameters. ARCHITECT/ZAFF consumes
this public ORACLE state as `LOCAL_HBOND_CLOSED_FORM`; the variational
`VARIATIONAL_QEQ_SQE` route remains the teacher and the fallback outside a
calibrated synthon domain.

For electrostatically embedded QM/MM calculations, the electronic-structure
program sees the current MM Gaussian charges and APOC returns CM5 charges from
the resulting QM density. ORACLE uses those CM5 values and the cross-boundary
hydrogen bonds to polarize the MM fragment. QM--QM response remains owned by
the QM program; across the QM/MM boundary the local rule retains only the
fragment-neutral MM polarization and projects net charge transfer to zero.
MM--MM contacts may retain both channels. This prevents double counting of
QM--MM electrostatics and preserves the fixed QM electron number and MM charge
of ordinary electrostatic embedding.

ORACLE also exposes accuracy-ladder-aware geometry refinement. Core--valence
(CV) is a transversal contribution available at every valence rung; BL1
conjugation and PL1 selected-pair corrections are attached only to L1. These
names deliberately replace no legacy algorithm name:

```python
from matrix_oracle import ValenceLevel, build_accuracy_ladder_plan

atomic_numbers = [6] * 6 + [1] * 6  # benzene example
plan = build_accuracy_ladder_plan(
    contract.primitives,
    atomic_numbers,
    valence_level=ValenceLevel.L1,
)
```

See `docs/ORACLE_QUICKSTART.md` in the MATRIX source distribution for the
complete installation and reproducibility contract.

ORACLE deliberately stops at perception, redundant PIC/B construction and
primitive-space structural improvement. The current L1-to-PL1 workflow is one
calibrated application of that general operation. SMITH constructs SONIC coordinates and
ARCHITECT owns Hessian reduction, force-field parameterization and ZAFF. The
versioned ownership contract is available as
`matrix_oracle.oracle_scope_contract()` and is embedded in every complete
analysis report.

For Cartesian basin initialization, ORACLE can also issue a frozen initial
topology contract. A classical GAFF/OpenMM preoptimization is eligible only
when geometry-first ORACLE connectivity agrees exactly with the trusted
constitutional graph and GAFF typing is complete. The accepted topology is
then reusable for SONIC. When connectivity is not fixed, Keymaker uses a
topology-flexible MATRIX-GFN2 calculation first and asks ORACLE to freeze the final
basin topology afterward.

The ORACLE--SMITH--LINK geometry-refinement path can request
`ARCHITECT/ZAFF` without naming one field. The shared resolver first searches
the ZAFF library using ORACLE atom order and covalent topology plus charge and
multiplicity; when no validated entry exists, LINK realizes the step with
xTB/GFN-FF. This keeps energy evaluation under LINK ownership while giving
ORACLE refinement the same library-first policy as LINK and SENTINEL.

The optional desktop command opens the same canonical ORACLE window as the
MATRIX shell, rather than the historical multi-tool project dashboard:

```bash
python -m pip install 'matrix-oracle[qt]'
oracle-gui benzene.xyzin
```

The publication wheel is deliberately narrower than the migration worktree:
it contains only the public perception, PIC/B, refinement, CLI and canonical
GUI-launcher modules. Historical dashboard, GICForge/SMITH and GF/TRINITY
helpers are excluded from the standalone artifact and are checked by the
clean-install verifier.
