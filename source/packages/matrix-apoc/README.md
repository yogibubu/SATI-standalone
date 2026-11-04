# APOC

**APOC — Atomic and Pairwise Observables from the Charge density** is the
MATRIX tool that normalizes the electronic part of QM results.

APOC owns two related backend-independent contracts:

- a population contract with Hirshfeld and CM5 atomic charges, the complete
  Mayer bond-order matrix, electron count, molecular charge, and provenance;
- an electronic-state contract containing the AO metric, orbitals,
  occupations, density matrices, optional Fock matrices, and transition
  densities.

APOC also owns the accepted-state history used between geometry evaluations.
Resident and process-based calculators export their converged state to APOC;
APOC validates compatibility and prepares projected or extrapolated guesses.
LINK consumes those guesses without maintaining a second matrix lifecycle.

The canonical data are written to `#QM_POPULATION` using schema
`matrix.qm.population.v1`.  Gaussian/GDV output is parsed directly.  Other
programs can currently export a complete Molden file (basis plus occupied
molecular orbitals), from which APOC reconstructs the AO density and evaluates
the same Hirshfeld/CM5 and Mayer definitions through PySCF. Molden is an import
adapter, not the native persistence contract. Complete electronic states are
stored in TREXIO (HDF5 by default). PySCF and Psi4 wavefunctions can be captured
directly through their Python APIs, without writing Molden; the same direct
adapter pattern is the contract for the remaining electronic-structure
backends.

ECP wavefunctions use an explicit valence-space population contract.  Neutral
Hirshfeld pro-atoms are calculated with the same orbital basis and ECP,
`q_A = Z_eff,A - N_val,A`, while CM5 receives the physical atomic number
`Z_A = Z_eff,A + N_core,A`.  Per-atom frozen-core electron counts are
serialized and the reported total electron count includes them.  Mayer bond
orders remain defined in the explicit valence AO space.  No artificial core
density is added.

```bash
apoc gaussian calculation.log --output calculation.apoc.json --xyzin molecule.xyzin
apoc gaussian-fchk calculation.fchk --output calculation.apoc.json \
  --xyzin molecule.xyzin
apoc orca calculation.gbw --output calculation.apoc.json --xyzin molecule.xyzin
apoc molden orbitals.molden --output calculation.apoc.json --xyzin molecule.xyzin
apoc pyscf-output pyscf_job.out --output calculation.apoc.json --xyzin molecule.xyzin
apoc inspect molecule.xyzin

# Remove the dependency on the Molden executable and preserve the wavefunction.
apoc state import-molden orbitals.molden molecule.trexio.h5
apoc state inspect molecule.trexio.h5

# Natural orbitals; equal-occupation spaces may be recanonicalized without
# changing the density that defines them.
apoc state natural-orbitals molecule.trexio.h5 natural.trexio.h5 --recanonicalize

# Follow TD roots by transition-density overlap. At different geometries APOC
# needs the cross-AO overlap, supplied directly or evaluated from the two
# temporary Molden exports during migration.
apoc state track point_0.trexio.h5 point_1.trexio.h5 \
  --reference-molden point_0.molden --candidate-molden point_1.molden --strict
```

For a Gaussian formatted checkpoint, the direct adapter imports the AO basis,
orbitals, occupations and available density matrices into APOC's
backend-neutral electronic-state contract. No Molden file is created or read.
APOC then evaluates its common Hirshfeld/CM5 and Mayer definitions. The
Gaussian log adapter remains available as an independent check when
`Pop=Hirshfeld IOp(6/80=1)` was requested.

State tracking uses a one-to-one maximum-overlap assignment. It reports the
selected overlap, phase, runner-up overlap and assignment margin. Energy order
is diagnostic only: it never defines electronic-state identity.

For TD scans and optimizations, `follow_excited_state` exposes the same rule as
a programmatic step guard. A failed minimum-overlap or ambiguity test raises
`StateContinuityError` in strict mode, allowing the driver to reject the step,
reduce its size or request additional roots. This makes root following
independent of backend-specific root numbering.

`natural_orbitals(D, S)` accepts any correlated one-particle density, not only
a mean-field density. `recanonicalize_orbitals(C, F, S)` diagonalizes the Fock
operator only inside equal-occupation subspaces by default; consequently the
natural-orbital density is unchanged and the resulting orbitals can be handed
to another code through TREXIO.

ORACLE's L0 population acquisition normally uses PBE0/def2-TZVP and the native
def2 ECP where required.  The future L1 acquisition is MP2 with a cardinal
basis family still to be fixed.  Both levels preserve the complete APOC
electronic state, one-particle density, natural orbitals and natural
occupations.  Both natural-orbital spaces can seed PNO work.  L0 still needs a
subsequent pair-correlation construction, while L1 additionally retains the
MP2 T2 amplitudes and pair densities and is therefore directly PNO-ready.
The correlated MP2 rung is also the preferred one for cardinal-basis
extrapolations and related convergence models.  MP2 avoids DFT quadrature,
but Hirshfeld integration itself remains numerical at both levels.

ORACLE may consume APOC data in place of its geometry-only estimates.
ARCHITECT/ZAFF requires the APOC CM5/Mayer contract by default. Alternative
charge or bond-order definitions must be supplied and labelled explicitly;
they never silently replace CM5 or Mayer.

### Dyson orbitals for molecular continua

`DysonOrbital` is APOC's common real Dyson representation. It binds signed AO
coefficients to an `ElectronicState`, with removal energy (eV), pole strength,
spin channel, electronic method, channel identifier and source. Construction
checks the AO-metric squared norm against the pole strength. It neither
normalises the orbital nor multiplies it by an additional pole-strength factor.
`DysonOrbital.from_mo` transforms a complete canonical-MO vector using the
existing state's AO convention. SPARKS consumes it with
`AODysonOrbital.from_apoc(dyson)`; electronic-file parsing belongs to APOC.

`read_gdv_ept_dysons(log, state, frozen_core=..., method=...)` is the GDV EPT
adapter. It reads complete signed `SPARKS_EPT_DYSON_V1` diagnostic blocks from
L909, not the thresholded, rounded component weights in ordinary output. Each
block contains symmetry/root and active dimensions, pole energy and norm, every
active MO index/energy/coefficient, and a terminator. Frozen-core count is
explicit. The adapter checks completeness, index uniqueness, orbital energies
against the checkpoint and the resulting AO norm. Current support is restricted
real electron-removal poles with no omitted virtual orbitals.

Import the electronic state **from the same run's checkpoint**. Energy checks
cannot detect independent MO phase flips between different SCF runs. The
producer phase is retained; channel and phase tracking between displaced
geometries is not performed by this reader. H2O NR2 import and the SPARKS
dipole-source bridge were qualified on 26 October 2026: the AO norm agrees with the exported pole strength within
5.70e-10. See `docs/reports/sparks_dyson_20261026`. This qualification does not
establish continuum convergence or a physical FCHT spectrum.

For geometry derivatives, `reorient_electronic_state` and
`reorient_dyson_orbital` put each checkpoint into the **intended displaced
geometry's frame**, using a proper rigid transform. Geometry residuals and the
AO overlap metric are checked; p/d/... AO tensors rotate with their centres.
The transform is returned as explicit provenance. The helper does not fit a
displaced molecule to its equilibrium shape. `follow_dyson_orbital` selects an
isolated channel by normalized cross-AO overlap and fixes its overall sign,
retaining the candidate's original pole strength. Ambiguous or low-overlap
matches fail. `cross_ao_overlap` reuses the existing PySCF integral backend;
callers may supply cross-overlap matrices from another backend instead.
