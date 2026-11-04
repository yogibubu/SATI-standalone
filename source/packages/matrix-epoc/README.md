# matrix-epoc

The [nonequilibrium solvation interface](../../docs/architecture/EPOC_NONEQUILIBRIUM_SOLVATION.md)
adds explicit fast/slow polarization, prepared-state records, analytic fixed-site
derivatives and APOC/SATI library interfaces. Local numerical and functional
qualification passes 122 tests; external solvent SCF backends and molecular
spectra require separate qualification.

`matrix-epoc` owns ellipsoidal boundaries, reaction fields, confinement, and
their analytic first- and second-order evaluations. Other MATRIX tools consume
this runtime through KEYMAKER-owned workflows.

The package owns four public artifact contracts:

- `matrix.epoc.boundary.v1`: ellipsoid geometry and boundary policy;
- `matrix.epoc.reaction_field.v1`: dielectric reaction-field configuration;
- `matrix.epoc.confinement.v1`: finite-domain confinement configuration;
- `matrix.epoc.evaluation.v1`: energy/gradient/Hessian evaluation results.

Legacy `matrix.zaff.*` records remain readable during migration, but are not
part of the EPOC writer contract.

The historical `matrix_zaff.{cpcm,confinement,variational_cavity}` and
`matrix_architect.{cpcm,confinement,variational_cavity}` modules are deprecated
import-only facades. They contain no scientific implementation and can be
removed after downstream consumers have completed one compatibility cycle.

## KEYMAKER solvent SCF

The [installed KEYMAKER protocol](../../docs/protocols/epoc-ellipsoidal-solvent-scf.md)
accepts explicit electronic methods and Gaussian bases. The JSON adapter supports
PySCF RHF/RKS equilibrium ground states; RHF/6-31G and B3LYP/6-31G have been
validated through the CLI on the small HF ellipsoid. FCI is an optional reference
test, not a protocol requirement. Other excited-state owners can connect via
`iterate_solvation` with certified total-state densities and continuity.

The [TDA-DFT / full TD-DFT provision](../../docs/protocols/epoc-ellipsoidal-solvent-td-dft.md)
adds explicit optical linear-response routes and a native response adapter.
Its separate KEYMAKER manifest is locally qualified for HF/6-31G with B3LYP/PBE,
including independent response matrices and vacuum controls. This is LR
absorption; the approved ground-state solvent SCF route remains available.
