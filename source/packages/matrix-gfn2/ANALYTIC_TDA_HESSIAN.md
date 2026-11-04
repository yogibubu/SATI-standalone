# Molecular analytic TDA Hessians

The molecular producer now supplies the relaxed second derivative of the TDA
matrix to the existing analytic root-response operator. Both closed-shell
sTDA-xTB and restricted-reference SF-TDA are available through `GFN2Calculator`.
The methods use zero-based root indices.

```python
ground = calculator.evaluate(coordinates_angstrom, retain_orbitals=True,
                             compute_cm5=False, compute_bond_orders=False)
operator = calculator.closed_shell_stda_hessian(
    ground.orbital_state, root=0, spectral_range_ev=7.0)
# For the open-shell reference instead:
# operator = calculator.sf_tda_hessian(ground.orbital_state, root=0)

H_final = operator.materialize(symmetrize=False)  # Ha/bohr^2
H_times_v = operator.matvec(direction_bohr)       # flat 3N vector
g_final = (ground.gradient_hartree_per_bohr
           + operator.excitation_gradient_hartree_per_bohr)
```

`materialize(symmetrize=False)` exposes reciprocity errors for qualification.
The default materialization symmetrizes rounding-level residuals. For a fixed
orthonormal Cartesian chart, `operator.project(basis)` uses the existing
projected-operator contract. A mass-weighted basis is generally not Cartesian
orthonormal: form its force constants as

```python
F_final = Q.T @ np.column_stack([operator.matvec(Q[:, k])
                               for k in range(Q.shape[1])])
```

For a VH calculation, evaluate the ground-state minimum once and use its own
Hessian for the initial force field and this **total final-state Hessian at
the same geometry** for the final force field. The final gradient remains
explicit. Feed those fields to TRINITY/SATI with `pes_model="VH"`. Analyze
the final curvature before applying a bound harmonic spectrum: negative modes
are not converted to positive frequencies. A coordinate-dependent chart also
needs its gradient-dependent second transformation term; the constant-chart
projection above does not construct that term.

## Analytic chain

For each direction, native AO recurrences provide `S_Rv`, dipole and second
moment derivatives. Internal NumPy mixed jets propagate first and mixed
derivatives through H0, coordination, shell SCC and third-order terms,
cumulative multipoles, AES/AXC and charge-dependent D4. The differentiated
generalized eigenproblem solves a second, mixed CPSCF response using the
existing Fock-density Jacobian and the same orbital-rotation response matrix.
This includes metric normalization and canonical orbital-energy response.

The complete TDA matrix is then differentiated, including Lowdin square roots,
orbital gaps, both semiempirical kernels, the native diagonal shift and the
rejected-configuration perturbative lowering. Parent orbital windows,
primary/secondary membership and external configurations remain frozen at
the reference geometry. The root Hessian contracts

```
omega_Rv = X.T A_Rv X + 2 X.T A_R X_v
H_final v = H_ground v + omega_Rv
```

Geometrically displaced SCF calculations enter the independent numerical
qualification only. The Hessian implementation has no finite-difference step.
Primal Fock, first Fock/TDA derivatives, mixed eigenproblem and normalization
residuals are checked while evaluating the operator. Supplied roots must match
the differentiated Hamiltonian; posterior energy shifts are rejected as
derivative inputs. Operators retain their reference even after the calculator
evaluates another geometry.

## ALPB

Configure the same solvent as for the fundamental state:

```python
calculator = GFN2Calculator(atoms, alpb_solvent="water",
                            scf_energy_tolerance_hartree=1e-11,
                            scf_wavefunction_tolerance_electron=1e-9,
                            scf_mixer_tolerance=1e-9)
```

ALPB enters the self-consistent reference and both CPSCF response orders.
The mixed producer differentiates the same Born radii, p16 screening,
electrostatic size and Lebedev cavity/hydrogen-bond charge kernel as the
ground-state ALPB functional. This supplies `K_Rv q`, the mixed geometry-charge
terms and charge relaxation. Charge-independent cavity/dispersion and
reference-state terms remain in the existing complete ground-state Hessian.
The existing optional dependency for ground-state analytic ALPB Hessians is
unchanged; the new mixed kernel itself uses NumPy and the standard library.
The semiempirical TDA interaction keeps its declared native definition.

## Current scope

This is a dense molecular reference producer. It requires integer occupations,
a complete canonical AO/MO basis, resolved canonical orbital gaps and an
isolated TDA root. Sparse D4 geometry, NN overlays and fixed external point
charges are explicitly rejected. The latter is distinct from ALPB: the
pre-existing dense first-response and complete ground-Hessian routes need
additional fixed-charge embedding work before that case can be qualified.

Oracle evidence and source/binary hashes are in
`docs/reports/tda_vh_analytic_20261022/`. Molecular derivative qualification
does not establish excitation-energy or experimental spectral accuracy.

## Complete VH application

`docs/reports/tda_vh_case_audit_20261022/` supplies the full 24-mode pyrazine
S1 n-pi* VH-FCHT calculation in vacuum and ALPB water, with positive final
curvatures, independent reconverged energy-curvature checks, Duschinsky mixing
and TI/TD qualification at 0/300 K. The raw native gaps are negative; only the
final spectrum receives the previously fitted constant n-pi* endpoint.
No coordinate-dependent energy correction or its derivatives enter the
analytic force fields. The dense molecular qualification and this application
do not establish sparse asymptotic scaling or experimental spectral accuracy.
