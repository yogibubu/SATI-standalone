TRINITY anharmonicity and Raman diagnostics
=========================================

Anharmonicities
--------------

``sonic_vpt2_x_diagnostics(field)`` returns the normal-mode frequencies,
``x_matrix_cm``, and every additive contribution to the upper triangle.
Each contribution contains zero-based matrix and mode indices, the expression,
its numerator and denominator, potential/kinetic/mixed parts, and running sum.
The production ``sonic_vpt2_x_matrix`` uses the same term generator.
``result.write_json(path)`` preserves all terms, including exact zeros.
Resonant denominators require a polyad; no singular term is silently omitted.
The decomposition is specific to the fixed modal representation supplied.

Cartesian Raman input
---------------------

``decode_polarizability_derivatives(raw, natoms)`` examines Cartesian-major
and component-major FCHK orderings, validates rigid-translation sum rules,
and returns a canonical 6 x 3N Jacobian. Components are xx, xy, yy, xz, yz, zz.
``polarizability_jacobian_from_gaussian_fchk(path)`` reads the corresponding
FCHK block. The return object also records candidate residuals.
An explicit ``layout`` must pass the same sum rule. Two distinct valid
orderings require an explicit choice; equivalent zero arrays are accepted.
``validate_cartesian_polarizability_jacobian`` checks already decoded arrays.
The KeyMaker Cartesian polarizability acquisition invokes this validator
before transformation to internal coordinates at each evaluated point.
These sum rules apply to Cartesian derivatives, not to modal derivative sums.

Independent numerical controls
------------------------------

``benchmarks/trinity_tars_extended_check.py --tars-root PATH --output FILE``
compares synthetic two-, three- and four-mode fields with the supplied TARS:
potential, kinetic and mixed x matrices, explicit-sum fundamentals, resonant
Hamiltonians and eigenvalues, Cartesian IR/Raman moments, and an independently
mixed Fermi-polyad property calculation. Three-quantum differences are retained.
The supplied TARS has a normalization discrepancy for |1_i,2_j>; a separate
normalized-oscillator calculation checks the TRINITY electrical contribution.
No curvilinear intensity implementation is attributed to TARS.

``benchmarks/trinity_raman_translation_check.py --data-root PATH --output FILE``
checks all thirteen saved H2O Cartesian polarizability Jacobians and replays
the ESALC-C/I acquisition and Raman calculation after rigid translations.
It submits no electronic jobs. Separate fixed-force-field calculations isolate
the property acquisition; modal phases are matched. Geometry derivative steps
1e-4 and 1e-3 are both retained to expose finite-difference roundoff. Frequency
and Raman criteria are reported separately. The synthetic formula comparisons
and molecular translation replay do not establish experimental accuracy.
