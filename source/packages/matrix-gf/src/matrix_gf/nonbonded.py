"""Legacy nonbonded-Hessian exports.

The analytic kernel remains in :mod:`matrix_chem` for compatibility. ORACLE
owns its perception inputs, whereas ARCHITECT/ZAFF owns force-field selection,
parameterization and runtime use.
"""

from matrix_chem.nonbonded import (
    NonbondedHessianComponents,
    nonbonded_cartesian_hessian_components,
    nonbonded_cartesian_hessian_correction,
    preferred_atomic_charges_from_xyzin,
    synthon_charges_from_xyzin,
    topology_bonds_from_xyzin,
)

__all__ = [
    "NonbondedHessianComponents",
    "nonbonded_cartesian_hessian_components",
    "nonbonded_cartesian_hessian_correction",
    "preferred_atomic_charges_from_xyzin",
    "synthon_charges_from_xyzin",
    "topology_bonds_from_xyzin",
]
