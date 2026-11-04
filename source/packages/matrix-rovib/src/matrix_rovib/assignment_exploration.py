"""QM-seeded coarse catalogues preserving the physical LAM and symmetry core."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class AssignmentExploration:
    surface: object
    operator_manifest: object | None
    audit: dict


def build_assignment_exploration(surface, *, rotational_order=2,
                                 operator_manifest=None, essential_parameters=()):
    """Suppress only pure rotational corrections above a declared order.

    Entire shared-parameter groups are kept if any term carries momentum,
    a nonconstant LAM function, or an explicitly essential parameter. No
    coordinate, kinetic metric, spin surface or symmetry action is removed.
    Suppressed manifest terms have zero prefactors, retaining their registry.
    """
    from .general_lam_operator import normalize_general_lam_operator_manifest

    if isinstance(rotational_order, bool) or rotational_order not in (2, 4, 6, 8):
        raise ValueError('exploratory rotational order must be 2, 4, 6 or 8')
    changed = {}
    suppressed = []
    for field, order in (('quartic_distortion_MHz', 4), ('sextic_distortion_MHz', 6)):
        value = getattr(surface, field)
        if value is not None and order > rotational_order:
            changed[field] = np.zeros_like(value)
            suppressed.append(field)
    manifest = None
    essential = set(essential_parameters)
    if operator_manifest is not None:
        manifest = normalize_general_lam_operator_manifest(operator_manifest)
        if essential - manifest.parameters_MHz.keys():
            raise ValueError('essential parameter is absent from the manifest')
        protected = essential | {t.parameter for t in manifest.terms if
            len(t.rotation_word) <= rotational_order or t.momentum_axes or any(f != 'constant' and h != 0
                                  for f, h in zip(t.functions, t.harmonics, strict=True))}
        disabled = {t.parameter for t in manifest.terms
                    if len(t.rotation_word) > rotational_order} - protected
        # Keep whole parameter groups (potentially symmetry orbits), never
        # individual Cartesian members of a common physical coefficient.
        terms = tuple(replace(t, prefactor=0.) if t.parameter in disabled else t
                      for t in manifest.terms)
        manifest = replace(manifest, terms=terms)
        suppressed.extend('operator.' + p for p in sorted(disabled))
    elif essential:
        raise ValueError('essential parameters require an operator manifest')
    return AssignmentExploration(replace(surface, **changed), manifest, dict(
        schema='matrix.lock.assignment_exploration.v1', rotational_order=rotational_order,
        suppressed=suppressed, physical_LAM_core_preserved=True,
        symmetry_metadata_preserved=True, parameter_groups_preserved=True, spin_surfaces_preserved=True,
        scope='coarse assignment model; not a converged precision Hamiltonian'))


def exploration_catalog(model, *, j_values, lam_states=12, dipole_au=None,
                        temperature_K=None, prediction_uncertainty_MHz,
                        solve_options=None,spin_model=None,two_f_values=None):
    """Run the native solver and return persistent, indexed candidate lines.

    The prediction uncertainty is a declared QM/model matching scale, not the
    experimental sigma. Spin-resolved catalogues can instead enter the common
    candidate matcher directly with their own state IDs and selectors.
    """
    from .lam_rotation import solve_lam_rotation, lam_rotation_transitions

    scale = float(prediction_uncertainty_MHz)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('declare a positive finite QM prediction uncertainty')
    if spin_model is not None:
        from .assignment_spin import spin_exploration_catalog
        return spin_exploration_catalog(model.surface,spin_model,j_values=j_values,
            two_f_values=two_f_values,lam_states=lam_states,dipole_au=dipole_au,
            temperature_K=temperature_K,prediction_uncertainty_MHz=scale,
            solve_options=solve_options,operator_manifest=model.operator_manifest)
    if two_f_values is not None:raise ValueError('F blocks require an explicit spin model')
    if model.surface.spin_tensor_surfaces_MHz and any(
            np.any(v) for v in model.surface.spin_tensor_surfaces_MHz.values()):
        raise ValueError('spin-active models need a spin-resolved catalogue; do not drop spin')
    options = dict(solve_options or {})
    if {'operator_manifest', 'j_values', 'lam_state_count'} & options.keys():
        raise ValueError('duplicate exploration solver controls')
    result = solve_lam_rotation(model.surface, j_values=j_values,
        lam_state_count=lam_states, operator_manifest=model.operator_manifest, **options)
    rows = lam_rotation_transitions(model.surface, result, dipole_au=dipole_au,
        temperature_K=temperature_K, include_level_indices=True)
    catalogue = []
    for row in rows:
        selectors = {}
        for side, suffix in (('lower', 'l'), ('upper', 'u')):
            j = int(row['J' + suffix]); index = int(row[side + '_level_index'])
            selectors[side] = dict(J=j, reference_index=index,
                state_id=f'J={j}:reference={index}',
                reference_energy_MHz=float(result.blocks[j].energies_MHz[index]))
        family = '|'.join(str(row[k]) for k in (
            'Branch', 'vt_l', 'vt_u', 'tors_sym_l', 'tors_sym_u',
            'rovib_irrep_l', 'rovib_irrep_u'))
        catalogue.append(dict(id=f"{selectors['lower']['state_id']}->{selectors['upper']['state_id']}",
            frequency_MHz=row['Frequency (MHz)'], uncertainty_MHz=scale,
            relative_intensity=float(row['Relative intensity']) / 100., family=family,
            **selectors))
    return result, catalogue


__all__ = ['AssignmentExploration', 'build_assignment_exploration', 'exploration_catalog']
