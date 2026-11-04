"""Valence ESALCs and a diagnostic delocalized chart for vibronic calculations.

The ORACLE primitive inventory is reused; SOOP normals are not extrapolated
between different electronic-state minima. This is an explicit alternative
chart, not a silent change to ordinary SMITH consumers.
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
from .models import FrozenGIC
from .definition_serialization import read_gic_definition_from_xyzin
from .oracle_contract_consumer import load_validated_oracle_candidate_pool
from .numerics import _analytic_b_row, _primitive_value


def build_vibronic_valence_chart(path, *, rank_relative_tolerance=1e-10):
    """Diagnostic reduction of the full ORACLE valence pool by Wilson-B SVD.

    Production uses build_vibronic_esalc_chart, retaining chemical families
    and explicit torsions. This comparator does not replace that chart.

    Bonds, angles and proper dihedrals are used first. Other angular primitives
    complete missing directions only when necessary (e.g. pyramidal inversion).
    Dihedrals are continued about their reference, before delocalization.
    Coefficients and reference remain fixed for both electronic states.
    Every evaluation still needs a full-rank/conditioning audit at that geometry.
    Linear molecules and changing connectivity are outside this chart factory.
    """
    if not np.isfinite(rank_relative_tolerance) or not 0 < rank_relative_tolerance < 1:
        raise ValueError('rank_relative_tolerance must lie in (0,1)')
    path = Path(path)
    pool = load_validated_oracle_candidate_pool(path)
    base = read_gic_definition_from_xyzin(path)
    xyz = np.asarray(pool.coordinates_angstrom)
    target = xyz.size-6
    primitives = tuple(p for p in pool.candidates if p.function in {'R','A','D'})

    def matrix(items):
        return np.array([_analytic_b_row(p, xyz) for p in items])

    b = matrix(primitives)
    if not len(b): raise ValueError('ORACLE valence pool is empty')
    singular = np.linalg.svd(b, compute_uv=False)
    if np.count_nonzero(singular > singular[0]*rank_relative_tolerance) < target:
        primitives += tuple(p for p in pool.candidates if p.function in {'U','H','L'})
        b = matrix(primitives)
    u, singular, _ = np.linalg.svd(b, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0]*rank_relative_tolerance))
    if rank != target:
        raise ValueError(f'ORACLE valence pool has rank {rank}, expected {target}; a regular nonlinear-molecule chart is required')
    primitives = tuple(replace(p, chart='PERIODIC_CONTINUATION',
        chart_reference_radian=float(_primitive_value(p, xyz))) if p.function=='D' else p for p in primitives)
    coefficients = u[:, :target].T
    # Canonical signs only; degenerate subspace rotations do not change the
    # nonlinear chart span. No physical coupling or coefficient is thresholded.
    for row in coefficients:
        if row[np.argmax(abs(row))] < 0: row *= -1
    gics = tuple(FrozenGIC(f'VDIC{k+1:04d}', f'VDIC{k+1:04d}', 'DELOCALIZED', '',
        primitives[0].identifier, 'NONE', tuple((p.identifier,float(c)) for p,c in zip(primitives,row,strict=True) if c != 0))
        for k,row in enumerate(coefficients))
    return replace(base, backend='matrix-smith-vibronic-valence-dic.v1', symmetrize=False,
        primitives=primitives, gics=gics, rank=rank, candidate_count=len(primitives),
        symmetry_diagnostics=None, reduction_diagnostics=None, periodic_coordinate_estimates=(),
        reference_coordinates_angstrom=pool.coordinates_angstrom,
        semantic_diagnostics=(*base.semantic_diagnostics,
            'VIBRONIC_CHART FIXED_FULL_ORACLE_VALENCE_SVD',
            f'VIBRONIC_RANK_RELATIVE_TOLERANCE {rank_relative_tolerance:.17g}'))


__all__ = ['build_vibronic_valence_chart']


def audit_vibronic_torsion_retention(definition, *, coordinates_angstrom=None):
    """Check that bridge-rotor directions remain in explicit torsional GICs.

    Rotor bonds come from the existing SMITH periodic-coordinate contract;
    connectivity comes from its retained valence stretches. No atom indices or
    rotor count are supplied by a molecule-specific rule. Cyclic torsions need
    a constrained path and are reported separately from bridge rotors.
    """
    from .evaluation import build_gic_b_matrix
    xyz=np.asarray(definition.reference_coordinates_angstrom if coordinates_angstrom is None else coordinates_angstrom,float)
    bonds=sorted({tuple(sorted(bond)) for item in definition.periodic_coordinate_estimates
        if item.family=='TORSION' for bond in item.central_bonds})
    edges={tuple(sorted(p.atoms)) for p in definition.primitives if p.function=='R' and len(p.atoms)==2}
    torsion_rows=[k for k,g in enumerate(definition.gics) if g.family=='TORSION']
    matrix=np.asarray(build_gic_b_matrix(definition,coordinates_angstrom=xyz).rows)
    velocities=[];records=[]
    for left,right in bonds:
        adjacent={k:set() for k in range(1,len(xyz)+1)}
        for a,b in edges-{(left,right)}:adjacent[a].add(b);adjacent[b].add(a)
        component={left};front=[left]
        while front:
            atom=front.pop()
            for other in adjacent[atom]-component:component.add(other);front.append(other)
        if right in component:
            records.append(dict(central_bond=[left,right],kind='CYCLIC_REQUIRES_CONSTRAINED_PATH'))
            continue
        dihedrals=[p for p in definition.primitives if p.function=='D' and set(p.atoms[1:3])=={left,right}]
        if not dihedrals:raise ValueError(f'protected rotor bond {left}-{right} has no retained dihedrals')
        axis=xyz[right-1]-xyz[left-1];axis/=np.linalg.norm(axis)
        velocity=np.zeros_like(xyz)
        for atom in component:velocity[atom-1]=np.cross(axis,xyz[atom-1]-xyz[left-1])
        velocities.append(velocity.ravel())
        records.append(dict(central_bond=[left,right],kind='BRIDGE_ROTOR',retained_dihedrals=len(dihedrals)))
    if velocities:
        response=matrix@np.column_stack(velocities)
        torsion_response=response[torsion_rows]
        full_rank=int(np.linalg.matrix_rank(response));torsion_rank=int(np.linalg.matrix_rank(torsion_response)) if len(torsion_response) else 0
        if torsion_rank != len(velocities) or full_rank != len(velocities):
            raise ValueError('explicit torsional GICs do not retain every protected bridge-rotor direction')
        singular=np.linalg.svd(torsion_response,compute_uv=False).tolist()
    else:full_rank=torsion_rank=0;singular=[]
    return dict(protected_bonds=records,bridge_rotor_count=len(velocities),retained_motion_rank=full_rank,
        explicit_torsional_motion_rank=torsion_rank,torsional_response_singular_values=singular)


def build_vibronic_esalc_chart(path):
    """Canonical symmetry-adapted valence chart, before SOOP materialization.

    Preserve SMITH's chemical families and explicit torsional GICs. A frozen
    Cartesian SOOP chart is a local harmonic representation, not the same
    finite-displacement chart as these curvilinear valence ESALCs.
    """
    from .definition import build_gic_definition_from_xyzin
    definition=build_gic_definition_from_xyzin(Path(path),symmetrize=True)
    if any(p.function in {'SOOP','SYCART'} for p in definition.primitives):
        raise ValueError('a curvilinear valence chart is required for finite vibronic displacements')
    audit_vibronic_torsion_retention(definition)
    return replace(definition,semantic_diagnostics=(*definition.semantic_diagnostics,
        'VIBRONIC_CHART CANONICAL_VALENCE_ESALC_BEFORE_SOOP',
        'VIBRONIC_TORSIONS EXPLICIT_BRIDGE_ROTOR_DIRECTIONS_RETAINED'))


__all__ += ['build_vibronic_esalc_chart','audit_vibronic_torsion_retention']
