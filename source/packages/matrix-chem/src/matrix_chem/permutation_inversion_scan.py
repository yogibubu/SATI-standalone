"""Certify deformable bridge-rotor permutations on a complete labelled scan.

Graph automorphisms only propose candidates. Admission requires a unique
proper-rotation geometric match at every point AND a uniform translation of
one periodic scan axis. A permutation that is only a graph symmetry is never
accepted. Budgets are explicit and exhausted searches fail without truncation.
"""
from dataclasses import replace
import hashlib

import networkx as nx
import numpy as np

from .permutation_inversion import (
    InternalRotorPermutation, PermutationInversionOperation, PIExpansionLimit, _closure,
)
from .permutation_inversion_grid import derive_permutation_inversion_grid_action


def extend_permutation_inversion_from_scan(
    contract, geometries_angstrom, masses_amu, *, grid_shape,
    tolerance_angstrom=2e-4, max_candidates_per_top=256, max_operations=4096,
):
    """Return a shared ORACLE contract certified on a periodic Cartesian grid.

    ``grid_shape`` is supplied by the coordinate producer (SMITH), in flattened
    C order. No group, rotor order, permutation or torsion name is supplied.
    The scan hash records the actual samples; it is not evidence of a QM method
    or an energetically accessible path. Those remain producer provenance.
    """
    shape = tuple(grid_shape)
    if not shape or any(type(n) is not int or n < 3 for n in shape):
        raise ValueError("scan certification requires periodic grid axes of at least three samples")
    if type(max_candidates_per_top) is not int or max_candidates_per_top < 1:
        raise ValueError("candidate budget must be a positive integer")
    if not contract.labelled_bonds:
        raise ValueError("scan certification requires the frozen ORACLE labelled bond graph; rebuild the PI contract")
    xyz = np.asarray(geometries_angstrom, dtype=float)
    size = int(np.prod(shape)); n = len(contract.atoms)
    if xyz.shape != (size,n,3) or not np.all(np.isfinite(xyz)):
        raise ValueError("scan shape and finite Cartesian samples disagree")
    # Validate the existing group on the actual scan before extending it.
    derive_permutation_inversion_grid_action(contract, xyz, masses_amu, tolerance_angstrom=tolerance_angstrom)
    digest = hashlib.sha256(np.asarray(shape,dtype='<i8').tobytes()+np.asarray(xyz,dtype='<f8').tobytes()).hexdigest()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for a,b,label in contract.labelled_bonds:
        graph.add_edge(a,b,label=label)
    if not nx.is_connected(graph):
        raise ValueError("scan rotor certification requires a connected ORACLE graph")
    indices = np.array(np.unravel_index(np.arange(size),shape)).T
    generators, sources, rotors = list(contract.generators), list(contract.generator_sources), list(contract.rotors)
    diagnostics = list(contract.rotor_diagnostics)
    certified_axes = {rotor.axis for rotor in rotors}
    identity = PermutationInversionOperation(tuple(range(n)))
    for a,b in sorted(nx.bridges(graph)):
        if graph.edges[a,b]['label'] != 'SINGLE':
            continue
        cut = graph.copy(); cut.remove_edge(a,b)
        for fixed,moving in ((a,b),(b,a)):
            if (fixed,moving) in certified_axes:
                continue
            top = tuple(sorted(nx.node_connected_component(cut,moving)))
            if len(top) < 3:
                continue
            axis_vector = xyz[0,moving] - xyz[0,fixed]
            length = np.linalg.norm(axis_vector)
            if length <= tolerance_angstrom:
                continue
            axis_vector /= length
            relative = xyz[0] - xyz[0,moving]
            radial = relative - (relative @ axis_vector)[:,None]*axis_vector
            radii = np.linalg.norm(radial,axis=1)
            frame = [atom for atom in range(n) if atom not in top]
            if max(radii[list(top)]) <= tolerance_angstrom or max(radii[frame]) <= tolerance_angstrom:
                continue
            anchored = graph.copy()
            for atom in anchored:
                anchored.nodes[atom]['label'] = (contract.atoms[atom],contract.nuclear_labels[atom],
                    atom if atom not in top or atom == moving else -1)
            matcher = nx.algorithms.isomorphism.GraphMatcher(anchored,anchored,
                node_match=nx.algorithms.isomorphism.categorical_node_match('label',None),
                edge_match=nx.algorithms.isomorphism.categorical_edge_match('label',None))
            accepted = []
            for count,mapping in enumerate(matcher.isomorphisms_iter(),1):
                if count > max_candidates_per_top:
                    raise PIExpansionLimit(f"scan candidate budget exhausted for bridge {fixed},{moving}")
                op = PermutationInversionOperation(tuple(mapping[i] for i in range(n)))
                if op == identity or any(op.permutation[i] == i and radii[i] > tolerance_angstrom for i in top):
                    continue
                operations,words = _closure(n,(op,),max_operations)
                cyclic = replace(contract,generators=(op,),generator_sources=('SCAN_CANDIDATE',),
                    operations=operations,words=words,rotors=())
                try:
                    action = derive_permutation_inversion_grid_action(cyclic,xyz,masses_amu,tolerance_angstrom=tolerance_angstrom)
                except ValueError:
                    continue  # Not a symmetry of this complete labelled scan.
                delta = (indices[action.targets[1]]-indices) % np.asarray(shape)
                if not np.all(delta == delta[0]) or np.count_nonzero(delta[0]) != 1:
                    continue
                axis = int(np.flatnonzero(delta[0])[0])
                step = int(delta[0,axis])
                if shape[axis] // np.gcd(shape[axis],step) != len(operations):
                    continue
                accepted.append((len(operations),step,op,action.maximum_rmsd_angstrom))
            if not accepted:
                diagnostics.append(f"AXIS:{fixed},{moving}:NO_CERTIFIED_SCAN_TRANSLATION")
                continue
            # Largest cyclic subgroup; smallest positive shift chooses its
            # primitive orientation, independent of GraphMatcher traversal.
            order,step,op,residual = sorted(accepted,key=lambda row:(-row[0],row[1],row[2].permutation))[0]
            rotors.append(InternalRotorPermutation((fixed,moving),top,order,op,residual,'scan_translation',digest))
            if op not in generators:
                generators.append(op); sources.append(f"SCAN_ROTOR:{fixed},{moving}:C{order}:{digest}")
    operations,words = _closure(n,tuple(generators),max_operations)
    extended = replace(contract,generators=tuple(generators),generator_sources=tuple(sources),
        operations=operations,words=words,rotors=tuple(rotors),rotor_diagnostics=tuple(diagnostics),
        tolerance_angstrom=max(contract.tolerance_angstrom,float(tolerance_angstrom)))
    # The combination must also close geometrically; independently valid
    # generators are not sufficient if their errors accumulate beyond tolerance.
    derive_permutation_inversion_grid_action(extended,xyz,masses_amu,tolerance_angstrom=tolerance_angstrom)
    return extended
