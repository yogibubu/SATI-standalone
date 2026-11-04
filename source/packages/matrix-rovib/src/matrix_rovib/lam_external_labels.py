"""External torsional translation labels resolved without molecule-name maps."""
from __future__ import annotations
from numbers import Integral


def product_translation_sector(labels,orders):
    labels,orders=tuple(labels),tuple(orders)
    if not labels or len(labels)!=len(orders) or any(
        isinstance(v,bool) or not isinstance(v,Integral) for v in labels+orders
    ) or any(n<1 for n in orders):
        raise ValueError('one integer translation label per positive fold is required')
    result=0
    for label,order in zip(labels,orders,strict=True):result=result*order+(label%order)
    return result


def resolve_lam_energy_rank(result,*,J,rank,translation_labels,orders,torsional_state):
    """Resolve declared 1-based within-band energy rank and physical sigmas.

    The band and rank convention must come from the source. This function
    never guesses a band from a measured frequency or permutes species to
    improve residuals. IDIAG/symmetry-block labels require their own importer.
    """
    if (any(isinstance(v,bool) or not isinstance(v,Integral) for v in (J,rank,torsional_state))
        or J<0 or rank<1 or torsional_state<0 or J not in result.blocks):
        raise ValueError('valid J, positive energy rank and nonnegative torsional band are required')
    sector=product_translation_sector(translation_labels,orders)
    candidates=sorted((level for level in result.blocks[J].levels
        if level.symmetry_sector==sector and level.torsional_state==torsional_state),
        key=lambda level:(level.energy_MHz,level.index))
    if rank>len(candidates):
        raise ValueError(f'energy rank {rank} exceeds retained band {torsional_state} in sector {sector}, J={J}')
    return candidates[rank-1]


def resolve_lam_erham_level(result,*,J,label,translation_labels,orders,torsional_state):
    """ERHAM IDIAG label: retain Ka and order its current asymmetry pair.

    Labels 1..2J+1 follow the canonical Ka families; Kc is assigned by energy
    within the family, including an inverted doublet. Global energy sorting
    across families and a molecule-specific species permutation are invalid.
    """
    if (any(isinstance(v,bool) or not isinstance(v,Integral) for v in (J,label,torsional_state))
        or J<0 or not 1<=label<=2*J+1 or torsional_state<0 or J not in result.blocks):
        raise ValueError('valid ERHAM J, label and torsional manifold are required')
    ka=label//2;kc=(2*J+2-label)//2
    sector=product_translation_sector(translation_labels,orders)
    candidates=sorted((level for level in result.blocks[J].levels
        if level.symmetry_sector==sector and level.torsional_state==torsional_state and level.Ka==ka),
        key=lambda level:(level.energy_MHz,level.index))
    expected=1 if ka==0 else 2
    if len(candidates)!=expected:
        raise ValueError(f'incomplete ERHAM Ka={ka} family in band {torsional_state}, sector {sector}, J={J}')
    return candidates[0 if ka==0 or kc==J-ka+1 else 1]
