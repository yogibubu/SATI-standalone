"""Declarative construction of a reaction-channel coordinate contract."""
from pathlib import Path
from matrix_core import read_sectioned_lines,section_content
from matrix_chem import preprocess_to_enriched_xyz,write_validation_section
from matrix_fragments import write_fragment_build_section
from .definition import write_sonic_build_sections_from_cartesian


def write_reaction_path_build_sections_from_cartesian(
    source: Path, target: Path, *, symmetry_group: str,
    source_kind: str = 'auto', sycart: bool = False,
):
    """Build at a chosen channel reference using its common symmetry cap.

    ``symmetry_group`` is an a priori maximum: use the lowest common symmetry
    retained along the channel, not an endpoint's possibly higher group.
    ORACLE creates missing chemical/TS contracts and SMITH builds the basis.
    Callers retain this definition for transport; they must not rebuild with
    higher endpoint symmetry. Cs selects a pointwise molecular mirror when
    available rather than an atom-exchanging endpoint-only mirror.
    """
    group=str(symmetry_group).strip()
    if not group:raise ValueError('an a priori reaction-channel symmetry_group is required')
    source=Path(source);target=Path(target)
    lines=read_sectioned_lines(source)
    required=('VALIDATION','TOPOLOGY','SYNTHONS','SYMMETRY')
    if not all(section_content(lines,key) for key in required):
        preprocess_to_enriched_xyz(source,target,source_kind=source_kind)
    elif source.resolve()!=target.resolve():
        target.write_text(source.read_text(encoding='utf-8'),encoding='utf-8')
    lines=read_sectioned_lines(target)
    if not section_content(lines,'FRAGMENTS'):write_fragment_build_section(target)
    write_validation_section(target)
    if not section_content(read_sectioned_lines(target),'ORACLE_SONIC_CONTRACT'):
        from matrix_oracle import write_oracle_sonic_contract_from_xyzin
        write_oracle_sonic_contract_from_xyzin(target)
    # The high-level workflow asks the owner to construct a TS atlas. The
    # lower-level SMITH builder retains its strict no-inference TS gate.
    if not section_content(read_sectioned_lines(target),'ORACLE_TRANSITION_STATE_GEOMETRY'):
        from matrix_oracle.transition_state_geometry import write_oracle_transition_state_geometry_contract_from_xyzin
        write_oracle_transition_state_geometry_contract_from_xyzin(target)
    return write_sonic_build_sections_from_cartesian(target,symmetrize=True,
        symmetry_group=group,sycart=sycart,fragment_context='transition_state')
