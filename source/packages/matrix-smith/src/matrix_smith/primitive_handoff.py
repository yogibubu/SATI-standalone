"""Bind SMITH expressions to frozen ORACLE measurements, failing closed."""
from dataclasses import replace
from .contracts import GICForgeContractError


_SMITH_DERIVED_FUNCTIONS = frozenset({"SOOP", "SYCART"})


def measurement_key(primitive):
    function = primitive.function
    if function in {"IR", "IR3", "ER"}:
        function = "R"
    atoms = tuple(primitive.atoms)
    if function == "R":
        atoms = tuple(sorted(atoms))
    elif function in {"A", "D"}:
        atoms = min(atoms, atoms[::-1])
    elif function == "L" and len(primitive.ref_atoms) == 1:
        # Reversal preserves L(-1); L(-2) becomes 2*pi-L(-2).
        # Retain the requested expression below, with the same frozen gauge.
        atoms = min(atoms, atoms[::-1])
    elif function == "H":
        atoms = (*sorted(atoms[:3]), atoms[3])
    elif function == "U" and len(atoms) == 4:
        atoms = (atoms[0], *sorted(atoms[1:3]), atoms[3])
    ref = tuple(primitive.ref_atoms)
    if function in {"FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST", "FTRANS", "FROT"}:
        atoms, ref = tuple(sorted(atoms)), tuple(sorted(ref))
    return (function, atoms, primitive.mode, ref,
            tuple(primitive.frame_atoms), tuple(primitive.ref_frame_atoms))


def bind_oracle_measurements(definition, pool, *, allow_missing: bool = False):
    """Every used elementary term must have a frozen ORACLE source.

    Nonlinear distance maps remain SMITH functions of an ORACLE distance.
    This check concerns the terms used in the chart, not redundant candidates
    which a rank-revealing construction is entitled to discard.
    """
    sources = {}
    for primitive in pool.candidates:
        sources.setdefault(measurement_key(primitive), primitive)
    used = {identifier for gic in definition.gics
            for identifier, coefficient in (gic.coefficients or ((gic.primitive_id, 1.),))
            if coefficient != 0.}
    identifiers = {p.identifier for p in definition.primitives}
    if len(identifiers) != len(definition.primitives):
        raise GICForgeContractError("SMITH primitive identifiers must be unique")
    if used - identifiers:
        raise GICForgeContractError(
            "SMITH expressions reference undefined primitives: " +
            ", ".join(sorted(used - identifiers))
        )
    missing = [p for p in definition.primitives
               if p.identifier in used
               and p.function not in _SMITH_DERIVED_FUNCTIONS
               and measurement_key(p) not in sources]
    if missing and not allow_missing:
        details = "; ".join(f"{p.function}{p.atoms} mode={p.mode} ref={p.ref_atoms} frames={p.frame_atoms}/{p.ref_frame_atoms} [{p.family}]" for p in missing)
        raise GICForgeContractError("SMITH expressions require missing ORACLE measurements: " + details)
    primitives = []
    signs = {}
    for primitive in definition.primitives:
        source = sources.get(measurement_key(primitive))
        if source is None:
            primitives.append(primitive)
            continue
        def parity(atoms):
            return -1. if sum(a > b for i, a in enumerate(atoms) for b in atoms[i+1:]) % 2 else 1.
        if primitive.function == "H":
            signs[primitive.identifier] = parity(primitive.atoms[:3]) * parity(source.atoms[:3])
        elif primitive.function == "U" and len(primitive.atoms) == 4:
            signs[primitive.identifier] = parity(primitive.atoms[1:3]) * parity(source.atoms[1:3])
        primitives.append(replace(primitive,
            atoms=(tuple(sorted(source.atoms)) if primitive.function in {"R", "IR", "IR3", "ER"}
                   else primitive.atoms if primitive.function == "L" else source.atoms), mode=source.mode,
            ref_atoms=source.ref_atoms, frame_atoms=source.frame_atoms,
            ref_frame_atoms=source.ref_frame_atoms,
            provenance=(primitive.provenance if primitive.provenance in {"USER", "ORACLE_TS_CONTRACT"}
                        else source.provenance + f":ORACLE_PRIMITIVE={source.identifier}")))
    gics = tuple(replace(gic, coefficients=tuple((identifier, coefficient * signs.get(identifier, 1.))
                       for identifier, coefficient in (gic.coefficients or ((gic.primitive_id, 1.),))))
                 for gic in definition.gics)
    return replace(definition, primitives=tuple(primitives), gics=gics,
                   semantic_diagnostics=definition.semantic_diagnostics +
                   (f"ORACLE_PRIMITIVE_HANDOFF USED_TERMS={len(used)} "
                    f"UNBOUND_TERMS={len(missing)} "
                    f"POLICY={'DEFERRED_SYCART' if missing and allow_missing else 'FAIL_CLOSED'}",))
