"""LINK-owned transition-state certification and KINETICS handoff."""

from __future__ import annotations

from pathlib import Path
import json

from .optimizer import IRC_VERIFICATION_SCHEMA, OptimizerResult


LINK_TS_HANDOFF_SCHEMA = "matrix.link.transition_state_handoff.v1"


def write_transition_state_handoff(
    path: Path | str,
    result: OptimizerResult,
    *,
    transition_state_id: str,
    reactant_id: str,
    product_id: str,
    geometry_path: Path | str,
    provenance: dict[str, object] | None = None,
) -> Path:
    """Publish a TS only after LINK has certified its Hessian and IRC branches."""

    if result.settings.stationary_point != "transition_state":
        raise ValueError("only a LINK transition-state optimization can create this handoff")
    frequencies = tuple(float(value) for value in result.final_frequencies_cm)
    imaginary = tuple(value for value in frequencies if value < -1.0e-6)
    irc = result.irc_verification or {}
    validation = {
        "converged": bool(result.converged),
        "exact_hessian": bool(result.exact_final_hessian),
        "one_imaginary_frequency": len(imaginary) == 1,
        "exact_mode_overlap": bool(result.final_convergence.get("exact_mode_overlap")),
        "irc_verified": (
            irc.get("schema") == IRC_VERIFICATION_SCHEMA and irc.get("verified") is True
        ),
        "endpoints_distinct": irc.get("endpoints_distinct") is True,
    }
    failed = [name for name, passed in validation.items() if not passed]
    if failed:
        raise ValueError("LINK cannot publish an invalid transition state: " + ", ".join(failed))
    required_paths = {
        "geometry_path": Path(geometry_path).expanduser().resolve(),
        "hessian_path": (
            Path(result.final_cartesian_hessian_path).expanduser().resolve()
            if result.final_cartesian_hessian_path is not None
            else Path()
        ),
        "irc_path": (
            Path(result.irc_path).expanduser().resolve()
            if result.irc_path is not None
            else Path()
        ),
    }
    missing = [name for name, candidate in required_paths.items() if not candidate.is_file()]
    if missing:
        raise FileNotFoundError("LINK transition-state artifacts missing: " + ", ".join(missing))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": LINK_TS_HANDOFF_SCHEMA,
        "transition_state_id": str(transition_state_id),
        "reactant_id": str(reactant_id),
        "product_id": str(product_id),
        "imaginary_frequency_cm-1": imaginary[0],
        **{name: str(candidate) for name, candidate in required_paths.items()},
        "validation": validation,
        "provenance": {
            "owner": "LINK",
            "optimizer_summary": str(result.summary_path),
            **dict(provenance or {}),
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = ["LINK_TS_HANDOFF_SCHEMA", "write_transition_state_handoff"]
