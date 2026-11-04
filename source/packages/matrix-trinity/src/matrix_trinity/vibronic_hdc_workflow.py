"""Native ESALC/GF requests for completed vibrational and vibronic Hamiltonians."""

from copy import deepcopy
import hashlib
from inspect import signature
import json
import numpy as np
from .hdc_hamiltonian import (
    solve_sam_hdcpt2,
    sam_polynomial_operator,
    hdc_transition_spectrum,
    sam_thermal_populations,
)
from .vibronic_harmonic import harmonic_vibronic_from_esalc
from .vibronic_hdcpt2 import hdc_vibronic_ti
from .vibronic_vpt2_workflow import _prepare_esalc_sam_state
from .vibronic_workflow import _complex_input, vibronic_json_value


def run_esalc_hdc_workflow(configuration):
    request = deepcopy(configuration)
    schema = request.pop("schema", None)
    controls = dict(request.pop("controls", {}))
    if schema == "matrix.sati.esalc_hdc_request.v1":
        allowed = {
            "initial",
            "final",
            "electronic_gap_cm",
            "transition_dipole_au",
            "transition_gradient_esalc_au",
            "transition_reference_esalc",
        }
        if set(request) - allowed:
            raise ValueError("unknown SATI HDC request fields")
        allowed_controls = set(signature(hdc_vibronic_ti).parameters) - {
            "model",
            "initial_field",
            "final_field",
        }
        if set(controls) - allowed_controls:
            raise ValueError("unknown SATI HDC controls")
        initial, fi = _prepare_esalc_sam_state(request["initial"])
        final, ff = _prepare_esalc_sam_state(request["final"])
        model = harmonic_vibronic_from_esalc(
            initial,
            final,
            electronic_gap_cm=request["electronic_gap_cm"],
            transition_dipole_au=_complex_input(request["transition_dipole_au"]),
            transition_gradient_esalc_au=_complex_input(
                request.get("transition_gradient_esalc_au")
            ),
            transition_reference_esalc=request.get("transition_reference_esalc"),
            pes_model="AH",
        )
        result = hdc_vibronic_ti(model, fi, ff, **controls)
    elif schema == "matrix.trinity.esalc_hdc_request.v1":
        if set(request) - {"state", "operator_terms", "spectrum", "operator_units"}:
            raise ValueError("unknown TRINITY HDC request fields")
        if set(controls) - (set(signature(solve_sam_hdcpt2).parameters) - {"field"}):
            raise ValueError("unknown TRINITY HDC controls")
        _, field = _prepare_esalc_sam_state(request["state"])
        solved = solve_sam_hdcpt2(field, **controls)
        result = dict(energies_cm=solved.energies_cm, diagnostics=solved.diagnostics)
        if "operator_terms" in request:
            terms = []
            for term in request["operator_terms"]:
                if set(term) != {"coefficient", "operators"}:
                    raise ValueError("invalid modal property term")
                terms.append((_complex_input(term["coefficient"]), term["operators"]))
            operator = sam_polynomial_operator(solved, terms)
            options = dict(request.get("spectrum", {}))
            if set(options) - {
                "initial_levels",
                "final_levels",
                "initial_populations",
                "temperature_K",
            }:
                raise ValueError("unknown vibrational spectrum controls")
            temperature = options.pop("temperature_K", None)
            if temperature is not None:
                if "initial_populations" in options:
                    raise ValueError("choose temperature_K or explicit populations, not both")
                ids = options.get("initial_levels", [0])
                index = np.asarray(ids)
                if (
                    index.ndim != 1
                    or index.dtype.kind not in "iu"
                    or np.any(index < 0)
                    or np.any(index >= len(solved.energies_cm))
                ):
                    raise ValueError("invalid initial_levels for thermal population")
                options["initial_populations"] = sam_thermal_populations(solved, temperature)[index]
            result["spectrum"] = hdc_transition_spectrum(solved, solved, operator, **options)
            result["spectrum"]["diagnostics"].update(
                temperature_K=temperature,
                population_model="finite_completed_hamiltonian_boltzmann"
                if temperature is not None
                else "explicit_absolute",
            )
            result["operator_units"] = request.get("operator_units", "supplied coefficient units")
        elif "spectrum" in request:
            raise ValueError("spectrum requires operator_terms")
    else:
        raise ValueError("unsupported native HDC completion request schema")
    identity = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return vibronic_json_value(
        dict(
            schema=schema.replace("request", "result"),
            configuration_sha256=identity,
            status="finite_hdc_completion_evaluated",
            result=result,
        )
    )


__all__ = ["run_esalc_hdc_workflow"]
