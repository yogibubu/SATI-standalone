"""Autonomous loader for the universal D4 reference-system database."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
from math import exp, floor, sqrt

import numpy as np

from .dispersion import D4AtomModel, D4ReferenceState


def _key(indices: list[str]) -> tuple[int, ...]:
    return tuple(int(item) for item in indices if item != ":")


@lru_cache(maxsize=1)
def load_d4_reference_models() -> tuple[D4AtomModel, ...]:
    """Return complete D4 models for Z=1..118; GFN2 uses Z<=86."""

    root = files("matrix_gfn2").joinpath("data")
    raw = json.loads(root.joinpath("d4-reference.json").read_text(encoding="utf-8"))
    element = json.loads(root.joinpath("d4-elements.json").read_text(encoding="utf-8"))
    refcn = json.loads(root.joinpath("d4-refcn.json").read_text(encoding="utf-8"))
    arrays = {
        name: {_key(indices): values for indices, values in entries}
        for name, entries in raw["arrays"].items()
    }
    models = []
    for atomic_number in range(1, 119):
        count = int(arrays["refn"].get((atomic_number,), [0])[0])
        coordination_groups: dict[int, int] = {0: 1}
        for reference in range(1, count + 1):
            reference_key = f"{reference},{atomic_number}"
            if reference_key not in refcn:
                continue
            key = min(floor(float(refcn[reference_key]) + 0.5), 19)
            coordination_groups[key] = coordination_groups.get(key, 0) + 1
        references = []
        for reference in range(1, count + 1):
            index = (reference, atomic_number)
            if index not in arrays["refsys"]:
                if atomic_number <= 86:
                    raise RuntimeError(f"missing D4 reference {index}")
                continue
            system = int(arrays["refsys"][index][0])
            spectrum = np.asarray(arrays["alphaiw"][(reference, atomic_number)])
            hydrogen_count = arrays["hcount"][index][0]
            if system:
                secondary = abs(system)
                zeff_secondary = element["effective_nuclear_charge"][secondary - 1]
                hardness_secondary = element["chemical_hardness"][secondary - 1]
                qmod = arrays["refh"][index][0] + zeff_secondary
                if qmod < 0.0:
                    scale = exp(3.0)
                else:
                    scale = exp(
                        3.0
                        * (
                            1.0
                            - exp(
                                2.0
                                * hardness_secondary
                                * (1.0 - zeff_secondary / qmod)
                            )
                        )
                    )
                secondary_spectrum = (
                    arrays["sscale"][(secondary,)][0]
                    * np.asarray(arrays["secaiw"][(secondary,)])
                    * scale
                )
                spectrum = spectrum - hydrogen_count * secondary_spectrum
            spectrum = np.maximum(arrays["ascale"][index][0] * spectrum, 0.0)
            references.append(
                D4ReferenceState(
                    coordination=arrays["refcovcn"][index][0],
                    charge=arrays["refq"][index][0],
                    polarizabilities=tuple(map(float, spectrum)),
                    gaussian_count=(
                        coordination_groups[
                            min(
                                floor(float(refcn[f"{reference},{atomic_number}"]) + 0.5),
                                19,
                            )
                        ]
                        * (
                            coordination_groups[
                                min(
                                    floor(
                                        float(refcn[f"{reference},{atomic_number}"]) + 0.5
                                    ),
                                    19,
                                )
                            ]
                            + 1
                        )
                        // 2
                    ),
                )
            )
        zeff = float(element["effective_nuclear_charge"][atomic_number - 1])
        hardness = float(element["chemical_hardness"][atomic_number - 1])
        raw_r4r2 = float(element["r4_over_r2"][atomic_number - 1])
        models.append(
            D4AtomModel(
                effective_nuclear_charge=zeff,
                hardness=hardness,
                r2r4=sqrt(0.5 * raw_r4r2 * sqrt(atomic_number)),
                references=tuple(references),
            )
        )
    return tuple(models)


__all__ = ["load_d4_reference_models"]
