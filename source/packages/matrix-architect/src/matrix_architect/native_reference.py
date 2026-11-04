"""Compile chemically typed local derivatives into the native ZAFF field."""

from matrix_chem.native_reference import native_reference_from_derivatives
from matrix_chem.primitive_force_field import (
    PrimitiveFunctionalParameter,
    TorsionFourierParameter,
    TorsionFourierTerm,
)


def compile_native_reference_parameter(
    type_id,
    primitive_indices,
    kind,
    reference,
    derivatives,
    *,
    dissociation_energy=None,
    library_alpha=None,
    periodicity=None,
):
    """Produce existing ZAFF parameter records from orders zero through four.

    QM derivatives refer to the residual bonded field after subtracting full
    nonbonded interactions. Library values and chemical periodicity are explicit
    inputs; no molecular name or inferred barrier enters this compilation.
    """
    potential = native_reference_from_derivatives(
        kind,
        reference,
        derivatives,
        dissociation_energy=dissociation_energy,
        library_alpha=library_alpha,
        periodicity=periodicity,
    )
    source = "QM_LOCAL_DERIVATIVES_0_TO_4_WITH_ZAFF_ASYMPTOTES"
    if kind == "torsion":
        import math

        a, b, c, d, e = potential.coefficients
        terms = []
        for n, cosine, sine in ((periodicity, -c - 2 * e, b + d), (2 * periodicity, e / 2, -d / 2)):
            # Existing field stores Fourier coefficients in the absolute angle.
            C = cosine * math.cos(n * reference) - sine * math.sin(n * reference)
            S = cosine * math.sin(n * reference) + sine * math.cos(n * reference)
            terms.append(TorsionFourierTerm(n, math.atan2(S, C), math.hypot(C, S), C, S, source))
        return TorsionFourierParameter(
            str(type_id),
            tuple(primitive_indices),
            reference,
            float(derivatives[2]),
            float(derivatives[1]),
            a + c + 1.5 * e,
            tuple(terms),
            source,
            1.0,
            1.0,
            True,
            0.0,
            0.0,  # Local derivatives alone do not replace a periodic scan.
        )
    parameters = [(f"C{i}", float(c)) for i, c in enumerate(potential.coefficients)]
    if kind == "stretch":
        form = "inverse_distance_quartic_switched"
        parameters.extend(
            (
                ("R0", reference),
                ("E_INFINITY", potential.asymptotic_energy),
                ("ALPHA", potential.alpha),
            )
        )
    else:
        form = "cosine_quartic"
        parameters.append(("THETA0", reference))
    return PrimitiveFunctionalParameter(
        str(type_id),
        tuple(primitive_indices),
        "bond" if kind == "stretch" else "angle",
        form,
        reference,
        float(derivatives[1]),
        float(derivatives[2]),
        tuple(parameters),
        source,
        None,
    )
