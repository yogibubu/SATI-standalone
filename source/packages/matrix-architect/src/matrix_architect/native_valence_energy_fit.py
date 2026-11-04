"""Energy fitting of existing native valence functions with a frozen reference G/H.

This is a molecular valence PES fit, not a transferable nonbonded force field.
No charges or fragment thresholds are inferred. All radial/angular products
use the resident asymptotic PIC owner and its defining-bond support switches.
A finite dissociation limit does not certify the compressed-bond wall: inverse
distance correction powers require an independent short-range shape audit.
"""

from dataclasses import asdict, replace
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from matrix_chem.primitive_force_field import (
    PrimitiveMMForceField,
    PrimitiveFunctionalParameter,
    PrimitiveMMCoupling,
)
from .asymptotic_pic import (
    AsymptoticPicFactor,
    AsymptoticPicCorrection,
    asymptotic_pic_bonded_energy_gradient,
    asymptotic_pic_correction_design,
)
from matrix_chem.primitive_coordinates import eval_primitives, primitive_b_matrix
from matrix_zaff.radial import SPFStretchPotential
from .native_refinement import native_refinement_catalog
from .native_nonlocal_support import equilibrated_minimum_norm_fit, extend_native_nonlocal_support


class NativeValenceEnergySurface:
    """Resident analytic native PES with reference G/H in physical internals."""

    def __init__(
        self,
        primitives,
        reference,
        gradient,
        hessian,
        radial_exponent,
        corrections=(),
        metadata=None,
        *,
        radial_representation="switched_native_products",
    ):
        self.primitives = tuple(primitives)
        self.reference = np.asarray(reference, float)
        self.gradient = np.asarray(gradient, float)
        self.hessian = np.asarray(hessian, float)
        self.radial_exponent = float(radial_exponent)
        self.corrections = tuple(corrections)
        self.metadata = dict(metadata or {})
        self.radial_representation = radial_representation
        if radial_representation not in {"switched_native_products", "spf_inverse_distance"}:
            raise ValueError("Unsupported radial representation")
        if radial_representation == "spf_inverse_distance" and self.radial_exponent != 1.0:
            raise ValueError("ZAFF SPF requires the fixed inverse-distance coordinate P=1")
        n = len(self.reference)
        if (
            len(self.primitives) != n
            or self.gradient.shape != (n,)
            or self.hessian.shape != (n, n)
            or not all(
                np.all(np.isfinite(x)) for x in [self.reference, self.gradient, self.hessian]
            )
            or not np.allclose(self.hessian, self.hessian.T, atol=1e-12)
            or not np.isfinite(self.radial_exponent)
            or self.radial_exponent <= 0
        ):
            raise ValueError("Finite physical reference G/H and positive radial exponent required")
        labels = tuple(f"primitive_{i}" for i in range(n))
        functions = []
        linear = []
        for i, (primitive, q0) in enumerate(zip(self.primitives, self.reference)):
            if primitive.kind == "bond":
                if q0 <= 0:
                    raise ValueError("Positive reference bond required")
                curvature = self.hessian[i, i] + self.gradient[i] * (self.radial_exponent + 1) / q0
                params = dict(
                    R0=q0,
                    P=self.radial_exponent,
                    D=curvature * q0 * q0 / (2 * self.radial_exponent**2),
                )
                form = "inverse_power_stretch"
            elif primitive.kind == "angle":
                if not 0 < q0 < np.pi:
                    raise ValueError("Nonsingular reference angle required")
                curvature = self.hessian[i, i] - self.gradient[i] / np.tan(q0)
                params = dict(THETA0=q0, A=curvature / (2 * np.sin(q0) ** 2))
                form = "cosine"
            else:
                raise ValueError("This energy-fit owner supports bond/valence-angle primitives")
            if curvature <= 0:
                raise ValueError("Native diagonal reference curvature must be positive")
            functions.append(
                PrimitiveFunctionalParameter(
                    labels[i],
                    (i,),
                    primitive.kind,
                    form,
                    float(q0),
                    0.0,
                    float(curvature),
                    tuple(params.items()),
                    "finite energy fit with reference quadratic constraint",
                    None,
                )
            )
            if self.gradient[i] != 0:
                linear.append(
                    AsymptoticPicCorrection(
                        float(self.gradient[i]), (AsymptoticPicFactor(i, "native"),)
                    )
                )
        couplings = tuple(
            PrimitiveMMCoupling(
                (i, j),
                (labels[i], labels[j]),
                float(self.hessian[i, j]),
                0.0,
                "REFERENCE_QUADRATIC",
                "physical internal G/H anchor",
            )
            for i, j in combinations(range(n), 2)
            if self.hessian[i, j] != 0
        )
        self.field = PrimitiveMMForceField(
            self.reference,
            self.hessian,
            np.diag(self.hessian),
            np.diag(self.hessian),
            self.reference,
            (),
            (),
            labels,
            self.hessian,
            couplings,
            (),
            (),
            tuple(functions),
            0.0,
            0,
            0,
            0,
            0,
            "REFERENCE_CONSTRAINED",
            0.0,
            0.0,
            0.0,
            0.0,
            "hartree/angstrom/radian",
            coupling_reference_values=self.reference,
        )
        self._all_corrections = tuple(linear) + self.corrections

        if self.radial_representation == "spf_inverse_distance":
            # SPF owns the complete pure radial series. The existing asymptotic
            # owner retains angular and mixed products with defining-bond support.
            self._radial_series = {}
            mixed = []
            for correction in self._all_corrections:
                if (
                    len(correction.factors) == 1
                    and self.primitives[correction.factors[0].primitive_index].kind == "bond"
                ):
                    factor = correction.factors[0]
                    if factor.form not in {"inverse_distance", "native"}:
                        raise ValueError("SPF radial factors must be powers of inverse distance")
                    series = self._radial_series.setdefault(factor.primitive_index, {})
                    series[factor.power] = series.get(factor.power, 0.0) + correction.coefficient
                else:
                    mixed.append(correction)
            functions = []
            self._spf = {}
            for index, parameter in enumerate(self.field.functional_parameters):
                if self.primitives[index].kind != "bond":
                    functions.append(parameter)
                    continue
                parameters = dict(parameter.parameters)
                r0 = self.reference[index]
                coefficients = self._radial_series.setdefault(index, {})
                coefficients[2] = coefficients.get(2, 0.0) + parameters["D"] / r0**2
                highest = max(2, max(coefficients, default=2))
                self._spf[index] = SPFStretchPotential(
                    r0, tuple(coefficients.get(n, 0.0) * r0**n for n in range(2, highest + 1))
                )
                parameters["D"] = 0.0
                functions.append(replace(parameter, parameters=tuple(parameters.items())))
            self._mixed_field = replace(self.field, functional_parameters=tuple(functions))
            self._mixed_corrections = tuple(mixed)

    def energy_gradient(self, coordinates_angstrom):
        """Energy and Cartesian gradient in Eh and Eh/angstrom."""
        if self.radial_representation == "switched_native_products":
            return asymptotic_pic_bonded_energy_gradient(
                self.field, self.primitives, coordinates_angstrom, corrections=self._all_corrections
            )
        xyz = np.asarray(coordinates_angstrom, float)
        energy, gradient = asymptotic_pic_bonded_energy_gradient(
            self._mixed_field, self.primitives, xyz, corrections=self._mixed_corrections
        )
        q = eval_primitives(self.primitives, xyz)
        radial_gradient = np.zeros(len(self.primitives))
        for index, potential in self._spf.items():
            jet = potential.derivatives(q[index])
            r0 = self.reference[index]
            linear = self._radial_series[index].get(1, 0.0)
            energy += jet.energy + linear * r0 * (1 - r0 / q[index])
            radial_gradient[index] = jet.first + linear * r0**2 / q[index] ** 2
        gradient += (primitive_b_matrix(self.primitives, xyz).T @ radial_gradient).reshape(
            gradient.shape
        )
        return float(energy), gradient

    def __call__(self, coordinates_angstrom):
        """Analytic native E/G/H in the shared NativePicPoint atomic units."""
        from types import SimpleNamespace
        from matrix_chem import BOHR_TO_ANGSTROM
        from matrix_zaff.basic_primitive_derivative import (
            analytic_basic_primitive_second_derivative,
        )
        from .native_pic_field import NativePicField, NativePicPoint

        xyz = np.asarray(coordinates_angstrom, float)
        point = self._native_runtime()(xyz)
        hessian = point.hessian_hartree_per_bohr2 / BOHR_TO_ANGSTROM**2
        if self.radial_representation == "spf_inverse_distance":
            q = eval_primitives(self.primitives, xyz)
            b = primitive_b_matrix(self.primitives, xyz)
            for index, potential in self._spf.items():
                jet = potential.derivatives(q[index])
                r0 = self.reference[index]
                linear = self._radial_series[index].get(1, 0.0)
                first = jet.first + linear * r0**2 / q[index] ** 2
                second = jet.second - 2 * linear * r0**2 / q[index] ** 3
                hessian += second * np.outer(b[index], b[index])
                item = SimpleNamespace(
                    function="R", atoms=tuple(a + 1 for a in self.primitives[index].atoms)
                )
                for i, j, value in analytic_basic_primitive_second_derivative(item, xyz).entries:
                    hessian[i, j] += first * value
        energy, gradient = self.energy_gradient(xyz)
        return NativePicPoint(
            energy,
            np.asarray(gradient).reshape(-1) * BOHR_TO_ANGSTROM,
            (hessian + hessian.T) / 2 * BOHR_TO_ANGSTROM**2,
        )

    def _native_runtime(self):
        from types import SimpleNamespace
        from .native_pic_field import NativePicField
        if not hasattr(self, "_hessian_runtime"):

            def zero_nonbonded(coordinates):
                size = np.asarray(coordinates).size
                return SimpleNamespace(
                    total_energy=0.0, total_gradient=np.zeros(size), total=np.zeros((size, size))
                )

            field = (
                self._mixed_field
                if self.radial_representation == "spf_inverse_distance"
                else self.field
            )
            corrections = (
                self._mixed_corrections
                if self.radial_representation == "spf_inverse_distance"
                else self._all_corrections
            )
            self._hessian_runtime = NativePicField(
                self.primitives, field, zero_nonbonded, corrections=corrections
            )
        return self._hessian_runtime

    def energy(self, coordinates_angstrom):
        return self.energy_gradient(coordinates_angstrom)[0]

    def energies(self, geometries_angstrom):
        """Batch native energies without constructing gradients on a DVR grid."""
        xyz = np.asarray(geometries_angstrom, float)
        energies = self._native_runtime().energies(xyz, batch_size=512)
        if self.radial_representation == "spf_inverse_distance" and len(xyz):
            q = np.array([eval_primitives(self.primitives, x) for x in xyz])
            for index, potential in self._spf.items():
                y = 1 - potential.r0 / q[:, index]
                linear = self._radial_series[index].get(1, 0.0) * potential.r0
                energies += np.polynomial.polynomial.polyval(y, (0., linear, *potential.coefficients))
        return energies

    def save(self, path):
        data = dict(
            schema=(
                "matrix.architect.native_valence_energy_surface.spf.v1"
                if self.radial_representation == "spf_inverse_distance"
                else "matrix.architect.native_valence_energy_surface.v1"
            ),
            primitives=[
                dict(kind=p.kind, atoms=list(p.atoms), function=p.function) for p in self.primitives
            ],
            reference=self.reference.tolist(),
            gradient=self.gradient.tolist(),
            hessian=self.hessian.tolist(),
            radial_exponent=self.radial_exponent,
            radial_representation=self.radial_representation,
            corrections=[asdict(c) for c in self.corrections],
            metadata=self.metadata,
            model_scope="total molecular energy fit in native valence functions; no explicit charge/nonbonded decomposition or independently qualified fragment thresholds",
        )
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        from matrix_chem.primitive_coordinates import Primitive

        d = json.loads(Path(path).read_text())
        if d["schema"] not in {
            "matrix.architect.native_valence_energy_surface.v1",
            "matrix.architect.native_valence_energy_surface.spf.v1",
        }:
            raise ValueError("Unsupported native energy artifact")
        if (
            d["schema"].endswith(".spf.v1")
            and d.get("radial_representation") != "spf_inverse_distance"
        ):
            raise ValueError("SPF artifact representation is inconsistent with its schema")
        if d.get("radial_form", "inverse_power_stretch") != "inverse_power_stretch":
            raise ValueError(
                "Unsupported experimental radial form; not a ZAFF inverse-distance artifact"
            )
        primitives = [
            Primitive(p["kind"], tuple(p["atoms"]), smith_function=p["function"])
            for p in d["primitives"]
        ]
        corrections = [
            AsymptoticPicCorrection(
                c["coefficient"], tuple(AsymptoticPicFactor(**f) for f in c["factors"])
            )
            for c in d["corrections"]
        ]
        return cls(
            primitives,
            d["reference"],
            d["gradient"],
            d["hessian"],
            d["radial_exponent"],
            corrections,
            d["metadata"],
            radial_representation=d.get("radial_representation", "switched_native_products"),
        )


def fit_native_valence_energies(
    primitives,
    reference,
    gradient,
    hessian,
    geometries,
    relative_energies_hartree,
    *,
    permutations=(),
    maximum_active_primitives=2,
    exponent_bounds=(0.25, 4.0),
    training_tolerance_cm=5.0,
    axis_degrees=(),
    axis_degrees_by_kind=None,
    bond_correction_form="inverse_distance",
    radial_exponent=None,
    radial_representation="switched_native_products",
):
    """Fit finite energies only; hold-outs must be passed to a separate validator.

    Native cubic/quartic-coordinate products preserve the reference G/H. These
    are functional parameters, not finite-difference cubic/quartic targets.
    A scalar radial-shape fit wraps the existing equilibrated linear fitter.
    Supplied primitive permutations tie equivalent coefficients exactly.
    ``spf_inverse_distance`` fixes P=1 and uses the resident ZAFF SPF owner
    for the complete pure radial series, without a separate radial envelope.
    Angular and mixed products retain the existing bond support switches.
    """
    if radial_representation == "spf_inverse_distance":
        if radial_exponent not in (None, 1.0):
            raise ValueError("ZAFF SPF requires P=1, not a fitted real inverse power")
        if bond_correction_form != "inverse_distance":
            raise ValueError("ZAFF SPF requires inverse-distance fit products")
        radial_exponent = 1.0
    xyz = np.asarray(geometries, float)
    target = np.asarray(relative_energies_hartree, float)
    n = len(reference)
    perms = [tuple(range(n)), *[tuple(p) for p in permutations]]
    if any(sorted(p) != list(range(n)) for p in perms):
        raise ValueError("True primitive permutations required")
    if xyz.ndim != 3 or target.shape != (len(xyz),) or not np.all(np.isfinite(target)):
        raise ValueError("Finite acquired energy samples required")
    catalog = native_refinement_catalog(
        primitives, maximum_active_primitives=maximum_active_primitives
    )
    if axis_degrees and axis_degrees_by_kind:
        raise ValueError("Select a common or primitive-family axis extension, not both")
    family_degrees = dict(axis_degrees_by_kind or {})
    if any(kind not in {p.kind for p in primitives} for kind in family_degrees):
        raise ValueError("Axis extension family absent from primitive catalog")
    selected_degrees = {
        i: tuple(family_degrees.get(p.kind, axis_degrees)) for i, p in enumerate(primitives)
    }
    for index, degrees in selected_degrees.items():
        if not degrees:
            continue
        extended, _ = extend_native_nonlocal_support(
            primitives, range(n), (), degrees=degrees, excited_indices=(index,)
        )
        # Use only declared univariate powers, never the response decorations
        # supplied by the general nonlocal catalog.
        catalog += tuple(term for term in extended if len(term) == 1 and term[0].power in degrees)
        if primitives[index].kind == "bond" and max(degrees) >= 6 and radial_exponent is None:
            raise ValueError("Sixth-degree bond extensions require a fixed radial exponent")
    if bond_correction_form not in {"inverse_distance", "native"}:
        raise ValueError("Use an existing resident bond correction coordinate")
    if bond_correction_form == "native":
        catalog = tuple(
            tuple(
                replace(f, form="native") if primitives[f.primitive_index].kind == "bond" else f
                for f in term
            )
            for term in catalog
        )
    catalog = tuple(dict.fromkeys(catalog))

    def termkey(term, perm):
        return tuple(sorted((perm[f.primitive_index], f.form, f.power, f.harmonic) for f in term))

    groups = {}
    for i, term in enumerate(catalog):
        groups.setdefault(min(termkey(term, p) for p in perms), []).append(i)
    groups = list(groups.values())

    def evaluate(exponent):
        parent = NativeValenceEnergySurface(
            primitives,
            reference,
            gradient,
            hessian,
            exponent,
            radial_representation=radial_representation,
        )
        baseline = parent.energies(xyz)
        raw = np.array(
            [asymptotic_pic_correction_design(parent.field, primitives, x, catalog)[0] for x in xyz]
        )
        if radial_representation == "spf_inverse_distance":
            coordinates = np.array([eval_primitives(primitives, x) for x in xyz])
            for column, term in enumerate(catalog):
                if len(term) == 1 and primitives[term[0].primitive_index].kind == "bond":
                    index = term[0].primitive_index
                    r0 = reference[index]
                    raw[:, column] = (r0 * (1 - r0 / coordinates[:, index])) ** term[0].power
        design = np.column_stack([raw[:, g].sum(axis=1) for g in groups])
        norms = np.linalg.norm(design, axis=0)
        observed = norms > 1e-14
        coefficients = np.zeros(len(groups))
        fitted, diagnostics = equilibrated_minimum_norm_fit(design[:, observed], target - baseline)
        coefficients[observed] = fitted
        residual = baseline + design @ coefficients - target
        return residual, parent, coefficients, diagnostics

    lo, hi = map(float, exponent_bounds)
    if not 0 < lo < hi:
        raise ValueError("Explicit positive radial exponent bounds required")

    def objective(exponent):
        residual = evaluate(exponent)[0]
        return float(residual @ residual)

    if radial_exponent is None:
        fit = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
        exponent = float(fit.x)
    else:
        exponent = float(radial_exponent)
        if not lo <= exponent <= hi:
            raise ValueError("Fixed exponent outside declared bounds")
    residual, parent, coefficients, diagnostics = evaluate(exponent)
    corrections = tuple(
        AsymptoticPicCorrection(float(c), catalog[i])
        for c, group in zip(coefficients, groups)
        for i in group
        if c != 0.0
    )
    surface = NativeValenceEnergySurface(
        primitives,
        reference,
        gradient,
        hessian,
        exponent,
        corrections,
        radial_representation=radial_representation,
    )
    report = dict(
        radial_representation=radial_representation,
        bond_correction_form=bond_correction_form,
        radial_exponent=exponent,
        radial_exponent_fitted=radial_exponent is None,
        axis_extension_degrees=list(axis_degrees),
        axis_extension_degrees_by_kind={k: list(v) for k, v in family_degrees.items()},
        training_count=len(target),
        catalog_terms=len(catalog),
        symmetry_tied_coefficients=len(groups),
        training_max_error_cm=float(np.max(abs(residual)) * 219474.6313705),
        training_rms_error_cm=float(
            np.linalg.norm(residual) / np.sqrt(len(residual)) * 219474.6313705
        ),
        linear_fit=diagnostics,
        linear_nullity=len(groups) - diagnostics["resolved_rank"],
        linear_parameters_identified=bool(len(groups) == diagnostics["resolved_rank"]),
        higher_QM_derivative_targets_used=False,
        held_out_used_in_fit=False,
        explicit_nonbonded_decomposition=False,
        training_tolerance_cm=training_tolerance_cm,
        training_passed=bool(np.max(abs(residual)) * 219474.6313705 <= training_tolerance_cm),
        physical_field_qualified=False,
    )
    surface.metadata.update(energy_fit=report)
    return surface, report


def regularize_spf_radial_monotonicity(
    surface,
    protected_geometries,
    *,
    orders=(8, 10, 12, 14, 16),
    maximum_energy_change_cm=5.0,
    central_radius=1e-3,
):
    """Stabilize pure 1/R branches with higher even SPF powers only.

    For a radial polynomial P(u), adding a*u**n gives a linear lower bound
    a >= -P'(u)/(n*u**(n-1)) outside the central well. Its extrema solve
    u*P''-(n-1)*P'=0. Candidate powers are fixed before inspecting the
    protected training energies. No excluded energy or electronic callback
    enters this shape constraint. Floating polynomial roots are a numerical
    certificate; the full multidimensional PES still needs a minima audit.
    """
    from numpy.polynomial import Polynomial

    if surface.radial_representation != "spf_inverse_distance":
        raise ValueError("Radial monotonicity regularization requires the ZAFF SPF representation")
    if (
        not orders
        or any(type(n) is not int or n <= 4 or n % 2 for n in orders)
        or maximum_energy_change_cm <= 0
        or central_radius <= 0
    ):
        raise ValueError("Positive budgets and declared even powers above fourth order required")
    geometries = np.asarray(protected_geometries, float)
    if not len(geometries):
        raise ValueError("Protected training geometries required")
    original = surface.energies(geometries)
    attempts = []
    for order in orders:
        additions = []
        rows = []
        for index, coefficients in surface._radial_series.items():
            polynomial = Polynomial(
                [coefficients.get(k, 0.0) for k in range(max(coefficients) + 1)]
            )
            u = Polynomial([0.0, 1.0])
            stationary = u * polynomial.deriv(2) - (order - 1) * polynomial.deriv()
            candidates = [surface.reference[index]]
            candidates += [
                float(z.real)
                for z in stationary.roots()
                if abs(z.imag) < 1e-8
                and abs(z.real) > central_radius
                and z.real < surface.reference[index]
            ]
            coefficient = max(
                [0.0] + [-polynomial.deriv()(v) / (order * v ** (order - 1)) for v in candidates]
            )
            coefficient *= 1.0001
            if coefficient:
                additions.append(
                    AsymptoticPicCorrection(
                        float(coefficient), (AsymptoticPicFactor(index, "inverse_distance", order),)
                    )
                )
            corrected = polynomial + Polynomial([0.0] * order + [coefficient])
            points = [
                float(z.real)
                for z in corrected.deriv().roots()
                if abs(z.imag) < 1e-7 and z.real < surface.reference[index]
            ]
            minima = [v for v in points if corrected.deriv(2)(v) > 1e-8]
            maxima = [v for v in points if corrected.deriv(2)(v) < -1e-8]
            rows.append(
                dict(
                    primitive_index=index,
                    added_coefficient=float(coefficient),
                    stationary_inverse_coordinates=points,
                    minima=len(minima),
                    maxima=len(maxima),
                    positive_even_leading_term=bool(
                        corrected.degree() % 2 == 0 and corrected.coef[-1] > 0
                    ),
                )
            )
        result = NativeValenceEnergySurface(
            surface.primitives,
            surface.reference,
            surface.gradient,
            surface.hessian,
            1.0,
            surface.corrections + tuple(additions),
            metadata=dict(surface.metadata),
            radial_representation="spf_inverse_distance",
        )
        change = float(np.max(abs(result.energies(geometries) - original)) * 219474.6313705)
        shape = all(
            row["minima"] == 1 and row["maxima"] == 0 and row["positive_even_leading_term"]
            for row in rows
        )
        report = dict(
            order=order,
            radial_checks=rows,
            maximum_protected_energy_change_cm=change,
            shape_passed=shape,
            passed=bool(shape and change <= maximum_energy_change_cm),
        )
        attempts.append(report)
        if report["passed"]:
            summary = dict(
                attempts=attempts,
                candidate_orders=list(orders),
                central_inverse_coordinate_radius=central_radius,
                coefficient_margin_factor=1.0001,
                selected_order=order,
                maximum_allowed_change_cm=maximum_energy_change_cm,
                reference_derivatives_preserved_through=4,
                excluded_energies_used=False,
                new_electronic_evaluations=0,
                global_multidimensional_shape_qualified=False,
                physical_field_qualified=False,
            )
            result.metadata["radial_regularization"] = summary
            return result, summary
    raise ValueError("Declared SPF stabilization powers fail the shape or protected-energy budget")
