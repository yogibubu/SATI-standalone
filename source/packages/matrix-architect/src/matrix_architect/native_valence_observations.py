"""SPF columns for the existing native multi-anchor analytic E/G/H fitter."""

from types import SimpleNamespace
import numpy as np
from matrix_chem.primitive_coordinates import eval_primitives, primitive_b_matrix
from matrix_zaff.basic_primitive_derivative import analytic_basic_primitive_second_derivative
from matrix_zaff.radial import SPFStretchPotential
from .asymptotic_pic_hessian import NativeCorrectionHessianDesign


class NativeValenceCorrectionDesign:
    """Use resident native columns, replacing only pure radial SPF columns."""

    def __init__(self, parent, factors):
        self.parent = parent
        self.factors = tuple(tuple(t) for t in factors)
        self.kernel = NativeCorrectionHessianDesign(
            parent.primitives,
            parent.field,
            [
                tuple(
                    (f.primitive_index, f.form, f.power, f.harmonic, f.window_start, f.window_end)
                    for f in term
                )
                for term in factors
            ],
        )

    def value(self, geometry):
        xyz = np.asarray(geometry, float)
        energy, gradient, hessian = self.kernel.value(xyz)
        if self.parent.radial_representation != "spf_inverse_distance":
            return energy, gradient, hessian
        q = eval_primitives(self.parent.primitives, xyz)
        b = primitive_b_matrix(self.parent.primitives, xyz)
        curvature = {}
        for column, term in enumerate(self.factors):
            if len(term) != 1 or self.parent.primitives[term[0].primitive_index].kind != "bond":
                continue
            factor = term[0]
            index = factor.primitive_index
            r0 = self.parent.reference[index]
            if factor.form not in ("inverse_distance", "native") or factor.power < 2:
                raise ValueError("Pure radial fitting columns require SPF powers >= 2")
            coefficients = [0.0] * (factor.power - 1)
            coefficients[-1] = r0**factor.power
            jet = SPFStretchPotential(r0, tuple(coefficients)).derivatives(q[index])
            if index not in curvature:
                matrix = np.zeros((xyz.size, xyz.size))
                primitive = self.parent.primitives[index]
                item = SimpleNamespace(function="R", atoms=tuple(a + 1 for a in primitive.atoms))
                for i, j, value in analytic_basic_primitive_second_derivative(item, xyz).entries:
                    matrix[i, j] = value
                curvature[index] = matrix
            energy[column] = jet.energy
            gradient[:, column] = jet.first * b[index]
            hessian[:, :, column] = (
                jet.second * np.outer(b[index], b[index]) + jet.first * curvature[index]
            )
        return energy, gradient, hessian
