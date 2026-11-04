"""Resident GFN2-XTB + primitive Delta(ZAFF) energy surface for LINK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from matrix_gfn2 import GFN2Calculator


@dataclass(frozen=True)
class Gfn2ZaffCompositeEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray | None = None


class Gfn2ZaffCompositeSession:
    """Compose both resident terms at each geometry in the same Cartesian frame."""

    def __init__(
        self,
        atoms: Sequence[str],
        *,
        force_field: Path | str,
        charge: int = 0,
        multiplicity: int = 1,
        required_fit_protocol_id: str | None = None,
    ) -> None:
        from matrix_zaff.artifact import load_zaff_artifact
        from matrix_zaff.primitive_residual import PrimitiveResidualRuntime

        self.atoms = tuple(str(atom) for atom in atoms)
        self.force_field = load_zaff_artifact(force_field)
        if self.force_field.atoms != self.atoms:
            raise ValueError("Delta(ZAFF) atom order differs from the GFN2 molecule")
        metadata = self.force_field.metadata
        if required_fit_protocol_id is not None and metadata.get("protocol_id") != required_fit_protocol_id:
            raise ValueError("Delta(ZAFF) field does not match the required fit protocol")
        if metadata.get("protocol_id") not in {"zaff-gfn2-residual-fit.v2", "zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4", "zaff-gfn2-morse-cubic-refit.v1"}:
            raise ValueError("GFN2+Delta(ZAFF) requires the approved primitive residual fit")
        from matrix_core import load_keymaker_protocol_library

        approved = load_keymaker_protocol_library().resolve(metadata["protocol_id"])
        if metadata.get("protocol_sha256") != approved.sha256:
            raise ValueError("Delta(ZAFF) fit protocol hash differs from the approved version")
        if (metadata.get("gfn2_charge"), metadata.get("gfn2_multiplicity")) != (
            int(charge), int(multiplicity)
        ):
            raise ValueError("GFN2 molecular charge or multiplicity differs from the residual fit")
        expected_schema = (
            "matrix.zaff.primitive_residual.v2"
            if metadata["protocol_id"] in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4", "zaff-gfn2-morse-cubic-refit.v1"}
            else "matrix.zaff.primitive_residual.v1"
        )
        if metadata.get("primitive_residual_model", {}).get("schema") != expected_schema:
            raise ValueError("GFN2+Delta(ZAFF) requires a fitted SMITH primitive residual")
        if required_fit_protocol_id in {"zaff-gfn2-residual-fit.v3", "zaff-gfn2-residual-fit.v4"}:
            residual = metadata["primitive_residual_model"]
            expected_gfn2_alpha_source = (
                "TWO_XH_PRIMITIVE_TANGENT_ENERGIES_AND_OTHER_STRETCH_GRADIENT_STENCILS"
                if required_fit_protocol_id.endswith(".v4")
                else "PROJECTED_GFN2_PRIMITIVE_THIRD_DERIVATIVE"
            )
            if (residual.get("stretch_form") != "QM_ZAFF_MORSE_MINUS_GFN2_DERIVED_MORSE"
                    or residual.get("morse_alpha_source") != "ZAFF_MM3_SYNTHON_INTERPOLATION_WITH_NATIVE_GFN2_MAYER_AND_ZERO_ZAFF_CHARGES"
                    or residual.get("gfn2_morse_alpha_source") != expected_gfn2_alpha_source):
                raise ValueError("Delta(ZAFF) requires the frozen two-Morse construction")
        charges = metadata.get("zaff_atomic_charges_e")
        if not isinstance(charges, list) or len(charges) != len(self.atoms) or any(float(value) != 0.0 for value in charges):
            raise ValueError("GFN2+Delta(ZAFF) requires zero ZAFF charges")
        if metadata.get("zaff_vdw_terms") != [] or metadata.get("nonbonded_terms") != "NONE":
            raise ValueError("GFN2+Delta(ZAFF) must not add ZAFF van der Waals terms")
        if float(self.force_field.energy_reference_hartree) != 0.0:
            raise ValueError("Delta(ZAFF) reference energy must be exactly zero")
        if self.force_field.geminal_xh_couplings_present:
            raise ValueError("primitive Delta(ZAFF) must not add separate geminal couplings")
        self.correction = PrimitiveResidualRuntime(metadata["primitive_residual_model"])
        self.calculator = GFN2Calculator(self.atoms, charge=charge, multiplicity=multiplicity)

    def evaluate(
        self, coordinates_angstrom: np.ndarray, *, hessian: bool = False
    ) -> Gfn2ZaffCompositeEvaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("composite coordinates must be finite natoms x 3")
        if hessian:
            native, native_hessian = self.calculator.evaluate_independent_hessian(
                coordinates
            )
        else:
            native = self.calculator.evaluate_independent(
                coordinates, compute_cm5=False, compute_bond_orders=False,
            )
        correction = self.correction.evaluate(coordinates, include_hessian=hessian)
        combined_hessian = None
        if hessian:
            combined_hessian = (
                np.asarray(native_hessian.hessian_hartree_per_bohr2, dtype=float)
                + np.asarray(correction.hessian_hartree_per_bohr2, dtype=float)
            )
            combined_hessian = 0.5 * (combined_hessian + combined_hessian.T)
        return Gfn2ZaffCompositeEvaluation(
            energy_hartree=float(native.energy_hartree) + float(correction.energy_hartree),
            gradient_hartree_per_bohr=(
                np.asarray(native.gradient_hartree_per_bohr, dtype=float).reshape(-1)
                + np.asarray(correction.gradient_hartree_per_bohr, dtype=float).reshape(-1)
            ),
            hessian_hartree_per_bohr2=combined_hessian,
        )

    def bond_order_matrix(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        """Return native GFN2 bond orders; the ZAFF residual has none."""

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("composite coordinates must be finite natoms x 3")
        native = self.calculator.evaluate_independent(
            coordinates, compute_cm5=False, compute_bond_orders=True,
        )
        return np.asarray(native.bond_orders, dtype=float).copy()


__all__ = ["Gfn2ZaffCompositeEvaluation", "Gfn2ZaffCompositeSession"]
