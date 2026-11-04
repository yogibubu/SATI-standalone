"""Geometry-dependent parameter state for native NN-xTB model profiles.

This module prepares parameters only.  It must not be mistaken for an NN energy
calculator until every consumer of these parameters is wired and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .hamiltonian import AtomParameters, OrbitalParameters
from .dispersion import D4Parameters
from .multipole import AnisotropicParameters
from .parameters import GFN2ElementParameters, GFN2ParameterSet


_XTB_LEGACY_RADIUS_SCALE = 0.529177210903 / 0.52917726


@dataclass(frozen=True)
class NNXTBParameterState:
    elements: tuple[GFN2ElementParameters, ...]
    atoms: tuple[AtomParameters, ...]
    orbitals: tuple[dict[str, OrbitalParameters], ...]
    third_scales: tuple[dict[str, float], ...]
    globals: dict[str, float]
    floor_events: tuple[str, ...]
    profile: str = 'provisional'
    active_heads: frozenset[str] = frozenset()

    @property
    def basis_elements(self):
        """Official libcint inference keeps STO-nG integrals unmodified."""
        if self.profile != 'codeocean-v1':
            return self.elements
        from .parameters import load_gfn2_parameters
        base = load_gfn2_parameters().elements
        return tuple(replace(element, shells=tuple(
            replace(shell, slater_exponent=reference.slater_exponent)
            for shell, reference in zip(element.shells, base[element.atomic_number].shells, strict=True)
        )) for element in self.elements)

    def d4_parameters(self, base: D4Parameters) -> D4Parameters:
        """Return the D4 damping/scaling coefficients from global heads."""
        return replace(
            base, s6=self.globals["s6"], s8=self.globals["s8"], s9=self.globals["s9"],
            s10=self.globals["s10"],
            neutral_atm=self.profile == "codeocean-v1",
            a1=self.globals["a1"], a2=self.globals["a2"],
        )

    def anisotropic_parameters(self, coordination: np.ndarray) -> AnisotropicParameters:
        """Build per-atom AES kernels and shifted CN-dependent damping radii."""
        cn = np.asarray(coordination, dtype=float)
        if cn.shape != (len(self.elements),) or not np.all(np.isfinite(cn)):
            raise ValueError("AES coordination must be finite with one value per atom")
        base = np.asarray([element.multipole_radius_bohr for element in self.elements])
        vcn = np.asarray([
            element.multipole_valence_coordination for element in self.elements
        ])
        exponent = -self.globals["aesexp"] * (
            cn - vcn - self.globals["aesshift"]
        )
        # Stable logistic, including geometries well outside the training set.
        logistic = np.exp(-np.logaddexp(0.0, exponent))
        radii = base + (self.globals["aesrmax"] - base) * logistic
        return AnisotropicParameters(
            radii,
            np.asarray([element.dipole_xc for element in self.elements]),
            np.asarray([element.quadrupole_xc for element in self.elements]),
            a3=self.globals["aesdmp3"], a5=self.globals["aesdmp5"],
        )


def _global_baseline(parameters: GFN2ParameterSet) -> dict[str, float]:
    raw = parameters.global_parameters
    hamiltonian = raw["hamiltonian"]["xtb"]
    shell = hamiltonian["shell"]
    dispersion = raw["dispersion"]["d4"]
    repulsion = raw["repulsion"]["effective"]
    multipole = raw["multipole"]["damped"]
    charge = raw["charge"]["effective"]
    return {
        "wexp": float(hamiltonian["wexp"]),
        "klight": float(repulsion["klight"]),
        "s6": float(dispersion["s6"]),
        "s10": float(dispersion.get("s10", 0.0)),
        "ks": float(shell["ss"]),
        "kp": float(shell["pp"]),
        "kd": float(shell["dd"]),
        "ksd": float(shell["sd"]),
        "kpd": float(shell["pd"]),
        "enscale": float(hamiltonian["enscale"]),
        "kexp": float(repulsion["kexp"]),
        "aesdmp3": float(multipole["dmp3"]),
        "aesdmp5": float(multipole["dmp5"]),
        "aesexp": float(multipole["kexp"]),
        "aesshift": float(multipole["shift"]),
        "aesrmax": float(multipole["rmax"]),
        "a1": float(dispersion["a1"]),
        "a2": float(dispersion["a2"]),
        "s8": float(dispersion["s8"]),
        "s9": float(dispersion["s9"]),
        "alphaj": float(charge["gexp"]),
    }


def build_nn_parameter_state(
    atomic_numbers: Sequence[int],
    parameters: GFN2ParameterSet,
    matrix_shifts: Mapping[str, np.ndarray],
) -> NNXTBParameterState:
    """Apply *all declared* NN heads without averaging equivalent elements.

    The shifts must already have been converted from the provisional model's
    raw xTB units by ``NNXTBOverlay.matrix_parameter_shifts``.  Missing shell
    couplings remain missing, as in the reference implementation.
    """
    from .calculator import _atomic_radii, _covalent_radii

    numbers = tuple(int(number) for number in atomic_numbers)
    natoms = len(numbers)
    if not numbers:
        raise ValueError("NN-xTB parameter state requires at least one atom")
    allowed_atom = {
        "gam", "gam3", "repa", "repb", "dpol", "qpol", "en",
        "rcov", "arad", "mprad", "mpvcn",
    }
    allowed_shell = {"lev", "exp", "h0exp", "kcn", "poly", "lpar", "third"}
    globals_base = _global_baseline(parameters)
    profile = getattr(matrix_shifts, 'profile', 'provisional')
    official = profile == 'codeocean-v1'
    for name, shift in matrix_shifts.items():
        if name.startswith("atom."):
            if name[5:] not in allowed_atom or np.shape(shift) != (natoms, 1):
                raise ValueError(f"invalid atom head {name!r} or shape")
        elif name.startswith("shell."):
            if name[6:] not in allowed_shell or np.shape(shift) != (natoms, 4):
                raise ValueError(f"invalid shell head {name!r} or shape")
        elif name.startswith("global."):
            if name[7:] not in globals_base or np.shape(shift) != (1, 1):
                raise ValueError(f"invalid global head {name!r} or shape")
        else:
            raise ValueError(f"unknown NN-xTB parameter head {name!r}")
        if not np.all(np.isfinite(shift)):
            raise ValueError(f"nonfinite NN-xTB parameter head {name!r}")

    def delta(name: str, atom: int = 0, slot: int = 0) -> float:
        value = matrix_shifts.get(name)
        return 0.0 if value is None else float(value[atom, slot])

    covalent = _covalent_radii()
    atomic = _atomic_radii()
    elements = []
    atoms = []
    orbitals = []
    third_scales = []
    floor_events = []

    def floored(value: float, floor: float, label: str) -> float:
        if official:
            # No provisional safety floor may alter the official surface.
            # Zero Slater exponents/hardness are singular in the native SCF.
            if value <= 0.0 and label.rsplit(": ", 1)[-1] in {"exp", "gam", "repa", "rcov", "arad"}:
                raise ValueError(f"Official NNxTB produced singular parameter {label}: {value}")
            return value
        if value >= floor:
            return value
        floor_events.append(label)
        return floor

    for atom, number in enumerate(numbers):
        original = parameters.elements[number]
        gam = floored(original.hubbard_hartree + delta("atom.gam", atom),
                      1e-3, f"atom {atom}: gam")
        repa = floored(original.repulsion_exponent + delta("atom.repa", atom),
                       1e-3, f"atom {atom}: repa")
        # The reference stores the D3-scaled CN radius. MATRIX's CN kernel
        # applies the 4/3 scale itself, so convert back at the boundary.
        rcov = floored((4.0 / 3.0) * float(covalent[number - 1])
                        * _XTB_LEGACY_RADIUS_SCALE
                        + delta("atom.rcov", atom),
                        0.1, f"atom {atom}: rcov")
        arad = floored(float(atomic[number - 1])
                        * _XTB_LEGACY_RADIUS_SCALE + delta("atom.arad", atom),
                        0.1, f"atom {atom}: arad")
        mprad = floored(original.multipole_radius_bohr + delta("atom.mprad", atom),
                         0.1, f"atom {atom}: mprad")
        shifted_shells = []
        orbital_values = {}
        third_values = {}
        for slot, shell in enumerate(original.shells):
            if slot >= 4:
                raise ValueError(f"element Z={number} has more than four NN shell slots")
            # Zero table entries encode absent optional interactions in the
            # bundled baseline.  A head must not create an absent coupling.
            kcn = (shell.coordination_shift_hartree + delta("shell.kcn", atom, slot)
                   if official or shell.coordination_shift_hartree != 0.0 else 0.0)
            poly = (shell.polynomial + delta("shell.poly", atom, slot)
                    if official or shell.polynomial != 0.0 else 0.0)
            lpar = (shell.hubbard_scale + delta("shell.lpar", atom, slot)
                    if official or shell.hubbard_scale != 1.0 else 1.0)
            if official and gam * lpar <= 0.0:
                raise ValueError(f"Official NNxTB produced singular SCC hardness at atom {atom}, shell {slot}")
            changed = replace(
                shell,
                level_hartree=shell.level_hartree + delta("shell.lev", atom, slot),
                slater_exponent=floored(
                    shell.slater_exponent + delta("shell.exp", atom, slot) + delta("shell.h0exp", atom, slot),
                    0.05, f"atom {atom} shell {slot}: exp",
                ),
                coordination_shift_hartree=kcn,
                polynomial=poly,
                hubbard_scale=lpar,
            )
            shifted_shells.append(changed)
            orbital_values[shell.label] = OrbitalParameters(
                changed.level_hartree, changed.coordination_shift_hartree,
                changed.polynomial,
            )
            third_values[shell.label] = 1.0 + delta("shell.third", atom, slot)
        element = replace(
            original,
            hubbard_hartree=gam,
            hubbard_derivative_hartree=(
                original.hubbard_derivative_hartree + delta("atom.gam3", atom)
            ),
            repulsion_exponent=repa,
            effective_nuclear_charge=(
                original.effective_nuclear_charge + delta("atom.repb", atom)
            ),
            electronegativity=(
                original.electronegativity + delta("atom.en", atom)
            ),
            dipole_xc=original.dipole_xc + delta("atom.dpol", atom),
            quadrupole_xc=original.quadrupole_xc + delta("atom.qpol", atom),
            multipole_radius_bohr=mprad,
            multipole_valence_coordination=(
                original.multipole_valence_coordination + delta("atom.mpvcn", atom)
            ),
            shells=tuple(shifted_shells),
        )
        elements.append(element)
        atoms.append(AtomParameters(
            # MATRIX's H0 pair prefactor stores twice the *atomic* radius;
            # its pair average then equals the reference's radius sum.
            repa, element.effective_nuclear_charge, 0.75 * rcov, 2.0 * arad,
            element.electronegativity,
        ))
        orbitals.append(orbital_values)
        third_scales.append(third_values)
    global_values = {
        key: baseline + delta(f"global.{key}")
        for key, baseline in globals_base.items()
    }
    return NNXTBParameterState(
        tuple(elements), tuple(atoms), tuple(orbitals),
        tuple(third_scales), global_values, tuple(floor_events), profile, frozenset(matrix_shifts),
    )


def analytic_repulsion_head_adjoints(
    atomic_numbers: Sequence[int], coordinates_bohr: np.ndarray,
    state: NNXTBParameterState,
) -> dict[str, np.ndarray]:
    """Exact derivatives of the pair repulsion with respect to its NN shifts.

    Light-light pairs use ``klight`` (one for the provisional profile), while
    ``atom.repa`` has the safety floor of the selected parameter profile.
    Returned adjoints use MATRIX shift units; the caller applies the NN
    head's raw-to-MATRIX scale.
    """
    numbers = np.asarray(atomic_numbers, dtype=int)
    positions = np.asarray(coordinates_bohr, dtype=float)
    if positions.shape != (len(numbers), 3):
        raise ValueError("repulsion adjoint coordinates must be natoms x 3")
    natoms = len(numbers)
    repa = np.asarray([atom.alpha for atom in state.atoms], dtype=float)
    repb = np.asarray(
        [atom.effective_nuclear_charge for atom in state.atoms], dtype=float,
    )
    da = np.zeros((natoms, 1), dtype=float)
    db = np.zeros((natoms, 1), dtype=float)
    dk = np.zeros((1, 1), dtype=float)
    dklight = np.zeros((1, 1), dtype=float)
    kexp = state.globals["kexp"]
    for left in range(natoms):
        for right in range(left + 1, natoms):
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if distance <= 0.0:
                raise ValueError("distinct atoms cannot occupy the same point")
            light_pair = numbers[left] <= 2 and numbers[right] <= 2
            exponent = state.globals["klight"] if light_pair else kexp
            power = distance**exponent
            alpha = np.sqrt(repa[left] * repa[right])
            damping = np.exp(-alpha * power) / distance
            pair = repb[left] * repb[right] * damping
            db[left, 0] += repb[right] * damping
            db[right, 0] += repb[left] * damping
            da[left, 0] -= pair * alpha * power / (2.0 * repa[left])
            da[right, 0] -= pair * alpha * power / (2.0 * repa[right])
            (dklight if light_pair else dk)[0, 0] -= pair * alpha * power * np.log(distance)
    # At the floor the mapping is non-smooth; take the one-sided clamped
    # derivative, consistent with the parameter-state builder's branch.
    if state.profile != "codeocean-v1":
        da[repa <= 1.0e-3, 0] = 0.0
    result = {"atom.repa": da, "atom.repb": db, "global.kexp": dk}
    if "global.klight" in state.active_heads:
        result["global.klight"] = dklight
    return result


def analytic_repulsion_head_adjoint_directional(
    atomic_numbers: Sequence[int], coordinates_bohr: np.ndarray,
    state: NNXTBParameterState, coordinate_direction: np.ndarray,
    parameter_direction: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Directional derivative of the repulsion head adjoints.

    A coordinate direction gives the explicit mixed block ``E_pR``; a head
    direction gives the parameter Hessian action ``E_pp dp``. Both may be
    supplied together. This is an exact pair-rule derivative, with no SCF
    contribution because the repulsion is density independent.
    """
    numbers = np.asarray(atomic_numbers, dtype=int)
    positions = np.asarray(coordinates_bohr, dtype=float)
    dr_xyz = np.asarray(coordinate_direction, dtype=float)
    natoms = len(numbers)
    if positions.shape != (natoms, 3) or dr_xyz.shape != positions.shape:
        raise ValueError("repulsion directional coordinates must be natoms x 3")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(dr_xyz)):
        raise ValueError("repulsion directional coordinates must be finite")
    shapes = {"atom.repa": (natoms, 1), "atom.repb": (natoms, 1),
              "global.kexp": (1, 1), "global.klight": (1, 1)}
    if set(parameter_direction) - shapes.keys():
        raise ValueError("unsupported repulsion parameter direction")
    direction = {}
    for name, shape in shapes.items():
        value = np.asarray(parameter_direction.get(name, np.zeros(shape)), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid repulsion direction for {name}")
        direction[name] = value
    repa = np.asarray([atom.alpha for atom in state.atoms], dtype=float)
    repb = np.asarray([atom.effective_nuclear_charge for atom in state.atoms], dtype=float)
    da = direction["atom.repa"][:, 0].copy()
    if state.profile != "codeocean-v1":
        da[repa <= 1.0e-3] = 0.0
    db = direction["atom.repb"][:, 0]
    dk = float(direction["global.kexp"][0, 0])
    out_a = np.zeros((natoms, 1))
    out_b = np.zeros((natoms, 1))
    out_k = np.zeros((1, 1))
    out_klight = np.zeros((1, 1))
    for left in range(natoms):
        for right in range(left + 1, natoms):
            difference = positions[left] - positions[right]
            radius = float(np.linalg.norm(difference))
            if radius <= 0.0:
                raise ValueError("distinct atoms cannot occupy the same point")
            dlogr = float(np.dot(difference, dr_xyz[left] - dr_xyz[right])) / radius**2
            light_pair = numbers[left] <= 2 and numbers[right] <= 2
            kexp = state.globals["klight"] if light_pair else state.globals["kexp"]
            exponent_direction = float(direction["global.klight"][0, 0]) if light_pair else dk
            dlogalpha = 0.5 * (da[left] / repa[left] + da[right] / repa[right])
            log_radius = np.log(radius)
            power = np.sqrt(repa[left] * repa[right]) * radius**kexp
            dpower = power * (dlogalpha + kexp * dlogr
                              + exponent_direction * log_radius)
            damping = np.exp(-power) / radius
            ddamping = damping * (-dpower - dlogr)
            charge_product = repb[left] * repb[right]
            dcharge_product = db[left] * repb[right] + repb[left] * db[right]
            pair = charge_product * damping
            dpair = dcharge_product * damping + charge_product * ddamping
            out_b[left, 0] += db[right] * damping + repb[right] * ddamping
            out_b[right, 0] += db[left] * damping + repb[left] * ddamping
            out_a[left, 0] -= (
                dpair * power + pair * dpower - pair * power * da[left] / repa[left]
            ) / (2.0 * repa[left])
            out_a[right, 0] -= (
                dpair * power + pair * dpower - pair * power * da[right] / repa[right]
            ) / (2.0 * repa[right])
            (out_klight if light_pair else out_k)[0, 0] -= (
                (dpair * power + pair * dpower) * log_radius
                + pair * power * dlogr
            )
    if state.profile != "codeocean-v1":
        out_a[repa <= 1.0e-3, 0] = 0.0
    return {"atom.repa": out_a, "atom.repb": out_b, "global.kexp": out_k,
            "global.klight": out_klight}


def analytic_h0_head_adjoints(
    atomic_numbers: Sequence[int], coordinates_bohr: np.ndarray,
    state: NNXTBParameterState, overlap: np.ndarray,
    density: np.ndarray, ao_atom: Sequence[int], ao_shell: Sequence[str],
    *, return_fock_derivatives: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Stationary band-energy adjoints for the basis-preserving H0 heads.

    ``density`` is the converged AO density. By stationarity, the electronic
    derivative is ``Tr(P dH0/dp)``; no orbital finite differences are needed.
    Basis-moving ``shell.exp`` and CN-moving ``atom.rcov`` are not included.
    """
    from .hamiltonian import _angular_classes, _h0_global_scalings, coordination_numbers

    numbers = tuple(int(number) for number in atomic_numbers)
    positions = np.asarray(coordinates_bohr, dtype=float)
    p = np.asarray(density, dtype=float)
    s = np.asarray(overlap, dtype=float)
    atom = np.asarray(ao_atom, dtype=int)
    labels = tuple(ao_shell)
    norbitals = len(labels)
    if positions.shape != (len(numbers), 3) or p.shape != (norbitals, norbitals) or s.shape != p.shape:
        raise ValueError("H0 adjoint state has incompatible dimensions")
    if atom.shape != (norbitals,):
        raise ValueError("H0 adjoint AO atom map has incompatible dimensions")
    slots = []
    levels = np.empty(norbitals)
    poly = np.empty(norbitals)
    slater = np.empty(norbitals)
    kcn = np.empty(norbitals)
    for mu, (a, label) in enumerate(zip(atom, labels, strict=True)):
        slot = next(i for i, sh in enumerate(state.elements[a].shells) if sh.label == label)
        shell = state.elements[a].shells[slot]
        slots.append(slot)
        levels[mu] = shell.level_hartree
        kcn[mu] = shell.coordination_shift_hartree
        poly[mu] = shell.polynomial
        slater[mu] = shell.slater_exponent
    cn = coordination_numbers(numbers, positions, state.atoms)
    level = levels - kcn * cn[atom]
    angular, enscale = _h0_global_scalings(state.globals)
    classes = _angular_classes(labels)
    en = np.asarray([value.electronegativity for value in state.atoms])
    radius = np.asarray([value.d4_covalent_radius_bohr for value in state.atoms])
    natoms = len(numbers)
    result = {
        "shell.lev": np.zeros((natoms, 4)),
        "shell.kcn": np.zeros((natoms, 4)),
        "shell.poly": np.zeros((natoms, 4)),
        "shell.h0exp": np.zeros((natoms, 4)),
        "atom.arad": np.zeros((natoms, 1)),
        "atom.en": np.zeros((natoms, 1)),
        **{f"global.{name}": np.zeros((1, 1)) for name in (
            "ks", "kp", "kd", "ksd", "kpd", "enscale", "wexp",
        )},
    }
    fock = (
        {name: np.zeros(value.shape + (norbitals, norbitals))
         for name, value in result.items()}
        if return_fock_derivatives else None
    )
    for mu in range(norbitals):
        a = atom[mu]
        slot_a = slots[mu]
        result["shell.lev"][a, slot_a] += p[mu, mu]
        result["shell.kcn"][a, slot_a] -= p[mu, mu] * cn[a]
        if fock is not None:
            fock["shell.lev"][a, slot_a, mu, mu] = 1.0
            fock["shell.kcn"][a, slot_a, mu, mu] = -cn[a]
        for nu in range(mu + 1, norbitals):
            b = atom[nu]
            if a == b:
                continue
            weight = 2.0 * p[mu, nu] * s[mu, nu]
            if weight == 0.0 and fock is None:
                continue
            slot_b = slots[nu]
            distance = float(np.linalg.norm(positions[a] - positions[b]))
            rr = 0.5 * (radius[a] + radius[b])
            root = np.sqrt(distance / rr)
            fi = 1.0 + poly[mu] * root
            fj = 1.0 + poly[nu] * root
            polynomial = fi * fj
            zeta_base = 2.0 * np.sqrt(slater[mu] * slater[nu]) / (slater[mu] + slater[nu])
            zeta = zeta_base ** state.globals.get("wexp", 0.5)
            ca, cb = classes[mu], classes[nu]
            k = angular[ca, cb]
            delta_en = en[a] - en[b]
            en_factor = 1.0 + enscale * delta_en**2
            prefactor = 0.5 * weight * zeta
            dlevel = prefactor * k * en_factor * polynomial
            result["shell.lev"][a, slot_a] += dlevel
            result["shell.lev"][b, slot_b] += dlevel
            result["shell.kcn"][a, slot_a] -= dlevel * cn[a]
            result["shell.kcn"][b, slot_b] -= dlevel * cn[b]
            hsum = level[mu] + level[nu]
            common = prefactor * hsum
            for endpoint, end_atom, end_slot in ((mu, a, slot_a), (nu, b, slot_b)):
                derivative_log = state.globals.get("wexp", .5) * (
                    .5 / slater[endpoint] - 1. / (slater[mu] + slater[nu]))
                result["shell.h0exp"][end_atom, end_slot] += common * k * en_factor * polynomial * derivative_log
            result["global.wexp"][0, 0] += common * k * en_factor * polynomial * np.log(zeta_base)
            result["shell.poly"][a, slot_a] += common * k * en_factor * root * fj
            result["shell.poly"][b, slot_b] += common * k * en_factor * root * fi
            dr = -root * (poly[mu] * fj + poly[nu] * fi) / (2.0 * rr)
            # rr = (2*arad_a + 2*arad_b)/2.
            result["atom.arad"][a, 0] += common * k * en_factor * dr
            result["atom.arad"][b, 0] += common * k * en_factor * dr
            result["atom.en"][a, 0] += common * k * polynomial * 2.0 * enscale * delta_en
            result["atom.en"][b, 0] -= common * k * polynomial * 2.0 * enscale * delta_en
            result["global.enscale"][0, 0] += common * k * polynomial * delta_en**2
            angular_weight = common * en_factor * polynomial
            pair = (min(ca, cb), max(ca, cb))
            if pair == (0, 0):
                result["global.ks"][0, 0] += angular_weight
            elif pair == (0, 1):
                result["global.ks"][0, 0] += 0.5 * angular_weight
                result["global.kp"][0, 0] += 0.5 * angular_weight
            elif pair[0] == 0:
                result["global.ksd"][0, 0] += angular_weight
            elif pair == (1, 1):
                result["global.kp"][0, 0] += angular_weight
            elif pair[0] == 1:
                result["global.kpd"][0, 0] += angular_weight
            else:
                result["global.kd"][0, 0] += angular_weight
            if fock is not None:
                pair_derivatives = {
                    "shell.lev": ((a, slot_a, 0.5 * s[mu, nu] * zeta * k * en_factor * polynomial),
                                  (b, slot_b, 0.5 * s[mu, nu] * zeta * k * en_factor * polynomial)),
                    "shell.kcn": ((a, slot_a, -0.5 * s[mu, nu] * zeta * k * en_factor * polynomial * cn[a]),
                                  (b, slot_b, -0.5 * s[mu, nu] * zeta * k * en_factor * polynomial * cn[b])),
                    "shell.poly": ((a, slot_a, 0.5 * s[mu, nu] * zeta * hsum * k * en_factor * root * fj),
                                   (b, slot_b, 0.5 * s[mu, nu] * zeta * hsum * k * en_factor * root * fi)),
                    "atom.arad": ((a, 0, 0.5 * s[mu, nu] * zeta * hsum * k * en_factor * dr),
                                  (b, 0, 0.5 * s[mu, nu] * zeta * hsum * k * en_factor * dr)),
                    "atom.en": ((a, 0, s[mu, nu] * zeta * hsum * k * polynomial * enscale * delta_en),
                                (b, 0, -s[mu, nu] * zeta * hsum * k * polynomial * enscale * delta_en)),
                    "global.enscale": ((0, 0, 0.5 * s[mu, nu] * zeta * hsum * k * polynomial * delta_en**2),),
                    "global.wexp": ((0, 0, 0.5 * s[mu, nu] * zeta * hsum * k * en_factor * polynomial * np.log(zeta_base)),),
                }
                angular_derivative = 0.5 * s[mu, nu] * zeta * hsum * en_factor * polynomial
                pair_derivatives["shell.h0exp"] = tuple(
                    (end_atom, end_slot, angular_derivative * k * state.globals.get("wexp", .5)
                     * (.5 / slater[endpoint] - 1. / (slater[mu] + slater[nu])))
                    for endpoint, end_atom, end_slot in ((mu, a, slot_a), (nu, b, slot_b))
                )
                if pair == (0, 0):
                    pair_derivatives["global.ks"] = ((0, 0, angular_derivative),)
                elif pair == (0, 1):
                    pair_derivatives["global.ks"] = ((0, 0, 0.5 * angular_derivative),)
                    pair_derivatives["global.kp"] = ((0, 0, 0.5 * angular_derivative),)
                elif pair[0] == 0:
                    pair_derivatives["global.ksd"] = ((0, 0, angular_derivative),)
                elif pair == (1, 1):
                    pair_derivatives["global.kp"] = ((0, 0, angular_derivative),)
                elif pair[0] == 1:
                    pair_derivatives["global.kpd"] = ((0, 0, angular_derivative),)
                else:
                    pair_derivatives["global.kd"] = ((0, 0, angular_derivative),)
                for name, components in pair_derivatives.items():
                    for component_a, component_b, value in components:
                        fock[name][component_a, component_b, mu, nu] += value
                        fock[name][component_a, component_b, nu, mu] += value
    for a, element in enumerate(state.elements):
        for slot, shell in enumerate(element.shells):
            if shell.coordination_shift_hartree == 0.0 and state.profile != "codeocean-v1":
                result["shell.kcn"][a, slot] = 0.0
                if fock is not None:
                    fock["shell.kcn"][a, slot] = 0.0
            if shell.polynomial == 0.0 and state.profile != "codeocean-v1":
                result["shell.poly"][a, slot] = 0.0
                if fock is not None:
                    fock["shell.poly"][a, slot] = 0.0
    for a in range(natoms):
        if f"atom {a}: arad" in state.floor_events:
            result["atom.arad"][a, 0] = 0.0
            if fock is not None:
                fock["atom.arad"][a, 0] = 0.0
    return result if fock is None else (result, fock)


def analytic_h0_head_density_response_adjoints(
    fock_derivatives: Mapping[str, np.ndarray], density_direction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Contract the analytic H0 parameter sources with a CPSCF density.

    This is the orbital-relaxation part of the H0 ``E_pp`` or ``E_pR``
    action. Explicit derivatives of the sources themselves must be added by
    the corresponding parameter/geometry rules before assembling a Hessian.
    """
    density = np.asarray(density_direction, dtype=float)
    if density.ndim != 2 or density.shape[0] != density.shape[1] or not np.all(np.isfinite(density)):
        raise ValueError("H0 density response must be a finite square matrix")
    response = {}
    for name, derivative in fock_derivatives.items():
        matrix = np.asarray(derivative, dtype=float)
        if matrix.ndim != 4 or matrix.shape[2:] != density.shape:
            raise ValueError(f"invalid H0 Fock source shape for {name}")
        response[name] = np.einsum("ab,ijab->ij", density, matrix)
    return response


def analytic_h0_head_adjoint_directional(
    atomic_numbers: Sequence[int], coordinates_bohr: np.ndarray,
    state: NNXTBParameterState, overlap: np.ndarray, overlap_direction: np.ndarray,
    density: np.ndarray, density_direction: np.ndarray,
    ao_atom: Sequence[int], ao_shell: Sequence[str],
    parameter_direction: Mapping[str, np.ndarray],
    *, coordinate_direction: np.ndarray | None = None,
    coordination_direction: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """H0 head-adjoint direction, including optional geometric motion.

    The caller supplies the overlap and density directions. CN and pair
    geometry are differentiated here through the existing GFN2 CN kernel.
    """
    from .hamiltonian import _angular_classes, _h0_global_scalings, coordination_numbers

    numbers = tuple(int(value) for value in atomic_numbers)
    positions = np.asarray(coordinates_bohr, dtype=float)
    natoms = len(numbers)
    atom = np.asarray(ao_atom, dtype=int)
    labels = tuple(ao_shell)
    nao = len(labels)
    p = np.asarray(density, dtype=float)
    dp = np.asarray(density_direction, dtype=float)
    s = np.asarray(overlap, dtype=float)
    ds = np.asarray(overlap_direction, dtype=float)
    if (positions.shape != (natoms, 3) or atom.shape != (nao,)
            or any(matrix.shape != (nao, nao) for matrix in (p, dp, s, ds))):
        raise ValueError("H0 adjoint direction dimensions are inconsistent")
    dxyz = (np.zeros_like(positions) if coordinate_direction is None else
            np.asarray(coordinate_direction, dtype=float))
    if dxyz.shape != positions.shape or not np.all(np.isfinite(dxyz)):
        raise ValueError("invalid H0 coordinate direction")
    names = ("shell.lev", "shell.kcn", "shell.poly", "shell.exp", "shell.h0exp",
             "atom.arad", "atom.en", "global.ks", "global.kp", "global.kd",
             "global.ksd", "global.kpd", "global.enscale", "global.wexp")
    shapes = {name: ((natoms, 4) if name.startswith("shell.") else
                     (natoms, 1) if name.startswith("atom.") else (1, 1))
              for name in names}
    if set(parameter_direction) - shapes.keys():
        raise ValueError("unsupported H0 adjoint parameter direction")
    values = {}
    for name, shape in shapes.items():
        value = np.asarray(parameter_direction.get(name, np.zeros(shape)), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid H0 direction for {name}")
        values[name] = value
    cn = coordination_numbers(numbers, positions, state.atoms)
    from .gradients import coordination_number_directional_derivative_sparse
    dcn = coordination_number_directional_derivative_sparse(
        positions, dxyz,
        np.asarray([item.covalent_radius_bohr for item in state.atoms]),
        cutoff_bohr=max(24.0, float(np.max(np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2,
        ))) + 1.0),
    )
    if coordination_direction is not None:
        extra_cn = np.asarray(coordination_direction, dtype=float)
        if extra_cn.shape != (natoms,) or not np.all(np.isfinite(extra_cn)):
            raise ValueError("invalid H0 coordination direction")
        dcn += extra_cn
    slots = []
    levels = np.empty(nao)
    dlevels = np.empty(nao)
    poly = np.empty(nao)
    dpoly = np.empty(nao)
    zeta = np.empty(nao)
    dzeta = np.empty(nao)
    kcn = np.empty(nao)
    for mu, (a, label) in enumerate(zip(atom, labels, strict=True)):
        slot = next(i for i, shell in enumerate(state.elements[a].shells)
                    if shell.label == label)
        shell = state.elements[a].shells[slot]
        slots.append(slot)
        levels[mu] = shell.level_hartree - shell.coordination_shift_hartree * cn[a]
        dlevels[mu] = (values["shell.lev"][a, slot]
                       - (values["shell.kcn"][a, slot]
                          if shell.coordination_shift_hartree != 0.0 or state.profile == "codeocean-v1" else 0.0) * cn[a]
                       - shell.coordination_shift_hartree * dcn[a])
        poly[mu] = shell.polynomial
        dpoly[mu] = values["shell.poly"][a, slot] if shell.polynomial != 0.0 or state.profile == "codeocean-v1" else 0.0
        zeta[mu] = shell.slater_exponent
        dzeta[mu] = (0.0 if f"atom {a} shell {slot}: exp" in state.floor_events
                     else values["shell.exp"][a, slot] + values["shell.h0exp"][a, slot])
        kcn[mu] = shell.coordination_shift_hartree
    angular, enscale = _h0_global_scalings(state.globals)
    angular_direction, denscale = _h0_global_scalings({
        name[7:]: float(values[name][0, 0])
        for name in values if name.startswith("global.")
    })
    classes = _angular_classes(labels)
    en = np.asarray([item.electronegativity for item in state.atoms])
    den = values["atom.en"][:, 0]
    radius = np.asarray([item.d4_covalent_radius_bohr for item in state.atoms])
    # AtomParameters stores twice the atomic H0 radius.
    dradius = 2.0 * values["atom.arad"][:, 0].copy()
    for a in range(natoms):
        if f"atom {a}: arad" in state.floor_events:
            dradius[a] = 0.0
    result = {
        "shell.lev": np.zeros((natoms, 4)),
        "shell.kcn": np.zeros((natoms, 4)),
        "shell.poly": np.zeros((natoms, 4)),
        "shell.h0exp": np.zeros((natoms, 4)),
        "atom.arad": np.zeros((natoms, 1)),
        "atom.en": np.zeros((natoms, 1)),
        **{f"global.{name}": np.zeros((1, 1)) for name in (
            "ks", "kp", "kd", "ksd", "kpd", "enscale", "wexp",
        )},
    }
    for mu in range(nao):
        a = atom[mu]
        slot_a = slots[mu]
        result["shell.lev"][a, slot_a] += dp[mu, mu]
        result["shell.kcn"][a, slot_a] -= dp[mu, mu] * cn[a] + p[mu, mu] * dcn[a]
        for nu in range(mu + 1, nao):
            b = atom[nu]
            if a == b:
                continue
            slot_b = slots[nu]
            weight = 2.0 * p[mu, nu] * s[mu, nu]
            dweight = 2.0 * (dp[mu, nu] * s[mu, nu] + p[mu, nu] * ds[mu, nu])
            distance = float(np.linalg.norm(positions[a] - positions[b]))
            ddistance = float(np.dot(
                positions[a] - positions[b], dxyz[a] - dxyz[b],
            ) / distance)
            rr = 0.5 * (radius[a] + radius[b])
            drr = 0.5 * (dradius[a] + dradius[b])
            root = np.sqrt(distance / rr)
            droot = 0.5 * root * (ddistance / distance - drr / rr)
            fi = 1.0 + poly[mu] * root
            fj = 1.0 + poly[nu] * root
            dfi = dpoly[mu] * root + poly[mu] * droot
            dfj = dpoly[nu] * root + poly[nu] * droot
            polynomial = fi * fj
            dpolynomial = dfi * fj + fi * dfj
            log_zeta = np.log(2.0 * np.sqrt(zeta[mu] * zeta[nu]) / (zeta[mu] + zeta[nu]))
            dlog_zeta = (0.5 * (dzeta[mu] / zeta[mu] + dzeta[nu] / zeta[nu])
                         - (dzeta[mu] + dzeta[nu]) / (zeta[mu] + zeta[nu]))
            wexp = state.globals.get("wexp", 0.5)
            zeta_factor = np.exp(wexp * log_zeta)
            dzeta_factor = zeta_factor * (wexp * dlog_zeta + values["global.wexp"][0, 0] * log_zeta)
            ca, cb = classes[mu], classes[nu]
            k = angular[ca, cb]
            delta_en = en[a] - en[b]
            ddelta_en = den[a] - den[b]
            en_factor = 1.0 + enscale * delta_en**2
            den_factor = denscale * delta_en**2 + 2.0 * enscale * delta_en * ddelta_en
            prefactor = 0.5 * weight * zeta_factor
            dprefactor = 0.5 * (dweight * zeta_factor + weight * dzeta_factor)
            hsum = levels[mu] + levels[nu]
            dhsum = dlevels[mu] + dlevels[nu]
            common = prefactor * hsum
            dcommon = dprefactor * hsum + prefactor * dhsum
            pair_value = common * k * en_factor * polynomial
            pair_direction = (dcommon * k * en_factor * polynomial
                + common * angular_direction[ca, cb] * en_factor * polynomial
                + common * k * den_factor * polynomial + common * k * en_factor * dpolynomial)
            for endpoint, end_atom, end_slot in ((mu, a, slot_a), (nu, b, slot_b)):
                log_base = .5 / zeta[endpoint] - 1. / (zeta[mu] + zeta[nu])
                log_direction = (-.5 * dzeta[endpoint] / zeta[endpoint]**2
                    + (dzeta[mu]+dzeta[nu]) / (zeta[mu]+zeta[nu])**2)
                result["shell.h0exp"][end_atom, end_slot] += (
                    pair_direction * wexp * log_base + pair_value * (
                        values["global.wexp"][0,0] * log_base + wexp * log_direction))
            result["global.wexp"][0, 0] += (
                (dcommon * k * en_factor * polynomial
                 + common * angular_direction[ca, cb] * en_factor * polynomial
                 + common * k * den_factor * polynomial
                 + common * k * en_factor * dpolynomial) * log_zeta
                + common * k * en_factor * polynomial * dlog_zeta
            )
            dlevel = (dprefactor * k * en_factor * polynomial
                      + prefactor * angular_direction[ca, cb] * en_factor * polynomial
                      + prefactor * k * den_factor * polynomial
                      + prefactor * k * en_factor * dpolynomial)
            result["shell.lev"][a, slot_a] += dlevel
            result["shell.lev"][b, slot_b] += dlevel
            result["shell.kcn"][a, slot_a] -= dlevel * cn[a] + (
                prefactor * k * en_factor * polynomial
            ) * dcn[a]
            result["shell.kcn"][b, slot_b] -= dlevel * cn[b] + (
                prefactor * k * en_factor * polynomial
            ) * dcn[b]
            dpoly_a = (
                dcommon * k * en_factor * root * fj
                + common * angular_direction[ca, cb] * en_factor * root * fj
                + common * k * den_factor * root * fj
                + common * k * en_factor * (droot * fj + root * dfj)
            )
            dpoly_b = (
                dcommon * k * en_factor * root * fi
                + common * angular_direction[ca, cb] * en_factor * root * fi
                + common * k * den_factor * root * fi
                + common * k * en_factor * (droot * fi + root * dfi)
            )
            result["shell.poly"][a, slot_a] += dpoly_a
            result["shell.poly"][b, slot_b] += dpoly_b
            poly_sum = poly[mu] * fj + poly[nu] * fi
            dpoly_sum = (dpoly[mu] * fj + poly[mu] * dfj
                         + dpoly[nu] * fi + poly[nu] * dfi)
            radial = -root * poly_sum / (2.0 * rr)
            dradial = -(
                (droot * poly_sum + root * dpoly_sum) / (2.0 * rr)
                - root * poly_sum * drr / (2.0 * rr**2)
            )
            arad_dot = (
                dcommon * k * en_factor * radial
                + common * angular_direction[ca, cb] * en_factor * radial
                + common * k * den_factor * radial
                + common * k * en_factor * dradial
            )
            result["atom.arad"][a, 0] += arad_dot
            result["atom.arad"][b, 0] += arad_dot
            en_dot = 2.0 * (
                (dcommon * k + common * angular_direction[ca, cb])
                * polynomial * enscale * delta_en
                + common * k * dpolynomial * enscale * delta_en
                + common * k * polynomial * (
                    denscale * delta_en + enscale * ddelta_en
                )
            )
            result["atom.en"][a, 0] += en_dot
            result["atom.en"][b, 0] -= en_dot
            result["global.enscale"][0, 0] += (
                (dcommon * k + common * angular_direction[ca, cb])
                * polynomial * delta_en**2
                + common * k * dpolynomial * delta_en**2
                + common * k * polynomial * 2.0 * delta_en * ddelta_en
            )
            angular_dot = (
                dcommon * en_factor * polynomial
                + common * den_factor * polynomial
                + common * en_factor * dpolynomial
            )
            pair = (min(ca, cb), max(ca, cb))
            if pair == (0, 0):
                result["global.ks"][0, 0] += angular_dot
            elif pair == (0, 1):
                result["global.ks"][0, 0] += 0.5 * angular_dot
                result["global.kp"][0, 0] += 0.5 * angular_dot
            elif pair[0] == 0:
                result["global.ksd"][0, 0] += angular_dot
            elif pair == (1, 1):
                result["global.kp"][0, 0] += angular_dot
            elif pair[0] == 1:
                result["global.kpd"][0, 0] += angular_dot
            else:
                result["global.kd"][0, 0] += angular_dot
    for a, element in enumerate(state.elements):
        for slot, shell in enumerate(element.shells):
            if shell.coordination_shift_hartree == 0.0 and state.profile != "codeocean-v1":
                result["shell.kcn"][a, slot] = 0.0
            if shell.polynomial == 0.0 and state.profile != "codeocean-v1":
                result["shell.poly"][a, slot] = 0.0
        if f"atom {a}: arad" in state.floor_events:
            result["atom.arad"][a, 0] = 0.0
    return result


def analytic_third_order_head_adjoints(
    atomic_numbers: Sequence[int], state: NNXTBParameterState,
    shell_charges: np.ndarray, shell_atoms: Sequence[int],
    shell_labels: Sequence[str],
) -> dict[str, np.ndarray]:
    """Exact frozen-SCC adjoints of the cubic monopole energy."""
    numbers = tuple(int(number) for number in atomic_numbers)
    q = np.asarray(shell_charges, dtype=float)
    if q.shape != (len(shell_atoms),) or len(shell_labels) != len(shell_atoms):
        raise ValueError("third-order adjoint shell arrays are inconsistent")
    atom_gam3 = np.zeros((len(numbers), 1))
    shell_third = np.zeros((len(numbers), 4))
    angular_scale = {"s": 1.0, "p": 0.5, "d": 0.25}
    for shell_index, (atom, label) in enumerate(zip(shell_atoms, shell_labels, strict=True)):
        slot = next(
            i for i, shell in enumerate(state.elements[atom].shells)
            if shell.label == label
        )
        base = angular_scale[label[-1]] * q[shell_index]**3 / 3.0
        atom_gam3[atom, 0] += base * state.third_scales[atom][label]
        shell_third[atom, slot] += base * state.elements[atom].hubbard_derivative_hartree
    return {"atom.gam3": atom_gam3, "shell.third": shell_third}


def analytic_scc_hardness_head_adjoints(
    atomic_numbers: Sequence[int], coordinates_bohr: np.ndarray,
    state: NNXTBParameterState, shell_charges: np.ndarray,
    shell_atoms: Sequence[int], shell_labels: Sequence[str],
) -> dict[str, np.ndarray]:
    """Analytic Klopman--Ohno head adjoints at fixed shell charges."""
    numbers = tuple(int(number) for number in atomic_numbers)
    positions = np.asarray(coordinates_bohr, dtype=float)
    q = np.asarray(shell_charges, dtype=float)
    nshells = len(shell_atoms)
    if positions.shape != (len(numbers), 3) or q.shape != (nshells,) or len(shell_labels) != nshells:
        raise ValueError("SCC hardness adjoint state has incompatible dimensions")
    slots = []
    hardness = np.empty(nshells)
    for index, (atom, label) in enumerate(zip(shell_atoms, shell_labels, strict=True)):
        slot = next(
            i for i, shell in enumerate(state.elements[atom].shells)
            if shell.label == label
        )
        slots.append(slot)
        hardness[index] = (
            state.elements[atom].hubbard_hartree
            * state.elements[atom].shells[slot].hubbard_scale
        )
    exponent = state.globals["alphaj"]
    d_eta = np.zeros(nshells)
    d_exponent = 0.0
    for u in range(nshells):
        for v in range(nshells):
            a, b = shell_atoms[u], shell_atoms[v]
            r = 0.0 if a == b else float(np.linalg.norm(positions[a] - positions[b]))
            fav = 0.5 * (hardness[u] + hardness[v])
            if r == 0.0:
                dgamma_dfav = 1.0
            else:
                rg = r**exponent
                fg = fav**(-exponent)
                total = rg + fg
                gamma = total**(-1.0 / exponent)
                dgamma_dfav = fg / fav * gamma / total
                dtotal_dexponent = rg * np.log(r) - fg * np.log(fav)
                dgamma_dexponent = gamma * (
                    np.log(total) / exponent**2
                    - dtotal_dexponent / (exponent * total)
                )
                d_exponent += 0.5 * q[u] * q[v] * dgamma_dexponent
            d_eta[u] += 0.5 * q[u] * q[v] * dgamma_dfav
    d_gam = np.zeros((len(numbers), 1))
    d_lpar = np.zeros((len(numbers), 4))
    for index, (atom, slot) in enumerate(zip(shell_atoms, slots, strict=True)):
        shell = state.elements[atom].shells[slot]
        d_gam[atom, 0] += d_eta[index] * shell.hubbard_scale
        if shell.hubbard_scale != 1.0 or state.profile == "codeocean-v1":
            d_lpar[atom, slot] += d_eta[index] * state.elements[atom].hubbard_hartree
    for atom in range(len(numbers)):
        if f"atom {atom}: gam" in state.floor_events:
            d_gam[atom, 0] = 0.0
    return {
        "atom.gam": d_gam,
        "shell.lpar": d_lpar,
        "global.alphaj": np.array([[d_exponent]]),
    }


def analytic_scc_parameter_fock_direction(
    coordinates_bohr: np.ndarray, state: NNXTBParameterState,
    shell_charges: np.ndarray, shell_atoms: Sequence[int],
    shell_labels: Sequence[str], overlap: np.ndarray,
    orbital_shell: Sequence[int], parameter_direction: Mapping[str, np.ndarray],
    *, return_shell_potential: bool = False,
) -> np.ndarray:
    """Explicit fixed-density SCC Fock source for a parameter direction.

    Covers Klopman--Ohno hardness/exponent and cubic-shell heads. The
    resulting AO matrix is the inhomogeneity for the ordinary CPSCF solver;
    the density kernel must be supplied by the converged overlaid state.
    """
    positions = np.asarray(coordinates_bohr, dtype=float)
    natoms = len(state.elements)
    nshells = len(shell_atoms)
    charges = np.asarray(shell_charges, dtype=float)
    metric = np.asarray(overlap, dtype=float)
    orbital_map = np.asarray(orbital_shell, dtype=int)
    if (positions.shape != (natoms, 3) or charges.shape != (nshells,)
            or len(shell_labels) != nshells or metric.shape != (len(orbital_map),) * 2
            or np.any(orbital_map < 0) or np.any(orbital_map >= nshells)):
        raise ValueError("SCC parameter source dimensions are inconsistent")
    shapes = {"atom.gam": (natoms, 1), "atom.gam3": (natoms, 1),
              "shell.lpar": (natoms, 4), "shell.third": (natoms, 4),
              "global.alphaj": (1, 1)}
    if set(parameter_direction) - shapes.keys():
        raise ValueError("unsupported SCC parameter direction")
    values = {}
    for name, shape in shapes.items():
        value = np.asarray(parameter_direction.get(name, np.zeros(shape)), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid SCC direction for {name}")
        values[name] = value
    hardness = np.empty(nshells)
    d_hardness = np.empty(nshells)
    d_third = np.empty(nshells)
    angular_scale = {"s": 1.0, "p": 0.5, "d": 0.25}
    for index, (atom, label) in enumerate(zip(shell_atoms, shell_labels, strict=True)):
        slot = next(i for i, shell in enumerate(state.elements[atom].shells)
                    if shell.label == label)
        element = state.elements[atom]
        shell = element.shells[slot]
        gam = element.hubbard_hartree
        scale = shell.hubbard_scale
        dgam = (0.0 if f"atom {atom}: gam" in state.floor_events
                else values["atom.gam"][atom, 0])
        dscale = (values["shell.lpar"][atom, slot]
                  if scale != 1.0 or state.profile == "codeocean-v1" else 0.0)
        hardness[index] = gam * scale
        d_hardness[index] = dgam * scale + gam * dscale
        third_scale = state.third_scales[atom][label]
        d_third[index] = angular_scale[label[-1]] * (
            values["atom.gam3"][atom, 0] * third_scale
            + element.hubbard_derivative_hartree
            * values["shell.third"][atom, slot]
        )
    exponent = state.globals["alphaj"]
    d_exponent = float(values["global.alphaj"][0, 0])
    d_interaction = np.zeros((nshells, nshells))
    for u in range(nshells):
        for v in range(u, nshells):
            a, b = shell_atoms[u], shell_atoms[v]
            fav = 0.5 * (hardness[u] + hardness[v])
            dfav = 0.5 * (d_hardness[u] + d_hardness[v])
            if a == b:
                derivative = dfav
            else:
                radius = float(np.linalg.norm(positions[a] - positions[b]))
                if radius <= 0.0:
                    raise ValueError("distinct SCC atoms cannot coincide")
                radial = radius**exponent
                onsite = fav**(-exponent)
                total = radial + onsite
                gamma = total**(-1.0 / exponent)
                dgamma_dfav = onsite / fav * gamma / total
                dtotal_dexponent = radial * np.log(radius) - onsite * np.log(fav)
                dgamma_dexponent = gamma * (
                    np.log(total) / exponent**2
                    - dtotal_dexponent / (exponent * total)
                )
                derivative = dgamma_dfav * dfav + dgamma_dexponent * d_exponent
            d_interaction[u, v] = d_interaction[v, u] = derivative
    potential = d_interaction @ charges + d_third * charges**2
    if return_shell_potential:
        return potential
    orbital_potential = potential[orbital_map]
    return -0.5 * metric * (
        orbital_potential[:, None] + orbital_potential[None, :]
    )


def analytic_scc_head_adjoint_directional(
    coordinates_bohr: np.ndarray, state: NNXTBParameterState,
    shell_charges: np.ndarray, shell_charge_direction: np.ndarray,
    shell_atoms: Sequence[int], shell_labels: Sequence[str],
    parameter_direction: Mapping[str, np.ndarray],
    *, coordinate_direction: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Analytic SCC head response to parameters, charges and geometry.

    The charge direction can be supplied from the *coupled* CPSCF density
    response, including cross-family perturbations. No numerical parameter
    or charge differences are used by this routine.
    """
    positions = np.asarray(coordinates_bohr, dtype=float)
    dxyz = (np.zeros_like(positions) if coordinate_direction is None else
            np.asarray(coordinate_direction, dtype=float))
    if dxyz.shape != positions.shape or not np.all(np.isfinite(dxyz)):
        raise ValueError("invalid SCC coordinate direction")
    natoms = len(state.elements)
    nshells = len(shell_atoms)
    q = np.asarray(shell_charges, dtype=float)
    dq = np.asarray(shell_charge_direction, dtype=float)
    if (positions.shape != (natoms, 3) or q.shape != (nshells,)
            or dq.shape != q.shape or len(shell_labels) != nshells):
        raise ValueError("SCC adjoint direction dimensions are inconsistent")
    shapes = {"atom.gam": (natoms, 1), "atom.gam3": (natoms, 1),
              "shell.lpar": (natoms, 4), "shell.third": (natoms, 4),
              "global.alphaj": (1, 1)}
    if set(parameter_direction) - shapes.keys():
        raise ValueError("unsupported SCC adjoint parameter direction")
    values = {}
    for name, shape in shapes.items():
        value = np.asarray(parameter_direction.get(name, np.zeros(shape)), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid SCC adjoint direction for {name}")
        values[name] = value
    hardness = np.empty(nshells)
    d_hardness = np.empty(nshells)
    slots = []
    for index, (atom, label) in enumerate(zip(shell_atoms, shell_labels, strict=True)):
        slot = next(i for i, shell in enumerate(state.elements[atom].shells)
                    if shell.label == label)
        slots.append(slot)
        element = state.elements[atom]
        scale = element.shells[slot].hubbard_scale
        dgam = (0.0 if f"atom {atom}: gam" in state.floor_events
                else values["atom.gam"][atom, 0])
        dscale = (values["shell.lpar"][atom, slot]
                  if scale != 1.0 or state.profile == "codeocean-v1" else 0.0)
        hardness[index] = element.hubbard_hartree * scale
        d_hardness[index] = dgam * scale + element.hubbard_hartree * dscale
    k = state.globals["alphaj"]
    dk = float(values["global.alphaj"][0, 0])
    d_eta = np.zeros(nshells)
    dd_eta = np.zeros(nshells)
    d_k = 0.0
    dd_k = 0.0
    for u in range(nshells):
        for v in range(nshells):
            a, b = shell_atoms[u], shell_atoms[v]
            fav = 0.5 * (hardness[u] + hardness[v])
            dfav = 0.5 * (d_hardness[u] + d_hardness[v])
            if a == b:
                gamma_f, dgamma_f = 1.0, 0.0
                gamma_k, dgamma_k = 0.0, 0.0
            else:
                radius = float(np.linalg.norm(positions[a] - positions[b]))
                if radius <= 0.0:
                    raise ValueError("distinct SCC atoms cannot coincide")
                logr = np.log(radius)
                dlogr = float(np.dot(positions[a] - positions[b],
                                    dxyz[a] - dxyz[b])) / radius**2
                logf = np.log(fav)
                radial = radius**k
                onsite = fav**(-k)
                total = radial + onsite
                logtotal = np.log(total)
                gamma = total**(-1.0 / k)
                d_radial = radial * (logr * dk + k * dlogr)
                d_onsite = onsite * (-logf * dk - k * dfav / fav)
                d_total = d_radial + d_onsite
                d_loggamma = -d_total / (k * total) + logtotal * dk / k**2
                d_gamma = gamma * d_loggamma
                gamma_f = gamma * onsite / (fav * total)
                dgamma_f = gamma_f * (
                    d_loggamma + d_onsite / onsite - dfav / fav
                    - d_total / total
                )
                t_k = radial * logr - onsite * logf
                d_t_k = (d_radial * logr + radial * dlogr
                         - d_onsite * logf - onsite * dfav / fav)
                bracket = logtotal / k**2 - t_k / (k * total)
                d_bracket = (
                    d_total / (total * k**2)
                    - 2.0 * logtotal * dk / k**3
                    - d_t_k / (k * total)
                    + t_k * dk / (k**2 * total)
                    + t_k * d_total / (k * total**2)
                )
                gamma_k = gamma * bracket
                dgamma_k = d_gamma * bracket + gamma * d_bracket
            weight = 0.5 * q[u] * q[v]
            dweight = 0.5 * (dq[u] * q[v] + q[u] * dq[v])
            d_eta[u] += weight * gamma_f
            dd_eta[u] += dweight * gamma_f + weight * dgamma_f
            d_k += weight * gamma_k
            dd_k += dweight * gamma_k + weight * dgamma_k
    gam = np.zeros((natoms, 1))
    lpar = np.zeros((natoms, 4))
    gam3 = np.zeros((natoms, 1))
    third = np.zeros((natoms, 4))
    angular_scale = {"s": 1.0, "p": 0.5, "d": 0.25}
    for index, (atom, label) in enumerate(zip(shell_atoms, shell_labels, strict=True)):
        slot = slots[index]
        element = state.elements[atom]
        scale = element.shells[slot].hubbard_scale
        dgam = (0.0 if f"atom {atom}: gam" in state.floor_events
                else values["atom.gam"][atom, 0])
        dscale = (values["shell.lpar"][atom, slot]
                  if scale != 1.0 or state.profile == "codeocean-v1" else 0.0)
        gam[atom, 0] += dd_eta[index] * scale + d_eta[index] * dscale
        if scale != 1.0 or state.profile == "codeocean-v1":
            lpar[atom, slot] += (
                dd_eta[index] * element.hubbard_hartree
                + d_eta[index] * dgam
            )
        angular = angular_scale[label[-1]]
        base = angular * q[index]**3 / 3.0
        dbase = angular * q[index]**2 * dq[index]
        gam3[atom, 0] += (
            dbase * state.third_scales[atom][label]
            + base * values["shell.third"][atom, slot]
        )
        third[atom, slot] += (
            dbase * element.hubbard_derivative_hartree
            + base * values["atom.gam3"][atom, 0]
        )
    for atom in range(natoms):
        if f"atom {atom}: gam" in state.floor_events:
            gam[atom, 0] = 0.0
    return {"atom.gam": gam, "shell.lpar": lpar,
            "global.alphaj": np.array([[dd_k]]),
            "atom.gam3": gam3, "shell.third": third}


def analytic_aes_head_adjoints(
    coordinates_bohr: np.ndarray, coordination: np.ndarray,
    state: NNXTBParameterState, multipoles,
) -> dict[str, np.ndarray]:
    """Exact AES damping/onsite adjoints at fixed multipoles and CN."""
    from .multipole import prepare_anisotropic_geometry

    positions = np.asarray(coordinates_bohr, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    natoms = len(state.elements)
    if positions.shape != (natoms, 3) or cn.shape != (natoms,):
        raise ValueError("AES head adjoint geometry has incompatible dimensions")
    parameters = state.anisotropic_parameters(cn)
    geometry = prepare_anisotropic_geometry(positions, parameters)
    left, right = geometry.left, geometry.right
    r = np.sqrt(geometry.distance2)
    radius = 0.5 * (
        parameters.damping_radius_bohr[left]
        + parameters.damping_radius_bohr[right]
    )
    q = np.asarray(multipoles.charges)
    mu = np.asarray(multipoles.dipoles_bohr)
    theta = np.asarray(multipoles.quadrupoles_bohr2)
    if q.shape != (natoms,) or mu.shape != (natoms, 3) or theta.shape != (natoms, 3, 3):
        raise ValueError("AES head adjoint multipoles have incompatible dimensions")
    displacement = geometry.displacement
    left_mu = np.einsum("pi,pi->p", mu[left], displacement)
    right_mu = np.einsum("pi,pi->p", mu[right], displacement)
    left_theta = np.einsum("pi,pij,pj->p", displacement, theta[left], displacement)
    right_theta = np.einsum("pi,pij,pj->p", displacement, theta[right], displacement)
    angular3 = -q[left] * right_mu + q[right] * left_mu
    angular5 = (
        q[left] * right_theta + q[right] * left_theta
        - 3.0 * left_mu * right_mu
        + np.einsum("pi,pi->p", mu[left], mu[right]) * geometry.distance2
    )
    ratio = radius / r
    power3 = ratio**parameters.a3
    power5 = ratio**parameters.a5
    fraction3 = 6.0 * power3 / (1.0 + 6.0 * power3)
    fraction5 = 6.0 * power5 / (1.0 + 6.0 * power5)
    d_radius = np.zeros(natoms)
    pair_radius = (
        -geometry.f3 * angular3 * fraction3 * parameters.a3 / radius
        -geometry.f5 * angular5 * fraction5 * parameters.a5 / radius
    )
    np.add.at(d_radius, left, 0.5 * pair_radius)
    np.add.at(d_radius, right, 0.5 * pair_radius)
    base = np.asarray([element.multipole_radius_bohr for element in state.elements])
    vcn = np.asarray([element.multipole_valence_coordination for element in state.elements])
    k = state.globals["aesexp"]
    shift = state.globals["aesshift"]
    maximum = state.globals["aesrmax"]
    logistic = np.exp(-np.logaddexp(0.0, -k * (cn - vcn - shift)))
    logistic_prime = logistic * (1.0 - logistic)
    span = maximum - base
    d_mprad = d_radius * (1.0 - logistic)
    d_mpvcn = -d_radius * span * k * logistic_prime
    d_aesexp = np.sum(d_radius * span * (cn - vcn - shift) * logistic_prime)
    d_aesshift = -np.sum(d_radius * span * k * logistic_prime)
    d_aesrmax = np.sum(d_radius * logistic)
    for a in range(natoms):
        if f"atom {a}: mprad" in state.floor_events:
            d_mprad[a] = 0.0
    logratio = np.log(ratio)
    d_aesdmp3 = -np.sum(geometry.f3 * angular3 * fraction3 * logratio)
    d_aesdmp5 = -np.sum(geometry.f5 * angular5 * fraction5 * logratio)
    return {
        "atom.dpol": np.sum(mu**2, axis=1)[:, None],
        "atom.qpol": np.sum(theta**2, axis=(1, 2))[:, None],
        "atom.mprad": d_mprad[:, None],
        "atom.mpvcn": d_mpvcn[:, None],
        "global.aesdmp3": np.array([[d_aesdmp3]]),
        "global.aesdmp5": np.array([[d_aesdmp5]]),
        "global.aesexp": np.array([[d_aesexp]]),
        "global.aesshift": np.array([[d_aesshift]]),
        "global.aesrmax": np.array([[d_aesrmax]]),
    }


def analytic_aes_parameter_moment_potential_direction(
    coordinates_bohr: np.ndarray, coordination: np.ndarray,
    state: NNXTBParameterState, multipoles,
    parameter_direction: Mapping[str, np.ndarray],
    *, coordination_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit AES moment-potential source for a parameter direction."""
    from .multipole import AtomicMultipoles, anisotropic_moment_potential_direction

    positions = np.asarray(coordinates_bohr, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    natoms = len(state.elements)
    if positions.shape != (natoms, 3) or cn.shape != (natoms,):
        raise ValueError("AES parameter direction has incompatible geometry")
    shapes = {"atom.dpol": (natoms, 1), "atom.qpol": (natoms, 1),
              "atom.mprad": (natoms, 1), "atom.mpvcn": (natoms, 1),
              **{f"global.{name}": (1, 1) for name in (
                  "aesdmp3", "aesdmp5", "aesexp", "aesshift", "aesrmax",
              )}}
    if set(parameter_direction) - shapes.keys():
        raise ValueError("unsupported AES parameter direction")
    values = {}
    for name, shape in shapes.items():
        value = np.asarray(parameter_direction.get(name, np.zeros(shape)), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid AES direction for {name}")
        values[name] = value
    base = np.asarray([element.multipole_radius_bohr for element in state.elements])
    vcn = np.asarray([element.multipole_valence_coordination for element in state.elements])
    k = state.globals["aesexp"]
    shift = state.globals["aesshift"]
    maximum = state.globals["aesrmax"]
    argument = cn - vcn - shift
    logistic = np.exp(-np.logaddexp(0.0, -k * argument))
    dbase = values["atom.mprad"][:, 0].copy()
    for atom in range(natoms):
        if f"atom {atom}: mprad" in state.floor_events:
            dbase[atom] = 0.0
    dargument = (-values["atom.mpvcn"][:, 0]
                 - values["global.aesshift"][0, 0])
    if coordination_direction is not None:
        cndot = np.asarray(coordination_direction, dtype=float)
        if cndot.shape != cn.shape or not np.all(np.isfinite(cndot)):
            raise ValueError("invalid AES coordination direction")
        dargument = dargument + cndot
    dlogistic = logistic * (1.0 - logistic) * (
        values["global.aesexp"][0, 0] * argument + k * dargument
    )
    radius_direction = (
        dbase * (1.0 - logistic)
        + values["global.aesrmax"][0, 0] * logistic
        + (maximum - base) * dlogistic
    )
    zero_moments = AtomicMultipoles(
        np.zeros(natoms), np.zeros((natoms, 3)), np.zeros((natoms, 3, 3)),
    )
    return anisotropic_moment_potential_direction(
        multipoles, zero_moments, positions, np.zeros_like(positions),
        state.anisotropic_parameters(cn), radius_direction,
        exponent_direction3=values["global.aesdmp3"][0, 0],
        exponent_direction5=values["global.aesdmp5"][0, 0],
        dipole_xc_direction=values["atom.dpol"][:, 0],
        quadrupole_xc_direction=values["atom.qpol"][:, 0],
    )


def analytic_aes_head_adjoint_directional(
    coordinates_bohr: np.ndarray, coordination: np.ndarray,
    state: NNXTBParameterState, multipoles, multipole_direction,
    parameter_direction: Mapping[str, np.ndarray],
    *, coordinate_direction: np.ndarray | None = None,
    coordination_direction: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Analytic AES head response to parameters, multipoles, geometry and CN."""
    positions = np.asarray(coordinates_bohr, dtype=float)
    cn = np.asarray(coordination, dtype=float)
    natoms = len(state.elements)
    dxyz = (np.zeros_like(positions) if coordinate_direction is None else
            np.asarray(coordinate_direction, dtype=float))
    dcn = (np.zeros_like(cn) if coordination_direction is None else
           np.asarray(coordination_direction, dtype=float))
    if (dxyz.shape != positions.shape or dcn.shape != cn.shape
            or not np.all(np.isfinite(dxyz)) or not np.all(np.isfinite(dcn))):
        raise ValueError("invalid AES geometry direction")
    if positions.shape != (natoms, 3) or cn.shape != (natoms,):
        raise ValueError("AES adjoint direction has incompatible geometry")
    shapes = {"atom.dpol": (natoms, 1), "atom.qpol": (natoms, 1),
              "atom.mprad": (natoms, 1), "atom.mpvcn": (natoms, 1),
              **{f"global.{name}": (1, 1) for name in (
                  "aesdmp3", "aesdmp5", "aesexp", "aesshift", "aesrmax",
              )}}
    if set(parameter_direction) - shapes.keys():
        raise ValueError("unsupported AES adjoint parameter direction")
    values = {}
    for name, shape in shapes.items():
        value = np.asarray(parameter_direction.get(name, np.zeros(shape)), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid AES adjoint direction for {name}")
        values[name] = value
    q, dq = np.asarray(multipoles.charges), np.asarray(multipole_direction.charges)
    mu, dmu = np.asarray(multipoles.dipoles_bohr), np.asarray(multipole_direction.dipoles_bohr)
    theta = np.asarray(multipoles.quadrupoles_bohr2)
    dtheta = np.asarray(multipole_direction.quadrupoles_bohr2)
    if (q.shape != (natoms,) or dq.shape != q.shape or mu.shape != (natoms, 3)
            or dmu.shape != mu.shape or theta.shape != (natoms, 3, 3)
            or dtheta.shape != theta.shape):
        raise ValueError("AES adjoint multipole directions are inconsistent")
    base = np.asarray([element.multipole_radius_bohr for element in state.elements])
    vcn = np.asarray([element.multipole_valence_coordination for element in state.elements])
    k = state.globals["aesexp"]
    shift = state.globals["aesshift"]
    maximum = state.globals["aesrmax"]
    argument = cn - vcn - shift
    logistic = np.exp(-np.logaddexp(0.0, -k * argument))
    logistic_prime = logistic * (1.0 - logistic)
    span = maximum - base
    db = values["atom.mprad"][:, 0].copy()
    for atom in range(natoms):
        if f"atom {atom}: mprad" in state.floor_events:
            db[atom] = 0.0
    dv = values["atom.mpvcn"][:, 0]
    dk = float(values["global.aesexp"][0, 0])
    dshift = float(values["global.aesshift"][0, 0])
    dmax = float(values["global.aesrmax"][0, 0])
    dargument = dcn - dv - dshift
    dlogistic = logistic_prime * (dk * argument + k * dargument)
    dlogistic_prime = (1.0 - 2.0 * logistic) * dlogistic
    radius = base + span * logistic
    dradius = db * (1.0 - logistic) + dmax * logistic + span * dlogistic
    radius_adj = np.zeros(natoms)
    dradius_adj = np.zeros(natoms)
    exponent_adj = np.zeros(2)
    dexponent_adj = np.zeros(2)
    for left in range(natoms):
        for right in range(left + 1, natoms):
            vector = positions[right] - positions[left]
            dvector = dxyz[right] - dxyz[left]
            distance2 = float(np.dot(vector, vector))
            ddistance2 = 2.0 * float(np.dot(vector, dvector))
            distance = np.sqrt(distance2)
            dlogdistance = 0.5 * ddistance2 / distance2
            pair_radius = 0.5 * (radius[left] + radius[right])
            dpair_radius = 0.5 * (dradius[left] + dradius[right])
            logratio = np.log(pair_radius / distance)
            dlogratio = dpair_radius / pair_radius - dlogdistance
            lp = float(np.dot(mu[left], vector))
            rp = float(np.dot(mu[right], vector))
            dlp = float(np.dot(dmu[left], vector) + np.dot(mu[left], dvector))
            drp = float(np.dot(dmu[right], vector) + np.dot(mu[right], dvector))
            lt = float(vector @ theta[left] @ vector)
            rt = float(vector @ theta[right] @ vector)
            dlt = float(vector @ dtheta[left] @ vector
                        + dvector @ theta[left] @ vector
                        + vector @ theta[left] @ dvector)
            drt = float(vector @ dtheta[right] @ vector
                        + dvector @ theta[right] @ vector
                        + vector @ theta[right] @ dvector)
            angular3 = -q[left] * rp + q[right] * lp
            dangular3 = (-dq[left] * rp - q[left] * drp
                         + dq[right] * lp + q[right] * dlp)
            angular5 = (q[left] * rt + q[right] * lt - 3.0 * lp * rp
                        + float(np.dot(mu[left], mu[right])) * distance2)
            dangular5 = (
                dq[left] * rt + q[left] * drt + dq[right] * lt + q[right] * dlt
                - 3.0 * (dlp * rp + lp * drp)
                + (float(np.dot(dmu[left], mu[right]))
                   + float(np.dot(mu[left], dmu[right]))) * distance2
                + float(np.dot(mu[left], mu[right])) * ddistance2
            )
            pair_radius_derivative = 0.0
            dpair_radius_derivative = 0.0
            for index, (power, angular, dangular, exponent, dexp) in enumerate((
                (3, angular3, dangular3, state.globals["aesdmp3"],
                 float(values["global.aesdmp3"][0, 0])),
                (5, angular5, dangular5, state.globals["aesdmp5"],
                 float(values["global.aesdmp5"][0, 0])),
            )):
                ratio_power = 6.0 * np.exp(exponent * logratio)
                fraction = ratio_power / (1.0 + ratio_power)
                value = distance**(-power) / (1.0 + ratio_power)
                dlog_power = dexp * logratio + exponent * dlogratio
                dfraction = fraction * (1.0 - fraction) * dlog_power
                dvalue = -value * (fraction * dlog_power + power * dlogdistance)
                dvalue_radius = -value * fraction * exponent / pair_radius
                ddvalue_radius = -(
                    (dvalue * fraction * exponent
                     + value * dfraction * exponent
                     + value * fraction * dexp) / pair_radius
                    - value * fraction * exponent * dpair_radius / pair_radius**2
                )
                dvalue_exponent = -value * fraction * logratio
                ddvalue_exponent = -(
                    (dvalue * fraction + value * dfraction) * logratio
                    + value * fraction * dlogratio
                )
                pair_radius_derivative += angular * dvalue_radius
                dpair_radius_derivative += (
                    dangular * dvalue_radius + angular * ddvalue_radius
                )
                exponent_adj[index] += angular * dvalue_exponent
                dexponent_adj[index] += (
                    dangular * dvalue_exponent + angular * ddvalue_exponent
                )
            radius_adj[left] += 0.5 * pair_radius_derivative
            radius_adj[right] += 0.5 * pair_radius_derivative
            dradius_adj[left] += 0.5 * dpair_radius_derivative
            dradius_adj[right] += 0.5 * dpair_radius_derivative
    base_jacobian = 1.0 - logistic
    dbase_jacobian = -dlogistic
    vcn_jacobian = -span * k * logistic_prime
    dvcn_jacobian = -(
        (dmax - db) * k * logistic_prime
        + span * dk * logistic_prime + span * k * dlogistic_prime
    )
    k_jacobian = span * argument * logistic_prime
    dk_jacobian = (
        (dmax - db) * argument * logistic_prime
        + span * dargument * logistic_prime
        + span * argument * dlogistic_prime
    )
    maximum_jacobian = logistic
    dmaximum_jacobian = dlogistic
    mprad = dradius_adj * base_jacobian + radius_adj * dbase_jacobian
    for atom in range(natoms):
        if f"atom {atom}: mprad" in state.floor_events:
            mprad[atom] = 0.0
    return {
        "atom.dpol": (2.0 * np.sum(mu * dmu, axis=1))[:, None],
        "atom.qpol": (2.0 * np.sum(theta * dtheta, axis=(1, 2)))[:, None],
        "atom.mprad": mprad[:, None],
        "atom.mpvcn": (dradius_adj * vcn_jacobian
                        + radius_adj * dvcn_jacobian)[:, None],
        "global.aesdmp3": np.array([[dexponent_adj[0]]]),
        "global.aesdmp5": np.array([[dexponent_adj[1]]]),
        "global.aesexp": np.array([[np.sum(
            dradius_adj * k_jacobian + radius_adj * dk_jacobian,
        )]]),
        "global.aesshift": np.array([[np.sum(
            dradius_adj * vcn_jacobian + radius_adj * dvcn_jacobian,
        )]]),
        "global.aesrmax": np.array([[np.sum(
            dradius_adj * maximum_jacobian
            + radius_adj * dmaximum_jacobian,
        )]]),
    }


def analytic_rcov_head_adjoints(
    coordinates_bohr: np.ndarray, state: NNXTBParameterState,
    h0_level_adjoints: np.ndarray, aes_mpvcn_adjoints: np.ndarray,
) -> np.ndarray:
    """Chain the frozen H0/AES CN adjoint through each NN covalent radius."""
    positions = np.asarray(coordinates_bohr, dtype=float)
    natoms = len(state.atoms)
    if (
        positions.shape != (natoms, 3)
        or np.shape(h0_level_adjoints) != (natoms, 4)
        or np.shape(aes_mpvcn_adjoints) != (natoms, 1)
    ):
        raise ValueError("NN covalent-radius adjoint state has incompatible dimensions")
    d_cn = -np.asarray(aes_mpvcn_adjoints, dtype=float)[:, 0].copy()
    for atom, element in enumerate(state.elements):
        for slot, shell in enumerate(element.shells):
            d_cn[atom] -= shell.coordination_shift_hartree * h0_level_adjoints[atom, slot]
    result = np.zeros((natoms, 1))
    radius = np.asarray([item.covalent_radius_bohr for item in state.atoms])
    for a in range(natoms):
        for b in range(a + 1, natoms):
            distance = float(np.linalg.norm(positions[a] - positions[b]))
            if distance <= 0.0:
                raise ValueError("distinct atoms cannot occupy the same point")
            scale1 = 4.0 * (radius[a] + radius[b]) / 3.0
            scale2 = scale1 + 2.0
            first = 1.0 / (1.0 + np.exp(-10.0 * (scale1 / distance - 1.0)))
            second = 1.0 / (1.0 + np.exp(-20.0 * (scale2 / distance - 1.0)))
            derivative = (4.0 / (3.0 * distance)) * (
                10.0 * first * (1.0 - first) * second
                + 20.0 * second * (1.0 - second) * first
            )
            contribution = 0.75 * (d_cn[a] + d_cn[b]) * derivative
            result[a, 0] += contribution
            result[b, 0] += contribution
    for atom in range(natoms):
        if f"atom {atom}: rcov" in state.floor_events:
            result[atom, 0] = 0.0
    return result


def analytic_rcov_coordination_jacobian(
    coordinates_bohr: np.ndarray, state: NNXTBParameterState,
    *, coordinate_direction: np.ndarray | None = None,
    radius_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CN radius Jacobian and its exact joint geometry/radius direction.

    Columns refer to MATRIX ``atom.rcov`` shifts, including the 3/4
    conversion to the atom radius and the parameter floor branch.
    """
    xyz = np.asarray(coordinates_bohr, dtype=float)
    n = len(state.atoms)
    dx = np.zeros_like(xyz) if coordinate_direction is None else np.asarray(coordinate_direction, dtype=float)
    dr = np.zeros((n, 1)) if radius_direction is None else np.asarray(radius_direction, dtype=float)
    if (xyz.shape != (n, 3) or dx.shape != xyz.shape or dr.shape != (n, 1)
            or not all(np.all(np.isfinite(v)) for v in (xyz, dx, dr))):
        raise ValueError("invalid CN radius response dimensions")
    active = np.asarray([f"atom {a}: rcov" not in state.floor_events for a in range(n)])
    raw_dot = dr[:, 0] * active
    radii = np.asarray([a.covalent_radius_bohr for a in state.atoms])
    jac = np.zeros((n, n))
    jac_dot = np.zeros_like(jac)
    for a in range(n):
        for b in range(a + 1, n):
            vector = xyz[a] - xyz[b]
            r = float(np.linalg.norm(vector))
            if r <= 0.0:
                raise ValueError("distinct atoms cannot occupy the same point")
            rdot = float(vector @ (dx[a] - dx[b])) / r
            s1 = (4.0 / 3.0) * (radii[a] + radii[b])
            s2 = s1 + 2.0
            sdot = raw_dot[a] + raw_dot[b]
            f1 = np.exp(-np.logaddexp(0.0, -10.0 * (s1 / r - 1.0)))
            f2 = np.exp(-np.logaddexp(0.0, -20.0 * (s2 / r - 1.0)))
            p1, p2 = f1 * (1.0 - f1), f2 * (1.0 - f2)
            d1 = 10.0 * (sdot / r - s1 * rdot / r**2)
            d2 = 20.0 * (sdot / r - s2 * rdot / r**2)
            value = (10.0 * p1 * f2 + 20.0 * p2 * f1) / r
            value_dot = (
                10.0 * (p1 * (1.0 - 2.0 * f1) * d1 * f2 + p1 * p2 * d2)
                + 20.0 * (p2 * (1.0 - 2.0 * f2) * d2 * f1 + p2 * p1 * d1)
            ) / r - value * rdot / r
            for row in (a, b):
                for column in (a, b):
                    jac[row, column] += value * active[column]
                    jac_dot[row, column] += value_dot * active[column]
    return jac, jac_dot


def analytic_rcov_head_adjoint_directional(
    coordinates_bohr: np.ndarray, state: NNXTBParameterState,
    h0_level_adjoints: np.ndarray, aes_mpvcn_adjoints: np.ndarray,
    h0_level_direction: np.ndarray, aes_mpvcn_direction: np.ndarray,
    parameter_direction: Mapping[str, np.ndarray],
    *, coordinate_direction: np.ndarray | None = None,
) -> np.ndarray:
    """Full radius-head response, including the H0/AES CN cross terms."""
    n = len(state.atoms)
    jac, jac_dot = analytic_rcov_coordination_jacobian(
        coordinates_bohr, state, coordinate_direction=coordinate_direction,
        radius_direction=parameter_direction.get("atom.rcov"),
    )
    cn_adjoint = -np.asarray(aes_mpvcn_adjoints)[:, 0].copy()
    cn_adjoint_dot = -np.asarray(aes_mpvcn_direction)[:, 0].copy()
    dkcn = np.asarray(parameter_direction.get("shell.kcn", np.zeros((n, 4))))
    for atom, element in enumerate(state.elements):
        for slot, shell in enumerate(element.shells):
            cn_adjoint[atom] -= shell.coordination_shift_hartree * h0_level_adjoints[atom, slot]
            cn_adjoint_dot[atom] -= (
                shell.coordination_shift_hartree * h0_level_direction[atom, slot]
                + (dkcn[atom, slot] if shell.coordination_shift_hartree != 0.0 else 0.0)
                * h0_level_adjoints[atom, slot]
            )
    return (jac_dot.T @ cn_adjoint + jac.T @ cn_adjoint_dot)[:, None]
