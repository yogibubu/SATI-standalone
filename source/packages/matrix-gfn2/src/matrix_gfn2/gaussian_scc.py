"""Fixed, atom-resolved Gaussian corrections to dense SCC interactions.

Only selected interregion couplings involving a positive-width atom change.
Point sites can belong to the solvent region without receiving a parameter.
Intraregion interactions retain the native kernel. The same radial function
supplies SCF values, coordinate derivatives and mixed molecular response.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np

from .gaussian_charge import gaussian_charge_coulomb


@dataclass(frozen=True)
class GaussianSCCProfile:
    atom_widths_bohr: tuple[float, ...]
    weight: float = 1.0
    interaction_region: tuple[bool, ...] | None = None

    def __post_init__(self):
        widths = tuple(float(x) for x in self.atom_widths_bohr)
        if not widths or any(not np.isfinite(x) or x < 0 for x in widths):
            raise ValueError("Gaussian SCC widths must be finite and nonnegative")
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("Gaussian SCC weight must be between zero and one")
        object.__setattr__(self, "atom_widths_bohr", widths)
        object.__setattr__(self, "weight", float(self.weight))
        if self.interaction_region is not None:
            region = tuple(self.interaction_region)
            if len(region) != len(widths) or any(type(x) is not bool for x in region):
                raise ValueError("Gaussian interaction region requires one boolean per atom")
            if any(width > 0 and not inside for width, inside in zip(widths, region)):
                raise ValueError("positive-width atoms must belong to the interaction region")
            object.__setattr__(self, "interaction_region", region)

    @property
    def is_identity(self):
        flags = [x > 0 for x in self.atom_widths_bohr]
        if self.interaction_region is None:
            return self.weight == 0 or all(flags) or not any(flags)
        return self.weight == 0 or not any(flags) or all(self.interaction_region)

    @property
    def content_sha256(self):
        payload = dict(widths=self.atom_widths_bohr, weight=self.weight)
        if self.interaction_region is not None:
            payload["interaction_region"] = self.interaction_region
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def shell_data(self, partition):
        widths = np.asarray(self.atom_widths_bohr)[np.asarray(partition.shell_atom)]
        positive = widths > 0
        region = positive if self.interaction_region is None else np.asarray(
            self.interaction_region)[np.asarray(partition.shell_atom)]
        pair_width = np.hypot(widths[:, None], widths[None, :])
        mask = (region[:, None] != region[None, :]) & (pair_width > 0)
        return mask, pair_width

    def modify_values(self, matrix, coordinates, partition):
        mask, widths = self.shell_data(partition)
        i, j = np.where(mask)
        centers = np.asarray(coordinates)[np.asarray(partition.shell_atom)]
        distance = np.linalg.norm(centers[i] - centers[j], axis=1)
        gaussian = gaussian_charge_coulomb(distance, widths[i, j])[0]
        out = np.array(matrix, copy=True)
        out[i, j] += self.weight * (gaussian - out[i, j])
        return wrap_scc_interaction(out, coordinates, partition)

    def modify_jet(self, gamma, coordinates, partition):
        mask, widths = self.shell_data(partition)
        i, j = np.where(mask)
        atom = np.asarray(partition.shell_atom)
        distance = ((coordinates[atom[i]] - coordinates[atom[j]])**2).sum(axis=1)**0.5
        gaussian = (distance / widths[i, j]).erf() / distance
        gamma[i, j] = (1-self.weight)*gamma[i, j] + self.weight*gaussian
        return gamma


def water_oxygen_gaussian_profile(atoms, water_fragments, *, width_scale):
    """One global oxygen-width scale; no hydrogen or mixture parameters.

    Oxygen's reference width matches its Gaussian Coulomb self-interaction
    to the published atomic Hubbard parameter: sigma_0=sqrt(2/pi)/U_O.
    Only oxygen-to-solute SCC pairs change. Hydrogen interactions and all
    interactions inside the declared water region retain the baseline.
    The charge distribution is normalized; charges are never rescaled.
    """
    from numbers import Integral
    from .parameters import load_gfn2_parameters

    if isinstance(width_scale, bool) or not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError("oxygen width scale must be finite and positive")
    widths = [0.] * len(atoms)
    region = [False] * len(atoms)
    sigma = float(width_scale) * np.sqrt(2/np.pi) / load_gfn2_parameters().elements[8].hubbard_hartree
    used = set()
    for fragment in water_fragments:
        indices = tuple(fragment)
        if len(indices) != 3 or any(isinstance(i, bool) or not isinstance(i, Integral)
                or i < 0 or i >= len(atoms) for i in indices):
            raise ValueError("a water fragment requires three valid atom indices")
        if len(set(indices)) != 3 or used.intersection(indices):
            raise ValueError("water fragments must be distinct and disjoint")
        if sorted(str(atoms[i]) for i in indices) != ["H", "H", "O"]:
            raise ValueError("a water fragment must contain O/H/H")
        used.update(indices)
        for i in indices:
            region[i] = True
            if atoms[i] == "O":
                widths[i] = sigma
    return GaussianSCCProfile(tuple(widths), weight=1., interaction_region=tuple(region))


class GaussianSCCInteraction(np.ndarray):
    """Dense matrix with exact gas-phase SCC geometry derivatives.

    The matrix may also contain ALPB values. ALPB geometry derivatives stay
    with the existing solvent routines; matrix-times-charge derivatives
    include its electronic response in the usual way.
    """

    def __new__(cls, matrix, coordinates, partition):
        obj = np.asarray(matrix).view(cls)
        obj.coordinates = np.array(coordinates, copy=True)
        obj.partition = partition
        return obj

    def __array_finalize__(self, source):
        if source is not None:
            self.coordinates = getattr(source, "coordinates", None)
            self.partition = getattr(source, "partition", None)

    def _pairs(self):
        p = self.partition
        atom = np.asarray(p.shell_atom)
        i, j = np.tril_indices(len(atom), -1)
        different = atom[i] != atom[j]
        i, j = i[different], j[different]
        a, b = atom[i], atom[j]
        delta = self.coordinates[a] - self.coordinates[b]
        r = np.linalg.norm(delta, axis=1)
        hardness = np.asarray(p.hardness_hartree)
        average = 0.5*(hardness[i]+hardness[j])
        softened = r*r + average**-2
        first = -r * softened**-1.5
        second = -softened**-1.5 + 3*r*r*softened**-2.5
        profile = p.electrostatic_profile
        if not getattr(profile, 'geometry_dependent', False):
            mask, widths = profile.shell_data(p)
            selected = mask[i, j]
            _, g1, g2 = gaussian_charge_coulomb(r[selected], widths[i[selected], j[selected]])
            first[selected] += profile.weight*(g1-first[selected])
            second[selected] += profile.weight*(g2-second[selected])
        return i, j, a, b, delta, r, first, second

    def evaluate(self, charges, *, compute_gradient=True):
        q = np.asarray(charges)
        potential = np.asarray(self) @ q
        gradient = np.zeros_like(self.coordinates)
        if compute_gradient:
            i, j, a, b, delta, r, first, second = self._pairs()
            ratio = np.divide(first, r, out=second.copy(), where=r>0)
            pair = (q[i]*q[j]*ratio)[:, None]*delta
            np.add.at(gradient, a, pair)
            np.add.at(gradient, b, -pair)
        if compute_gradient and getattr(self.partition.electrostatic_profile, 'geometry_dependent', False):
            i,j,correction = self.partition.electrostatic_profile.correction_jet(
                self.coordinates,self.partition,gradient=True)
            gradient += (.5*(correction*(q[i]*q[j])).sum()).a.reshape(self.coordinates.shape)
        return SimpleNamespace(energy_hartree=0.5*float(q@potential),
                               potential=potential, gradient_hartree_per_bohr=gradient)

    def potential_direction(self, charges, charge_direction, direction):
        q = np.asarray(charges)
        i, j, a, b, delta, r, first, second = self._pairs()
        ratio = np.divide(first, r, out=second.copy(), where=r>0)
        dot = ratio*np.einsum('ij,ij->i', delta, np.asarray(direction)[a]-np.asarray(direction)[b])
        result = np.asarray(self) @ charge_direction
        np.add.at(result, i, dot*q[j])
        np.add.at(result, j, dot*q[i])
        if getattr(self.partition.electrostatic_profile, 'geometry_dependent', False):
            i,j,correction = self.partition.electrostatic_profile.correction_jet(
                self.coordinates,self.partition,direction=direction)
            np.add.at(result,i,correction.b*q[j])
        return result

    def hessian_vector_product(self, charges, direction):
        q = np.asarray(charges)
        i, j, a, b, delta, r, first, second = self._pairs()
        relative = np.asarray(direction)[a]-np.asarray(direction)[b]
        ratio = np.divide(first, r, out=second.copy(), where=r>0)
        radial = np.divide(second-ratio, r*r, out=np.zeros_like(r), where=r>0)
        pair = (q[i]*q[j])[:, None] * (
            ratio[:, None]*relative
            + (radial*np.einsum('ij,ij->i',delta,relative))[:, None]*delta)
        result = np.zeros_like(self.coordinates)
        np.add.at(result, a, pair)
        np.add.at(result, b, -pair)
        if getattr(self.partition.electrostatic_profile, 'geometry_dependent', False):
            i,j,correction = self.partition.electrostatic_profile.correction_jet(
                self.coordinates,self.partition,direction=direction,gradient=True)
            result += (.5*(correction*(q[i]*q[j])).sum()).ab.reshape(self.coordinates.shape)
        return result


def wrap_scc_interaction(matrix, coordinates, partition):
    profile = partition.electrostatic_profile
    if profile is None or profile.is_identity:
        return matrix
    return GaussianSCCInteraction(matrix, coordinates, partition)
