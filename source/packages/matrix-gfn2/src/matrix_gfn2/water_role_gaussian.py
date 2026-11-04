"""Continuous donor/acceptor roles for oxygen-centered Gaussian water charges."""
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import numpy as np
from ._mixed_jet import MixedJet
from .gaussian_scc import GaussianSCCProfile, water_oxygen_gaussian_profile


def water_donor_weights(coordinates, waters, solute_acceptors, solute_donor_bonds):
    """Smooth contact balance without distance/angle cutoffs or fitted selectors.

    Every D-H...A contribution is [(1+cos(theta))/2]^2 / r_HA^6,
    where theta=0 is a linear donor-heavy-atom/H/acceptor arrangement.
    Returned values are water-donor fractions. Topology is supplied once;
    both water roles, including their derivatives, vary with geometry.
    """
    jet = isinstance(coordinates, MixedJet)
    zero = lambda: MixedJet.constant(0., coordinates.n) if jet else 0.
    def contact(d, h, a):
        dh = coordinates[h]-coordinates[d]
        ha = coordinates[a]-coordinates[h]
        h2 = (ha*ha).sum()
        cosine = (dh*ha).sum()/((dh*dh).sum()*h2)**.5
        return ((1+cosine)/2)**2/h2**3
    result = []
    for oxygen, h1, h2 in waters:
        if not solute_acceptors or not solute_donor_bonds:
            fraction = .5 if not solute_acceptors and not solute_donor_bonds else (
                0. if not solute_acceptors else 1.)
            result.append(zero()+fraction)
            continue
        donor, acceptor = zero(), zero()
        for hydrogen in (h1, h2):
            for site in solute_acceptors:
                donor = donor+contact(oxygen, hydrogen, site)
        for heavy, hydrogen in solute_donor_bonds:
            acceptor = acceptor+contact(heavy, hydrogen, oxygen)
        total = donor+acceptor
        value = total.v if jet else total
        if not np.isfinite(value) or value <= 0:
            raise ValueError("water role requires nonzero finite hydrogen-bond scores")
        result.append(donor/total)
    return result


@dataclass(frozen=True)
class WaterRoleGaussianProfile(GaussianSCCProfile):
    waters: tuple[tuple[int, int, int], ...] = ()
    solute_acceptors: tuple[int, ...] = ()
    solute_donor_bonds: tuple[tuple[int, int], ...] = ()
    sigma_donor_bohr: float = 1.
    sigma_acceptor_bohr: float = 1.

    def __post_init__(self):
        super().__post_init__()
        for value in (self.sigma_donor_bohr, self.sigma_acceptor_bohr):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("both role widths must be finite and positive")
        # Factory validates atomic composition; direct construction also checks
        # immutable topology and separation of water and solute sites.
        if self.interaction_region is None:
            raise ValueError("water role profiles require an explicit interaction region")
        raw_indices = [i for w in self.waters for i in w] + list(self.solute_acceptors) + [
            i for bond in self.solute_donor_bonds for i in bond]
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in raw_indices):
            raise ValueError("water role topology requires integer atom indices")
        waters = tuple(tuple(int(i) for i in w) for w in self.waters)
        acceptors = tuple(int(i) for i in self.solute_acceptors)
        donors = tuple(tuple(int(i) for i in bond) for bond in self.solute_donor_bonds)
        n = len(self.atom_widths_bohr)
        if any(len(w) != 3 or len(set(w)) != 3 for w in waters):
            raise ValueError("water topology requires distinct O/H/H triples")
        members = [i for w in waters for i in w]
        if len(members) != len(set(members)):
            raise ValueError("water fragments must be disjoint")
        if any(len(b) != 2 or b[0] == b[1] for b in donors):
            raise ValueError("solute donor topology requires heavy-atom/H pairs")
        if len(acceptors) != len(set(acceptors)) or len(donors) != len(set(donors)):
            raise ValueError("water role topology must not contain duplicate sites")
        sites = list(acceptors)+[i for bond in donors for i in bond]
        if any(i < 0 or i >= n for i in members+sites) or set(members).intersection(sites):
            raise ValueError("water and solute role sites must be valid and disjoint")
        if any(not self.interaction_region[i] for i in members):
            raise ValueError("water role atoms must lie inside the selected region")
        object.__setattr__(self, 'sigma_donor_bohr', float(self.sigma_donor_bohr))
        object.__setattr__(self, 'sigma_acceptor_bohr', float(self.sigma_acceptor_bohr))
        object.__setattr__(self, 'waters', waters)
        object.__setattr__(self, 'solute_acceptors', acceptors)
        object.__setattr__(self, 'solute_donor_bonds', donors)

    @property
    def geometry_dependent(self):
        return True

    @property
    def content_sha256(self):
        payload = dict(base=super().content_sha256, waters=self.waters,
                       solute_acceptors=self.solute_acceptors,
                       solute_donor_bonds=self.solute_donor_bonds,
                       sigma_donor=self.sigma_donor_bohr,
                       sigma_acceptor=self.sigma_acceptor_bohr,
                       selector='angular2-distance6-v1')
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _correction(self, coordinates, partition):
        atom = np.asarray(partition.shell_atom)
        mask, _ = self.shell_data(partition)
        i, j = np.where(mask)
        widths = MixedJet.constant(self.atom_widths_bohr, coordinates.n)
        donor = water_donor_weights(coordinates, self.waters,
                                   self.solute_acceptors, self.solute_donor_bonds)
        for water, fraction in zip(self.waters, donor):
            widths[water[0]] = self.sigma_acceptor_bohr + fraction*(
                self.sigma_donor_bohr-self.sigma_acceptor_bohr)
        distance = ((coordinates[atom[i]]-coordinates[atom[j]])**2).sum(axis=1)**.5
        sigma = (widths[atom[i]]**2+widths[atom[j]]**2)**.5
        hardness = np.asarray(partition.hardness_hartree)
        average = .5*(hardness[i]+hardness[j])
        native = (distance**2+average**-2)**-.5
        return i, j, self.weight*((distance/sigma).erf()/distance-native)

    def correction_jet(self, coordinates, partition, *, direction=None, gradient=False):
        x = np.asarray(coordinates)
        first = np.eye(x.size).reshape(x.size,*x.shape) if gradient else np.zeros((0,*x.shape))
        return self._correction(MixedJet(x,first,direction),partition)

    def modify_values(self, matrix, coordinates, partition):
        from .gaussian_scc import wrap_scc_interaction
        i,j,delta = self.correction_jet(coordinates,partition)
        out = np.array(matrix,copy=True);out[i,j] += delta.v
        return wrap_scc_interaction(out,coordinates,partition)

    def modify_jet(self, gamma, coordinates, partition):
        i,j,delta = self._correction(coordinates,partition)
        gamma[i,j] = gamma[i,j]+delta
        return gamma


def water_role_gaussian_profile(atoms, water_fragments, *, solute_acceptors,
                                solute_donor_bonds, width_scale_donor,
                                width_scale_acceptor):
    """Two O widths; H parameters/charges and intraregion pairs stay unchanged."""
    fragments = tuple(tuple(w) for w in water_fragments)
    donor = water_oxygen_gaussian_profile(atoms,fragments,width_scale=width_scale_donor)
    acceptor = water_oxygen_gaussian_profile(atoms,fragments,width_scale=width_scale_acceptor)
    waters = tuple((next(i for i in w if atoms[i]=='O'),
                    *(i for i in w if atoms[i]=='H')) for w in fragments)
    first_o = waters[0][0] if waters else None
    return WaterRoleGaussianProfile(
        donor.atom_widths_bohr, weight=1., interaction_region=donor.interaction_region,
        waters=waters, solute_acceptors=tuple(solute_acceptors),
        solute_donor_bonds=tuple(tuple(b) for b in solute_donor_bonds),
        sigma_donor_bohr=donor.atom_widths_bohr[first_o] if waters else 1.,
        sigma_acceptor_bohr=acceptor.atom_widths_bohr[first_o] if waters else 1.)


def water_hbond_gaussian_profile(atoms, water_fragments, *, solute_acceptors,
                                 solute_donor_bonds, width_scale):
    """One O-width scale with fixed donor/acceptor role weights 1 and 1/2.

    The factors define an effective Gaussian-width convention, rather than
    an energy ratio between an electron pair and a hydrogen atom. Geometry
    dependence and its analytic derivatives are inherited unchanged.
    """
    return water_role_gaussian_profile(
        atoms, water_fragments, solute_acceptors=solute_acceptors,
        solute_donor_bonds=solute_donor_bonds,
        width_scale_donor=width_scale, width_scale_acceptor=.5*width_scale)
