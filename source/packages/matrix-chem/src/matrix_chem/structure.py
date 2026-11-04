from typing import List, Tuple, Optional

from .average_atomic_masses import atomic_mass
from .isotopes_table import get_default_isotope, get_isotope
from .topology.elements import atomic_number

Coord3 = Tuple[float, float, float]


class Structure:
    """
    Self-contained molecular structure.
    """

    def __init__(
        self,
        symbols: List[str],
        coords: List[Coord3],
        isotopes: Optional[List[Optional[int]]] = None,
    ):
        if len(symbols) != len(coords):
            raise ValueError("symbols and coords must have the same length")

        self.natoms = len(symbols)
        self.symbols = list(symbols)

        # coordinates
        self.coords = []
        for c in coords:
            if len(c) != 3:
                raise ValueError(f"Invalid coordinate (not length 3): {c}")
            self.coords.append(tuple(float(x) for x in c))

        # atomic numbers (from elements module)
        self.Z = []
        for sym in self.symbols:
            Z = atomic_number(sym)
            if Z is None:
                raise ValueError(f"Unknown atomic symbol: {sym}")
            self.Z.append(Z)

        # isotopes
        if isotopes is None:
            self.isotopes = [None] * self.natoms
        else:
            if len(isotopes) != self.natoms:
                raise ValueError("isotopes must have same length as symbols")
            self.isotopes = list(isotopes)

        # masses
        self.mass_average = []
        self.mass_isotope = []

        # Use the index so we can *store* the chosen default isotope mass number (A)
        # when isotopes were not explicitly provided.
        for i, (Z, iso) in enumerate(zip(self.Z, self.isotopes)):
            m_avg = atomic_mass(Z)
            self.mass_average.append(m_avg)

            if iso is None:
                try:
                    default_iso = get_default_isotope(Z)  # default must be first in isotopes_table
                    m_iso = default_iso.mass
                    # Persist chosen isotope A so it can be reported downstream
                    self.isotopes[i] = default_iso.A
                except Exception:
                    m_iso = m_avg
            else:
                # Explicit isotope A provided
                try:
                    isotope = get_isotope(Z, int(iso))
                    if isotope is None:
                        isotope = get_default_isotope(Z)
                    m_iso = isotope.mass
                except Exception:
                    m_iso = m_avg

            self.mass_isotope.append(m_iso)

        self.total_mass_average = sum(self.mass_average)
        self.total_mass_isotope = sum(self.mass_isotope)

        # ---------- IMMUTABILITY (STEP 1) ----------
        self.symbols = tuple(self.symbols)
        self.coords = tuple(self.coords)
        self.Z = tuple(self.Z)
        self.isotopes = tuple(self.isotopes)
        self.mass_average = tuple(self.mass_average)
        self.mass_isotope = tuple(self.mass_isotope)

    @classmethod
    def from_atoms_coords(
        cls,
        atoms: List[str],
        coords: List[Coord3],
        isotopes: Optional[List[Optional[int]]] = None,
    ):
        return cls(atoms, coords, isotopes=isotopes)

    def __repr__(self):
        return (
            f"<Structure natoms={self.natoms} "
            f"M_avg={self.total_mass_average:.6f} "
            f"M_iso={self.total_mass_isotope:.6f}>"
        )
