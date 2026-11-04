import numpy as np

from .structure import Structure


def center_of_mass(structure: Structure, isotopic: bool = True):
    """
    Compute the center of mass.

    Returns
    -------
    com : ndarray, shape (3,)
        Center of mass in Å
    """
    masses = structure.mass_isotope if isotopic else structure.mass_average
    mtot = sum(masses)

    com = np.zeros(3)
    for m, c in zip(masses, structure.coords):
        com += m * np.array(c)

    return com / mtot


def inertia_tensor(structure: Structure, isotopic: bool = True):
    """
    Compute the inertia tensor with respect to the center of mass.

    Units: amu Å^2
    """
    masses = structure.mass_isotope if isotopic else structure.mass_average

    coords = np.array(structure.coords)
    com = center_of_mass(structure, isotopic=isotopic)
    coords = coords - com

    I = np.zeros((3, 3))
    for m, (x, y, z) in zip(masses, coords):
        I[0, 0] += m * (y**2 + z**2)
        I[1, 1] += m * (x**2 + z**2)
        I[2, 2] += m * (x**2 + y**2)
        I[0, 1] -= m * x * y
        I[0, 2] -= m * x * z
        I[1, 2] -= m * y * z

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I


def principal_moments(structure: Structure, isotopic: bool = True):
    """
    Principal moments of inertia.

    Returns
    -------
    moments : tuple of float
        Principal moments in amu Å^2 (sorted ascending)
    """
    I = inertia_tensor(structure, isotopic=isotopic)
    vals, _ = np.linalg.eigh(I)
    return tuple(vals)
