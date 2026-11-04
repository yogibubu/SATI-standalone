# functions_ring.py
# ============================================================
# Geometric utility functions for ring analysis
#
# This module contains pure geometric helpers used by Ring
# objects. No topology, no chemistry, no GNIC logic.
# ============================================================

import numpy as np


# ============================================================
# Plane fitting and planarity
# ============================================================


def best_fit_plane(xyz):
    """
    Compute the best-fit plane to a set of points.

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)
        Cartesian coordinates of points.

    Returns
    -------
    point : ndarray, shape (3,)
        Centroid of the points.
    normal : ndarray, shape (3,)
        Unit normal vector of the best-fit plane.
    """
    xyz = np.asarray(xyz)
    centroid = xyz.mean(axis=0)

    # SVD of covariance matrix
    u, s, vh = np.linalg.svd(xyz - centroid)
    normal = vh[-1]

    # Normalize
    normal /= np.linalg.norm(normal)

    return centroid, normal


def ring_planarity(xyz):
    """
    Quantify planarity as RMS distance from best-fit plane.

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)

    Returns
    -------
    float
        RMS distance from the plane.
    """
    centroid, normal = best_fit_plane(xyz)
    distances = np.dot(xyz - centroid, normal)
    return np.sqrt(np.mean(distances**2))


# ============================================================
# Cyclic topology helpers
# ============================================================


def cyclic_triplets(atoms):
    """
    Generate cyclic triplets (i,j,k) from an ordered ring.

    Parameters
    ----------
    atoms : list[int]

    Returns
    -------
    list of tuple
        (i,j,k) triplets.
    """
    n = len(atoms)
    triplets = []

    for i in range(n):
        i_prev = atoms[(i - 1) % n]
        j = atoms[i]
        i_next = atoms[(i + 1) % n]
        triplets.append((i_prev, j, i_next))

    return triplets


def cyclic_quartets(atoms):
    """
    Generate cyclic quartets (i,j,k,l) from an ordered ring.

    Parameters
    ----------
    atoms : list[int]

    Returns
    -------
    list of tuple
        (i,j,k,l) quartets.
    """
    n = len(atoms)
    quartets = []

    for i in range(n):
        i1 = atoms[i % n]
        i2 = atoms[(i + 1) % n]
        i3 = atoms[(i + 2) % n]
        i4 = atoms[(i + 3) % n]
        quartets.append((i1, i2, i3, i4))

    return quartets


# ============================================================
# Orientation helpers (optional, future use)
# ============================================================


def ring_normal(xyz):
    """
    Return unit normal vector to a ring.

    Parameters
    ----------
    xyz : ndarray, shape (N,3)

    Returns
    -------
    ndarray, shape (3,)
    """
    _, normal = best_fit_plane(xyz)
    return normal


def signed_distance_from_plane(xyz, plane_point, plane_normal):
    """
    Signed distances of points from a plane.

    Parameters
    ----------
    xyz : ndarray, shape (N,3)
    plane_point : ndarray, shape (3,)
    plane_normal : ndarray, shape (3,)

    Returns
    -------
    ndarray
        Signed distances.
    """
    return np.dot(xyz - plane_point, plane_normal)
