"""Select observable native cubic products before redundant primitive fitting."""

from itertools import combinations_with_replacement, permutations
import numpy as np
from scipy.linalg import qr
from matrix_chem.primitive_coordinates import primitive_b_matrix
from .native_sparse_fit import native_zaff_candidate_powers
from .asymptotic_pic import AsymptoticPicFactor


def independent_native_cubic_catalog(primitives, reference_angstrom, *, relative_rank_cutoff=1e-8):
    """Pivot the third-order physical response, without QM or spectral targets.

    Native coordinates are normalized to unit first derivative with respect to
    their primitive at the reference. Rigid Cartesian directions are removed
    by the existing B matrix. QR acts on unique symmetric cubic components,
    with multiplicity weights preserving the full tensor norm. Products that
    are linearly dependent because of redundant primitives are not all fitted.
    Units are the native angstrom/radian convention, without column rescaling.
    """
    if not 0 < relative_rank_cutoff < 1:
        raise ValueError("invalid cubic rank cutoff")
    b = np.asarray(primitive_b_matrix(primitives, reference_angstrom))
    _, singular, vh = np.linalg.svd(b, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * relative_rank_cutoff))
    a = b @ vh[:rank].T
    powers = native_zaff_candidate_powers(
        [p.atoms for p in primitives],
        degrees=(3,),
        maximum_active_primitives=3,
        require_connected=False,
    )
    products = np.array(
        [tuple(i for i, count in enumerate(row) for _ in range(count)) for row in powers]
    )
    triples = np.array(list(combinations_with_replacement(range(rank), 3)))
    design = np.zeros((len(triples), len(products)), order="F")
    for perm in permutations(range(3)):
        design += (
            a[products[:, 0][None, :], triples[:, perm[0]][:, None]]
            * a[products[:, 1][None, :], triples[:, perm[1]][:, None]]
            * a[products[:, 2][None, :], triples[:, perm[2]][:, None]]
        )
    multiplicity = np.array([len(set(permutations(row))) for row in triples])
    design *= np.sqrt(multiplicity)[:, None]
    r, pivots = qr(design, mode="r", pivoting=True, overwrite_a=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    resolved = int(np.count_nonzero(diagonal > diagonal[0] * relative_rank_cutoff))
    selected = sorted(pivots[:resolved].tolist())
    factors = []
    for j in selected:
        factors.append(
            tuple(
                AsymptoticPicFactor(
                    i,
                    "inverse_distance"
                    if primitives[i].kind == "bond"
                    else "native"
                    if primitives[i].kind == "angle"
                    else "sin",
                    power,
                )
                for i, power in enumerate(powers[j])
                if power
            )
        )
    return tuple(factors), {
        "primitive_rank": rank,
        "cubic_response_rank": resolved,
        "candidate_count": len(powers),
        "maximum_cubic_rank": len(triples),
    }
