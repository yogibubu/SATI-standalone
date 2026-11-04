"""Real saddle derivatives from Gaussian's signed-frequency anharmonic tables.

The FI column uses complex mode normalization for unstable modes. Its modulus
is recovered with the sign of a physical real derivative, checked against the
printed physical columns and their rounding intervals. No imaginary-frequency
phase or mode-order convention is guessed.
"""
from dataclasses import dataclass
from decimal import Decimal
from itertools import permutations
from pathlib import Path
import hashlib
import re

import numpy as np

from .fchk import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class GaussianSaddleQFF:
    signed_frequencies_cm: np.ndarray
    cubic_cm: np.ndarray
    quartic_cm: np.ndarray
    diagnostics: dict


def _number(token):
    value = float(token.replace('D', 'E').replace('d', 'e'))
    if not np.isfinite(value):
        raise ValueError('nonfinite Gaussian saddle coefficient')
    return value


def _half_unit(token):
    return float(Decimal(10) ** Decimal(token.lower().replace('d', 'e')).as_tuple().exponent) / 2


def read_gaussian_saddle_qff(path, *, omitted_terms, quartic_coverage,
                             allow_physical_reconstruction=False):
    """Read a declared printed-threshold, semidiagonal quartic saddle model.

    ``omitted_terms='printed_threshold_model'`` explicitly closes unprinted
    cubic and repeated-index quartic entries at zero at the acquisition print
    threshold; it is not a claim that they vanish physically. Four-distinct
    quartics are omitted only under the declared ``semidiagonal_model``.
    Arrays follow Gaussian's explicit harmonic (H) numbering, not reversal.
    """
    from matrix_gf.harmonic import CM_PER_HARTREE, HESSIAN_EIGENVALUE_TO_CM

    if omitted_terms != 'printed_threshold_model':
        raise ValueError('explicit printed_threshold_model omission policy required')
    if quartic_coverage != 'semidiagonal_model':
        raise ValueError('printed Gaussian saddle input requires explicit semidiagonal_model')
    if not isinstance(allow_physical_reconstruction,(bool,np.bool_)):
        raise ValueError('explicit boolean physical reconstruction policy required')
    reconstructed=[]
    source = Path(path); text = source.read_text(); lines = text.splitlines()
    if (text.rfind('Normal termination of Gaussian') < 0
            or text.rfind('Error termination') > text.rfind('Normal termination of Gaussian')):
        raise ValueError('normally terminated Gaussian calculation required')
    def heading(label):
        found = [i for i, line in enumerate(lines) if label in line]
        if not found:
            raise ValueError('missing Gaussian saddle section: '+label)
        return found[-1]
    q = heading('QUADRATIC FORCE CONSTANTS IN NORMAL MODES')
    c = heading('CUBIC FORCE CONSTANTS IN NORMAL MODES')
    t = heading('QUARTIC FORCE CONSTANTS IN NORMAL MODES')
    frequencies = {}
    for line in lines[q:c]:
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit() and parts[0] == parts[1]:
            label = int(parts[0])
            if label in frequencies:
                raise ValueError('duplicate Gaussian quadratic coefficient')
            frequencies[label] = _number(parts[2])
    n = len(frequencies)
    if not n or set(frequencies) != set(range(1, n+1)):
        raise ValueError('incomplete Gaussian saddle quadratic table')
    wa = np.array([frequencies[i] for i in range(1, n+1)])
    if np.count_nonzero(wa < 0) != 1 or np.any(wa == 0):
        raise ValueError('exactly one unstable Gaussian mode required')
    mapping = {}
    start = heading('Input/Output information')
    for i in range(start, q):
        if re.match(r'^\s*\(H\)\s*\|', lines[i]) and i+1 < q:
            if not lines[i+1].strip().startswith('(A)'):
                raise ValueError('malformed Gaussian H/A equivalence table')
            h = [int(v) for v in re.findall(r'\d+', lines[i])]
            a = [int(v) for v in re.findall(r'\d+', lines[i+1])]
            if len(h) != len(a) or any(v in mapping for v in h):
                raise ValueError('duplicate or incomplete Gaussian H/A mapping')
            mapping.update(zip(h, a))
    if set(mapping) != set(range(1, n+1)) or set(mapping.values()) != set(mapping):
        raise ValueError('complete explicit Gaussian H/A equivalence table required')
    a_to_h = {a: h-1 for h, a in mapping.items()}
    wh = wa[[mapping[h]-1 for h in range(1, n+1)]]
    scales = HESSIAN_EIGENVALUE_TO_CM / np.sqrt(CM_PER_HARTREE*np.abs(wa))
    tensors = []; counts = {}; thresholds = {}; maximum_rounding_fraction = 0.
    for degree, begin, finish in ((3, c, t), (4, t, len(lines))):
        tensor = np.zeros((n,)*degree); seen = set(); imaginary = 0; ended = False
        for line in lines[begin:finish]:
            if f'Num. of {degree}' in line:
                m = re.search(r'larger than\s+(\S+):\s*(\d+)\s+over\s+(\d+)', line)
                if not m or int(m[2]) != len(seen):
                    raise ValueError('Gaussian saddle printed coefficient count mismatch')
                thresholds[str(degree)] = dict(printed_threshold=_number(m[1]), total_slots=int(m[3]))
                ended = True; break
            parts = line.split()
            if len(parts) < degree+3 or not all(v.isdigit() for v in parts[:degree]):
                continue
            labels = tuple(int(v) for v in parts[:degree])
            if any(v not in a_to_h for v in labels):
                raise ValueError('unknown Gaussian saddle mode index')
            key = tuple(sorted(a_to_h[v] for v in labels))
            if key in seen:
                raise ValueError('duplicate Gaussian saddle derivative')
            if degree == 4 and len(set(labels)) == 4:
                raise ValueError('four-distinct quartic conflicts with semidiagonal input declaration')
            physical_k,physical_K=map(_number,parts[degree+1:degree+3])
            fi_missing=bool(parts[degree]) and set(parts[degree])=={'*'}
            if fi_missing and not allow_physical_reconstruction:
                raise ValueError('unprinted reduced coefficient requires explicit physical-column reconstruction')
            fi=None if fi_missing else _number(parts[degree])
            if (physical_k and physical_K and np.sign(physical_k) != np.sign(physical_K)):
                raise ValueError('inconsistent real physical derivative signs')
            physical = physical_k or physical_K
            if not physical and (fi is None or fi):
                raise ValueError('physical sign unresolved at Gaussian print precision')
            scale = CM_PER_HARTREE * float(np.prod(scales[np.array(labels)-1]))
            # k is in attoJ / amu^(d/2) / angstrom^d; K in Eh / amu^(d/2) / bohr^d.
            conversion = BOHR_TO_ANGSTROM**degree / 4.3597447222071
            if fi_missing:
                intervals=[((value-_half_unit(token))*factor*scale,
                            (value+_half_unit(token))*factor*scale)
                           for token,value,factor in ((parts[degree+1],physical_k,conversion),
                                                      (parts[degree+2],physical_K,1.))]
                lower=max(row[0] for row in intervals);upper=min(row[1] for row in intervals)
                if lower>upper or lower<=0<=upper:
                    raise ValueError('physical columns do not resolve one signed real derivative')
                real=.5*(lower+upper);fi=abs(real)
                reconstructed.append(dict(degree=degree,indices_1based=[v+1 for v in key],
                    reconstructed_real_coefficient_cm=real,
                    physical_column_rounding_interval_cm=[lower,upper],
                    source_reduced_token=parts[degree]))
            else:
                for token, value, factor in ((parts[degree+1], physical_k, conversion),
                                              (parts[degree+2], physical_K, 1.)):
                    tolerance = (_half_unit(token)*factor*scale + _half_unit(parts[degree])
                                 + 2e-7*max(1., abs(fi)))
                    residual = abs(abs(value*factor*scale)-abs(fi))
                    if residual > tolerance:
                        raise ValueError('FI modulus disagrees with real physical derivative and rounding')
                    maximum_rounding_fraction = max(maximum_rounding_fraction, residual/tolerance)
            im = 'Im.Val.' in line
            if im != (sum(frequencies[v] < 0 for v in labels) % 2 == 1):
                raise ValueError('imaginary-value flag disagrees with unstable-mode parity')
            real = float(np.copysign(abs(fi), physical)) if fi else 0.
            for permuted in set(permutations(key)):
                tensor[permuted] = real
            seen.add(key); imaginary += im
        if not ended or not seen:
            raise ValueError('incomplete Gaussian saddle derivative table')
        counts[str(degree)] = dict(parsed=len(seen), imaginary_flagged=imaginary)
        tensors.append(tensor)
    return GaussianSaddleQFF(wh, *tensors, dict(
        schema='matrix.gaussian.saddle_qff.v1', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        harmonic_to_anharmonic_1based=[mapping[h] for h in range(1,n+1)],
        coordinate_convention='real dimensionless q scaled with absolute harmonic frequency; signed quadratic retained',
        conversion='FI modulus with independently checked real physical k/K sign and rounding intervals',
        omitted_terms=omitted_terms, quartic_coverage=quartic_coverage,
        counts=counts, source_print_thresholds=thresholds,
        maximum_physical_column_rounding_fraction=maximum_rounding_fraction,
        physical_reconstruction_allowed=allow_physical_reconstruction,
        reconstructed_reduced_coefficients=reconstructed,
    ))
