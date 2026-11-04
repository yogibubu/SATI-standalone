"""
Centralized numerical parameters for continuous molecular descriptors.
All parameters are collected here to allow easy optimization.
"""

# ------------------------------------------------------------
# Continuous Coordination Number (CNA)
# ------------------------------------------------------------
#
# The continuous coordination number is computed as:
#
#   g = 0.5 * [ 1 - erf( CNA_ALPHA * ( R / Reff - 1 ) ) ]
#
# where the effective reference distance is:
#
#   Reff = CNA_SCALE * ( Rcov_i + Rcov_j )
#
# The scaling factor CNA_SCALE follows the Grimme philosophy
# (4/3 in early works) and ensures that
# standard covalent bonds give g ≈ 1.
#
# ------------------------------------------------------------

CNA_ALPHA = 8.0
CNA_SCALE = 4.0 / 3.0

# ------------------------------------------------------------
# Bond order parameters (normalized Pauling form)
# ------------------------------------------------------------
#
# The bond order used in atomic_synthons is:
#
#   BO = exp( - ( R / (Rcov_i + Rcov_j) - 1 ) / lambda )
#
# The decay parameter lambda is dimensionless and related to the
# classical Pauling decay length lambda_P (in Å) through:
#
#   lambda = lambda_P / (Rcov_i + Rcov_j)
#
# To reproduce the Pauling bond order for a C–C bond, we adopt:
#
#   Rcov(C)  = 0.75 Å
#   lambda_P = 0.30 Å   (strong bonds, BO ≳ 1)
#
# giving:
#
#   lambda_strong = 0.30 / (2 * 0.75) = 0.20
#
# It is known that for weak bonds and transition states (BO < 1)
# a larger Pauling decay length provides a more realistic decay.
# We therefore also define:
#
#   lambda_P_weak = 0.60 Å
#   lambda_weak   = 0.60 / (2 * 0.75) = 0.40
#
# The use of these two limits is deferred to atomic_synthons.
# ------------------------------------------------------------

BO_LAMBDA_STRONG = 0.20
BO_LAMBDA_WEAK = 0.40


# Smooth switching parameter for bond-order decay
# Controls the width of the strong-to-weak bond transition
# Typical values: 15–30; default chosen to give a crossover over
# ~0.05–0.1 Å for C–C bonds
ALPHA_LAMBDA = 20.0

# Width of the C2 ramps used to resolve a continuous multiplicity order into
# sigma, first-pi and second-pi (pi-pi) occupancies.  Outside the short ramp
# immediately above integer order 1 or 2, the decomposition is exactly the
# usual ordinal one and its components sum identically to the total BO.
BO_COMPONENT_RAMP_WIDTH = 0.10

# Geometric Pauling bond order used when Mayer data are unavailable:
#
#   M_ij = exp[(Rcov_i + Rcov_j - R_ij) / BO_PAULING_DECAY_ANGSTROM]
#
# Unlike the coordination-normalized topological edge index, this quantity
# deliberately retains single/double/triple-bond radial information and can
# therefore be resolved into sigma, first-pi and second-pi occupancies.
BO_PAULING_DECAY_ANGSTROM = 0.30


# Minimum bond order used in atomic descriptors
# (robustness for TS and weak bonds)
BO_MIN_DESC = 1.0


# ------------------------------------------------------------
# EAN normalization
# ------------------------------------------------------------

EAN_SIGMA = 1.0

# The production metric uses the four-component continuous synthon core
# (q, C, D, S) and a sign-constrained first principal component fitted to the
# C/N/O CM5/Mayer population in LCB25.  The
# model is stored in data/ean_pca_lcb25.json; the constants below keep the
# hot path free of file I/O and are checked against that artifact in tests.
# The bounded amplitude leaves headroom for environments outside LCB25 while
# retaining the element-preserving limits Z-1/2 and Z+1/2.  The arbitrary
# The constrained direction requires q <= -0.05 and C >= +0.05: positive
# charge lowers Zeff, while increasing compactness has the opposite loading.
EAN_RAW_MAX = 4.0
EAN_RADIUS_MAX = 2.0
# Historical weighted parameters are retained for audit/reproduction only.
EAN_WEIGHT_COVALENCY = 0.10
EAN_WEIGHT_DELOCALIZATION = 0.20
EAN_WEIGHT_STRAIN = 0.30
EAN_WEIGHT_CHARGE = 0.40
EAN_DELOCALIZATION_SCALE = 5.0
EAN_CHARGE_SCALE = 0.12

EAN_PCA_MEAN = (
    -0.13734891022163218,
    0.4824380525481825,
    1.716369402396528,
    0.1812098962371966,
)
EAN_PCA_SCALE = (
    0.1669606535412635,
    0.01620347815126661,
    0.6780359588282264,
    0.1600734214110363,
)
EAN_PCA_PC1 = (
    -0.04999999999999991,
    0.6108907966204574,
    0.5983707522244285,
    -0.5160086021441839,
)
EAN_PCA_EIGENVALUE = 2.157597772933552
EAN_PCA_UNCONSTRAINED_EIGENVALUE = 2.294834404174794
EAN_PCA_VARIANCE_LOSS = 0.05980241144702181
EAN_PCA_SCORE_CENTER = 0.5418505456328849
EAN_PCA_SCORE_SCALE = 2.1860288719985888
EAN_PCA_AMPLITUDE = 0.50


# ------------------------------------------------------------
# Reference angular sums for strain (degrees)
# One reference geometry per coordination number
# ------------------------------------------------------------

REF_ANGLE_SUM = {
    2: 180.0,  # linear
    3: 360.0,  # trigonal planar
    4: 6 * 109.47122,  # tetrahedral (Platonic)
    5: 720.0,  # trigonal bipyramidal (approx., non-Platonic)
    6: 12 * 90.0,  # octahedral (Platonic)
    7: 1260.0,  # capped octahedron (approx., non-Platonic)
    8: 24 * 90.0,  # cubic (Platonic)
    12: 30 * 63.43495,  # icosahedral (Platonic)
}
