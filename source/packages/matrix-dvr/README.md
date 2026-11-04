# Curvilinear nuclear integration additions

`matrix_dvr.curvilinear_sine.CurvilinearSineDVR` is a matrix-free sine DVR
operator for a supplied complete positive kinetic tensor, potential and
physical volume gradient. Input coordinates retain their physical units;
kinetic coefficients include the conversion to cm^-1. For the flat-measure
wavefunction, it contracts

`1/2 (d - dlog(rho)/2)^dagger G (d - dlog(rho)/2) + V`.

All off-diagonal kinetic terms are retained. The existing sine Galerkin
values/derivatives are evaluated on an oversampled Gaussian rule; square
first-derivative collocation, which can introduce null modes, is avoided.
An orthogonal DST converts between the sine expansion and returned DVR nodal
coefficients. No dense Hamiltonian, electronic acquisition or potential fitting
is introduced by this operator. Grid, quadrature and domain convergence remain
separate from potential validation.

`matrix_dvr.triatomic_chart.valence_triatomic_metric` supplies r01, r02, theta102
for any nonlinear valence triatomic with arbitrary masses. It delegates to the
existing complete Cartesian/internal/rotation metric owner, inverts the full
metric before selecting its J=0 block, and returns the physical density
proportional to `r01^2*r02^2*sin(theta)`. It does not substitute a frozen Wilson
metric or omit the rotational contribution to the volume.

Four focused controls run on MAC-STUDIO check the positive odd-size sine
spectrum, radial volume connection, complete triatomic metric against the
independent Wilson formula, and tensor action against an independently
assembled full weak-form matrix. These are mathematical/operator checks;
they do not qualify an unfitted molecular PES.

Every molecular production DVR potential follows the existing KeyMaker
`dvr-sparse-analytic-fit.v1` finite-acquisition/analytic-fit protocol. The solver
never launches electronic jobs during grid or domain refinement.
