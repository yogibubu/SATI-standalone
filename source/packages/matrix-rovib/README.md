# matrix-rovib (LOCK)

`matrix-rovib` provides the LOCK rotational, torsional, tunnelling and
hyperfine solvers, fit contracts and reproducible artifacts. Use `matrix lock`
through the MATRIX command line, or launch LOCK Studio from Keymaker for the
graphical workflow.

## General spectral continuation

`continue_spectral_coupling(reference_energies, coupling, ...)` follows the
physical eigenstates of a fixed Hermitian Hamiltonian as its coupling is
switched on. It uses the shared projector transport and refines the coupling
step until selected labels meet an overlap threshold. Finite attempt and
minimum-step limits reject unresolved labels. An optional checkpoint callback
and restart state preserve completed work; callers own input identity and
compute-host policy. The API accepts real or complex operators, contains no
molecule-specific parameters, and does not acquire potential-energy points.
Check both basis size and initial-step refinement before using occupation
labels in a downstream observable. Energies within a mixed span remain the
physical eigenvalues, not diagonal entries of a rotated Hamiltonian.
