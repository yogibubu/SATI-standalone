# Grimme sTDA reference data

`grimme-stda-hardness.json` transcribes the 94-element `setrep` atomic-hardness
table distributed with Grimme's `stda` implementation.  The initial MATRIX
spin-flip profile also reproduces the `stda -xtb -sf` settings `ax = 0.36`,
`beta = 3.0`, and the SF hardness scale `1.4 ax`.

The upstream implementation is distributed under GNU LGPL-3.0-or-later.  The
scientific sources identified by the program are:

- S. Grimme, *J. Chem. Phys.* **2013**, 138, 244104.
- C. Bannwarth and S. Grimme, *Comput. Theor. Chem.* **2014**, 1040–1041,
  45–53.
- M. de Wergifosse and S. Grimme, *J. Phys. Chem. A* **2021**, 125,
  3841–3851.

These values define a reference-compatibility profile.  They do not imply that
the later MATRIX-specific parametrization has been fitted or qualified.
