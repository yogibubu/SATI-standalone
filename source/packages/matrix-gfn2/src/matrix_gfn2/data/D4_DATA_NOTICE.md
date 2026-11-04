# D4 reference data notice

The numerical reference-system data in `d4-reference.json` and elemental data
in `d4-elements.json` originate from the DFT-D4 project and the publications
of Caldeweyher and co-workers. The migration snapshot was taken from DFT-D4
3.7.0 (`reference.inc`, `zeff.f90`, `hardness.f90`, and `r4r2.f90`), licensed
under LGPL-3.0-or-later. MATRIX contains an independent Python loader and model
implementation; it does not link to or import DFT-D4 at runtime. Retain this
notice and the LGPL attribution when redistributing these data.
