
### Approved Cartesian templates

`write_gaussian_cartesian_template(template, output, coordinates_angstrom)`
changes only an explicit four-column Cartesian coordinate block. Routes,
resources, atom labels, ExtraOverlays and trailing sections are retained.
Multiple jobs, ambiguous geometry sections and in-place overwriting fail.
This is the route-preserving path for a bounded electronic derivative stencil;
it does not add Force, symmetry or convergence options.
