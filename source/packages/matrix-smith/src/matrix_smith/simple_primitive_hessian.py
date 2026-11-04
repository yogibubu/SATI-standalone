"""SH using exactly the xTB ddvopt primitive model, on any SONIC chart.

No extra out-of-plane/improper terms or primitive reparametrization are added.
Pseudo-bond distances are already included by ddvopt's all-atom pair loop.
"""


def build_simple_primitive_sh(path, *, coordinates_angstrom=None, s6=20.):
    # Compatibility facade only: primitive-Hessian construction belongs to ORACLE.
    from matrix_oracle import build_simple_hessian
    return build_simple_hessian(path,coordinates_angstrom=coordinates_angstrom,s6=s6)
