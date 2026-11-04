"""Elementary ORACLE face/contact operators with analytic local Jacobians.

Faces are ordered polygons. Their plane passes through the centroid, with
normal given by the oriented polygon area vector. Reversing the order
reverses the normal. No plane fitting or eigenvector differentiation is used.
"""
import numpy as np

FACE_OPERATORS = {
    "signed_face_flap": "FACE_FLAP",
    "center_face_distance": "CENTER_FACE_DIST",
    "face_direction_angle": "FACE_DIRECTION_ANGLE",
    "face_hinge": "FACE_HINGE",
    "contact_center_angle": "CONTACT_CENTER_ANGLE",
}


def face_from_smith(primitive):
    from .primitive_coordinates import Primitive
    return Primitive(next(k for k, v in FACE_OPERATORS.items() if v == primitive.function),
                     tuple(a-1 for a in primitive.atoms), ref=tuple(a-1 for a in primitive.ref_atoms),
                     frame_atoms=tuple(a-1 for a in primitive.frame_atoms))


def compiled_face_value_gradient(primitive, coordinates):
    from .primitive_coordinates import compiled_smith_primitive_plan_csr
    integer = lambda v: np.asarray(v, dtype=np.intp)
    arrays = []
    for atoms in (primitive.atoms, primitive.ref, primitive.frame_atoms, ()):
        arrays.extend((integer(atoms), integer([0, len(atoms)])))
    code = list(FACE_OPERATORS).index(primitive.kind) + 18
    raw = compiled_smith_primitive_plan_csr(
        coordinates, coordinates, integer([code]), integer([0]), *arrays,
        integer([]), integer([]).reshape(0,4), np.array([],dtype=float), integer([0,0]))
    if raw is None:
        # Wheels built before the face-op table was added can still expose the
        # generic CSR entry point but return ``None`` for these opcodes.  Keep
        # the scientific path total and deterministic by using the exact
        # analytic implementation below; this is a compatibility fallback,
        # not a numerical approximation.
        return face_value_gradient(primitive, coordinates)
    values, data, indices, _ = raw
    return float(values[0]), tuple((int(i), float(v)) for i,v in zip(indices,data))


def face_value_gradient(primitive, coordinates):
    support = sorted(set(primitive.atoms + primitive.ref + primitive.frame_atoms))
    xyz = np.asarray(coordinates, dtype=float)
    if not support or min(support) < 0 or max(support) >= len(xyz):
        raise ValueError("invalid face/contact support")
    size = 3 * len(support)
    points = {atom: (xyz[atom], np.eye(size)[3*i:3*i+3]) for i, atom in enumerate(support)}

    def sub(a, b): return a[0]-b[0], a[1]-b[1]
    def dot(a, b): return np.dot(a[0], b[0]), a[0] @ b[1] + b[0] @ a[1]
    def cross(a, b):
        return np.cross(a[0], b[0]), (np.cross(a[1].T, b[0]) + np.cross(a[0], b[1].T)).T
    def unit(a):
        length = np.linalg.norm(a[0])
        if length < 1.e-12:
            raise FloatingPointError("singular face normal, edge or direction")
        v = a[0]/length
        return v, (np.eye(3)-np.outer(v, v)) @ a[1]/length
    def center(atoms):
        if not atoms or len(set(atoms)) != len(atoms):
            raise ValueError("centers require nonempty unique atom sets")
        return tuple(np.mean([points[a][k] for a in atoms], axis=0) for k in (0, 1))
    def normal(atoms):
        if len(atoms) < 3:
            raise ValueError("a face requires at least three ordered atoms")
        c = center(atoms)
        pairs = [cross(sub(points[a], c), sub(points[b], c)) for a, b in zip(atoms, atoms[1:]+atoms[:1])]
        return unit(tuple(sum(p[k] for p in pairs) for k in (0, 1)))
    def angle(a, b):
        cosine, derivative = dot(unit(a), unit(b))
        sine = np.sqrt(max(0., 1.-cosine*cosine))
        if sine < 1.e-10:
            raise FloatingPointError("collinear angular chart; use transverse components")
        return np.arccos(np.clip(cosine, -1., 1.)), -derivative/sine

    kind = primitive.kind
    delta = sub(center(primitive.atoms), center(primitive.ref))
    if kind == "signed_face_flap":
        value, gradient = dot(delta, normal(primitive.ref))
    elif kind == "center_face_distance":
        value = np.linalg.norm(delta[0])
        gradient = unit(delta)[0] @ delta[1]
    elif kind == "face_direction_angle":
        value, gradient = angle(normal(primitive.ref), delta)
    elif kind == "contact_center_angle":
        origin = center(primitive.frame_atoms)
        value, gradient = angle(sub(center(primitive.atoms), origin), sub(center(primitive.ref), origin))
    elif kind == "face_hinge":
        edge = primitive.frame_atoms
        if len(edge) != 2 or not set(edge) <= set(primitive.atoms) & set(primitive.ref):
            raise ValueError("hinge requires an oriented edge shared by both faces")
        n1, n2 = normal(primitive.atoms), normal(primitive.ref)
        x, dx = dot(n1, n2)
        y, dy = dot(unit(sub(points[edge[1]], points[edge[0]])), cross(n1, n2))
        if x*x+y*y < 1.e-20:
            raise FloatingPointError("singular hinge chart")
        value, gradient = np.arctan2(y, x), (x*dy-y*dx)/(x*x+y*y)
    else:
        raise ValueError(f"unknown face operator {kind}")
    entries = tuple((3*atom+axis, float(gradient[3*i+axis]))
                    for i, atom in enumerate(support) for axis in range(3))
    return float(value), entries
