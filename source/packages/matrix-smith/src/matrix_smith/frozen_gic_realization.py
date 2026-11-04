"""Frozen displacement realization using SMITH's existing Eckart inverse.

This adapter supplies the analytic first/second Cartesian chain rule for a
caller-defined subspace. It performs geometry operations only, never PES calls.
"""
from pathlib import Path
import hashlib
import inspect
import numpy as np
from .eckart_chart import build_eckart_constrained_gic_chart
from .evaluation import build_gic_b_matrix
from .second_derivatives import analytic_primitive_hessian
from matrix_zaff.primitive_second_derivative import analytic_primitive_second_derivative


def _geometry_runtime_identity():
    """Bind cached chain rules to the analytic implementations that made them."""
    providers=(frozen_gic_realization,build_eckart_constrained_gic_chart,build_gic_b_matrix,
               analytic_primitive_hessian,analytic_primitive_second_derivative)
    sources=sorted({inspect.getsourcefile(provider) for provider in providers})
    return b''.join(hashlib.sha256(Path(source).read_bytes()).digest() for source in sources)


def frozen_gic_realization(definition,reference,masses,displacements,cache_directory):
    reference=np.asarray(reference,float);masses=np.asarray(masses,float);l=np.asarray(displacements,float)
    if (reference.ndim!=2 or reference.shape[1]!=3 or masses.shape!=(len(reference),)
            or l.ndim!=2 or l.shape[0]!=reference.size or np.any(masses<=0)
            or not all(np.all(np.isfinite(x)) for x in (reference,masses,l))):
        raise ValueError('aligned finite geometry, masses and displacement subspace required')
    inverse=build_eckart_constrained_gic_chart(definition,reference,masses,tolerance=2e-12)
    identifiers={p.identifier:i for i,p in enumerate(definition.primitives)}
    combinations=np.zeros((len(definition.gics),len(identifiers)))
    for i,g in enumerate(definition.gics):
        for key,value in g.coefficients:combinations[i,identifiers[key]]=value
    k=np.asarray(build_gic_b_matrix(definition,coordinates_angstrom=reference).rows)@l
    n=l.shape[1];cache=Path(cache_directory);cache.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(_geometry_runtime_identity()+repr(definition).encode()
        +reference.tobytes()+masses.tobytes()+l.tobytes()).digest()
    def point(q):
        q=np.asarray(q,float)
        if q.shape!=(n,) or not np.all(np.isfinite(q)):raise ValueError('aligned finite displacement required')
        path=cache/(hashlib.sha256(identity+q.tobytes()).hexdigest()+'.npz')
        if path.exists():
            with np.load(path) as d:return d['xyz'],d['first'],d['second']
        xyz=inverse.geometry_at_displacement_angstrom(k@q)
        bg=np.asarray(build_gic_b_matrix(definition,coordinates_angstrom=xyz).rows)
        primitive=[]
        for p in definition.primitives:
            if p.function in ('SOOP','SYCART'):h=analytic_primitive_hessian(p,xyz)
            else:
                h=np.zeros((xyz.size,xyz.size))
                for i,j,v in analytic_primitive_second_derivative(p,xyz,reference_coordinates=reference).entries:h[i,j]+=v
            primitive.append(h)
        hg=np.einsum('ap,pij->aij',combinations,np.asarray(primitive))
        system=np.vstack((bg,inverse.rigid_constraint_rows))
        first=np.linalg.solve(system,np.vstack((k,np.zeros((6,n)))))
        rhs=np.einsum('aij,ib,jc->abc',hg,first,first,optimize=True)
        second=-np.linalg.solve(system,np.vstack((rhs.reshape(len(k),-1),np.zeros((6,n*n))))).reshape(xyz.size,n,n)
        np.savez_compressed(path,xyz=xyz,first=first,second=second)
        return xyz,first,second
    return point
