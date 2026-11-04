"""Complete a native scalar field on a measured fixed-LAM stationary section.

The existing full Cartesian quadratic jet is retained. Unobserved mixed LAM
terms retain the resident functional closure; no new source points are implied.
"""
import numpy as np
from .native_coordinate_quartic import NativeCoordinateQuartic
from .native_variable_quartic import _contract


class ProjectedNativeKernel:
    def __init__(self,kernel,projection):
        self.base=kernel;self.projection=np.asarray(projection,float)

    def value(self,xyz):
        u,j,h=self.base.value(xyz);p=self.projection
        if p.ndim!=2 or p.shape[1]!=len(u) or not np.all(np.isfinite(p)):
            raise ValueError('finite projection matching the native kernel required')
        return p@u,j@p.T,np.einsum('ija,ba->ijb',h,p)


def complete_conditional_fourjet(resident,kernel,realization,projection,
                                resident_cubic,resident_quartic,target_cubic,target_quartic):
    """Match a conditional stationary C3/Q4 without changing any full F2.

    Potential tensors are in hartree and use the supplied realization's
    coordinates. Projection maps full native variables to independent local
    variables with a nonsingular Jacobian on that realization. The general
    stationary chain rule retains its induced mixed radial/angular terms.
    """
    p=np.asarray(projection,float);n=len(resident.quadratic)
    if p.ndim!=2 or p.shape[1]!=n or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError('finite aligned conditional native projection required')
    m=len(p)
    if m>n or np.linalg.matrix_rank(p)!=m:raise ValueError('independent conditional native projection required')
    for value,degree in ((resident_cubic,3),(target_cubic,3),(resident_quartic,4),(target_quartic,4)):
        value=np.asarray(value)
        if value.shape!=(m,)*degree or not np.all(np.isfinite(value)):
            raise ValueError('finite aligned conditional potential tensors required')
    dc=np.asarray(target_cubic)-np.asarray(resident_cubic)
    dq=np.asarray(target_quartic)-np.asarray(resident_quartic)
    correction=NativeCoordinateQuartic.from_geometry_realization(
        np.zeros((m,m)),dc,dq,ProjectedNativeKernel(kernel,p),realization)
    return NativeCoordinateQuartic(resident.quadratic.copy(),
        resident.cubic+_contract(correction.cubic,p),
        resident.quartic+_contract(correction.quartic,p))

