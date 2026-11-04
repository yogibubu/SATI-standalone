"""Add an archived Hermite residual to a frozen periodic scalar surface."""
from dataclasses import dataclass
from .periodic_shepard import PeriodicShepardHermiteField


@dataclass(frozen=True)
class RelaxedQuadraticBathSurface:
    """Minimum of one declared frozen scalar plus its quadratic bath.

    Keeping the frozen scalar explicit permits direct checks against acquired
    frozen-geometry energies, rather than energies reconstructed with a second
    independently fitted relaxation model.
    """
    frozen_surface: object
    bath: object
    baseline: object

    def evaluate(self, points, derivative=None, *, allow_extrapolation=False):
        import numpy as np
        q = np.asarray(points, float)
        order = np.zeros(q.shape[1], int) if derivative is None else np.asarray(derivative, int)
        if order.shape != (q.shape[1],) or np.any(order < 0) or order.sum() > 2:
            raise ValueError('Relaxed scalar supports derivatives through order two')
        degree = int(order.sum())
        fields = self.bath.evaluate(q, derivatives=degree > 0, second=degree == 2)
        if degree == 0:
            lowering = fields['energy_lowering']
        elif degree == 1:
            lowering = fields['energy_lowering_first'][:, np.flatnonzero(order)[0]]
        else:
            i, j = np.repeat(np.arange(q.shape[1]), order)
            lowering = fields['energy_lowering_second'][:, i, j]
        return self.frozen_surface.evaluate(q, order, allow_extrapolation=allow_extrapolation) + lowering

    def to_dict(self):
        return dict(schema='matrix.dvr.relaxed_quadratic_bath_surface.v1',
            frozen_surface=self.frozen_surface.to_dict(), bath=self.bath.to_dict(),
            baseline=self.baseline.to_dict())

    @classmethod
    def from_dict(cls, data):
        from .periodic_hermite import periodic_surface_from_dict, PeriodicHermiteSurface
        from .periodic_fields import PeriodicQuadraticBath
        if data.get('schema') != 'matrix.dvr.relaxed_quadratic_bath_surface.v1':
            raise ValueError('Unsupported relaxed quadratic scalar')
        return cls(periodic_surface_from_dict(data['frozen_surface']),
            PeriodicQuadraticBath.from_dict(data['bath']),
            PeriodicHermiteSurface.from_dict(data['baseline']))


@dataclass(frozen=True)
class HermiteCorrectedPeriodicSurface:
    base: object
    correction: PeriodicShepardHermiteField

    @property
    def baseline(self):
        return self.base.baseline

    def evaluate(self,points,derivative=None,*,allow_extrapolation=False):
        return (self.base.evaluate(points,derivative,allow_extrapolation=allow_extrapolation)
                +self.correction.evaluate(points,derivative)[:,0])

    def to_dict(self):
        return dict(schema='matrix.dvr.hermite_corrected_periodic_surface.v1',
                    base=self.base.to_dict(),correction=self.correction.to_dict())

    @classmethod
    def from_dict(cls,data):
        from .periodic_hermite import periodic_surface_from_dict
        from .response_bath import response_field_from_dict
        if data.get('schema')!='matrix.dvr.hermite_corrected_periodic_surface.v1':raise ValueError('Unsupported periodic scalar correction')
        return cls(periodic_surface_from_dict(data['base']),response_field_from_dict(data['correction']))
