"""Periodic contact field plus the established reference-anchored residual."""
from dataclasses import dataclass
from .anchored_periodic_eg import AnchoredPeriodicEG


@dataclass(frozen=True)
class ContactCorrectedPeriodicEG:
    contact: object
    correction: AnchoredPeriodicEG

    def evaluate(self, angles, *, derivative=0):
        return (self.contact.evaluate(angles, derivative=derivative)
                + self.correction.evaluate(angles, derivative=derivative))

    def to_dict(self):
        return dict(schema='matrix.dvr.contact_corrected_periodic_eg.v1',
                    contact=self.contact.to_dict(), correction=self.correction.to_dict())

    @classmethod
    def from_dict(cls, data):
        from matrix_zaff.rigid_orbit_contact import RigidOrbitContactPotential
        if data.get('schema') != 'matrix.dvr.contact_corrected_periodic_eg.v1':
            raise ValueError('unsupported contact-corrected periodic potential')
        return cls(RigidOrbitContactPotential.from_dict(data['contact']),
                   AnchoredPeriodicEG.from_dict(data['correction']))
