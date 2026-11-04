"""Immutable, owner-neutral sampled molecular surfaces (standard library only).

Arrays use C order, explicit units and a declared Cartesian frame. This is an
exchange contract, not a field fit, a kinetic model or a physical certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite, prod
from pathlib import Path

from .atomic_io import atomic_write_text

MOLECULAR_SURFACE_SCHEMA = 'matrix.core.molecular_surface.v1'


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError('surface numbers must be finite and real; booleans are not numbers')
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'expected exact surface fields: {sorted(names)}')


@dataclass(frozen=True)
class SurfaceTensor:
    """Dense finite sampled tensor, flattened in C order, with explicit units."""
    shape: tuple[int, ...]
    values: tuple[float, ...]
    unit: str
    frame: str

    def __post_init__(self):
        shape = tuple(_integer(n, 'tensor dimension', 1) for n in self.shape)
        values = tuple(_number(v) for v in self.values)
        if not shape or prod(shape) != len(values):
            raise ValueError('surface tensor shape differs from its value count')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'values', values)
        _text(self.unit, 'tensor unit'); _text(self.frame, 'tensor frame')

    @classmethod
    def from_nested(cls, values, *, unit, frame):
        """Accept rectangular lists or array-like inputs without importing NumPy."""
        if hasattr(values, 'tolist'):
            values = values.tolist()
        def flatten(node):
            if isinstance(node, (list, tuple)):
                if not node:
                    raise ValueError('empty surface tensor axis')
                rows = [flatten(v) for v in node]
                if any(shape != rows[0][0] for shape, _ in rows):
                    raise ValueError('ragged surface tensor')
                return (len(rows),) + rows[0][0], tuple(v for _, row in rows for v in row)
            return (), (_number(node),)
        shape, data = flatten(values)
        return cls(shape, data, unit, frame)

    def to_dict(self):
        return dict(shape=list(self.shape), values=list(self.values), unit=self.unit, frame=self.frame)

    @classmethod
    def from_dict(cls, value):
        _keys(value, ('shape', 'values', 'unit', 'frame'))
        return cls(tuple(value['shape']), tuple(value['values']), value['unit'], value['frame'])


@dataclass(frozen=True)
class SurfaceCoordinate:
    identifier: str
    values: tuple[float, ...]
    unit: str
    period: float | None = None
    frequency_cm: float | None = None

    def __post_init__(self):
        _text(self.identifier, 'coordinate identifier')
        values = tuple(_number(v) for v in self.values)
        if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError('surface axes must be strictly increasing with at least two nodes')
        if self.unit not in {'radian', 'angstrom', 'dimensionless'}:
            raise ValueError('coordinate unit must be radian, angstrom or dimensionless')
        object.__setattr__(self, 'values', values)
        if self.period is not None:
            period = _number(self.period)
            if period <= values[-1] - values[0]:
                raise ValueError('periodic surface axes must exclude their duplicate endpoint')
            object.__setattr__(self, 'period', period)
        if self.frequency_cm is not None:
            frequency = _number(self.frequency_cm)
            if frequency <= 0 or self.unit != 'dimensionless' or self.period is not None:
                raise ValueError('modal frequencies require positive, nonperiodic dimensionless axes')
            object.__setattr__(self, 'frequency_cm', frequency)

    def to_dict(self):
        return dict(identifier=self.identifier, values=list(self.values), unit=self.unit,
                    period=self.period, frequency_cm=self.frequency_cm)

    @classmethod
    def from_dict(cls, value):
        _keys(value, ('identifier', 'values', 'unit', 'period', 'frequency_cm'))
        return cls(value['identifier'], tuple(value['values']), value['unit'], value['period'], value['frequency_cm'])


@dataclass(frozen=True)
class ElectronicSurface:
    """One explicitly identified electronic state on the shared geometry grid."""
    identifier: str
    energies: SurfaceTensor
    gradients: SurfaceTensor | None = None
    hessians: SurfaceTensor | None = None
    properties: tuple[tuple[str, SurfaceTensor], ...] = ()

    def __post_init__(self):
        _text(self.identifier, 'electronic state identifier')
        properties = tuple((name, tensor) for name, tensor in self.properties)
        if len({name for name, _ in properties}) != len(properties):
            raise ValueError('duplicate surface property')
        for name, tensor in properties:
            _text(name, 'property name')
            if name in {'energies', 'gradients', 'hessians', 'geometries'}:
                raise ValueError('property names cannot shadow structural surface channels')
            if not isinstance(tensor, SurfaceTensor):
                raise ValueError('properties must be SurfaceTensor instances')
        object.__setattr__(self, 'properties', properties)

    def property(self, name):
        try:
            return dict(self.properties)[name]
        except KeyError as exc:
            raise KeyError(f'property {name!r} absent in state {self.identifier!r}') from exc

    def to_dict(self):
        return dict(identifier=self.identifier, energies=self.energies.to_dict(),
                    gradients=self.gradients.to_dict() if self.gradients is not None else None,
                    hessians=self.hessians.to_dict() if self.hessians is not None else None,
                    properties={k: v.to_dict() for k, v in self.properties})

    @classmethod
    def from_dict(cls, value):
        _keys(value, ('identifier', 'energies', 'gradients', 'hessians', 'properties'))
        return cls(value['identifier'], SurfaceTensor.from_dict(value['energies']),
                   SurfaceTensor.from_dict(value['gradients']) if value['gradients'] is not None else None,
                   SurfaceTensor.from_dict(value['hessians']) if value['hessians'] is not None else None,
                   tuple((k, SurfaceTensor.from_dict(v)) for k, v in value['properties'].items()))


@dataclass(frozen=True)
class MolecularSurface:
    """Shared immutable geometries, E/G/H and properties, without a solver."""
    identifier: str
    atom_ids: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    masses_amu: tuple[float, ...]
    coordinates: tuple[SurfaceCoordinate, ...]
    geometries: SurfaceTensor
    states: tuple[ElectronicSurface, ...]
    cartesian_frame: str
    energy_reference: str
    provenance_json: str
    representation: str = 'INTERNAL_PATH'
    charge: int = 0
    reference_geometry: SurfaceTensor | None = None
    modal_displacements: SurfaceTensor | None = None

    def __post_init__(self):
        for name in ('identifier', 'cartesian_frame', 'energy_reference'):
            _text(getattr(self, name), name)
        for name in ('atom_ids', 'atomic_numbers', 'masses_amu', 'coordinates', 'states'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.atom_ids)
        if not n or len(set(self.atom_ids)) != n:
            raise ValueError('nonempty, unique atom IDs in frozen atom order required')
        for atom in self.atom_ids:
            _text(atom, 'atom ID')
        if len(self.atomic_numbers) != n or len(self.masses_amu) != n:
            raise ValueError('atom, element and isotope mass counts differ')
        for z in self.atomic_numbers:
            if _integer(z, 'atomic number', 1) > 118:
                raise ValueError('unsupported atomic number')
        masses = tuple(_number(m) for m in self.masses_amu)
        if min(masses) <= 0:
            raise ValueError('isotope masses must be positive')
        object.__setattr__(self, 'masses_amu', masses)
        if isinstance(self.charge, bool) or not isinstance(self.charge, int):
            raise ValueError('charge must be an integer')
        if (not self.coordinates or any(not isinstance(c, SurfaceCoordinate) for c in self.coordinates)
                or len({c.identifier for c in self.coordinates}) != len(self.coordinates)):
            raise ValueError('nonempty, uniquely identified coordinate axes required')
        if (not self.states or any(not isinstance(s, ElectronicSurface) for s in self.states)
                or len({s.identifier for s in self.states}) != len(self.states)):
            raise ValueError('nonempty, uniquely identified electronic states required')
        shape = self.grid_shape
        def require(tensor, wanted, unit, frame):
            if not isinstance(tensor, SurfaceTensor) or (tensor.shape, tensor.unit, tensor.frame) != (wanted, unit, frame):
                raise ValueError(f'surface tensor requires shape={wanted}, unit={unit}, frame={frame}')
        require(self.geometries, shape + (n, 3), 'angstrom', self.cartesian_frame)
        for state in self.states:
            require(state.energies, shape, 'hartree', 'scalar')
            if state.gradients is not None:
                require(state.gradients, shape + (n, 3), 'hartree/bohr', self.cartesian_frame)
            if state.hessians is not None:
                require(state.hessians, shape + (3*n, 3*n), 'hartree/bohr^2', self.cartesian_frame)
            for name, tensor in state.properties:
                if tensor.shape[:len(shape)] != shape or tensor.frame not in {'scalar', self.cartesian_frame}:
                    raise ValueError('property grid or Cartesian frame differs from the surface')
                if name == 'dipole':
                    if tensor.shape != shape + (3,) or tensor.unit not in {'e*bohr', 'debye'} or tensor.frame != self.cartesian_frame:
                        raise ValueError('dipoles require Cartesian vectors in e*bohr or debye')
                if name == 'polarizability':
                    require(tensor, shape + (3, 3), 'bohr^3', self.cartesian_frame)
        provenance = json.loads(self.provenance_json, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
        if not isinstance(provenance, dict):
            raise ValueError('surface provenance must be a JSON object')
        for key in ('producer', 'source_identifier'):
            _text(provenance.get(key), f'provenance {key}')
        object.__setattr__(self, 'provenance_json', _canonical(provenance))
        if self.representation == 'LINEAR_CARTESIAN_MODAL':
            require(self.reference_geometry, (n, 3), 'angstrom', self.cartesian_frame)
            require(self.modal_displacements, (len(shape), n, 3), 'angstrom', self.cartesian_frame)
            if any(c.frequency_cm is None for c in self.coordinates):
                raise ValueError('linear modal surfaces require every modal frequency')
        elif self.representation != 'INTERNAL_PATH' or self.reference_geometry is not None or self.modal_displacements is not None:
            raise ValueError('internal paths cannot declare a linear modal embedding')

    @property
    def grid_shape(self):
        return tuple(len(c.values) for c in self.coordinates)

    @property
    def provenance(self):
        return json.loads(self.provenance_json)

    def state(self, identifier):
        for state in self.states:
            if state.identifier == identifier:
                return state
        raise KeyError(f'electronic state {identifier!r} absent from shared surface')

    def to_dict(self):
        return dict(schema=MOLECULAR_SURFACE_SCHEMA, identifier=self.identifier,
                    atom_ids=list(self.atom_ids), atomic_numbers=list(self.atomic_numbers),
                    masses_amu=list(self.masses_amu), coordinates=[c.to_dict() for c in self.coordinates],
                    geometries=self.geometries.to_dict(), states=[s.to_dict() for s in self.states],
                    cartesian_frame=self.cartesian_frame, energy_reference=self.energy_reference,
                    provenance=self.provenance, representation=self.representation, charge=self.charge,
                    reference_geometry=self.reference_geometry.to_dict() if self.reference_geometry is not None else None,
                    modal_displacements=self.modal_displacements.to_dict() if self.modal_displacements is not None else None)

    @property
    def sha256(self):
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        fields = ('schema', 'identifier', 'atom_ids', 'atomic_numbers', 'masses_amu', 'coordinates',
                  'geometries', 'states', 'cartesian_frame', 'energy_reference', 'provenance',
                  'representation', 'charge', 'reference_geometry', 'modal_displacements')
        _keys(value, fields)
        if value['schema'] != MOLECULAR_SURFACE_SCHEMA:
            raise ValueError('unsupported molecular surface schema')
        return cls(value['identifier'], tuple(value['atom_ids']), tuple(value['atomic_numbers']),
                   tuple(value['masses_amu']), tuple(SurfaceCoordinate.from_dict(c) for c in value['coordinates']),
                   SurfaceTensor.from_dict(value['geometries']), tuple(ElectronicSurface.from_dict(s) for s in value['states']),
                   value['cartesian_frame'], value['energy_reference'], _canonical(value['provenance']),
                   value['representation'], value['charge'],
                   SurfaceTensor.from_dict(value['reference_geometry']) if value['reference_geometry'] is not None else None,
                   SurfaceTensor.from_dict(value['modal_displacements']) if value['modal_displacements'] is not None else None)


def _invalid_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'duplicate JSON key: {name}')
        result[name] = value
    return result


def write_molecular_surface(path, surface):
    """Atomically persist a typed surface with its complete content digest."""
    if not isinstance(surface, MolecularSurface):
        raise TypeError('a validated MolecularSurface is required')
    target = Path(path)
    atomic_write_text(target, _canonical(dict(sha256=surface.sha256, surface=surface.to_dict()))+'\n')
    return target


def read_molecular_surface(path, *, maximum_bytes=128*1024*1024):
    """Read one bounded, strict JSON artifact; never deserialize executable data."""
    _integer(maximum_bytes, 'maximum_bytes', 1)
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum_bytes+1)
    if len(raw) > maximum_bytes:
        raise ValueError('molecular surface exceeds the declared byte budget')
    envelope = json.loads(raw, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
    _keys(envelope, ('sha256', 'surface'))
    surface = MolecularSurface.from_dict(envelope['surface'])
    if envelope['sha256'] != surface.sha256:
        raise ValueError('molecular surface content digest mismatch')
    return surface


def create_molecular_surface(
    *, identifier, atom_ids, atomic_numbers, masses_amu, coordinates, geometries_angstrom,
    electronic_states, cartesian_frame, energy_reference, provenance,
    representation='INTERNAL_PATH', charge=0, reference_geometry_angstrom=None,
    modal_displacements_angstrom=None,
):
    """Package already computed arrays without fitting or launching calculations.

    ``electronic_states`` maps explicit state IDs to dictionaries containing
    ``energies_hartree``, optional ``gradients_hartree_per_bohr`` and
    ``hessians_hartree_per_bohr2``, and optional ``properties``. Properties map
    names to SurfaceTensor objects with their own declared units and frame.
    Axes are SurfaceCoordinate instances. Missing values are never filled.
    """
    def sequence(values):
        return tuple(values.tolist() if hasattr(values, 'tolist') else values)
    def tensor(values, unit, frame=cartesian_frame):
        return SurfaceTensor.from_nested(values, unit=unit, frame=frame)
    if not isinstance(electronic_states, dict) or not electronic_states:
        raise ValueError('explicit electronic-state data dictionaries are required')
    states = []
    for name, data in electronic_states.items():
        allowed = {'energies_hartree', 'gradients_hartree_per_bohr',
                   'hessians_hartree_per_bohr2', 'properties'}
        if not isinstance(data, dict) or set(data)-allowed or 'energies_hartree' not in data:
            raise ValueError('unknown or incomplete electronic-state arrays')
        gradient = data.get('gradients_hartree_per_bohr')
        hessian = data.get('hessians_hartree_per_bohr2')
        states.append(ElectronicSurface(
            name, tensor(data['energies_hartree'], 'hartree', 'scalar'),
            tensor(gradient, 'hartree/bohr') if gradient is not None else None,
            tensor(hessian, 'hartree/bohr^2') if hessian is not None else None,
            tuple(data.get('properties', {}).items()),
        ))
    return MolecularSurface(
        identifier, sequence(atom_ids), sequence(atomic_numbers), sequence(masses_amu),
        tuple(coordinates), tensor(geometries_angstrom, 'angstrom'), tuple(states),
        cartesian_frame, energy_reference, _canonical(provenance), representation, charge,
        tensor(reference_geometry_angstrom, 'angstrom') if reference_geometry_angstrom is not None else None,
        tensor(modal_displacements_angstrom, 'angstrom') if modal_displacements_angstrom is not None else None,
    )
