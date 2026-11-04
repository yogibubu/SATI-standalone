"""Portable official NNxTB encoder, without MACE/dxtb/TBLite runtime imports.

The graph is exported from the strictly loaded Code Ocean checkpoint. Only
the neural parameter predictor is compiled; electronic structure remains in
MATRIX. Coordinates entering this module are in Bohr.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

from .nn_xtb import (GFN2NNFormatError, GFN2NNHeadSpec, GFN2NNModelInfo,
                     NNXTBOverlay, _MATRIX_SHIFT_FACTORS)

MATRIX_ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
# tad-mctc CODATA conversion used by the archived reference. The native
# GFN2 table deliberately retains its original 27.21138505 convention.
OFFICIAL_EV_TO_HARTREE = 0.03674932217565494
ELEMENT_HEADS = ('3rd_scale', 'levels', 'slater', 'shpoly', 'kcn', 'gam', 'lgam',
                 'arad', 'rcov', 'gam3', 'zeff', 'arep', 'en', 'dkernel', 'qkernel',
                 'mprad', 'mpvcn')
ELEMENT_LENGTHS = (3, 3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
GLOBAL_HEADS = ('wexp', 'kpol', 'enscale', 'ss', 'pp', 'dd', 'sd', 'pd', 's6',
                's8', 'a1', 'a2', 's9', 's10', 'kexp', 'klight', 'gexp', 's', 'p',
                'd', 'dmp3', 'dmp5', 'shift', 'rmax')


_model_load_lock = Lock()


def _file_identity(path):
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


@lru_cache(maxsize=1)
def _verified_model(path, identity, expected_digest):
    import torch

    encoder_path = Path(path)
    if _file_identity(encoder_path) != identity:
        raise GFN2NNFormatError('Official encoder changed before loading')
    with encoder_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected_digest or _file_identity(encoder_path) != identity:
        raise GFN2NNFormatError('Official encoder checksum mismatch')
    model = torch.jit.load(str(encoder_path), map_location='cpu').eval()
    if _file_identity(encoder_path) != identity:
        raise GFN2NNFormatError('Official encoder changed while loading')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class OfficialNNXTBEncoder:
    """Strict CPU/double loader of the exported equivariant parameter graph."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if (m.get('format') != 'MATRIX-NNXTB-CodeOcean' or m.get('version') != 1
                or m.get('parameter_profile') != 'codeocean-v1-libcint-fixed-basis'
                or m.get('reference_ev_to_hartree') != OFFICIAL_EV_TO_HARTREE
                or m.get('coordinate_unit') != 'angstrom'
                or tuple(m.get('element_heads', ())) != ELEMENT_HEADS
                or tuple(m.get('element_head_lengths', ())) != ELEMENT_LENGTHS
                or tuple(m.get('global_heads', ())) != GLOBAL_HEADS
                or m.get('elements') != list(range(1, 101))
                or m.get('cutoff_angstrom') != 5.0):
            raise GFN2NNFormatError('Unsupported official NNxTB manifest')
        encoder_path = (self.path.parent / m['encoder']).resolve()
        if encoder_path.parent != self.path.parent:
            raise GFN2NNFormatError('Encoder must be a sibling of its manifest')
        identity = _file_identity(encoder_path)
        with _model_load_lock:
            self.model = _verified_model(str(encoder_path), identity,
                                         m.get('encoder_sha256'))

    def predict_torch(self, atomic_numbers, coordinates_bohr):
        import torch

        numbers = np.asarray(atomic_numbers)
        if (numbers.ndim != 1 or not len(numbers) or not np.all(numbers == numbers.astype(int))
                or np.any(numbers < 1) or np.any(numbers > 100)):
            raise ValueError('Official NNxTB requires atomic numbers 1..100')
        if coordinates_bohr.shape != (len(numbers), 3):
            raise ValueError('Official NNxTB coordinates must have shape (atoms, 3)')
        if coordinates_bohr.dtype != torch.float64 or coordinates_bohr.device.type != 'cpu':
            raise ValueError('Official encoder requires CPU float64 coordinates')
        if not torch.isfinite(coordinates_bohr).all():
            raise ValueError('Nonfinite official NNxTB coordinates')
        xyz = coordinates_bohr / MATRIX_ANGSTROM_TO_BOHR
        # Discrete connectivity is held fixed while differentiating smooth
        # radial/ angular features, as in the reference molecular graph.
        r = xyz.detach().numpy()
        distance = np.linalg.norm(r[:, None] - r[None, :], axis=-1)
        mask = ~np.eye(len(numbers), dtype=bool)
        if np.any(distance[mask] < 1e-10):
            raise ValueError('Coincident atoms in official NNxTB input')
        edges = torch.as_tensor(np.array(np.where(mask & (distance < 5.0))), dtype=torch.long)
        z = torch.as_tensor(numbers.copy(), dtype=torch.long)
        atom, global_ = self.model(xyz, z, edges)
        if atom.shape != (len(numbers), 29) or global_.shape != (1, 24):
            raise GFN2NNFormatError('Official encoder returned incorrect parameter dimensions')
        if not torch.isfinite(atom).all() or not torch.isfinite(global_).all():
            raise ArithmeticError('Nonfinite official NNxTB parameter prediction')
        return atom, global_

    def predict(self, atomic_numbers, coordinates_bohr):
        import torch

        with torch.no_grad():
            values = self.predict_torch(atomic_numbers, torch.as_tensor(
                np.asarray(coordinates_bohr), dtype=torch.float64))
        return tuple(value.numpy() for value in values)


class OfficialShifts(dict):
    """Retain the official parameter semantics through frozen SCF workers."""
    profile = 'codeocean-v1'


class OfficialNNXTBOverlay(NNXTBOverlay):
    """Official network mapped to the existing native parameter/AD boundary."""

    def __init__(self, path):
        self.encoder = OfficialNNXTBEncoder(path)
        self._last_shifts = None
        atom_names = ('gam', 'gam3', 'repa', 'repb', 'dpol', 'qpol', 'en',
                      'rcov', 'arad', 'mprad', 'mpvcn')
        shell_names = ('lev', 'h0exp', 'kcn', 'poly', 'lpar', 'third')
        global_names = ('wexp', 'enscale', 's6', 's8', 'a1', 'a2', 's9', 's10',
                        'kexp', 'klight', 'alphaj', 'aesdmp3', 'aesdmp5',
                        'aesexp', 'aesshift', 'aesrmax')
        heads = tuple(GFN2NNHeadSpec('atom.'+n, 'atomwise', 1, 1.) for n in atom_names)
        heads += tuple(GFN2NNHeadSpec('shell.'+n, 'atomwise', 4, 1.) for n in shell_names)
        heads += tuple(GFN2NNHeadSpec('global.'+n, 'global', 1, 1.) for n in global_names)
        self.info = GFN2NNModelInfo(str(self.encoder.path), 1,
            {'arch':'codeocean-mace', 'shift_bound':'none',
             'parameter_profile':self.encoder.manifest['parameter_profile'],
             'source_checkpoint_sha256':self.encoder.manifest['source_checkpoint_sha256'],
             'encoder_sha256':self.encoder.manifest['encoder_sha256']},
            self.encoder.manifest['loaded_source_tensors'], 0, heads=heads)

    def torch_parameter_shifts(self, numbers, coordinates, **_):
        import torch
        from .parameters import EV_PER_HARTREE, load_gfn2_parameters

        table = load_gfn2_parameters()
        if any(int(z) not in table.elements for z in numbers):
            raise ValueError('Native MATRIX-GFN2 parameter coverage is H through Rn')

        atom, global_ = self.encoder.predict_torch(numbers, coordinates)
        values = dict(zip(ELEMENT_HEADS, torch.split(atom, ELEMENT_LENGTHS, dim=1)))
        globals_ = dict(zip(GLOBAL_HEADS, global_.split(1, dim=1)))
        result = {}
        for old, new in {'gam':'gam', 'gam3':'gam3', 'arep':'repa', 'zeff':'repb',
                         'dkernel':'dpol', 'qkernel':'qpol', 'en':'en', 'rcov':'rcov',
                         'arad':'arad', 'mprad':'mprad', 'mpvcn':'mpvcn'}.items():
            result['atom.'+new] = values[old]
        for old, new in {'levels':'lev', 'slater':'h0exp', 'kcn':'kcn',
                         'shpoly':'poly', 'lgam':'lpar'}.items():
            value = values[old]
            if new in ('lev', 'kcn'):
                attribute = 'level_hartree' if new == 'lev' else 'coordination_shift_hartree'
                base_hartree = coordinates.new_tensor([
                    [getattr(sh, attribute) for sh in table.elements[int(z)].shells]
                    + [0.] * (3-len(table.elements[int(z)].shells)) for z in numbers
                ])
                value = ((base_hartree * EV_PER_HARTREE + value) * OFFICIAL_EV_TO_HARTREE
                         - base_hartree)
            elif new == 'h0exp':
                base_slater = coordinates.new_tensor([
                    [sh.slater_exponent for sh in table.elements[int(z)].shells]
                    + [1.] * (3-len(table.elements[int(z)].shells)) for z in numbers
                ])
                value = torch.relu(base_slater + value) - base_slater
            result['shell.'+new] = torch.nn.functional.pad(value, (0,1))
        # The reference adds the global and per-atom third-order scales.
        # MATRIX stores a multiplier relative to the s/p/d baseline.
        angular_index = torch.tensor([
            ['spd'.index(sh.label[-1]) for sh in table.elements[int(z)].shells]
            + [0] * (3-len(table.elements[int(z)].shells)) for z in numbers
        ], dtype=torch.long)
        base = coordinates.new_tensor([1., .5, .25])[angular_index]
        angular = torch.cat([globals_[key] for key in ('s','p','d')], dim=1)[0, angular_index]
        third = torch.relu(base + angular + values['3rd_scale']) / base - 1.
        result['shell.third'] = torch.nn.functional.pad(third, (0,1))
        for old, new in {'wexp':'wexp', 'enscale':'enscale', 's6':'s6', 's8':'s8',
                         'a1':'a1', 'a2':'a2', 's9':'s9', 's10':'s10',
                         'kexp':'kexp', 'klight':'klight', 'gexp':'alphaj',
                         'dmp3':'aesdmp3', 'dmp5':'aesdmp5', 'shift':'aesshift',
                         'rmax':'aesrmax'}.items():
            result['global.'+new] = globals_[old]
        result['global.aesexp'] = globals_['kexp']
        # H0 uses learned CN radii, whereas the official AES term uses the
        # unshifted radii. Express that distinction at the existing native
        # CN/vcn boundary; all coordinate and parameter chain rules remain AD.
        from .calculator import _atomic_radii, _covalent_radii
        from .nn_xtb_parameters import _XTB_LEGACY_RADIUS_SCALE
        from .kernels import batched_gfn2_coordination
        rcov0 = coordinates.new_tensor(_covalent_radii()[np.asarray(numbers, dtype=int)-1]) * _XTB_LEGACY_RADIUS_SCALE
        rcov = torch.relu(4. / 3. * rcov0 + values['rcov'][:, 0])
        result['atom.rcov'] = (rcov - 4. / 3. * rcov0)[:, None]
        cn_learned = batched_gfn2_coordination(coordinates, .75 * rcov, backend='torch')
        cn_fixed = batched_gfn2_coordination(coordinates, rcov0, backend='torch')
        vcn0 = coordinates.new_tensor([table.elements[int(z)].multipole_valence_coordination for z in numbers])
        result['atom.mpvcn'] = (torch.relu(vcn0 + values['mpvcn'][:, 0]) - vcn0
                                + cn_learned - cn_fixed)[:, None]
        arad0 = coordinates.new_tensor(_atomic_radii()[np.asarray(numbers, dtype=int)-1])[:, None] * _XTB_LEGACY_RADIUS_SCALE
        result['atom.arad'] = torch.relu(arad0 + values['arad']) - arad0
        for head, attribute, source in (
            ('repb', 'effective_nuclear_charge', 'zeff'),
            ('repa', 'repulsion_exponent', 'arep'),
            ('mprad', 'multipole_radius_bohr', 'mprad'),
            ('en', 'electronegativity', 'en'),
        ):
            base_value = coordinates.new_tensor([getattr(table.elements[int(z)], attribute) for z in numbers])[:, None]
            result['atom.'+head] = torch.relu(base_value + values[source]) - base_value
        # kpol and ss/pp/dd/sd/pd are present in the checkpoint but bypassed
        # by the official Hamiltonian's direct ParameterModule.param reads.
        # Do not activate scientifically different couplings in this port.
        return {name: value/_MATRIX_SHIFT_FACTORS.get(name, 1.)
                for name,value in result.items()}

    def parameter_shifts(self, atomic_numbers, coordinates_bohr, **kwargs):
        import torch

        with torch.no_grad():
            values = self.torch_parameter_shifts(atomic_numbers, torch.as_tensor(
                np.asarray(coordinates_bohr), dtype=torch.float64), **kwargs)
        return {name:value.numpy() for name,value in values.items()}

    def parameter_shift_curvature_matrix(
        self, atomic_numbers, coordinates_bohr, adjoints, **kwargs,
    ):
        """Differentiate all Cartesian directions using one NN forward graph."""
        import torch

        xyz = torch.as_tensor(np.asarray(coordinates_bohr, dtype=float), dtype=torch.float64)
        if xyz.shape != (len(atomic_numbers), 3) or not torch.isfinite(xyz).all():
            raise ValueError('Official NNxTB Hessian coordinates are invalid')
        extra = set(adjoints) - {head.name for head in self.info.heads}
        if extra or not adjoints:
            raise ValueError(f'Invalid official NNxTB adjoints: {sorted(extra)}')
        weights = {name: torch.as_tensor(value, dtype=torch.float64)
                   for name, value in adjoints.items()}

        def weighted_shifts(positions):
            outputs = self.torch_parameter_shifts(atomic_numbers, positions, **kwargs)
            if any(weights[name].shape != outputs[name].shape for name in weights):
                raise ValueError('Official NNxTB adjoint shape mismatch')
            return sum((outputs[name] * weight).sum() for name, weight in weights.items())

        curvature = torch.autograd.functional.hessian(
            weighted_shifts, xyz, vectorize=False,
        )
        return curvature.detach().numpy().reshape((xyz.numel(), xyz.numel()))

    def matrix_parameter_shifts(self, atomic_numbers, coordinates_bohr, **kwargs):
        coordinates = np.asarray(coordinates_bohr, dtype=float)
        key = (tuple(atomic_numbers), coordinates.shape, coordinates.tobytes(),
               tuple(sorted(kwargs.items())))
        cached = self._last_shifts
        if cached is not None and cached[0] == key:
            return OfficialShifts(cached[1])
        result = OfficialShifts(super().matrix_parameter_shifts(
            atomic_numbers, coordinates_bohr, **kwargs))
        # Cache only immutable predictions, never a live differentiation graph.
        self._last_shifts = key, result
        return OfficialShifts(result)
