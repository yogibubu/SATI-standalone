"""Internal strict JSON and identity primitives for scientific orchestration."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import isfinite
from pathlib import Path
import re


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} requires a nonempty string')
    return value


def number(value, name, *, minimum=0):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} requires a finite number >= {minimum}')
    return float(value)


def digest(value, name='sha256'):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} requires a lowercase SHA-256')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seal(value, key):
    result = deepcopy(value)
    if key in result:
        raise ValueError(f'already sealed: {key}')
    result[key] = fingerprint(result)
    return result


def verify(value, key, schema):
    if not isinstance(value, dict) or value.get('schema') != schema:
        raise ValueError(f'expected {schema}')
    result = deepcopy(value)
    expected = result.pop(key, None)
    digest(expected, key)
    if fingerprint(result) != expected:
        raise ValueError(f'invalid {key}')
    return deepcopy(value)


def fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys()-set(required)-set(optional):
        raise ValueError(f'expected fields {sorted(required)} with optional {sorted(optional)}')


def read_json(path, *, maximum_bytes=16*1024*1024):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum_bytes+1)
    if len(raw) > maximum_bytes:
        raise ValueError('scientific JSON exceeds its size budget')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)  # Also reject numeric exponent overflow, e.g. 1e999.
    return value


IDENTITY_FIELDS = ('molecule_id', 'isotopologue_id', 'electronic_state', 'method',
                   'basis', 'coordinate_frame', 'geometry_sha256', 'atom_order_sha256', 'units')


def scientific_identity(value):
    fields(value, IDENTITY_FIELDS)
    for key in IDENTITY_FIELDS[:-1]:
        text(value[key], key)
    digest(value['geometry_sha256'], 'geometry_sha256')
    digest(value['atom_order_sha256'], 'atom_order_sha256')
    if not isinstance(value['units'], dict) or not value['units']:
        raise ValueError('identity requires explicit channel-to-unit declarations')
    for channel, unit in value['units'].items():
        text(channel, 'channel'); text(unit, 'unit')
    return deepcopy(value)


def verify_plan(plan, library):
    from .scientific_planning import SCIENTIFIC_PLAN_SCHEMA, build_scientific_plan

    verify(plan, 'plan_sha256', SCIENTIFIC_PLAN_SCHEMA)
    # Rebuild to verify gates, routes, dependencies and policies, rather than
    # accepting a self-consistent digest on an arbitrarily modified plan.
    if build_scientific_plan(plan['request'], library=library) != plan:
        raise ValueError('plan differs from the current frozen protocol library')
    if not plan['selected_protocol_id'] or not plan['steps']:
        raise ValueError('an explicitly resolved scientific plan is required')
    return deepcopy(plan)
