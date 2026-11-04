"""Atomic, content-bound checkpoints for assignment workflows and stage fits."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def assignment_fingerprint(*objects):
    """Bind complete scientific inputs, initial values and declared budgets."""
    def canonical(value):
        if isinstance(value,np.ndarray):
            array=np.ascontiguousarray(value)
            if array.dtype.hasobject:
                raise TypeError('object arrays cannot identify an assignment calculation')
            return dict(shape=array.shape,dtype=array.dtype.str,
                        sha256=hashlib.sha256(array.tobytes()).hexdigest())
        if isinstance(value,np.generic):return canonical(value.item())
        if is_dataclass(value):
            return dict(type=type(value).__module__+'.'+type(value).__name__,
                        fields={f.name:canonical(getattr(value,f.name)) for f in fields(value)})
        if isinstance(value,Mapping):
            return {str(k):canonical(v) for k,v in sorted(value.items(),key=lambda p:str(p[0]))}
        if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
        if isinstance(value,Path):return str(value)
        if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
        if isinstance(value,float) and not np.isfinite(value):return dict(float=str(value))
        if value is None or isinstance(value,(str,int,float,bool)):return value
        raise TypeError(f'unsupported assignment identity value: {type(value).__name__}')
    encoded=json.dumps(canonical(objects),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def save_assignment_checkpoint(path,state,*,contract):
    """Reuse the pickle-free typed NPZ writer and its atomic file replacement."""
    from .lam_checkpoint import write_lam_workspace
    return write_lam_workspace(path,dict(schema='matrix.lock.assignment_checkpoint.v1',state=state),
                               contract=contract)


def load_assignment_checkpoint(path,*,contract):
    from .lam_checkpoint import read_lam_workspace
    payload=read_lam_workspace(path,contract=contract)
    if payload.get('schema')!='matrix.lock.assignment_checkpoint.v1':
        raise ValueError('unsupported assignment checkpoint schema')
    return payload['state']


__all__=['assignment_fingerprint','save_assignment_checkpoint','load_assignment_checkpoint']
