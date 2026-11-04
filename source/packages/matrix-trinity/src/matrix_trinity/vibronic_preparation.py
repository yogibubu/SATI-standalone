"""Explicit bounded reuse of physical Gaussian preparations and exact results."""
from collections import OrderedDict
from functools import wraps
import hashlib
import threading
from pathlib import Path
import numpy as np


def _fingerprint(*values):
    digest=hashlib.sha256()
    for value in values:
        if isinstance(value,str):digest.update(value.encode());continue
        array=np.ascontiguousarray(value)
        digest.update(str((array.dtype.str,array.shape)).encode())
        if array.size:digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _bytes(value):
    if isinstance(value,np.ndarray):return value.nbytes
    if isinstance(value,dict):return sum(_bytes(x) for x in value.values())
    if isinstance(value,(tuple,list)):return sum(_bytes(x) for x in value)
    if hasattr(value,'keys') and hasattr(value,'slots') and hasattr(value,'values'):
        return value.keys.nbytes+value.slots.nbytes+value.values.nbytes
    return 0


class NativePreparationCache:
    """User-owned LRU cache. Concurrent requests serialize on a reentrant lock.

    Keys hash array contents, temperature, origin, operator and time grid as
    appropriate. No tolerance-based equality, global cache or disk discovery.
    Returned preparations and finite overlap indices/values are immutable.
    Overlap reuse requires exactly the same states and active insertions.
    """
    def __init__(self,*,maximum_bytes=128*1024**2,maximum_entries=32):
        for value in (maximum_bytes,maximum_entries):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('Positive cache limits required')
        self.maximum_bytes=maximum_bytes;self.maximum_entries=maximum_entries
        self.lock=threading.RLock();self.entries=OrderedDict();self.bytes=0;self.hits={};self.misses={}

    def _get(self,kind,key,factory,*,immutable=False):
        key=(kind,key)
        with self.lock:
            if key in self.entries:
                self.hits[kind]=self.hits.get(kind,0)+1
                value,size=self.entries.pop(key);self.entries[key]=(value,size);return value
            self.misses[kind]=self.misses.get(kind,0)+1;value=factory();size=_bytes(value)
            if immutable:
                def freeze(x):
                    if isinstance(x,np.ndarray):x.setflags(write=False)
                    elif isinstance(x,dict):
                        for y in x.values():freeze(y)
                    elif isinstance(x,(tuple,list)):
                        for y in x:freeze(y)
                freeze(value)
            if size<=self.maximum_bytes:
                while self.entries and (self.bytes+size>self.maximum_bytes or len(self.entries)>=self.maximum_entries):
                    _,(_,removed)=self.entries.popitem(last=False);self.bytes-=removed
                self.entries[key]=(value,size);self.bytes+=size
            return value

    def physical_key(self,library,model,temperature):
        path=Path(library.path);stat=path.stat()
        return _fingerprint(str(path),[stat.st_ino,stat.st_size,stat.st_mtime_ns],
            model.initial_frequencies_cm,model.final_frequencies_cm,model.mode_map,
            model.final_shift_dimensionless,[model.electronic_gap_cm,temperature])

    def gaussian(self,library,model,temperature):
        key=self.physical_key(library,model,temperature)
        geometry=self._get('geometry',key,lambda:library.prepare_geometry(model,temperature),immutable=True)
        return library.insert_dipole(geometry,model)

    def checkpoint_gaussian(self,library,model,temperature):
        from .vibronic_harmonic import harmonic_gaussian_generator
        key=_fingerprint(self.physical_key(library,model,temperature),model.transition_gradient_dimensionless_au,
            model.transition_momentum_dimensionless_au)
        return self._get('checkpoint_identity',key,lambda:harmonic_gaussian_generator(model),immutable=True)

    def finite_integrals(self,library,model,temperature,gaussian,linear,states,budget,cap):
        from .vibronic_selection import PackedHermiteCache
        key=_fingerprint(self.physical_key(library,model,temperature),states,np.any(linear!=0,axis=0),[budget,cap])
        def evaluate():
            cache=PackedHermiteCache(gaussian,linear,budget,cap,native_library=library)
            ids,exhausted=cache.evaluate(states)
            if exhausted:raise MemoryError('TI recurrence cache budget exceeded; reduce the requested state space')
            return ids,cache.values[:cache.count].copy()
        return self._get('overlaps',key,evaluate,immutable=True)

    def correlation(self,library,model,times,temperature,direction):
        key=_fingerprint(self.physical_key(library,model,temperature),model.transition_dipole_au,
            model.transition_gradient_dimensionless_au,model.transition_momentum_dimensionless_au,times,[direction])
        values,details=self._get('correlation',key,lambda:library.correlation(model,times,temperature,direction),immutable=True)
        return values.copy(),dict(details)

    def snapshot(self):
        with self.lock:return dict(bytes=self.bytes,entries=len(self.entries),hits=dict(self.hits),misses=dict(self.misses),
            maximum_bytes=self.maximum_bytes,maximum_entries=self.maximum_entries)

    def clear(self):
        with self.lock:self.entries.clear();self.bytes=0


def preparation_session(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        cache=kwargs.get('preparation_cache')
        if cache is None:return function(*args,**kwargs)
        if not isinstance(cache,NativePreparationCache):raise TypeError('NativePreparationCache required')
        with cache.lock:return function(*args,**kwargs)
    return wrapped
