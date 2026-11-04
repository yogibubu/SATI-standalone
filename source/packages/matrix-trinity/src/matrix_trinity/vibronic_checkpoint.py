"""Atomic, non-pickle restart archives for adaptive harmonic TI engines."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from numba import njit, types
from numba.typed import List
from .vibronic_selection import ConditionalOccupationTree, _packed_find
from .vibronic_storage import ArrayBlocks,FileArrayBlocks,array_info,write_npz_blocks

_HEAP_ITEM_TYPE = types.Tuple((types.float64,types.int64,types.int64))


def _digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        # Preserve the original scalar fingerprint and C-order concatenation.
        if not isinstance(value,ArrayBlocks):value=np.ascontiguousarray(value)
        dtype,shape,chunks=array_info(value)
        digest.update(json.dumps([name,dtype.str,shape]).encode())
        for chunk in chunks:digest.update(chunk)
    return digest.hexdigest()


@njit(cache=True)
def _rehash(keys, slots, count):
    for index in range(count):
        found, slot = _packed_find(keys[index], keys, slots)
        if found >= 0:
            raise ValueError('duplicate overlap keys in checkpoint')
        slots[slot] = index


@njit(cache=True)
def _heap_arrays_numba(heap):
    """Preserve heap order without boxing millions of native tuples in Python."""
    costs=np.empty(len(heap),np.float64)
    nodes=np.empty(len(heap),np.int64)
    axes=np.empty(len(heap),np.int64)
    for index in range(len(heap)):
        costs[index],nodes[index],axes[index]=heap[index]
    return costs,nodes,axes


def _heap_arrays(heap):
    if hasattr(heap,'native_arrays'):return heap.native_arrays()
    return _heap_arrays_numba(heap)


@njit(cache=True)
def _restore_heap(costs,nodes,axes):
    heap=List.empty_list(_HEAP_ITEM_TYPE)
    for index in range(len(costs)):
        heap.append((costs[index],nodes[index],axes[index]))
    return heap


class AdaptiveCheckpoint:
    """Budgets, workers, batch size and target may change; physics may not.

    No arbitrary Python objects are deserialized. Writes replace the previous
    complete archive atomically. One calculation must own each checkpoint path.
    """
    def __init__(self, path, resume, identity, arrays, interval_seconds=60., *, asynchronous=False):
        if not isinstance(asynchronous,bool):raise ValueError('checkpoint asynchronous must be boolean')
        self.asynchronous=asynchronous;self._executor=None;self._future=None;self._closed=False
        self.statistics=dict(scheduled=0,completed=0,skipped_busy=0,snapshot_seconds=0.,write_seconds=0.,wait_seconds=0.,maximum_pending=0)
        if not np.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError('checkpoint interval seconds must be finite and nonnegative')
        if not isinstance(resume, bool): raise ValueError('resume must be boolean')
        self.interval_seconds = interval_seconds
        self.last_save = time.monotonic()
        self.path = str(Path(path).expanduser().resolve()) if path is not None else None
        self.resumed = False
        self.identity = hashlib.sha256((json.dumps(identity, sort_keys=True, allow_nan=False)
                                       +_digest(arrays)).encode()).hexdigest()
        self.archive = None
        if resume and self.path is None:
            raise ValueError('resume requires checkpoint_path')
        if self.path is None:
            return
        if not resume:
            if Path(self.path).exists():
                raise FileExistsError('checkpoint exists; use resume=True or a new path')
            return
        with np.load(self.path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
            if metadata.get('schema') != 'matrix.sati.adaptive_checkpoint.v1':
                raise ValueError('unsupported TI checkpoint schema')
            if metadata.get('identity') != self.identity:
                raise ValueError('checkpoint model/operator/domain identity mismatch')
            arrays = {name: archive[name] for name in archive.files if name != 'metadata'}
        if metadata['sha256'] != _digest(arrays):
            raise ValueError('checkpoint integrity check failed')
        self.archive = (metadata, arrays)
        self.resumed = True

    def due(self):
        self.poll()
        return self.path is not None and time.monotonic()-self.last_save >= self.interval_seconds

    def restore(self, cache, budget):
        if self.archive is None:
            return None, {}
        metadata, arrays = self.archive
        state = metadata['tree']
        count = metadata['cache_count']
        if budget < state['visited'] or len(cache.values) < count:
            raise ValueError('resume budgets cannot be smaller than saved visited states/cache')
        if state.get('type')=='window_forest_v1':
            from .vibronic_window_selection import WindowOccupationForest
            tree=WindowOccupationForest.restore(state,arrays,budget,getattr(cache,'native_library',None))
        else:
            tree = ConditionalOccupationTree.__new__(ConditionalOccupationTree)
            for name in ('allocated', 'visited', 'modes', 'qi', 'qf', 'maximum_class'):
                setattr(tree, name, state[name])
            tree.budget = budget
            for name in ('costs', 'values', 'parents', 'original'):
                setattr(tree, name, arrays['tree_'+name])
            for name in ('ancestors', 'axes', 'ranks'):
                value = np.zeros(2*budget+2, np.int64)
                saved = arrays['tree_'+name]
                value[:len(saved)] = saved
                setattr(tree, name, value)
            if not (len(arrays['heap_cost']) == len(arrays['heap_node']) == len(arrays['heap_axis'])):
                raise ValueError('checkpoint heap array lengths differ')
            if getattr(cache,'native_library',None) is not None:
                from .vibronic_native_kernels import SavedNativeHeap
                tree.heap=SavedNativeHeap(arrays['heap_cost'],arrays['heap_node'],arrays['heap_axis'])
            else:tree.heap = _restore_heap(arrays['heap_cost'],arrays['heap_node'],arrays['heap_axis'])
        if arrays['cache_keys'].shape != (count, cache.keys.shape[1]):
            raise ValueError('checkpoint overlap key shape mismatch')
        cache.keys[:count] = arrays['cache_keys']; cache.values[:count] = arrays['cache_values']
        cache.count = count; cache.edge_count = metadata['cache_edge_count']
        cache.slots.fill(-1)
        if getattr(cache,'native_library',None) is not None:cache.native_library.rehash(cache)
        else:_rehash(cache.keys, cache.slots, count)
        self.archive = None
        return tree, {name[8:]: value for name, value in arrays.items() if name.startswith('payload_')}

    def poll(self):
        if self._future is not None and self._future.done():self.flush()

    def flush(self):
        if self._future is None:return
        future=self._future;self._future=None;before=time.perf_counter()
        try:
            elapsed=future.result();self.statistics['write_seconds']+=elapsed
            self.statistics['completed']+=1
        finally:self.statistics['wait_seconds']+=time.perf_counter()-before

    def close(self):
        if self._closed:return
        try:self.flush()
        finally:
            if self._executor is not None:self._executor.shutdown(wait=True)
            self._closed=True

    def save(self, tree, cache, payload, *, final=False, append_only=False):
        """Save one coherent snapshot, with at most one background write.

        append_only is an internal ownership contract: tree/cache prefixes and
        line blocks never change after a batch. Mutable reduction payloads are
        copied regardless. Generic callers receive a full owned snapshot.
        """
        if self._closed:raise RuntimeError('Checkpoint writer is closed')
        if self.path is None:return False
        if final:self.flush()
        else:
            self.poll()
            if self._future is not None:
                self.statistics['skipped_busy']+=1;return False
        before=time.perf_counter()
        if getattr(tree,'is_window_forest',False):
            arrays,tree_state=tree.checkpoint_arrays()
        else:
            arrays = {'tree_'+name: getattr(tree, name) for name in ('costs', 'values', 'parents', 'original')}
            arrays.update({'tree_'+name: getattr(tree, name)[:tree.allocated+1] for name in ('ancestors', 'axes', 'ranks')})
            costs,nodes,axes = _heap_arrays(tree.heap)
            arrays.update(heap_cost=costs,heap_node=nodes,heap_axis=axes)
            tree_state={name:getattr(tree,name) for name in ('allocated','visited','modes','qi','qf','maximum_class')}
        arrays.update(cache_keys=cache.keys[:cache.count],cache_values=cache.values[:cache.count])
        arrays.update({'payload_'+name: value if isinstance(value,ArrayBlocks) else np.asarray(value) for name,value in payload.items()})
        metadata = dict(schema='matrix.sati.adaptive_checkpoint.v1', identity=self.identity,
            cache_count=cache.count, cache_edge_count=cache.edge_count,
            tree=tree_state)
        if self.asynchronous:
            def snapshot(value,share):
                if isinstance(value,FileArrayBlocks):
                    if share:return FileArrayBlocks(value.path,value.dtype,value.shape)
                    value=value.materialize()
                if isinstance(value,ArrayBlocks):
                    return ArrayBlocks([snapshot(part,share) for part in value.parts],axis=value.axis)
                result=np.asarray(value).view() if share else np.array(value,copy=True)
                result.flags.writeable=False;return result
            arrays={name:snapshot(value,append_only and
                (name.startswith(('tree_','cache_','heap_','payload_column_')))) for name,value in arrays.items()}
        self.statistics['snapshot_seconds']+=time.perf_counter()-before
        self.statistics['scheduled']+=1;self.last_save=time.monotonic()
        if self.asynchronous:
            if self._executor is None:self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='matrix-checkpoint')
            self._future=self._executor.submit(self._write,arrays,metadata)
            self.statistics['maximum_pending']=1
            if final:self.flush()
        else:
            self.statistics['write_seconds']+=self._write(arrays,metadata);self.statistics['completed']+=1
        return True

    def _write(self,arrays,metadata):
        before=time.perf_counter();metadata={**metadata,'sha256':_digest(arrays)}
        destination = Path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name+'.', suffix='.tmp', dir=destination.parent)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                write_npz_blocks(output,dict(metadata=np.asarray(json.dumps(metadata,sort_keys=True)),**arrays))
                output.flush(); os.fsync(output.fileno())
            os.replace(temporary, destination)

        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return time.perf_counter()-before
