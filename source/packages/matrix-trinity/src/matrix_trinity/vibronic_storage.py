"""Blockwise, non-pickle NumPy archives for large vibronic line tables."""
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np


class ArrayBlocks:
    """A logical concatenation whose C-order bytes are streamed without joining."""
    def __init__(self,parts,*,axis=0,empty=None):
        self.parts=tuple(np.asarray(x) for x in parts)
        if not self.parts:
            if empty is None:raise ValueError('Empty block arrays require an explicit shape/dtype')
            self.parts=(np.asarray(empty),)
        first=self.parts[0]
        if not 0<=axis<first.ndim:raise ValueError('Valid concatenation axis required')
        self.axis=axis;self.dtype=np.result_type(*[x.dtype for x in self.parts])
        if self.dtype.hasobject:raise ValueError('Object blocks are forbidden')
        shape=list(first.shape);shape[axis]=0
        for x in self.parts:
            if x.ndim!=first.ndim or any(x.shape[k]!=first.shape[k] for k in range(x.ndim) if k!=axis):
                raise ValueError('Incompatible array block dimensions')
            shape[axis]+=x.shape[axis]
        self.shape=tuple(shape);self.size=int(np.prod(self.shape,dtype=object));self.nbytes=self.size*self.dtype.itemsize

    def chunks(self):
        # A column concatenation needs prefix-major iteration for C-order bytes.
        for prefix in np.ndindex(self.shape[:self.axis]):
            for part in self.parts:
                chunk=np.ascontiguousarray(part[prefix],dtype=self.dtype)
                if chunk.size:yield memoryview(chunk).cast('B')

    def materialize(self,path=None):
        if path is not None:
            _atomic_npy(path,self)
            return np.load(path,mmap_mode='r',allow_pickle=False)
        result=np.empty(self.shape,self.dtype);offset=0
        for part in self.parts:
            selection=[slice(None)]*len(self.shape);selection[self.axis]=slice(offset,offset+part.shape[self.axis])
            result[tuple(selection)]=part;offset+=part.shape[self.axis]
        return result


def array_info(value):
    if isinstance(value,ArrayBlocks):return value.dtype,value.shape,value.chunks()
    value=np.asarray(value)
    if value.dtype.hasobject:raise ValueError('Object arrays are forbidden')
    if value.flags.c_contiguous:
        chunks=(() if not value.size else (memoryview(value).cast('B'),))
    else:
        def buffered():
            iterator=np.nditer(value,flags=['external_loop','buffered','zerosize_ok'],order='C',buffersize=131072)
            for block in iterator:
                yield memoryview(np.ascontiguousarray(block)).cast('B')
        chunks=buffered()
    return value.dtype,value.shape,chunks


def write_npy(stream,value):
    dtype,shape,chunks=array_info(value)
    np.lib.format.write_array_header_2_0(stream,dict(descr=np.lib.format.dtype_to_descr(dtype),fortran_order=False,shape=shape))
    for chunk in chunks:stream.write(chunk)


def write_npz_blocks(stream,arrays,*,compression_level=0):
    """Standard NPZ, with explicit lossless compression and bounded write buffers."""
    if isinstance(compression_level,bool) or not isinstance(compression_level,int) or not 0<=compression_level<=9:
        raise ValueError('Integer compression_level between zero and nine required')
    compression=zipfile.ZIP_STORED if compression_level==0 else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(stream,'w',compression=compression,allowZip64=True,
            compresslevel=None if compression_level==0 else compression_level) as archive:
        for name,value in arrays.items():
            if not isinstance(name,str) or '/' in name or '\\' in name:raise ValueError('Simple array names required')
            with archive.open(name+'.npy','w',force_zip64=True) as member:write_npy(member,value)


def save_vibronic_columns(path,columns,*,compression_level=0):
    """Atomically export a line table; arrays and ArrayBlocks share one format."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            write_npz_blocks(stream,columns,compression_level=compression_level);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return str(path.resolve())


def _atomic_npy(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            write_npy(stream,value);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


class FileArrayBlocks(ArrayBlocks):
    """One append-only raw field, exposed as standard C-order array blocks."""
    def __init__(self,path,dtype,shape):
        self.path=Path(path);self.dtype=np.dtype(dtype);self.shape=tuple(shape)
        self.size=int(np.prod(self.shape,dtype=object));self.nbytes=self.size*self.dtype.itemsize

    def chunks(self):
        remaining=self.nbytes
        with self.path.open('rb') as stream:
            while remaining:
                chunk=stream.read(min(8*1024**2,remaining))
                if not chunk:raise IOError('Incomplete disk-backed transition field')
                remaining-=len(chunk);yield chunk

    def materialize(self,path=None):
        if path is None:return np.fromfile(self.path,dtype=self.dtype,count=self.size).reshape(self.shape)
        return super().materialize(path)


class LineBlockStore:
    """Append six typed transition fields without retaining a full line table."""
    names=('states','energy','population','amplitude','strength','weighted')
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='.line-blocks-',dir=self.path)
        self.streams=[];self.dtypes=[];self.tails=[];self.rows=0

    def append(self,batch):
        arrays=tuple(np.asarray(x) for x in batch)
        if len(arrays)!=len(self.names) or any(x.ndim<1 or len(x)!=len(arrays[0]) or x.dtype.hasobject for x in arrays):
            raise ValueError('Six compatible non-object transition fields required')
        if not self.streams:
            self.dtypes=[x.dtype for x in arrays];self.tails=[x.shape[1:] for x in arrays]
            self.streams=[(Path(self.temporary.name)/(name+'.raw')).open('wb') for name in self.names]
        for x,dtype,tail,stream in zip(arrays,self.dtypes,self.tails,self.streams):
            if x.shape[1:]!=tail:raise ValueError('Transition field dimensions changed')
            # Cast to the established archive dtype; occupation domain is fixed.
            if not np.can_cast(x.dtype,dtype,casting='safe'):raise ValueError('Transition field dtype changed')
            value=np.ascontiguousarray(x,dtype=dtype)
            if value.size:stream.write(memoryview(value).cast('B'))
        self.rows+=len(arrays[0])

    def arrays(self):
        for stream in self.streams:stream.flush()
        return tuple(FileArrayBlocks(Path(self.temporary.name)/(name+'.raw'),dtype,(self.rows,)+tail)
            for name,dtype,tail in zip(self.names,self.dtypes,self.tails))

    def materialize(self):
        arrays=self.arrays()
        return tuple(array.materialize(self.path/(name+'.npy')) for name,array in zip(self.names,arrays))

    def close(self):
        for stream in self.streams:stream.close()
        self.streams=[];self.temporary.cleanup()

    def __del__(self):
        if hasattr(self,'temporary'):self.close()
