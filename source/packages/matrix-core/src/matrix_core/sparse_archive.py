"""Atomic SciPy-compatible sparse archives with bounded compression overhead."""
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def save_sparse_archive(path, matrix, *, compression_level=0):
    """Write exact sparse arrays to an atomic standard NPZ archive.

    The default disables compression for checkpoint throughput. No
    floating-point values, explicit zeros, indices or duplicate entries change.
    The exact supplied filename is retained. Readers use scipy.sparse.load_npz.
    """
    from scipy import sparse
    if isinstance(compression_level,bool) or not isinstance(compression_level,int) or not 0<=compression_level<=9:
        raise ValueError('integer ZIP compression level from zero to nine required')
    if not sparse.issparse(matrix):raise TypeError('SciPy sparse matrix or array required')
    arrays={}
    if matrix.format in ('csr','csc','bsr'):
        arrays.update(indices=matrix.indices,indptr=matrix.indptr)
    elif matrix.format=='dia':arrays['offsets']=matrix.offsets
    elif matrix.format=='coo':
        if matrix.ndim==2:arrays.update(row=matrix.row,col=matrix.col)
        else:arrays['coords']=matrix.coords
    else:raise NotImplementedError('Unsupported sparse archive format: '+matrix.format)
    arrays.update(format=matrix.format.encode('ascii'),shape=matrix.shape,data=matrix.data)
    if isinstance(matrix,sparse.sparray):arrays['_is_array']=True
    path=Path(path)
    handle=tempfile.NamedTemporaryFile(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent,delete=False)
    temporary=Path(handle.name);handle.close()
    try:
        method=zipfile.ZIP_DEFLATED if compression_level else zipfile.ZIP_STORED
        with zipfile.ZipFile(temporary,'w',compression=method,
                             compresslevel=compression_level if compression_level else None,
                             allowZip64=True) as archive:
            for name,array in arrays.items():
                with archive.open(name+'.npy','w',force_zip64=True) as stream:
                    np.save(stream,np.asarray(array),allow_pickle=False)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__=['save_sparse_archive']
