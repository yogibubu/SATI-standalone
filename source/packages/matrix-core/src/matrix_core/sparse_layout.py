"""Read-only dense views of explicitly full, canonical CSR storage."""
import numpy as np
from scipy.sparse import isspmatrix_csr


def dense_csr_view(matrix):
    """Return a zero-copy dense view only after verifying every column index.

    Explicit stored zeros are retained. Unsorted rows, duplicate entries,
    missing entries, non-CSR formats and non-contiguous values return None.
    Pattern checking uses bounded row chunks and never trusts cached CSR flags.
    """
    if not isspmatrix_csr(matrix):return None
    rows,columns=matrix.shape
    if (not rows or not columns or matrix.nnz!=rows*columns
            or not matrix.data.flags.c_contiguous
            or not np.all(np.diff(matrix.indptr)==columns)):
        return None
    # Avoid an unbounded column-index temporary on unusual very wide inputs.
    budget=8*1024**2
    if columns*matrix.indices.dtype.itemsize>budget:return None
    expected=np.arange(columns,dtype=matrix.indices.dtype)
    indices=matrix.indices.reshape(rows,columns)
    batch=max(1,budget//columns)
    for start in range(0,rows,batch):
        if not np.all(indices[start:start+batch]==expected):return None
    result=matrix.data.reshape(rows,columns).view()
    result.setflags(write=False)
    return result
