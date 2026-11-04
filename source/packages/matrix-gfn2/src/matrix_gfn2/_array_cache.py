"""Small immutable index tables shared by resident GFN2 kernels."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=128)
def triangular_indices(size: int, diagonal: bool = True):
    rows, columns = np.tril_indices(int(size), k=0 if diagonal else -1)
    rows.setflags(write=False)
    columns.setflags(write=False)
    return rows, columns
