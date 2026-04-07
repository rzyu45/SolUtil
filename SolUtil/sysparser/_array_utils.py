import numpy as np


def to_writable_array(values, dtype=None):
    # pandas 3 Copy-on-Write can hand out read-only NumPy views for Series-backed data.
    return np.array(values, dtype=dtype, copy=True)
