import numpy as np
from numba import njit, prange


@njit
def get_min_max_range_numba(data):
    min_val = max_val = data[0]
    for x in data:
        if x < min_val:
            min_val = x
        elif x > max_val:
            max_val = x
    return min_val, max_val - min_val


@njit
def get_min_numba(data):
    min_val = data[0]
    for x in data:
        if x < min_val:
            min_val = x
    return min_val


@njit
def get_max_numba(data):
    max_val = data[0]
    for x in data:
        if x > max_val:
            max_val = x
    return max_val


@njit
def get_dx_numba(xdata):
    n = xdata.shape[0] - 1
    out = np.empty(n, dtype=xdata.dtype)
    for i in range(n):
        out[i] = abs(xdata[i + 1] - xdata[i])
    return out


@njit(parallel=True)
def get_dx_numba_parallel(xdata):
    n = xdata.shape[0] - 1
    out = np.empty(n, dtype=xdata.dtype)
    for i in prange(n):
        out[i] = abs(xdata[i + 1] - xdata[i])
    return out


@njit(parallel=True)
def get_IJ(x, xmin, xrng, xres):
    n = x.shape[0]
    xrng_recip = 1 / xrng
    out = np.empty(n, dtype=np.int32)
    for i in prange(n):
        out[i] = int((x[i] - xmin) * xrng_recip * (xres - 1))
    return out


@njit(parallel=True)
def scale_to_range(x, xmin, xrng, a=0, b=1):
    n = x.shape[0]
    out = np.empty(n, dtype=x.dtype)
    for i in prange(n):
        out[i] = a + b * (x[i] - xmin) / xrng
    return out
