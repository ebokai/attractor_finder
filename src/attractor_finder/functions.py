import functools
import time

import numpy as np
from numba import njit, prange

from attractor_finder.functions_numba import get_min_max_range_numba


def get_min_max_range(data):
    max_val = np.max(data)
    min_val = np.min(data)
    data_range = max_val - min_val
    return min_val, data_range


def get_dx(xdata):
    dx = np.abs(xdata[1:] - xdata[:-1])
    return dx


def zalpha(z, zmin, zrng, a_min=0):
    """return alpha based on z depth"""
    alpha = a_min + (1 - a_min) * (z - zmin) / zrng
    return alpha


def get_index(x, xmin, xrng, xres):
    """map coordinate to array index"""
    return int((x - xmin) / xrng * (xres - 1))


def set_aspect(xdata, ydata, width, height, debug=False, margin=1.1):
    """get boundaries for given aspect ratio w/h"""
    xmin, xrng = get_min_max_range_numba(xdata)
    ymin, yrng = get_min_max_range_numba(ydata)

    if debug:
        print(" Data Summary")
        print("────────────────────────────────────────────")
        print(f"• X Range:          {xrng:.2f}")
        print(f"• Y Range:          {yrng:.2f}")
        print(f"• Aspect Ratio:     {xrng/yrng:.2f}\n")

    xmid = xmin + xrng / 2
    ymid = ymin + yrng / 2

    data_aspect = xrng / yrng
    target_aspect = width / height

    if data_aspect < target_aspect:
        xrng = target_aspect * yrng
    else:
        yrng = xrng / target_aspect

    xrng *= margin
    yrng *= margin

    xmin = xmid - xrng / 2.0
    ymin = ymid - yrng / 2
    if debug:
        print(" Rescaled Data")
        print("────────────────────────────────────────────")
        print(f"• X Range:          {xrng:.2f}")
        print(f"• Y Range:          {yrng:.2f}")
        print(f"• Aspect Ratio:     {xrng/yrng:.2f}\n")

    return xmin, ymin, xrng, yrng


def pixel_density(xl, yl, xres=320, yres=320):
    """check for density of points in image"""

    xmin, ymin, xrng, yrng = set_aspect(xl, yl, xres, yres)
    render = np.zeros((yres, xres))

    try:
        for x, y in zip(xl, yl):
            J = get_index(x, xmin, xrng, xres)
            I = get_index(y, ymin, yrng, yres)
            render[I, J] += 1
    except ValueError:
        print("Invalid value")
        return False

    return check_density(render)


def check_density(render, min_fill=1.5):
    """check if pixel density exceeds threshold"""
    filled_pixels = np.count_nonzero(render)
    fill_percentage = 100 * filled_pixels / np.size(render)
    if fill_percentage > min_fill:
        return True
    return False


def time_this(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{elapsed:.2f}s")
        return result

    return wrapper
