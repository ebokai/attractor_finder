import numpy as np 
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from attractor_finder.functions import set_aspect
from attractor_finder.functions_numba import get_dx_numba_parallel, get_max_numba, get_min_max_range_numba
from attractor_finder.renderer_batch import compute_burn, compute_render_slice
from attractor_finder.renderer import render_pixels

import time 

def burn_worker(args):
    return np.asarray(compute_burn(*args))

def render_worker(args):
    return np.asarray(compute_render_slice(*args))

def construct_burn_args(
    xres, yres, xa, ya, za, dx, dy, dz, 
    bounds, max_deltas, alpha, n_processes, 
    burn_factors = np.array([0.75, 1.00, 1.25])):

    n_iterates = np.size(xa[1:])
    it_ranges = np.linspace(1, n_iterates, n_processes + 1).astype(int)
    xmin, ymin, zmin, xrng, yrng, zrng = bounds

    args_list_burn = []
    for i in range(n_processes):
        i0, i1 = it_ranges[i], it_ranges[i+1]
        args = (
            xres, yres, 
            xa[i0:i1], ya[i0:i1], za[i0:i1],
            dx[i0-1:i1-1], dy[i0-1:i1-1], dz[i0-1:i1-1],
            xrng, xmin, yrng, ymin, zrng, zmin,
            alpha, max_deltas, burn_factors)
        args_list_burn.append(args)

    return args_list_burn

def construct_render_args(
    xres, yres, 
    full_burn, n_processes, 
    bgcolor = np.array([0.90, 0.90, 0.85])):

    y_slice = np.linspace(0, yres, n_processes+1).astype(int)
    args_list_render = [(xres, yres, y_slice[i], y_slice[i+1], bgcolor, full_burn) for i in range(n_processes)]
    return args_list_render

def construct_render_args_one_pass(
    xres, yres, 
    xa, ya, za, dx, dy, dz, 
    bounds, alpha, 
    bgcolor = np.array([0.90, 0.90, 0.85]), 
    burn_factors = np.array([0.75, 1.00, 1.25])):

    xmin, ymin, zmin, xrng, yrng, zrng = bounds
    args_list_render_one_pass = (xres, yres, xa[1:], ya[1:], za[1:], dx, dy, dz, xrng, xmin, yrng, ymin, zrng, zmin, alpha, bgcolor, burn_factors)
    return args_list_render_one_pass


def compute_deltas(xa, ya, za):
    start = time.perf_counter()
    dx = get_dx_numba_parallel(xa)
    dy = get_dx_numba_parallel(ya)
    dz = get_dx_numba_parallel(za)

    max_dx = get_max_numba(dx)
    max_dy = get_max_numba(dy)
    max_dz = get_max_numba(dz)

    max_deltas = np.array([max_dx, max_dy, max_dz])
    print(f"• Difference Arrays:   {time.perf_counter()-start:.1f} s")

    return dx, dy, dz, max_deltas

def get_bounds(xa, ya, za, xres, yres):
    xmin, ymin, xrng, yrng = set_aspect(xa, ya, xres, yres, debug=True)
    zmin, zrng = get_min_max_range_numba(za)
    return [xmin, ymin, zmin, xrng, yrng, zrng], np.isnan(xrng)

def burn_pool(args_list_burn, n_processes, xres, yres):
    burn_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers = n_processes) as executor:
        burn_batch = list(executor.map(burn_worker, args_list_burn))

    full_burn = np.ones((yres, xres, 3))
    for batch in burn_batch:
        full_burn *= batch

    print(f"• Burn Values:         {time.perf_counter()-burn_start:.1f} s")
    return full_burn

def render_pool(args_list_render, n_processes, xres, yres):
    pixel_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers = n_processes) as executor:
        render_batch = list(executor.map(render_worker, args_list_render))

    render = np.zeros((yres, xres, 3))
    for batch in render_batch:
        render += batch
    render = np.clip(render, 0, 1)

    print(f"• Pixel Colors:        {time.perf_counter()-pixel_start:.1f} s")
    return render

def render_one_pass(args_list_render_one_pass):
    render_start = time.perf_counter()
    render = np.asarray(render_pixels(*args))
    print(f"• One-Pass Render:     {time.perf_counter()-render_start:.1f} s")
    return render

def save_image(render, dimension, seed):
    output_dir = Path(__file__).parents[2] / "output" 
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f'D{dimension}-{seed}.png'
    plt.imsave(fname, render, dpi=300)
    print(f"• Output:              {fname}\n")


def render_attractor(xl, yl, zl, dimension, seed, alpha = 0.0075, xres = 3200, yres = 1800, n_processes = 6):

    xa = np.asarray(xl)
    ya = np.asarray(yl)
    za = np.asarray(zl)

    bounds, xnan = get_bounds(xa, ya, za, xres, yres)

    if not xnan:

        start = time.perf_counter()
        print(" Rendering")
        print("────────────────────────────────────────────")
        dx, dy, dz, max_deltas = compute_deltas(xa, ya, za)

        if len(xa) > 10_000_000:

            multi_start = time.perf_counter()

            args_list_burn = construct_burn_args(xres, yres, xa, ya, za, dx, dy, dz, bounds, max_deltas, alpha, n_processes)
            full_burn = burn_pool(args_list_burn, n_processes, xres, yres)

            args_list_render = construct_render_args(xres, yres, full_burn, n_processes)
            render = render_pool(args_list_render, n_processes, xres, yres)

            print(f"• Multi-Pass Render:   {time.perf_counter()-multi_start:.1f} s")

        else:
            args_list_render_one_pass = construct_render_args_one_pass(xres, yres, xa, ya, za, dx, dy, dz, bounds, alpha)
            render = render_one_pass(args_list_render_one_pass)

        save_image(render, dimension, seed)
        print("────────────────────────────────────────────")
        print(f" Total Render Time:    {time.perf_counter()-start:.1f} s")

