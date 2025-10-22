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
    xres, yres, xa, ya, za, dx, dy, dz, xrng, xmin, yrng, ymin, zrng, zmin, alpha, max_ds, burn_factors = args
    return np.asarray(compute_burn(xres, yres, xa, ya, za, dx, dy, dz, xrng, xmin, yrng, ymin, zrng, zmin, alpha, max_ds, burn_factors))

def render_worker(args):
    xres, yres, ymin, ymax, bgcolor, burn_factor = args
    return np.array(compute_render_slice(xres, yres, ymin, ymax, bgcolor, burn_factor))

def compute_histogram(full_burn, dimension, seed):
    burn_values = np.ravel(full_burn[:,:,0])
    burn_values = burn_values[np.isfinite(burn_values)]
    counts, _ = np.histogram(burn_values, bins = 100)
    return np.all(counts >= 1e3)



def render_attractor(xl, yl, zl, coeff, dimension, seed, tag, alpha = 0.0075, xres = 3200, yres = 1800, n_processes = 6):

    xa = np.asarray(xl)
    ya = np.asarray(yl)
    za = np.asarray(zl)

    xmin, ymin, xrng, yrng = set_aspect(xa, ya, xres, yres, debug=True)
    zmin, zrng = get_min_max_range_numba(za)


    bgcolor = np.array([0.9,0.9,0.85])
    burn_factors = np.array([0.75,1.00,1.25])


    n_iterates = np.size(xa[1:])

    if not np.isnan(xrng):

        print(" Rendering")
        print("────────────────────────────────────────────")
        start = time.perf_counter()

        dxs = get_dx_numba_parallel(xl)
        dys = get_dx_numba_parallel(yl)
        dzs = get_dx_numba_parallel(zl)

        max_dx = get_max_numba(dxs)
        max_dy = get_max_numba(dys)
        max_dz = get_max_numba(dzs)

        max_ds = np.array([max_dx, max_dy, max_dz])
        print(f"• Difference Arrays:   {time.perf_counter()-start:.1f} s")

        if n_iterates > 10_000_000:
            burn_start = time.perf_counter()
            multi_start = time.perf_counter()
            
            s = np.linspace(1,n_iterates,n_processes+1).astype(int)
            args_list_burn = [(xres, yres, xa[s[i]:s[i+1]], ya[s[i]:s[i+1]], za[s[i]:s[i+1]], 
                dxs[s[i]-1:s[i+1]-1], dys[s[i]-1:s[i+1]-1], dzs[s[i]-1:s[i+1]-1],
                xrng, xmin, yrng, ymin, zrng, zmin, alpha, max_ds, burn_factors) for i in range(n_processes)]

            with ProcessPoolExecutor(max_workers = n_processes) as executor:
                burn_batch = list(executor.map(burn_worker, args_list_burn))
            full_burn = np.ones((yres, xres, 3))
            for batch in burn_batch:
                full_burn *= batch
            print(f"• Burn Values:         {time.perf_counter()-burn_start:.1f} s")

            # if not (compute_histogram(full_burn, dimension, seed)):
            #     print("────────────────────────────────────────────")
            #     print('Sparse histogram - skipping rendering')
            #     return

            pixel_start = time.perf_counter()

            y_slice = np.linspace(0,yres,n_processes+1).astype(int)
            args_list_render = [(xres, yres, y_slice[i], y_slice[i+1], bgcolor, full_burn) for i in range(n_processes)]

            with ProcessPoolExecutor(max_workers = n_processes) as executor:
                render_batch = list(executor.map(render_worker, args_list_render))
            render = np.zeros((yres, xres, 3))
            for batch in render_batch:
                render += batch
            render = np.clip(render, 0, 1)
            print(f"• Pixel Colors:        {time.perf_counter()-pixel_start:.1f} s")
            print(f"• Multi-Pass Render:   {time.perf_counter()-multi_start:.1f} s")

        else:
            print('Rendering using one-pass rendering')
            render_start = time.perf_counter()
            render = np.asarray(render_pixels(xres,yres,xa[1:],ya[1:],za[1:],dxs,dys,dzs,xrng,xmin,yrng,ymin,zrng,zmin,alpha,bgcolor,burn_factors))
            print(f"• One-Pass Render:     {time.perf_counter()-render_start:.1f} s")


        output_dir = Path(__file__).parents[2] / "output" 
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f'D{dimension}-{seed}-{tag}.png'
        
        plt.imsave(fname, render, dpi=300)
        print(f"• Output:              {fname}\n")
        print("────────────────────────────────────────────")
        print(f" Total Render Time:    {time.perf_counter()-start:.1f} s")

