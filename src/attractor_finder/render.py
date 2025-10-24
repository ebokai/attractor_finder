from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from attractor_finder.functions import set_aspect, time_this
from attractor_finder.functions_numba import (
    get_dx_numba_parallel, get_max_numba, get_min_max_range_numba)
from attractor_finder.renderer_batch import compute_burn, compute_burn_optimized, compute_render_slice
from attractor_finder.renderer import render_pixels


def burn_worker(args):
    return np.asarray(compute_burn(*args))

def pixel_worker(args):
    return np.asarray(compute_render_slice(*args))

class AttractorRenderPipeline():

    def __init__(self, data, xres, yres, alpha = 0.025):

        self.dimension = (data.shape)[1]

        self._xa = np.asarray(data[10000:, self.dimension - 3])
        self._ya = np.asarray(data[10000:, self.dimension - 2])
        self._za = np.asarray(data[10000:, self.dimension - 1])

        self.xres = xres
        self.yres = yres
        self.alpha = alpha

        self._bgcolor = np.asarray([0.9,0.9,0.85])
        self._burn_factors = np.asarray([0.75,1.00,1.25])

        self.n_processes = 6

        self._args_list_burn = None
        self._args_list_pixel = None
        self._args_list_one_pass = None
        self._full_burn = None
        self._render = None

        self.get_bounds()
        if not self._xnan:
            self.compute_deltas()

    @time_this
    def get_bounds(self):
        """
        rescale bounding box to print size
        """
        print('... get_bounds', end=" ")
        xmin, ymin, xrng, yrng = set_aspect(self._xa, self._ya, self.xres, self.yres)
        zmin, zrng = get_min_max_range_numba(self._za)
        self._bounds = {
        'xmin': xmin, 'ymin': ymin, 'zmin': zmin, 
        'xrng': xrng, 'yrng': yrng, 'zrng': zrng
        }

        self._xnan = np.isnan(xrng)

    @time_this
    def compute_deltas(self):
        """
        compute delta = x[i+1]-x[i] for each coordinate array
        get maximum delta for each coordinate (used in pixel colors)
        """
        print('... compute_deltas', end=" ")
        self._dx = get_dx_numba_parallel(self._xa)
        self._dy = get_dx_numba_parallel(self._ya)
        self._dz = get_dx_numba_parallel(self._za)
        max_dx = get_max_numba(self._dx)
        max_dy = get_max_numba(self._dy)
        max_dz = get_max_numba(self._dz)
        self._max_deltas = np.asarray([max_dx, max_dy, max_dz])

    @time_this
    def construct_args_list_burn(self):
        """
        construct argument list to pass to burn_worker
        """
        print('... construct_args_list_burn', end=" ")
        self._args_list_burn = []
        it_ranges = np.linspace(1, len(self._xa[1:]), self.n_processes + 1)
        it_ranges = it_ranges.astype(int)

        for i in range(self.n_processes):
            i0, i1 = it_ranges[i], it_ranges[i+1]
            args = (
                self.xres,
                self.yres,
                self._xa[i0:i1],
                self._ya[i0:i1],
                self._za[i0:i1],
                self._dx[i0+1:i1+1],
                self._dy[i0+1:i1+1],
                self._dz[i0+1:i1+1],
                self._bounds['xrng'],
                self._bounds['xmin'],
                self._bounds['yrng'],
                self._bounds['ymin'],
                self._bounds['zrng'],
                self._bounds['zmin'],
                self.alpha,
                self._max_deltas,
                self._burn_factors
                )
            self._args_list_burn.append(args)

    @time_this
    def construct_args_list_pixel(self):
        """
        construct argument list to pass to pixel_worker
        """
        print('... construct_args_list_pixel', end=" ")
        y_slice = np.linspace(0, self.yres, self.n_processes + 1)
        y_slice = y_slice.astype(int)
        self._args_list_pixel = [(
            self.xres,
            self.yres,
            y_slice[i],
            y_slice[i+1],
            self._bgcolor,
            self._full_burn) for i in range(self.n_processes)]

    @time_this
    def construct_args_list_one_pass(self):
        """
        construct argument list to pass to render_pixels (one-pass render)
        """
        self._args_list_one_pass = (
            self.xres,
            self.yres,
            self._xa[1:],
            self._ya[1:],
            self._za[1:],
            self._dx,
            self._dy,
            self._dz,
            self._bounds['xrng'],
            self._bounds['xmin'],
            self._bounds['yrng'],
            self._bounds['ymin'],
            self._bounds['zrng'],
            self._bounds['zmin'],
            self.alpha,
            self._bgcolor,
            self._burn_factors)

    @time_this
    def burn_pool(self):
        """
        compute burn factors which are used to darken pixels
        uses multiprocessing
        """
        print('... burn_pool', end=" ")
        with ProcessPoolExecutor(max_workers = self.n_processes) as executor:
            burn_batch = list(executor.map(burn_worker, self._args_list_burn))
        self._full_burn = np.ones((self.yres, self.xres, 3))
        for batch in burn_batch:
            self._full_burn *= batch

    @time_this
    def pixel_pool(self):
        print('... pixel_pool', end=" ")
        with ProcessPoolExecutor(max_workers = self.n_processes) as executor:
            pixel_batch = list(executor.map(pixel_worker, self._args_list_pixel))
        self._render = np.zeros((self.yres, self.xres, 3))
        for batch in pixel_batch:
            self._render += batch
        self._render = np.clip(self._render, 0, 1)

    @time_this
    def render_one_pass(self):
        self._render = np.asarray(render_pixels(*self._args_list_one_pass))

    def _multi_pass_render(self, n_processes):
        print('... multi_pass_render')
        self.n_processes = n_processes
        self.construct_args_list_burn()
        self.burn_pool()
        self.construct_args_list_pixel()
        self.pixel_pool()

    def _one_pass_render(self):
        print('... one_pass_render')
        self.construct_args_list_one_pass()
        self.render_one_pass()

    @time_this
    def _save_image(self, seed):
        print('... save_image', end=" ")
        output_dir = Path(__file__).parents[2] / "output" 
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f'D{self.dimension}-{seed}.png'
        plt.imsave(fname, self._render, dpi=300)

    def render_attractor(self, seed, multi = True, n_processes = 6):
        if multi:
            self._multi_pass_render(n_processes)
        else:
            self._one_pass_render()
        self._save_image(seed)
