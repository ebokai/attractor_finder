from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from attractor_finder.functions import set_aspect, time_this
from attractor_finder.functions_numba import (
    get_dx_numba_parallel, get_max_numba, get_min_max_range_numba)
from attractor_finder.renderer_batch import compute_burn, compute_render_slice
from attractor_finder.renderer import render_pixels


def burn_worker(args):
    return np.asarray(compute_burn(*args))

def pixel_worker(args):
    return np.asarray(compute_render_slice(*args))

class AttractorRenderPipeline():

    def __init__(self, data, xres, yres, dimension, used_seed = 0, alpha = 0.025):

        self.xa = np.asarray(data[10000:, dimension - 3])
        self.ya = np.asarray(data[10000:, dimension - 2])
        self.za = np.asarray(data[10000:, dimension - 1])
        
        self.dimension = dimension
        self.seed = used_seed

        self.xres = xres 
        self.yres = yres
        self.alpha = alpha
        self.bgcolor = np.asarray([0.9,0.9,0.85])
        self.burn_factors = np.asarray([0.75,1.00,1.25])

        self.n_processes = 6

        self.get_bounds()
        if not self.xnan:
            self.compute_deltas()
    
    @time_this
    def get_bounds(self):
        print('... get_bounds', end=" ")
        xmin, ymin, xrng, yrng = set_aspect(self.xa, self.ya, self.xres, self.yres)
        zmin, zrng = get_min_max_range_numba(self.za)
        self.bounds = {
        'xmin': xmin, 'ymin': ymin, 'zmin': zmin, 
        'xrng': xrng, 'yrng': yrng, 'zrng': zrng
        }

        self.xnan = np.isnan(xrng)

    @time_this
    def compute_deltas(self):
        print('... compute_deltas', end=" ")
        self.dx = get_dx_numba_parallel(self.xa)
        self.dy = get_dx_numba_parallel(self.ya)
        self.dz = get_dx_numba_parallel(self.za)
        max_dx = get_max_numba(self.dx)
        max_dy = get_max_numba(self.dy)
        max_dz = get_max_numba(self.dz)
        self.max_deltas = np.asarray([max_dx, max_dy, max_dz])

    @time_this
    def construct_args_list_burn(self):
        print('... construct_args_list_burn', end=" ")
        self.args_list_burn = []
        it_ranges = np.linspace(1, len(self.xa[1:]), self.n_processes + 1)
        it_ranges = it_ranges.astype(int)

        for i in range(self.n_processes):
            i0, i1 = it_ranges[i], it_ranges[i+1]
            args = (
                self.xres, 
                self.yres,
                self.xa[i0:i1],
                self.ya[i0:i1],
                self.za[i0:i1],
                self.dx[i0+1:i1+1],
                self.dy[i0+1:i1+1],
                self.dz[i0+1:i1+1],
                self.bounds['xrng'],
                self.bounds['xmin'],
                self.bounds['yrng'],
                self.bounds['ymin'],
                self.bounds['zrng'],
                self.bounds['zmin'],
                self.alpha,
                self.max_deltas,
                self.burn_factors
                )
            self.args_list_burn.append(args)

    @time_this
    def construct_args_list_pixel(self):
        print('... construct_args_list_pixel', end=" ")
        y_slice = np.linspace(0, self.yres, self.n_processes + 1)
        y_slice = y_slice.astype(int)
        self.args_list_pixel = [(self.xres, self.yres, y_slice[i], y_slice[i+1], self.bgcolor, self.full_burn) for i in range(self.n_processes)]


    @time_this
    def burn_pool(self):
        print('... burn_pool', end=" ")
        with ProcessPoolExecutor(max_workers = self.n_processes) as executor:
            burn_batch = list(executor.map(burn_worker, self.args_list_burn))
        self.full_burn = np.ones((self.yres, self.xres, 3))
        for batch in burn_batch:
            self.full_burn *= batch

    @time_this
    def pixel_pool(self):
        print('... pixel_pool', end=" ")
        with ProcessPoolExecutor(max_workers = self.n_processes) as executor:
            pixel_batch = list(executor.map(pixel_worker, self.args_list_pixel))
        self.render = np.zeros((self.yres, self.xres, 3))
        for batch in pixel_batch:
            self.render += batch
        self.render = np.clip(self.render, 0, 1)

    @time_this
    def render_one_pass(self):
        self.render = np.asarray(render_pixels(*self.args_list_one_pass))

    @time_this
    def construct_args_list_one_pass(self):
        self.args_list_one_pass = (
            self.xres,
            self.yres,
            self.xa[1:],
            self.ya[1:],
            self.za[1:],
            self.dx,
            self.dy,
            self.dz,
            self.bounds['xrng'],
            self.bounds['xmin'],
            self.bounds['yrng'],
            self.bounds['ymin'],
            self.bounds['zrng'],
            self.bounds['zmin'],
            self.alpha,
            self.bgcolor,
            self.burn_factors)


    def multi_pass_render(self, n_processes):
        print('... multi_pass_render')
        self.n_processes = n_processes
        self.construct_args_list_burn()
        self.burn_pool()
        self.construct_args_list_pixel()
        self.pixel_pool()

    def one_pass_render(self):
        print('... one_pass_render')
        self.construct_args_list_one_pass()
        self.render_one_pass()

    @time_this
    def save_image(self):
        print('... save_image', end=" ")
        output_dir = Path(__file__).parents[2] / "output" 
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f'D{self.dimension}-{self.seed}.png'
        plt.imsave(fname, self.render, dpi=300)

    def render_attractor(self, multi = True, n_processes = 6):
        if multi:
            self.multi_pass_render(n_processes)
        else:
            self.one_pass_render()
        self.save_image()
