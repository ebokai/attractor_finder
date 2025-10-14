import numpy as np 
import matplotlib.pyplot as plt
from functions import *
from renderer import render_pixels
from batch_renderer import compute_burn, compute_render_slice
from numpy_renderer import compute_burn_numpy

from concurrent.futures import ProcessPoolExecutor

import time 

def burn_worker(args):
	xres, yres, xa, ya, za, dx, dy, dz, xrng, xmin, yrng, ymin, zrng, zmin, alpha, max_ds, burn_factors = args
	return np.asarray(compute_burn(xres, yres, xa, ya, za, dx, dy, dz, xrng, xmin, yrng, ymin, zrng, zmin, alpha, max_ds, burn_factors))

def render_worker(args):
	xres, yres, ymin, ymax, bgcolor, burn_factor = args
	return np.array(compute_render_slice(xres, yres, ymin, ymax, bgcolor, burn_factor))

def render_attractors(xl, yl, zl, coeff, dimension, seed, tag, alpha = 0.0075, xres = 3200, yres = 1800, n_processes = 6):

	xa = np.asarray(xl)
	ya = np.asarray(yl)
	za = np.asarray(zl)

	xmin, ymin, xrng, yrng, xdr, ydr = set_aspect(xa, ya, 
		xres, yres, debug=True)
	zmin, zrng = get_minmax_rng(za)

	bgcolor = np.array([0.9,0.9,0.85])
	burn_factors = np.array([0.75,1.00,1.25])

	n_iterates = np.size(xa[1:])

	if not np.isnan(xrng):

		print('Calculating pixel values')
		start = time.time()

		dxs = get_dx(xl)
		dys = get_dx(yl)
		dzs = get_dx(zl)

		max_dx = dxs.max()
		max_dy = dys.max()
		max_dz = dzs.max()

		max_ds = np.array([max_dx, max_dy, max_dz])
		print(f'Calculated difference arrays: {time.time()-start:.1f} seconds')

		if n_iterates > 10_000_000:

			print('Rendering using multi-pass rendering')
			burn_start = time.time()
			multi_start = time.time()
			
			s = np.linspace(1,n_iterates,n_processes+1).astype(int)
			args_list_burn = [(xres, yres, xa[s[i]:s[i+1]], ya[s[i]:s[i+1]], za[s[i]:s[i+1]], 
				dxs[s[i]-1:s[i+1]-1], dys[s[i]-1:s[i+1]-1], dzs[s[i]-1:s[i+1]-1],
				xrng, xmin, yrng, ymin, zrng, zmin, alpha, max_ds, burn_factors) for i in range(n_processes)]

			with ProcessPoolExecutor(max_workers = n_processes) as executor:
				burn_batch = list(executor.map(burn_worker, args_list_burn))
			full_burn = np.ones((yres, xres, 3))
			for batch in burn_batch:
				full_burn *= batch
			print(f'Calculated burn values: {time.time()-burn_start:.1f} seconds')

			pixel_start = time.time()

			y_slice = np.linspace(0,yres,n_processes+1).astype(int)
			args_list_render = [(xres, yres, y_slice[i], y_slice[i+1], bgcolor, full_burn) for i in range(n_processes)]

			with ProcessPoolExecutor(max_workers = n_processes) as executor:
				render_batch = list(executor.map(render_worker, args_list_render))
			render = np.zeros((yres, xres, 3))
			for batch in render_batch:
				render += batch
			render = np.clip(render, 0, 1)
			print(f'Calculated pixel colors: {time.time()-pixel_start:.1f} seconds')
			print(f'Multi-pass rendering: {time.time()-multi_start:.1f} seconds')

		else:
			print('Rendering using one-pass rendering')
			render_start = time.time()
			render = np.asarray(render_pixels(xres,yres,xa[1:],ya[1:],za[1:],dxs,dys,dzs,xrng,xmin,yrng,ymin,zrng,zmin,alpha,bgcolor,burn_factors))
			print(f'One-pass rendering: {time.time()-render_start:.1f} seconds')


		fname = f'render/D{dimension}-{seed}-{tag}.png'
		plt.imsave(fname, render, dpi=300)
		print('Saved ' + fname)
		print(f'Total rendering time: {time.time()-start:.1f} seconds')
