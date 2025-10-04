from search import search_attractors
from compute import compute_attractors
from render import render_attractors as renderer
from print_sizes import print_sizes

import numpy as np 
import time 


save_data = False

search_iterates = 2000
render_iterates = 10000000
render_check_ratio = 0.01
n_attractors = 100
alpha = 0.025

xres, yres = print_sizes['A4']

for i in range(n_attractors):

	start = time.time()

	dimension = np.random.randint(2,8)

	coeffs, seed = search_attractors(search_iterates, dimension)
	itdata, error = compute_attractors(coeffs, render_iterates, render_check_ratio, dimension)

	if save_data:
		np.save(f'd_{dimension}_{seed}_arr', itdata)

	if not error:

		renderer(
			itdata[10000:, dimension - 3], 
			itdata[10000:, dimension - 2], 
			itdata[10000:, dimension - 1], 
			coeffs, dimension, seed, 'xyz-v3', alpha = alpha, xres = xres, yres = yres)


	print(f'Total time: {time.time()-start:.2f} seconds')


