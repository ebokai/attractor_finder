from search import search_attractors
from render import render_attractors as renderer
from compute import compute_attractors
from iterator_cubic import iteration_cubic

import numpy as np 
import time 
import json

if __name__ == "__main__":

	save_data = False

	search_iterates = 2000
	render_iterates = 50_000_000
	render_check_ratio = 0.01
	n_attractors = 100
	alpha = 0.025


	with open('print_sizes.json', 'r') as f:
		print_sizes = json.load(f)
	xres, yres = print_sizes['A4']

	for i in range(n_attractors):

		start = time.time()

		dimension = np.random.randint(2,7)

		coeffs, seed = search_attractors(search_iterates, dimension)
		itdata, error = compute_attractors(coeffs, render_iterates, render_check_ratio, dimension)

		if save_data:
			np.save(f'd_{dimension}_{seed}_arr', itdata)

		if not error:

			renderer(
				itdata[10000:, dimension - 3], 
				itdata[10000:, dimension - 2], 
				itdata[10000:, dimension - 1], 
				coeffs, dimension, seed, 'xyz', alpha = alpha, xres = xres, yres = yres)

		print(f'Total time: {time.time()-start:.2f} seconds')


