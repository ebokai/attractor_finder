from attractor_finder import search_attractor, compute_attractor, render_attractor
from attractor_finder.iterator import iterator
from pathlib import Path

import numpy as np 
import time 
import json

if __name__ == "__main__":

	save_data = False


	render_iterates = 1_000_000
	n_attractors = 100
	alpha = 0.025


	
	data_path = Path(__file__).parents[1] / "data" / "print_sizes.json"
	with open(data_path, 'r') as f:
		print_sizes = json.load(f)

	xres, yres = print_sizes['A4']

	for i in range(n_attractors):

		start = time.perf_counter()

		dimension = np.random.randint(2,4)

		coeffs, seed = search_attractor(dimension)
		itdata, error = compute_attractor(coeffs, render_iterates, dimension)

		if save_data:
			np.save(f'd_{dimension}_{seed}_arr', itdata)

		if not error:

			render_attractor(
				itdata[10000:, dimension - 3], 
				itdata[10000:, dimension - 2], 
				itdata[10000:, dimension - 1], 
				coeffs, dimension, seed, 'xyz', alpha = alpha, xres = xres, yres = yres)

		print(f" Total Runtime:        {time.perf_counter()-start:.1f} s")
		print("────────────────────────────────────────────\n")


