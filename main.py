from search import search_attractors
from compute import compute_attractors
from render_v1 import render_attractors as render1
from render_v2 import render_attractors as render2
from render_v3 import render_attractors as render3
import numpy as np 
import time 

search_iterates = 2000
render_iterates = 10000000
render_check_ratio = 0.01
n_attractors = 100

alpha = 0.025

for i in range(n_attractors):

	start = time.time()

	dimension = np.random.randint(2,8)

	coeffs, seed = search_attractors(search_iterates, dimension)
	itdata, error = compute_attractors(coeffs, render_iterates, render_check_ratio, dimension)

	# np.save(f'd_{dimension}_{seed}_arr', itdata)

	if not error:

		# render1(
		# 	itdata[10000:, dimension - 3], 
		# 	itdata[10000:, dimension - 2], 
		# 	itdata[10000:, dimension - 1], 
		# 	coeffs, dimension, seed, 'xyz-v1', alpha = alpha)
		
		# render2(
		# 	itdata[10000:, dimension - 3], 
		# 	itdata[10000:, dimension - 2], 
		# 	itdata[10000:, dimension - 1], 
		# 	coeffs, dimension, seed, 'xyz-v2', alpha = alpha)

		render3(
			itdata[10000:, dimension - 3], 
			itdata[10000:, dimension - 2], 
			itdata[10000:, dimension - 1], 
			coeffs, dimension, seed, 'xyz-v3', alpha = alpha, xres = 2481, yres = 3508)


	print(f'Total time: {time.time()-start:.2f} seconds')


