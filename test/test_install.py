from attractor_finder import search_attractor

import numpy as np

def test_search_seed():
	coeffs1, _ = search_attractor(dimension = 2, seed = 1)
	coeffs2, _ = search_attractor(dimension = 2, seed = 1)
	assert np.allclose(coeffs1, coeffs2)


