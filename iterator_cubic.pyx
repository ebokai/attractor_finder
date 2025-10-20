#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np 


def iteration_cubic(int n_iterations, double[:] coeffs, int dimension):

	cdef double fsum = 0

	cdef int t 
	cdef int m 
	cdef int n 
	cdef int i
	cdef int j 
	cdef int k 

	cdef double[:] coords = np.random.uniform(-1e-1, 1e-1, (dimension + 1))
	cdef double[:] sums = np.zeros(dimension)
	cdef double[:,:] itdata = np.zeros((n_iterations, dimension))

	coords[0] = 1

	for t in range(n_iterations):

		n = 0

		for m in range(dimension):

			fsum = 0

			for i in range(dimension + 1):
				for j in range(i, dimension + 1):
					for k in range(j, dimension + 1):

						fsum = fsum + coeffs[n] * coords[i] * coords[j] * coords[k]

						n += 1

			sums[m] = fsum

		coords[1:] = sums
		itdata[t,:] = coords[1:]

	return itdata






