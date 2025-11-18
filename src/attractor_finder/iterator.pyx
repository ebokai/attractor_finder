#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np

def iterator(int n_iterations, double[:] coeffs, double[:] x0, int dimension):

	"""
	parameters
	----------

	n_iterations : int
		number of iterations

	coeffs : np.ndarray of shape (num_coeffs,)
		attractor coefficients

	x0 : np.ndarray of shape (dimension + 1,)
		initial position vector

	dimension : int
		number of variables
	"""

	cdef double fsum = 0

	cdef int t
	cdef int m
	cdef int n
	cdef int i
	cdef int j
	cdef int k

	cdef double[:] coords = np.copy(x0)
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

def iterator_optimized(int n_iterations, double[:] coeffs, double[:] x0, int dimension):

	"""
	parameters
	----------

	n_iterations : int
		number of iterations

	coeffs : np.ndarray of shape (num_coeffs,)
		attractor coefficients

	x0 : np.ndarray of shape (dimension + 1,)
		initial position vector

	dimension : int
		number of variables
	"""

	cdef double fsum = 0

	cdef int t
	cdef int m
	cdef int n
	cdef int i
	cdef int j
	cdef int k

	cdef int d1 = dimension + 1

	cdef double[:] coords = np.copy(x0)
	cdef double[:] sums = np.zeros(dimension)
	cdef double[:,:] itdata = np.zeros((n_iterations, dimension))

	coords[0] = 1

	for t in range(n_iterations):

		n = 0

		for m in range(dimension):

			fsum = 0

			for i in range(d1):
				for j in range(i, d1):
					for k in range(j, d1):

						fsum = fsum + coeffs[n] * coords[i] * coords[j] * coords[k]

						n += 1

			sums[m] = fsum

		for i in range(dimension):
			coords[i + 1] = sums[i]
			itdata[t, i] = sums[i]

	return itdata
