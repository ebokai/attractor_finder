#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np


def render_pixels(int xres, int yres, 
	double[:] xa, double[:] ya, double[:] za, 
	double[:] dxs, double[:] dys, double[:] dzs,
	double xrng, double xmin, double yrng, double ymin, 
	double zrng, double zmin, double alpha):

    cdef double[:,:,:] render = np.zeros((yres, xres, 3))
    cdef int length = np.size(xa)
    cdef int I, J
    cdef double z_alpha
    cdef double mdx = max(dxs)
    cdef double mdy = max(dys)
    cdef double mdz = max(dzs)
    cdef double x, y, z

    bgcolor = [0.9,0.9,0.85]

    # Fill background
    for I in range(yres):
        for J in range(xres):
            render[I,J,0] = bgcolor[0]
            render[I,J,1] = bgcolor[1]
            render[I,J,2] = bgcolor[2]

    # Draw attractor points in black with low alpha
    for i in range(length):
        x = xa[i]
        y = ya[i]
        z = za[i]

        J = <int>((x-xmin)/xrng * (xres-1))
        I = <int>((y-ymin)/yrng * (yres-1))

        z_alpha = 0.1 + 0.9*(z-zmin)/zrng  # scale alpha slightly with z

        # Additive darkening toward black (burn effect)
        render[I,J,0] -= alpha * z_alpha * 0.5
        render[I,J,1] -= alpha * z_alpha * 0.8
        render[I,J,2] -= alpha * z_alpha * 1.1

    return render
