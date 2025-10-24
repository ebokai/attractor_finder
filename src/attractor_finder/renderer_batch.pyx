#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np


def compute_burn(int xres, int yres, 
    double[:] xa, double[:] ya, double[:] za, 
    double[:] dxs, double[:] dys, double[:] dzs,
    double xrng, double xmin, double yrng, double ymin, 
    double zrng, double zmin, double alpha, double[:] max_deltas, double[:] burn_factors):

    cdef double[:,:,:] render = np.ones((yres, xres, 3))
    cdef int length = np.size(xa)
    cdef int I, J
    cdef double z_alpha
    cdef double mdx = max_deltas[0]
    cdef double mdy = max_deltas[1]
    cdef double mdz = max_deltas[2]
    cdef double x, y, z

    bfr = burn_factors[0]
    bfg = burn_factors[1]
    bfb = burn_factors[2]

    # Draw attractor points in black with low alpha
    for i in range(length):
        x = xa[i]
        y = ya[i]
        z = za[i]
        dx = dxs[i]
        dy = dys[i]
        dz = dzs[i]

        J = <int>((x - xmin) / xrng * (xres - 1))
        I = <int>((y - ymin) / yrng * (yres - 1))

        z_alpha = 0.1 + 0.9 * (z - zmin) / zrng  # scale alpha slightly with z
        
        burn_factor_r = alpha * z_alpha * (1 + dx / mdx) * bfr
        burn_factor_g = alpha * z_alpha * (1 + dy / mdy) * bfg
        burn_factor_b = alpha * z_alpha * (1 + dz / mdz) * bfb

        # Multiplicative burn (scale toward black)
        render[I,J,0] *= (1 - burn_factor_r * render[I,J,0])
        render[I,J,1] *= (1 - burn_factor_g * render[I,J,1])
        render[I,J,2] *= (1 - burn_factor_b * render[I,J,2])

    return render

def compute_burn_optimized(int xres, int yres, 
    int[:] Is, int[:] Js, double[:] z_alphas, 
    double[:] dxm, double[:] dym, double[:] dzm,
    double alpha, double[:] burn_factors):

    cdef double[:,:,:] render = np.ones((yres, xres, 3), dtype=np.float32)
    cdef int length = np.size(Is)
    cdef int I, J
    cdef double z, z_alpha
    cdef double dx, dy, dz
    cdef double burn_factor_r, burn_factor_g, burn_factor_b
    cdef double r, g, b
    cdef double bfr, bfg, bfb
    cdef double abfr, abfg, abfb

    bfr = burn_factors[0]
    bfg = burn_factors[1]
    bfb = burn_factors[2]

    abfr = alpha * bfr 
    abfg = alpha * bfg 
    abfb = alpha * bfb

    # Draw attractor points in black with low alpha
    for i in range(length):

        dx = dxm[i]
        dy = dym[i]
        dz = dzm[i]

        I = Is[i]
        J = Js[i]

        z_alpha = 0.1 + 0.9 * z_alphas[i]  # scale alpha slightly with z
        
        burn_factor_r = z_alpha * (1 + dx) * abfr
        burn_factor_g = z_alpha * (1 + dy) * abfg
        burn_factor_b = z_alpha * (1 + dz) * abfb

        r = render[I,J,0]
        g = render[I,J,1]
        b = render[I,J,2]

        # Multiplicative burn (scale toward black)
        render[I,J,0] = r * (1 - burn_factor_r * r)
        render[I,J,1] = g * (1 - burn_factor_g * g)
        render[I,J,2] = b * (1 - burn_factor_b * b)

    return render

def compute_render_slice(int xres, int yres, int ymin, int ymax, double[:] bgcolor, double[:,:,:] burn_array):

    cdef double[:,:,:] render = np.zeros((yres, xres, 3))
    cdef int x, y

    for x in range(xres):
        for y in range(ymin, ymax):
            for k in range(3):
                render[y,x,k] = bgcolor[k] * burn_array[y,x,k]
                if render[y,x,k] > 1.0:
                    render[y,x,k] = 1.0
                elif render[y,x,k] < 0.0:
                    render[y,x,k] = 0.0

    return render
