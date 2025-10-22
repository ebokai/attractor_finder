import numpy as np
import time 

from attractor_finder.iterator import iterator
from attractor_finder.functions import pixel_density

def search_attractor(dimension, search_iterates = 2000, seed = None):

    start = time.perf_counter()

    found = False

    d = dimension

    ncoeffs = int(d + 11/6 * d**2 + d**3 + d**4/6)

    if seed is None:
        seed = np.random.randint(1, 2e9)
    np.random.seed(seed)

    while not found:
                
        coeffs = np.random.randint(-10, 11, ncoeffs)/(10 + 2 * dimension)
        x0 = np.random.uniform(-1e-1, 1e-1, (dimension + 1))
        itdata = np.asarray(iterator(search_iterates, coeffs, x0, dimension))

        test = itdata[-1,-1]

        if np.isnan(test) or np.isinf(test):
            out_of_bounds = True
        else:
            out_of_bounds = False

        if not out_of_bounds:
            xa = itdata[:,0]
            ya = itdata[:,1]
            if pixel_density(xa,ya):
                print("────────────────────────────────────────────")
                print(" Attractor Found")
                print("────────────────────────────────────────────")
                print(f"• Dimension:        {d}")
                print(f"• Seed:             {seed}")
                print(f"• Discovery Time:   {time.perf_counter()-start:.1f} s\n")
                found = True
            
    return coeffs, seed
