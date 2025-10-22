import time
import numpy as np

from attractor_finder.iterator import iterator, iterator_optimized
from concurrent.futures import ProcessPoolExecutor

def compute_attractor_single_thread(coeffs, render_iterates, dimension, render_check_ratio = 0.01):

    check_index = int(render_iterates * render_check_ratio)
    x0 = np.random.uniform(-1e-1, 1e-1, (dimension + 1))
    itdata = np.asarray(iterator_optimized(check_index, coeffs, x0, dimension))

    if np.isnan(itdata[-1,-1]) or np.isinf(itdata[-1,-1]):
        print('Error during calculation\n')
        error = True
    else:
        start = time.perf_counter()
        itdata = np.asarray(iterator_optimized(render_iterates, coeffs, x0, dimension))
        end = time.perf_counter()
        iteration_time = end - start

        print(" Iteration")
        print("────────────────────────────────────────────")
        print(f"• Duration:         {iteration_time:.1f} s")
        print(f"• Rate:             {render_iterates/iteration_time/1e6:.2f} M it/s\n")

        error = False

    return itdata, error

def worker(args):
    thread_iterates, coeffs, x0, dimension = args
    return np.asarray(iterator_optimized(thread_iterates, coeffs, x0, dimension))

def compute_attractor(coeffs, render_iterates, dimension, render_check_ratio = 0.01, n_processes = 6):

    check_index = int(render_iterates * render_check_ratio)
    x0 = np.random.uniform(-1e-1, 1e-1, (dimension + 1))
    itdata = np.asarray(iterator_optimized(check_index, coeffs, x0, dimension))

    if np.isnan(itdata[-1,-1]) or np.isinf(itdata[-1,-1]):
        print(' Error during calculation\n')
        return None, True
        
    thread_iterates = render_iterates // n_processes
    x0_pool = np.random.uniform(-1e-1, 1e-1, (n_processes, dimension + 1))
    args_list = [(thread_iterates, coeffs, x0_pool[i], dimension) for i in range(n_processes)]

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers = n_processes) as executor:
        it_proc = list(executor.map(worker, args_list))
    end = time.perf_counter()
    iteration_time = end - start

    print(" Iteration Phase")
    print("────────────────────────────────────────────")
    print(f"• Duration:         {iteration_time:.1f} s")
    print(f"• Rate:             {render_iterates/iteration_time/1e6:.2f} M it/s\n")

    itdata = np.vstack(it_proc)

    return itdata, False


