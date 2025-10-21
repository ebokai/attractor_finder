import time
import numpy as np

from attractor_finder.iterator import iterator
from concurrent.futures import ProcessPoolExecutor

def compute_attractor_single_thread(coeffs, render_iterates, dimension, render_check_ratio = 0.01):

    check_index = int(render_iterates * render_check_ratio)
    itdata = np.asarray(iterator(check_index,coeffs,dimension))

    if np.isnan(itdata[-1,-1]) or np.isinf(itdata[-1,-1]):
        print('Error during calculation\n')
        error = True
    else:
        start = time.perf_counter()
        itdata = np.asarray(iterator(render_iterates,coeffs,dimension))
        end = time.perf_counter()
        iteration_time = end - start

        print(" Iteration")
        print("────────────────────────────────────────────")
        print(f"• Duration:         {iteration_time:.1f} s")
        print(f"• Rate:             {render_iterates/iteration_time/1e6:.2f} M it/s\n")

        error = False

    return itdata, error

def worker(args):
    thread_iterates, coeffs, dimension = args
    return np.asarray(iterator(thread_iterates, coeffs, dimension))

def compute_attractor(coeffs, render_iterates, dimension, render_check_ratio = 0.01, n_processes = 6):

    check_index = int(render_iterates * render_check_ratio)
    itdata = np.asarray(iterator(check_index, coeffs, dimension))

    if np.isnan(itdata[-1,-1]) or np.isinf(itdata[-1,-1]):
        print(' Error during calculation\n')
        return None, True
        
    thread_iterates = render_iterates // n_processes
    args_list = [(thread_iterates, coeffs, dimension) for _ in range(n_processes)]

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


