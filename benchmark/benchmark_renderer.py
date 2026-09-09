import json
from pathlib import Path
from time import perf_counter

import numpy as np

from attractor_finder.functions import get_dx, get_min_max_range, set_aspect
from attractor_finder.functions_numba import (
    get_dx_numba,
    get_dx_numba_parallel,
    get_min_max_range_numba,
)
from attractor_finder.render import AttractorRenderPipeline


def benchmark_difference_arrays(data, func, n_runs=25):
    xa, ya, za = data[:, 0], data[:, 1], data[:, 2]
    print(f"• Running benchmark ({func.__name__})...", end=" ")
    times = []
    for _ in range(n_runs):
        start = perf_counter()
        dxs = func(xa)
        dys = func(ya)
        dzs = func(za)
        end = perf_counter()
        times.append(end - start)
    print("done.")

    avg_time = np.mean(times[1:])
    std_time = np.std(times[1:])

    print(f"• {n_runs} runs @ {avg_time:.3f} +/- {std_time:.3f} s/run")


def benchmark_aspect_ratio(data, xres, yres, n_runs=25):
    xa, ya, za = data[:, 0], data[:, 1], data[:, 2]
    print(f"• Running benchmark (set_aspect)...", end=" ")
    times = []
    for _ in range(n_runs):
        start = perf_counter()
        _ = set_aspect(xa, ya, xres, yres)
        end = perf_counter()
        times.append(end - start)
    print("done.")

    avg_time = np.mean(times[1:])
    std_time = np.std(times[1:])

    print(f"• {n_runs} runs @ {avg_time:.3f} +/- {std_time:.3f} s/run")


def benchmark_min_max(data, func, n_runs=25):
    xa, ya, za = data[:, 0], data[:, 1], data[:, 2]
    print(f"• Running benchmark ({func.__name__})...", end=" ")
    times = []
    for _ in range(n_runs):
        start = perf_counter()
        _ = func(xa)
        end = perf_counter()
        times.append(end - start)
    print("done.")

    avg_time = np.mean(times[1:])
    std_time = np.std(times[1:])

    print(f"• {n_runs} runs @ {avg_time:.3f} +/- {std_time:.3f} s/run")


def load_print_sizes():
    data_path = Path(__file__).parents[1] / "data" / "print_sizes.json"
    with open(data_path, "r") as f:
        return json.load(f)


def load_data():
    print("Loading data...", end=" ")
    data_path = Path(__file__).parents[1] / "data" / "d3_test_data_50M_arr.npy"
    data = np.load(data_path)
    print("done.\n")
    return data


def benchmark_render_utilities():
    print_sizes = load_print_sizes()
    xres, yres = print_sizes["A4"]

    data = load_data()

    print("\nDifference arrays")
    print("-----------------")
    benchmark_difference_arrays(data, get_dx)
    benchmark_difference_arrays(data, get_dx_numba)
    benchmark_difference_arrays(data, get_dx_numba_parallel)

    print("\nAspect ratio")
    print("------------")
    benchmark_aspect_ratio(data, xres, yres)

    print("\nMin/max values")
    print("------------")
    benchmark_min_max(data, get_min_max_range)
    benchmark_min_max(data, get_min_max_range_numba)


def benchmark_render():
    print_sizes = load_print_sizes()
    xres, yres = print_sizes["A4"]

    data = load_data()

    render_pipeline = AttractorRenderPipeline(data, xres, yres)

    start = perf_counter()
    for _ in range(10):
        render_pipeline.construct_args_list_burn()
        render_pipeline.burn_pool()
    print(f"avg time: {(perf_counter()-start)/10:.2f}s")


if __name__ == "__main__":
    # benchmark_render_utilities()
    benchmark_render()
