from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from attractor_finder import search_attractor
from attractor_finder.iterator import iterator, iterator_optimized


def test_iterators(dimension=2, seed=None):
    if seed is None:
        seed = np.random.randint(1, 2_000_000_000)
    np.random.seed(seed)

    coeffs, used_seed = search_attractor(dimension=dimension, seed=seed)
    x0 = np.random.uniform(-1e-1, 1e-1, (dimension + 1))
    itdata_std = iterator(
        n_iterations=100_000, coeffs=coeffs, x0=x0, dimension=dimension
    )
    itdata_opt = iterator_optimized(
        n_iterations=100_000, coeffs=coeffs, x0=x0, dimension=dimension
    )
    same = np.allclose(itdata_std, itdata_opt)
    if same:
        print(
            f"✅ Iterators produce equivalent results for dimension {dimension} (seed: {used_seed})"
        )
    else:
        print(
            f"❌ Iterators differ for dimension {dimension} (seed: {used_seed}) — check for numerical or logical discrepancies"
        )


def benchmark_iterator(func, seed, n_runs=10, n_iterations=1_000_000, dimension=2):
    if seed is None:
        seed = np.random.randint(1, 2_000_000_000)
    np.random.seed(seed)

    coeffs, used_seed = search_attractor(dimension=dimension, seed=seed)
    x0 = np.random.uniform(-1e-1, 1e-1, (dimension + 1))

    times = []
    for _ in range(n_runs):
        start = perf_counter()
        _ = func(n_iterations=n_iterations, coeffs=coeffs, x0=x0, dimension=dimension)
        end = perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_its_per_sec = n_iterations / avg_time
    std_its_per_sec = n_iterations * std_time / (avg_time**2)

    print(
        f"Dimension {dimension}: {avg_its_per_sec/1e6:.3f} ± {std_its_per_sec/1e6:.3f} M it/s ({func.__name__})"
    )
    return avg_its_per_sec, std_its_per_sec


def run_benchmarks_across_dimensions(
    func, max_dim=6, n_runs=10, n_iterations=100_000, seed=None
):
    dims = list(range(2, max_dim + 1))
    avg_results = []
    std_results = []

    for d in dims:
        print(f"\nRunning benchmark for dimension {d} using {func.__name__}...")
        avg_its, std_its = benchmark_iterator(
            func, seed=seed, n_runs=n_runs, n_iterations=n_iterations, dimension=d
        )
        avg_results.append(avg_its)
        std_results.append(std_its)

    return dims, avg_results, std_results


def plot_results(dims, avg1, std1, avg2, std2):
    avg1_m = [r / 1e6 for r in avg1]
    std1_m = [s / 1e6 for s in std1]
    avg2_m = [r / 1e6 for r in avg2]
    std2_m = [s / 1e6 for s in std2]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        dims, avg1_m, yerr=std1_m, fmt="o-", label="iterator", color="teal", capsize=5
    )
    plt.errorbar(
        dims,
        avg2_m,
        yerr=std2_m,
        fmt="s--",
        label="iterator_optimized",
        color="orange",
        capsize=5,
    )
    plt.title("Average Iterations per Second vs Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Iterations per Second (Millions)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    for d in range(2, 7):
        test_iterators(dimension=d)
    dims, avg_orig, std_orig = run_benchmarks_across_dimensions(
        iterator, max_dim=6, n_runs=100, n_iterations=1_000_000, seed=5
    )
    _, avg_opt, std_opt = run_benchmarks_across_dimensions(
        iterator_optimized, max_dim=6, n_runs=100, n_iterations=1_000_000, seed=5
    )
    plot_results(dims, avg_orig, std_orig, avg_opt, std_opt)
