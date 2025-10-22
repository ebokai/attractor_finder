import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from attractor_finder import search_attractor
from attractor_finder.iterator import iterator

def benchmark_iterator(n_runs=10, n_iterations=100_000, dimension=2, seed=None):
    if seed is None:
        seed = np.random.randint(1, 2_000_000_000)
    np.random.seed(seed)

    coeffs, used_seed = search_attractor(dimension=dimension, seed=seed)

    times = []
    for _ in range(n_runs):
        start = perf_counter()
        _ = iterator(n_iterations=n_iterations, coeffs=coeffs, dimension=dimension)
        end = perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_its_per_sec = n_iterations / avg_time
    std_its_per_sec = n_iterations * std_time / (avg_time ** 2)

    print(f"Dimension {dimension}: {avg_its_per_sec/1e6:.3f}M it/s ± {std_its_per_sec/1e6:.3f}")
    return avg_its_per_sec, std_its_per_sec

def run_benchmarks_across_dimensions(max_dim=6, n_runs=10, n_iterations=100_000):
    dims = list(range(2, max_dim + 1))
    avg_results = []
    std_results = []

    for d in dims:
        print(f"\nRunning benchmark for dimension {d}...")
        avg_its, std_its = benchmark_iterator(n_runs=n_runs, n_iterations=n_iterations, dimension=d)
        avg_results.append(avg_its)
        std_results.append(std_its)

    return dims, avg_results, std_results

def plot_results(dims, avg_results, std_results):
    avg_millions = [r / 1e6 for r in avg_results]
    std_millions = [s / 1e6 for s in std_results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(dims, avg_millions, yerr=std_millions, fmt='o-', color='teal', capsize=5)
    plt.title("Average Iterations per Second vs Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Iterations per Second (Millions)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dims, avg_results, std_results = run_benchmarks_across_dimensions(max_dim=6, n_runs=100, n_iterations=100_000)
    plot_results(dims, avg_results, std_results)