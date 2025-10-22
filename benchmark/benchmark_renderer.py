from pathlib import Path 
import numpy as np
from time import perf_counter
from attractor_finder.functions import get_dx

def benchmark_renderer(n_runs = 100):

	data = load_data()
	xa, ya, za = data[:,0], data[:,1], data[:,2]

	times_dx = []
	for _ in range(n_runs):
		start = perf_counter()
		dxs = get_dx(xa)
		dys = get_dx(ya)
		dzs = get_dx(za)
		end = perf_counter()
		times_dx.append(end - start)

	avg_time_dx = np.mean(times_dx)
	std_time_dx = np.std(times_dx)

	print(f'Difference arrays: {avg_time_dx:.3f} +/- {std_time_dx:.3f} s')




def load_data():
	data_path = Path(__file__).parents[1] / "data" / "d3_test_data_50M_arr.npy"
	return np.load(data_path)

benchmark_renderer()