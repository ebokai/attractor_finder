from attractor_finder import search_attractor, compute_attractor, render_attractor
from pathlib import Path

import argparse
import numpy as np 
import time 
import json


def load_print_sizes():
	data_path = Path(__file__).parents[1] / "data" / "print_sizes.json"
	with open(data_path, 'r') as f:
		return json.load(f)

def generate_attractor(render_iterates, xres, yres, alpha, save_data):
	dimension = np.random.randint(2,4)
	coeffs, seed = search_attractor(dimension)
	itdata, error = compute_attractor(coeffs, render_iterates, dimension)

	if not error:
		render_attractor(
			itdata[10000:, dimension - 3], 
			itdata[10000:, dimension - 2], 
			itdata[10000:, dimension - 1], 
			coeffs, dimension, seed, 'xyz', 
			alpha=alpha, xres=xres, yres=yres)

		if save_data:
			np.save(f'd_{dimension}_{seed}_arr', itdata)

def main():

	parser = argparse.ArgumentParser(description="Generate iterated map attractors.")
	parser.add_argument("--render_iterates", type=int, default=10_000_000, help="Number of iterations per attractor.")
	parser.add_argument("--n_attractors", type=int, default=100, help="Number of attractors to generate.")
	parser.add_argument("--alpha", type=float, default=0.025, help="Alpha blending value. Set lower if using a large number of iterates.")
	parser.add_argument("--save_data", action="store_true", help="Save iteration data as .npy file.")
	args = parser.parse_args()

	print_sizes = load_print_sizes()
	xres, yres = print_sizes['A4']

	for i in range(args.n_attractors):
		start = time.perf_counter()
		generate_attractor(args.render_iterates, xres, yres, args.alpha, args.save_data)
		print(f" Total Runtime:        {time.perf_counter()-start:.1f} s")
		print("────────────────────────────────────────────\n")

if __name__ == "__main__":
	main()