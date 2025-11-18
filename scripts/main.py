import argparse
import json
import time
from pathlib import Path

import numpy as np

from attractor_finder import compute_attractor, search_attractor
from attractor_finder.render import AttractorRenderPipeline


def load_print_sizes():
    data_path = Path(__file__).parents[1] / "data" / "print_sizes.json"
    with open(data_path, "r") as f:
        return json.load(f)


def generate_attractor(render_iterates, xres, yres, alpha):
    dimension = 2  # np.random.randint(2,4)
    coeffs, seed = search_attractor(dimension)
    itdata, error = compute_attractor(coeffs, render_iterates, dimension)

    if not error:
        attractor_pipeline = AttractorRenderPipeline(itdata, xres, yres, alpha)
        attractor_pipeline.render_attractor(seed)


def main():

    print_sizes = load_print_sizes()

    parser = argparse.ArgumentParser(description="Generate iterated map attractors.")
    parser.add_argument(
        "--render_iterates",
        type=int,
        default=10_000_000,
        help="Number of iterations per attractor.",
    )
    parser.add_argument(
        "--n_attractors",
        type=int,
        default=100,
        help="Number of attractors to generate.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.025,
        help="Alpha blending value. Set lower if using a large number of iterates.",
    )
    parser.add_argument("--size", choices=print_sizes.keys(), default="A4")
    args = parser.parse_args()

    xres, yres = print_sizes[args.size]

    for i in range(args.n_attractors):
        start = time.perf_counter()
        generate_attractor(args.render_iterates, xres, yres, args.alpha)
        print(f" Total Runtime:        {time.perf_counter()-start:.1f} s")
        print("────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
