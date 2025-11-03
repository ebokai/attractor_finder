# Cubic Iterator Attractor Renderer

A high-performance Python+Cython pipeline for generating and visualizing high-dimensional **iterated map attractors**. It explores chaotic structures defined by discrete cubic maps in 2+ dimensions using optimized numerical kernels and custom rendering.

## Installation

From the project root:

```bash
python -m pip install -e .
```

## Usage

Run the attractor generator:

```bash
./scripts/run.bat        # Windows
python scripts/main.py   # Cross-platform
```

### Optional arguments:
```bash
--render_iterates   # Number of iterations per attractor. Default: 10,000,000.
--n_attractors      # Total attractors to generate. Default: 100.
--alpha             # Alpha blending for rendering. Float between 0 and 1. Default: 0.025.
--size              # Output resolution. Options: A5, A4, A3, A2, A1, Letter, 8x10, 11x14, 16x20, 18x24, Square_8, Square_12
```

Example:
```bash
python scripts/main.py --render_iterates 25000000 --n_attractors 50 --alpha 0.02 --size A4
```
---
