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
--render_iterates   # Number of iterations per attractor
--n_attractors      # Total attractors to generate
--alpha             # Alpha blending for rendering
--save_data         # Save iteration data as .npy
```

Example:
```bash
python scripts/main.py --render_iterates 25000000 --n_attractors 50 --alpha 0.02 --save_data
```
---
