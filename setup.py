from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
Extension("attractor_finder.iterator", ["src/attractor_finder/iterator.pyx"]),
Extension("attractor_finder.renderer_batch", ["src/attractor_finder/renderer_batch.pyx"]),
Extension("attractor_finder.renderer", ["src/attractor_finder/renderer.pyx"])]

setup(
	name = "attractor_finder",
	ext_modules = cythonize(extensions, 
		build_dir = "cython_build",
		annotate = True, 
		language_level = "3"))