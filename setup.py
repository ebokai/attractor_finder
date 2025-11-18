from Cython.Build import cythonize
from setuptools import Extension, setup

openmp_args = {"extra_compile_args": ["-fopenmp"], "extra_link_args": ["-fopenmp"]}

extensions = [
    Extension(
        "attractor_finder.iterator",
        ["src/attractor_finder/iterator.pyx"],
        **openmp_args,
    ),
    Extension(
        "attractor_finder.renderer_batch",
        ["src/attractor_finder/renderer_batch.pyx"],
        **openmp_args,
    ),
    Extension(
        "attractor_finder.renderer",
        ["src/attractor_finder/renderer.pyx"],
        **openmp_args,
    ),
]

setup(
    name="attractor_finder",
    ext_modules=cythonize(
        extensions, build_dir="cython_build", annotate=True, language_level="3"
    ),
)
