from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import os

# Path to your renderer.pyx
source_file = "batch_renderer.pyx"

# Compiler and linker flags for OpenMP on MSYS2/MinGW
compile_args = ["-O3", "-fopenmp"]
link_args = ["-fopenmp"]

extensions = [
    Extension(
        "batch_renderer",               # Name of the resulting module
        [source_file],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c",
    )
]

setup(
    name="batch_renderer",
    ext_modules=cythonize(extensions, compiler_directives={"boundscheck": False, "wraparound": False}),
)
