from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import os

# Path to your renderer.pyx
source_file = "renderer.pyx"

# Compiler and linker flags for OpenMP on MSYS2/MinGW
compile_args = ["-O3", "-fopenmp"]
link_args = ["-fopenmp"]

extensions = [
    Extension(
        "renderer",               # Name of the resulting module
        [source_file],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c",
    )
]

setup(
    name="renderer",
    ext_modules=cythonize(extensions, compiler_directives={"boundscheck": False, "wraparound": False}),
)
