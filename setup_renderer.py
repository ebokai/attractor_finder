from setuptools import setup
from Cython.Build import cythonize

setup(name="renderer", ext_modules=cythonize('renderer.pyx', compiler_directives={'language_level' : "3"}),)