from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name="iterator_cubic", ext_modules=cythonize('iterator_cubic.pyx'),include_dirs=[numpy.get_include()])