__author__ = 'qdengpercy'


from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='qsort',
    include_dirs=[np.get_include()],
    ext_modules=cythonize("auxiliary/quicksort.pyx",
                          language="c++",
                          )
)

# run setupqsort.py build_ext --inplace