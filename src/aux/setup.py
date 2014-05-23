__author__ = 'qdengpercy'



from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='stoc_coor_cython',
    include_dirs=[np.get_include()],
    ext_modules=cythonize("ksmall.pyx",
                          language="c++",
                          )
)

# run setup.py build_ext --inplace