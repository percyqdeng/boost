__author__ = 'qdengpercy'



from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='ksmallest',
    include_dirs=[np.get_include()],
    ext_modules=cythonize("ksmall.pyx",
                          language="c++",
                          )
)

# run setup_ksmall.py build_ext --inplace