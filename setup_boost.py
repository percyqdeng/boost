__author__ = 'qdengpercy'


from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
# p1 = '/Users/qdengpercy/workspace/boost/src/aux'
setup(
    name='boost_cy',
    include_dirs=[np.get_include()],
    ext_modules=cythonize("src/boost_cy.pyx",
                          language="c++",
                          )
)

# run setup_boost.py build_ext --inplace