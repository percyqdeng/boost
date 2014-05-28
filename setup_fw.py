__author__ = 'qdengpercy'


from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
# p1 = '/Users/qdengpercy/workspace/boost/src/aux'
setup(
    name='fw_cython',
    include_dirs=[np.get_include()],
    ext_modules=cythonize("src/frank_wolfe_cy.pyx",
                          language="c++",
                          )
)

# run setup_fw.py build_ext --inplace