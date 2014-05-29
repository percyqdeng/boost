# distutils: language = c++
cimport numpy as np

from libcpp cimport bool
from libc.math cimport fabs
from libc cimport math
from libc.math cimport fmax
from libc.math cimport fmin
cimport cython
from auxiliary.quicksort import sort as qsort
ctypedef np.float_t dtype_t


@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.float64_t smallest(np.ndarray[np.float64_t] u):
    cdef unsigned int n = u.shape[0]
    cdef Py_ssize_t i
    cdef np.float64_t res = u[0]
    for i in xrange(1, n):
        if u[i] < res:
            res = u[i]
    return res


@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.float64_t k_average(np.ndarray[np.float64_t]v, unsigned int k):
    cdef np. ndarray[np.float64_t] u = v.copy()
    qsort(u)
    cdef np.float64_t res = 0
    cdef Py_ssize_t i
    for i in xrange(k):
        res += u[i]
    return res / k

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void proj_cap_ent_cy(np.ndarray[np.float64_t]d0, np.float64_t v):
    """
    projection with the entropy distance, with capped distribution
    min KL(d,d0) sub to max_i d(i) <=v
    """
    cdef np.ndarray[np.float64_t] u = d0.copy()
    cdef int i, m = d0.shape[0]
    if v < 1.0 / m:
        print "error"
    qsort(u)
    assert u[0] <= u[m-1]
    cdef np.float64_t e, z = 0

    for i in range(m):
        z += u[i]

    # for i in range(m):
    #     e = (1 - v * i) / z
    #     if e * u[i] <= v:
    #         break
    #     z -= u[i]
    for i in xrange(m-1, -1, -1):
        e = 1 - v*(m-1-i) / z
        if e * u[i] <= v:
            break
        z -= u[i]
    for i in range(m):
        d0[i] = fmin(v, e *d0[i])
        # d = np.minimum(v, e * d0)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void exp_descent(np.ndarray[np.float64_t]v, np.ndarray[np.float64_t]x0,
                 double * x, np.float64_t sigma, dist_option=2):
    """
    prox-mapping  argmin_x   <v,x> + 1/sigma D(x0,x)
    distance option:
    dist_option:    1  euclidean distance, 0.5||x-x0||^2
                    2  kl divergence
    """
    cdef Py_ssize_t n = v.shape[0]
    assert n == x0.shape[0]
    cdef Py_ssize_t i
    cdef np.float64_t s = 0
    if dist_option == 1:
        for i in xrange(n):
            x[i] = x0[i] - sigma * v[i]
    elif dist_option == 2:
        for i in xrange(n):
            x[i] = x0[i] * math.exp(-sigma * v[i])
            s += x[i]
        for i in xrange(n):
            x[i] /= s


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void mat_vec(np.ndarray[np.float64_t, ndim=2]aa, np.ndarray[np.float64_t]b, double * c):
    """
    c = aa * b
    """
    cdef Py_ssize_t n = aa.shape[0]
    cdef Py_ssize_t m = aa.shape[1]
    # assert(c.shape[0] == n)
    cdef Py_ssize_t i, j
    for i in range(n):
        c[i] = 0
    for i in range(n):
        for j in range(m):
            c[i] += aa[i,j] * b[j]

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void vec_mat(np.ndarray[np.float64_t] a,
                  np.ndarray[np.float64_t, ndim=2]bb, double *c):
    # not most efficient!!!!!!!!!
    """
    a.T * bb = c.T
    """
    cdef Py_ssize_t n = bb.shape[0]
    cdef Py_ssize_t m = bb.shape[1]
    # assert(c.shape[0] == m)

    for j in range(m):
        c[j] = 0
        for i in range(n):
            c[j] += a[i] * bb[i,j]


cdef np.float64_t l1norm(np.ndarray[np.float64_t] v):
    cdef unsigned int n = v.shape[0]
    cdef unsigned int i
    cdef np.float64_t res = 0
    for i in range(n):
        res += fabs(v[i])

    return res

