# distutils: language = c++
import numpy as np
cimport numpy as np

from libcpp cimport bool
from libc.math cimport fabs
from libc cimport math
from libc.math cimport fmax
from libc.math cimport fmin
cimport cython
from libcpp.vector cimport vector
from aux.quicksort import sort as qsort
ctypedef np.float_t dtype_t

cpdef pm_boost_cy(np.ndarray[double,ndim=2]hh,double epsi=0.01, double ratio=0.1, int steprule=1, bool has_dcap=False):
    """
    primal-dual boost with capped probability ||d||_infty <= 1/k
    a, d : primal dual variables to be updated,
    a_tilde, d_tilde : intermediate variables,
    a_bar, d_bar : average value as output.
    """
    print '----------------primal-dual boost-------------------'
    hh = np.hstack((hh, -hh))
    (n, p) = hh.shape
    cdef Py_ssize_t n = hh.shape[0]
    cdef Py_ssize_t p = hh.shape[1]
    cdef double c = math.log(n*p)
    cdef int nu = int(n * ratio)
    cdef int max_iter = int(np.log(n * p) / epsi)
    cdef Py_ssize_t showtimes = int(5)
    cdef np.ndarray[double] d = np.ones(n) / n
    cdef np.ndarray[double] d_bar = np.ones(n) / n
    cdef np.ndarray[double] a_bar = np.ones(p) / p
    cdef np.ndarray[double] a = np.ones(p) / p
    # a_bar = a
    cdef np.ndarray[double] a_tilde = np.ones(p) / p
    # d_tilde = np.zeros(p)
    cdef double theta = 1
    cdef double sig = 1
    cdef double tau = 1
    cdef Py_ssize_t t = 0
    for t in range(max_iter):
        d = prox_mapping(np.dot(hh, a_tilde), d, tau, 2)
        if has_dcap:
            d2 = proj_cap_ent(d, 1.0 / nu)
            # d_new = d_new/d_new.sum()
            if np.abs(d.sum() - d2.sum()) > 0.0001:
                print 'error'
            d = d2
        d_tilde = d
        dthh = np.dot(d_tilde, hh)
        a_new = prox_mapping(-dthh, a, sig, 2)
        # a_new = proj_l1ball(tmp, 1)
        a_tilde = a_new + theta * (a_new - a)
        a = a_new
        d_bar *= t / (t + 1.0)
        d_bar += 1.0 / (t + 1) * d
        a_bar *= t / (t + 1.0)
        a_bar += 1.0 / (t + 1) * a
        h_a = np.dot(hh, a_bar)
        if has_dcap:
            min_margin = ksmallest(h_a, nu)
            _primal_obj.append(-np.mean(min_margin))
        else:
            _primal_obj.append(- np.min(h_a))
        _margin.append(-_primal_obj[-1])
        _dual_obj.append(-np.max(np.dot(d_bar, hh)))
        _gap.append(_primal_obj[-1] - _dual_obj[-1])
        err_tr.append(np.mean(h_a < 0))
        if t % (max_iter / showtimes) == 0:
            print 'iter ' + str(t) + ' ' + str(_gap[-1])
        if _gap[-1] < epsi:
            break
    alpha = a_bar[:p / 2] - a_bar[p / 2:]
    d = d_bar
