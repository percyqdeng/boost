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


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef fw_boost_cy(np.ndarray[double, ndim=2]hh,
                  double epsi=0.01, double ratio=0.1, int steprule=1, bool has_dcap=False):
    """
    frank-wolfe boost for binary classification with weak learner as matrix hh
    min_a max_d   d^T(-hha) sub to:  ||a||_1\le 1
    capped probability constraint:
        max_i d_i <= 1/(n*ratio)
    Args:
        hh : output matrix of weak learners
    """
    cdef unsigned int n = hh.shape[0]
    cdef unsigned int p = hh.shape[1]
    cdef vector[double] margin
    cdef vector[double] primal_obj
    cdef vector[double] dual_obj
    cdef vector[double] gap
    cdef vector[double] err_tr
    cdef np.ndarray[double] alpha = np.zeros(p)
    d0_py = np.ones(n) / n
    cdef np.ndarray[double] d0 = d0_py
    cdef np.ndarray[double] d= np.zeros(n)
    cdef np.ndarray[double] d_next = np.zeros(n)
    cdef np.ndarray[double] dt_h = np.zeros(p)
    cdef double mu = epsi / (2 * math.log(n))
    cdef np.ndarray[double] h_a = np.zeros(n)
    cdef double ej
    cdef double eta
    cdef unsigned int max_iter = int(math.log(n) / epsi**2)
    cdef unsigned int nu = int(n * ratio)
    mat_vec(hh, alpha, <np.float_t*>h_a.data)
    cdef Py_ssize_t t, j, i, k
    cdef double res, tmp

    print " fw-boosting: maximal iter #: "+str(max_iter)
    for t in range(max_iter):
        exp_descent(h_a, d0, <np.float_t*>d.data, 1 / mu)
        # print "haha"
        # assert not math.isnan(d_next[0])
        if has_dcap:
            proj_cap_ent(d, 1.0 / nu)
        # d = d_
        vec_mat(d, hh, <np.float_t*>dt_h.data)
        ej = fabs(dt_h[0])
        j = 0
        for i in range(1,p):
            if fabs(dt_h[i]) > ej:
                ej = fabs(dt_h[i])
                j = i
        # j = np.argmax(np.abs(dt_h))
        # ej = np.zeros(p)
        # ej[j] = np.sign(dt_h[j])
        if dt_h[j] < 0:
            ej = dt_h[j]
        res = 0
        for i in xrange(p):
            res -= dt_h[i] * alpha[i]
        res += dt_h[j] * ej
        gap.push_back(res)
        # gap.push(np.dot(dt_h, ej - alpha))
        if has_dcap:
            min_margin = k_average(h_a, nu)
            margin.push_back(min_margin)
        else:
            margin.push_back(smallest(h_a))
        # primal_obj.append(mu * math.log(1.0 / n * np.sum(np.exp(-h_a / mu))))
        primal_obj.push_back(primal_objective(h_a, mu))
        # dual_obj.append(-np.max(np.abs(dt_h)) - mu * np.dot(d, np.log(d)) + mu * np.log(n))
        if steprule == 1:
            res = 0
            for i in xrange(p):
                if i != j:
                   res += fabs(alpha[i])
                else:
                    res += fabs(alpha[i]-ej)
            eta = fmax(0, fmin(1, mu * gap[t] / res ** 2))
        elif steprule == 2:
            res = fabs(h_a[0] - hh[0,j] * ej)
            for i in xrange(1,p):
                tmp = fabs(h_a[i] - hh[i,j] * ej)
                if tmp > res:
                    res = tmp
            eta = fmax(0, fmin(1, mu * gap[t] / res ** 2))
        else:
            print "steprule 3, to be done"
        for i in xrange(p):
            alpha[i] *= 1-eta
        # alpha *= (1 - eta)
        alpha[j] += eta * ej
        for i in xrange(p):
            h_a[i] *= (1 - eta)
            h_a[i] += hh[i, j] * (eta * ej)
        res = 0
        for i in xrange(n):
            if h_a < 0:
                res += 1
        err_tr.push_back(res/n)
        if gap[t] < epsi:
            break
        if t % (max_iter/10) == 0:
            print "iter " + str(t) + " gap :" + str(gap[t])

    return alpha, primal_obj, gap, err_tr

cdef double primal_objective(np.ndarray[double] z, double mu):
    """
    mu * log(1/n * sum(-z_i/mu))
    """
    cdef double res = 0
    cdef unsigned int i, n = z.shape[0]
    for i in range(n):
        res += math.exp(-z[i]/mu)
    res = mu * math.log(res / n)
    return res

cdef double k_average(np.ndarray[double]v, unsigned int k):
    u = v.copy()
    qsort(u)
    cdef double res = 0
    cdef Py_ssize_t i
    for i in xrange(k):
        res += u[i]
    return res / k

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void proj_cap_ent(np.ndarray[double]d0, double v):
    """
    projection with the entropy distance, with capped distribution
    min KL(d,d0) sub to max_i d(i) <=v
    """
    u = d0.copy()
    cdef int m = d0.shape[0]
    if v < 1.0 / m:
        print "error"
    qsort(u)
    cdef double z = 0
    for i in range(m):
        z += u[i]
    for i in range(m):
        e = (1 - v * i) / z
        if e * u[i] <= v:
            break
        z -= u[i]
    for i in range(m):
        d0[i] = fmin(v, e *d0[i])
    # d = np.minimum(v, e * d0)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void exp_descent(np.ndarray[double]v, np.ndarray[double]x0,
                 double * x, double sigma, dist_option=2):
    """
    prox-mapping  argmin_x   <v,x> + 1/sigma D(x0,x)
    distance option:
    dist_option:    1  euclidean distance, 0.5||x-x0||^2
                    2  kl divergence
    """
    cdef Py_ssize_t n = v.shape[0]
    assert n == x0.shape[0]
    cdef Py_ssize_t i
    cdef double s = 0
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
cdef void mat_vec(np.ndarray[double, ndim=2]aa, np.ndarray[double]b, double * c):
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
cdef void vec_mat(np.ndarray[double] a,
                  np.ndarray[double, ndim=2]bb, double *c):
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


cdef double l1norm(np.ndarray[double] v):
    cdef unsigned int n = v.shape[0]
    cdef unsigned int i
    cdef double res = 0
    for i in range(n):
        res += fabs(v[i])
    return res

cdef double smallest(np.ndarray[double] u):
    cdef unsigned int n = u.shape[0]
    cdef Py_ssize_t i
    cdef double res = u[0]
    for i in xrange(1, n):
        if u[i] < res:
            res = u[i]
    return res
