__author__ = 'qdengpercy'
# coding=utf8
# from boost import *
import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype
ctypedef np.int32_t itype
from libc cimport math
from libc.math cimport fmax
from libc.math cimport fmin
from libcpp cimport bool
cimport cython

cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int N, double *X, int incX,
                             double *Y, int incY)

cdef class ParaBoost_cy():
    """Primal-Dual Parallel Boosting method as a saddle point matrix game
    Parameters:
    ------------
    epsi ：optimization tolerance
    has_dcap : boolean
        has capped probability in the dual variable (distribution of samples)
    has_pcap : boolean, has capped weight in the primal variable (weight of weak learners)
    ratio : float the capped probability
    alpha : array type, the weights of weak learner
    err_tr : array type, training error of each iteration
    _primal_obj : array type, primal objective in each iteration
    _margin : array type, margin in each iteration
    _dual_obj : array type, dual objective in each iteration
    _gap : array type, primal-dual gap in each iteration
    """

    cdef __init__(self, epsi=0.01, has_dcap=False, ratio=0.1):
        self.epsi = epsi
        self.has_dcap = has_dcap
        self.ratio = ratio
        self._primal_obj = []
        self._dual_obj = []
        self._margin = []
        self._gap = []
        self.err_tr = []
        self.alpha = []
        self.iter_num = []

    def to_name(self):
        return "paraboost"

    cpdef void train(self, np.ndarray[dtype,ndim=2]xtr, np.ndarray[itype]ytr, bool early_stop=False):

        # xtr = self._process_train_data(xtr)
        # xtr = np.hstack((xtr, np.ones((ntr, 1))))
        yH = ytr[:, np.newaxis] * xtr
        # yH = np.hstack((yH, -yH))
        self._para_boosting(yH, early_stop)

    cpdef void train_h(self, np.ndarray[dtype,ndim=2]h, np.ndarray[itype]ytr, bool early_stop=False):
        yh = ytr[:, np.newaxis] * h
        self._para_boosting(yh, early_stop)

    cpdef np.ndarray[itype] test_h(self,np.ndarray[dtype,ndim=2] h):
        cdef np.ndarray[itype] pred = np.sign(np.dot(h, self.alpha))
        return pred

    cpdef  test(self, np.ndarray[dtype,ndim=2]xte, np.ndarray[itype]yte):
        cdef Py_ssize_t nte = xte.shape[0]
        # xte = self._process_test_data(xte)
        # xte = np.hstack((xte, np.ones((nte, 1))))
        pred = np.sign(np.dot(xte, self.alpha))
        return np.mean(pred != yte)


    cdef _para_boosting(self, np.ndarray[dtype,ndim=2] H, bool earlystop=False):
        """
        primal-dual boost with capped probability ||d||_infty <= 1/k
        a, d : primal dual variables to be updated,
        a_tilde, d_tilde : intermediate variables,
        a_bar, d_bar : average value as output.
        """
        # print '----------------primal-dual boost-------------------'
        H = np.hstack((H, -H))
        (n, p) = H.shape
        cdef Py_ssize_t n = H.shape[0]
        cdef Py_ssize_t p = H.shape[1]
        self.c = math.log(n*p)
        cdef int nu = int(n * self.ratio)
        cdef Py_ssize_t max_iter
        cdef Py_ssize_t delta
        if earlystop:
            max_iter = 200
        else:
            max_iter = int(np.log(n * p) / self.epsi)
        if max_iter < 1000:
            delta = 4
        else:
            delta = 40
        showtimes = int(5)
        cdef np.ndarray[dtype] d = np.ones(n) / n
        cdef np.ndarray[dtype] d_bar = np.ones(n) / n
        cdef np.ndarray[dtype] a_bar = np.ones(p) / p
        cdef np.ndarray[dtype] d_next = np.ones(n) / n
        cdef np.ndarray[dtype] a = np.ones(p) / p
        # a_bar = a
        cdef np.ndarray[dtype] a_tilde = np.ones(p) / p
        cdef np.ndarray[dtype] h_a = np.dot(H, a_tilde)
        # d_tilde = np.zeros(p)
        cdef double theta = 1
        cdef double sig = 1
        cdef tau = 1
        cdef Py_ssize_t t = 0
        # print " pd-boosting(python): maximal iter #: "+str(max_iter)
        for t in range(max_iter):
            multi_update(<dtype *>h_a.data, <dtype *>d.data, <dtype *>d_next.data, tau, 2)
            if self.has_dcap:
                proj_cap_ent_cy(d, 1.0 / nu)
                # # d_new = d_new/d_new.sum()
                # if np.abs(d.sum() - d2.sum()) > 0.0001:
                #     print 'error'
                # d = d2
            d_tilde = d
            dtH = np.dot(d_tilde, H)
            a_new = prox_mapping(-dtH, a, sig, 2)
            # a_new = proj_l1ball(tmp, 1)
            a_tilde = a_new + theta * (a_new - a)
            a = a_new
            d_bar *= t / (t + 1.0)
            d_bar += 1.0 / (t + 1) * d
            a_bar *= t / (t + 1.0)
            a_bar += 1.0 / (t + 1) * a
            h_a = np.dot(H, a_bar)
            if t % delta == 0:
                self.iter_num.append(t)
                if self.has_dcap:
                    min_margin = ksmallest(h_a, nu)
                    self._primal_obj.append(-np.mean(min_margin))
                else:
                    self._primal_obj.append(- np.min(h_a))
                self._margin.append(-self._primal_obj[-1])
                self._dual_obj.append(-np.max(np.dot(d_bar, H)))
                self._gap.append(self._primal_obj[-1] - self._dual_obj[-1])
                self.err_tr.append(np.mean(h_a < 0))
            # if t % (max_iter / showtimes) == 0:
            #     print 'iter ' + str(t) + ' ' + str(self._gap[-1])
            if self._gap[-1] < self.epsi:
                break
        self.alpha = a_bar[:p / 2] - a_bar[p / 2:]
        self.d = d_bar


@cython.boundscheck(False)
@cython.cdivision(True)
cdef dtype smallest(np.ndarray[dtype] u):
    cdef unsigned int n = u.shape[0]
    cdef Py_ssize_t i
    cdef dtype res = u[0]
    for i in xrange(1, n):
        if u[i] < res:
            res = u[i]
    return res


@cython.boundscheck(False)
@cython.cdivision(True)
cdef dtype k_average(np.ndarray[dtype]v, unsigned int k):
    cdef np. ndarray[dtype] u = v.copy()
    qsort(u)
    cdef dtype res = 0
    cdef Py_ssize_t i
    for i in xrange(k):
        res += u[i]
    return res / k

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void proj_cap_ent_cy(np.ndarray[dtype]d0,  dtype v):
    """
    projection with the entropy distance, with capped distribution
    min KL(d,d0) sub to max_i d(i) <=v
    """
    cdef np.ndarray[dtype] u = d0.copy()
    cdef int i, m = d0.shape[0]
    if v < 1.0 / m:
        print "error"
    qsort(u)
    assert u[0] <= u[m-1]
    cdef dtype e, z = 0

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
cdef void multi_update(double *v, double * x0, double *x, int n,  double sigma):
    """
    prox-mapping  argmin_x   <v,x> + 1/sigma D(x0,x)
    distance option:
    dist_option:    1  euclidean distance, 0.5||x-x0||^2
                    2  kl divergence
    """
    # cdef Py_ssize_t n = v.shape[0]
    # assert n == x0.shape[0]
    cdef Py_ssize_t i
    cdef double s = 0
    for i in xrange(n):
        x[i] = x0[i] * math.exp(-sigma * v[i])
        s += x[i]
    for i in xrange(n):
        x[i] /= s


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void mat_vec(np.ndarray[dtype, ndim=2]aa, np.ndarray[dtype]b, double * c):
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
cdef void vec_mat(np.ndarray[dtype] a,
                  np.ndarray[dtype, ndim=2]bb, double *c):
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


cdef dtype l1norm(np.ndarray[dtype] v):
    cdef unsigned int n = v.shape[0]
    cdef unsigned int i
    cdef dtype res = 0
    for i in range(n):
        res += fabs(v[i])
    return res

