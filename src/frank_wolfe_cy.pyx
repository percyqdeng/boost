
import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libc.math cimport fabs
cimport cython


@cython.boundscheck(False)
cpdef fw_boost_cy(double[::1,:]hh, double epsi, double ratio, int steprule, bool has_dcap):
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
    # alpha = np.ones(p)/p
    cdef double[:] alpha = np.zeros(p)
    d0_py = np.ones(n) / n
    cdef double[::1] d0 = d0_py
    cdef double[::1] d= np.zeros(n)
    cdef double[::1] d_next = np.zeros(n)
    cdef double[::1] dt_h = np.zeros(p)
    cdef double mu = epsi / (2 * np.log(n))
    cdef double[::1] h_a = np.zeros(n)
    cdef double ej
    cdef unsigned int max_iter = int(np.log(n) / epsi**2)
    cdef unsigned int nu = int(n * ratio)
    cdef unsigned int t = 0
    # h_a = np.dot(hh, alpha)
    mat_vec(hh, alpha, h_a)
    # d0 = np.ones(n)/n
    cdef unsigned int t, j, i, k
    cdef double res
    print " fw-boosting: maximal iter #: "+str(max_iter)
    for t in range(max_iter):
        d_next = prox_mapping(h_a, d0, 1 / mu)
        # assert not math.isnan(d_next[0])
        if has_dcap:
            d_next = proj_cap_ent(d_next, 1.0 / nu)
        d = d_next
        vec_mat(d, hh, dt_h)
        # dt_h = np.dot(d, hh)
        ej = fabs(dt_h[0])
        j = 0
        for i in range(1,p):
            if fabs(dt_h[i]) > ej:
                ej = fabs(dt_h[i])
                j = i
        # j = np.argmax(np.abs(dt_h))
        # ej = np.zeros(p)
        # ej[j] = np.sign(dt_h[j])
        res = 0
        for i in range(p):
            res -= dt_h[i] * alpha[i]
        res += dt_h[j] * ej
        gap.push_back(res)
        # gap.push(np.dot(dt_h, ej - alpha))
        if has_dcap:
            min_margin = ksmallest(h_a, nu)
            margin.append(np.mean(min_margin))
        else:
            margin.append(np.min(h_a))
        primal_obj.append(mu * np.log(1.0 / n * np.sum(np.exp(-h_a / mu))))
        # dual_obj.append(-np.max(np.abs(dt_h)) - mu * np.dot(d, np.log(d)) + mu * np.log(n))
        if steprule == 1:
            eta = np.maximum(0, np.minimum(1, mu * gap[-1] / np.sum(np.abs(alpha - ej)) ** 2))
        elif steprule == 2:
            eta = np.maximum(0, np.minimum(1, mu * gap[-1] / LA.norm(h_a - hh[:, j] * ej[j], np.inf) ** 2))
        else:
            #
            # do line search
            #
            print "steprule 3, to be done"
        alpha *= (1 - eta)
        alpha[j] += eta * ej[j]
        h_a *= (1 - eta)
        h_a += hh[:, j] * (eta * ej[j])
        err_tr.append(np.mean(h_a <= 0))
        if gap[-1] < epsi:
            break
        if t % (max_iter/10) == 0:
            print "iter " + str(t) + " gap :" + str(gap[-1])
    d = d

cdef double cmp_primal_obj():
    pass

cdef void proj_simplex(double[::1] u, double[::1] z, double[::1] w):
    """
    find w :  min 0.5*||w-u||^2 s.t. w>=0; w1+w2+...+wn = z; z>0
    """
    p = u.shape[0]
    ind = np.argsort(u, kind='quicksort')[::-1]
    mu = u[ind]
    s = np.cumsum(mu)
    tmp = 1.0 / np.asarray([i + 1 for i in range(p)])
    tmp *= (s - z)
    I = np.where((mu - tmp) > 0)[0]
    rho = I[-1]
    w = np.maximum(u - tmp[rho], 0)
    # return w


cdef proj_l1ball(u, z):
    """
    find w :  min 0.5*||w-u||^2 s.t. ||w||_1 <= z
    """
    if l1norm(u) <= z:
        w = u
        return w
    sign = np.sign(u)
    w = proj_simplex(np.abs(u), z)
    w = w * sign
    return w


cdef proj_cap_ent(d0, v):
    """
    projection with the entropy distance, with capped distribution
    min KL(d,d0) sub to max_i d(i) <=v
    """
    d = d0
    m = len(d)
    if v < 1.0 / m:
        print "error"
    ind = np.argsort(d0, kind='quicksort')[::-1]
    u = d[ind]
    Z = u.sum()
    for i in range(m):
        e = (1 - v * i) / Z
        if e * u[i] <= v:
            break
        Z -= u[i]
    d = np.minimum(v, e * d0)
    return d


def ksmallest(u0, k):
    u = u0.tolist()
    mins = u[:k]
    mins.sort()
    for i in u[k:]:
        if i < mins[-1]:
            mins.append(i)
            # np.append(mins, i)
            mins.sort()
            mins = mins[:k]
    return np.asarray(mins)


def prox_mapping(v, x0, sigma, dist_option=2):
    """
    prox-mapping  argmin_x   <v,x> + 1/sigma D(x0,x)
    distance option:
    dist_option:    1  euclidean distance, 0.5||x-x0||^2
                    2  kl divergence
    """
    if dist_option == 1:
        x = x0 - sigma * v
    elif dist_option == 2:
        x = x0 * np.exp(-sigma * v)
        x = x / x.sum()

    return x

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void mat_vec(double[:,::1]aa, double[::1]b, double [::1] c):
    """
    c = aa * b
    """
    cdef Py_ssize_t n = aa.shape[0]
    cdef Py_ssize_t m = aa.shape[1]
    assert(c.shape[0] == n)
    for i in range(n):
        c[i] = 0
    cdef Py_ssize_t i, j
    for i in range(n):
        for j in range(m):
            c[i] += aa[i,j] * b[j]

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void vec_mat(double [::1]a, double[::1,:]bb, double[::1]c):
    # not most efficient!!!!!!!!!
    """
    a.T * bb = c.T
    """
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t m = bb.shape[1]
    assert(c.shape[0] == m)
    for j in range(m):
        c[j] = 0
    for j in range(m):
        for i in range(n):
            c[j] += a[i] * bb[i,j]


cdef double l1norm(double[::1] v):
    cdef unsigned int n = v.shape[0]
    cdef unsigned int i
    cdef double res = 0
    for i in range(n):
        res += fabs(v[i])
    return res