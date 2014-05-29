__author__ = 'qdengpercy'

import numpy as np
import timeit
import time
from boost import ksmallest
from boost import ksmallest2
import profile
n = 5000

def check_ksm():
    n = 50000
    k = 5000
    rep = 20
    t1 = 0
    t2 = 0
    for i in range(rep):
        arr = np.random.normal(0, 1, n)
        # timeit.timeit("ksmallest(arr,k")
        start = time.time()
        ksmallest(arr, k)
        t1 += time.time() - start
        start = time.time()
        ksmallest2(arr, k)
        t2 += time.time() - start
    print "%d runs, average time t1 %f, t2 %f" % (rep, t1/rep, t2/rep)

def check_obj():
    a = np.array([[1,2], [3, 4]])
    b = a.T
    print "are a, b same objects: %s" % (a is b)
    b[1, 1] = -1
    print a
    print b

def check_mat_vec():
    a = np.random.normal(0, 1, (n, n))
    b = np.random.normal(0, 1, n)
    start = time.time()
    c = np.dot(a, b)
    t1 = time.time() - start

    a2 = np.asfortranarray(a)
    start = time.time()
    e = np.dot(b, a2)
    t3 = time.time() - start

    start = time.time()
    d = np.dot(b, a.T)
    t2 = time.time() - start

    print "t1%f, t2%f, t3%f" % (t1, t2, t3)


if __name__=='__main__':
    # profile.run(check_ksm())
    # profile.runctx("check_ksm()", globals(), locals())
    check_ksm()