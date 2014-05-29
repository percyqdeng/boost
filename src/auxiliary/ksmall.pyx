# distutils: language = c++
__author__ = 'qdengpercy'

# from libcpp.vector cimport vector
import random
import numpy as np
cimport numpy as np
cimport cython
import time

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T &)
        void pop_back()
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()


cdef extern from "<algorithm>" namespace "std":
    cdef cppclass RandomAccessIterator:
        cppclass Compare
    void make_heap [RandomAccessIterator](RandomAccessIterator first, RandomAccessIterator last)
    void pop_heap [RandomAccessIterator](RandomAccessIterator first, RandomAccessIterator last)
    void sort_heap [RandomAccessIterator](RandomAccessIterator first, RandomAccessIterator last)
    void push_heap [RandomAccessIterator](RandomAccessIterator first, RandomAccessIterator last)


cpdef demo():
    n = 15
    arr = range(n)
    random.shuffle(arr)
    cdef vector[int] v
    print "random list of number"
    print arr
    for i, x in enumerate(arr):
        # print str(x) + ' ',
        v.push_back(x)
    make_heap[vector[int].iterator](v.begin(), v.end())
    # sort_heap[vector[int].iterator](v.begin(), v.end())
    print ""
    print "after heap sort"
    for i in range(n):
        print str(v[i])+' ',
    print
    for i in range(n):
        print "pop out" + str(v[0])
        pop_heap[vector[int].iterator](v.begin(), v.end())
        v.pop_back()
        for j in range(len(v)):
            print str(v[j]) + ' ',
        print

cpdef test_ksmallest():
    pass

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef heapsort(np.ndarray[double, mode='c']u):
    cdef int n = u.shape[0]
    cdef vector[double] v
    for i in xrange(n):
        v.push_back(u[i])
    make_heap[vector[double].iterator](v.begin(), v.end())
    sort_heap[vector[double].iterator](v.begin(), v.end())


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double k_avg(np.ndarray[double, mode='c']u, unsigned int k):
    """
    average of ksmallest numbers in the list
    """
    cdef int n = u.shape[0]
    # for i in xrange(n):
    #     print str(u[i]) + ' ',
    # print
    cdef unsigned i, j
    assert 0 <= k < n
    cdef vector[double] v

    for i in xrange(k):
        v.push_back(u[i])
    make_heap[vector[double].iterator](v.begin(), v.end())
    # sort_heap[vector[double].iterator](v.begin(), v.end())
    # print "the first k ele: "
    # for i in range(k):
    #     print str(v[i])+' ',
    # print
    for i in xrange(k,n):
        v.push_back(u[i])
        # print 'push '+str(u[i])+ ' ',
        push_heap[vector[double].iterator](v.begin(), v.end())
        pop_heap[vector[double].iterator](v.begin(), v.end())
        # print 'move '+str(v[0])
        v.pop_back()
    # if debug:
    # for i in xrange(len(v)):
    #     print str(v[i])+' ',
    cdef double res = 0
    for i in xrange(k):
        res += v[i]
    return res / k


cpdef ksmallest(np.ndarray[double, mode='c']u, unsigned int k):
    """
    compute the ksmallest numbers
    """
    cdef int n = u.shape[0]
    # for i in xrange(n):
    #     print str(u[i]) + ' ',
    # print
    cdef unsigned i, j
    assert 0 <= k < n
    cdef vector[double] v

    for i in xrange(k):
        v.push_back(u[i])
    make_heap[vector[double].iterator](v.begin(), v.end())
    # sort_heap[vector[double].iterator](v.begin(), v.end())
    # print "the first k ele: "
    # for i in range(k):
    #     print str(v[i])+' ',
    # print
    for i in xrange(k,n):
        v.push_back(u[i])
        # print 'push '+str(u[i])+ ' ',
        push_heap[vector[double].iterator](v.begin(), v.end())
        pop_heap[vector[double].iterator](v.begin(), v.end())
        # print 'move '+str(v[0])
        v.pop_back()
    # if debug:
    # for i in xrange(len(v)):
    #     print str(v[i])+' ',
    return v


cpdef cmp_time(int n):
    arr = np.random.normal(0, 1, n)
    start = time.time()
    v = ksmallest(arr, 100)
    print "ksmallest " + str(time.time() - start)

    start = time.time()
    heapsort(arr)
    print "heap sort " + str(time.time() - start)