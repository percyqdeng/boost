cimport numpy as np
#https://gist.github.com/zed/1257360
DEF CUTOFF = 17


cpdef sort(np.ndarray[np.float_t,ndim=1]arr):
    # cdef double[::1] a = arr
    # qsort(&a[0], 0, a.shape[0])
    qsort(<np.float_t*>arr.data, 0, arr.shape[0])


cdef void qsort(double * a, int start, int end):
    if (end - start) < CUTOFF:
        insertion_sort(a, start, end)
        return
    cdef int boundary = partition(a, start, end)
    qsort(a, start, boundary)
    qsort(a, boundary+1, end)


cdef int partition(double* a, int start, int end):
    assert end > start
    cdef int i = start, j = end-1
    cdef double pivot = a[j]
    while True:
        # assert all(x < pivot for x in a[start:i])
        # assert all(x >= pivot for x in a[j:end])

        while a[i] < pivot:
            i += 1
        while i < j and pivot <= a[j]:
            j -= 1
        if i >= j:
            break
        assert a[j] < pivot <= a[i]
        swap(a, i, j)
        assert a[i] < pivot <= a[j]
    assert i >= j and i < end
    swap(a, i, end-1)
    assert a[i] == pivot
    # assert all(x < pivot for x in a[start:i])
    # assert all(x >= pivot for x in a[i:end])
    return i

cdef inline void swap(double* a, int i, int j):
    a[i], a[j] = a[j], a[i]

cdef void insertion_sort(double* a, int start, int end):
    cdef int i, j
    cdef double v
    for i in range(start, end):
        #invariant: [start:i) is sorted
        v = a[i]; j = i-1
        while j >= start:
            if a[j] <= v: break
            a[j+1] = a[j]
            j -= 1
        a[j+1] = v