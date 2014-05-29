# import pyximport
# pyximport.install()
from timeit import default_timer as timer

from quicksort import sort as qsort

def measure(f):
    def wrapper(L):
        start = timer()
        try: f(L)
        finally:
            return timer() - start
    return wrapper

measure_qsort = measure(qsort)

def run_tests():
    import numpy as np
    for L in map(np.array, [[], [1.], [1.]*10]):
        old = L.copy()
        qsort(L)
        assert np.all(old == L)

    for n in np.r_[np.arange(2, 10),11,101,100,1000]:
        L = np.random.random(int(n))
        old = sorted(L.copy())
        t = float("+inf")
        for _ in xrange(10):
            np.random.shuffle(L)
            t = min(measure_qsort(L), t)
            assert np.allclose(old, L), L
        report_time(n, t)

    for n in np.r_[1e4,1e5,1e6]:
        t = float("+inf")
        for _ in range(3):
            t = min(measure_qsort(np.random.random_sample(n)), t)
        report_time(n, t)

def report_time(n, t):
    print "N=%09d took us %.2g\t%s" % (n, t*1000, "ms")

if __name__=="__main__":
    run_tests()