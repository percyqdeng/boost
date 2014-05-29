__author__ = 'qdengpercy'

import numpy as np
import ksmall
import time
import timeit
import random
# heapfun.demo()

# heapfun.ksmallest()

# arr = np.random.normal(0, 1, 10)

# arr = np.random.normal(0, 1, 10000000)
# start = time.time()
# v = ksmall.ksmallest(arr, 100)
# print time.time() - start
#
# start = time.time()
# ksmall.heapsort(arr)
# print time.time() - start

ksmall.cmp_time(10000000)