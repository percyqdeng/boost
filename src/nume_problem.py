__author__ = 'qdengpercy'
import numpy as np


n = 300
a = np.random.uniform(0, 1, n)

a[2:50] *= 10**(-14)
a[50:-1] *= 10**(-18)
a = np.sort(a)[::-1]
a /= a.sum()

z = a.sum()
for i, x in enumerate(a):
    z -= a[i]
    if z < 0:
        print "error z:%f" % z