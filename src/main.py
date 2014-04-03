__author__ = 'qdengpercy'

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy
import heapq
from boost import *

def gen_syn(ftr_type, ntr, nte):
    '''
    get synthetic dataset as in Collin's paper
    '''
    p = 50
    mean = np.zeros(p)
    cov = np.eye(p)
    w = np.random.multivariate_normal(mean, cov, 1)
    w = np.squeeze(w)
    w /= LA.norm(w, 1)
    # w = np.ones(p)/np.float(p)
    if ftr_type == 'real':
        x = np.random.multivariate_normal(mean, cov, ntr+nte)
        y = np.sign(np.dot(x, w))
        ytr = y[:ntr]
        yte = y[ntr:]
        # add noise to data
        noise = np.random.multivariate_normal(mean, 0.8*cov, ntr+nte)
        x += noise
        xtr = x[:ntr, :]
        xte = x[ntr:, :]
        yH = ytr[:, np.newaxis] * xtr
        margin = np.min(np.dot(yH, w))
    elif ftr_type == 'disc':
        x = np.random.binomial(1, 0.5, (ntr+nte, p))
        x[x == 0] = -1
        y = np.sign(np.dot(x, w))
        ytr = y[:ntr]
        yte = y[ntr:]
        # add noise to data
        # flips = np.random.binomial(1, 0.95, (ntr+nte, p))
        # flips[flips == 0] = -1
        # x *= flips
        xtr = x[:ntr, :]
        xte = x[ntr:, :]
        yH = ytr[:, np.newaxis] * xtr
        margin = np.min(np.dot(yH, w))
    return xtr, ytr, yH, margin, w, xte, yte



ntr = 2000
(xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000)
(n,p) = xtr.shape

num_iter = 4000
k = 200
row = 2
col = 2
hasCap = False
# yH = ytr[:, np.newaxis]*xtr
(w1, d1, gaps1, eta1, dual1, m1, total_iter1) = dboost(yH, 0.01, hasCap, k, num_iter, 1)
pred = np.sign(np.dot(xte, w1))
err1 = np.mean(pred != yte)

# (w2, d2, gaps2, eta2, dual2) = dboost(yH, 0.01, hasCap, k, num_iter,  2)


# plt.plot(range(1, num_iter+1), (gaps2[1:]), 'b', label='dboost stepsize 2 gap')
# plt.subplot(row,col,2)
# plt.plot(range(1, num_iter+1), (dual1[1:]), 'g', label='dboost stepsize 1 dual')
# plt.plot(range(1, num_iter+1), (dual2[1:]), 'y', label='dboost stepsize 2 dual')

(w3, d3, gaps3, m3, total_iter3) = pdboost3(yH, 0.01, hasCap, k, num_iter)

# plt.figure()
# plt.subplot(row, col, 1)
# plt.plot(range(1, total_iter3+1), (primal3[1:total_iter3+1]), 'b', label='primal')
# plt.plot(range(1, total_iter3+1), (dual3[1:total_iter3+1]), 'b', label='dual')

plt.figure()

pred = np.sign(np.dot(xte, w3[:p]))
err3 = np.mean(pred != yte)
plt.subplot(row, col, 1)
plt.plot(range(1, total_iter1+1), (np.log(gaps1[1:total_iter1+1])), 'r', label='dboost')
plt.plot(range(1, total_iter3+1), (np.log(gaps3[1:total_iter3+1])), 'b', label='pdboost')
plt.xlabel('# iteration')
plt.ylabel('log of gap')
plt.subplot(row, col, 2)

plt.plot(range(1, total_iter1+1), (gaps1[1:total_iter1+1]), 'r', label='dboost')
plt.plot(range(1, total_iter3+1), (gaps3[1:total_iter3+1]), 'b', label='pdboost')
plt.xlabel('# iteration',size='small')
plt.ylabel('gap')

plt.subplot(row, col, 3)
plt.title('111')
plt.plot(range(1, total_iter1+1), (m1[1:total_iter1+1]), 'r', label='dboost')
plt.plot(range(1, total_iter3+1), (m3[1:total_iter3+1]), 'b', label='pdboost')
plt.legend(loc='best')

plt.savefig('../output/cmp.eps')




