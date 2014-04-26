__author__ = 'qdengpercy'

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.io
import os
import copy
import heapq
from boost import *
from sklearn import svm


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
        flips = np.random.binomial(1, 0.8, (ntr+nte, p))
        flips[flips == 0] = -1
        x *= flips
        # noise = np.random.normal(0,0.1,(ntr+nte,p))
        # x += noise
        xtr = x[:ntr, :]
        xte = x[ntr:, :]
        yH = ytr[:, np.newaxis] * xtr
        margin = np.min(np.dot(yH, w))
    return xtr, ytr, yH, margin, w, xte, yte


def toy_test():
    ntr = 2000
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000)
    (n,p) = xtr.shape

    num_iter = 4000
    ratio = 0.1
    row = 2
    col = 2
    hasCap = True
    epsi = 0.001
    # yH = ytr[:, np.newaxis]*xtr
    (w1, d1, gaps1, eta1, dual1, m1, total_iter1) = dboost(yH, epsi, hasCap, ratio, num_iter, 1)
    pred = np.sign(np.dot(xte, w1))
    err1 = np.mean(pred != yte)
    (w3, d3, gaps3, primal3, dual3, m3) = pdboost(yH, epsi, hasCap, ratio, num_iter)

    plt.figure()
    pred = np.sign(np.dot(xte, w3[:p]))
    err3 = np.mean(pred != yte)
    plt.subplot(row, col, 1)
    plt.plot(range(1, total_iter1+1), (np.log(gaps1[1:total_iter1+1])), 'r', label='dboost')
    plt.plot((np.log(gaps3)), 'b', label='pdboost')
    plt.plot(np.log((np.log(ntr*p)/range(len(gaps3)))),'g')
    # plt.plot((np.log(ntr*p)/range(1,total_iter1)),'g')
    plt.xlabel('# iteration')
    plt.ylabel('log of gap')
    plt.subplot(row, col, 2)

    plt.plot(range(1, total_iter1+1), (gaps1[1:total_iter1+1]), 'r', label='dboost')
    plt.plot((np.log(ntr*p)/range(len(gaps3))),'g')
    plt.plot(gaps3, 'b', label='pdboost')
    plt.xlabel('# iteration',size='small')
    plt.ylabel('gap')

    plt.subplot(row, col, 3)
    plt.title('111')
    plt.plot(range(1, total_iter1+1), (m1[1:total_iter1+1]), 'r', label='dboost')
    plt.legend(loc='best')

    plt.subplot(row,col,4)
    plt.title('pdboost')
    plt.plot(primal3,'r', label='primal')
    plt.plot(dual3,'b',label='dual')

    plt.savefig('../output/cmp.eps')

def load_data(dtname = 'ringnormmat.mat'):
    data = scipy.io.loadmat(dtname)
    x = data['x']
    y = data['t']
    trInd = data['train']
    teInd = data['test']
    return data



path = '/Users/qdengpercy/workspace/boost/dataset/'
dtname = 'heartmat.mat'
data = load_data(path+dtname)

