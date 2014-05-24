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


def gen_syn(ntr, nte, ftr_type="disc", has_noise=False):
    """
    get synthetic dataset as in Collin's paper
    """
    p = 100
    mean = np.zeros(p)
    cov = np.eye(p)
    w = np.random.multivariate_normal(mean, cov, 1)

    w = np.squeeze(w)
    w /= LA.norm(w, 1)
    # w = np.ones(p)/np.float(p)
    if ftr_type == 'real':
        x = np.random.multivariate_normal(mean, cov, ntr + nte)
        x[x<-1] = -1
        x[x>1] = 1
        y = np.sign(np.dot(x, w))
        ytr = y[:ntr]
        yte = y[ntr:]
        # add noise to data
        # noise = np.random.multivariate_normal(mean, 0.8 * cov, ntr + nte)
        # x += noise
        xtr = x[:ntr, :]
        xte = x[ntr:, :]
        yH = ytr[:, np.newaxis] * xtr
        margin = np.min(np.dot(yH, w))
        print 'real margin %.4f ' % margin
    elif ftr_type == 'disc':
        x = np.random.binomial(1, 0.5, (ntr + nte, p))
        x[x == 0] = -1
        y = np.sign(np.dot(x, w))
        ytr = y[:ntr]
        yte = y[ntr:]
        yH = ytr[:, np.newaxis] * x[:ntr, :]
        margin = np.min(np.dot(yH, w))
        print 'real margin %.4f ' % margin
        # add noise to data
        if has_noise:
            """
            revert the feature of 30% data
            """
            # flips = np.random.binomial(1, 0.9, (ntr + nte, p))
            # flips[flips == 0] = -1
            # x *= flips
            # noise = np.random.normal(0, 0.1, (ntr + nte, p))
            # x += noise
            # x =
            ind = np.random.choice(p, int(0.3*p), replace=False)
            x2 = x.copy()
            x2[:, ind] = - x2[:, ind]
            x = np.hstack((x, x2))
        xtr = x[:ntr, :]
        xte = x[ntr:, :]


        yH = ytr[:, np.newaxis] * xtr
        # margin = np.min(np.dot(yH, w))
    return xtr, ytr,  xte, yte, w



def load_data(dtname='ringnormmat.mat'):
    data = scipy.io.loadmat(dtname)
    x = data['x']
    y = data['t']
    trInd = data['train']
    teInd = data['test']
    return data


if __name__ == '__main__':
    if os.name == "posix":
        path = '/Users/qdengpercy/workspace/boost/dataset/'
    elif os.name == "nt":
        path = "..\\dataset\\"
    # dtname = 'heartmat.mat'
    # data = load_data(path+dtname)
    toy_test2()
