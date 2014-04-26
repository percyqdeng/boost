__author__ = 'qdengpercy'


from boost import *
import numpy as np
import scipy.io


def BenchMark(data):
    x = data['x']
    y = data['t']
    # trInd = data['train']
    # teInd = data['test']

    rep = data['train'].shape[0]
    rep = 1
    err_te = np.zeros(rep)
    for i in range(rep):
        trInd = data['train'][i,:] - 1
        teInd = data['test'][i,:] - 1

        xtr = x[trInd, :]
        ytr = y[trInd]
        xte = x[teInd, :]
        yte = y[teInd]

        yH = ytr[:, np.newaxis] * xtr
        yH = np.vstack(yH, ytr)

        epsi = 0.001
        hasCap = False
        ratio = 0.1
        max_iter = 1000
        alpha, d, gaps = fwboost(yH, epsi, hasCap, ratio, max_iter)

        pred = np.sign(np.dot(xte, alpha))
        err_te[i] = np.mean(pred != yte)


if __name__ == '__main__':

    path = '/Users/qdengpercy/workspace/boost/dataset/'
    dtname = 'heartmat.mat'
    data = scipy.io.loadmat(path+dtname)
    BenchMark(data)