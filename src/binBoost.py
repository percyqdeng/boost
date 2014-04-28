__author__ = 'qdengpercy'


from boost import *
import numpy as np
import scipy.io
import math
from gen_ftrs import *
from sklearn.ensemble import AdaBoostClassifier

class BoostPara:
    def __init__(self, epsi=0.01, hasDualCap = False, ratio=0.1, max_iter=100, steprule = 1):
        self.epsi = epsi
        self.hasDualCap = hasDualCap
        self.ratio = ratio
        self.max_iter = max_iter
        self.steprule = steprule


def normalize_data(x):
    Z = np.std(x, 0)
    avg = np.mean(x, 0)
    x = (x-avg[np.newaxis, :])/Z[np.newaxis, :]
    return x

class FwBoost:

    def train(self, xtr, ytr, para):
        # self.Z = np.std(xtr, 0)
        # self.mu = np.mean(xtr, 0)
        # xtr = (xtr - self.mu[np.newaxis, :])/self.Z[np.newaxis, :]
        ntr = xtr.shape[0]
        xtr = np.hstack((xtr, np.ones((ntr, 1))))
        yH = ytr[:, np.newaxis] * xtr
        self.alpha, self.d, self.gaps, self.eta = self.frank_wolfe_boosting(yH, para)


    def test(self, xte, yte):
        # normalize and add the intercept
        # xte = (xte-self.mu[np.newaxis, :])/self.Z[np.newaxis, :]
        nte = xte.shape[0]
        xte = np.hstack((xte, np.ones((nte, 1))))
        pred = np.sign(np.dot(xte, self.alpha))
        return np.mean(pred != yte)
    def frank_wolfe_boosting_2(self, para):
        '''
        frank-wolfe boost for binary classification with user defined weak learners
        '''

    def frank_wolfe_boosting(self, H, para):
        '''
        frank-wolfe boost for binary classification
            min_a max_d   d^T(-Ha) sub to:  ||a||_1\le 1
        '''
        [n, p] = H.shape
        gaps = np.zeros(para.max_iter)
        alpha = np.zeros(p)
        d0 = np.ones(n)/n
        mu = para.epsi/(2*np.log(n))
        # mu = 1
        nu = int(n * para.ratio)
        t = 0
        Ha = np.dot(H, alpha)
        # d0 = np.ones(n)/n
        while t < para.max_iter:
            d_next = prox_mapping(Ha, d0, 1/mu)
            if math.isnan(d_next[0]):
                print 'omg'
            if para.hasDualCap:
                d_next = proj_cap_ent(d_next, 1.0/nu)
            d = d_next
            dtH = np.dot(d, H)
            j = np.argmax(np.abs(dtH))
            ej = np.zeros(p)
            ej[j] = np.sign(dtH[j])
            gaps[t] = np.dot(dtH, ej-alpha)
            if para.steprule == 1:
                eta = np.maximum(0, np.minimum(1, mu*gaps[t]/np.sum(np.abs(alpha-ej))**2))
            elif para.steprule == 2:
                eta = np.maximum(0, np.minimum(1, mu*gaps[t]/LA.norm(Ha-H[:, j]*ej[j], np.inf)**2))
            else:
                '''
                do line search
                '''
                print "steprule 3, to be done"
            alpha *= (1-eta)
            alpha[j] += eta*ej[j]
            Ha *= (1-eta)
            Ha += H[:, j] * (eta*ej[j])
            if(gaps[t] < para.epsi):
                break
            t += 1

        return alpha, d, gaps, eta

def benchMark(data):
    x = data['x']
    y = data['t']
    y = np.squeeze(y)
    x = normalize_data(x)
    para = BoostPara(epsi=0.01, hasDualCap=False, ratio=0.1, max_iter=1000, steprule=2)
    # rep = data['train'].shape[0]
    rep = 1
    err_te = np.zeros(rep)

    for i in range(rep):
        trInd = data['train'][i, :] - 1
        teInd = data['test'][i, :] - 1

        xtr = x[trInd, :]
        ytr = y[trInd]
        xte = x[teInd, :]
        yte = y[teInd]

        # yH = ytr[:, np.newaxis] * xtr
        # yH = np.vstack(yH, ytr)
        booster = FwBoost()
        booster.train(xtr, ytr, para)

        # max_iter = 1000
        err_te[i] = booster.test(xte, yte)
    return booster

def toyTest():
    ntr = 2000
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000)
    para = BoostPara(epsi=0.001, hasDualCap=False, ratio=0.1, max_iter=1000, steprule=1)
    booster = FwBoost()
    booster.train(xtr, ytr, para)
    plt.plot(booster.gaps,'rx-')


if __name__ == '__main__':

    path = '/Users/qdengpercy/workspace/boost/dataset/'
    dtname = 'heartmat.mat'
    data = scipy.io.loadmat(path+dtname)
    booster = benchMark(data)
    # booster = toyTest()

