# coding=utf-8
__author__ = 'qdengpercy'

import os
import numpy as np
import scipy.io
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from boost import *
from gen_ftrs import *

# class BoostPara:
#     def __init__(self, epsi=0.01, hasDualCap = False, ratio=0.1, max_iter=100, steprule = 1):
#         self.epsi = epsi
#         self.hasDualCap = hasDualCap
#         self.ratio = ratio
#         self.max_iter = max_iter
#         self.steprule = steprule


def preprocess_data(x):
    Z = np.std(x, 0)
    avg = np.mean(x, 0)
    x = (x - avg[np.newaxis, :]) Z[np.newaxis, :]
    return x


class AdaFwBoost:
    """Adaptive Frank-Wolfe Boosting method,
    Parameters:
    ------------
    epsi ：optimization tolerance,
    has_dcap : boolean
        has capped probability in the dual variable (distribution of samples)
    has_pcap : boolean
        has capped weight in the primal variable (weight of weak learners)
    ratio : float the capped probability
    steprule : int, the stepsize choice in FW method
        1 : derive from quadratic surrogate function one
        2 : quadratic surrogate function two
        3 : line search
    max_iter :  int,
        maximal iteration number, or the maximal number of weak learners in the adaptive setting
    learners : list type, set of weak learners, (used in the adaptive fw boosting)
    alpha : weights of weak learners
    err_tr : array type, training error of each iteration
    err_wl : array type, training error of each weak learner
    _primal_obj :  array type, primal objective in each iteration
    _margin :  array type, margin in each iteration
    _dual_obj : array type, dual objective in each iteration
    _gap : array type, primal-dual gap in each iteration
    """
    def __init__(self, epsi=0.01, has_dcap=False, ratio=0.1, max_iter=100, steprule=1):
        """

        :type max_iter: int
        """
        self.epsi = epsi
        self.has_dcap = has_dcap
        self.ratio = ratio
        self.max_iter = max_iter
        self.steprule = steprule

        self._primal_obj = []
        self._dual_obj = []
        self._margin = []
        self._gap = []
        self.learners = []
        self.err_tr = []
        self.err_wl = []
        self.alpha = np.zeros(max_iter)

    def train(self, xtr, ytr):
        self.ada_fw_boosting(xtr, ytr)

    def test(self, xte, yte):
        pred = self.pred(xte)
        err = np.mean(pred != yte)
        return err

    def adaboost(self, x, y):
        # for t in range(self.max_iter):
        pass

    def ada_fw_boosting(self, x, y):
        """
        frank-wolfe boost for binary classification with user defined weak learners, choose decision stump
        as default
        x : array of shape n*p
        y : array of shape n*1
        """
        n, p = x.shape
        d = np.ones(n) / n
        Ha = np.zeros(n)  # the current margin
        # z = np.zeros(n)   # the current output of learner
        mu = self.epsi / (2 * np.log(n))
        # mu = 1
        nu = int(n * self.ratio)
        for t in range(self.max_iter):
            if t % (self.max_iter/10) == 0:
                print "iter "+str(t)
            d_next = prox_mapping(Ha, d, 1 / mu)
            assert not math.isnan(d_next[0])
            if self.has_dcap:
                d_next = proj_cap_ent(d_next, 1.0 / nu)
            d = d_next
            assert np.abs(1-d.sum()) < 0.0001
            weak_learner = DecisionTreeClassifier(max_depth=1)
            weak_learner.fit(x, y, sample_weight=d)
            self.learners.append(weak_learner)
            pred = weak_learner.predict(x)
            h = pred * y
            self.err_wl.append(np.dot(h<=0, d))
            self._gap.append(np.dot(d, h - Ha))
            if self.has_dcap:
                min_margin = ksmallest(Ha, nu)
                self._margin.append(np.mean(min_margin))
            else:
                self._margin.append(np.min(Ha))
            self._primal_obj.append(mu * np.log(1.0 / n * np.sum(np.exp(-Ha / mu))))
            self.err_tr.append(np.mean(Ha <= 0))
            if self.steprule == 1:
                eta = np.maximum(0, np.minimum(1, mu * self._gap[-1] / (1 + self.alpha.sum()) ** 2))
            elif self.steprule == 2:
                eta = np.maximum(0, np.minimum(1, mu * self._gap[-1] / (LA.norm(h - Ha, np.inf)) ** 2))
            else:
                """
                line search
                """
            self.alpha *= (1 - eta)
            self.alpha[t] = eta
            Ha *= (1-eta)
            Ha += eta * h
            if self._gap[-1] < self.epsi:
                break
        self.d = d
        # gaps = gaps[:t]
        # self._margin = self._margin[:t]
        # prim_obj = prim_obj[:t]
        # dual_obj = dual_obj[:t]

    def pred(self, x):
        n, p = x.shape
        h = np.zeros(n)
        for t, weak_learner in enumerate(self.learners):
            pred = weak_learner.predict(x)
            h += pred * self.alpha[t]
        return np.sign(h)

    def plot_result(self):
        r = 2
        c = 2
        plt.subplot(r, c, 1)
        plt.plot(self._gap, 'rx-', markersize=0.3, label='gap')
        plt.title('primal-dual gap')
        plt.subplot(r, c, 2)
        plt.plot(self._margin, 'bo-', markersize=0.3)
        plt.title('margin')
        plt.subplot(r, c, 3)
        plt.plot(self.err_tr, 'x-', color='b', markersize=0.2, label="train err")
        plt.plot(self.err_wl, 'o-', color='r', markersize=0.2, label="wl err")
        plt.savefig('result.eps', format='eps')
        plt.title('training error')
        plt.legend()
        plt.tight_layout()
        plt.savefig('adafwboost.pdf')


class FwBoost:
    """Frank-Wolfe Boosting method as a saddle point matrix game
    Parameters:
    ------------
    epsi ：optimization tolerance,
    has_dcap : boolean
        has capped probability in the dual variable (distribution of samples)
    has_pcap : boolean
        has capped weight in the primal variable (weight of weak learners)
    ratio : float the capped probability
    steprule : int, the stepsize choice in FW method
        1 : derive from quadratic surrogate function one
        2 : quadratic surrogate function two
        3 : line search
    max_iter :  int,
        maximal iteration number, or the maximal number of weak learners in the adaptive setting
    alpha : array type, the weights of weak learner
    err_tr : array type, training error of each iteration
    _primal_obj : array type, primal objective in each iteration
    _margin : array type, margin in each iteration
    _dual_obj : array type, dual objective in each iteration
    _gap : array type, primal-dual gap in each iteration
    """

    def __init__(self, epsi=0.01, has_dcap=False, ratio=0.1, max_iter=100, steprule=1):
        self.epsi = epsi
        self.has_dcap = has_dcap
        self.ratio = ratio
        self.max_iter = max_iter
        self.steprule = steprule
        self._primal_obj = []
        self._dual_obj = []
        self._margin = []
        self._gap = []
        self.err_tr = []
        self.alpha = []

    def train(self, xtr, ytr):
        # self.Z = np.std(xtr, 0)
        # self.mu = np.mean(xtr, 0)
        # xtr = (xtr - self.mu[np.newaxis, :])/self.Z[np.newaxis, :]
        ntr = xtr.shape[0]
        xtr = np.hstack((xtr, np.ones((ntr, 1))))
        yH = ytr[:, np.newaxis] * xtr
        self.fw_boosting(yH)

    def test(self, xte, yte):
        # normalize and add the intercept
        # xte = (xte-self.mu[np.newaxis, :])/self.Z[np.newaxis, :]
        nte = xte.shape[0]
        xte = np.hstack((xte, np.ones((nte, 1))))
        pred = np.sign(np.dot(xte, self.res['alpha']))
        return np.mean(pred != yte)

    def fw_boosting(self, H):
        """
        frank-wolfe boost for binary classification with weak learner as matrix H
        min_a max_d   d^T(-Ha) sub to:  ||a||_1\le 1
        """
        [n, p] = H.shape
        # self.alpha = np.ones(p)/p
        self.alpha = np.zeros(p)
        d0 = np.ones(n) / n
        mu = self.epsi / (2 * np.log(n))
        # mu = 1
        nu = int(n * self.ratio)
        t = 0
        Ha = np.dot(H, self.alpha)

        # d0 = np.ones(n)/n
        for t in range(self.max_iter):
            if t % n == 0:
                print "iter "+str(t)
            d_next = prox_mapping(Ha, d0, 1 / mu)
            assert not math.isnan(d_next[0])
            if self.has_dcap:
                d_next = proj_cap_ent(d_next, 1.0 / nu)
            d = d_next
            dtH = np.dot(d, H)
            j = np.argmax(np.abs(dtH))
            ej = np.zeros(p)
            ej[j] = np.sign(dtH[j])
            self._gap.append(np.dot(dtH, ej - self.alpha))
            if self.has_dcap:
                min_margin = ksmallest(Ha, nu)
                self._margin.append(np.mean(min_margin))
            else:
                self._margin.append(np.min(Ha))
            self._primal_obj.append(mu * np.log(1.0 / n * np.sum(np.exp(-Ha / mu))))
            self._dual_obj.append(-np.max(np.abs(dtH)) - mu * np.dot(d, np.log(d)) + mu * np.log(n))
            self.err_tr.append(np.mean(Ha <= 0))
            if self.steprule == 1:
                eta = np.maximum(0, np.minimum(1, mu * self._gap[-1] / np.sum(np.abs(self.alpha - ej)) ** 2))
            elif self.steprule == 2:
                eta = np.maximum(0, np.minimum(1, mu * self._gap[-1] / LA.norm(Ha - H[:, j] * ej[j], np.inf) ** 2))
            else:
                #
                # do line search
                #
                print "steprule 3, to be done"
            self.alpha *= (1 - eta)
            self.alpha[j] += eta * ej[j]
            Ha *= (1 - eta)
            Ha += H[:, j] * (eta * ej[j])
            if self._gap[-1] < self.epsi:
                break


    def pdboost(self, H):
        """
        primal-dual boost with capped probability ||d||_infty <= 1/k
        """
        print '----------------primal-dual boost-------------------'
        H = np.hstack((H, -H))
        (n, p) = H.shape
        nu = int(n * self.ratio)
        gaps = np.zeros(self.max_iter)
        margin = np.zeros(self.max_iter)
        primal_val = np.zeros(self.max_iter)
        dual_val = np.zeros(self.max_iter)
        # gaps[0] = 100
        showtimes = 5
        d = np.ones(n) / n

        d_bar = np.ones(n) / n
        a_bar = np.ones(p) / p
        a = np.ones(p) / p
        # a_bar = a
        a_tilde = np.ones(p) / p
        # d_tilde = np.zeros(p)
        theta = 1
        sig = 1
        tau = 1
        t = 0
        while t < self.max_iter:

            d = prox_mapping(np.dot(H, a_tilde), d, tau, 2)

            if self.has_dcap:
                d2 = proj_cap_ent(d, 1.0 / nu)
                # d_new = d_new/d_new.sum()
                if np.abs(d.sum() - d2.sum()) > 0.0001:
                    print 'error'
                d = d2
            d_tilde = d
            dtH = np.dot(d_tilde, H)
            a_new = prox_mapping(-dtH, a, sig, 2)
            # a_new = proj_l1ball(tmp, 1)
            a_tilde = a_new + theta * (a_new - a)
            a = a_new
            d_bar *= t / (t + 1.0)
            d_bar += 1.0 / (t + 1) * d
            a_bar *= t / (t + 1.0)
            a_bar += 1.0 / (t + 1) * a

            if self.has_dcap:
                Ha = np.dot(H, a_bar)
                min_margin = ksmallest(Ha, nu)
                primal_val[t] = -np.mean(min_margin)
            else:
                primal_val[t] = - np.min(np.dot(H, a_bar))
            margin[t] = -primal_val[t]
            dual_val[t] = -np.max(np.dot(d_bar, H))
            gaps[t] = primal_val[t] - dual_val[t]
            if t % np.int(self.max_iter / showtimes) == 0:
                print 'iter ' + str(t) + ' ' + str(gaps[t])
            # print 'primal: '+str(-(ksmallest(Ha, k)).sum()/k)
            # print 'dual: '+str(-LA.norm(dtH, np.inf))
            if gaps[t] < self.epsi:
                break
            t += 1
        gaps = gaps[:t]
        primal_val = primal_val[:t]
        dual_val = dual_val[:t]
        return a_bar, d_bar, gaps, primal_val, dual_val, margin

    def plot_result(self):
        r = 2
        c = 2
        plt.subplot(r, c, 1)
        plt.plot(self._gap, 'rx-', markersize=0.3, label='gap')
        plt.title('primal-dual gap')
        plt.subplot(r, c, 2)
        plt.plot(self._margin, 'bo-', markersize=0.3)
        plt.title('margin')
        plt.subplot(r, c, 3)
        plt.plot(self._primal_obj, 'rx-', markersize=0.3, label='primal')
        plt.plot(self._dual_obj, color='g', marker='o', markersize=0.3, label='dual')
        plt.title('primal objective')
        plt.legend()
        plt.subplot(r, c, 4)
        plt.plot(self.err_tr, 'x-', color='b', markersize=0.3, label="train err")
        plt.savefig('result.eps', format='eps')
        plt.title('training error')
        plt.legend()
        plt.tight_layout()


def benchmark(data):
    x = data['x']
    y = data['t']
    y = np.squeeze(y)
    x = preprocess_data(x)
    # para = BoostPara(epsi=0.01, has_dcap=False, ratio=0.11, max_iter=20000, steprule=2)
    # rep = data['train'].shape[0]
    rep = 1
    err_te = np.zeros(rep)
    err_tr = np.zeros(rep)
    for i in range(rep):
        trInd = data['train'][i, :] - 1
        teInd = data['test'][i, :] - 1

        xtr = x[trInd, :]
        ytr = y[trInd]
        xte = x[teInd, :]
        yte = y[teInd]
        booster = FwBoost(epsi=0.01, has_dcap=False, ratio=0.11, max_iter=20000, steprule=2)
        booster.train(xtr, ytr)

        # max_iter = 1000
        err_te[i] = booster.test(xte, yte)
        err_tr[i] = booster.test(xtr, ytr)
        print "rounds " + str(i + 1) + "err_tr " + str(err_tr[i]) + " err_te " + str(err_te[i])
    # plt.plot(booster.result['gap'], 'bx-')
    return booster, err_te


def test_adafwboost():
    ntr = 500
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000)
    booster = AdaFwBoost(epsi=0.001, has_dcap=False, ratio=0.1, max_iter=100, steprule=1)
    booster.train(xtr, ytr)
    # plt.plot(booster.gaps,'rx-')
    booster.plot_result()
    return booster


def test_fwboost():
    ntr = 500
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000, hasNoise=True)
    # para = BoostPara(epsi=0.001, has_dcap=False, ratio=0.1, max_iter=500000, steprule=1)
    booster = FwBoost(epsi=0.001, has_dcap=True, ratio=0.1, max_iter=5000, steprule=2)
    booster.train(xtr, ytr)
    # plt.plot(booster.gaps,'rx-')
    booster.plot_result()
    return booster


def plot_2d_data(data):
    x = data['x']
    y = data['t']
    y = np.squeeze(y)
    plt.subplot(2, 2, 1)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', c='r', label='+')
    plt.subplot(2, 2, 2)
    plt.scatter(x[y == -1, 0], x[y == -1, 1], marker='x', c='b', label='-')
    plt.subplot(2, 2, 3)
    plt.scatter(x[:, 0], x[:, 1])
    plt.title('banana')

# plt.show()
# plt.savefig('2dplot.eps')

if __name__ == '__main__':
    if os.name == "nt":
        path = "..\\dataset\\"
    elif os.name == "posix":
        path = '/Users/qdengpercy/workspace/boost/dataset/'

    dtname = 'bananamat.mat'
    # data = scipy.io.loadmat(path+dtname)
    # plot_2d_data(data)
    # plt.figure()
    # booster, err = benchmark(data)
    # booster = test_fwboost()
    booster = test_adafwboost()

