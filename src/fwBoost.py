# coding=utf-8
__author__ = 'qdengpercy'

import os
import numpy as np
import math
import scipy.io
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from boost import *
from gen_ftrs import *
import frank_wolfe_cy as fw_cy

class AdaFwBoost(Boost):
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
    mu : smoothing parameter
    err_tr : array type, training error of each iteration
    err_wl : array type, training error of each weak learner
    _primal_obj :  array type, primal objective in each iteration
    _margin :  array type, margin in each iteration
    _dual_obj : array type, dual objective in each iteration
    _gap : array type, primal-dual gap in each iteration
    """
    def __init__(self, epsi=0.01, has_dcap=False, ratio=0.1,  steprule=1):
        """

        :type max_iter: int
        """
        self.epsi = epsi
        self.has_dcap = has_dcap
        self.ratio = ratio

        self.steprule = steprule
        self.primal_obj = []
        self.dual_obj = []
        self.margin = []
        self.gap = []
        self.learners = []
        self.err_tr = []
        self.err_wl = []
        # self.alpha = np.zeros(max_iter)
        self.mu = 1

    def train(self, xtr, ytr):
        self.ada_fw_boosting(xtr, ytr)

    def test(self, xte, yte):
        pred = self.pred(xte)
        err = np.mean(pred != yte)
        return err

    def adaboost(self, x, y):

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
        self.mu = self.epsi / (2 * np.log(n))
        max_iter = int(np.log(n) / self.epsi**2)
        max_iter = 4000
        self.alpha = np.zeros(max_iter)
        # mu = 1
        nu = int(n * self.ratio)
        for t in range(max_iter):
            if t % (max_iter/10) == 0:
                print "iter "+str(t)
            d_next = prox_mapping(Ha, d, 1 / self.mu)
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
            self.gap.append(np.dot(d, h - Ha))
            if self.has_dcap:
                min_margin = ksmallest(Ha, nu)
                self.margin.append(np.mean(min_margin))
            else:
                self.margin.append(np.min(Ha))
            self.primal_obj.append(self.mu * np.log(1.0 / n * np.sum(np.exp(-Ha / self.mu))))
            self.err_tr.append(np.mean(Ha <= 0))
            if self.steprule == 1:
                eta = np.maximum(0, np.minimum(1, self.mu * self.gap[-1] / (1 + self.alpha.sum()) ** 2))
            elif self.steprule == 2:
                eta = np.maximum(0, np.minimum(1, self.mu * self.gap[-1] / (LA.norm(h - Ha, np.inf)) ** 2))
            else:
                """
                line search
                """
            self.alpha *= (1 - eta)
            self.alpha[t] = eta
            Ha *= (1-eta)
            Ha += eta * h
            if self.gap[-1] < self.epsi:
                break
        self.d = d
        # gaps = gaps[:t]
        # self.margin = self.margin[:t]
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
        nBins = 6
        plt.subplot(r, c, 1)
        plt.plot(np.log(self.gap), 'rx-', markersize=0.3, label='gap')
        T = len(self.gap)
        bound = 1/(self.mu * np.arange(1, 1+T))
        plt.plot(np.log(bound), 'bo-', markersize=0.2, label='bound')
        plt.title('log primal-dual gap')
        plt.legend(loc='best')
        plt.locator_params(axis='x', nbins=nBins)
        plt.subplot(r, c, 2)
        plt.plot(self.margin, 'bo-', markersize=0.3)
        plt.title('margin')
        plt.locator_params(axis='x', nbins=nBins)
        plt.subplot(r, c, 3)
        plt.plot(self.err_tr, 'b-',  label="train err")
        plt.plot(self.err_wl, 'r-', label="wl err")
        plt.savefig('result.eps', format='eps')
        plt.title('training error')
        plt.legend(loc='best')
        plt.locator_params(axis='x', nbins=nBins)
        plt.tight_layout()
        plt.savefig('adafwboost.png')


class FwBoost(Boost):
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
    alpha : array type, the weights of weak learner
    mu : smoothing parameter
    err_tr : array type, training error of each iteration
    _primal_obj : array type, primal objective in each iteration
    _margin : array type, margin in each iteration
    _gap : array type, Frank-Wolfe gap in each iteration
    """

    def __init__(self, epsi=0.01, has_dcap=False, ratio=0.1, steprule=1):
        self.epsi = epsi
        self.has_dcap = has_dcap
        self.ratio = ratio
        self.steprule = steprule
        self.primal_obj = []
        self.dual_obj = []
        self.margin = []
        self.gap = []
        self.err_tr = []
        self.alpha = []
        self.num_zeros = []
        self.iter_num = []


    def to_name(self):
        return "fwboost"

    def train(self, xtr, ytr, codetype="cy", approx_margin=True, early_stop=False, ftr='raw'):
        """
        Argument:
        codetype:  string,
            "cy", call the faster cython function; "py", call the python code
        approx_margin:  bool type,
            True, approximately maximize the margin; False, solve exponential loss
        early_stop: bool
            True, for debug use; False, normal use
        ftr: string, feature type
            'wl', features from weak learner
            'raw', raw data
        """
        print "-------fw boost training---------"
        ntr = xtr.shape[0]
        if ftr == 'raw':
            xtr = self.process_train_data(xtr)
            xtr = np.hstack((xtr, np.ones((ntr, 1))))
            y_h = ytr[:, np.newaxis] * xtr
        elif ftr == 'wl':
            y_h = ytr[:, np.newaxis] * xtr
        else:
            print "error, fwboost: train()"
        if approx_margin:
            self.mu = self.epsi / (2 * np.log(ntr))
        else:
            self.mu = 1

        if early_stop:
            self.max_iter = 10
        else:
            self.max_iter = int(108*np.log(ntr) / self.epsi**2)

        if codetype == "cy":
            self.alpha, self.primal_obj, self.gap, self.err_tr, self.margin, self.iter_num, self.num_zeros = \
                fw_cy.fw_boost_cy(y_h, np.float32(self.epsi), self.ratio, self.steprule, self.has_dcap, self.mu, self.max_iter)
        elif codetype == 'py':
            self._fw_boosting(y_h)


    def test(self, xte, yte):
        # normalize and add the intercept
        # xte = (xte-self.mu[np.newaxis, :])/self.Z[np.newaxis, :]
        nte = xte.shape[0]
        # xte = self._process_test_data(xte)
        # xte = np.hstack((xte, np.ones((nte, 1))))
        pred = np.sign(np.dot(xte, self.alpha))
        return np.mean(pred != yte)

    def _fw_boosting(self, H):
        """
        frank-wolfe boost for binary classification with weak learner as matrix H
        min_a max_d   d^T(-Ha) sub to:  ||a||_1\le 1
        capped probability constraint:
            max_i d_i <= 1/(n*ratio)
        Args:
            H : output matrix of weak learners
        """

        [n, p] = H.shape
        used = np.zeros(p)
        total_zeros = p
        # self.alpha = np.ones(p)/p
        self.alpha = np.zeros(p)
        d0 = np.ones(n) / n
        # self.mu = self.epsi / (2 * np.log(n))

        # max_iter = 100
        # mu = 1
        nu = int(n * self.ratio)
        if self.max_iter < 100:
            delta = 1
        else:
            delta = 40
        h_a = np.dot(H, self.alpha)
        # d0 = np.ones(n)/n
        print " fw-boosting(python): maximal iter #: "+str(self.max_iter)
        for t in range(self.max_iter):
            d_next = prox_mapping(h_a, d0, 1 / self.mu)

            # assert not math.isnan(d_next.max())
            # if math.isnan(d_next.max()) or d_next.min()<0 or d_next.max()>1:
            #     print d_next.max()
            if self.has_dcap:
                d = proj_cap_ent(d_next, 1.0 / nu)
                if d.max() > 1.0/nu or d.min()<0:
                    d = proj_cap_ent(d_next, 1.0 / nu)
                    print 'dmax %f, cap, %f' % (d.max(), 1.0/nu)
                    # assert d.max() <= 1.0/nu
            else:
                d = d_next
            # if math.isnan(d.max()) or d.min() < 0 or d.max()>1:
            #     print d.max()
            dt_h = np.dot(d, H)
            j = np.argmax(np.abs(dt_h))
            if used[j] == 0:
                used[j] = 1
                total_zeros -= 1
            ej = np.zeros(p)
            ej[j] = np.sign(dt_h[j])
            curr_gap = np.dot(dt_h, ej - self.alpha)
            # print 'iter %s, gap: %s ' %(t, curr_gap)
            if t % delta == 0:
                self.iter_num.append(t)
                if self.has_dcap:
                    min_margin = ksmallest(h_a, nu)
                    self.margin.append(np.mean(min_margin))
                else:
                    self.margin.append(np.min(h_a))
                self.gap.append(curr_gap)
                self.err_tr.append(np.mean(h_a <= 0))
                self.primal_obj.append(self.mu * np.log(1.0 / n * np.sum(np.exp(-h_a / self.mu))))
                self.num_zeros.append(total_zeros)
            # self.dual_obj.append(-np.max(np.abs(dt_h)) - self.mu * np.dot(d, np.log(d)) + self.mu * np.log(n))
            if self.steprule == 1:
                eta = np.maximum(0, np.minimum(1, self.mu * curr_gap / np.sum(np.abs(self.alpha - ej)) ** 2))
            elif self.steprule == 2:
                eta = np.maximum(0, np.minimum(1, self.mu * curr_gap / LA.norm(h_a - H[:, j] * ej[j], np.inf) ** 2))
            else:
                # do line search
                #
                print "steprule 3, to be done"
            self.alpha *= (1 - eta)
            self.alpha[j] += eta * ej[j]
            h_a *= (1 - eta)
            h_a += H[:, j] * (eta * ej[j])
            if curr_gap < self.epsi:
                break
            if t % (self.max_iter/10) == 0:
                print ("iter# %d, gap %.5f, dmax %f" % (t, curr_gap, d.max()))
        # self.d = d

    def plot_result(self):
        r = 2
        c = 2
        nBins = 6
        plt.figure()
        plt.subplot(r, c, 1)
        plt.plot(np.log(self.gap), 'r-', label='gap')
        T = len(self.gap)
        bound = 1/(self.mu * np.arange(1, 1+T))
        plt.plot(np.log(bound), 'b-', label='bound')
        plt.title('log primal-dual gap')
        plt.legend(loc='best')
        plt.locator_params(axis='x', nbins=nBins)
        plt.subplot(r, c, 2)
        plt.plot(self.margin, 'b-')
        plt.title('margin')
        plt.locator_params(axis='x', nbins=nBins)
        plt.subplot(r, c, 3)
        plt.plot(self.primal_obj, 'r-', label='primal')
        # plt.plot(self.dual_obj, color='g', label='dual')
        plt.title('primal objective')
        plt.legend(loc='best')
        plt.locator_params(axis='x', nbins=nBins)
        plt.subplot(r, c, 4)
        plt.plot(self.err_tr, 'b-', label="train err")
        plt.savefig('result.eps', format='eps')
        plt.title('training error')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.locator_params(axis='x', nbins=nBins)
        plt.savefig('fwboost.png', format='png')


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


if __name__ == '__main__':
    # if os.name == "nt":
    #     path = "..\\dataset\\"
    # elif os.name == "posix":
    #     path = '/Users/qdengpercy/workspace/boost/dataset/'
    #
    # dtname = 'bananamat.mat'
    # data = scipy.io.loadmat(path+dtname)
    # plot_2d_data(data)
    # plt.figure()
    # booster, err = benchmark(data)
    # booster = test_fwboost()
    # booster = test_adafwboost()
    pass
