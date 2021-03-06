__author__ = 'qdengpercy'
# coding=utf8
from boost import *
import matplotlib.pyplot as plt
import pdb


class ParaBoost(Boost):
    """Primal-Dual Parallel Boosting method as a saddle point matrix game
    Parameters:
    ------------
    epsi ：optimization tolerance
    has_dcap : boolean
        has capped probability in the dual variable (distribution of samples)
    has_pcap : boolean, has capped weight in the primal variable (weight of weak learners)
    ratio : float the capped probability
    alpha : array type, the weights of weak learner
    err_tr : array type, training error of each iteration
    primal_obj : array type, primal objective in each iteration
    margin : array type, margin in each iteration
    dual_obj : array type, dual objective in each iteration
    gap : array type, primal-dual gap in each iteration
    """

    def __init__(self, epsi=0.01, has_dcap=False, ratio=0.1):
        self.epsi = epsi
        self.has_dcap = has_dcap
        self.ratio = ratio
        self.primal_obj = []
        self.dual_obj = []
        self.margin = []
        self.gap = []
        self.err_tr = []
        self.alpha = []
        self.iter_num = []
        self.max_iter = -1

    def to_name(self):
        return "paraboost"

    def train(self, xtr, ytr, max_iter=None, ftr='wl'):
        if ftr == 'raw':
            h = self._process_train_data(xtr)
            # h = np.hstack((h, np.ones((ntr, 1))))
            y_h = ytr[:, np.newaxis] * h
        elif ftr == 'wl':
            y_h = ytr[:, np.newaxis] * xtr
        else:
            pass
        n, p = y_h.shape
        if max_iter is None:
            self.max_iter = int(np.log(n * p) / self.epsi)
        else:
            self.max_iter = max_iter
        self._para_boosting(y_h)

    def test_h(self, xte, ftr='wl'):
        if ftr == 'wl':
            pred = np.sign(np.dot(xte, self.alpha))
        else:
            h = self._process_test_data(xte)
            pred = np.sign(np.dot(h, self.alpha))
        return pred

    def test(self, xte, yte):

        nte = xte.shape[0]
        # xte = self._process_test_data(xte)
        # xte = np.hstack((xte, np.ones((nte, 1))))
        pred = np.sign(np.dot(xte, self.alpha))
        return np.mean(pred != yte)

    def _para_boosting(self, H):
        """
        primal-dual boost with capped probability ||d||_infty <= 1/k
        a, d : primal dual variables to be updated,
        a_tilde, d_tilde : intermediate variables,
        a_bar, d_bar : average value as output.
        """
        # print '----------------primal-dual boost-------------------'
        H = np.hstack((H, -H))
        # H_ft = np.asfortranarray((H.copy()))
        (n, p) = H.shape
        self.c = np.log(n*p)
        nu = int(n * self.ratio)

        if self.max_iter < 50:
            delta = 1
        else:
            delta = 40
        d = np.ones(n) / n
        d_bar = np.ones(n) / n
        a_bar = np.ones(p) / p
        a = np.ones(p) / p
        h_a = np.sum(H, axis=1) / p
        h_a_bar = h_a.copy()
        # a_bar = a
        # a_tilde = np.ones(p) / p
        h_a_tilde = h_a.copy()
        # d_tilde = np.zeros(p)
        theta = 1
        sig = 1
        tau = 1
        t = 0
        logscale = 0
        for t in range(self.max_iter):
            d = prox_mapping(h_a_tilde, d, tau, 2)
            if self.has_dcap:
                d2 = proj_cap_ent(d, 1.0 / nu)
                # d_new = d_new/d_new.sum()
                if np.abs(d.sum() - d2.sum()) > 0.0001:
                    print 'error'
                d = d2
            d_tilde = d
            dtH = np.dot(d_tilde, H)
            # dtH = np.dot(H.T, d_tilde)
            a_new = prox_mapping(-dtH, a, sig, 2)
            h_a_new = np.dot(H, a_new)
            # a_tilde = a_new + theta * (a_new - a)
            h_a_tilde = (1+theta) * h_a_new - theta * h_a
            a = a_new
            h_a = h_a_new
            d_bar *= t / (t + 1.0)
            d_bar += 1.0 / (t + 1) * d
            a_bar *= t / (t + 1.0)
            a_bar += 1.0 / (t + 1) * a
            # h_a_bar = np.dot(H, a_bar)
            h_a_bar = t / (t + 1.0) * h_a_bar + 1.0/(t+1) * h_a
            if int(np.log(t+1)) == logscale:
                logscale += 1
                self.iter_num.append(t)
                if self.has_dcap:
                    min_margin = ksmallest2(h_a_bar, nu)
                    self.primal_obj.append(-np.mean(min_margin))
                else:
                    self.primal_obj.append(- np.min(h_a_bar))
                self.margin.append(-self.primal_obj[-1])
                self.dual_obj.append(-np.max(np.dot(d_bar, H)))
                self.gap.append(self.primal_obj[-1] - self.dual_obj[-1])
                self.err_tr.append(np.mean(h_a_bar < 0))
            # if t % 100 == 0:
            #     print 'iter ' + str(t) + ' ' + str(self.gap[-1])
            if self.gap[-1] < self.epsi:
                break
        self.alpha = a_bar[:p / 2] - a_bar[p / 2:]
        self.d = d_bar
        print " pd-boosting(python): max iter#%d: , actual iter#%d" % (self.max_iter, t)

    def plot_result(self):
        r = 2
        c = 2
        nBins = 6
        plt.figure()
        plt.subplot(r, c, 1)
        plt.plot(np.log(self.gap), 'r-', label='gap')
        T = len(self.gap)
        bound = self.c / np.arange(1, 1 + T)
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
        plt.plot(self.dual_obj, color='g', label='dual')
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
        plt.savefig('result_paraboost.png', format='png')


if __name__ == "__main__":
    pass