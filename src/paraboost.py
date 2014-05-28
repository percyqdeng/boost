__author__ = 'qdengpercy'
# coding=utf8
from boost import *
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
    _primal_obj : array type, primal objective in each iteration
    _margin : array type, margin in each iteration
    _dual_obj : array type, dual objective in each iteration
    _gap : array type, primal-dual gap in each iteration
    """

    def __init__(self, epsi=0.01, has_dcap=False, ratio=0.1):
        self.epsi = epsi
        self.has_dcap = has_dcap
        self.ratio = ratio
        self._primal_obj = []
        self._dual_obj = []
        self._margin = []
        self._gap = []
        self.err_tr = []
        self.alpha = []
        self.iter_num = []

    def to_name(self):
        return "paraboost"

    def train(self, xtr, ytr, early_stop=False):

        # xtr = self._process_train_data(xtr)
        # xtr = np.hstack((xtr, np.ones((ntr, 1))))
        yH = ytr[:, np.newaxis] * xtr
        # yH = np.hstack((yH, -yH))
        self._para_boosting(yH, early_stop)

    def train_h(self, h, ytr, early_stop=False):
        yh = ytr[:, np.newaxis] * h
        self._para_boosting(yh, early_stop)

    def test_h(self, h):
        pred = np.sign(np.dot(h, self.alpha))
        return pred

    def test(self, xte, yte):

        nte = xte.shape[0]
        # xte = self._process_test_data(xte)
        # xte = np.hstack((xte, np.ones((nte, 1))))
        pred = np.sign(np.dot(xte, self.alpha))
        return np.mean(pred != yte)

    def plot_result(self):
        r = 2
        c = 2
        nBins = 6
        plt.figure()
        plt.subplot(r, c, 1)
        plt.plot(np.log(self._gap), 'r-', label='gap')
        T = len(self._gap)
        bound = self.c / np.arange(1, 1 + T)
        plt.plot(np.log(bound), 'b-', label='bound')
        plt.title('log primal-dual gap')
        plt.legend(loc='best')
        plt.locator_params(axis='x', nbins=nBins)
        plt.subplot(r, c, 2)
        plt.plot(self._margin, 'b-')
        plt.title('margin')
        plt.locator_params(axis='x', nbins=nBins)
        plt.subplot(r, c, 3)
        plt.plot(self._primal_obj, 'r-', label='primal')
        plt.plot(self._dual_obj, color='g', label='dual')
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

    def _para_boosting(self, H, earlystop=False):
        """
        primal-dual boost with capped probability ||d||_infty <= 1/k
        a, d : primal dual variables to be updated,
        a_tilde, d_tilde : intermediate variables,
        a_bar, d_bar : average value as output.
        """
        # print '----------------primal-dual boost-------------------'
        H = np.hstack((H, -H))
        (n, p) = H.shape
        self.c = np.log(n*p)
        nu = int(n * self.ratio)
        if earlystop:
            max_iter = 200
        else:
            max_iter = int(np.log(n * p) / self.epsi)
        if max_iter < 1000:
            delta = 4
        else:
            delta = 40
        showtimes = int(5)
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

        for t in range(max_iter):
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
            h_a = np.dot(H, a_bar)
            if t % delta == 0:
                self.iter_num.append(t)
                if self.has_dcap:
                    min_margin = ksmallest(h_a, nu)
                    self._primal_obj.append(-np.mean(min_margin))
                else:
                    self._primal_obj.append(- np.min(h_a))
                self._margin.append(-self._primal_obj[-1])
                self._dual_obj.append(-np.max(np.dot(d_bar, H)))
                self._gap.append(self._primal_obj[-1] - self._dual_obj[-1])
                self.err_tr.append(np.mean(h_a < 0))
            # if t % (max_iter / showtimes) == 0:
            #     print 'iter ' + str(t) + ' ' + str(self._gap[-1])
            if self._gap[-1] < self.epsi:
                break
        self.alpha = a_bar[:p / 2] - a_bar[p / 2:]
        self.d = d_bar
        print " pd-boosting(python): max iter#%d: , actual iter#%d" % (max_iter, t)

if __name__ == "__main__":
    pass