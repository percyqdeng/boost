__author__ = 'qdengpercy'

import profile

import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from fwboost import *
from paraboost import *
from extract_features import *
class TestCase(object):
    """
    TestCase: experimental comparison among different boosting algorithms.
    """
    def __init__(self, pathname='../../dataset/benchmark_uci/', dtname='bananamat.mat'):
        dtpath = pathname
        data = scipy.io.loadmat(dtpath + dtname)
        self.x = data['x']
        # self.x = preprocess_data(self.x)
        self.y = data['t']
        self.y = (np.squeeze(self.y)).astype(np.intc)
        self.train_ind = data['train'] - 1
        self.test_ind = data['test'] - 1
        print "use dataset: "+str(dtname)
        print "n = "+str(self.x.shape[0])+" p = "+str(self.x.shape[1])

    def rand_test_boost(self):
        xtr, ytr, xte, yte = self._gen_i_th(i=-1)
        xtr, xte = TestCase._normalize_features(xtr, xte)
        self.booster1 = ParaBoost(epsi=0.01, has_dcap=True, ratio=0.3)
        self.booster1.train(xtr, ytr)
        self.booster2 = FwBoost(epsi=0.01, has_dcap=True, ratio=0.3, steprule=1)
        self.booster2.train(xtr, ytr, codetype='cy', approx_margin=True)

        plt.figure()
        plt.plot(self.booster1.err_tr, 'r-', label='pboost')
        plt.plot(self.booster2.err_tr, 'b-', label='fwboost')
        plt.ylabel('train error')
        plt.legend()
        plt.figure()
        plt.plot(self.booster1.margin, 'r-', label='pboost')
        plt.plot(self.booster2.margin, 'b-', label='fwboost')
        plt.legend()

    @staticmethod
    def _normalize_features(xtr, xte):
        std = np.std(xtr, 0)
        avg = np.mean(xtr, 0)
        xtr = (xtr - avg[np.newaxis, :]) / std[np.newaxis, :]
        xte = (xte - avg[np.newaxis, :]) / std[np.newaxis, :]
        return xtr, xte

    def gen_i_th(self, i=-1):
        rep = self.train_ind.shape[0]
        if i < 0 or i >= rep:
            i = np.random.randint(low=0, high=rep)
        xtr = self.x[self.train_ind[i, :], :]
        ytr = self.y[self.train_ind[i, :]]
        xte = self.x[self.test_ind[i, :], :]
        yte = self.y[self.test_ind[i, :]]
        return xtr, ytr, xte, yte

    @staticmethod
    def gen_syn_data(n1=1000, n2=100):
        n1 = 1000
        n2 = 100
        x = np.random.uniform(0, 1, size=(n1, n2))
        # add 0.5 to the first 10 columns of first 1000 rows, and rescale the first 1000 rows
        x[:n1, :10] += 0.5
        x[:n1, :] /= x[:n1, :10].max()
        x = np.hstack((x, -x))
        x2 = np.random.uniform(0, 1, size=(n1, n2))
        x2 = np.hstack((x2, -x2))
        x = np.vstack((x, x2))
        y = np.vstack((np.ones((n1, 1)), -np.ones((n1, 1))))
        y = np.squeeze(y)
        ind = np.random.choice(2*n1, int(0.2*n1), replace=False)
        y[ind] = - y[ind]
        return x, y


    @staticmethod
    def synthetic_soft_margin():
        """
        synthetic test on soft margin, follow the experiment in Warmuth's NIPS paper
        """
        n1 = 1000
        n2 = 100
        x, y = TestCase.gen_syn_data(n1, n2)
        epsi = 0.001
        has_dcap = True
        n_iter = 2
        ss = cv.ShuffleSplit(2*n1, n_iter=n_iter, train_size=0.25, test_size=0.75)
        rate = np.linspace(0.1, 0.8, 8, endpoint=True)
        te_err_fw = np.zeros((n_iter, len(rate)))
        tr_err_fw = np.zeros((n_iter, len(rate)))
        te_err_pd = np.zeros((n_iter, len(rate)))
        tr_err_pd = np.zeros((n_iter, len(rate)))
        k = 0
        for tr_ind, te_ind in ss:
            xtr = x[tr_ind, :]
            ytr = y[tr_ind]
            xte = x[te_ind, :]
            yte = y[te_ind]
            for i, r in enumerate(rate):
                fw = FwBoost(epsi, has_dcap, ratio=r)
                fw.train(xtr, ytr, codetype="cy", approx_margin=True, early_stop=False, ftr='wl')
                pd = ParaBoost(epsi, has_dcap, ratio=r)
                pd.train(xtr, ytr, early_stop=False, ftr='wl')
                te_err_fw[k, i] = fw.test(xte, yte)
                tr_err_fw[k, i] = fw.err_tr[-1]
                te_err_pd[k, i] = pd.test(xte, yte)
                tr_err_pd[k, i] = pd.err_tr[-1]
            k += 1

        plt.figure()
        # plt.subplot(121)
        plt.plot(rate, np.mean(te_err_fw, axis=0), 'bx-', label='fw test err')
        plt.plot(rate, np.mean(tr_err_fw, axis=0), 'ro-', label='fw train err')
        plt.plot(rate, np.mean(te_err_pd, axis=0), 'yx-', label='pd test err')
        plt.plot(rate, np.mean(tr_err_pd, axis=0), 'go-', label='pd train err')
        plt.xlabel('r')
        plt.legend(loc='best')
        plt.savefig('../output/syn_cmp_rate.pdf')

        plt.figure()
        epsi = 0.01
        r = 0.4
        pd = ParaBoost(epsi, has_dcap, ratio=r)
        pd.train(xtr, ytr, early_stop=False, ftr='wl')
        fw = FwBoost(epsi, has_dcap=has_dcap, ratio=r)
        fw.train(xtr, ytr, codetype="cy", approx_margin=True, early_stop=False, ftr='wl')
        plt.subplot(121)
        plt.plot(pd.iter_num, pd.margin, 'b-', label='pd')
        plt.plot(fw.iter_num, fw.margin, 'r-', label='fw')
        plt.ylabel('margin')
        plt.xlabel('number of iteration')
        plt.subplot(122)
        plt.plot(pd.iter_num, pd.gap, 'b-', label='pd')
        plt.plot(fw.iter_num, fw.gap, 'r-', label='fw')
        plt.ylabel('primal dual gap')
        plt.xlabel('number of iteration')
        # plt.semilogx(pd.iter_num, pd.margin)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()
        plt.savefig('../output/syn_cmp_margin.pdf')



    def bench_mark(self):
        """
        test on uci benchmark
        """
        n_estimators = np.minimum(1000, int(self.x.shape[0]*0.7))
        # n_estimators = 1
        n_samples = self.x.shape[0]
        n_reps = 20
        ss = cv.ShuffleSplit(n_samples, n_reps, train_size=0.6, test_size=0.4, random_state=1)
        ada_tr_err = np.zeros(n_reps)
        ada_te_err = np.zeros(n_reps)
        pd_tr_err = np.zeros(n_reps)
        pd_te_err = np.zeros(n_reps)
        k = 0
        ratio_list = np.array([0.05, 0.1, 0.15, 0.2, 0.3])
        for tr_ind, te_ind in ss:
            print " iter#: %d" % (k)
            xtr = self.x[tr_ind, :]
            ytr = self.y[tr_ind]
            xte = self.x[te_ind, :]
            yte = self.y[te_ind]
            ada_disc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                          n_estimators=n_estimators, algorithm="SAMME")
            ada_disc.fit(xtr, ytr)
            pred = ada_disc.predict(xtr)
            ada_tr_err[k] = zero_one_loss(ytr, pred)
            pred = ada_disc.predict(xte)
            ada_te_err[k] = zero_one_loss(yte, pred)
            htr = TestCase.weak_learner_pred(ada_disc, xtr)
            hte = TestCase.weak_learner_pred(ada_disc, xte)
            # htr = np.zeros((ytr.shape[0], n_estimators))
            # for i, y_pred in enumerate(ada_disc.staged_predict(xtr)):
            #     htr[:, i] = y_pred
            #     ada_tr_err[k, i] = zero_one_loss(y_true=ytr, y_pred=y_pred)

            best_ratio = TestCase.cross_valid(htr, ytr, ratio_list)
            # best_ratio = 0.1
            print "best ratio %f " % (best_ratio)
            pd = ParaBoost(epsi=0.005, has_dcap=True, ratio=best_ratio)
            pd.train(htr, ytr, ftr='wl')
            pred = pd.test_h(hte)
            pd_tr_err[k] = pd.err_tr[-1]
            pd_te_err[k] = zero_one_loss(y_true=yte, y_pred=pred)
            k += 1

        print "trainerr %f, testerr %f" % (np.mean(pd_tr_err), np.mean(pd_te_err))

        n_terms = 2 * 2
        err = np.zeros(n_terms)
        std = np.zeros(n_terms)
        err[0] = np.mean(ada_tr_err)
        err[1] = np.mean(ada_te_err)
        err[2] = np.mean(pd_tr_err)
        err[3] = np.mean(pd_te_err)
        std[0] = np.std(ada_tr_err)
        std[1] = np.std(ada_te_err)
        std[2] = np.std(pd_tr_err)
        std[3] = np.std(pd_te_err)
        # plt.figure()
        # plt.plot(np.mean(ada_tr_err, axis=0), 'r-', label="ada_train")
        # plt.plot(np.mean(ada_te_err, axis=0), 'b-', label="ada_test")
        # plt.show()
        return err, std,

    @staticmethod
    def cross_valid(h, y, ratio_list):
        """
        cross validation to tune the best cap probability for soft-margin boosting
        """
        print " find optimal ratio"
        n_samples = h.shape[0]
        n_folds = 4
        ntr = n_samples/n_folds
        ratio_list = ratio_list[ratio_list >= 1.0/ntr]
        kf = cv.KFold(n=n_samples, n_folds=n_folds)
        err_tr = np.zeros((n_folds, len(ratio_list)))
        err_te = np.zeros((n_folds, len(ratio_list)))
        k = 0
        for tr_ind, te_ind in kf:
            print "nfold: %d" % (k)
            xtr, ytr, xte, yte = h[tr_ind, :], y[tr_ind], h[te_ind, :], y[te_ind]
            for i, r in enumerate(ratio_list):
                pd = ParaBoost(epsi=0.01, has_dcap=True, ratio=r)
                pd.train_h(xtr, ytr)
                pred = pd.test_h(xte)
                err_te[k, i] = zero_one_loss(y_true=yte, y_pred=pred)
                err_tr[k, i] = pd.err_tr[-1]
            k += 1
        err_te_avg = np.mean(err_te, axis=0)
        err_tr_avg = np.mean(err_tr, axis=0)
        arg = np.argmin(err_te_avg)
        best_ratio = ratio_list[arg]
        err = err_te_avg[arg]
        return best_ratio



def profile_paraboost():
    x, y = TestCase.gen_syn_data(n1=2000, n2=2000)
    pd = ParaBoost(epsi=0.005, has_dcap=True, ratio=0.1)
    profile.runctx("pd.train(x,y)", globals(), locals())




if os.name == "nt":
    ucipath = "..\\..\\dataset\\ucibenchmark\\"
    uspspath = "..\\..\\dataset\\usps\\"
elif os.name == "posix":
    ucipath = '../../dataset/benchmark_uci/'
    uspspath = '../../dataset/usps/'
ucifile = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
uspsfile = 'usps_all.mat'
mnistfile = 'mnist_all.mat'


if __name__ == "__main__":

    TestCase.synthetic_hard_margin()
    # newtest = TestCase(ucipath, ucifile[0])
    # fw = newtest.cmp_sparsity()
    # newtest.bench_mark()
    # newtest.rand_test_boost()
    # bfw, bpd, w = cmp_margin()
    # newtest.synthetic2()
    # TestCase.synthetic_soft_margin()
    # TestCase.debug()
    # profile_paraboost()
