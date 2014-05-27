__author__ = 'qdengpercy'

from fwboost import *
from paraboost import *
import sklearn.cross_validation as cv

class TestCase(object):
    """
    TestCase: experimental comparison among different boosting algorithms.
    """

    def __init__(self, dtname='bananamat.mat'):
        if os.name == "nt":
            dtpath = "..\\..\\dataset\\ucibenchmark\\"
        elif os.name == "posix":
            dtpath = '../../dataset/benchmark_uci/'
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
        # rep = self.train_ind.shape[0]
        # i = np.random.randint(low=0, high=rep)
        # xtr = self.x[self.train_ind[i, :], :]
        # ytr = self.y[self.train_ind[i, :]]
        # xte = self.x[self.test_ind[i, :], :]
        # yte = self.y[self.test_ind[i, :]]
        # print 'i = '+str(i)
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
        plt.plot(self.booster1._margin, 'r-', label='pboost')
        plt.plot(self.booster2._margin, 'b-', label='fwboost')
        plt.legend()

    @staticmethod
    def _normalize_features(xtr, xte):
        std = np.std(xtr, 0)
        avg = np.mean(xtr, 0)
        xtr = (xtr - avg[np.newaxis, :]) / std[np.newaxis, :]
        xte = (xte - avg[np.newaxis, :]) / std[np.newaxis, :]
        return xtr, xte

    def _gen_i_th(self, i=-1):
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
    def debug():
        n1 = 1000
        n2 = 100
        x, y = TestCase.gen_syn_data(n1, n2)
        ss = cv.ShuffleSplit(2*n1, n_iter=1, train_size=0.25, test_size=0.75)
        epsi = 0.01
        has_dcap = True
        r = 0.5
        print
        for tr_ind, te_ind in ss:
            xtr = x[tr_ind, :]
            ytr = y[tr_ind]
            xte = x[te_ind, :]
            yte = y[te_ind]
            fw = FwBoost(epsi, has_dcap, ratio=r)
            fw.train(xtr, ytr, codetype="py", approx_margin=True, early_stop=False)
            fw = FwBoost(epsi, has_dcap, ratio=r)
            fw.train(xtr, ytr, codetype="cy", approx_margin=True, early_stop=False)

    @staticmethod
    def synthetic_soft_margin():
        """
        synthetic test on soft margin
        """
        n1 = 1000
        n2 = 100
        x, y = TestCase.gen_syn_data(n1, n2)
        epsi = 0.01
        has_dcap = True
        ss = cv.ShuffleSplit(2*n1, n_iter=1, train_size=0.25, test_size=0.75)
        rate = np.linspace(0.1, 0.8, 8, endpoint=True)
        te_err = np.zeros(len(rate))
        tr_err = np.zeros(len(rate))
        for tr_ind, te_ind in ss:
            xtr = x[tr_ind, :]
            ytr = y[tr_ind]
            xte = x[te_ind, :]
            yte = y[te_ind]
            for i, r in enumerate(rate):
                pd = FwBoost(epsi, has_dcap, ratio=r)
                pd.train(xtr, ytr, codetype="cy", approx_margin=True, early_stop=False)
                # pd = ParaBoost(epsi, has_dcap, ratio=r)
                # pd.train(xtr, ytr, early_stop=False)
                te_err[i] = pd.test(xte, yte)
                tr_err[i] = pd.err_tr[-1]

        plt.figure()
        plt.subplot(121)
        plt.plot(rate, te_err, 'bx-', label='test err')
        plt.plot(rate, tr_err, 'ro-', label='train err')
        plt.legend(loc='best')
        # plt.savefig('../output/syn_soft_margin1.pdf')

        epsi = 0.01
        r = 0.4
        pd = ParaBoost(epsi, has_dcap, ratio=r)
        pd.train(xtr, ytr, early_stop=False)
        # plt.figure()
        plt.subplot(122)
        plt.plot(pd.iter_num, pd._margin)
        # plt.semilogx(pd.iter_num, pd._margin)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()
        plt.savefig('../output/syn_soft_margin.pdf')


    def synthetic_hard_margin(self):
        """
        compare margin
        """
        n = 1000
        n1 = 100
        n2 = 1000
        x = np.random.random_integers(0, 1, size=(n, n1))
        x = 2*x.astype(np.float) - 1
        z= np.sum(x[:, :5], axis=1)
        y = np.sign(z)
        m = np.min(np.abs(z))
        # repeat feature with infused noise
        x2 = np.repeat(x, n2, axis=1)
        x2 += np.random.normal(0, 0.1, size=(n, n1*n2))
        x = np.hstack(((x, x2)))
        x[x>1] = 1
        x[x<-1] = -1
        print "generate data n=%s, n1=%s, n2=%s, margin=%s" % (n, n1, n2, m)
        cap_prob = False
        stop = False
        fw = FwBoost(0.01, has_dcap=cap_prob, ratio=0.1)
        fw.train(x, y, codetype='cy', approx_margin=True, early_stop=stop)
        pd = ParaBoost(epsi=0.01, has_dcap=cap_prob, ratio=0.1)
        pd.train(xtr=x, ytr=y, early_stop=stop)

        # print "margin fw:%s pd%s" % (fw._margin[-1], pd._margin[-1])
        print "margin pd%s" % ( pd._margin[-1])
        plt.figure()
        plt.ylim(0.05, 0.2)
        # plt.plot(fw.iter_num, fw._margin,'r-',label='fw')
        plt.plot(pd.iter_num, pd._margin,'b-',label='pd')
        plt.ylabel('margin')
        plt.legend(loc='best')
        plt.savefig("../output/synth_hard_margin.pdf")

        plt.figure()
        plt.ylim(0.05, 0.2)
        # plt.semilogx(fw.iter_num, fw._margin,'r-',label='fw')
        plt.semilogx(pd.iter_num, pd._margin,'b-',label='pd')
        plt.ylabel('margin')
        plt.legend(loc='best')
        plt.savefig("../output/synth_hard_margin_log.pdf")

    @staticmethod
    def test_hard_margin():
        # ntr = 1000
        """
        toy example with hard margin
        """
        (xtr, ytr, yH, margin, w, xte, yte) = gen_syn(ftr_type='disc', ntr=1000, nte=100)
        booster1 = ParaBoost(has_dcap=False, ratio=0.3)
        booster2 = FwBoost(has_dcap=False, ratio=0.3)

        booster1.train(xtr, ytr)
        booster2.train(xtr, ytr)
        row = 1
        col = 2
        plt.subplot(row, col, 1)
        plt.plot(booster1._gap, 'r-', label=booster1.to_name())
        plt.plot(booster2._gap, 'b-', label=booster2.to_name())
        plt.ylabel('primal-dual gap')
        plt.subplot(row, col, 2)
        plt.plot(booster1.err_tr, 'r-', label=booster1.to_name())
        plt.plot(booster2.err_tr, 'b-', label=booster2.to_name())
        plt.ylabel('train err')
        # plot gap
        # booster1.plot_result()
        plt.ticklabel_format(style='sci')
        plt.legend(bbox_to_anchor=(-1.2, 1.01, 2.2, .1), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        print booster1.err_tr[0]
        print booster2.err_tr[0]


def test_adafwboost():
    ntr = 500
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000)
    booster = AdaFwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=1)
    booster.train(xtr, ytr)
    # plt.plot(booster.gaps,'rx-')
    booster.plot_result()
    return booster


def cmp_margin():
    ntr = 700
    (xtr, ytr, xte, yte, w) = gen_syn(ntr, 10, ftr_type='disc', has_noise=True)
    booster1 = FwBoost(epsi=0.001, has_dcap=False, ratio=0.1, steprule=1)
    booster1.train(xtr, ytr, codetype='py', approx_margin=True)
    # booster2.plot_result()
    booster2 = ParaBoost(epsi=0.001, has_dcap=False, ratio=0.1)
    booster2.train(xtr, ytr)

    row = 1
    col = 2
    plt.figure()
    plt.plot(booster1._margin,'r-', label='fw')
    plt.plot(booster2._margin, 'b-', label='pd')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('number of iteration')
    plt.ylabel('margin')
    plt.legend(loc='best')
    plt.savefig('../output/cmp_marginreal.pdf')
    plt.figure()
    plt.plot(booster1.err_tr, 'r-', label='fw')
    plt.plot(booster2.err_tr, 'b-', label='pd')
    plt.figure()
    plt.plot(booster1._primal_obj, 'r-', label='fw')
    plt.plot(booster2._primal_obj, 'b-', label='pd')
    plt.ylabel('primal objective')
    plt.legend(loc='best')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    return booster1, booster2, w


if __name__ == "__main__":
    filename = ["bananamat", "breast_cancermat", "cvt_bench", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
    # newtest = TestCase(filename[1])
    # newtest.rand_test_boost()
    # bfw, bpd, w = cmp_margin()
    # newtest.synthetic2()
    # TestCase.synthetic_soft_margin()
    TestCase.debug()