__author__ = 'qdengpercy'

from fwboost import *
from paraboost import *


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
        self.booster1 = ParaBoost(epsi=0.01, has_dcap=False, ratio=0.3)
        self.booster1.train(xtr, ytr)
        self.booster2 = FwBoost(epsi=0.01, has_dcap=False, ratio=0.3, steprule=1)
        self.booster2.train(xtr, ytr, codetype='py', approx_margin=True)

        plt.figure()
        plt.plot(self.booster1.err_tr, 'r-')
        plt.plot(self.booster2.err_tr, 'b-')

        plt.figure()
        plt.plot(self.booster1._margin, 'r-')
        plt.plot(self.booster2._margin, 'b-')

    @staticmethod
    # def _normalize_features(xtr, xte):
    #     std = np.std(xtr, 0)
    #     avg = np.mean(xtr, 0)
    #     xtr = (xtr - avg[np.newaxis, :]) / std[np.newaxis, :]
    #     xte = (xte - avg[np.newaxis, :]) / std[np.newaxis, :]
    #     return xtr, xte

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
    def test_hard_margin():
        # ntr = 1000
        """
        toy example with hard margin
        """
        (xtr, ytr, yH, margin, w, xte, yte) = gen_syn(ftr_type='disc', ntr=1000, nte=100)
        booster1 = ParaBoost(has_dcap=False, ratio=0.2)
        booster2 = FwBoost(has_dcap=False, ratio=0.2)

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
    # para = BoostPara(epsi=0.001, has_dcap=False, ratio=0.1, max_iter=500000, steprule=1)
    # xtr = np.array([[1,-1], [ 1,-1]], dtype=np.float)
    # ytr = np.array([1, 1])
    # booster1 = FwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=2)
    # booster1.train(xtr, ytr, codetype='py', approx_margin=True)
    # booster1.plot_result()
    booster2 = FwBoost(epsi=0.001, has_dcap=False, ratio=0.01, steprule=1)
    booster2.train(xtr, ytr, codetype='py', approx_margin=True)
    # booster2.plot_result()
    booster3 = ParaBoost(epsi=0.001, has_dcap=False, ratio=0.01)
    booster3.train(xtr, ytr)

    row = 1
    col = 2
    plt.figure()
    # plt.subplot(row, col, 1)
    plt.plot(booster2._margin,'r-', label='fw')
    plt.plot(booster3._margin, 'b-', label='pd')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('number of iteration')
    plt.ylabel('margin')
    plt.legend(loc='best')
    plt.savefig('../output/cmp_marginreal.pdf')

    plt.figure()
    plt.plot(booster2.err_tr, 'r-', label='fw')
    plt.plot(booster3.err_tr, 'b-', label='pd')
    plt.figure()
    plt.plot(booster2._primal_obj, 'r-', label='fw')
    plt.plot(booster3._primal_obj, 'b-', label='pd')
    plt.ylabel('primal objective')
    plt.legend(loc='best')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    return booster2, booster3, w



if __name__ == "__main__":

    filename = ["bananamat", "breast_cancermat", "cvt_bench", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
    # newtest = TestCase(filename[1])
    # newtest.rand_test_boost()
    # newtest.rand_test_boost(1)
    # newtest.rand_test_boost(2)
    # TestCase.cmp_hard_margin(1)
    # TestCase.test_hard_margin()
    bfw, bpd, w = cmp_margin()
    # xtr = np.array([[1.0,-1, 1], [ - 1.0,-1,1],[1,-1,-1]], dtype=np.float)
    # ytr = np.array([1, 1,-1])
    # booster = FwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=2)
    # booster.train(xtr, ytr, codetype='cyf')
    # booster.train(xtr, ytr, codetype='cy')