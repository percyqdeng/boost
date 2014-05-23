__author__ = 'qdengpercy'

from fwboost import *
from paraboost import *


class TestCase(object):
    """
    TestCase: experimental comparison among different boosting algorithms.
    """

    def __init__(self, dtname='bananamat.mat'):
        if os.name == "nt":
            dtpath = "..\\dataset\\"
        elif os.name == "posix":
            dtpath = '/Users/qdengpercy/workspace/dataset/benchmark_uci/'
        data = scipy.io.loadmat(dtpath + dtname)
        self.x = data['x']
        # self.x = preprocess_data(self.x)
        self.y = data['t']
        self.y = np.squeeze(self.y)
        self.train_ind = data['train'] - 1
        self.test_ind = data['test'] - 1
        print "use dataset: "+str(dtname)
        print "n = "+str(self.x.shape[0])+" p = "+str(self.x.shape[1])

    def rand_test_boost(self, choice=1):
        rep = self.train_ind.shape[0]
        i = np.random.randint(low=0, high=rep)
        xtr = self.x[self.train_ind[i, :], :]
        ytr = self.y[self.train_ind[i, :]]
        xte = self.x[self.test_ind[i, :], :]
        yte = self.y[self.test_ind[i, :]]
        print 'i = '+str(i)
        if choice == 1:
            self.booster = ParaBoost(has_dcap=True, ratio=0.2)
        elif choice == 2:
            self.booster = FwBoost(epsi=0.01, has_dcap=True, ratio=0.2, steprule=1)
        else:
            print "TBD"
        self.booster.train(xtr, ytr)
        self.booster.plot_result()
        self.booster.test(xte, yte)


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


    def benchmark(self, dtname = 'bananamat'):
        if os.name == "nt":
            path = "..\\dataset\\"
        elif os.name == "posix":
            path = '/Users/qdengpercy/workspace/dataset/benchmark_uci/'
        data = scipy.io.loadmat(path + dtname)
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
            booster = FwBoost(epsi=0.01, has_dcap=False, ratio=0.11, steprule=2)
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
    booster = AdaFwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=1)
    booster.train(xtr, ytr)
    # plt.plot(booster.gaps,'rx-')
    booster.plot_result()
    return booster


def cmp_margin():
    ntr = 500
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn(ntr, 1000, ftr_type='disc', has_noise=False)
    # para = BoostPara(epsi=0.001, has_dcap=False, ratio=0.1, max_iter=500000, steprule=1)
    # xtr = np.array([[1,-1], [ 1,-1]], dtype=np.float)
    # ytr = np.array([1, 1])
    # booster1 = FwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=2)
    # booster1.train(xtr, ytr, codetype='py', approx_margin=True)
    # booster1.plot_result()
    booster2 = FwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=1)
    booster2.train(xtr, ytr, codetype='py', approx_margin=True)
    # booster2.plot_result()
    booster3 = ParaBoost(epsi=0.005, has_dcap=False, ratio=0.1)
    booster3.train(xtr, ytr)

    row = 1
    col = 2
    plt.figure()
    # plt.subplot(row, col, 1)
    plt.plot(booster2._margin,'r-', label='fw')
    plt.plot(booster3._margin, 'b-', label='pd')

    plt.figure()
    plt.plot(booster2.err_tr,'r-', label='fw')
    plt.plot(booster3.err_tr,'b-', label='pd')

    plt.legend(loc='best')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))





if __name__ == "__main__":

    filename = ["bananamat", "breast_cancermat", "cvt_bench", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
    # newtest = TestCase(filename[-2])
    # newtest.rand_test_boost(1)
    # newtest.rand_test_boost(2)
    # TestCase.cmp_hard_margin(1)
    # TestCase.test_hard_margin()
    cmp_margin()
    # xtr = np.array([[1.0,-1, 1], [ - 1.0,-1,1],[1,-1,-1]], dtype=np.float)
    # ytr = np.array([1, 1,-1])
    # booster = FwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=2)
    # booster.train(xtr, ytr, codetype='cyf')
    # booster.train(xtr, ytr, codetype='cy')