__author__ = 'qdengpercy'

from fwboost import *
from paraboost import *


class TestCase(object):
    """
    TestCase
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

    def rand_test_paraboost(self):
        rep = self.train_ind.shape[0]
        i = np.random.randint(low=0, high=rep)
        xtr = self.x[self.train_ind[i, :], :]
        ytr = self.y[self.train_ind[i, :]]
        xte = self.x[self.test_ind[i, :], :]
        yte = self.y[self.test_ind[i, :]]
        self.booster = ParaBoost(has_dcap=True)
        self.booster.train(xtr, ytr)
        self.booster.plot_result()
        self.booster.test(xte, yte)


def test_adafwboost():
    ntr = 500
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000)
    booster = AdaFwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=1)
    booster.train(xtr, ytr)
    # plt.plot(booster.gaps,'rx-')
    booster.plot_result()
    return booster


def test_fwboost():
    ntr = 500
    (xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000, hasNoise=False)
    # para = BoostPara(epsi=0.001, has_dcap=False, ratio=0.1, max_iter=500000, steprule=1)
    booster = FwBoost(epsi=0.01, has_dcap=False, ratio=0.1, steprule=2)
    booster.train(xtr, ytr)
    # plt.plot(booster.gaps,'rx-')
    booster.plot_result()
    return booster


def test_paraboost():
    """
    simple test
    """
    testcase = TestCase()

def benchmark(dtname = 'bananamat'):
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


if __name__ == "__main__":
    filename = ["bananamat", "breast_cancermat", "cvt_bench", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormat", "splicemat"]
    newtest = TestCase(filename[-1])
    newtest.rand_test_paraboost()