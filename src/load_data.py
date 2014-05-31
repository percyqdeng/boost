__author__ = 'qdengpercy'

import os
import scipy.io

if os.name == "nt":
    ucipath = "..\\..\\dataset\\ucibenchmark\\"
    uspspath = "..\\..\\dataset\\usps\\"
    mnistpath = "..\\..\\dataset\\mnist\\"
elif os.name == "posix":
    ucipath = '../../dataset/benchmark_uci/'
    uspspath = '../../dataset/usps/'
    mnistpath = '../../dataset/mnist/'
ucifile = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
uspsfile = 'usps_all.mat'
mnistfile = 'mnist_all.mat'


def load_usps():
    data = scipy.io.loadmat(uspspath+'usps_all.mat')
    x = {}
    for i in range(10):
        x[str(i)] = data['data'][:,:,i].T / 255.0
    return x


def load_mnist():
    data = scipy.io.loadmat(mnistpath+'mnist_all.mat')
    x_train ={}
    x_test = {}
    for i in range(10):
        x_train[str(i)] = data['train'+str(i)]/255.0
        x_test[str(i)] = data['test'+str(i)]/255.0
    return x_train, x_test


def load_uci():


if __name__ == '__main__':
    pass