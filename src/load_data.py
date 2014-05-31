__author__ = 'qdengpercy'

import os
import scipy.io
import numpy as np

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


def convert_one_vs_one(data, pos_ind, neg_ind):
    """
    convert 0-9 digits to binary dataset, with digit pos_ind vs neg_ind
    """
    assert 0 <= pos_ind <= 9
    assert 0 <= neg_ind <= 9
    x_pos = data[str(pos_ind)]
    x_neg = data[str(neg_ind)]
    x = np.vstack((x_pos, x_neg))
    y = np.ones(x.shape[0], dtype=np.int32)
    y[x_pos.shape[0]:-1] = -1
    return x, y

def convert_one_vs_all(data, pos_ind):
    """
    convert 0-9 digits to binary dataset, with digit pos_ind vs rest
    """
    assert 0 <= pos_ind <= 9
    x_pos = data[str(pos_ind)]
    x_neg = None
    for i in range(10):
        if i != pos_ind:
            if x_neg is None:
                x_neg = data[str(i)]
            else:
                x_neg = np.vstack((x_neg, data[str(i)]))
    x = np.vstack((x_pos, x_neg))
    y = np.ones(x.shape[0], dtype=np.int32)
    y[x_pos.shape[0]:-1] = -1
    return x, y


def load_usps():
    data = scipy.io.loadmat(uspspath+'usps_all.mat')
    x = {}
    for i in range(10):
        x[str(i)] = data['data'][:,:,i].T / 255.0
    return x


def load_mnist_split():
    """
    load mnist with split train test
    """
    data = scipy.io.loadmat(mnistpath+'mnist_all.mat')
    x_train ={}
    x_test = {}
    for i in range(10):
        x_train[str(i)] = data['train'+str(i)]/255.0
        x_test[str(i)] = data['test'+str(i)]/255.0
    return x_train, x_test


def load_mnist():
    data = scipy.io.loadmat(mnistpath+'mnist_all.mat')
    x = {}
    for i in range(10):
        x[str(i)] = np.vstack((data['train'+str(i)]/255.0, data['test'+str(i)]/255.0))
    return x

def load_uci():
    pass

if __name__ == '__main__':
    pass