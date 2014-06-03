__author__ = 'qdengpercy'

import numpy as np
import os
from fwboost import *
from paraboost import *
from sklearn import cross_validation
# from sklearn import KFold
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from test_boost import *


if os.name == "nt":
     dtpath = "..\\..\\dataset\\ucibenchmark\\"
elif os.name == "posix":
    dtpath = '../../dataset/benchmark_uci/'

if __name__ == '__main__':
    filenames = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
    n_dataset = len(filenames)
    err = np.zeros((n_dataset, 4))
    std = np.zeros((n_dataset, 4))
    for i, name in enumerate(filenames):
        experiment = TestCase(dtpath, name)
        res = experiment.bench_mark()
        err[i, :] = res[0]
        std[i, :] = res[1]

    np.save('../output/bench_err', err)
    np.save('../output/bench_std', std)