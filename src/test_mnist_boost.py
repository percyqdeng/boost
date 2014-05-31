__author__ = 'qdengpercy'

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from fwboost import *
from paraboost import *
from load_data import *
import sklearn.cross_validation as cv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from extract_features import *

# def experiment_sparse_pattern():
if __name__ == "__main__":
    """
    show sparsity pattern
    """
    digit1 = 3
    digit2 = 5
    data = load_mnist()
    x, y = convert_one_vs_one(data, digit1, digit2)
    print "-------------------load mnist data-------------------"
    # ----------------pass through adaboost-----------------
    n_estimators = 200
    n_samples = x.shape[0]
    n_reps = 1
    x_train, x_test, y_train, y_test = cv.train_test_split(x, y, train_size=0.3, test_size=0.7, random_state=0)

    ada_disc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                  n_estimators=n_estimators, algorithm="SAMME")
    ada_disc.fit(x_train, y_train)
    htr = weak_learner_pred(ada_disc, x_train)
    print "-----------------obtain weak learner feature------------------"
    fw = FwBoost(epsi=0.005, has_dcap=True, ratio=0.1)
    fw.train(htr, y_train, codetype="cy", ftr='wl')
    pd = ParaBoost(0.005, has_dcap=True, ratio=0.1)
    pd.train(htr, y_train, ftr='wl')
    plt.figure()
    plt.subplot(121)
    plt.plot(fw.iter_num, fw.num_zeros, 'rx-')
    plt.ylim(0, n_estimators)
    plt.subplot(122)
    plt.plot(fw.iter_num, fw.err_tr, 'rx-', label='fw')
    plt.plot(pd.iter_num, pd.err_tr, 'bx-', label='pd')
    plt.legend(loc='best')
    plt.ticklabel_format(style='sci')
    plt.tight_layout()

    plt.figure()
    plt.subplot(121)
    a = np.fabs(fw.alpha)
    a /= a.max()
    a = np.sort(a, kind='quicksort')[::-1]
    b = np.fabs(pd.alpha)
    b /= b.max()
    b = np.sort(b, kind='quicksort')[::-1]
    plt.plot(a, 'rx-', label='fw')
    plt.plot(b, 'bo-', label='pd')
    # plt.show()
