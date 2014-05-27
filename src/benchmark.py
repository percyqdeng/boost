__author__ = 'qdengpercy'

import numpy as np
from fwboost import *
from paraboost import *
from sklearn import cross_validation
# from sklearn import KFold
from sklearn import datasets


def cmp_convergence():
    n_folds = 1
    err = np.zeros(n_folds)
    kf = cross_validation.KFold(n, n_folds, indices=True)
    for trainind, testind in kf:
        

dataset_path = '../../dataset/benchmark_uci'

if __name__ == '__main__':
    iris = datasets.load_iris()
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #     iris.data, iris.target, test_size=0.4, random_state=0)

    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=8)

    kf = cross_validation.KFold(10, n_folds=4, indices=True)
    for train, test in kf:
        print "%s %s " %(train, test)