"""
=============================
Discrete versus Real AdaBoost
=============================

This example is based on Figure 10.2 from Hastie et al 2009 [1] and illustrates
the difference in performance between the discrete SAMME [2] boosting
algorithm and real SAMME.R boosting algorithm. Both algorithms are evaluated
on a binary classification task where the target Y is a non-linear function
of 10 input features.

Discrete SAMME AdaBoost adapts based on errors in predicted class labels
whereas real SAMME.R uses the predicted class probabilities.

.. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
    Learning Ed. 2", Springer, 2009.

.. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
"""
print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>,
#         Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import pylab as pl
import os
from test_boost import *


# A learning rate of 1. may not be optimal for both SAMME and SAMME.R
learning_rate = 1.

# X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

# X_test, y_test = X[2000:], y[2000:]
# X_train, y_train = X[:2000], y[:2000]
if os.name == "nt":
    dtpath = '../../dataset/benchmark_uci'
elif os.name == "posix":
    dtpath = '../../dataset/benchmark_uci/'
filename = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
newtest = TestCase(dtpath, filename[-2])
X_train, y_train, X_test, y_test = newtest.gen_i_th()
n_estimators = 1000
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)

print "fit ada_discrete"
ada_discrete = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")
ada_discrete.fit(X_train, y_train)

h = TestCase.weak_learner_pred(ada_discrete, X_train)
ada_wk_tr_err = np.zeros(h.shape[1])
for i in range(h.shape[1]):
    ada_wk_tr_err[i] = zero_one_loss(y_train, h[:,i])

hte = TestCase.weak_learner_pred(ada_discrete, X_test)
ada_wk_te_err = np.zeros(hte.shape[1])
for i in range(hte.shape[1]):
    ada_wk_te_err[i] = zero_one_loss(y_test, hte[:,i])

fig = pl.figure()
pl.plot(range(n_estimators), ada_wk_tr_err, 'r-')
pl.plot(range(n_estimators), ada_wk_te_err, 'b-')
plt.show()
print "fit ada_real"
ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
ada_real.fit(X_train, y_train)

fig = pl.figure()
ax = fig.add_subplot(111)

ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
        label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
        label='Decision Tree Error')


ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)

ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
        label='Discrete AdaBoost Test Error',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
        label='Discrete AdaBoost Train Error',
        color='blue')
ax.plot(np.arange(n_estimators) + 1, ada_real_err,
        label='Real AdaBoost Test Error',
        color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
        label='Real AdaBoost Train Error',
        color='green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)

pl.show()

h = np.zeros((X_train.shape[0], n_estimators))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    h[:, i] = y_pred

pd = ParaBoost(epsi=0.01, has_dcap=True, ratio=0.1)
pd.train_h(h, y_train)
pred = pd.test_h(h)
err_tr = zero_one_loss(y_train, pred)

hte = np.zeros((X_test.shape[0], n_estimators))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    hte[:, i] = y_pred

pred = pd.test_h(hte)
err_te = zero_one_loss(y_test, pred)

print "err_tr%f, err_te%f" % (err_tr, err_te)