__author__ = 'qdengpercy'


__author__ = 'qdengpercy'

from fwboost import *
from paraboost import *
from boost import *
from test_boost import TestCase
import sklearn.cross_validation as cv
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# def debug_hard_margin():

# if __name__ == '__main__':



n = 1000
n1 = 10
n2 = 100
x = np.random.random_integers(0, 1, size=(n, n1))
x = 2*x.astype(np.float) - 1
z = np.sum(x[:, :5], axis=1)
y = np.sign(z)
m = np.min(np.abs(z))
# repeat feature with infused noise
x2 = np.repeat(x, n2, axis=1)
x2 += np.random.normal(0, 0.1, size=(n, n1*n2))
x = np.hstack((x, x2))
x[x>1] = 1
x[x<-1] = -1


# n1 = 100
# n2 = 100
# x, y = TestCase.gen_syn_data(n1, n2)
x_train, x_test, y_train, y_test = cv.train_test_split(x, y, train_size=0.6, test_size=0.4, random_state=12)

epsi = 0.01
has_dcap = False
# max_iter = 10000
max_iter = None
r = 0.3
# fw = FwBoost(epsi, has_dcap=has_dcap, ratio=r, steprule=1)
# fw.train(x_train, y_train, codetype="cy", approx_margin=False, max_iter=max_iter, ftr='wl')
fw1 = FwBoost(epsi, has_dcap=has_dcap, ratio=r, steprule=3)
fw1.train(x_train, y_train, codetype="py", approx_margin=False, max_iter=max_iter, ftr='wl')

fw2 = FwBoost(epsi, has_dcap=has_dcap, ratio=r, steprule=1)
fw2.train(x_train, y_train, codetype="py", approx_margin=False, max_iter=max_iter, ftr='wl')

row = 1
col = 2
plt.subplot(row, col, 1)
plt.semilogy(fw1.iter_num, fw1.gap,'r-')
plt.semilogy(fw2.iter_num, fw2.gap, 'b-')

