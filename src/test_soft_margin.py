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
n1 = 100
n2 = 100
x, y = TestCase.gen_syn_data(n1, n2)
x_train, x_test, y_train, y_test = cv.train_test_split(x, y, train_size=0.25, test_size=0.75, random_state=12)

epsi = 0.005
has_dcap = True
# max_iter = 10000
max_iter = None
r = 0.4
fw = FwBoost(epsi, has_dcap=has_dcap, ratio=r, steprule=1)
fw.train(x_train, y_train, codetype="cy", approx_margin=False, max_iter=max_iter, ftr='wl')
# fw = FwBoost(epsi, has_dcap=has_dcap, ratio=r, steprule=2)
# fw.train(x_train, y_train, codetype="py", approx_margin=True, max_iter=max_iter, ftr='wl')
pd = ParaBoost(epsi, has_dcap=has_dcap, ratio=r)
pd.train(x_train, y_train, max_iter=max_iter, ftr='wl')

row = 2
col = 2
plt.subplot(row, col, 1)
plt.semilogx(pd.iter_num, pd.margin, 'b-', label='pd')
plt.semilogx(fw.iter_num, fw.margin, 'r-', label='fw')
plt.ylabel('margin')
plt.xlabel('log number of iteration')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend(loc='best')
plt.subplot(row, col, 2)
plt.semilogx(pd.iter_num, pd.gap, 'b-', label='pd')
plt.semilogx(fw.iter_num, fw.gap, 'r-', label='fw')
plt.ylabel('primal dual gap')
plt.xlabel('log number of iteration')
plt.legend(loc='best')
plt.savefig('../output/synth2_soft_margin_gap.pdf', format='pdf')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.subplot(row, col, 3)
plt.figure()
plt.semilogx(pd.iter_num, pd.primal_obj, 'b-', label='pd')
plt.semilogx(fw.iter_num, fw.primal_obj, 'r-', label='fw')
plt.xlabel('log number of iteration')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend(loc='best')
plt.tight_layout()
# plt.show()

plt.figure()
a = np.fabs(fw.alpha)
a /= a.max()
a = np.sort(a, kind='quicksort')[::-1]
b = np.fabs(pd.alpha)
b /= b.max()
b = np.sort(b, kind='quicksort')[::-1]
n = len(a)
plt.plot(range(1, n+1), a, 'r-', label='fw')
plt.plot(range(1, n+1), b, 'b-', label='pd')
plt.ylabel("relative magnitude of weights")
plt.xlabel('sorted weights ')
plt.legend(loc='best')
plt.ylim(ymax=1.2)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig('../output/synth2_soft_margin_sparsity.pdf', format='pdf')
