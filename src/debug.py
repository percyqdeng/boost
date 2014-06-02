__author__ = 'qdengpercy'

from fwboost import *
from paraboost import *
from test_boost import TestCase
import sklearn.cross_validation as cv
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


if __name__ == '__main__':
    n1 = 1000
    n2 = 100
    x, y = TestCase.gen_syn_data(n1, n2)
    x_train, x_test, y_train, y_test = cv.train_test_split(x, y, train_size=0.25, test_size=0.75, random_state=10)
    # ss = cv.ShuffleSplit(2*n1, n_iter=1, train_size=0.25, test_size=0.75)
    epsi = 0.001
    has_dcap = True
    r = 0.4
    pd = ParaBoost(epsi, has_dcap=has_dcap, ratio=r)
    pd.train(x_train, y_train, early_stop=False, ftr='wl')
    fw = FwBoost(epsi, has_dcap=has_dcap, ratio=r)
    fw.train(x_train, y_train, codetype="cy", approx_margin=True, early_stop=False, ftr='wl')
    row = 2
    col = 2
    plt.subplot(row, col, 1)
    plt.plot(pd.iter_num, pd.margin, 'b-', label='pd')
    plt.plot(fw.iter_num, fw.margin, 'r-', label='fw')
    plt.ylabel('margin')
    plt.xlabel('number of iteration')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.subplot(row, col, 2)
    plt.plot(pd.iter_num, pd.gap, 'b-', label='pd')
    plt.plot(fw.iter_num, fw.gap, 'r-', label='fw')
    plt.ylabel('primal dual gap')
    plt.xlabel('number of iteration')
    # plt.semilogx(pd.iter_num, pd.margin)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.subplot(row, col, 3)
    plt.plot(pd.iter_num, pd.primal_obj, 'b-', label='pd')
    plt.plot(fw.iter_num, fw.primal_obj, 'r-', label='fw')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout()