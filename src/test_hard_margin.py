__author__ = 'qdengpercy'


from fwboost import *
from paraboost import *
from extract_features import *

if __name__ == '__main__':
    """
    compare margin
    """
    print "--------- test hard margin -------------"
    n = 500
    n1 = 10
    n2 = 10
    x = np.random.random_integers(0, 1, size=(n, n1))
    x = 2*x.astype(np.float) - 1
    z = np.sum(x[:, :5], axis=1)
    y = np.sign(z)
    m = np.min(np.abs(z))
    # repeat feature with infused noise
    x2 = np.repeat(x, n2, axis=1)
    x2 += np.random.normal(0, 0.1, size=(n, n1*n2))
    x = np.hstack(((x, x2)))
    x[x>1] = 1
    x[x<-1] = -1
    print "generate data n=%s, n1=%s, n2=%s, margin=%s" % (n, n1, n2, m)
    cap_prob = False
    stop = False
    epsi = 0.01
    fw = FwBoost(epsi=epsi, has_dcap=cap_prob, ratio=0.1)
    fw.train(x, y, codetype='cy', approx_margin=True, early_stop=stop)
    pd = ParaBoost(epsi=epsi, has_dcap=cap_prob, ratio=0.1)
    pd.train(xtr=x, ytr=y, early_stop=stop)

    # print "margin fw:%s pd%s" % (fw.margin[-1], pd.margin[-1])
    print "margin pd%s" % (pd.margin[-1])
    plt.figure()
    plt.ylim(0.05, 0.2)
    plt.plot(fw.iter_num, fw.margin,'r-',label='fw')
    plt.plot(pd.iter_num, pd.margin, 'b-', label='pd')
    plt.ylabel('margin')
    plt.legend(loc='best')
    plt.savefig("../output/synth_hard_margin.pdf")

    plt.figure()
    plt.ylim(0.05, 0.2)
    plt.plot(fw.iter_num, fw.margin, 'r-', label='fw')
    plt.plot(pd.iter_num, pd.margin, 'b-', label='pd')
    plt.ylabel('margin')
    plt.legend(loc='best')
    plt.savefig("../output/synth_hard_margin_log.pdf")