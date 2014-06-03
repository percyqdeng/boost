__author__ = 'qdengpercy'
import numpy as np
from numpy import linalg as LA
import time

path = '/Users/qdengpercy/workspace/boost'


class Boost(object):
    normalizer = 1
    avg = 0

    def __init__(self):
        pass

    def train(self, x, y):
        pass

    def test(self, x, y):
        pass

    def plot_result(self):
        pass

    def to_name(self):
        return "boost"

    def _process_train_data(self, xtr):
        self.normalizer = np.std(xtr, 0)
        self.avg = np.mean(xtr, 0)
        xtr = (xtr - self.avg[np.newaxis, :]) / self.normalizer[np.newaxis, :]
        return xtr

    def _process_test_data(self, xte):
        # normalize and add the intercept
        xte = (xte - self.avg[np.newaxis, :]) / self.normalizer[np.newaxis, :]
        return xte

    def _print_algo_info(self):
        pass


def proj_simplex(u, z):
    """
    find w :  min 0.5*||w-u||^2 s.t. w>=0; w1+w2+...+wn = z; z>0
    """
    p = len(u)
    ind = np.argsort(u, kind='quicksort')[::-1]
    mu = u[ind]
    s = np.cumsum(mu)
    tmp = 1.0 / np.asarray([i + 1 for i in range(p)])
    tmp *= (s - z)
    I = np.where((mu - tmp) > 0)[0]
    rho = I[-1]
    w = np.maximum(u - tmp[rho], 0)
    return w


def proj_l1ball(u, z):
    """
    find w :  min 0.5*||w-u||^2 s.t. ||w||_1 <= z
    """
    if LA.norm(u, 1) <= z:
        w = u
        return w
    sign = np.sign(u)
    w = proj_simplex(np.abs(u), z)
    w = w * sign
    return w


def proj_cap_ent(d0, v):
    """
    projection with the entropy distance, with capped distribution
    min KL(d,d0) sub to max_i d(i) <=v
    """
    d = d0
    m = len(d)
    if v < 1.0 / m:
        print "error"
    # ind = np.argsort(d, kind='quicksort')[::-1]
    # uu = d[ind]
    uu = np.sort(d, kind='quicksort')[::-1]
    Z = uu.sum()
    try:
        for i in xrange(m):
            # if Z == 0:
            #     break
            e = (1 - v * i) / Z
            if e * uu[i] <= v:
                break
            Z -= uu[i]
            if e < 0:
                print ""
    except FloatingPointError:
        print "Z: %f, sum: %d" % (Z, uu.sum())
    # except Exception as err:
    #     pdb.set_trace()
    if d.max()>1 or d.min()<0:
        print ''
    d = np.minimum(v, e * d)
    return d


def cmp_obj_cap(h_a, mu, nu):
    n = h_a.shape
    d0 = np.ones(n) / n
    d = prox_mapping(h_a, d0, 1 / mu)
    d = proj_cap_ent(d, 1.0 / nu)
    res = -np.dot(d, h_a) - mu * np.dot(d, np.log(d*n))
    return res


def ksmallest2(u0, k):
    u = np.sort(u0, kind='quicksort')
    res = u[:k]
    return res


def k_avg(u0, k):
    v = ksmallest2(u0, k)
    return np.mean(v)


def ksmallest(u0, k):
    u = u0.tolist()
    mins = u[:k]
    mins.sort()
    for i, x in enumerate(u[k:]):
        if x < mins[-1]:
            mins.append(x)
            # np.append(mins, x)
            mins.sort()
            mins = mins[:k]
    return np.asarray(mins)

def check_ksm():
    n = 50000
    k = 5000
    rep = 20
    t1 = 0
    t2 = 0
    for i in xrange(rep):
        arr = np.random.normal(0, 1, n)
        # timeit.timeit("ksmallest(arr,k")
        start = time.time()
        ksmallest(arr, k)
        t1 += time.time() - start
        start = time.time()
        ksmallest2(arr, k)
        t2 += time.time() - start
    print "%d runs, average time t1 %f, t2 %f" % (rep, t1/rep, t2/rep)


def prox_mapping(v, x0, sigma, dist_option=2):
    """
    prox-mapping  argmin_x   <v,x> + 1/sigma D(x0,x)
    distance option:
    dist_option:    1  euclidean distance, 0.5||x-x0||^2
                    2  kl divergence
    """
    if dist_option == 1:
        x = x0 - sigma * v
    elif dist_option == 2:
        x = x0 * np.exp(-sigma * v)
        x = x / x.sum()

    return x


if __name__ == '__main__':
    pass