__author__ = 'qdengpercy'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy
import heapq
import bisect


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
    d = d0.copy()/d0.sum()
    m = len(d)
    if v < 1.0 / m:
        print "error"
    ind = np.argsort(d0, kind='quicksort')[::-1]
    uu = d[ind]
    Z = uu.sum()
    for i in range(m):
        e = (1 - v * i) / Z
        if e * uu[i] <= v:
            break
        Z -= uu[i]

    # if d.max()>1 or d.min()<0:
    #     print ''
    d = np.minimum(v, e * d0)
    return d


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