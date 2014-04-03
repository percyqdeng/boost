__author__ = 'qdengpercy'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy
import heapq
import bisect
path = '/Users/qdengpercy/workspace/boost'
def proj_simplex(u, z):
    '''
    find w :  min 0.5*||w-u||^2 s.t. w>=0; w1+w2+...+wn = z; z>0
    '''
    p = len(u)
    ind = np.argsort(u,kind='quicksort')[::-1]
    mu = u[ind]
    s = np.cumsum(mu)
    tmp = 1.0/np.asarray([i+1 for i in range(p)])
    tmp *= (s-z)
    I = np.where((mu-tmp) > 0)[0]
    rho = I[-1]
    w = np.maximum(u-tmp[rho], 0)
    return w

def proj_l1ball(u, z):
    '''
    find w :  min 0.5*||w-u||^2 s.t. ||w||_1 <= z
    '''
    if LA.norm(u, 1) <= z:
        w = u
        return w
    sign = np.sign(u)
    w = proj_simplex(np.abs(u), z)
    w = w*sign
    return w

def proj_cap_ent(d0, v):
    '''
    projection with the entropy distance
    '''
    d = d0
    m = len(d)
    ind = np.argsort(d0, kind='quicksort')[::-1]

    u = d[ind]
    Z = u.sum()
    for i in range(m):
        e = (1-v*i)/Z
        if e*u[i] <= v:
            break
        Z -= u[i]

    d = np.minimum(v, e * d0)
    return d

def ksmallest(u0, k):
    u = u0.tolist()
    mins = u[:k]
    mins.sort()
    for i in u[k:]:
        if i < mins[-1]:
            mins.append(i)
            # np.append(mins, i)
            mins.sort()
            mins = mins[:k]
    return np.asarray(mins)
def pdboost3(H, epsi, hasCap, k, max_iter):
    '''
    primal-dual boost with capped probability ||d||_infty <= 1/k
    similar to pdboost, the only difference is w becomes the primal variable now.
    '''
    print '----------------primal-dual boost3-------------------'
    H = np.hstack((H, -H))
    (n, p) = H.shape
    gaps = np.zeros(max_iter+1)
    margins = np.zeros(max_iter+1)
    gaps[0] = 100
    showtimes = 5
    d = np.ones(n)/n

    d_bar = d
    a = np.ones(p)/p
    a_bar = a
    theta = 1
    sig = 1
    tau = 1
    # sig = 1/np.sqrt(p)
    # tau = 1/np.sqrt(p)
    # sig = 1.0/(p/np.log(p))
    # tau = 1.0/np.log(p)
    t = 0
    while gaps[t] > epsi and t < max_iter:
        t += 1

        d_new = d * np.exp(-tau * np.dot(H, a_bar))
        d_new = d_new/d_new.sum()
        if hasCap:
            d_new = proj_cap_ent(d_new, 1.0/k)
            # d_new = d_new/d_new.sum()
        d_bar = d_new
        dtH = np.dot(d_bar, H)
        tmp = a + sig * dtH
        a_new = a * np.exp(sig * dtH)
        a_new = a_new/a_new.sum()
        # a_new = proj_l1ball(tmp, 1)
        a_bar = a_new + theta*(a_new - a)
        a = a_new
        # something wrong, it should be unnecessary to renormalize
        d = d_new
        # dtH = np.dot(d, H)
        Ha = np.dot(H, a)

        if hasCap:
            gaps[t] = LA.norm(dtH, np.inf) - (ksmallest(Ha, k)).sum()/k
            margins[t] = (ksmallest(Ha, k)).sum()/k
        else:
            gaps[t] = LA.norm(dtH, np.inf) - np.min(Ha)
            margins[t] = np.min(Ha)
        if gaps[t] < 0:
            print 'error'
        if t % np.int(max_iter/showtimes) == 0:
            print 'iter '+str(t)+' '+str(gaps[t])
            # ind = np.argsort(Ha, kind='quicksort')
            # d_s = np.zeros(n)
            # d_s[ind[:k]] = 1.0/k
            print 'primal: '+str(-(ksmallest(Ha, k)).sum()/k)
            print 'dual: '+str(-LA.norm(dtH, np.inf))
            # print 'intermediate: '+str(-np.dot(dtH, a))
            # print 'norm '+str(d.sum())
    return a, d, gaps, margins, t,

def pdboost2(H, epsi, hasCap, k, max_iter):
    '''
    primal-dual boost with capped probability ||d||_infty <= 1/k
    similar to pdboost, the only difference is w becomes the primal variable now.
    '''
    print '----------------primal-dual boost-------------------'
    (n, p) = H.shape
    gaps = np.zeros(max_iter+1)
    margins = np.zeros(max_iter+1)
    gaps[0] = 100
    showtimes = 5
    d = np.ones(n)/n
    d_bar = d
    a = np.zeros(p)
    a_bar = a
    theta = 1
    # sig = .1
    # tau = .1
    sig = 1/np.sqrt(p)
    tau = 1/np.sqrt(p)
    # sig = 1.0/(p/np.log(p))
    # tau = 1.0/np.log(p)
    t = 0
    while gaps[t] > epsi and t < max_iter:
        t += 1

        d_new = d * np.exp(-tau * np.dot(H, a_bar))
        d_new = d_new/d_new.sum()
        if hasCap:
            d_new = proj_cap_ent(d_new, 1.0/k)
            # d_new = d_new/d_new.sum()
        d_bar = d_new
        dtH = np.dot(d_bar, H)
        tmp = a + sig * dtH
        a_new = proj_l1ball(tmp, 1)
        a_bar = a_new + theta*(a_new - a)
        a = a_new
        # something wrong, it should be unnecessary to renormalize
        d = d_new
        # dtH = np.dot(d, H)
        Ha = np.dot(H, a)

        if hasCap:
            gaps[t] = LA.norm(dtH, np.inf) - (ksmallest(Ha, k)).sum()/k
            margins[t] = (ksmallest(Ha, k)).sum()/k
        else:
            gaps[t] = LA.norm(dtH, np.inf) - np.min(Ha)
            margins[t] = np.min(Ha)
        if gaps[t] < 0:
            print 'error'
        if t % np.int(max_iter/showtimes) == 0:
            print 'iter '+str(t)+' '+str(gaps[t])
            # ind = np.argsort(Ha, kind='quicksort')
            # d_s = np.zeros(n)
            # d_s[ind[:k]] = 1.0/k
            print 'primal: '+str(-(ksmallest(Ha, k)).sum()/k)
            print 'dual: '+str(-LA.norm(dtH, np.inf))
            print 'intermediate: '+str(-np.dot(dtH, a))
            # print 'norm '+str(d.sum())
    return a, d, gaps, margins, t


def pdboost(H, epsi, hasCap, k, max_iter):
    '''
    primal-dual boost with capped probability ||d||_infty <= 1/k
    '''
    print '----------------primal-dual boost-------------------'
    (n, p) = H.shape
    gaps = np.zeros(max_iter+1)
    margins = np.zeros(max_iter+1)
    gaps[0] = 100
    margins = np.zeros(max_iter+1)
    primals = np.zeros(max_iter+1)
    duals = np.zeros(max_iter+1)
    showtimes = 5
    d = np.ones(n)/n
    d_bar = d
    a = np.zeros(p)
    theta = 1
    # sig = .1
    # tau = .1
    sig = 1/np.sqrt(p)
    tau = 1/np.sqrt(p)
    # sig = 1.0/(p/np.log(p))
    # tau = 1.0/np.log(p)
    t = 0
    while gaps[t] > epsi and t < max_iter:
        t += 1
        dtH = np.dot(d_bar, H)
        tmp = a + sig * dtH
        a_new = proj_l1ball(tmp, 1)
        d_new = d * np.exp(-tau * np.dot(H, a_new))
        d_new = d_new/d_new.sum()
        if hasCap:
            d_new = proj_cap_ent(d_new, 1.0/k)
            # d_new = d_new/d_new.sum()

        d_bar = d_new + theta*(d_new - d)
        a = a_new
        # something wrong, it should be unnecessary to renormalize
        d = d_new
        dtH = np.dot(d, H)
        Ha = np.dot(H, a)
        margins[t] = np.min(Ha)
        primals[t] = LA.norm(dtH, np.inf)
        duals[t] = (ksmallest(Ha, k)).sum()/k
        if hasCap:
            gaps[t] = LA.norm(dtH, np.inf) - (ksmallest(Ha, k)).sum()/k
            margins[t] = (ksmallest(Ha, k)).sum()/k
        else:
            gaps[t] = LA.norm(dtH, np.inf) - np.min(Ha)
            margins[t] = np.min(Ha)
        if gaps[t] < 0:
            print 'error'
        if t % np.int(max_iter/showtimes) == 0:
            print 'iter '+str(t)+' '+str(gaps[t])
            # ind = np.argsort(Ha, kind='quicksort')
            # d_s = np.zeros(n)
            # d_s[ind[:k]] = 1.0/k
            print 'primal: '+str(LA.norm(dtH, np.inf))
            print 'dual: '+str((ksmallest(Ha, k)).sum()/k)
            print 'intermediate: '+str(np.dot(dtH, a))
            print 'norm '+str(d.sum())
    return a, d, gaps, margins, t, primals, duals

def dboost(A, epsi, hasCap, k, max_iter, rule):
    '''
    boosting with conditional-gradient algorithm,from Shai Shwartz & Yoram Singer's boosting paper
    '''
    print '----------------dual boost-------------------'
    [n, m] =A.shape
    w = np.zeros(m)
    if rule == 2:
        w = np.ones(m)/np.float(m)
    Aw = np.dot(A, w)
    beta = epsi/(2*np.log(n))
    t = 0
    gaps = np.zeros(max_iter+1)
    margins = np.zeros(max_iter+1)
    eta = np.zeros(max_iter+1)
    dual = np.zeros(max_iter+1)
    gaps[0] = 100
    showtimes = 5
    while gaps[t] > epsi and t < max_iter:
        t += 1
        d = np.exp(-Aw/beta)
        d /= d.sum()
        if hasCap:
            d = proj_cap_ent(d, 1.0/k)

        dTA = np.dot(d, A)
        j = np.argmax(np.abs(dTA))
        # j = np.argmax(dTA)
        if np.sign(dTA[j]) > 0:
            sign = 1
        else:
            sign = -1
        tmp = sign * A[:, j] - Aw
        gaps[t] = np.dot(d.T, tmp)
        dual[t] = np.dot(d, Aw) + beta*(np.log(n)+(d*np.log(d)).sum())
        if gaps[t] < epsi:
            break
        else:
            if rule == 1:
                eta[t] =beta * gaps[t]/(LA.norm(tmp, np.inf)**2)
                eta[t] = np.maximum(0, np.minimum(1, eta[t]))
            elif rule == 2:
                eta[t] = 2.0/(t+1)
            Aw = (1-eta[t]) * Aw + eta[t] * sign * A[:, j]
            # w_new = (1-eta[t])*w
            # w_new[j] += eta[t]
            w *= (1-eta[t])
            w[j] += eta[t]
            margins[t] = np.min(Aw)
        if t % np.int(max_iter/showtimes) == 0:
            print 'iter '+str(t)
            # print 'eta: '+str(eta[t])
            print 'dual: '+str(dual[t])
            # print 'dual: '+str(np.dot(A, w).min()) +' and '+str((np.dot(A, w_new).min()))
            print 'primal: '+str(beta*(np.log(n)+(d*np.log(d)).sum())+LA.norm(np.dot(d, A), np.inf))
            print 'gaps: '+str(gaps[t])
            # print 'increm: '+str(gaps[t]**2*beta/LA.norm(tmp, np.inf)**2)
            # print 'margin: '+str()

    return w, d, gaps, eta, dual, margins, t,



if __name__ == '__main__':
    '''
    toy example
    '''
    x = np.array([[-1, 1], [1, -1]])
    # w = np.array([1, 1])

    w = w / np.float(LA.norm(w, 2))
    y = np.sign(np.dot(x, w))
    num_iter = 1000
    k = 400
    row = 2
    col = 2
    hasCap = False
    yH = y[:, np.newaxis] * x
    (w1, d1, gaps1, eta1, dual1, m1, total_iter1) = dboost(yH, 0.01, hasCap, k, num_iter, 1)
    # return x
    plt.plot(range(1, total_iter1+1), ((gaps1[1:total_iter1+1])), 'r', label='dboost')
    (w3, d3, gaps3, m3, total_iter3) = pdboost3(x, 0.01, hasCap, k, num_iter)

    plt.plot(range(1, total_iter3+1), ((gaps3[1:total_iter3+1])), 'b', label='pdboost')


    '''
        test 3
            '''
