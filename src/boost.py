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
    projection with the entropy distance, with capped distribution
    min KL(d,d0) sub to max_i d(i) <=v
    '''
    d = d0

    m = len(d)
    if v < 1.0/m:
        print "error"
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

def prox_mapping(v, x0, sigma, dist_option=2):
    '''
    prox-mapping  argmin_x   <v,x> + 1/sigma D(x0,x)
    distance option:
    dist_option:    1  euclidean distance, 0.5||x-x0||^2
                    2  kl divergence
    '''
    if dist_option == 1:
        x = x0 - sigma * v
    elif dist_option ==2:
        x = x0 * np.exp(-sigma * v)
        x = x/x.sum()

    return x

def pdboost(H, epsi, hasCap, r, max_iter):
    '''
    primal-dual boost with capped probability ||d||_infty <= 1/k
    '''

    print '----------------primal-dual boost-------------------'
    H = np.hstack((H, -H))
    (n, p) = H.shape
    nu = int(n * r)
    gaps = np.zeros(max_iter)
    margin = np.zeros(max_iter)
    primal_val = np.zeros(max_iter)
    dual_val = np.zeros(max_iter)
    # gaps[0] = 100
    showtimes = 5
    d = np.ones(n)/n

    d_bar = np.ones(n)/n
    a_bar = np.ones(p)/p
    a = np.ones(p)/p
    # a_bar = a
    a_tilde = np.ones(p)/p
    # d_tilde = np.zeros(p)
    theta = 1
    sig = 1
    tau = 1
    t = 0
    while t < max_iter:

        d = prox_mapping(np.dot(H, a_tilde), d, tau, 2)

        if hasCap:
            d2 = proj_cap_ent(d, 1.0/nu)
            # d_new = d_new/d_new.sum()
            if np.abs(d.sum() - d2.sum())>0.0001:
                print 'error'
            d = d2
        d_tilde = d
        dtH = np.dot(d_tilde, H)
        a_new = prox_mapping(-dtH, a, sig, 2)
        # a_new = proj_l1ball(tmp, 1)
        a_tilde = a_new + theta*(a_new - a)
        a = a_new
        d_bar *= t/(t+1.0)
        d_bar += 1.0/(t+1)*d
        a_bar *= t/(t+1.0)
        a_bar += 1.0/(t+1)*a

        if hasCap:
            Ha = np.dot(H,a_bar)
            min_margin = ksmallest(Ha, nu)
            primal_val[t] = -np.mean(min_margin)
        else:
            primal_val[t] = - np.min(np.dot(H,a_bar))
        margin[t] = -primal_val[t]
        dual_val[t] = -np.max(np.dot(d_bar, H))
        gaps[t] = primal_val[t] - dual_val[t]
        if t % np.int(max_iter/showtimes) == 0:
            print 'iter '+str(t)+' '+str(gaps[t])
            # print 'primal: '+str(-(ksmallest(Ha, k)).sum()/k)
            # print 'dual: '+str(-LA.norm(dtH, np.inf))
        if gaps[t] <epsi:
            break
        t += 1
    gaps = gaps[:t]
    primal_val = primal_val[:t]
    dual_val = dual_val[:t]
    return a_bar, d_bar, gaps, primal_val, dual_val, margin





def dboost(A, epsi, hasCap, r, max_iter=100, rule=1):
    '''
    boosting with conditional-gradient algorithm,from Shai Shwartz & Yoram Singer's boosting paper
    '''
    print '----------------dual boost-------------------'
    [n, m] =A.shape
    nu = int(r*n)
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
            d = proj_cap_ent(d, 1.0/nu)

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

    return w, d, gaps, eta, dual, margins, t,



if __name__ == '__main__':
    '''
    toy example
    '''
