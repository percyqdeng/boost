__author__ = 'qdengpercy'

import os
import numpy as np
import scipy.io
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from boost import *
from gen_ftrs import *

# class BoostPara:
#     def __init__(self, epsi=0.01, hasDualCap = False, ratio=0.1, max_iter=100, steprule = 1):
#         self.epsi = epsi
#         self.hasDualCap = hasDualCap
#         self.ratio = ratio
#         self.max_iter = max_iter
#         self.steprule = steprule


def preprocess_data(x):
	Z = np.std(x, 0)
	avg = np.mean(x, 0)
	x = (x - avg[np.newaxis, :])/Z[np.newaxis, :]
	return x

class AdaFwBoost:
	"""Adaptive Frank-Wolfe Boosting method,similar to adaboost, but apply Frank-Wolfe method on the constraint
	problem
	Parameters:
	------------
	epsi ：optimization tolerance,
	has_dcap : boolean
		has capped probability in the dual variable
	ratio : float the capped probability
	steprule : int, the stepsize choice in FW method
		1 : derive from quadratic surrogate function one
		2 : quadratic surrogate function two
		3 : line search
	"""
	def ada_fw_boosting(self):
		"""
		frank-wolfe boost for binary classification with user defined weak learners, choose decision stump
		as default
		"""
class FwBoost:
	"""Frank-Wolfe Boosting method,
	Parameters:
	------------
	epsi ：optimization tolerance,
	has_dcap : boolean
		has capped probability in the dual variable
	ratio : float the capped probability
	steprule : int, the stepsize choice in FW method
		1 : derive from quadratic surrogate function one
		2 : quadratic surrogate function two
		3 : line search
	"""
	def __init__(self, epsi=0.01, has_dcap=False, ratio=0.1, max_iter=100, steprule=1):
		self.epsi = epsi
		self.has_dcap = has_dcap
		self.ratio = ratio
		self.max_iter = max_iter
		self.steprule = steprule

	def train(self, xtr, ytr):
		# self.Z = np.std(xtr, 0)
		# self.mu = np.mean(xtr, 0)
		# xtr = (xtr - self.mu[np.newaxis, :])/self.Z[np.newaxis, :]
		ntr = xtr.shape[0]
		xtr = np.hstack((xtr, np.ones((ntr, 1))))
		yH = ytr[:, np.newaxis] * xtr
		self.res = self.fw_boosting(yH)

	def test(self, xte, yte):
		# normalize and add the intercept
		# xte = (xte-self.mu[np.newaxis, :])/self.Z[np.newaxis, :]
		nte = xte.shape[0]
		xte = np.hstack((xte, np.ones((nte, 1))))
		pred = np.sign(np.dot(xte, self.res['alpha']))
		return np.mean(pred != yte)

	def ada_fw_boosting(self, x, y):
		"""
		frank-wolfe boost for binary classification with user defined weak learners, choose decision stump
		as default
		x : array of shape n*p
		y : array of shape n*1

		"""
		n, p = x.shape

	def fw_boosting(self, H):
		"""
		frank-wolfe boost for binary classification with weak learner as matrix H
		min_a max_d   d^T(-Ha) sub to:  ||a||_1\le 1
		"""
		[n, p] = H.shape
		gaps = np.zeros(self.max_iter)
		margins = np.zeros(self.max_iter)
		prim_obj = np.zeros(self.max_iter)
		dual_obj = np.zeros(self.max_iter)
		alpha = np.zeros(p)
		# alpha = np.ones(p)/p
		d0 = np.ones(n)/n
		mu = self.epsi/(2*np.log(n))
		# mu = 1
		nu = int(n * self.ratio)
		t = 0
		Ha = np.dot(H, alpha)
		# d0 = np.ones(n)/n
		while t < self.max_iter:
			d_next = prox_mapping(Ha, d0, 1/mu)
			if math.isnan(d_next[0]):
				print 'omg'
			if self.has_dcap:
				d_next = proj_cap_ent(d_next, 1.0/nu)
			d = d_next
			dtH = np.dot(d, H)
			j = np.argmax(np.abs(dtH))
			ej = np.zeros(p)
			ej[j] = np.sign(dtH[j])
			gaps[t] = np.dot(dtH, ej-alpha)
			if self.has_dcap:
				min_margin = ksmallest(Ha, nu)
				margins[t] = np.mean(min_margin)
			else:
				margins[t] = np.min(Ha)
			prim_obj[t] = mu * np.log(1.0/n*np.sum(np.exp(-Ha/mu)))
			dual_obj[t] = -np.max(np.abs(dtH)) - mu*np.dot(d,np.log(d)) + mu*np.log(n)
			if self.steprule == 1:
				eta = np.maximum(0, np.minimum(1, mu*gaps[t]/np.sum(np.abs(alpha-ej))**2))
			elif self.steprule == 2:
				eta = np.maximum(0, np.minimum(1, mu*gaps[t]/LA.norm(Ha-H[:, j]*ej[j], np.inf)**2))
			else:
				#
				# do line search
				#
				print "steprule 3, to be done"
			alpha *= (1-eta)
			alpha[j] += eta*ej[j]
			Ha *= (1-eta)
			Ha += H[:, j] * (eta*ej[j])
			if gaps[t] < self.epsi:
				t += 1
				break
			t += 1
		gaps = gaps[:t]
		margins = margins[:t]
		prim_ojb = prim_obj[:t]
		dual_obj = dual_obj[:t]
		res = {'alpha': alpha, 'd': d, 'gap': gaps, 'eta': eta, 'marg': margins, 'pri': prim_obj, 'dual': dual_obj}
		return res

	def pdboost(self, H):
		"""
		primal-dual boost with capped probability ||d||_infty <= 1/k
		"""

		print '----------------primal-dual boost-------------------'
		H = np.hstack((H, -H))
		(n, p) = H.shape
		nu = int(n * self.ratio)
		gaps = np.zeros(self.max_iter)
		margin = np.zeros(self.max_iter)
		primal_val = np.zeros(self.max_iter)
		dual_val = np.zeros(self.max_iter)
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
		while t < self.max_iter:

			d = prox_mapping(np.dot(H, a_tilde), d, tau, 2)

			if self.has_dcap:
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
			d_bar *= t / (t+1.0)
			d_bar += 1.0 / (t+1) * d
			a_bar *= t/(t+1.0)
			a_bar += 1.0/(t+1)*a

			if self.has_dcap:
				Ha = np.dot(H,a_bar)
				min_margin = ksmallest(Ha, nu)
				primal_val[t] = -np.mean(min_margin)
			else:
				primal_val[t] = - np.min(np.dot(H,a_bar))
			margin[t] = -primal_val[t]
			dual_val[t] = -np.max(np.dot(d_bar, H))
			gaps[t] = primal_val[t] - dual_val[t]
			if t % np.int(self.max_iter/showtimes) == 0:
				print 'iter '+str(t)+' '+str(gaps[t])
				# print 'primal: '+str(-(ksmallest(Ha, k)).sum()/k)
				# print 'dual: '+str(-LA.norm(dtH, np.inf))
			if gaps[t] <self.epsi:
				break
			t += 1
		gaps = gaps[:t]
		primal_val = primal_val[:t]
		dual_val = dual_val[:t]
		return a_bar, d_bar, gaps, primal_val, dual_val, margin

	def plot_result(self):
		r = 2
		c = 2
		plt.subplot(r,c,1)
		plt.plot((self.res['gap']),'rx-', markersize=0.3, label='gap')
		plt.title('primal-dual gap')
		plt.subplot(r, c, 2)
		plt.plot(self.res['marg'], 'bo-',markersize=0.3)
		plt.title('margin')
		plt.subplot(r, c, 3)
		p1 = plt.plot(self.res['pri'], 'rx-', markersize=0.3, label='primal')
		p2 = plt.plot(self.res['dual'], color='g',marker='o', markersize=0.3, label='dual')
		# plt.legend([p1],"primal")
		plt.title('primal objective')
		# plt.subplot(r,c,4)
		# plt.plot(self.res['dual'], 'rx-', markersize=0.3)
		plt.savefig('result.pdf')


def benchmark(data):
	x = data['x']
	y = data['t']
	y = np.squeeze(y)
	x = preprocess_data(x)
	para = BoostPara(epsi=0.01, has_dcap=False, ratio=0.11, max_iter=20000, steprule=2)
	# rep = data['train'].shape[0]
	rep = 1
	err_te = np.zeros(rep)
	err_tr = np.zeros(rep)
	for i in range(rep):
		trInd = data['train'][i, :] - 1
		teInd = data['test'][i, :] - 1

		xtr = x[trInd, :]
		ytr = y[trInd]
		xte = x[teInd, :]
		yte = y[teInd]

		booster = FwBoost(epsi=0.01, has_dcap=False, ratio=0.11, max_iter=20000, steprule=2)
		booster.train(xtr, ytr)

		# max_iter = 1000
		err_te[i] = booster.test(xte, yte)
		err_tr[i] = booster.test(xtr, ytr)
		print "rounds "+ str(i+1)+ "err_tr " + str(err_tr[i])+" err_te "+str(err_te[i])
	# plt.plot(booster.result['gap'], 'bx-')

	return booster, err_te


def toyTest():
	ntr = 1000
	(xtr, ytr, yH, margin, w, xte, yte) = gen_syn('disc', ntr, 1000)
	para = BoostPara(epsi=0.001, has_dcap=False, ratio=0.1, max_iter=500000, steprule=1)
	booster = FwBoost()
	booster.train(xtr, ytr, para)
	# plt.plot(booster.gaps,'rx-')
	booster.plot_result()
	return booster


def plot_2d_data(data):
	x = data['x']
	y = data['t']
	y = np.squeeze(y)
	plt.subplot(2, 2, 1)
	plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', c='r', label='+')
	plt.subplot(2, 2, 2)
	plt.scatter(x[y == -1, 0], x[y == -1, 1], marker='x', c='b', label='-')
	plt.subplot(2, 2, 3)
	plt.scatter(x[:, 0], x[:, 1])
	plt.title('banana')
	# plt.show()
	# plt.savefig('2dplot.eps')

if __name__ == '__main__':
	if os.name == "nt":
		path = "..\\dataset\\"
	elif os.name =="posix":
		path = '/Users/qdengpercy/workspace/boost/dataset/'


	dtname = 'bananamat.mat'
	# data = scipy.io.loadmat(path+dtname)
	# plot_2d_data(data)
	# plt.figure()
	# booster, err = benchmark(data)
	booster = toyTest()

