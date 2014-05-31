__author__ = 'qdengpercy'
"""
implement different ways to extract features from raw data
"""
import numpy as np
import sklearn.cross_validation as cv
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def weak_learner_pred(cls, x):
    """
    call scklearn's adaboost and obtain weak learners' output on raw feature x,
    """
    # weak_learners = cls.estimators_
    n_samples = x.shape[0]
    n_estimators = (cls.n_estimators)
    h = np.zeros((n_samples, n_estimators))
    classes = np.array([-1, 1])
    for i, estimator in enumerate(cls.estimators_):
        if cls.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            current_pred = cls._samme_proba(estimator, cls.n_classes_, x)
        else:  # elif self.algorithm == "SAMME":
            current_pred = estimator.predict(x)
            # current_pred = (current_pred == classes).T
        h[:, i] = classes.take(current_pred > 0, axis=0)
    return h
