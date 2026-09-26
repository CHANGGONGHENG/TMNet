"""Gentle AdaBoost (GentleBoost) classifier.

GentleBoost (Friedman, Hastie & Tibshirani, *Additive logistic regression: a statistical view of
boosting*, Annals of Statistics 2000) is one of the 30 models compared in the manuscript. It is not
available in scikit-learn, so it is implemented here as a scikit-learn compatible estimator using
depth-1 regression trees as base learners.

Algorithm
---------
``w_i = 1/n``; ``F(x) = 0``

for m = 1 ... M:
    fit a weighted least-squares regression stump ``f_m`` to the signed response ``y* = 2y - 1``
    with weights ``w``
    ``F(x) += learning_rate * f_m(x)``
    ``w_i *= exp(-y*_i * learning_rate * f_m(x_i))``, then renormalise

Prediction: ``P(y = 1) = 1 / (1 + exp(-2 F(x)))``
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor


class GentleBoostClassifier(BaseEstimator, ClassifierMixin):
    """GentleBoost with regression stumps."""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 1.0,
                 max_depth: int = 1, random_state: int | None = None) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        signed = 2.0 * y - 1.0
        n = len(y)

        weights = np.full(n, 1.0 / n)
        self.estimators_ = []
        self.F_train_ = np.zeros(n)

        for _ in range(int(self.n_estimators)):
            stump = DecisionTreeRegressor(max_depth=self.max_depth,
                                          random_state=self.random_state)
            stump.fit(x, signed, sample_weight=weights)
            pred = stump.predict(x)
            self.estimators_.append(stump)
            self.F_train_ += self.learning_rate * pred
            weights = weights * np.exp(-signed * self.learning_rate * pred)
            total = weights.sum()
            weights = weights / total if total > 0 else np.full(n, 1.0 / n)

        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        f = np.zeros(x.shape[0])
        for stump in self.estimators_:
            f += self.learning_rate * stump.predict(x)
        return 2.0 * f

    def predict_proba(self, x) -> np.ndarray:
        score = np.clip(self.decision_function(x), -50.0, 50.0)
        p1 = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
