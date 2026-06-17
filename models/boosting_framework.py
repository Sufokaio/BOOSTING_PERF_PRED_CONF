# Generic L2 gradient boosting loop, any sklearn-compatible regressor can serve as base learner.
import numpy as np


class GradientBoostingFramework:

    def __init__(self, base_learner_class, base_learner_params, M, eta):
        self.base_learner_class  = base_learner_class
        self.base_learner_params = base_learner_params
        self.M   = M
        self.eta = eta
        self.learners_ = []
        self.F0_ = 0.0

    def _make_learner(self, stage=0):
        return self.base_learner_class(**self.base_learner_params)

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        self.F0_ = float(np.mean(y_train))
        self.learners_ = []

        F = np.full(len(y_train), self.F0_)
        for m in range(self.M):
            residuals = y_train - F
            learner = self._make_learner(stage=m)
            learner.fit(X_train, residuals)
            self.learners_.append(learner)
            F = F + self.eta * learner.predict(X_train)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.F0_)
        for learner in self.learners_:
            F = F + self.eta * learner.predict(X)
        return F
