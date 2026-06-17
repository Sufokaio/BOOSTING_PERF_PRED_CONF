from sklearn.ensemble import GradientBoostingRegressor
from config import TREE_PARAMS, DEFAULT_M, DEFAULT_ETA


class TreeBoost:
    def __init__(self, M=DEFAULT_M, eta=DEFAULT_ETA):
        self.M   = M
        self.eta = eta
        self.model = GradientBoostingRegressor(
            n_estimators=M, learning_rate=eta,
            random_state=0, **TREE_PARAMS
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
