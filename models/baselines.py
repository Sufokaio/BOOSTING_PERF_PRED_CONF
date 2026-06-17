from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import xgboost as xgb


class XGBoostBaseline:
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params, verbosity=0)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class CARTBaseline:
    def __init__(self, **params):
        self.model = DecisionTreeRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class SVRBaseline:
    def __init__(self, **params):
        self.model = SVR(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RidgeLRBaseline:
    def __init__(self, **params):
        self.model = Ridge(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
