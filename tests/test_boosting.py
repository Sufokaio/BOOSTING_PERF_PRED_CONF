import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.svr_boost import SVRBoost
from models.ridge_boost import RidgeLRBoost
from models.mlp_boost import MLPBoost
from models.tree_boost import TreeBoost
from models.baselines import XGBoostBaseline, CARTBaseline, SVRBaseline, RidgeLRBaseline


def _simple_1d_data(n=50, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 1) * 4 - 2
    y = X[:, 0] ** 2 + rng.randn(n) * 0.1
    return X, y


def _linear_data(n=20):
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    return X, y


def test_boosting_reduces_residuals():
    X, y = _simple_1d_data(n=80)
    X_train, X_test = X[:60], X[60:]
    y_train, y_test = y[:60], y[60:]

    for ModelClass in [SVRBoost, RidgeLRBoost, MLPBoost, TreeBoost]:
        m1 = ModelClass(M=1,  eta=0.1)
        m50 = ModelClass(M=50, eta=0.1)
        m1.fit(X_train, y_train)
        m50.fit(X_train, y_train)

        mse1  = np.mean((y_train - m1.predict(X_train))  ** 2)
        mse50 = np.mean((y_train - m50.predict(X_train)) ** 2)
        assert mse50 <= mse1 + 1e-6, (
            f"{ModelClass.__name__}: M=50 MSE ({mse50:.4f}) not <= M=1 MSE ({mse1:.4f})"
        )


def test_svr_boost_manual():
    X, y = _linear_data(n=20)
    model = SVRBoost(M=1, eta=1.0)
    model.fit(X, y)

    assert abs(model.F0_ - np.mean(y)) < 1e-6, "F0_ must equal mean(y_train)"
    residuals_stage0 = y - model.F0_
    pred_residuals = model.learners_[0].predict(X)
    corr = np.corrcoef(residuals_stage0, pred_residuals)[0, 1]
    assert corr > 0.0, "Stage-0 learner predictions should positively correlate with residuals"


def test_tree_boost_matches_sklearn():
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.RandomState(7)
    X = rng.rand(60, 4)
    y = X[:, 0] + X[:, 1] ** 2 + rng.randn(60) * 0.05

    tb = TreeBoost(M=10, eta=0.1)
    tb.fit(X, y)

    sk = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1,
                                   max_depth=3, random_state=0)
    sk.fit(X, y)

    np.testing.assert_array_almost_equal(
        tb.predict(X), sk.predict(X), decimal=5,
        err_msg="TreeBoost must match sklearn GradientBoostingRegressor"
    )


def test_all_models_same_interface():
    rng = np.random.RandomState(0)
    X = rng.rand(30, 4)
    y = rng.rand(30)

    models = [
        SVRBoost(M=2, eta=0.1),
        RidgeLRBoost(M=2, eta=0.1),
        MLPBoost(M=2, eta=0.1),
        TreeBoost(M=2, eta=0.1),
        XGBoostBaseline(),
        CARTBaseline(),
        SVRBaseline(),
        RidgeLRBaseline(),
    ]
    for model in models:
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,), f"{type(model).__name__}: predict() shape mismatch"
        assert not np.any(np.isnan(preds)), f"{type(model).__name__}: NaN in predictions"
