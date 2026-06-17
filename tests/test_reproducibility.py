import sys
import os
import tempfile
import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import DataLoader, set_all_seeds


def _make_synthetic_csv(tmp_dir, n_rows=60, n_features=5):
    rng = np.random.RandomState(0)
    X = rng.rand(n_rows, n_features)
    y = rng.rand(n_rows)
    data = np.column_stack([X, y])
    path = os.path.join(tmp_dir, "synthetic.csv")
    cols = [f"f{i}" for i in range(n_features)] + ["target"]
    pd.DataFrame(data, columns=cols).to_csv(path, index=False)
    return path


def test_same_seed_same_split(tmp_path):
    path = _make_synthetic_csv(str(tmp_path))
    loader = DataLoader(path)
    r1 = loader.sample(20, seed=42)
    r2 = loader.sample(20, seed=42)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_different_splits(tmp_path):
    path = _make_synthetic_csv(str(tmp_path))
    loader = DataLoader(path)
    X_train1, _, _, _ = loader.sample(20, seed=42)
    X_train2, _, _, _ = loader.sample(20, seed=43)
    assert not np.array_equal(X_train1, X_train2)


def test_same_seed_same_model_output(tmp_path):
    from models.svr_boost import SVRBoost
    path = _make_synthetic_csv(str(tmp_path), n_rows=40)
    loader = DataLoader(path)
    X_train, X_test, y_train, _ = loader.sample(20, seed=42)

    set_all_seeds(42)
    m1 = SVRBoost(M=3, eta=0.1)
    m1.fit(X_train, y_train)
    p1 = m1.predict(X_test)

    set_all_seeds(42)
    m2 = SVRBoost(M=3, eta=0.1)
    m2.fit(X_train, y_train)
    p2 = m2.predict(X_test)

    np.testing.assert_array_almost_equal(p1, p2)


def test_run_i_uses_correct_seed(tmp_path):
    from unittest.mock import patch, MagicMock
    import config as cfg

    path = _make_synthetic_csv(str(tmp_path), n_rows=40)
    loader_mock = MagicMock()
    loader_mock.sample.return_value = (
        np.zeros((10, 5)), np.zeros((10, 5)), np.zeros(10), np.zeros(10)
    )

    seeds_used = []
    original_sample = DataLoader.sample

    with patch("data.loader.DataLoader") as MockLoader:
        instance = MockLoader.return_value
        def fake_sample(size, seed):
            seeds_used.append(seed)
            return (np.zeros((size, 5)), np.zeros((5, 5)), np.zeros(size), np.zeros(5))
        instance.sample.side_effect = fake_sample
        instance._load = MagicMock()
        instance._X = np.zeros((40, 5))
        instance._y = np.zeros(40)

        for i in range(5):
            seed = cfg.RANDOM_SEED_BASE + i
            instance.sample(10, seed)

    expected = [cfg.RANDOM_SEED_BASE + i for i in range(5)]
    assert seeds_used == expected
