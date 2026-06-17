import sys
import os
import tempfile
import math
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import DataLoader, set_all_seeds
from evaluation.metrics import compute_all
from evaluation.timing import timer
from models.svr_boost import SVRBoost
from models.baselines import CARTBaseline
from results.collector import ResultCollector
from results.checkpointing import Checkpoint


def _make_csv(path, n_rows=50, n_features=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_rows, n_features)).astype(float)
    y = rng.rand(n_rows) * 100
    cols = [f"f{i}" for i in range(n_features)] + ["perf"]
    pd.DataFrame(np.column_stack([X, y]), columns=cols).to_csv(path, index=False)


def test_end_to_end_mini(tmp_path):
    csv_path = str(tmp_path / "mini.csv")
    _make_csv(csv_path, n_rows=50, n_features=5)

    results_path     = str(tmp_path / "rq1_raw.csv")
    checkpoint_path  = str(tmp_path / "checkpoint.json")

    collector  = ResultCollector()
    checkpoint = Checkpoint(checkpoint_path)

    loader = DataLoader(csv_path)
    N_RUNS = 3
    SEED_BASE = 42
    SAMPLE_SIZE = 20

    models = [
        ("SVRBoost", lambda: SVRBoost(M=5, eta=0.1)),
        ("CART",     lambda: CARTBaseline()),
    ]

    for run in range(N_RUNS):
        seed = SEED_BASE + run
        set_all_seeds(seed)
        X_train, X_test, y_train, y_test = loader.sample(SAMPLE_SIZE, seed)

        for model_name, factory in models:
            if checkpoint.is_done("mini", model_name, SAMPLE_SIZE, run):
                continue
            model = factory()
            with timer() as tt:
                model.fit(X_train, y_train)
            with timer() as pt:
                y_pred = model.predict(X_test)
            metrics = compute_all(y_test, y_pred)
            collector.record("mini", model_name, SAMPLE_SIZE, run, metrics, tt.elapsed, pt.elapsed)
            checkpoint.mark_done("mini", model_name, SAMPLE_SIZE, run)
            checkpoint.save()

        collector.save(results_path)

    df = pd.read_csv(results_path)
    assert len(df) == N_RUNS * len(models) * 4, f"Expected 24 rows, got {len(df)}"

    assert df["value"].notna().all(), "NaN metric values found"
    assert np.all(np.isfinite(df["value"].values)), "Non-finite metric values found"

    for run in range(N_RUNS):
        for model_name, _ in models:
            assert checkpoint.is_done("mini", model_name, SAMPLE_SIZE, run)

    df2 = pd.read_csv(results_path)
    assert len(df2) == len(df)


def test_mini_rq2(tmp_path):
    csv_path = str(tmp_path / "mini2.csv")
    _make_csv(csv_path, n_rows=50, n_features=5)

    results_path    = str(tmp_path / "rq2_raw.csv")
    checkpoint_path = str(tmp_path / "ckpt2.json")

    collector  = ResultCollector()
    checkpoint = Checkpoint(checkpoint_path)
    loader     = DataLoader(csv_path)

    M_VALUES   = [5, 10]
    ETA_VALUES = [0.05, 0.1]
    N_RUNS     = 2
    SAMPLE_SIZE = 20
    SEED_BASE  = 42

    from models.svr_boost import SVRBoost
    from models.ridge_boost import RidgeLRBoost
    boosted_models = [
        ("SVRBoost",     SVRBoost),
        ("RidgeLRBoost", RidgeLRBoost),
    ]

    for run in range(N_RUNS):
        seed = SEED_BASE + run
        set_all_seeds(seed)
        X_train, X_test, y_train, y_test = loader.sample(SAMPLE_SIZE, seed)

        for M in M_VALUES:
            for eta in ETA_VALUES:
                for cls_name, cls in boosted_models:
                    model_name = f"{cls_name}_M{M}_eta{eta}"
                    if checkpoint.is_done("mini2", model_name, SAMPLE_SIZE, run):
                        continue
                    model = cls(M=M, eta=eta)
                    with timer() as tt:
                        model.fit(X_train, y_train)
                    with timer() as pt:
                        y_pred = model.predict(X_test)
                    metrics = compute_all(y_test, y_pred)
                    collector.record("mini2", model_name, SAMPLE_SIZE, run,
                                     metrics, tt.elapsed, pt.elapsed)
                    checkpoint.mark_done("mini2", model_name, SAMPLE_SIZE, run)
                    checkpoint.save()

        collector.save(results_path)

    df = pd.read_csv(results_path)
    expected = len(M_VALUES) * len(ETA_VALUES) * len(boosted_models) * N_RUNS * 4
    assert len(df) == expected, f"Expected {expected} rows, got {len(df)}"


def test_hinnperf_wrapper_mini(tmp_path):
    try:
        import tensorflow  # noqa
    except ImportError:
        pytest.skip("TensorFlow not available")

    from models.hinnperf_wrapper import HINNPerfWrapper

    rng = np.random.RandomState(0)
    X_train = rng.rand(20, 5)
    y_train = rng.rand(20) * 100
    X_test  = rng.rand(10, 5)

    wrapper = HINNPerfWrapper(seed=42, test_mode=True)
    wrapper.fit(X_train, y_train)
    y_pred = wrapper.predict(X_test)

    assert y_pred.shape == (10,), f"Expected (10,), got {y_pred.shape}"
    assert np.all(np.isfinite(y_pred)), "HINNPerf returned non-finite predictions"
