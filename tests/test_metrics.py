import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import mre, mae, rmse, mibre, compute_all


def test_mre_known_values():
    result = mre(np.array([100, 200]), np.array([110, 180]))
    assert abs(result - 10.0) < 1e-9


def test_mre_zero_true_skipped():
    result = mre(np.array([0, 100]), np.array([5, 110]))
    assert abs(result - 10.0) < 1e-9


def test_mre_all_zeros_returns_nan():
    result = mre(np.array([0, 0]), np.array([1, 2]))
    assert math.isnan(result)


def test_mae_known_values():
    result = mae(np.array([1, 2, 3]), np.array([2, 2, 2]))
    assert abs(result - 2.0 / 3) < 1e-9


def test_rmse_known_values():
    result = rmse(np.array([0, 0]), np.array([3, 4]))
    assert abs(result - math.sqrt(12.5)) < 1e-9


def test_mibre_known_values():
    result = mibre(np.array([10, 20]), np.array([12, 16]))
    expected = (2 / 12 + 4 / 20) / 2
    assert abs(result - expected) < 1e-6


def test_mibre_both_zero_skipped():
    result = mibre(np.array([0, 10]), np.array([0, 12]))
    assert abs(result - 2 / 12) < 1e-9


def test_compute_all_returns_four_keys():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_all(y, y + 0.1)
    assert set(result.keys()) == {"MRE", "MAE", "RMSE", "MIBRE"}


def test_no_nan_on_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_all(y, y)
    assert result["MAE"]  == 0.0
    assert result["RMSE"] == 0.0
    assert result["MRE"]  == 0.0
    assert result["MIBRE"] == 0.0
