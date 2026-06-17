# MRE, MAE, RMSE, MIBRE computation.
import logging
import numpy as np

logger = logging.getLogger(__name__)


def mre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        logger.warning("mre: all y_true values are zero; returning nan")
        return float("nan")
    n_skipped = np.sum(~mask)
    if n_skipped > 0:
        logger.warning("mre: skipping %d zero y_true entries", n_skipped)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mibre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), np.abs(y_pred))
    mask = denom != 0
    if not np.any(mask):
        logger.warning("mibre: all denominators are zero; returning nan")
        return float("nan")
    n_skipped = np.sum(~mask)
    if n_skipped > 0:
        logger.warning("mibre: skipping %d all-zero entries", n_skipped)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MRE":   mre(y_true, y_pred),
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MIBRE": mibre(y_true, y_pred),
    }
