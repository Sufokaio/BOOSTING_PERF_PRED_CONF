# Validation scores of every model variant.
import os

for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_v] = "1"

import argparse
import glob
import logging
import random
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from data.loader import DataLoader
from evaluation.metrics import compute_all
from experiments.rq1_accuracy import _model_factories
from models.svr_boost import SVRBoost
from models.ridge_boost import RidgeLRBoost
from models.mlp_boost import MLPBoost
from models.tree_boost import TreeBoost

warnings.filterwarnings("ignore")
logging.getLogger("evaluation.metrics").setLevel(logging.ERROR)

N_FOLDS = 5
STAGED = {"SVRBoost": SVRBoost, "RidgeLRBoost": RidgeLRBoost, "MLPBoost": MLPBoost, "TreeBoost": TreeBoost}
DIRECT = ["XGBoost", "CART", "SVRBaseline", "RidgeLRBaseline"]
TEST_REFIT = ["XGBoost"]
FAMILIES = list(STAGED) + DIRECT
COLUMNS = ["dataset", "model", "sample_size", "run", "metric", "value", "split"]

_loaders = {}


def _loader(dataset):
    if dataset not in _loaders:
        _loaders[dataset] = DataLoader(cfg.DATASETS[dataset])
    return _loaders[dataset]


def split_rows(loader, size, seed):
    X_raw, y = loader.load()
    n = len(y)
    rng = np.random.RandomState(seed)
    train_idx = rng.choice(n, size=size, replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    X_tr, X_te, y_tr, y_te = loader.sample(size, seed)
    scaler = MinMaxScaler().fit(X_raw[train_idx])
    if not (np.array_equal(scaler.transform(X_raw[train_idx]), X_tr)
            and np.array_equal(scaler.transform(X_raw[test_idx]), X_te)
            and np.array_equal(y[train_idx], y_tr) and np.array_equal(y[test_idx], y_te)):
        raise RuntimeError(f"split mismatch with DataLoader.sample (size={size}, seed={seed})")
    return X_raw, y, train_idx, (X_tr, X_te, y_tr, y_te)


def fold_indices(size, seed):
    return list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(np.arange(size)))


def staged_predictions(model, X, Ms):
    Ms = set(Ms)
    out = {}
    if isinstance(model, TreeBoost):
        for m, pred in enumerate(model.model.staged_predict(X), start=1):
            if m in Ms:
                out[m] = pred
        return out
    X = np.asarray(X, dtype=float)
    F = np.full(len(X), model.F0_)
    for m, learner in enumerate(model.learners_, start=1):
        F = F + model.eta * learner.predict(X)
        if m in Ms:
            out[m] = F.copy()
    return out


def grid_points(family):
    if family in STAGED:
        return [(f"{family}_M{M}_eta{eta}", M, eta) for eta in cfg.ETA_VALUES for M in cfg.M_VALUES]
    return [(name, fac, None) for name, fac in _model_factories(cfg) if name.split("_")[0] == family]


def fit_predict(family, X_fit, y_fit, X_pred):
    preds = {}
    if family in STAGED:
        M_max = max(cfg.M_VALUES)
        for eta in cfg.ETA_VALUES:
            model = STAGED[family](M=M_max, eta=eta).fit(X_fit, y_fit)
            for M, pred in staged_predictions(model, X_pred, cfg.M_VALUES).items():
                preds[f"{family}_M{M}_eta{eta}"] = pred
    else:
        for name, fac, _ in grid_points(family):
            preds[name] = fac().fit(X_fit, y_fit).predict(X_pred)
    return preds


def part_path(out_dir, dataset, size, family, run):
    return os.path.join(out_dir, "parts", dataset, str(size), f"{family}_run{run:02d}.csv")


def run_task(task):
    dataset, size, family, run, out_dir = task
    path = part_path(out_dir, dataset, size, family, run)
    if os.path.exists(path):
        return task, 0.0, "skipped"
    t0 = time.perf_counter()
    seed = cfg.RANDOM_SEED_BASE + run
    random.seed(seed)
    np.random.seed(seed)

    X_raw, y, train_idx, (X_tr, X_te, y_tr, y_te) = split_rows(_loader(dataset), size, seed)
    oof = {}
    for fit_pos, val_pos in fold_indices(size, seed):
        scaler = MinMaxScaler().fit(X_raw[train_idx[fit_pos]])
        X_fit = scaler.transform(X_raw[train_idx[fit_pos]])
        X_val = scaler.transform(X_raw[train_idx[val_pos]])
        for name, pred in fit_predict(family, X_fit, y[train_idx[fit_pos]], X_val).items():
            oof.setdefault(name, np.full(size, np.nan))[val_pos] = pred

    rows = []
    for name, pred in oof.items():
        for metric, value in compute_all(y[train_idx], pred).items():
            rows.append((dataset, name, size, run, metric, value, "val"))
    if family in TEST_REFIT:
        for name, pred in fit_predict(family, X_tr, y_tr, X_te).items():
            for metric, value in compute_all(y_te, pred).items():
                rows.append((dataset, name, size, run, metric, value, "test"))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return task, time.perf_counter() - t0, "done"


def build_tasks(out_dir, datasets, families):
    tasks = []
    for dataset in datasets:
        for size in cfg.SAMPLE_SIZES[dataset]:
            for family in families:
                for run in range(cfg.N_RUNS):
                    tasks.append((dataset, size, family, run, out_dir))
    priority = {"MLPBoost": 0, "SVRBoost": 1, "XGBoost": 2}
    tasks.sort(key=lambda t: (priority.get(t[2], 3), -t[1]))
    return tasks


def combine(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "parts", "*", "*", "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(out_dir, "val_raw.csv"), index=False)
    print(f"combined {len(files)} files, {len(df)} rows -> {os.path.join(out_dir, 'val_raw.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/val")
    parser.add_argument("--processes", type=int, default=12)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=FAMILIES)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()

    if args.combine:
        combine(args.out)
        return

    datasets = args.datasets or list(cfg.DATASETS)
    tasks = build_tasks(args.out, datasets, args.families)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "progress.log")
    t_start = time.perf_counter()
    n_done = 0
    with Pool(args.processes) as pool, open(log_path, "a") as log:
        for (dataset, size, family, run, _), elapsed, status in pool.imap_unordered(run_task, tasks):
            n_done += 1
            line = (f"{time.strftime('%H:%M:%S')} {n_done}/{len(tasks)} {status} "
                    f"{dataset} {size} {family} run{run:02d} {elapsed:.1f}s "
                    f"(wall {time.perf_counter() - t_start:.0f}s)")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    main()
