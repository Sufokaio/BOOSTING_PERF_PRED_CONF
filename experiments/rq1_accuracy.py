# RQ1 experiment: train all model variants over the full M×eta grid and collect metrics (and baselines).
import logging
import os
import sys
from itertools import product as _iproduct

from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from data.loader import DataLoader, set_all_seeds
from evaluation.metrics import compute_all
from evaluation.timing import timer
from models.svr_boost import SVRBoost
from models.ridge_boost import RidgeLRBoost
from models.mlp_boost import MLPBoost
from models.tree_boost import TreeBoost
from models.baselines import XGBoostBaseline, CARTBaseline, SVRBaseline, RidgeLRBaseline
from models.hinnperf_wrapper import HINNPerfWrapper

logger = logging.getLogger(__name__)


def _expand_grid(grid_dict):
    keys = list(grid_dict.keys())
    return [dict(zip(keys, combo)) for combo in _iproduct(*grid_dict.values())]


def _model_factories(config=cfg):
    factories = []

    for cls_name, model_class in [
        ("SVRBoost",     SVRBoost),
        ("RidgeLRBoost", RidgeLRBoost),
        ("MLPBoost",     MLPBoost),
        ("TreeBoost",    TreeBoost),
    ]:
        for M in config.M_VALUES:
            for eta in config.ETA_VALUES:
                name = f"{cls_name}_M{M}_eta{eta}"
                factories.append(
                    (name, lambda M=M, eta=eta, cls=model_class: cls(M=M, eta=eta))
                )

    for p in _expand_grid(config.XGBOOST_GRID):
        name = f"XGBoost_n{p['n_estimators']}_lr{p['learning_rate']}_d{p['max_depth']}_mcw{p['min_child_weight']}"
        factories.append((name, lambda p=p: XGBoostBaseline(**p)))

    for p in _expand_grid(config.CART_GRID):
        name = (
            f"CART_d{p['max_depth']}"
            f"_mss{p['min_samples_split']}"
            f"_msl{p['min_samples_leaf']}"
        )
        factories.append((name, lambda p=p: CARTBaseline(**p)))

    for p in _expand_grid(config.BASELINE_SVR_GRID):
        name = f"SVRBaseline_C{p['C']}_eps{p['epsilon']}"
        factories.append((name, lambda p=p: SVRBaseline(**p)))

    for p in _expand_grid(config.BASELINE_RIDGE_GRID):
        name = f"RidgeLRBaseline_a{p['alpha']}"
        factories.append((name, lambda p=p: RidgeLRBaseline(**p)))

    factories.append(("HINNPerf", lambda: HINNPerfWrapper()))

    return factories


def _evaluate_one_run(model_name, model_factory, loader, sample_size, run,
                      dataset_name, seed_base):
    seed = seed_base + run
    set_all_seeds(seed)
    X_train, X_test, y_train, y_test = loader.sample(sample_size, seed)

    logger.info("[RQ1] START  %-45s | %s | size=%d | run=%d", model_name, dataset_name, sample_size, run + 1)
    train_elapsed = float("nan")
    pred_elapsed  = float("nan")
    try:
        model = model_factory()
        with timer() as train_t:
            model.fit(X_train, y_train)
        train_elapsed = train_t.elapsed
        with timer() as pred_t:
            y_pred = model.predict(X_test)
        pred_elapsed = pred_t.elapsed
        metrics = compute_all(y_test, y_pred)
    except Exception as exc:
        logger.error("[RQ1] FAILED %s run=%d — %s", model_name, run + 1, exc)
        metrics = {k: float("nan") for k in ["MRE", "MAE", "RMSE", "MIBRE"]}
    return model_name, run, metrics, train_elapsed, pred_elapsed


def _evaluate_hinnperf_run(loader, sample_size, run, dataset_name, seed_base):
    # HINNPerf uses TF1-style graph/session code — must run in isolated processes
    # to avoid tf.placeholder/eager conflicts across concurrent workers.
    from models.hinnperf_wrapper import HINNPerfWrapper
    return _evaluate_one_run(
        "HINNPerf", lambda: HINNPerfWrapper(),
        loader, sample_size, run, dataset_name, seed_base,
    )


def run_rq1(collector, checkpoint, config=cfg):
    all_models = _model_factories(config)
    n_jobs = getattr(config, "N_JOBS", 1)
    logger.info(
        "[RQ1] Starting — %d datasets | %d model variants | %d runs each | N_JOBS=%d",
        len(config.DATASETS), len(all_models), config.N_RUNS, n_jobs,
    )

    for dataset_name, csv_path in config.DATASETS.items():
        loader = DataLoader(csv_path)
        logger.info("[RQ1] Dataset: %s — %d sample size(s)", dataset_name, len(config.SAMPLE_SIZES[dataset_name]))

        for sample_size in config.SAMPLE_SIZES[dataset_name]:
            for model_name, model_factory in all_models:

                pending_runs = [
                    run for run in range(config.N_RUNS)
                    if not checkpoint.is_done(dataset_name, model_name, sample_size, run)
                ]

                if not pending_runs:
                    logger.info(
                        "[RQ1]   %-45s | %s | size=%d — all %d runs already checkpointed",
                        model_name, dataset_name, sample_size, config.N_RUNS,
                    )
                    continue

                logger.info(
                    "[RQ1]   %-45s | %s | size=%d | %d/%d runs pending — dispatching to %s",
                    model_name, dataset_name, sample_size,
                    len(pending_runs), config.N_RUNS,
                    (f"{n_jobs} processes" if model_name == "HINNPerf" else f"{n_jobs} threads")
                    if n_jobs != 1 else "sequential",
                )

                if n_jobs == 1:
                    run_results = [
                        _evaluate_one_run(model_name, model_factory, loader, sample_size, run,
                                          dataset_name, config.RANDOM_SEED_BASE)
                        for run in pending_runs
                    ]
                elif model_name == "HINNPerf":
                    run_results = Parallel(n_jobs=n_jobs, prefer="processes")(
                        delayed(_evaluate_hinnperf_run)(
                            loader, sample_size, run,
                            dataset_name, config.RANDOM_SEED_BASE,
                        )
                        for run in pending_runs
                    )
                else:
                    run_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_evaluate_one_run)(
                            model_name, model_factory, loader, sample_size, run,
                            dataset_name, config.RANDOM_SEED_BASE,
                        )
                        for run in pending_runs
                    )

                for _, run_idx, metrics, train_elapsed, pred_elapsed in run_results:
                    collector.record(
                        dataset_name, model_name, sample_size, run_idx,
                        metrics, train_elapsed, pred_elapsed,
                    )
                    checkpoint.mark_done(dataset_name, model_name, sample_size, run_idx)

                    mre_val = metrics.get("MRE", float("nan"))
                    logger.info(
                        "[RQ1] DONE   %-45s | %s | size=%d | run=%d/%d | MRE=%.2f%% | train=%.3fs",
                        model_name, dataset_name, sample_size, run_idx + 1, config.N_RUNS,
                        mre_val if mre_val == mre_val else -1, train_elapsed,
                    )
                    print(
                        f"[RQ1] {dataset_name} | size={sample_size} | "
                        f"run={run_idx+1}/{config.N_RUNS} | {model_name} | "
                        f"MRE={mre_val:.2f}% | train={train_elapsed:.3f}s"
                    )

                collector.save(os.path.join(config.RESULTS_DIR, "rq1_raw.csv"))
                checkpoint.save()

                logger.info(
                    "[RQ1] %-45s | %s | size=%d — %d runs complete",
                    model_name, dataset_name, sample_size, len(run_results),
                )
