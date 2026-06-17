# RQ3 analysis: compute global and per-dataset cost-accuracy summaries and Pareto frontiers.
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from analysis.rq1_analysis import select_best_per_scenario, _filter_s1
from analysis.rq2_analysis import _parse_model_name


def _pareto_frontier(df, x_col, y_col):
    dominated = pd.Series(False, index=df.index)
    vals_x = df[x_col].values
    vals_y = df[y_col].values
    for i, idx in enumerate(df.index):
        dominated[idx] = any(
            (vals_x[j] <= vals_x[i]) and (vals_y[j] <= vals_y[i])
            and (vals_x[j] < vals_x[i] or vals_y[j] < vals_y[i])
            for j in range(len(df))
            if df.index[j] != idx
        )
    return ~dominated


def _agg_series(series, agg):
    return series.mean() if agg == "mean" else series.median()


def _compute_global_summary(timing_df, rq1_best, metric, agg):
    acc_col = f"{agg}_{metric}"

    model_timing = (
        timing_df.groupby("model")[["train_time_s", "pred_time_s"]]
        .median()
        .reset_index()
    )

    acc_df = rq1_best[rq1_best["metric"] == metric]
    model_acc = (
        acc_df.groupby("model")["value"]
        .agg(agg)
        .reset_index()
        .rename(columns={"value": acc_col})
    )

    summary = model_timing.merge(model_acc, on="model", how="inner")
    summary["efficiency"]      = (1.0 / summary[acc_col]) / np.log1p(summary["train_time_s"])
    summary["pred_efficiency"] = (1.0 / summary[acc_col]) / np.log1p(summary["pred_time_s"])
    summary["pareto_efficient"] = _pareto_frontier(summary, "train_time_s", acc_col)
    summary["pareto_pred"]      = _pareto_frontier(summary, "pred_time_s",  acc_col)
    return summary.sort_values("train_time_s").reset_index(drop=True)


def _compute_per_dataset_summary(timing_df, rq1_best, metric, agg):
    acc_col = f"{agg}_{metric}"

    timing_ds = (
        timing_df.groupby(["model", "dataset"])[["train_time_s", "pred_time_s"]]
        .median()
        .reset_index()
    )

    acc_df = rq1_best[rq1_best["metric"] == metric]
    acc_ds = (
        acc_df.groupby(["model", "dataset"])["value"]
        .agg(agg)
        .reset_index()
        .rename(columns={"value": acc_col})
    )

    summary = timing_ds.merge(acc_ds, on=["model", "dataset"], how="inner")
    summary["efficiency"]      = (1.0 / summary[acc_col]) / np.log1p(summary["train_time_s"])
    summary["pred_efficiency"] = (1.0 / summary[acc_col]) / np.log1p(summary["pred_time_s"])

    parts = []
    for ds in summary["dataset"].unique():
        sub = summary[summary["dataset"] == ds].copy()
        sub["pareto_efficient"] = _pareto_frontier(sub, "train_time_s", acc_col)
        sub["pareto_pred"]      = _pareto_frontier(sub, "pred_time_s",  acc_col)
        parts.append(sub)

    return pd.concat(parts).reset_index(drop=True)


def _compute_s1_summary(timing_df, rq1_best, metric, agg):
    return _compute_global_summary(
        _filter_s1(timing_df),
        _filter_s1(rq1_best),
        metric,
        agg,
    )


def _compute_per_dataset_s1_summary(timing_df, rq1_best, metric, agg):
    return _compute_per_dataset_summary(
        _filter_s1(timing_df),
        _filter_s1(rq1_best),
        metric,
        agg,
    )


def _compute_timing_vs_M(rq1_df, config):
    boosted_mask = rq1_df["model"].str.contains("_M") & rq1_df["model"].str.contains("_eta")
    df = rq1_df[boosted_mask].copy()
    if df.empty:
        return

    parsed = df["model"].apply(_parse_model_name)
    df["base_model"] = [p[0] for p in parsed]
    df["M"]          = [p[1] for p in parsed]
    df = df[df["base_model"].notna()].copy()

    timing_rows = df.drop_duplicates(subset=["dataset", "model", "sample_size", "run"])[
        ["base_model", "M", "dataset", "sample_size", "train_time_s"]
    ]

    timing_vs_M = (
        timing_rows.groupby(["base_model", "M", "dataset", "sample_size"])["train_time_s"]
        .median()
        .reset_index()
    )
    timing_vs_M.to_csv(os.path.join(config.RESULTS_DIR, "rq3_timing_vs_M.csv"), index=False)

    acc_rows = df.copy()
    acc_M = (
        acc_rows.groupby(["base_model", "M", "metric"])["value"]
        .agg(["median", "mean"])
        .reset_index()
        .rename(columns={"median": "median_acc", "mean": "mean_acc"})
    )
    acc_M.to_csv(os.path.join(config.RESULTS_DIR, "rq3_accuracy_vs_M.csv"), index=False)


def _timing_from_best(rq1_best):
    return (
        rq1_best
        .drop_duplicates(subset=["model", "dataset", "sample_size", "run"])
        [["model", "dataset", "sample_size", "train_time_s", "pred_time_s"]]
        .dropna(subset=["train_time_s", "pred_time_s"])
        .groupby(["model", "dataset", "sample_size"])[["train_time_s", "pred_time_s"]]
        .median()
        .reset_index()
    )


def analyse_rq3(timing_csv=None, rq1_csv=None, config=cfg):
    if rq1_csv is None:
        rq1_csv = os.path.join(config.RESULTS_DIR, "rq1_raw.csv")
    if not os.path.exists(rq1_csv):
        raise FileNotFoundError(f"RQ1 results not found at {rq1_csv}. Run --rq1 first.")

    rq1_df = pd.read_csv(rq1_csv)
    rq1_df = rq1_df[rq1_df["value"].notna()].copy()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    _compute_timing_vs_M(rq1_df, config)

    all_results = {}
    for metric in ["MRE"]:
        rq1_best = select_best_per_scenario(rq1_df, metric=metric)

        timing_for_metric = _timing_from_best(rq1_best)

        for agg in ["median", "mean"]:
            suffix = f"{metric.lower()}_{agg}"

            global_s = _compute_global_summary(timing_for_metric, rq1_best, metric, agg)
            global_s.to_csv(
                os.path.join(config.RESULTS_DIR, f"rq3_summary_{suffix}.csv"), index=False)

            per_ds = _compute_per_dataset_summary(timing_for_metric, rq1_best, metric, agg)
            per_ds.to_csv(
                os.path.join(config.RESULTS_DIR, f"rq3_summary_per_dataset_{suffix}.csv"), index=False)

            s1 = _compute_s1_summary(timing_for_metric, rq1_best, metric, agg)
            s1.to_csv(
                os.path.join(config.RESULTS_DIR, f"rq3_summary_s1_{suffix}.csv"), index=False)

            per_ds_s1 = _compute_per_dataset_s1_summary(timing_for_metric, rq1_best, metric, agg)
            per_ds_s1.to_csv(
                os.path.join(config.RESULTS_DIR, f"rq3_summary_per_dataset_s1_{suffix}.csv"), index=False)

            all_results[suffix] = global_s
            print(f"\n=== RQ3 Global Summary ({metric}, {agg}) ===")
            print(global_s.to_string(index=False))

    all_results["mre_median"].to_csv(
        os.path.join(config.RESULTS_DIR, "rq3_summary.csv"), index=False)

    return all_results["mre_median"]
