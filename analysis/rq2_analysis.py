# RQ2 analysis: M×eta sensitivity grid for boosted models.
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

_RENAME = {"RidgeLRBoost": "RidgeBoost"}


def _parse_model_name(model_name):
    parts = model_name.rsplit("_M", 1)
    if len(parts) != 2:
        return None, None, None
    base = parts[0]
    rest = parts[1].split("_eta")
    if len(rest) != 2:
        return None, None, None
    try:
        return base, int(rest[0]), float(rest[1])
    except ValueError:
        return None, None, None


def analyse_rq2(results_csv=None, config=cfg):
    if results_csv is None:
        results_csv = os.path.join(config.RESULTS_DIR, "rq1_raw.csv")
    if not os.path.exists(results_csv):
        raise FileNotFoundError(
            f"RQ1 results not found at {results_csv}. Run --rq1 first."
        )

    df = pd.read_csv(results_csv)
    df = df[df["value"].notna()].copy()

    boosted_mask = df["model"].str.contains("_M") & df["model"].str.contains("_eta")
    df = df[boosted_mask].copy()

    if df.empty:
        raise ValueError(
            "No boosted-model rows found in rq1_raw.csv. "
            "Ensure RQ1 has been run with the full M×eta grid."
        )

    parsed = df["model"].apply(_parse_model_name)
    df["base_model"] = [p[0] for p in parsed]
    df["M"]          = [p[1] for p in parsed]
    df["eta"]        = [p[2] for p in parsed]
    df["base_model"] = df["base_model"].map(lambda x: _RENAME.get(x, x))
    df = df[df["base_model"].notna()].copy()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    all_results = {}
    for metric in ["MRE"]:
        metric_df = df[df["metric"] == metric]
        col = f"median_{metric}"

        median_df = (
            metric_df.groupby(["base_model", "dataset", "sample_size", "M", "eta"])["value"]
            .median()
            .reset_index()
            .rename(columns={"value": col})
        )

        median_df["rank"] = median_df.groupby(
            ["base_model", "dataset", "sample_size"]
        )[col].rank()

        mean_rank = (
            median_df.groupby(["base_model", "M", "eta"])["rank"]
            .mean()
            .reset_index()
            .rename(columns={"rank": "mean_rank"})
        )

        best_configs = (
            mean_rank.loc[mean_rank.groupby("base_model")["mean_rank"].idxmin()]
            .reset_index(drop=True)
        )

        suffix = metric.lower()
        median_df.to_csv(
            os.path.join(config.RESULTS_DIR, f"rq2_median_{suffix}.csv"), index=False)
        mean_rank.to_csv(
            os.path.join(config.RESULTS_DIR, f"rq2_mean_rank_{suffix}.csv"), index=False)
        best_configs.to_csv(
            os.path.join(config.RESULTS_DIR, f"rq2_best_configs_{suffix}.csv"), index=False)

        all_results[metric] = (median_df, mean_rank, best_configs)

        print(f"\n=== RQ2 Best (M, eta) per boosted model (by {metric}) ===")
        print(best_configs.to_string(index=False))

    mre_median, mre_mean_rank, mre_best = all_results["MRE"]
    mre_mean_rank.to_csv(os.path.join(config.RESULTS_DIR, "rq2_mean_rank.csv"), index=False)
    mre_best.to_csv(os.path.join(config.RESULTS_DIR, "rq2_best_configs.csv"), index=False)

    return all_results["MRE"]
