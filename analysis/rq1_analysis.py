# RQ1 analysis: model comparison per scenario, run SK + Borda.
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from evaluation.statistical import run_full_statistical_analysis

_RENAME = {"RidgeLRBoost": "RidgeBoost", "SVRBaseline": "SVR", "RidgeLRBaseline": "RR"}


def _filter_s1(df):
    mins = df.groupby("dataset")["sample_size"].min().reset_index()
    mins.columns = ["dataset", "min_size"]
    m = df.merge(mins, on="dataset")
    return m[m["sample_size"] == m["min_size"]].drop(columns="min_size")


def _filter_smax(df):
    maxs = df.groupby("dataset")["sample_size"].max().reset_index()
    maxs.columns = ["dataset", "max_size"]
    m = df.merge(maxs, on="dataset")
    return m[m["sample_size"] == m["max_size"]].drop(columns="max_size")


def select_best_per_scenario(df, metric="MRE"):
    df = df.copy()
    df["model_family"] = df["model"].str.split("_").str[0]
    df["model_family"] = df["model_family"].map(lambda x: _RENAME.get(x, x))

    metric_df = df[df["metric"] == metric].copy()
    median_by_combo = (
        metric_df.groupby(["model_family", "dataset", "sample_size", "model"])["value"]
        .median()
        .reset_index()
        .rename(columns={"value": f"median_{metric}"})
    )
    idx = median_by_combo.groupby(
        ["model_family", "dataset", "sample_size"]
    )[f"median_{metric}"].idxmin()
    best_map = (
        median_by_combo.loc[idx][["model_family", "dataset", "sample_size", "model"]]
        .rename(columns={"model": "best_config"})
    )

    merged = df.merge(best_map, on=["model_family", "dataset", "sample_size"])
    out = merged[merged["model"] == merged["best_config"]].copy()
    out["model"] = out["model_family"]
    return out.drop(columns=["model_family", "best_config"])


def analyse_rq1(results_csv=None, config=cfg, selection_metrics=None):
    if results_csv is None:
        results_csv = os.path.join(config.RESULTS_DIR, "rq1_raw.csv")
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"RQ1 results not found at {results_csv}.")
    if selection_metrics is None:
        selection_metrics = ["MRE"]

    df = pd.read_csv(results_csv)
    nan_mask = df["value"].isna()
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"[RQ1 Analysis] Warning: {n_nan} NaN metric rows excluded.")
    df = df[~nan_mask].copy()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    all_results = {}
    for sel_metric in selection_metrics:
        print(f"\n[RQ1 Analysis] Running SK + Borda with selection by {sel_metric} …")
        df_sel = select_best_per_scenario(df, metric=sel_metric)
        sk_ranks_df, borda_per_scenario_df, borda_global_df = run_full_statistical_analysis(df_sel)

        suffix = f"_{sel_metric}"
        sk_ranks_df.to_csv(
            os.path.join(config.RESULTS_DIR, f"rq1_sk_ranks{suffix}.csv"), index=False)
        borda_per_scenario_df.to_csv(
            os.path.join(config.RESULTS_DIR, f"rq1_borda_scenario{suffix}.csv"), index=False)
        borda_global_df.to_csv(
            os.path.join(config.RESULTS_DIR, f"rq1_borda_global{suffix}.csv"), index=False)

        if sel_metric == "MRE":
            sk_ranks_df.to_csv(
                os.path.join(config.RESULTS_DIR, "rq1_sk_ranks.csv"), index=False)
            borda_per_scenario_df.to_csv(
                os.path.join(config.RESULTS_DIR, "rq1_borda_scenario.csv"), index=False)
            borda_global_df.to_csv(
                os.path.join(config.RESULTS_DIR, "rq1_borda_global.csv"), index=False)

        all_results[sel_metric] = (sk_ranks_df, borda_per_scenario_df, borda_global_df)

        print(f"=== RQ1 Global Borda Ranking (selected by {sel_metric}) ===")
        print(borda_global_df.to_string(index=False))

    mre_triple = all_results.get("MRE", (None, None, None))
    return mre_triple
