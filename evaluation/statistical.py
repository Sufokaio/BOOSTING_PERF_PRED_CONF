# Scott-Knott + Borda Count statistical analysis over per-scenario result DataFrames.
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sk_implementation import scott_knott


def run_scott_knott(results_dict: dict) -> dict:
    groups = [(name, list(values)) for name, values in results_dict.items()]
    sk_result = scott_knott(groups, a12_threshold=0.60, conf=0.01, seed=42)
    return {name: rank for rank, name, _med, _iqr, _vals in sk_result}


def borda_count(sk_ranks_per_metric: list) -> list:
    if not sk_ranks_per_metric:
        return []

    all_models = set()
    for d in sk_ranks_per_metric:
        all_models.update(d.keys())

    scores = {m: 0 for m in all_models}

    for rank_dict in sk_ranks_per_metric:
        if not rank_dict:
            continue
        K = max(rank_dict.values())
        for model, rank in rank_dict.items():
            scores[model] += (K - rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def run_full_statistical_analysis(raw_results_df, metrics=None):
    if metrics is None:
        metrics = ["MRE", "MAE", "RMSE", "MIBRE"]

    sk_rows = []
    borda_scenario_rows = []

    datasets    = raw_results_df["dataset"].unique()
    sample_sizes = raw_results_df["sample_size"].unique()

    for dataset in datasets:
        for size in sample_sizes:
            sub = raw_results_df[
                (raw_results_df["dataset"] == dataset) &
                (raw_results_df["sample_size"] == size)
            ]
            if sub.empty:
                continue

            sk_per_metric = []
            for metric in metrics:
                metric_sub = sub[sub["metric"] == metric]
                if metric_sub.empty:
                    continue
                results_dict = {}
                for model, grp in metric_sub.groupby("model"):
                    results_dict[model] = grp["value"].tolist()

                if not results_dict:
                    continue
                try:
                    ranks = run_scott_knott(results_dict)
                except Exception:
                    ranks = {m: 1 for m in results_dict}

                sk_per_metric.append(ranks)
                for model, rank in ranks.items():
                    sk_rows.append({
                        "dataset":     dataset,
                        "sample_size": size,
                        "metric":      metric,
                        "model":       model,
                        "sk_rank":     rank,
                    })

            borda_scores = borda_count(sk_per_metric)
            for model, score in borda_scores:
                borda_scenario_rows.append({
                    "dataset":     dataset,
                    "sample_size": size,
                    "model":       model,
                    "borda_score": score,
                })

    sk_ranks_df = pd.DataFrame(sk_rows)
    borda_per_scenario_df = pd.DataFrame(borda_scenario_rows)

    if borda_per_scenario_df.empty:
        borda_global_df = pd.DataFrame(columns=["model", "total_borda_score"])
    else:
        borda_global_df = (
            borda_per_scenario_df
            .groupby("model")["borda_score"]
            .sum()
            .reset_index()
            .rename(columns={"borda_score": "total_borda_score"})
            .sort_values("total_borda_score", ascending=False)
            .reset_index(drop=True)
        )

    return sk_ranks_df, borda_per_scenario_df, borda_global_df
