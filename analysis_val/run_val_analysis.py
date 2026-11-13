# Selection of the per-scenario variant on validation scores.
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation.statistical as st
from analysis.rq1_analysis import _RENAME, _filter_s1
from analysis.rq2_analysis import analyse_rq2

KEY = ["dataset", "model", "sample_size", "run", "metric"]
METRICS = ["MRE", "MAE", "RMSE", "MIBRE"]
SEL_METRIC = "MRE"

_sk_failures = []
_run_scott_knott = st.run_scott_knott


def _checked_scott_knott(results_dict):
    try:
        return _run_scott_knott(results_dict)
    except Exception as e:
        _sk_failures.append(repr(e))
        raise


st.run_scott_knott = _checked_scott_knott


def _family(models):
    return models.str.split("_").str[0].map(lambda x: _RENAME.get(x, x))


def chosen_variants(df, stat="median", metric=SEL_METRIC):
    df = df[df["metric"] == metric].assign(model_family=lambda d: _family(d["model"]))
    agg = df.groupby(["model_family", "dataset", "sample_size", "model"])["value"].agg(stat).reset_index()
    idx = agg.groupby(["model_family", "dataset", "sample_size"])["value"].idxmin()
    best = agg.loc[idx].rename(columns={"model": "chosen", "value": "score"})
    both = agg.merge(best[["model_family", "dataset", "sample_size", "score"]],
                     on=["model_family", "dataset", "sample_size"])
    ties = (both["value"] == both["score"]).groupby(
        [both["model_family"], both["dataset"], both["sample_size"]]).sum()
    best = best.merge(ties.rename("n_tied").reset_index(), on=["model_family", "dataset", "sample_size"])
    return best.reset_index(drop=True)


def keep_chosen(test_df, chosen):
    d = test_df.assign(model_family=_family(test_df["model"]))
    d = d.merge(chosen[["model_family", "dataset", "sample_size", "chosen"]],
                on=["model_family", "dataset", "sample_size"], how="left")
    for fam in d[d["chosen"].isna()]["model_family"].unique():
        if d[d["model_family"] == fam]["model"].nunique() != 1:
            raise RuntimeError(f"{fam} has no validation scores and more than one variant")
    d["chosen"] = d["chosen"].fillna(d["model"])
    return d[d["model"] == d["chosen"]].drop(columns=["model_family", "chosen"]).reset_index(drop=True)


def _budget(df, budget):
    return _filter_s1(df) if budget == "s1" else df


def summary_table(sel, sk, budget, report):
    data, ranks = _budget(sel, budget), _budget(sk, budget)
    rows = []
    for model in sorted(data["model"].unique()):
        row = {"model": model}
        for metric in METRICS:
            v = data[(data["model"] == model) & (data["metric"] == metric)]["value"]
            if report == "mean":
                center, spread = v.mean(), v.std()
            else:
                center, spread = v.median(), v.quantile(0.75) - v.quantile(0.25)
            row[metric] = center
            row[metric + "_spread"] = spread
            row[metric + "_rank"] = ranks[(ranks["model"] == model) & (ranks["metric"] == metric)]["sk_rank"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def rq2_medians(df, results_dir, split):
    path = os.path.join(results_dir, f"rq2_input_{split}.csv")
    df[df["model"].str.contains("_M") & df["model"].str.contains("_eta")][KEY + ["value"]].to_csv(path, index=False)
    c = types.SimpleNamespace(RESULTS_DIR=results_dir)
    analyse_rq2(results_csv=path, config=c)
    os.remove(path)
    median_df = pd.read_csv(os.path.join(results_dir, "rq2_median_mre.csv"))
    for f in ("rq2_median_mre.csv", "rq2_mean_rank_mre.csv", "rq2_best_configs_mre.csv",
              "rq2_mean_rank.csv", "rq2_best_configs.csv"):
        os.remove(os.path.join(results_dir, f))
    median_df.to_csv(os.path.join(results_dir, f"rq2_median_mre_{split}.csv"), index=False)
    return median_df
