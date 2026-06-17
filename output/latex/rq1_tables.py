# RQ1 LaTeX table generation. Output: {LATEX_DIR}/rq1/.
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg
from analysis.rq1_analysis import select_best_per_scenario, _filter_s1, _filter_smax

_BASELINE_PAIRS = {
    "SVRBoost":   "SVR",
    "TreeBoost":  "CART",
    "RidgeBoost": "RR",
    "XGBoost":    "CART",
}


def _bold(s):
    return r"\textbf{" + str(s) + r"}"


def _save(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _load_sk(results_dir, sel_metric):
    path = os.path.join(results_dir, f"rq1_sk_ranks_{sel_metric}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run --analyse first.")
    return pd.read_csv(path)


def _load_borda_scenario(results_dir, sel_metric):
    path = os.path.join(results_dir, f"rq1_borda_scenario_{sel_metric}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run --analyse first.")
    return pd.read_csv(path)


def _load_borda_global(results_dir, sel_metric):
    path = os.path.join(results_dir, f"rq1_borda_global_{sel_metric}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run --analyse first.")
    return pd.read_csv(path)


def _sorted_models(df):
    return sorted(df["model"].unique())


def _add_sample_rank(df):
    df = df.copy()
    sizes = (
        df[["dataset", "sample_size"]].drop_duplicates()
        .sort_values(["dataset", "sample_size"])
    )
    sizes["sample_rank"] = sizes.groupby("dataset").cumcount() + 1
    return df.merge(sizes, on=["dataset", "sample_size"])


def _t1_sk_summary(raw_df, sk_df, sel_metric, agg, budget_label, latex_dir, sub_type="mode"):
    metrics = ["MRE", "MAE", "RMSE", "MIBRE"]
    models = _sorted_models(raw_df)

    if budget_label == "s1":
        sk_sub = _filter_s1(sk_df)
        data_sub = _filter_s1(raw_df)
    elif budget_label == "smax":
        sk_sub = _filter_smax(sk_df)
        data_sub = _filter_smax(raw_df)
    else:
        sk_sub = sk_df
        data_sub = raw_df

    rows = []
    for model in models:
        row = {"Model": model}
        for metric in metrics:
            vals = data_sub[(data_sub["model"] == model) & (data_sub["metric"] == metric)]["value"]
            sk_r = sk_sub[
                (sk_sub["model"] == model) & (sk_sub["metric"] == metric)
            ]["sk_rank"]
            if not sk_r.empty:
                if agg == "median" or sub_type == "mode":
                    rank_num = float(int(sk_r.mode()[0]))
                    rank_display = str(int(sk_r.mode()[0]))
                else:
                    rank_num = sk_r.mean()
                    rank_display = f"{rank_num:.2f}"
            else:
                rank_num = 1.0
                rank_display = "1"
            if agg == "median":
                med = vals.median()
                iqr = vals.quantile(0.75) - vals.quantile(0.25)
                row[metric] = (f"{med:.2f} ({iqr:.2f})", rank_num, rank_display)
            else:
                mn  = vals.mean()
                std = vals.std()
                row[metric] = (f"{mn:.2f} ({std:.2f})", rank_num, rank_display)
        rows.append(row)

    best_rank = {m: min(r[m][1] for r in rows) for m in metrics}

    header = "Model & " + " & ".join(metrics) + r" \\"
    lines = [
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule", header, r"\midrule",
    ]
    for row in rows:
        cells = [row["Model"]]
        for metric in metrics:
            val_str, rank_num, rank_display = row[metric]
            cell = f"{val_str}$_{{{rank_display}}}$"
            if abs(rank_num - best_rank[metric]) < 1e-9:
                cell = _bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    sub_suffix = f"_sub_{sub_type}" if agg == "mean" else ""
    fname = f"t1_sk_{agg}_{sel_metric.lower()}_{budget_label}{sub_suffix}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t2_mean_rank(sk_df, sel_metric, budget_label, latex_dir):
    metrics = ["MRE", "MAE", "RMSE", "MIBRE"]

    if budget_label == "s1":
        sk_sub = _filter_s1(sk_df)
    elif budget_label == "smax":
        sk_sub = _filter_smax(sk_df)
    else:
        sk_sub = sk_df

    models = sorted(sk_sub["model"].unique())
    pivot = (
        sk_sub.groupby(["model", "metric"])["sk_rank"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="metric", values="sk_rank")
        .reindex(index=models, columns=metrics)
    )

    best_per_metric = {m: pivot[m].min() for m in metrics}

    lines = [
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Model & " + " & ".join(metrics) + r" \\",
        r"\midrule",
    ]
    for model in models:
        cells = [model]
        for metric in metrics:
            val = pivot.at[model, metric] if metric in pivot.columns else float("nan")
            cell = f"{val:.2f}" if not np.isnan(val) else "--"
            if not np.isnan(val) and abs(val - best_per_metric[metric]) < 1e-9:
                cell = _bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t2_mean_rank_{sel_metric.lower()}_{budget_label}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t3_best_cluster_count(sk_df, sel_metric, latex_dir):
    sk_s1 = _filter_s1(sk_df)
    models = sorted(sk_df["model"].unique())

    sel_all = sk_df[sk_df["metric"] == sel_metric]
    sel_s1  = sk_s1[sk_s1["metric"] == sel_metric]

    total_all = sel_all[["dataset", "sample_size"]].drop_duplicates().shape[0]
    total_s1  = sel_s1[["dataset", "sample_size"]].drop_duplicates().shape[0]

    best_all = sel_all[sel_all["sk_rank"] == 1].groupby("model").size().reindex(models, fill_value=0)
    best_s1  = sel_s1[sel_s1["sk_rank"] == 1].groupby("model").size().reindex(models, fill_value=0)

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & \#Best (all) & \%Best (all) & \#Best (S1) & \%Best (S1) \\",
        r"\midrule",
    ]
    for model in models:
        n_all = int(best_all[model])
        n_s1  = int(best_s1[model])
        pct_all = 100 * n_all / total_all if total_all > 0 else 0
        pct_s1  = 100 * n_s1  / total_s1  if total_s1  > 0 else 0
        lines.append(
            f"{model} & {n_all} & {pct_all:.1f}\\% & {n_s1} & {pct_s1:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t3_best_cluster_{sel_metric.lower()}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t4_borda(borda_scenario_df, borda_global_df, sk_df, sel_metric, budget_label, latex_dir):
    metrics = ["MRE", "MAE", "RMSE", "MIBRE"]

    if budget_label == "s1":
        borda_sub = _filter_s1(borda_scenario_df)
        sk_sub    = _filter_s1(sk_df)
        borda_global = (
            borda_sub.groupby("model")["borda_score"].sum()
            .reset_index().rename(columns={"borda_score": "total_borda_score"})
            .sort_values("total_borda_score", ascending=False)
        )
    else:
        borda_sub    = borda_scenario_df
        sk_sub       = sk_df
        borda_global = borda_global_df.sort_values("total_borda_score", ascending=False)

    models = borda_global["model"].tolist()

    borda_per_metric = {}
    for metric in metrics:
        m_sk = sk_sub[sk_sub["metric"] == metric].copy()
        if m_sk.empty:
            borda_per_metric[metric] = {m: 0 for m in models}
            continue
        K_map = (
            m_sk.groupby(["dataset", "sample_size"])["sk_rank"].max()
            .reset_index().rename(columns={"sk_rank": "K"})
        )
        m_sk = m_sk.merge(K_map, on=["dataset", "sample_size"])
        m_sk["pts"] = m_sk["K"] - m_sk["sk_rank"] + 1
        per_model = m_sk.groupby("model")["pts"].sum().to_dict()
        borda_per_metric[metric] = {m: int(per_model.get(m, 0)) for m in models}

    col_best = {m: max(borda_per_metric[m].get(mdl, 0) for mdl in models) for m in metrics}
    total_best = borda_global["total_borda_score"].max()

    lines = [
        r"\begin{tabular}{l" + "c" * (len(metrics) + 1) + "}",
        r"\toprule",
        "Model & " + " & ".join(metrics) + r" & Total \\",
        r"\midrule",
    ]
    for model in models:
        cells = [model]
        for metric in metrics:
            val = borda_per_metric[metric].get(model, 0)
            cells.append(_bold(val) if val == col_best[metric] else str(val))
        total = int(borda_global[borda_global["model"] == model]["total_borda_score"].values[0])
        cells.append(_bold(total) if total == total_best else str(total))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t4_borda_{sel_metric.lower()}_{budget_label}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t4_borda_per_dataset(borda_scenario_df, sel_metric, latex_dir):
    models = sorted(borda_scenario_df["model"].unique())
    datasets = sorted(borda_scenario_df["dataset"].unique())

    pivot = (
        borda_scenario_df.groupby(["dataset", "model"])["borda_score"].sum()
        .reset_index().pivot(index="model", columns="dataset", values="borda_score")
        .reindex(index=models, columns=datasets)
    )

    lines = [
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\toprule",
        "Model & " + " & ".join(datasets) + r" \\",
        r"\midrule",
    ]
    for model in models:
        cells = [model]
        for ds in datasets:
            val = pivot.at[model, ds] if ds in pivot.columns else 0
            best = pivot[ds].max()
            cell = str(int(val)) if not np.isnan(val) else "0"
            if not np.isnan(val) and val == best:
                cell = _bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t4_borda_{sel_metric.lower()}_per_dataset.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t4_borda_per_samplesize(borda_scenario_df, sel_metric, latex_dir):
    df = _add_sample_rank(borda_scenario_df)
    models = sorted(df["model"].unique())
    ranks = sorted(df["sample_rank"].unique())
    rank_labels = [f"S{r}" for r in ranks]

    pivot = (
        df.groupby(["sample_rank", "model"])["borda_score"].sum()
        .reset_index().pivot(index="model", columns="sample_rank", values="borda_score")
        .reindex(index=models, columns=ranks)
    )

    lines = [
        r"\begin{tabular}{l" + "c" * len(ranks) + "}",
        r"\toprule",
        "Model & " + " & ".join(rank_labels) + r" \\",
        r"\midrule",
    ]
    for model in models:
        cells = [model]
        for r in ranks:
            val = pivot.at[model, r] if r in pivot.columns else 0
            best = pivot[r].max()
            cell = str(int(val)) if not np.isnan(val) else "0"
            if not np.isnan(val) and val == best:
                cell = _bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t4_borda_{sel_metric.lower()}_per_samplesize.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t5_per_dataset(raw_df, sk_df, sel_metric, display_metric, agg, budget_label, latex_dir):
    if budget_label == "s1":
        data = _filter_s1(raw_df)
        sk   = _filter_s1(sk_df)
    else:
        data = _filter_smax(raw_df)
        sk   = _filter_smax(sk_df)

    metric_data = data[data["metric"] == display_metric]
    models   = _sorted_models(raw_df)
    datasets = sorted(raw_df["dataset"].unique())

    lines = [
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        "Dataset & " + " & ".join(models) + r" \\",
        r"\midrule",
    ]
    for dataset in datasets:
        sub = metric_data[metric_data["dataset"] == dataset]
        cells = [dataset]
        vals = {}
        for model in models:
            s = sub[sub["model"] == model]["value"]
            vals[model] = (s.mean() if agg == "mean" else s.median()) if not s.empty else float("nan")

        valid = [v for v in vals.values() if not np.isnan(v)]
        best_val = min(valid) if valid else float("nan")

        for model in models:
            med = vals[model]
            sk_sub = sk[
                (sk["dataset"] == dataset) &
                (sk["metric"] == display_metric) &
                (sk["model"] == model)
            ]["sk_rank"]
            rank = int(sk_sub.mode()[0]) if not sk_sub.empty else 1
            cell = f"{med:.2f}$_{{{rank}}}$" if not np.isnan(med) else "--"
            if not np.isnan(med) and abs(med - best_val) < 1e-9:
                cell = _bold(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t5_per_dataset_{sel_metric.lower()}_{display_metric.lower()}_{agg}_{budget_label}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t6_mega(raw_df, sel_metric, display_metric, agg, latex_dir):
    data   = raw_df[raw_df["metric"] == display_metric]
    df_r   = _add_sample_rank(data)
    models = _sorted_models(raw_df)

    datasets = sorted(raw_df["dataset"].unique())
    lines = [
        r"\begin{tabular}{ll" + "c" * len(models) + "}",
        r"\toprule",
        "Dataset & Budget & " + " & ".join(models) + r" \\",
        r"\midrule",
    ]

    for dataset in datasets:
        ds_sub = df_r[df_r["dataset"] == dataset].copy()
        ranks  = sorted(ds_sub["sample_rank"].unique())
        n_rows = len(ranks)

        for i, rank in enumerate(ranks):
            rank_sub = ds_sub[ds_sub["sample_rank"] == rank]
            cells = []
            if i == 0:
                cells.append(r"\multirow{" + str(n_rows) + r"}{*}{" + dataset + "}")
            else:
                cells.append("")
            cells.append(f"S{rank}")

            vals = {}
            for model in models:
                s = rank_sub[rank_sub["model"] == model]["value"]
                vals[model] = (s.mean() if agg == "mean" else s.median()) if not s.empty else float("nan")

            valid = [v for v in vals.values() if not np.isnan(v)]
            best_val = min(valid) if valid else float("nan")

            for model in models:
                v = vals[model]
                cell = f"{v:.2f}" if not np.isnan(v) else "--"
                if not np.isnan(v) and abs(v - best_val) < 1e-9:
                    cell = _bold(cell)
                cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    fname = f"t6_mega_{display_metric.lower()}_{agg}_{sel_metric.lower()}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t7_relative_improvement(raw_df, display_metric, agg, sel_metric, latex_dir):
    data = raw_df[raw_df["metric"] == display_metric]
    fn   = "mean" if agg == "mean" else "median"
    datasets = sorted(raw_df["dataset"].unique())

    col_spec = "l" + "c" * (2 + len(datasets))
    ds_header = " & ".join(datasets)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"Pair & {agg.capitalize()} (all) & S1 & {ds_header}" + r" \\",
        r"\midrule",
    ]

    for boosted, baseline in _BASELINE_PAIRS.items():
        b_data = data[data["model"] == boosted].set_index(
            ["dataset", "sample_size", "run"])["value"]
        s_data = data[data["model"] == baseline].set_index(
            ["dataset", "sample_size", "run"])["value"]

        common = b_data.index.intersection(s_data.index)
        if common.empty:
            lines.append(f"{boosted} vs {baseline}" + " & --" * (2 + len(datasets)) + r" \\")
            continue

        rel = ((s_data[common] - b_data[common]) / s_data[common].abs() * 100)
        rel_df = rel.reset_index()
        rel_df.columns = ["dataset", "sample_size", "run", "rel_imp"]

        overall = getattr(rel_df["rel_imp"], fn)()
        s1_mask = rel_df["sample_size"] == rel_df.groupby("dataset")["sample_size"].transform("min")
        s1_val  = getattr(rel_df.loc[s1_mask, "rel_imp"], fn)()
        per_ds  = rel_df.groupby("dataset")["rel_imp"].agg(fn)

        cells = [f"{boosted} vs {baseline}", f"{overall:+.1f}\\%", f"{s1_val:+.1f}\\%"]
        for ds in datasets:
            v = per_ds[ds] if ds in per_ds.index else float("nan")
            cells.append(f"{v:+.1f}\\%" if not np.isnan(v) else "--")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t7_rel_improvement_{display_metric.lower()}_{agg}_{sel_metric.lower()}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def generate_rq1_tables(config=cfg):
    results_dir = config.RESULTS_DIR
    latex_dir   = os.path.join(config.LATEX_DIR, "rq1")
    os.makedirs(latex_dir, exist_ok=True)

    raw_df_orig = pd.read_csv(os.path.join(results_dir, "rq1_raw.csv"))
    raw_df_orig = raw_df_orig[raw_df_orig["value"].notna()].copy()

    for sel_metric in ["MRE"]:
        print(f"\n[RQ1 Tables] Generating tables for selection_metric={sel_metric}")
        raw_df   = select_best_per_scenario(raw_df_orig, metric=sel_metric)
        sk_df    = _load_sk(results_dir, sel_metric)
        # borda_sc = _load_borda_scenario(results_dir, sel_metric)
        # borda_gl = _load_borda_global(results_dir, sel_metric)

        for budget in ["all", "s1"]:
            _t1_sk_summary(raw_df, sk_df, sel_metric, "mean", budget, latex_dir, sub_type="mrank")

        # for budget in ["all", "s1"]:
        #     _t1_sk_summary(raw_df, sk_df, sel_metric, "median", budget, latex_dir)
        #     _t1_sk_summary(raw_df, sk_df, sel_metric, "mean", budget, latex_dir, sub_type="mode")
        #
        # for budget in ["all", "s1"]:
        #     _t2_mean_rank(sk_df, sel_metric, budget, latex_dir)
        #
        # _t3_best_cluster_count(sk_df, sel_metric, latex_dir)
        #
        # for budget in ["all", "s1"]:
        #     _t4_borda(borda_sc, borda_gl, sk_df, sel_metric, budget, latex_dir)
        # _t4_borda_per_dataset(borda_sc, sel_metric, latex_dir)
        # _t4_borda_per_samplesize(borda_sc, sel_metric, latex_dir)
        #
        # for display_metric in ["MRE", "MAE"]:
        #     for agg in ["median", "mean"]:
        #         for budget in ["s1", "smax"]:
        #             _t5_per_dataset(raw_df, sk_df, sel_metric, display_metric, agg, budget, latex_dir)
        #
        # for display_metric in ["MRE", "MAE"]:
        #     for agg in ["median", "mean"]:
        #         _t6_mega(raw_df, sel_metric, display_metric, agg, latex_dir)
        #
        # for display_metric in ["MRE", "MAE"]:
        #     for agg in ["median", "mean"]:
        #         _t7_relative_improvement(raw_df, display_metric, agg, sel_metric, latex_dir)

    print(f"\n[RQ1 Tables] All tables written to {latex_dir}")
