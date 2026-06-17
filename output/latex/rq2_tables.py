# RQ2 LaTeX table generation. Output: {LATEX_DIR}/rq2/.
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg
from analysis.rq1_analysis import _filter_s1

_BOOSTED = ["SVRBoost", "RidgeBoost", "MLPBoost", "TreeBoost"]


def _bold(s):
    return r"\textbf{" + str(s) + r"}"


def _save(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _add_sample_rank(df):
    df = df.copy()
    sizes = (
        df[["dataset", "sample_size"]].drop_duplicates()
        .sort_values(["dataset", "sample_size"])
    )
    sizes["sample_rank"] = sizes.groupby("dataset").cumcount() + 1
    return df.merge(sizes, on=["dataset", "sample_size"])


def _t4_sensitivity(median_df, metric, agg, base_models, M_vals, eta_vals, latex_dir):
    col = f"median_{metric}"
    for base in base_models:
        sub = median_df[median_df["base_model"] == base]
        if sub.empty:
            continue
        fn = "mean" if agg == "mean" else "median"
        grid = sub.groupby(["M", "eta"])[col].agg(fn).reset_index()
        pivot = grid.pivot(index="M", columns="eta", values=col)
        pivot = pivot.reindex(index=M_vals, columns=eta_vals)

        best_val = np.nanmin(pivot.values)
        eta_strs = [str(e) for e in eta_vals]

        lines = [
            f"% {base} -- M x eta sensitivity ({metric}, {agg})",
            r"\begin{tabular}{l" + "c" * len(eta_vals) + "}",
            r"\toprule",
            r"$M$ \textbackslash $\eta$ & " + " & ".join(eta_strs) + r" \\",
            r"\midrule",
        ]
        for M in M_vals:
            cells = [str(M)]
            for eta in eta_vals:
                val = pivot.at[M, eta] if (M in pivot.index and eta in pivot.columns) else float("nan")
                cell = f"{val:.2f}" if not np.isnan(val) else "--"
                if not np.isnan(val) and abs(val - best_val) < 1e-9:
                    cell = _bold(cell)
                cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]

        fname = f"t4_{base.lower()}_{metric.lower()}_{agg}.tex"
        _save(lines, os.path.join(latex_dir, fname))
        print(f"  wrote {fname}")


def _t_bestcfg_per_dataset(median_df, metric, agg, base_models, latex_dir):
    col = f"median_{metric}"
    fn  = "mean" if agg == "mean" else "median"
    datasets = sorted(median_df["dataset"].unique())

    lines = [
        r"\begin{tabular}{l" + "c" * len(base_models) + "}",
        r"\toprule",
        "Dataset & " + " & ".join(base_models) + r" \\",
        r"\midrule",
    ]
    for ds in datasets:
        cells = [ds]
        for base in base_models:
            sub = median_df[
                (median_df["base_model"] == base) &
                (median_df["dataset"] == ds)
            ]
            if sub.empty:
                cells.append("--")
                continue
            agg_df = sub.groupby(["M", "eta"])[col].agg(fn).reset_index()
            best_row = agg_df.loc[agg_df[col].idxmin()]
            best_M   = int(best_row["M"])
            best_eta = float(best_row["eta"])
            best_val = float(best_row[col])
            cells.append(f"({best_M}, {best_eta}) [{best_val:.2f}]")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t_bestcfg_per_dataset_{metric.lower()}_{agg}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t_bestcfg_per_ssize(median_df, metric, agg, base_models, latex_dir):
    col = f"median_{metric}"
    fn  = "mean" if agg == "mean" else "median"

    df = _add_sample_rank(median_df)
    ranks = sorted(df["sample_rank"].unique())

    lines = [
        r"\begin{tabular}{l" + "c" * len(base_models) + "}",
        r"\toprule",
        "Budget & " + " & ".join(base_models) + r" \\",
        r"\midrule",
    ]
    for rank in ranks:
        cells = [f"S{rank}"]
        for base in base_models:
            sub = df[
                (df["base_model"] == base) &
                (df["sample_rank"] == rank)
            ]
            if sub.empty:
                cells.append("--")
                continue
            agg_df = sub.groupby(["M", "eta"])[col].agg(fn).reset_index()
            best_row = agg_df.loc[agg_df[col].idxmin()]
            best_M   = int(best_row["M"])
            best_eta = float(best_row["eta"])
            best_val = float(best_row[col])
            cells.append(f"({best_M}, {best_eta}) [{best_val:.2f}]")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t_bestcfg_per_ssize_{metric.lower()}_{agg}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t_heatmap_per_dataset(median_df, metric, agg, base_models, M_vals, eta_vals, latex_dir):
    col = f"median_{metric}"
    fn  = "mean" if agg == "mean" else "median"
    datasets = sorted(median_df["dataset"].unique())

    for ds in datasets:
        ds_sub = median_df[median_df["dataset"] == ds]
        if ds_sub.empty:
            continue

        M_strs = [f"$M={M}$" for M in M_vals]
        lines = [
            f"% Dataset: {ds} -- M x eta heatmap ({metric}, {agg})",
            r"\begin{tabular}{ll" + "c" * len(M_vals) + "}",
            r"\toprule",
            r"Model & $\eta$ & " + " & ".join(M_strs) + r" \\",
            r"\midrule",
        ]

        for b_idx, base in enumerate(base_models):
            if b_idx > 0:
                lines.append(r"\midrule")
            b_sub = ds_sub[ds_sub["base_model"] == base]
            if b_sub.empty:
                lines.append(base + r" & \multicolumn{" + str(len(M_vals) + 1) + r"}{c}{--} \\")
                continue

            agg_df = b_sub.groupby(["M", "eta"])[col].agg(fn).reset_index()
            pivot  = agg_df.pivot(index="eta", columns="M", values=col)
            pivot  = pivot.reindex(index=eta_vals, columns=M_vals)
            best_val = np.nanmin(pivot.values)

            for e_idx, eta in enumerate(eta_vals):
                row_label_model = base if e_idx == 0 else ""
                cells = [row_label_model, str(eta)]
                for M in M_vals:
                    val = pivot.at[eta, M] if (eta in pivot.index and M in pivot.columns) else float("nan")
                    cell = f"{val:.2f}" if not np.isnan(val) else "--"
                    if not np.isnan(val) and abs(val - best_val) < 1e-9:
                        cell = _bold(cell)
                    cells.append(cell)
                lines.append(" & ".join(cells) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}"]

        safe_ds = ds.replace(" ", "_").replace("/", "-").lower()
        fname   = f"t_heatmap_{safe_ds}_{metric.lower()}_{agg}.tex"
        _save(lines, os.path.join(latex_dir, fname))
        print(f"  wrote {fname}")


def generate_rq2_tables(config=cfg):
    results_dir = config.RESULTS_DIR
    latex_dir   = os.path.join(config.LATEX_DIR, "rq2")
    os.makedirs(latex_dir, exist_ok=True)

    M_vals   = sorted(config.M_VALUES)
    eta_vals = sorted(config.ETA_VALUES)

    for metric in ["MRE"]:
        suffix = metric.lower()
        path = os.path.join(results_dir, f"rq2_median_{suffix}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found — run --analyse first")
            continue
        median_df = pd.read_csv(path)
        base_models = [b for b in _BOOSTED if b in median_df["base_model"].unique()]

        # for agg in ["median", "mean"]:
        #     print(f"\n[RQ2 Tables] {metric} / {agg}")
        #     _t4_sensitivity(median_df, metric, agg, base_models, M_vals, eta_vals, latex_dir)
        #     _t_bestcfg_per_dataset(median_df, metric, agg, base_models, latex_dir)
        #     _t_bestcfg_per_ssize(median_df, metric, agg, base_models, latex_dir)
        #     _t_heatmap_per_dataset(median_df, metric, agg, base_models, M_vals, eta_vals, latex_dir)

    print(f"\n[LaTeX] RQ2 tables written to {latex_dir}")
