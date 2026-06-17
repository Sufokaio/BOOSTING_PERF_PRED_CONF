# RQ3 figure generation: cost-accuracy scatter plots with Pareto frontier. Output: {FIGURES_DIR}/rq3/.
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg
from output.figures.plot_utils import get_color, save_figure
from analysis.rq3_analysis import _pareto_frontier

_BOOSTED = ["SVRBoost", "RidgeBoost", "MLPBoost", "TreeBoost"]


def _f6_scatter(summary, metric, agg, time_col, budget_label, figures_dir):
    acc_col  = f"{agg}_{metric}"
    is_pred  = time_col == "pred_time_s"
    t_scale  = 1000 if is_pred else 1
    xlabel   = "Median Pred Time (ms, log scale)" if is_pred else "Median Train Time (s, log scale)"
    time_tag = "pred" if is_pred else "train"

    summary = summary.copy()
    summary["_pareto"] = _pareto_frontier(summary, time_col, acc_col)

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in summary.iterrows():
        model  = row["model"]
        t_val  = row[time_col] * t_scale
        acc    = row[acc_col]
        color  = get_color(model)
        pareto = bool(row["_pareto"])
        marker = "*" if pareto else "o"
        sz     = 120 if pareto else 60
        ax.scatter(t_val, acc, color=color, s=sz, marker=marker, zorder=3)
        ax.annotate(model, (t_val, acc), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)

    pareto_df = summary[summary["_pareto"]].sort_values(time_col)
    if len(pareto_df) > 1:
        ax.step(pareto_df[time_col] * t_scale, pareto_df[acc_col],
                where="post", linestyle="--", color="gray", alpha=0.6,
                label="Pareto frontier")
        ax.legend(fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(f"{agg.capitalize()} {metric}", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = f"f6_scatter_{metric.lower()}_{agg}_{time_tag}_{budget_label}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def _f7_pareto_per_dataset(per_ds, metric, agg, time_col, budget_label, figures_dir):
    acc_col  = f"{agg}_{metric}"
    is_pred  = time_col == "pred_time_s"
    t_scale  = 1000 if is_pred else 1
    xlabel   = "Pred Time (ms)" if is_pred else "Train Time (s)"
    time_tag = "pred" if is_pred else "train"

    datasets = sorted(per_ds["dataset"].unique())
    ncols = 4
    nrows = max(1, int(np.ceil(len(datasets) / ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, ds in enumerate(datasets):
        ax     = axes[ax_idx]
        ds_sub = per_ds[per_ds["dataset"] == ds].copy()
        ds_sub["_pareto"] = _pareto_frontier(ds_sub, time_col, acc_col)

        for _, row in ds_sub.iterrows():
            model  = row["model"]
            t_val  = row[time_col] * t_scale
            acc    = row[acc_col]
            color  = get_color(model)
            pareto = bool(row["_pareto"])
            marker = "*" if pareto else "o"
            sz     = 80 if pareto else 40
            ax.scatter(t_val, acc, color=color, s=sz, marker=marker, zorder=3)
            ax.annotate(model, (t_val, acc), textcoords="offset points",
                        xytext=(3, 2), fontsize=5)

        p_sub = ds_sub[ds_sub["_pareto"]].sort_values(time_col)
        if len(p_sub) > 1:
            ax.step(p_sub[time_col] * t_scale, p_sub[acc_col],
                    where="post", linestyle="--", color="gray", alpha=0.5)

        ax.set_xscale("log")
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(metric, fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    fig.tight_layout()

    fname = f"f7_pareto_per_dataset_{metric.lower()}_{agg}_{time_tag}_{budget_label}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def _f8_time_vs_M(timing_M_df, acc_M_df, metric, agg, config, figures_dir):
    M_vals = sorted(config.M_VALUES)
    acc_col = "median_acc" if agg == "median" else "mean_acc"
    acc_metric_df = acc_M_df[acc_M_df["metric"] == metric] if acc_M_df is not None else None

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    bases = [b for b in _BOOSTED if b in timing_M_df["base_model"].unique()]

    for base in bases:
        color = get_color(base)

        t_sub = timing_M_df[timing_M_df["base_model"] == base]
        t_grp = (
            t_sub.groupby("M")["train_time_s"]
            .median()
            .reindex(M_vals)
        )
        ax1.plot(M_vals, t_grp.values, marker="o", color=color,
                 linestyle="-", label=f"{base} (time)", linewidth=1.5)

        if acc_metric_df is not None:
            a_sub = acc_metric_df[acc_metric_df["base_model"] == base]
            a_grp = a_sub.groupby("M")[acc_col].first().reindex(M_vals)
            ax2.plot(M_vals, a_grp.values, marker="s", color=color,
                     linestyle="--", alpha=0.7, linewidth=1)

    ax1.set_xlabel(r"$M$ (boosting rounds)", fontsize=8)
    ax1.set_ylabel("Median Training Time (s)", fontsize=8, color="black")
    ax2.set_ylabel(f"{agg.capitalize()} {metric}", fontsize=8, color="dimgray")
    ax2.tick_params(axis="y", labelcolor="dimgray")

    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)
    fig.tight_layout()

    fname = f"f8_time_vs_M_{metric.lower()}_{agg}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def generate_rq3_figures(config=cfg):
    results_dir = config.RESULTS_DIR
    figures_dir = os.path.join(config.FIGURES_DIR, "rq3")
    os.makedirs(figures_dir, exist_ok=True)

    tvm_path = os.path.join(results_dir, "rq3_timing_vs_M.csv")
    timing_M = pd.read_csv(tvm_path) if os.path.exists(tvm_path) else None

    acc_M_path = os.path.join(results_dir, "rq3_accuracy_vs_M.csv")
    acc_M = pd.read_csv(acc_M_path) if os.path.exists(acc_M_path) else None

    for metric in ["MRE"]:
        for agg in ["median", "mean"]:
            suffix = f"{metric.lower()}_{agg}"
            print(f"\n[RQ3 Figures] {metric} / {agg}")

            if metric == "MRE" and agg == "mean":
                g_path  = os.path.join(results_dir, f"rq3_summary_{suffix}.csv")
                s1_path = os.path.join(results_dir, f"rq3_summary_s1_{suffix}.csv")
                if os.path.exists(g_path):
                    global_s = pd.read_csv(g_path)
                    _f6_scatter(global_s, metric, agg, "pred_time_s", "all", figures_dir)
                    # _f6_scatter(global_s, metric, agg, "train_time_s", "all", figures_dir)
                if os.path.exists(s1_path):
                    s1_s = pd.read_csv(s1_path)
                    _f6_scatter(s1_s, metric, agg, "pred_time_s", "s1", figures_dir)
                    # _f6_scatter(s1_s, metric, agg, "train_time_s", "s1", figures_dir)

            # F7: disabled
            # pd_path    = os.path.join(results_dir, f"rq3_summary_per_dataset_{suffix}.csv")
            # pd_s1_path = os.path.join(results_dir, f"rq3_summary_per_dataset_s1_{suffix}.csv")
            # if os.path.exists(pd_path):
            #     per_ds = pd.read_csv(pd_path)
            #     for time_col in ["train_time_s", "pred_time_s"]:
            #         _f7_pareto_per_dataset(per_ds, metric, agg, time_col, "all", figures_dir)
            # if os.path.exists(pd_s1_path):
            #     per_ds_s1 = pd.read_csv(pd_s1_path)
            #     for time_col in ["train_time_s", "pred_time_s"]:
            #         _f7_pareto_per_dataset(per_ds_s1, metric, agg, time_col, "s1", figures_dir)

            # F8: disabled
            # if timing_M is not None:
            #     _f8_time_vs_M(timing_M, acc_M, metric, agg, config, figures_dir)

    print(f"\n[Figures] RQ3 figures saved to {figures_dir}")
