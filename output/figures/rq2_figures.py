# RQ2 figure generation: M×eta sensitivity heatmaps (f5). Output: {FIGURES_DIR}/rq2/.
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
from analysis.rq1_analysis import _filter_s1

_BOOSTED = ["SVRBoost", "RidgeBoost", "MLPBoost", "TreeBoost"]

_F3F4_COLORS = {
    "SVRBoost":   "#0072B2",
    "RidgeBoost": "#E69F00",
    "MLPBoost":   "#CC79A7",
    "TreeBoost":  "#009E73",
}


def _band(series, fn, agg):
    if agg == "mean":
        center = series.agg(fn)
        std    = series.std()
        return center, center - std, center + std
    else:
        center = series.agg(fn)
        iqr    = series.quantile(0.75) - series.quantile(0.25)
        return center, center - iqr / 2, center + iqr / 2


def _f3_perf_vs_M(median_df, metric, agg, config, figures_dir):
    col   = f"median_{metric}"
    fn    = "mean" if agg == "mean" else "median"

    fig, ax = plt.subplots(figsize=(6, 4))
    for base in _BOOSTED:
        sub = median_df[
            (median_df["base_model"] == base) &
            (median_df["eta"] == config.DEFAULT_ETA)
        ]
        if sub.empty:
            continue
        M_vals = sorted(config.M_VALUES)
        centers, lowers, uppers = [], [], []
        for M in M_vals:
            s = sub[sub["M"] == M][col]
            c, lo, hi = _band(s, fn, agg)
            centers.append(c); lowers.append(lo); uppers.append(hi)

        color = _F3F4_COLORS.get(base, "#7f7f7f")
        ax.plot(M_vals, centers, marker="o", label=base, color=color)
        ax.fill_between(M_vals, lowers, uppers, alpha=0.2, color=color)

    ax.set_xlabel(r"$M$ (boosting rounds)", fontsize=8)
    ax.set_ylabel(f"{agg.capitalize()} {metric}", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = f"f3_perf_vs_M_{metric.lower()}_{agg}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def _f4_perf_vs_eta(median_df, metric, agg, config, figures_dir):
    col   = f"median_{metric}"
    fn    = "mean" if agg == "mean" else "median"

    fig, ax = plt.subplots(figsize=(6, 4))
    for base in _BOOSTED:
        sub = median_df[
            (median_df["base_model"] == base) &
            (median_df["M"] == config.DEFAULT_M)
        ]
        if sub.empty:
            continue
        eta_vals = sorted(config.ETA_VALUES)
        centers, lowers, uppers = [], [], []
        for eta in eta_vals:
            s = sub[sub["eta"] == eta][col]
            c, lo, hi = _band(s, fn, agg)
            centers.append(c); lowers.append(lo); uppers.append(hi)

        color = _F3F4_COLORS.get(base, "#7f7f7f")
        ax.plot(eta_vals, centers, marker="s", label=base, color=color)
        ax.fill_between(eta_vals, lowers, uppers, alpha=0.2, color=color)

    ax.set_xlabel(r"$\eta$ (learning rate)", fontsize=8)
    ax.set_ylabel(f"{agg.capitalize()} {metric}", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = f"f4_perf_vs_eta_{metric.lower()}_{agg}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def _f5_heatmap(median_df, metric, agg, budget_label, config, figures_dir):
    col = f"median_{metric}"
    fn  = "mean" if agg == "mean" else "median"

    if budget_label == "s1":
        data = _filter_s1(median_df)
        title_budget = "S1 budget"
    else:
        data = median_df
        title_budget = "all budgets"

    M_vals   = sorted(config.M_VALUES)
    eta_vals = sorted(config.ETA_VALUES)

    bases = [b for b in _BOOSTED if b in data["base_model"].unique()]
    ncols = min(2, len(bases))
    nrows = int(np.ceil(len(bases) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, base in enumerate(bases):
        ax  = axes[idx]
        sub = data[data["base_model"] == base]
        grid = sub.groupby(["M", "eta"])[col].agg(fn).reset_index()
        pivot = grid.pivot(index="M", columns="eta", values=col)
        pivot = pivot.reindex(index=M_vals, columns=eta_vals)

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=f"{agg.capitalize()} {metric}")
        ax.set_xticks(range(len(eta_vals)))
        ax.set_xticklabels([str(e) for e in eta_vals], fontsize=8)
        ax.set_yticks(range(len(M_vals)))
        ax.set_yticklabels([str(m) for m in M_vals], fontsize=8)
        ax.set_xlabel(r"$\eta$", fontsize=8)
        ax.set_ylabel(r"$M$", fontsize=8)
        ax.set_title(base, fontsize=9)

        for i in range(len(M_vals)):
            for j in range(len(eta_vals)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    for ax in axes[len(bases):]:
        ax.set_visible(False)

    fig.tight_layout()

    fname = f"f5_heatmap_{metric.lower()}_{agg}_{budget_label}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def generate_rq2_figures(config=cfg):
    results_dir = config.RESULTS_DIR
    figures_dir = os.path.join(config.FIGURES_DIR, "rq2")
    os.makedirs(figures_dir, exist_ok=True)

    for metric in ["MRE"]:
        suffix = metric.lower()
        path = os.path.join(results_dir, f"rq2_median_{suffix}.csv")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found — run --analyse first")
            continue
        median_df = pd.read_csv(path)

        # for agg in ["median", "mean"]:
        #     print(f"\n[RQ2 Figures] {metric} / {agg}")
        #     _f3_perf_vs_M(median_df, metric, agg, config, figures_dir)
        #     _f4_perf_vs_eta(median_df, metric, agg, config, figures_dir)
        #     for budget in ["all", "s1"]:
        #         _f5_heatmap(median_df, metric, agg, budget, config, figures_dir)

        print(f"\n[RQ2 Figures] {metric} / mean / all")
        _f5_heatmap(median_df, metric, "mean", "all", config, figures_dir)

    print(f"\n[Figures] RQ2 figures saved to {figures_dir}")
