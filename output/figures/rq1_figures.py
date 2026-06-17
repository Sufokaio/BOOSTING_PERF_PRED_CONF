# RQ1 figure generation: box-plot grids, Borda bar chart, budget curves. Output: {FIGURES_DIR}/rq1/.
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
from analysis.rq1_analysis import select_best_per_scenario, _filter_s1, _filter_smax


def _add_sample_rank(df):
    df = df.copy()
    sizes = (
        df[["dataset", "sample_size"]].drop_duplicates()
        .sort_values(["dataset", "sample_size"])
    )
    sizes["sample_rank"] = sizes.groupby("dataset").cumcount() + 1
    return df.merge(sizes, on=["dataset", "sample_size"])


def _f1_boxplots(raw_df, sk_df, sel_metric, display_metric, agg_fn, budget_label, figures_dir):
    if budget_label == "s1":
        data_sub = _filter_s1(raw_df)
        sk_sub   = _filter_s1(sk_df)
    else:
        data_sub = _filter_smax(raw_df)
        sk_sub   = _filter_smax(sk_df)

    metric_data = data_sub[data_sub["metric"] == display_metric]
    datasets    = sorted(metric_data["dataset"].unique())

    ncols = 4
    nrows = max(1, int(np.ceil(len(datasets) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        sub = metric_data[metric_data["dataset"] == dataset]

        sorted_models = (
            sub.groupby("model")["value"]
            .agg(agg_fn)
            .sort_values()
            .index.tolist()
        )
        data_per_model = [sub[sub["model"] == m]["value"].values for m in sorted_models]
        colors = [get_color(m) for m in sorted_models]

        bp = ax.boxplot(
            data_per_model, patch_artist=True, whis=1.5,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        for i, model in enumerate(sorted_models):
            sk_r = sk_sub[
                (sk_sub["dataset"] == dataset) &
                (sk_sub["metric"] == display_metric) &
                (sk_sub["model"] == model)
            ]["sk_rank"]
            rank = int(sk_r.mode()[0]) if not sk_r.empty else "?"
            ylim = ax.get_ylim()
            ax.text(i + 1, ylim[1], str(rank),
                    ha="center", va="top", fontsize=7, color="navy")

        ax.set_title(dataset, fontsize=9)
        ax.set_xticks(range(1, len(sorted_models) + 1))
        ax.set_xticklabels(sorted_models, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(f"{display_metric}", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[len(datasets):]:
        ax.set_visible(False)

    fig.tight_layout()
    fname = f"f1_boxplots_{display_metric.lower()}_{agg_name}_{budget_label}_{sel_metric.lower()}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def _f2_borda_bar(borda_global_df, sel_metric, budget_label, figures_dir):
    df = borda_global_df.sort_values("total_borda_score", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [get_color(m) for m in df["model"]]
    bars = ax.barh(df["model"], df["total_borda_score"], color=colors)
    ax.set_xlabel("Total Borda Score", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, df["total_borda_score"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fname = f"f2_borda_bar_{sel_metric.lower()}_{budget_label}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def _f3_budget_curves(raw_df, sel_metric, display_metric, agg, figures_dir):
    data = raw_df[raw_df["metric"] == display_metric].copy()
    data = _add_sample_rank(data)
    models   = sorted(data["model"].unique())
    ranks    = sorted(data["sample_rank"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    for model in models:
        sub = data[data["model"] == model]
        fn  = "mean" if agg == "mean" else "median"
        agg_per_rank = (
            sub.groupby("sample_rank")["value"]
            .agg(fn)
            .reindex(ranks)
        )
        ax.plot(ranks, agg_per_rank.values, marker="o", label=model, color=get_color(model))

    ax.set_xticks(ranks)
    ax.set_xticklabels([f"S{r}" for r in ranks], fontsize=8)
    ax.set_xlabel("Training budget", fontsize=8)
    ax.set_ylabel(f"{agg.capitalize()} {display_metric}", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"f3_budget_curves_{display_metric.lower()}_{agg}_{sel_metric.lower()}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def _f4_rank_stability(sk_df, sel_metric, display_metric, figures_dir):
    sub = sk_df[sk_df["metric"] == display_metric]
    models = sorted(sub["model"].unique(),
                    key=lambda m: sub[sub["model"] == m]["sk_rank"].median())
    data_per_model = [sub[sub["model"] == m]["sk_rank"].values for m in models]
    colors = [get_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(
        data_per_model, patch_artist=True, whis=1.5,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("SK Rank (lower = better)", fontsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = f"f4_rank_stability_{display_metric.lower()}_{sel_metric.lower()}"
    save_figure(fig, os.path.join(figures_dir, fname))
    print(f"  saved {fname}")


def generate_rq1_figures(config=cfg):
    results_dir = config.RESULTS_DIR
    figures_dir = os.path.join(config.FIGURES_DIR, "rq1")
    os.makedirs(figures_dir, exist_ok=True)

    raw_df_orig = pd.read_csv(os.path.join(results_dir, "rq1_raw.csv"))
    raw_df_orig = raw_df_orig[raw_df_orig["value"].notna()].copy()

    for sel_metric in ["MRE"]:
        print(f"\n[RQ1 Figures] Generating figures for selection_metric={sel_metric}")

        raw_df = select_best_per_scenario(raw_df_orig, metric=sel_metric)

        sk_path = os.path.join(results_dir, f"rq1_sk_ranks_{sel_metric}.csv")
        borda_path = os.path.join(results_dir, f"rq1_borda_global_{sel_metric}.csv")
        borda_sc_path = os.path.join(results_dir, f"rq1_borda_scenario_{sel_metric}.csv")
        if not os.path.exists(sk_path):
            print(f"  [skip] {sk_path} not found — run --analyse first")
            continue
        sk_df       = pd.read_csv(sk_path)
        borda_gl    = pd.read_csv(borda_path)

        # ── F1: Box-plot grids ────────────────────────────────────────────────
        # for display_metric in ["MRE"]:
        #     for budget in ["s1", "smax"]:
        #         _f1_boxplots(raw_df, sk_df, sel_metric, display_metric,
        #                      np.median, budget, figures_dir)

        # ── F2: Borda bar ─────────────────────────────────────────────────────
        # _f2_borda_bar(borda_gl, sel_metric, "all", figures_dir)
        # if os.path.exists(borda_sc_path):
        #     borda_sc = pd.read_csv(borda_sc_path)
        #     borda_s1 = _filter_s1(borda_sc)
        #     borda_s1_global = (
        #         borda_s1.groupby("model")["borda_score"].sum()
        #         .reset_index().rename(columns={"borda_score": "total_borda_score"})
        #     )
        #     _f2_borda_bar(borda_s1_global, sel_metric, "s1", figures_dir)

        # ── F3: Budget curves ─────────────────────────────────────────────────
        # for display_metric in ["MRE"]:
        #     for agg in ["median", "mean"]:
        #         _f3_budget_curves(raw_df, sel_metric, display_metric, agg, figures_dir)

        # ── F4: SK rank stability ─────────────────────────────────────────────
        # for display_metric in ["MRE"]:
        #     _f4_rank_stability(sk_df, sel_metric, display_metric, figures_dir)

    print(f"\n[RQ1 Figures] All figures saved to {figures_dir}")
