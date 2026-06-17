# RQ3 LaTeX table generation. Output: {LATEX_DIR}/rq3/.
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg


def _bold(s):
    return r"\textbf{" + str(s) + r"}"


def _save(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _pareto_star(flag):
    return r"$\star$" if str(flag) in ("True", "1", "true") else ""


def _fmt_eff(val, best_val):
    if np.isnan(val):
        return "--"
    s = f"{val:.4f}"
    return _bold(s) if not np.isnan(best_val) and abs(val - best_val) < 1e-9 else s


def _t6_global(summary, metric, agg, latex_dir):
    acc_col = f"{agg}_{metric}"
    summary = summary.sort_values("train_time_s").reset_index(drop=True)

    best_train    = summary["train_time_s"].min()
    best_pred     = summary["pred_time_s"].min()
    best_acc      = summary[acc_col].min()
    best_eff      = summary["efficiency"].max()       if "efficiency"      in summary.columns else float("nan")
    best_pred_eff = summary["pred_efficiency"].max()  if "pred_efficiency" in summary.columns else float("nan")

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        rf"Model & Train (s) & Pred (ms) & {agg.capitalize()} {metric} & Train Eff $\uparrow$ & Pred Eff $\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        model    = row["model"]
        t_s      = row["train_time_s"]
        p_ms     = row["pred_time_s"] * 1000
        acc      = row[acc_col]
        eff      = row.get("efficiency",      float("nan"))
        pred_eff = row.get("pred_efficiency", float("nan"))
        pareto   = _pareto_star(row.get("pareto_pred", False))

        c_t        = _bold(f"{t_s:.3f}")  if abs(t_s - best_train)         < 1e-9 else f"{t_s:.3f}"
        c_p        = _bold(f"{p_ms:.2f}") if abs(p_ms / 1000 - best_pred)  < 1e-9 else f"{p_ms:.2f}"
        c_acc      = _bold(f"{acc:.2f}")  if abs(acc - best_acc)            < 1e-9 else f"{acc:.2f}"
        c_eff      = _fmt_eff(eff,      best_eff)
        c_pred_eff = _fmt_eff(pred_eff, best_pred_eff)

        label = f"{model}{pareto}"
        lines.append(" & ".join([label, c_t, c_p, c_acc, c_eff, c_pred_eff]) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t6_global_{metric.lower()}_{agg}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t7_wide(per_ds, metric, agg, latex_dir):
    acc_col  = f"{agg}_{metric}"
    datasets = sorted(per_ds["dataset"].unique())

    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        rf"Dataset & Model & Train (s) & Pred (ms) & {agg.capitalize()} {metric} & Train Eff & Pred Eff & P \\",
        r"\midrule",
    ]
    for ds in datasets:
        ds_sub     = per_ds[per_ds["dataset"] == ds].sort_values("train_time_s")
        best_train = ds_sub["train_time_s"].min()
        best_acc   = ds_sub[acc_col].min()
        n = len(ds_sub)

        for i, (_, row) in enumerate(ds_sub.iterrows()):
            ds_label = rf"\multirow{{{n}}}{{*}}{{{ds}}}" if i == 0 else ""
            t_s      = row["train_time_s"]
            p_ms     = row["pred_time_s"] * 1000
            acc      = row[acc_col]
            eff      = row.get("efficiency",      float("nan"))
            pred_eff = row.get("pred_efficiency", float("nan"))
            star     = _pareto_star(row.get("pareto_efficient", False))

            c_t   = _bold(f"{t_s:.3f}") if abs(t_s  - best_train) < 1e-9 else f"{t_s:.3f}"
            c_acc = _bold(f"{acc:.2f}") if abs(acc  - best_acc)   < 1e-9 else f"{acc:.2f}"
            c_eff      = f"{eff:.4f}"      if not np.isnan(eff)      else "--"
            c_pred_eff = f"{pred_eff:.4f}" if not np.isnan(pred_eff) else "--"

            lines.append(
                " & ".join([ds_label, row["model"], c_t, f"{p_ms:.2f}", c_acc,
                            c_eff, c_pred_eff, star])
                + r" \\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")

    fname = f"t7_wide_{metric.lower()}_{agg}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t7_per_dataset(per_ds, metric, agg, latex_dir):
    acc_col  = f"{agg}_{metric}"
    datasets = sorted(per_ds["dataset"].unique())

    for ds in datasets:
        ds_sub     = per_ds[per_ds["dataset"] == ds].sort_values("train_time_s")
        best_train = ds_sub["train_time_s"].min()
        best_acc   = ds_sub[acc_col].min()

        lines = [
            f"% Dataset: {ds}  |  {metric} {agg}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            rf"Model & Train (s) & Pred (ms) & {agg.capitalize()} {metric} & Train Eff & Pred Eff & P \\",
            r"\midrule",
        ]
        for _, row in ds_sub.iterrows():
            t_s      = row["train_time_s"]
            p_ms     = row["pred_time_s"] * 1000
            acc      = row[acc_col]
            eff      = row.get("efficiency",      float("nan"))
            pred_eff = row.get("pred_efficiency", float("nan"))
            star     = _pareto_star(row.get("pareto_efficient", False))

            c_t        = _bold(f"{t_s:.3f}") if abs(t_s  - best_train) < 1e-9 else f"{t_s:.3f}"
            c_acc      = _bold(f"{acc:.2f}") if abs(acc  - best_acc)   < 1e-9 else f"{acc:.2f}"
            c_eff      = f"{eff:.4f}"      if not np.isnan(eff)      else "--"
            c_pred_eff = f"{pred_eff:.4f}" if not np.isnan(pred_eff) else "--"

            lines.append(
                " & ".join([row["model"], c_t, f"{p_ms:.2f}", c_acc,
                            c_eff, c_pred_eff, star])
                + r" \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]

        safe_ds = ds.replace(" ", "_").replace("/", "-").lower()
        fname   = f"t7_{safe_ds}_{metric.lower()}_{agg}.tex"
        _save(lines, os.path.join(latex_dir, fname))
        print(f"  wrote {fname}")


def _t8_s1(s1_summary, metric, agg, latex_dir):
    acc_col = f"{agg}_{metric}"
    s1_summary = s1_summary.sort_values("train_time_s").reset_index(drop=True)

    best_train    = s1_summary["train_time_s"].min()
    best_acc      = s1_summary[acc_col].min()
    best_eff      = s1_summary["efficiency"].max()       if "efficiency"      in s1_summary.columns else float("nan")
    best_pred_eff = s1_summary["pred_efficiency"].max()  if "pred_efficiency" in s1_summary.columns else float("nan")

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        rf"Model & Train (s) & Pred (ms) & {agg.capitalize()} {metric} (S1) & Train Eff $\uparrow$ & Pred Eff $\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in s1_summary.iterrows():
        t_s      = row["train_time_s"]
        p_ms     = row["pred_time_s"] * 1000
        acc      = row[acc_col]
        eff      = row.get("efficiency",      float("nan"))
        pred_eff = row.get("pred_efficiency", float("nan"))
        star     = _pareto_star(row.get("pareto_pred", False))

        c_t        = _bold(f"{t_s:.3f}") if abs(t_s  - best_train) < 1e-9 else f"{t_s:.3f}"
        c_acc      = _bold(f"{acc:.2f}") if abs(acc  - best_acc)   < 1e-9 else f"{acc:.2f}"
        c_eff      = _fmt_eff(eff,      best_eff)
        c_pred_eff = _fmt_eff(pred_eff, best_pred_eff)

        label = f"{row['model']}{star}"
        lines.append(" & ".join([label, c_t, f"{p_ms:.2f}", c_acc, c_eff, c_pred_eff]) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    fname = f"t8_s1_{metric.lower()}_{agg}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def _t9_pareto_count(per_ds, metric, agg, latex_dir):
    datasets = sorted(per_ds["dataset"].unique())
    models   = sorted(per_ds["model"].unique())

    if "pareto_pred" not in per_ds.columns:
        per_ds = per_ds.copy()
        per_ds["pareto_pred"] = False

    _tobool = lambda x: str(x) in ("True", "1", "true")

    def _make_pivot(col):
        piv = per_ds.pivot_table(
            index="model", columns="dataset", values=col, aggfunc="first"
        ).reindex(index=models, columns=datasets).fillna(False)
        return pd.DataFrame(
            {c: piv[c].apply(_tobool) for c in piv.columns}, index=piv.index)

    pivot_train = _make_pivot("pareto_efficient")
    pivot_pred  = _make_pivot("pareto_pred")

    lines = [
        r"\begin{tabular}{l" + "c" * len(datasets) + "cc}",
        r"\toprule",
        "Model & " + " & ".join(datasets) + r" & $\star$ & $\dagger$ \\",
        r"\midrule",
    ]
    for model in models:
        cells = [model]
        for ds in datasets:
            t = pivot_train.at[model, ds]
            p = pivot_pred.at[model, ds]
            if t and p:
                cells.append(r"$\star\dagger$")
            elif t:
                cells.append(r"$\star$")
            elif p:
                cells.append(r"$\dagger$")
            else:
                cells.append("")
        cells.append(str(int(pivot_train.loc[model].sum())))
        cells.append(str(int(pivot_pred.loc[model].sum())))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\midrule",
        r"\multicolumn{" + str(len(datasets) + 3) + r"}{l}{"
        + r"$\star$ = train-Pareto \quad $\dagger$ = pred-Pareto} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    fname = f"t9_pareto_count_{metric.lower()}_{agg}.tex"
    _save(lines, os.path.join(latex_dir, fname))
    print(f"  wrote {fname}")


def generate_rq3_tables(config=cfg):
    results_dir = config.RESULTS_DIR
    latex_dir   = os.path.join(config.LATEX_DIR, "rq3")
    os.makedirs(latex_dir, exist_ok=True)

    # for metric in ["MRE", "MAE"]:
    #     for agg in ["median", "mean"]:
    #         suffix = f"{metric.lower()}_{agg}"
    #         print(f"\n[RQ3 Tables] {metric} / {agg}")
    #         g_path = os.path.join(results_dir, f"rq3_summary_{suffix}.csv")
    #         if not os.path.exists(g_path):
    #             print(f"  [skip] {g_path} not found -- run --analyse first")
    #             continue
    #         global_s = pd.read_csv(g_path)
    #         _t6_global(global_s, metric, agg, latex_dir)
    #         pd_path = os.path.join(results_dir, f"rq3_summary_per_dataset_{suffix}.csv")
    #         if os.path.exists(pd_path):
    #             per_ds = pd.read_csv(pd_path)
    #             _t7_wide(per_ds, metric, agg, latex_dir)
    #             _t7_per_dataset(per_ds, metric, agg, latex_dir)
    #             _t9_pareto_count(per_ds, metric, agg, latex_dir)
    #         s1_path = os.path.join(results_dir, f"rq3_summary_s1_{suffix}.csv")
    #         if os.path.exists(s1_path):
    #             s1 = pd.read_csv(s1_path)
    #             _t8_s1(s1, metric, agg, latex_dir)
    # print(f"\n[LaTeX] RQ3 tables written to {latex_dir}")
