# Paper tables and figures with validation-based selection.
import os
import sys
import json
import math
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from analysis.rq1_analysis import select_best_per_scenario, _filter_s1
from analysis.rq3_analysis import _pareto_frontier
from analysis_val import run_val_analysis as rva
from evaluation.statistical import run_full_statistical_analysis
from output.figures.plot_utils import get_color

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 100, "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})

METRICS = rva.METRICS
KEY = rva.KEY
BOOSTED = ["SVRBoost", "RidgeBoost", "MLPBoost", "TreeBoost"]
TREES = ["CART", "TreeBoost", "XGBoost"]
SCATTER_SIZE = (466.04 / 72, 346.32 / 72)
HEATMAP_SIZE = (706.69 / 72, 562.32 / 72)
LABEL_FS = 10
CIRCLE_S, STAR_S = 70, 170

CAPTION_V = ("Predictive performance aggregated over all 40 evaluation scenarios.\n"
             "Each cell reports mean error with standard deviation (in parentheses) and the mean SK rank as a "
             "subscript. Cells with the best mean SK rank per metric are highlighted in bold.}\n")
CAPTION_VI = ("Predictive performance at the smallest training budget (S$_1$).\n"
              "Format as in \\Cref{tab:rq1-overall}.}\n")


def load(args):
    aws = pd.read_csv(args.test)
    aws = aws[aws["value"].notna()].copy()
    scores = pd.read_csv(args.val)
    val = scores[scores["split"] == "val"].drop(columns="split")
    xgb_pc = scores[scores["split"] == "test"].drop(columns="split")
    test = aws.merge(xgb_pc[KEY + ["value"]], on=KEY, how="left", suffixes=("", "_pc"))
    is_xgb = test["model"].str.startswith("XGBoost")
    if test.loc[is_xgb, "value_pc"].isna().any() or (~is_xgb & test["value_pc"].notna()).any():
        raise RuntimeError("XGBoost refit scores do not line up with the test results")
    test.loc[is_xgb, "value"] = test.loc[is_xgb, "value_pc"]
    test = test.drop(columns="value_pc")
    return aws, test, val


def budget(df, b):
    return _filter_s1(df) if b == "s1" else df


def cell(r, m, best):
    s = f"{r[m]:.2f} ({r[m + '_spread']:.2f})$_{{{r[m + '_rank']:.2f}}}$"
    return r"\textbf{" + s + "}" if abs(r[m + "_rank"] - best[m]) < 1e-9 else s


def table_block(tab, b):
    best = {m: tab[m + "_rank"].min() for m in METRICS}
    if b == "all":
        head = ["\\begin{table*}[htbp]", "\\centering", "\\scriptsize", "\\caption{" + CAPTION_V,
                "\\label{tab:rq1-overall}"]
    else:
        head = ["\\begin{table*}[htbp]", "\\centering", "\\scriptsize", "", "\\caption{" + CAPTION_VI.rstrip("\n"),
                "\\label{tab:rq1-s1}"]
    body = ["\\begin{tabular}{lcccc}", "\\toprule", "Model & MRE & MAE & RMSE & MIBRE \\\\", "\\midrule"]
    for _, r in tab.iterrows():
        body.append(" & ".join([r["model"]] + [cell(r, m, best) for m in METRICS]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(head + body) + "\n"


def timing_means(sel, b):
    d = budget(sel, b).drop_duplicates(["model", "dataset", "sample_size", "run"])
    return d.groupby("model")[["train_time_s", "pred_time_s"]].mean()


def hinnperf_pred(path, b):
    d = budget(pd.read_csv(path), b)
    per_run = d.groupby(["dataset", "sample_size", "run"])["pred_first_s"].median()
    by_shape = d.groupby(["num_block", "num_layer_pb"])["pred_first_s"].mean()
    return per_run.mean(), by_shape.min(), by_shape.max(), by_shape


def timing_table(t, path):
    lines = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
             "\\caption{Mean training time (s) and mean prediction time (ms) per run of the selected variants. "
             "HINNPerf training time is its hyperparameter search; its prediction time was measured separately "
             "without retraining (mean over its candidate network shapes).}",
             "\\label{tab:rq3-time}",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{Training (s)} & \\multicolumn{2}{c}{Prediction (ms)} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Model & All & S$_1$ & All & S$_1$ \\\\", "\\midrule"]
    for model, r in t.iterrows():
        lines.append(f"{model} & {fmt_s(r['train_all'])} & {fmt_s(r['train_s1'])} & "
                     f"{r['pred_all_ms']:.2f} & {r['pred_s1_ms']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fmt_s(v):
    return f"{v:.3f}" if v < 1 else f"{v:.2f}"


def _boxes_overlap(a, b, pad=0.0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segs_cross(p1, p2, p3, p4):
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def _seg_hits_box(p, q, b):
    inside = lambda r: b[0] <= r[0] <= b[2] and b[1] <= r[1] <= b[3]
    if inside(p) or inside(q):
        return True
    c = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
    return any(_segs_cross(p, q, c[i], c[(i + 1) % 4]) for i in range(4))


def _closest_on_box(p, b):
    return (min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3]))


def _inside(inner, outer, pad):
    return (inner[0] >= outer[0] + pad and inner[1] >= outer[1] + pad
            and inner[2] <= outer[2] - pad and inner[3] <= outer[3] - pad)


AMBIGUITY = 1.5


def _ambiguous(box, model, x, y, pts):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    own = math.hypot(cx - x, cy - y)
    return any(math.hypot(cx - x2, cy - y2) < AMBIGUITY * own for m2, x2, y2, _ in pts if m2 != model)


ANGLES = [45, 0, 90, 315, 135, 180, 270, 225, 22, 68, 112, 158, 202, 248, 292, 338]
RADII = [5, 8, 12, 17, 23, 30, 38, 48, 60, 75]


def place_labels(fig, ax, pts, segments):
    r = fig.canvas.get_renderer()
    pxpt = fig.dpi / 72.0
    axbox = ax.get_window_extent(r).extents
    sizes = {}
    for model, *_ in pts:
        t = ax.text(0, 0, model, fontsize=LABEL_FS)
        bb = t.get_window_extent(r)
        sizes[model] = (bb.width, bb.height)
        t.remove()
    point_boxes = {m: (x - rad, y - rad, x + rad, y + rad) for m, x, y, rad in pts}
    crowd = {m: sum(math.hypot(x - x2, y - y2) < 80 for _, x2, y2, _ in pts) for m, x, y, _ in pts}
    order = sorted(pts, key=lambda p: -crowd[p[0]])
    placed, leaders = {}, {}
    for model, x, y, rad in order:
        w, h = sizes[model]
        done = False
        for radius in RADII:
            for ang in ANGLES:
                c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                ox, oy = (rad / pxpt + radius) * c, (rad / pxpt + radius) * s
                ha = "left" if c > 0.3 else "right" if c < -0.3 else "center"
                va = "bottom" if s > 0.3 else "top" if s < -0.3 else "center"
                axx, axy = x + ox * pxpt, y + oy * pxpt
                x0 = axx if ha == "left" else axx - w if ha == "right" else axx - w / 2
                y0 = axy if va == "bottom" else axy - h if va == "top" else axy - h / 2
                box = (x0 - 2, y0 - 2, x0 + w + 2, y0 + h + 2)
                if not _inside(box, axbox, 3):
                    continue
                if any(_boxes_overlap(box, b, 3) for b in placed.values()):
                    continue
                if any(_boxes_overlap(box, b, 2) for b in point_boxes.values()):
                    continue
                if any(_seg_hits_box(p, q, box) for p, q in segments):
                    continue
                if any(_seg_hits_box(p, q, box) for p, q in leaders.values()):
                    continue
                if radius < 12 and _ambiguous(box, model, x, y, pts):
                    continue
                lead = None
                if radius >= 12:
                    relx = {"left": 0.0, "right": 1.0, "center": 0.5}[ha]
                    rely = {"bottom": 0.0, "top": 1.0, "center": 0.5}[va]
                    q = (x0 + relx * w, y0 + rely * h)
                    ang_ = math.atan2(q[1] - y, q[0] - x)
                    p = (x + rad * math.cos(ang_), y + rad * math.sin(ang_))
                    lead = (p, q)
                    if any(_seg_hits_box(p, q, b) for m2, b in placed.items()):
                        continue
                    if any(_seg_hits_box(p, q, b) for m2, b in point_boxes.items() if m2 != model):
                        continue
                    if any(_segs_cross(p, q, a, b) for a, b in segments + list(leaders.values())):
                        continue
                placed[model] = box
                if lead:
                    leaders[model] = lead
                pts_info = dict(offset_pt=(round(ox, 1), round(oy, 1)), ha=ha, va=va, leader=bool(lead))
                placed_info[model] = pts_info
                done = True
                break
            if done:
                break
        if not done:
            return None
    return placed_info


placed_info = {}


def obstacle_segments(ax, xs, ys, front, ranges):
    tr = ax.transData.transform
    segs = []
    f = front.sort_values("x")
    pts = [tr((a, b)) for a, b in zip(f["x"], f["y"])]
    for i in range(len(pts) - 1):
        corner = tr((f["x"].iloc[i + 1], f["y"].iloc[i]))
        segs += [(tuple(pts[i]), tuple(corner)), (tuple(corner), tuple(pts[i + 1]))]
    for (lo, hi, y) in ranges:
        a, b = tr((lo, y)), tr((hi, y))
        segs.append((tuple(a), tuple(b)))
        cap = 4 * plt.gcf().dpi / 72
        for e in (a, b):
            segs.append(((e[0], e[1] - cap), (e[0], e[1] + cap)))
    return segs


def draw_scatter(summary, time_col, scale, xlabel, path_no_ext, ranges=None):
    global placed_info
    s = summary.copy()
    s["x"] = s[time_col] * scale
    s["y"] = s["MRE"]
    s["pareto"] = _pareto_frontier(s, "x", "y")
    for margin in (0.08, 0.12, 0.16, 0.22):
        placed_info = {}
        fig = plt.figure(figsize=SCATTER_SIZE)
        ax = fig.add_subplot(111)
        for _, r in s.iterrows():
            ax.scatter(r["x"], r["y"], color=get_color(r["model"]), s=STAR_S if r["pareto"] else CIRCLE_S,
                       marker="*" if r["pareto"] else "o", zorder=3)
        rng = []
        for m, (lo, hi) in (ranges or {}).items():
            r = s[s["model"] == m].iloc[0]
            ax.errorbar(r["x"], r["y"], xerr=[[r["x"] - lo * scale], [hi * scale - r["x"]]], fmt="none",
                        ecolor=get_color(m), elinewidth=1.2, capsize=4, zorder=2)
            rng.append((lo * scale, hi * scale, r["y"]))
        front = s[s["pareto"]].sort_values("x")
        line = None
        if len(front) > 1:
            line = ax.step(front["x"], front["y"], where="post", linestyle="--", color="gray", alpha=0.6,
                           label="Pareto frontier")
        ax.set_xscale("log")
        ax.margins(x=margin, y=margin)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean MRE")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        tr = ax.transData.transform
        pxpt = fig.dpi / 72
        pts = []
        for _, r in s.iterrows():
            d = tr((r["x"], r["y"]))
            rad = math.sqrt(STAR_S if r["pareto"] else CIRCLE_S) / 2 * pxpt + 1
            pts.append((r["model"], d[0], d[1], rad))
        segs = obstacle_segments(ax, s["x"], s["y"], front, rng)
        info = place_labels(fig, ax, pts, segs)
        if info is None:
            plt.close(fig)
            continue
        anns = {}
        for model, x, y, rad in pts:
            p = info[model]
            r_ = s[s["model"] == model].iloc[0]
            relpos = ({"left": 0.0, "right": 1.0, "center": 0.5}[p["ha"]],
                      {"bottom": 0.0, "top": 1.0, "center": 0.5}[p["va"]])
            kw = dict(arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, shrinkA=1, relpos=relpos,
                                      shrinkB=rad / pxpt)) if p["leader"] else {}
            anns[model] = ax.annotate(model, (r_["x"], r_["y"]), xytext=p["offset_pt"], textcoords="offset points",
                                      ha=p["ha"], va=p["va"], fontsize=LABEL_FS, zorder=4, **kw)
        legend_loc = None
        if line is not None:
            for loc in ["upper right", "upper left", "lower right", "lower left", "center right", "center left",
                        "upper center", "lower center"]:
                leg = ax.legend(loc=loc)
                fig.canvas.draw()
                lb = leg.get_window_extent(fig.canvas.get_renderer()).extents
                ok = True
                ok &= not any(_boxes_overlap(lb, (x - rad, y - rad, x + rad, y + rad), 2) for _, x, y, rad in pts)
                ok &= not any(_boxes_overlap(lb, text_box(a), 2) for a in anns.values())
                ok &= not any(_seg_hits_box(p, q, lb) for p, q in segs)
                ok &= not any(_seg_hits_box(p, q, lb) for a in anns.values() for p, q in leader_segments(a))
                if ok:
                    legend_loc = loc
                    break
                leg.remove()
            if legend_loc is None:
                plt.close(fig)
                continue
        checks = verify(fig, ax, pts, segs, anns, legend_loc)
        if checks["failures"]:
            plt.close(fig)
            continue
        os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
        fig.savefig(path_no_ext + ".pdf")
        plt.close(fig)
        checks.update(margin=margin, legend=legend_loc, labels=info, size_pt=[round(v * 72, 2) for v in SCATTER_SIZE])
        return checks, s
    raise RuntimeError(f"no clean label layout found for {path_no_ext}")


def text_box(ann, renderer=None):
    return matplotlib.text.Text.get_window_extent(ann, renderer).extents


def leader_segments(ann):
    if ann.arrow_patch is None:
        return []
    out = []
    for path in ann.arrow_patch.get_path_in_displaycoord()[0]:
        v = path.vertices
        out += [(tuple(v[i]), tuple(v[i + 1])) for i in range(len(v) - 1)]
    return out


def verify(fig, ax, pts, segs, anns, legend_loc):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(r).extents
    failures = []
    boxes = {m: text_box(a, r) for m, a in anns.items()}
    leads = {m: leader_segments(a) for m, a in anns.items() if a.arrow_patch is not None}
    point_boxes = {m: (x - rad, y - rad, x + rad, y + rad) for m, x, y, rad in pts}
    for m, b in boxes.items():
        if m in leads:
            continue
        own = next(p for p in pts if p[0] == m)
        if _ambiguous(b, m, own[1], own[2], pts):
            failures.append(f"{m} label is ambiguous without a leader line")
    items = dict(boxes)
    if legend_loc:
        items["<legend>"] = ax.get_legend().get_window_extent(r).extents
    for m, b in items.items():
        if not _inside(b, axbox, 0):
            failures.append(f"{m} outside axes")
        for m2, b2 in items.items():
            if m < m2 and _boxes_overlap(b, b2):
                failures.append(f"{m} overlaps {m2}")
        for m2, pb in point_boxes.items():
            if _boxes_overlap(b, pb):
                failures.append(f"{m} overlaps point {m2}")
        for p, q in segs:
            if _seg_hits_box(p, q, b):
                failures.append(f"{m} overlaps a line")
        for m2, ls in leads.items():
            if m2 != m and any(_seg_hits_box(p, q, b) for p, q in ls):
                failures.append(f"leader of {m2} crosses {m}")
    for m, ls in leads.items():
        for p, q in ls:
            for m2, l2 in leads.items():
                if m2 > m and any(_segs_cross(p, q, a, b) for a, b in l2):
                    failures.append(f"leaders of {m} and {m2} cross")
            if any(_segs_cross(p, q, a, b) for a, b in segs):
                failures.append(f"leader of {m} crosses a line")
            for m2, pb in point_boxes.items():
                if m2 != m and _seg_hits_box(p, q, pb):
                    failures.append(f"leader of {m} crosses point {m2}")
    return {"failures": sorted(set(failures)), "n_labels": len(boxes), "n_leaders": len(leads)}


def draw_heatmap(median_df, path_no_ext):
    M_vals, eta_vals = sorted(cfg.M_VALUES), sorted(cfg.ETA_VALUES)
    fig, axes = plt.subplots(2, 2, figsize=HEATMAP_SIZE)
    for ax, base in zip(axes.flatten(), BOOSTED):
        grid = median_df[median_df["base_model"] == base].groupby(["M", "eta"])["median_MRE"].mean()
        pivot = grid.unstack("eta").reindex(index=M_vals, columns=eta_vals)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", origin="upper")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Mean MRE", fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax.set_xticks(range(len(eta_vals)))
        ax.set_xticklabels([str(e) for e in eta_vals], fontsize=11)
        ax.set_yticks(range(len(M_vals)))
        ax.set_yticklabels([str(m) for m in M_vals], fontsize=11)
        ax.set_xlabel(r"$\eta$", fontsize=12)
        ax.set_ylabel(r"$M$", fontsize=12)
        ax.set_title(base, fontsize=13)
        for i in range(len(M_vals)):
            for j in range(len(eta_vals)):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="output/final/results/rq1_raw.csv")
    parser.add_argument("--val", default="output/val/val_raw.csv")
    parser.add_argument("--hinnperf", default="output/val/hinnperf_time/hinnperf_pred_time.csv")
    parser.add_argument("--out", default="output/paper_final")
    args = parser.parse_args()

    out = {k: os.path.join(args.out, k) for k in ("tables", "figures", "numbers")}
    for d in out.values():
        os.makedirs(d, exist_ok=True)
    log = {}

    aws, test, val = load(args)
    chosen = rva.chosen_variants(val, "median")
    sel = select_best_per_scenario(rva.keep_chosen(test, chosen), metric="MRE")

    sk, _, _ = run_full_statistical_analysis(sel)
    log["sk_failures"] = len(rva._sk_failures)
    sk.to_csv(os.path.join(out["numbers"], "sk_ranks.csv"), index=False)
    chosen.to_csv(os.path.join(out["numbers"], "selected_variants.csv"), index=False)

    new_tab = {}
    for b, name in (("all", "table_V"), ("s1", "table_VI")):
        new_tab[b] = rva.summary_table(sel, sk, b, "mean")
        new_tab[b].to_csv(os.path.join(out["numbers"], f"{name}.csv"), index=False)
        with open(os.path.join(out["tables"], f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(table_block(new_tab[b], b))

    timing, summaries, hp = {}, {}, {}
    for b in ("all", "s1"):
        t = timing_means(sel, b)
        point, lo, hi, by_shape = hinnperf_pred(args.hinnperf, b)
        hp[b] = dict(point=point, lo=lo, hi=hi, by_shape=by_shape)
        t.loc["HINNPerf", "pred_time_s"] = point
        mre = budget(sel, b)
        mre = mre[mre["metric"] == "MRE"].groupby("model")["value"].mean().rename("MRE")
        summaries[b] = t.join(mre).reset_index()
        timing[b] = t
        by_shape.rename("mean_pred_first_s").reset_index().to_csv(
            os.path.join(out["numbers"], f"hinnperf_pred_time_by_shape_{b}.csv"), index=False)
    tt = pd.DataFrame({"train_all": timing["all"]["train_time_s"], "train_s1": timing["s1"]["train_time_s"],
                       "pred_all_ms": timing["all"]["pred_time_s"] * 1000,
                       "pred_s1_ms": timing["s1"]["pred_time_s"] * 1000}).sort_index()
    tt.to_csv(os.path.join(out["numbers"], "timing_table.csv"))
    timing_table(tt, os.path.join(out["tables"], "table_timing.tex"))

    log["figures"] = {}
    pareto_rows = []
    for b in ("all", "s1"):
        s = summaries[b]
        for kind, col, scale, xlabel, ranges in (
                ("pred", "pred_time_s", 1000, "Mean Prediction Time (ms)", None),
                ("train", "train_time_s", 1, "Mean Training Time (s)", None)):
            name = f"scatter_mre_mean_{kind}_{b}"
            checks, ss = draw_scatter(s, col, scale, xlabel, os.path.join(out["figures"], name), ranges)
            log["figures"][name] = checks
            for _, r in ss.iterrows():
                dom = ss[(ss["x"] <= r["x"]) & (ss["y"] <= r["y"]) & ((ss["x"] < r["x"]) | (ss["y"] < r["y"]))]
                pareto_rows.append(dict(budget=b, time=kind, model=r["model"], mean_MRE=r["y"],
                                        time_value=r["x"], time_unit="ms" if kind == "pred" else "s",
                                        pareto=bool(r["pareto"]), dominated_by=", ".join(dom["model"])))
    pareto = pd.DataFrame(pareto_rows)
    pareto.to_csv(os.path.join(out["numbers"], "rq3_pareto.csv"), index=False)

    ratios = []
    for b in ("all", "s1"):
        t = timing[b]
        for tree in TREES:
            ratios.append(dict(budget=b, versus=tree,
                               pred_ratio_hinnperf=t.loc["HINNPerf", "pred_time_s"] / t.loc[tree, "pred_time_s"],
                               train_ratio_hinnperf=t.loc["HINNPerf", "train_time_s"] / t.loc[tree, "train_time_s"]))
    pd.DataFrame(ratios).to_csv(os.path.join(out["numbers"], "rq3_ratios.csv"), index=False)

    rq2_tmp = os.path.join(out["numbers"], "rq2_tmp")
    os.makedirs(rq2_tmp, exist_ok=True)
    grid_rows = []
    med = rva.rq2_medians(test, rq2_tmp, "test")
    for b in ("all", "s1"):
        d = budget(med, b)
        draw_heatmap(d, os.path.join(out["figures"], f"heatmap_mre_mean_{b}"))
        g = d.groupby(["base_model", "M", "eta"])["median_MRE"].mean().reset_index()
        for base, gg in g.groupby("base_model"):
            bst, wst = gg.loc[gg["median_MRE"].idxmin()], gg.loc[gg["median_MRE"].idxmax()]
            grid_rows.append(dict(split="test", budget=b, model=base, best_M=bst["M"], best_eta=bst["eta"],
                                  best_mean_MRE=bst["median_MRE"], worst_M=wst["M"], worst_eta=wst["eta"],
                                  worst_mean_MRE=wst["median_MRE"]))
    os.replace(os.path.join(rq2_tmp, "rq2_median_mre_test.csv"),
               os.path.join(out["numbers"], "rq2_median_mre_test.csv"))
    os.rmdir(rq2_tmp)
    pd.DataFrame(grid_rows).to_csv(os.path.join(out["numbers"], "rq2_grid_range.csv"), index=False)

    ch = chosen[chosen["model_family"].isin(BOOSTED)].copy()
    ch["M"] = ch["chosen"].str.extract(r"_M(\d+)_").astype(int)
    ch["eta"] = ch["chosen"].str.extract(r"_eta([\d.]+)").astype(float)
    counts = ch.groupby(["model_family", "M", "eta"]).size().rename("n_scenarios").reset_index()
    counts.to_csv(os.path.join(out["numbers"], "rq2_selected_counts.csv"), index=False)
    edge = ch.groupby("model_family").apply(lambda g: pd.Series(dict(
        n=len(g), M_50=int((g["M"] == 50).sum()), eta_0_2=int((g["eta"] == 0.2).sum()),
        M_50_or_eta_0_2=int(((g["M"] == 50) | (g["eta"] == 0.2)).sum()),
        M_50_and_eta_0_2=int(((g["M"] == 50) & (g["eta"] == 0.2)).sum())))).reset_index()
    edge.to_csv(os.path.join(out["numbers"], "rq2_grid_edge.csv"), index=False)

    ps_rank = sk[sk["metric"] == "MRE"].groupby(["dataset", "model"])["sk_rank"].mean().rename("mean_sk_rank_MRE")
    ps_err = sel.groupby(["dataset", "model", "metric"])["value"].mean().unstack("metric")[METRICS]
    ps_err.columns = [f"mean_{m}" for m in METRICS]
    per_sys = ps_err.join(ps_rank).reset_index()
    per_sys.to_csv(os.path.join(out["numbers"], "per_system.csv"), index=False)
    best_sys = per_sys.loc[per_sys.groupby("dataset")["mean_sk_rank_MRE"].transform("min")
                           .sub(per_sys["mean_sk_rank_MRE"]).abs() < 1e-9, ["dataset", "model", "mean_sk_rank_MRE"]]
    best_sys.to_csv(os.path.join(out["numbers"], "per_system_best_rank_MRE.csv"), index=False)

    log["latex_mismatches"] = crosscheck(out)
    log["hinnperf_pred"] = {b: dict(point_ms=hp[b]["point"] * 1000, fastest_ms=hp[b]["lo"] * 1000,
                                    slowest_ms=hp[b]["hi"] * 1000) for b in hp}
    print(json.dumps(log, indent=1, default=str))


def crosscheck(out):
    import re
    bad = 0
    for name in ("table_V", "table_VI"):
        tab = pd.read_csv(os.path.join(out["numbers"], f"{name}.csv")).set_index("model")
        best_ = {m: tab[m + "_rank"].min() for m in METRICS}
        for line in open(os.path.join(out["tables"], f"{name}.tex"), encoding="utf-8"):
            if " & " not in line or line.startswith("Model"):
                continue
            cells = [c.strip() for c in line.strip().rstrip("\\").split("&")]
            model = cells[0]
            for m, c in zip(METRICS, cells[1:]):
                v = re.search(r"([\d.]+) \(([\d.]+)\)\$_\{([\d.]+)\}\$", c)
                r = tab.loc[model]
                ok = (v[1] == f"{r[m]:.2f}" and v[2] == f"{r[m + '_spread']:.2f}" and v[3] == f"{r[m + '_rank']:.2f}"
                      and ("textbf" in c) == (abs(r[m + "_rank"] - best_[m]) < 1e-9))
                bad += not ok
    tt = pd.read_csv(os.path.join(out["numbers"], "timing_table.csv"), index_col=0)
    for line in open(os.path.join(out["tables"], "table_timing.tex"), encoding="utf-8"):
        cells = [c.strip() for c in line.strip().rstrip("\\").split("&")]
        if len(cells) == 5 and cells[0] in tt.index:
            r = tt.loc[cells[0]]
            exp = [fmt_s(r["train_all"]), fmt_s(r["train_s1"]), f"{r['pred_all_ms']:.2f}", f"{r['pred_s1_ms']:.2f}"]
            bad += sum(a != b for a, b in zip(cells[1:], exp))
    return bad


if __name__ == "__main__":
    main()
