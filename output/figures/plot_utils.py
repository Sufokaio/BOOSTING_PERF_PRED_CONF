# Shared matplotlib settings, model color palette, and figure save helper.
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":        10,
    "axes.labelsize":   8,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "figure.dpi":       600,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

MODEL_COLORS = {
    "SVRBoost":    "#1f77b4",
    "RidgeBoost":  "#4fa3d1",
    "MLPBoost":    "#17becf",
    "TreeBoost":   "#2ca02c",
    "XGBoost":     "#98df8a",
    "CART":        "#d62728",
    "SVR":         "#ff7f0e",
    "RR":          "#ffbb78",
    "HINNPerf":    "#9467bd",
}

MODEL_FAMILIES = {
    "boosted":   ["SVRBoost", "RidgeBoost", "MLPBoost"],
    "tree":      ["TreeBoost", "XGBoost"],
    "baselines": ["CART", "SVR", "RR"],
    "hinnperf":  ["HINNPerf"],
}


def get_color(model_name):
    return MODEL_COLORS.get(model_name, "#7f7f7f")


def save_figure(fig, path_no_ext):
    import os
    os.makedirs(os.path.dirname(path_no_ext) if os.path.dirname(path_no_ext) else ".", exist_ok=True)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=600)
    plt.close(fig)
