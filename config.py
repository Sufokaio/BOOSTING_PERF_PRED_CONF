# Edit this file before running. All other files read from here.

DATASETS = {
    "Apache":  "data/apache.csv",
    "BDBC":    "data/bdbc.csv",
    "Sqlite":    "data/sqlite.csv",
    "x264":    "data/x264.csv",
    "VP8":  "data/vp8.csv",


    "Kanzi":    "data/kanzi.csv",

    "Lrzip":  "data/lrzip.csv",

    "Dune":  "data/dune.csv"

}

SAMPLE_SIZES = {

    #     "Apache": [9,18],
    # "BDBC":   [18],


    "Apache": [9,18,27,36,45],
    "BDBC":   [18,36,54,72,90],
        "Sqlite":   [14,28,42,56,70],
        "x264":    [16,32,48,64,80],
        "VP8":     [121,273,356,467,830],

    "Kanzi":   [31,62,93,124,155],
    "Lrzip":   [127,295,386,485,907],

    "Dune":   [224,692,1000,1365,1612]

}

N_RUNS = 30
RANDOM_SEED_BASE = 42

M_VALUES   = [5, 10, 20, 30, 50]
ETA_VALUES = [0.01, 0.05, 0.1, 0.2]

DEFAULT_M   = 20
DEFAULT_ETA = 0.1

SVR_PARAMS   = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
RIDGE_PARAMS = {"alpha": 1.0}
MLP_PARAMS   = {"hidden_layer_sizes": (16,), "max_iter": 200, "random_state": 0}
TREE_PARAMS  = {"max_depth": 3}

XGBOOST_GRID = {
    "n_estimators":     [20, 50, 100],
    "learning_rate":    [0.05, 0.1, 0.2],
    "max_depth":        [2, 4, 6],
    # "min_child_weight": [1, 3, 5],
    "min_child_weight": [1],

    "subsample":        [0.8],
    "colsample_bytree": [0.8],
    "random_state":     [0],
}
CART_GRID = {
    "max_depth":         [3, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [2, 6, 10],
    "random_state":      [0],
}
BASELINE_SVR_GRID = {
    "kernel":  ["rbf"],
    "C":       [1.0, 10.0, 50.0],
    "epsilon": [0.01, 0.05, 0.1],
    "gamma":   ["scale"],
}
BASELINE_RIDGE_GRID = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
}

RESULTS_DIR     = "output/results/"
LATEX_DIR       = "output/latex/"
FIGURES_DIR     = "output/figures/"
CHECKPOINT_FILE = "output/checkpoint.json"

N_JOBS = 11
