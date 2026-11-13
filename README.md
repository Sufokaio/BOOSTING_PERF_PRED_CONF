# Trees or Alternative Base Learners? Evaluating Design Choices in Gradient Boosting for Configurable Software Performance Prediction

This repository contains the source code, datasets, and raw results accompanying the paper. The study empirically investigates whether replacing the standard decision(regression)-tree base learner inside gradient boosting with alternatives (SVR, Ridge Regression, MLP) improves accuracy or cost-efficiency when predicting the performance of configurable software systems.

---

## Repository Structure

```
GBoosting/
├── run_experiments.py       Entry point for the full pipeline (experiments, analysis, outputs)
├── config.py                Configuration: edit sample sizes, parallelism (N_JOBS), etc.
├── sk_implementation.py     Scott-Knott + A12 effect size (standalone)
├── requirements.txt         (Python libraries)
│
├── data/                    Eight benchmark datasets (CSV) + loader
│   └── loader.py            
│
├── models/                  All model implementations
│   ├── boosting_framework.py  Core L2 gradient boosting loop
│   ├── svr_boost.py / ridge_boost.py / mlp_boost.py / tree_boost.py
│   ├── baselines.py         XGBoost, CART, SVR, Ridge baselines
│   └── hinnperf_wrapper.py  sklearn-compatible wrapper for HINNPerf
│
├── HINNPERF/                HINNPerf source (do not modify, see credit below)
│
├── experiments/
│   ├── rq1_accuracy.py      Trains all model variants, records metrics + timing
│   ├── rq2_sensitivity.py   (Not used anymore)  (RQ2 data lives in rq1_raw.csv)
│   ├── rq3_cost.py          Extracts timing summary from rq1_raw.csv
│   ├── val_scores.py        Validation scores of every model variant
│   └── hinnperf_pred_time.py  HINNPerf prediction time without retraining
│
├── analysis/
│   ├── rq1_analysis.py      Best-config selection, Scott-Knott
│   ├── rq2_analysis.py      M × η sensitivity analysis
│   └── rq3_analysis.py      Cost-accuracy summaries and Pareto frontiers
│
├── analysis_val/
│   ├── run_val_analysis.py  Validation-based selection
│   └── paper_final.py       Tables and figures of the paper
│
├── evaluation/
│   ├── metrics.py           MRE, MAE, RMSE, MIBRE
│   ├── statistical.py       Scott-Knott 
│   └── timing.py            timer() context manager
│
├── results/
│   ├── collector.py         Thread-safe incremental CSV writer
│   └── checkpointing.py     JSON checkpoint for crash recovery
│
├── output/
│   ├── figures/             Figure generation scripts (rq1/rq2/rq3)
│   ├── latex/               LaTeX table generation scripts (rq1/rq2/rq3)
│   ├── final/               Results of the main run (--tag final)
│   │   ├── results/         All analysis CSVs (rq1_raw.csv is the main source of raw results)
│   │   ├── latex/rq1/       LaTeX tables
│   │   └── figures/rq2,rq3/ Figures
│   ├── val/                 Validation scores and HINNPerf prediction times
│   └── paper_final/         Tables, figures and numbers used in the paper
```

---

## HINNPerf

HINNPerf is included from prior work. The original code is available at:  
<https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh>

The wrapper in `models/hinnperf_wrapper.py` makes it conform to the sklearn `fit`/`predict` interface without modifying the original source.

---

## Setup

Requires **Python 3.9.x**.

```bash
pip install -r requirements.txt
```

Follow any runtime messages, some dependencies may require additional setup steps depending on your platform.

---

## How to Run

All commands go through `run_experiments.py`. Use `--tag <name>` to isolate outputs under `output/<name>/` so runs never overwrite each other.

### Step 1 — Run the experiment

```bash
python run_experiments.py --rq1 --tag final
```

Trains all model variants across all datasets and sample sizes (in the config file). Results are written incrementally to `output/final/results/rq1_raw.csv` and checkpointed. If interrupted, re-running the same command resumes from where it stopped.

### Step 2 — Extract timing, run analysis, generate outputs

```bash
python run_experiments.py --rq3 --analyse --tables --figures --tag final
```

- `--rq3` extracts training/prediction timing from `rq1_raw.csv`
- `--analyse` runs Scott-Knott, sensitivity analysis (RQ2, no separate experiment needed, data is already in `rq1_raw.csv`), and cost-accuracy analysis (RQ3)
- `--tables` generates LaTeX tables into `output/final/latex/`
- `--figures` generates PDF + PNG figures into `output/final/figures/`

### Step 3 — Validation-based selection and paper outputs

For each model and scenario, the variant with the lowest median validation MRE (within each run's training sample) is selected, and its test errors are reported.

```bash
python experiments/val_scores.py
python experiments/val_scores.py --combine
python experiments/hinnperf_pred_time.py
python experiments/hinnperf_pred_time.py --combine
python analysis_val/paper_final.py
```

Outputs go to `output/val/` and `output/paper_final/`.

### The `--tag` flag

Every output file (results CSV, checkpoint, tables, figures) is written to `output/<tag>/`. Using the same tag across both steps keeps all outputs in the same folder. Using a different tag gives a fully isolated run.

### Other useful flags

| Flag | Effect |
|------|--------|
| `--sanity` | Quick check on synthetic data (run this first) |
| `--jobs N` | Override parallelism (default: 11 threads) |
