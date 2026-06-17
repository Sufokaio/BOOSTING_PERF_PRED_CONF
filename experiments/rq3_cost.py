import logging
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)


def run_rq3(config=cfg):
    rq1_path = os.path.join(config.RESULTS_DIR, "rq1_raw.csv")
    logger.info("[RQ3] Starting — loading RQ1 results from %s", rq1_path)
    if not os.path.exists(rq1_path):
        raise FileNotFoundError(
            f"RQ1 results not found at {rq1_path}. Run RQ1 first."
        )

    from analysis.rq1_analysis import select_best_per_scenario

    df = pd.read_csv(rq1_path)

    logger.info("[RQ3] Loaded %d rows", len(df))
    nan_mask = df["value"].isna()
    n_nan = nan_mask.sum()
    if n_nan > 0:
        logger.warning("[RQ3] %d NaN metric rows excluded before timing extraction", n_nan)
    df_clean = df[~nan_mask].copy()

    df_best = select_best_per_scenario(df_clean)

    timing_cols = ["dataset", "model", "sample_size", "run", "train_time_s", "pred_time_s"]
    timing_df = df_best[timing_cols].drop_duplicates(
        subset=["dataset", "model", "sample_size", "run"]
    ).copy()

    nan_timing = timing_df["train_time_s"].isna() | timing_df["pred_time_s"].isna()
    n_nan_t = nan_timing.sum()
    if n_nan_t > 0:
        logger.warning("[RQ3] %d rows have NaN timing (crashed runs)", n_nan_t)
        timing_df["flagged"] = nan_timing
    else:
        timing_df["flagged"] = False

    summary = (
        timing_df[~timing_df["flagged"]]
        .groupby(["model", "dataset", "sample_size"])[["train_time_s", "pred_time_s"]]
        .median()
        .reset_index()
    )

    out_path = os.path.join(config.RESULTS_DIR, "rq3_timing.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("[RQ3] Timing summary saved to %s (%d rows)", out_path, len(summary))
    return summary
