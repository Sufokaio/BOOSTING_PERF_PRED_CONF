# main file: run experiments, analysis, tables, and figures for the GB base-learner study.
import argparse
import logging
import os
import sys

import config as cfg
from results.collector import ResultCollector
from results.checkpointing import Checkpoint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiments")


def _makedirs():
    for d in [cfg.RESULTS_DIR, cfg.LATEX_DIR, cfg.FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def _check_config():
    if not cfg.DATASETS:
        logger.warning(
            "DATASETS is empty in config.py. "
            "RQ1/RQ2 will not run until you populate it."
        )
    if not cfg.SAMPLE_SIZES:
        logger.warning(
            "SAMPLE_SIZES is empty in config.py. "
            "RQ1/RQ2 will not run until you populate it."
        )


def run_sanity():
    logger.info("Running sanity checks via pytest …")
    import subprocess
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if ret.returncode != 0:
        logger.error("Sanity checks FAILED. Fix issues before running full experiment.")
        sys.exit(1)
    logger.info("Sanity checks PASSED.")


def run_rq1(collector, checkpoint, config):
    from experiments.rq1_accuracy import run_rq1 as _run
    _run(collector, checkpoint, config)


def run_rq2(collector, checkpoint, config):
    from experiments.rq2_sensitivity import run_rq2 as _run
    _run(collector, checkpoint, config)


def run_rq3(config):
    from experiments.rq3_cost import run_rq3 as _run
    _run(config)


def run_analyse(config):
    from analysis.rq1_analysis import analyse_rq1
    from analysis.rq2_analysis import analyse_rq2
    from analysis.rq3_analysis import analyse_rq3

    logger.info("Running RQ1 analysis …")
    try:
        analyse_rq1(config=config)
    except FileNotFoundError as e:
        logger.warning("RQ1 analysis skipped: %s", e)

    logger.info("Running RQ2 analysis …")
    try:
        analyse_rq2(config=config)
    except Exception as e:
        logger.exception("RQ2 analysis failed — full traceback:")

    logger.info("Running RQ3 analysis …")
    try:
        analyse_rq3(config=config)
    except FileNotFoundError as e:
        logger.warning("RQ3 analysis skipped: %s", e)


def run_tables(config):
    from output.latex.rq1_tables import generate_rq1_tables
    from output.latex.rq2_tables import generate_rq2_tables
    from output.latex.rq3_tables import generate_rq3_tables

    for fn, name in [
        (generate_rq1_tables, "RQ1"),
        (generate_rq2_tables, "RQ2"),
        (generate_rq3_tables, "RQ3"),
    ]:
        logger.info("Generating %s LaTeX tables …", name)
        try:
            fn(config)
        except Exception as e:
            logger.exception("%s table generation failed — full traceback:", name)


def run_figures(config):
    from output.figures.rq1_figures import generate_rq1_figures
    from output.figures.rq2_figures import generate_rq2_figures
    from output.figures.rq3_figures import generate_rq3_figures

    for fn, name in [
        (generate_rq1_figures, "RQ1"),
        (generate_rq2_figures, "RQ2"),
        (generate_rq3_figures, "RQ3"),
    ]:
        logger.info("Generating %s figures …", name)
        try:
            fn(config)
        except Exception as e:
            logger.warning("%s figure generation failed: %s", name, e)


def main():
    parser = argparse.ArgumentParser(description="Gradient Boosting Study")
    parser.add_argument("--rq1",     action="store_true", help="Run RQ1 experiment")
    parser.add_argument("--rq2",     action="store_true", help="Run RQ2 experiment") # Not Used
    parser.add_argument("--rq3",     action="store_true", help="Extract RQ3 timing from RQ1 results")
    parser.add_argument("--analyse", action="store_true", help="Run all statistical analysis")
    parser.add_argument("--tables",  action="store_true", help="Generate all LaTeX tables")
    parser.add_argument("--figures", action="store_true", help="Generate all figures")
    parser.add_argument("--all",     action="store_true", help="Run everything in order")
    parser.add_argument("--sanity",  action="store_true", help="Run sanity checks only")
    parser.add_argument("--jobs",    type=int, default=None, help="Override N_JOBS from config")
    parser.add_argument(
        "--tag", type=str, default=None,
        help=(
            "Experiment tag — redirects ALL output (results, figures, latex, checkpoint) "
            "to output/<tag>/. Use a different tag per experiment so runs never mix. "
            "Example: --tag paper_full  --tag test_apache"
        ),
    )
    args = parser.parse_args()

    if args.tag is not None:
        base = os.path.join("output", args.tag)
        cfg.RESULTS_DIR     = os.path.join(base, "results", "")
        cfg.LATEX_DIR       = os.path.join(base, "latex", "")
        cfg.FIGURES_DIR     = os.path.join(base, "figures", "")
        cfg.CHECKPOINT_FILE = os.path.join(base, "checkpoint.json")
        logger.info("Tag '%s' — all output under: %s", args.tag, base)

    if args.jobs is not None:
        cfg.N_JOBS = args.jobs
        logger.info("N_JOBS overridden to %d", cfg.N_JOBS)

    _makedirs()
    _check_config()

    if args.sanity:
        run_sanity()
        return

    collector  = ResultCollector()
    checkpoint = Checkpoint(cfg.CHECKPOINT_FILE)

    rq1_csv = os.path.join(cfg.RESULTS_DIR, "rq1_raw.csv")
    if os.path.exists(rq1_csv):
        collector.load(rq1_csv)
        for (dataset, model, sample_size, run) in collector.get_completed_runs():
            checkpoint.mark_done(dataset, model, sample_size, run)

    do_all = args.all

    ckpt_before = len(checkpoint)

    if do_all or args.rq1:
        logger.info("=== RQ1: Accuracy ===")
        run_rq1(collector, checkpoint, cfg)

    if do_all or args.rq2:
        logger.info("=== RQ2: Sensitivity (no separate experiment — data in rq1_raw.csv) ===")
        run_rq2(None, None, cfg)

    if do_all or args.rq3:
        logger.info("=== RQ3: Computational Cost ===")
        run_rq3(cfg)

    if do_all or args.analyse:
        logger.info("=== Analysis ===")
        run_analyse(cfg)

    if do_all or args.tables:
        logger.info("=== LaTeX Tables ===")
        run_tables(cfg)

    if do_all or args.figures:
        logger.info("=== Figures ===")
        run_figures(cfg)

    ckpt_after      = len(checkpoint)
    newly_completed = ckpt_after - ckpt_before
    skipped         = ckpt_before

    import pandas as pd
    failed = 0
    if os.path.exists(rq1_csv):
        try:
            df_out = pd.read_csv(rq1_csv)
            failed = int(df_out["value"].isna().sum() // 4)
        except Exception:
            pass

    attempted = newly_completed + failed
    logger.info(
        "=== Session Summary ===\n"
        "  Attempted this session : %d\n"
        "  Completed              : %d\n"
        "  Skipped (already done) : %d\n"
        "  Failed (NaN recorded)  : %d\n"
        "  Total in checkpoint    : %d",
        attempted, newly_completed, skipped, failed, ckpt_after,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
