import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

logger = logging.getLogger(__name__)


def run_rq2(collector, checkpoint, config=cfg):
    """
Depcretated: RQ2 is now collected as part of RQ1.
    """
    print(
        "[RQ2] No separate experiment needed.\n"
        "      The full M×eta grid for all boosted models is collected by --rq1.\n"
        "      Run --analyse to produce RQ2 sensitivity outputs from rq1_raw.csv."
    )
    logger.info("[RQ2] Skipped — data already collected in RQ1 (rq1_raw.csv).")
