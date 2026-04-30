"""
scheduler.py
Runs predict_batch.run_batch() automatically every 5 minutes using APScheduler.
Also supports a CRON_MODE env var to use cron-style scheduling instead.

Usage:
    python scheduler.py                  # interval mode (every 5 min)
    INTERVAL_MINUTES=10 python scheduler.py
"""

import os
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from predict_batch import run_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", 5))


def main():
    scheduler = BlockingScheduler()

    # interval-based schedule (e.g. every 5 minutes)
    scheduler.add_job(
        run_batch,
        trigger="interval",
        minutes=INTERVAL_MINUTES,
        id="batch_prediction",
        name="Batch Prediction Job",
        replace_existing=True,
    )

    log.info(f"Scheduler starting — batch job will run every {INTERVAL_MINUTES} minute(s).")
    log.info("Press Ctrl+C to stop.")

    # Run once immediately on startup, then on schedule
    run_batch()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()