"""
ML Scheduler Daemon

Runs continuous train -> predict cycles inside the ml container so the
pipeline keeps itself fresh while the Docker stack is online. Retrain
relevance is delegated to the existing age-based logic:

  - model_trainer.run_model_training() trains untrained tickers first,
    then retrains tickers whose models are older than TRAIN_MAX_AGE_DAYS,
    and skips everything that is fresh.
  - price_predictor.run_predictions() regenerates predictions older than
    PREDICT_MAX_AGE_DAYS and skips fresh ones.

Because both entry points are no-ops for fresh tickers, running the cycle
frequently is cheap: a cycle where nothing is stale finishes in seconds.

Configuration (environment variables, all optional):

  SCHEDULER_INTERVAL_HOURS   Hours between cycle starts (default: 6)
  TRAIN_MAX_AGE_DAYS         Retrain models older than this (default: 30)
  PREDICT_MAX_AGE_DAYS       Regenerate predictions older than this (default: 1)
  MAX_STOCKS_PER_CYCLE       Cap tickers trained per cycle; unset = no cap.
                             Useful to spread a large backlog across cycles
                             instead of one multi-day marathon.
  SCHEDULER_RUN_ON_START     Run a cycle immediately on boot (default: true)

Run manually:      python ml_scheduler.py
Run via compose:   see the ml-scheduler service in docker-compose.yml

The daemon handles SIGTERM/SIGINT so `docker compose stop` shuts it down
cleanly between phases instead of killing a cycle mid-write.
"""
import logging
import os
import signal
import threading
import time
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ml_scheduler")

# Event used both for graceful shutdown and as an interruptible sleep
_shutdown = threading.Event()


def _handle_signal(signum, frame):  # noqa: ARG001 (frame required by signal API)
    logger.info("Received signal %s — shutting down after current phase.", signum)
    _shutdown.set()


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def run_cycle(train_max_age_days, predict_max_age_days, max_stocks):
    """Run one train -> predict cycle. Each phase is isolated so a failure
    in one does not prevent the other, and no failure kills the daemon."""

    # --- Phase 1: training (includes the retrain-relevance check) ---
    if not _shutdown.is_set():
        logger.info("Phase 1/2: model training (max age %sd, max stocks %s)",
                    train_max_age_days, max_stocks if max_stocks else "unlimited")
        try:
            from model_trainer import run_model_training
            summary = run_model_training(
                max_model_age_days=train_max_age_days,
                max_stocks=max_stocks,
            )
            logger.info("Training phase finished: %s", summary)
        except Exception:  # daemon must survive any phase failure
            logger.exception("Training phase failed — continuing to prediction phase.")

    # --- Phase 2: predictions ---
    if not _shutdown.is_set():
        logger.info("Phase 2/2: predictions (max age %sd)", predict_max_age_days)
        try:
            from price_predictor import run_predictions
            summary = run_predictions(
                max_prediction_age_days=predict_max_age_days,
            )
            logger.info("Prediction phase finished: %s", summary)
        except Exception:
            logger.exception("Prediction phase failed — will retry next cycle.")


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    interval_hours = _env_int("SCHEDULER_INTERVAL_HOURS", 6)
    train_max_age_days = _env_int("TRAIN_MAX_AGE_DAYS", 30)
    predict_max_age_days = _env_int("PREDICT_MAX_AGE_DAYS", 1)
    max_stocks = _env_int("MAX_STOCKS_PER_CYCLE", 0) or None
    run_on_start = _env_bool("SCHEDULER_RUN_ON_START", True)

    logger.info(
        "ML scheduler starting: every %sh | retrain models > %sd old | "
        "refresh predictions > %sd old | max stocks/cycle: %s",
        interval_hours, train_max_age_days, predict_max_age_days,
        max_stocks if max_stocks else "unlimited",
    )

    first = True
    while not _shutdown.is_set():
        if first and not run_on_start:
            first = False
        else:
            first = False
            started = datetime.now()
            logger.info("=" * 60)
            logger.info("Cycle starting at %s", started.strftime("%Y-%m-%d %H:%M:%S"))
            run_cycle(train_max_age_days, predict_max_age_days, max_stocks)
            elapsed = datetime.now() - started
            logger.info("Cycle finished in %s", str(elapsed).split(".", maxsplit=1)[0])

        if _shutdown.is_set():
            break

        next_run = datetime.now() + timedelta(hours=interval_hours)
        logger.info("Sleeping until ~%s (interval %sh). Ctrl+C / compose stop to exit.",
                    next_run.strftime("%Y-%m-%d %H:%M"), interval_hours)
        # Interruptible sleep: wakes immediately on shutdown signal
        _shutdown.wait(timeout=interval_hours * 3600)

    logger.info("ML scheduler stopped cleanly.")


if __name__ == "__main__":
    main()
