from __future__ import annotations
import logging
from pathlib import Path


def get_logger(
    name: str = "credit_scoring", log_dir: str = "logs", level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger that writes to stdout and to logs/credit_scoring.log.
    Subsequent calls with the same name are idempotent (no duplicate handlers).
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "credit_scoring.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger._configured = True  # type: ignore[attr-defined]
    return logger
