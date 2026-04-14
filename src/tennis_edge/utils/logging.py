"""Structured logging setup with custom TRADE and OPPORTUNITY levels."""

from __future__ import annotations

import logging
from pathlib import Path

# Custom log levels (following polymarket-arbitrage pattern)
TRADE = 25
OPPORTUNITY = 26
logging.addLevelName(TRADE, "TRADE")
logging.addLevelName(OPPORTUNITY, "OPPORTUNITY")


class TennisEdgeLogger(logging.Logger):
    def trade(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(TRADE):
            self._log(TRADE, msg, args, **kwargs)

    def opportunity(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(OPPORTUNITY):
            self._log(OPPORTUNITY, msg, args, **kwargs)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the application."""
    logging.setLoggerClass(TennisEdgeLogger)

    fmt = "%(asctime)s [%(levelname)-11s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(path), encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
