"""
Structured, coloured logging via `rich`.
"""
from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=_console, rich_tracebacks=True, markup=True)],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
