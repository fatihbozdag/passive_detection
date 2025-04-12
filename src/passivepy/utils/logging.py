"""Logging configuration for PassivePy."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    rich_console: bool = True,
) -> None:
    """Set up logging configuration for the package.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        rich_console: Whether to use rich console formatting
    """
    handlers = []
    
    if rich_console:
        console_handler = RichHandler(
            level=level,
            show_time=True,
            show_path=True,
            markup=True,
        )
        handlers.append(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    
    # Set up package logger
    logger = logging.getLogger("passivepy")
    logger.setLevel(level) 