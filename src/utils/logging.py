import logging
from typing import Optional


def setup_logging(level: str = "INFO", fmt: Optional[str] = None) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
