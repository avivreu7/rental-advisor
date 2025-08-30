import logging
from pathlib import Path

def get_logger(name: str = "app", level=logging.INFO):
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def ensure_dir(path: Path):
    """Ensure a directory exists, create if missing."""
    path.mkdir(parents=True, exist_ok=True)
    return path
