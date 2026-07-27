"""Project path helpers resolved relative to the repository root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"


def ensure_directories() -> None:
    """Create standard project directories if they do not exist."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
