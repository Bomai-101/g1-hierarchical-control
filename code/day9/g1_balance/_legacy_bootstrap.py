"""Shared import setup for the retained Day 9 compatibility entrypoints."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_src_path() -> Path:
    src_dir = PROJECT_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return PROJECT_ROOT
