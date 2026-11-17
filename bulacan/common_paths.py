"""File locations shared by tide.py and common.py."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIDE_CSV = ROOT / "data/external/tide/h370.csv"
TIDE_MODEL = ROOT / "data/processed/manila_tide_harmonics.json"
