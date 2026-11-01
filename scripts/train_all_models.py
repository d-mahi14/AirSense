"""Fit four baselines and measure validation only; never evaluate the test split.
Run from project root: python -m scripts.train_all_models
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import fit_baselines


def main():
    return fit_baselines()


if __name__ == "__main__":
    main()
