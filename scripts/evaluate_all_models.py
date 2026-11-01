"""Evaluate a hash-verified, frozen candidate roster on the shared final test set.
Run after training/tuning: python -m scripts.evaluate_all_models
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import evaluate_all


def main():
    return evaluate_all()


if __name__ == "__main__":
    main()
