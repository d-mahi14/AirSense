"""Training-only temporal CV, full-data refits and validation-only selection.
Run from project root: python -m scripts.tune_models
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import tune_all


def main():
    return tune_all()


if __name__ == "__main__":
    main()
