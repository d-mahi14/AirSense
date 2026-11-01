"""Compatibility entry point for the corrected chronological experiment.

Nothing runs on import. The original random-split script is preserved in
reports/archive/pre_final_audit/scripts__run_all_models.py.
Training replaces saved artifacts: back up models/ and reports/ before running.
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from src.experiment import fit_baselines, tune_all, evaluate_all
    from scripts.analyze_features import main as analyze_features
    fit_baselines()
    tune_all()
    evaluate_all()
    analyze_features()


if __name__ == "__main__":
    main()
