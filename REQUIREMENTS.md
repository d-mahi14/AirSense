# AirSense — Technical Requirements & Environment Setup

**Project:** AirSense — Air Quality Prediction and Analysis Using Machine Learning  
**Workspace:** `C:\dev\AirSense\`  
**Date:** October 2026  

---

## 1. System & Environment Specifications

- **Operating System:** Windows 10/11 (Architecture: x64)
- **Command Shell:** Windows PowerShell
- **Base Python Runtime:** Python 3.11.14 (Verified active in workspace)
- **Path Format:** Windows backward/forward slash compatible (`data/raw/...` or `data\raw\...`)

---

## 2. Python Package Inventory & Status

The following packages constitute the required machine learning, visualization, and application stack for the project. An empirical inspection of the current environment reveals which dependencies are already present and which require installation:

| Package Category | Library | Minimum Recommended Version | Current Environment Status | Purpose in AirSense |
| :--- | :--- | :--- | :--- | :--- |
| **Data Processing** | `numpy` | `>= 1.24.0` | **Installed** (v2.3.3) | Numerical array computations, vectorization |
| **Data Processing** | `pandas` | `>= 2.0.0` | **Installed** (v2.3.3) | Tabular data manipulation, time-series operations |
| **Scientific Computing** | `scipy` | `>= 1.10.0` | **Installed** (v1.17.1) | Statistical distributions, hypothesis testing |
| **Data Visualization** | `matplotlib` | `>= 3.7.0` | **Installed** (v3.11.2) | Static publication-quality charts & histograms |
| **Data Visualization** | `seaborn` | `>= 0.12.0` | **Installed** (v0.13.2) | Statistical visualizations, heatmaps, boxplots |
| **Machine Learning** | `scikit-learn` | `>= 1.3.0` | **NOT INSTALLED** | Decision Trees, SVM, Random Forest, metrics, splitters |
| **Machine Learning** | `xgboost` | `>= 2.0.0` | **NOT INSTALLED** | Extreme Gradient Boosting classifier |
| **Explainability** | `shap` | `>= 0.43.0` | **NOT INSTALLED** | Shapley Additive exPlanations feature attribution |
| **Web Dashboard** | `streamlit` | `>= 1.30.0` | **NOT INSTALLED** | Interactive web application and UI |
| **Model Persistence** | `joblib` | `>= 1.3.0` | Installed as part of standard scientific stack / to be verified | Serialization of trained models and scalers |

---

## 3. Installation Instructions

To install all required packages into the current Python environment, execute the following PowerShell command from the project root (`C:\dev\AirSense`):

```powershell
pip install scikit-learn xgboost shap streamlit joblib
```

Or install using the provided `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

## 4. Hardware & Resource Considerations

- **Memory (RAM):**
  - Minimum: 8 GB RAM.
  - Recommended: 16 GB RAM.
  - *Note on SVM Computational Complexity:* Training kernel SVM has computational complexity of $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$ with respect to sample size $N$. Training SVM directly on 2.5 million rows (`station_hour.csv`) would cause memory exhaustion and excessive training times. The primary benchmark on `city_day.csv` (~24,850 valid rows) runs smoothly within standard consumer hardware limits (< 1 minute per fold).
- **Disk Space:**
  - Raw Dataset: ~288.8 MB across 5 CSV files.
  - Processed Datasets & Models: ~100 MB.
  - Total workspace footprint is under 500 MB.

---

## 5. Directory Structure Specification

The project follows a clean, modular structure:

```text
AirSense/
├── data/
│   ├── raw/
│   │   └── archive (3)/          # READ-ONLY raw data from Kaggle (5 CSV files)
│   └── processed/                # Cleaned, imputed, and split datasets
├── docs/
│   ├── IMPLEMENTATION_LOG.md     # Stage-by-stage implementation history
│   ├── AI_PROMPT_LOG.md          # Exact user prompts and responses
│   └── AI_FAILURE_LOG.md         # Errors, edge cases, and resolutions
├── models/                       # Saved trained model artifacts (.joblib/.pkl)
├── notebooks/                    # Optional student exploratory notebooks
├── reports/
│   ├── dataset_requirements.md   # Dataset audit report
│   ├── figures/                  # Generated plots and evaluation charts
│   └── research_report.md        # Academic synthesis report
├── src/
│   ├── __init__.py
│   ├── data/                     # Loading, cleaning, imputation, splitting
│   ├── features/                 # Feature extraction & encoding
│   ├── models/                   # DT, RF, SVM, XGBoost training & tuning
│   ├── evaluation/               # Metrics, confusion matrix, SHAP utilities
│   └── utils/                    # Helper functions and constants
├── app.py                        # Streamlit web dashboard application
├── PROJECT_PLAN.md               # Architecture and project plan
├── REQUIREMENTS.md               # Technical requirements (this file)
├── IMPLEMENTATION_ROADMAP.md     # Step-by-step roadmap
└── requirements.txt              # Pip dependency specification
```

