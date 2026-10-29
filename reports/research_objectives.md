# AirSense — Research Objectives & Project Scope

**Project:** AirSense — Air Quality Prediction and Analysis Using Machine Learning  
**Project Character:** Undergraduate / Student Machine Learning Mini-Project  
**Author:** AI Assistant (Antigravity) & Student Research Team  
**Dataset:** India Air Quality Dataset (2015–2020), Kaggle (Rohan Rao)  
**Primary Theoretical Reference:** Méndez, M., Merayo, M. G., & Núñez, M. (2023). *Machine learning algorithms to forecast air quality: a survey.* **Artificial Intelligence Review**, 56(9), 10031–10066. https://doi.org/10.1007/s10462-023-10424-4  

---

## 1. Problem Statement & Motivation

Ambient air pollution in Indian cities frequently reaches hazardous levels, posing acute and chronic cardiovascular and respiratory health risks. The Central Pollution Control Board (CPCB) classifies daily air quality into six categorical health risk tiers (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, and `Severe`).

While deep learning and sequence-to-sequence neural architectures are increasingly explored in the academic forecasting literature (as surveyed by Méndez et al., 2023), municipal agencies and environmental monitoring teams require robust, computationally tractable, interpretable, and reproducible machine learning models that can:
1. Accurately categorize air quality status from monitored criteria pollutants.
2. Maintain high sensitivity (recall) in detecting hazardous pollution spikes (`Very Poor` and `Severe`) despite severe class imbalance.
3. Provide transparent feature attributions to inform targeted regulatory actions.

---

## 2. Modest & Defensible Student-Project Contribution

### What AirSense Aims to Deliver
AirSense proposes a modest, technically disciplined, and methodologically sound contribution appropriate for an applied machine learning mini-project:

1. **Controlled, Leak-Free Algorithmic Benchmark:**
   - Implement, optimize, and systematically compare four canonical machine learning algorithms spanning three distinct paradigms:
     - **Tree-Based Rule Model:** Decision Tree (DT)
     - **Bagging Ensemble:** Random Forest (RF)
     - **Kernel-Based Margin Maximizer:** Support Vector Machine (SVM)
     - **Gradient-Boosted Tree Ensemble:** XGBoost
   - All models are evaluated on an identical, strictly segregated stratified split ($70\%$ Train, $15\%$ Validation, $15\%$ Test) using a fixed random seed (`42`).
   - Imputation and scaling parameters are fitted exclusively on training folds to prevent look-ahead bias and data leakage.
2. **Class Imbalance Mitigation & Fair Evaluation:**
   - Address severe category imbalance (`Moderate` and `Satisfactory` comprise $68.62\%$ of samples; `Good` and `Severe` each comprise only $\sim 5.4\%$) through balanced class weighting and threshold tuning.
   - Evaluate performance using balanced metrics: **Macro F1-Score**, **Balanced Accuracy**, **Per-Class Recall**, and **Multi-Class Confusion Matrices**.
3. **Multi-Pollutant Error Diagnostics & Feature Importance:**
   - Conduct detailed confusion matrix analysis to distinguish benign one-tier misclassifications from severe multi-tier failures.
   - Utilize Tree Feature Importances and post-hoc **SHAP (SHapley Additive exPlanations)** to quantify which pollutants drive classifications into hazardous tiers.
4. **Interactive Dashboard:**
   - Deploy an accessible **Streamlit web application** that allows students, educators, and citizens to explore multi-city historical trends and simulate air quality categories under varying pollutant scenarios.

### Explicit Boundary: What Is NOT Claimed
> [!IMPORTANT]
> - **No Algorithmic Novelty Claim:** AirSense does not claim to have invented a new machine learning algorithm, novel mathematical optimization technique, or new loss function. The project contributes a disciplined empirical comparison, rigorous pipeline engineering, error analysis, and interpretability study on real-world Indian CAAQMS data.
> - **No Forecasting Equivalence Claim:** In strict adherence to scientific rigor and project guidelines, **AirSense does not claim current-condition classification is future forecasting**. Diagnostic classification of time $t$ observations is clearly distinguished from multi-step ahead temporal forecasting.

---

## 3. Testable Research Hypotheses

AirSense formulates three testable empirical hypotheses:

- **Hypothesis 1 ($H_1$ — Ensemble Superiority):**
  Ensemble architectures (Random Forest and XGBoost) will achieve statistically and practically significant improvements in **Macro F1-Score** ($\ge 0.10$ absolute gain) and minority class recall on `Severe` days compared to a single depth-tuned Decision Tree, due to variance reduction and non-linear feature interaction modeling.
- **Hypothesis 2 ($H_2$ — Gradient Boosting Efficiency vs. Kernel SVM):**
  XGBoost will match or exceed the classification discrimination of kernel Support Vector Machines across high-risk categories (`Poor`, `Very Poor`, `Severe`), while requiring substantially lower training and inference latency on tabular pollutant matrices.
- **Hypothesis 3 ($H_3$ — Particulate Dominance in SHAP Attribution):**
  SHAP feature importance analysis will reveal that fine particulate matter ($PM_{2.5}$) and coarse particulate matter ($PM_{10}$), followed by combustion oxides ($NO_2$), have the highest positive marginal Shapley values in propelling days into the `Severe` category across Indian urban centers.

---

## 4. Methodological Constraints & Global Safeguards

1. **Weather Variable Absence:** The dataset contains zero meteorological variables. No weather features will be synthesized or assumed.
2. **Target Leakage Prohibition:** The numerical `AQI` column is mathematically derived from pollutant sub-indices; it is strictly excluded from feature matrix $X$ when predicting `AQI_Bucket`.
3. **Data Integrity:** Original CSV files in `data/raw/archive (3)/` remain completely read-only.
4. **Reproducibility:** A fixed random seed (`RANDOM_STATE = 42`) is applied across all splits, cross-validation folds, and stochastic estimators.

