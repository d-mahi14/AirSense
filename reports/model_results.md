> **HISTORICAL / SUPERSEDED:** The material below describes the earlier random-split experiment and is preserved as user work. It is not the final submission result. Use `final_results.csv`, `evaluation_results.json`, `model_comparison.md` and `research_paper.pdf` for the verified chronological experiment. The original file is also preserved in `reports/archive/pre_final_audit/`.

# AirSense — Empirical Model Results & Comparative Evaluation

**Project:** AirSense — Ambient Air Quality Classification Using Machine Learning  
**Dataset:** CPCB India Ambient Air Quality (`city_day.csv`), 24,850 labeled daily records  
**Evaluation Splits:** Stratified 70% Train (17,394), 15% Validation (3,728), 15% Test (3,728)  
**Models Evaluated:** Decision Tree · Random Forest · Support Vector Machine (SVM) · XGBoost  
**Target Variable:** `AQI_Bucket` (6-class ordinal CPCB scale: Good, Satisfactory, Moderate, Poor, Very Poor, Severe)  
**Artifact Directory:** `models/` (`.joblib` pipelines and `_config.json` manifests)

---

## 1. Executive Summary & Benchmark Comparison

All four models were trained under strict train/test isolation guarantees: imputation medians, robust scaling quantiles, and ordinal encoders were fit exclusively on the 70% training split. Hyperparameters were tuned on the validation split, and generalization performance was verified on the untouched held-out 15% test split.

### Overall Performance Matrix

| Model | Val Accuracy | Test Accuracy | Val Macro F1 | Test Macro F1 | Weighted F1 | Train Time (s) | Best Config / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **0.8090** | **0.8082** | **0.7868** | **0.7816** | **0.8074** | 4.69s | `max_depth=6, lr=0.08, n_estimators=400` (stopped at 264) |
| **Random Forest** | 0.7881 | 0.7881 | 0.7733 | 0.7654 | 0.7903 | 2.95s | `n_estimators=200, max_features=sqrt, oob_score=0.7944` |
| **Decision Tree** | 0.7465 | 0.7454 | 0.7301 | 0.7301 | 0.7490 | **0.26s** | `max_depth=10, min_samples_leaf=10, class_weight=balanced` |
| **SVM (RBF Kernel)** | 0.7438 | 0.7361 | 0.7208 | 0.7135 | 0.7410 | 8.10s | `C=2.0, gamma=scale, kernel=rbf, robust_scaled=True` |

### Key Findings:
1. **XGBoost is the Top Performer:** Achieved the highest overall Test Accuracy (**80.82%**) and Test Macro F1 (**0.7816**). It demonstrates superior capability in handling non-linear pollutant interactions and boundary thresholding without overfitting (generalization gap between Val and Test Accuracy is less than 0.08%).
2. **Random Forest Provides Robust Generalization:** Achieved **78.81%** accuracy with an Out-of-Bag (OOB) score of **0.7944**, confirming that bagging effectively stabilizes variance across noisy sensor recordings.
3. **Decision Tree Serves as an Agile Baseline:** Trains in a fraction of a second (0.26s) and reaches **74.54%** test accuracy. Its decision rules closely emulate environmental regulatory threshold trees.
4. **SVM Demonstrates Strong Separability but Scalability Constraints:** SVM with an RBF kernel reaches **73.61%** test accuracy. However, training time scales superlinearly ($\mathcal{O}(n^2)$ to $\mathcal{O}(n^3)$), requiring robust feature scaling and imputation.

---

## 2. Per-Class Performance Breakdown

Air quality datasets exhibit severe natural class imbalance: moderate and satisfactory conditions dominate the year, while clean episodes ("Good", 5.4% support) and hazardous smog episodes ("Severe", 5.4% support) represent rare events. Evaluating per-class metrics is therefore critical.

### 2.1 XGBoost Classifier (Champion Model)

```text
              precision    recall  f1-score   support
        Good     0.8462    0.7114    0.7730       201
Satisfactory     0.8232    0.8752    0.8484      1234
    Moderate     0.8310    0.8210    0.8260      1324
        Poor     0.7065    0.6811    0.6935       417
   Very Poor     0.7578    0.7578    0.7578       351
      Severe     0.8226    0.7612    0.7907       201

    accuracy                         0.8082      3728
   macro avg     0.7979    0.7680    0.7816      3728
weighted avg     0.8080    0.8082    0.8074      3728
```

- **Analysis:** XGBoost achieved balanced precision and recall across all tiers. Most notably, precision on the rare "Good" category was **84.62%**, and precision on "Severe" was **82.26%**, ensuring minimal false alarms for public health alerts while maintaining dependable hazardous event detection.

### 2.2 Random Forest Classifier

```text
              precision    recall  f1-score   support
        Good     0.6562    0.8358    0.7352       201
Satisfactory     0.8384    0.8282    0.8333      1234
    Moderate     0.8478    0.7696    0.8068      1324
        Poor     0.6058    0.7554    0.6724       417
   Very Poor     0.7699    0.7151    0.7415       351
      Severe     0.7951    0.8109    0.8030       201

    accuracy                         0.7881      3728
   macro avg     0.7522    0.7859    0.7654      3728
weighted avg     0.7971    0.7881    0.7903      3728
```

- **Analysis:** Random Forest achieves high recall across extreme buckets (**83.58%** on Good, **81.09%** on Severe). However, its precision on "Good" (65.62%) and "Poor" (60.58%) is lower than XGBoost due to ensemble averaging smoothing out steep boundary shifts.

### 2.3 Decision Tree Classifier (Pruned Baseline)

```text
              precision    recall  f1-score   support
        Good     0.5348    0.8408    0.6538       201
Satisfactory     0.7980    0.7650    0.7811      1234
    Moderate     0.8228    0.7085    0.7614      1324
        Poor     0.5709    0.7530    0.6494       417
   Very Poor     0.7623    0.7037    0.7319       351
      Severe     0.7767    0.8308    0.8029       201

    accuracy                         0.7454      3728
   macro avg     0.7109    0.7670    0.7301      3728
weighted avg     0.7627    0.7454    0.7490      3728
```

- **Analysis:** With `class_weight='balanced'`, the single tree achieves strong recall on the tails (84.08% Good, 83.08% Severe), but suffers from lower precision (53.48% Good) due to step-function approximations.

### 2.4 Support Vector Machine (RBF Kernel)

```text
              precision    recall  f1-score   support
        Good     0.4808    0.8706    0.6195       201
Satisfactory     0.7936    0.7415    0.7667      1234
    Moderate     0.8264    0.7190    0.7690      1324
        Poor     0.5772    0.7170    0.6396       417
   Very Poor     0.7310    0.7123    0.7215       351
      Severe     0.7688    0.7612    0.7650       201

    accuracy                         0.7361      3728
   macro avg     0.6963    0.7536    0.7135      3728
weighted avg     0.7569    0.7361    0.7410      3728
```

- **Analysis:** SVM exhibits the highest recall on "Good" (**87.06%**), but at the expense of lowest precision (**48.08%**), frequently classifying clean Satisfactory days as Good in the high-dimensional margin boundary.

---

## 3. Confusion Matrix & Error Topology

The confusion matrices reveal an essential characteristic of the AirSense classification problem: **errors are overwhelmingly off-by-one adjacent category confusions**. Discontinuous catastrophic misclassifications (e.g., predicting "Good" when air is "Severe") are near zero.

### Confusion Matrices (Test Set, N = 3,728)

#### XGBoost Confusion Matrix:
```
Pred →    Good   Satisfactory   Moderate   Poor   Very Poor   Severe
True ↓
Good       143         56           2       0          0        0
Satisfact   26       1080         127       1          0        0
Moderate     0        174        1087      56          4        3
Poor         0          2          90     284         40        1
Very Poor    0          0           1      55        266       29
Severe       0          0           1       6         41      153
```

#### Random Forest Confusion Matrix:
```
Pred →    Good   Satisfactory   Moderate   Poor   Very Poor   Severe
True ↓
Good       168         30           3       0          0        0
Satisfact   84       1022         125       3          0        0
Moderate     4        165        1019     129          3        4
Poor         0          2          53     315         45        2
Very Poor    0          0           0      64        251       36
Severe       0          0           2       9         27      163
```

### Topological Observations:
1. **Tridiagonal Dominance:** Over 98.5% of all observations fall on the diagonal or directly adjacent off-diagonals (e.g., Moderate confused with Satisfactory or Poor). This confirms the ordinal validity of the learned decision boundaries.
2. **CPCB Breakpoint Ambiguity:** Misclassifications cluster tightly around statutory threshold transitions:
   - AQI = 50 boundary (Good / Satisfactory)
   - AQI = 100 boundary (Satisfactory / Moderate)
   - AQI = 200 boundary (Moderate / Poor)
   - AQI = 300 boundary (Poor / Very Poor)
   - AQI = 400 boundary (Very Poor / Severe)
3. **Absence of Catastrophic Inversion:** Across all 3,728 test samples in XGBoost, 0 Severe samples were predicted as Good or Satisfactory, and 0 Good samples were predicted as Poor, Very Poor, or Severe.

---

## 4. Feature Importance & Interpretability

Feature importance analysis across the tree-based ensembles reveals remarkable physical consistency with environmental chemistry and the CPCB National Air Quality Index (NAQI) sub-index mechanism:

### Relative Feature Importances

| Feature | Decision Tree | Random Forest | XGBoost | Environmental / Regulatory Role |
| :--- | :---: | :---: | :---: | :--- |
| **`PM2.5`** | **34.33%** | **29.01%** | **27.58%** | Fine particulate matter; primary determinant of hazardous AQI in urban India |
| **`PM10`** | **32.16%** | **17.43%** | **20.35%** | Coarse inhalable dust; dominant during pre-monsoon & arid dry seasons |
| **`CO`** | **22.25%** | **14.87%** | **12.36%** | Carbon monoxide; key traffic & incomplete combustion tracer |
| **`NO` / `NO2`** | 2.50% | 8.85% | 6.84% | Nitrogen oxides; photochemical ozone precursors & vehicular indicators |
| **`O3`** | 2.68% | 4.53% | 4.14% | Ground-level ozone; secondary photochemical pollutant prominent in summer |
| **`Year` / `Month`** | 1.85% | 7.42% | 8.21% | Captures seasonal winter inversions and long-term multi-year trends |
| **`Season`** | 0.82% | 3.12% | 3.45% | Categorical seasonal regime (Monsoon washout vs Winter inversion) |
| **VOCs (`Benzene`, `Toluene`)**| 1.20% | 4.77% | 4.07% | Industrial & evaporative fuel emissions |

### Key Environmental Insights:
- **Particulate Dominance:** Combined, `PM2.5` and `PM10` account for **48% to 66%** of total model decision weight. This mirrors empirical EPA and CPCB observations where particulate matter is the critical driver exceeding permissible exposure limits.
- **CO as Synergistic Indicator:** Carbon monoxide represents ~12–22% of feature importance, acting as an effective differentiator during stagnant winter atmospheric inversions.
- **Seasonality and Inversion Dynamics:** Temporal and seasonal features provide significant secondary splitting power, reflecting the severe winter pollution spikes characteristic of the Indo-Gangetic Plain.

---

## 5. Latency, Computational Complexity, and Scalability

| Dimension | Decision Tree | Random Forest | SVM (RBF) | XGBoost |
| :--- | :---: | :---: | :---: | :---: |
| **Training Time (17,394 samples)** | 0.26s | 2.95s | 8.10s | 4.69s |
| **Inference Latency (per 1,000 samples)** | < 1 ms | ~ 12 ms | ~ 85 ms | ~ 4 ms |
| **Memory Footprint on Disk** | 68 KB | 49.5 MB | 2.5 MB | 6.1 MB |
| **Theoretical Training Complexity** | $\mathcal{O}(m \cdot n \log n)$ | $\mathcal{O}(B \cdot m \cdot n \log n)$ | $\mathcal{O}(n^2 \cdot m)$ to $\mathcal{O}(n^3)$ | $\mathcal{O}(K \cdot d \cdot m \cdot n)$ |
| **Parallelization Support** | Single-threaded | Embarrassingly parallel (`n_jobs=-1`) | Limited multicore speedup | Fully multithreaded tree construction |

- **Storage Consideration:** Random Forest required 49.5 MB due to storing 200 fully grown trees without depth restrictions, whereas XGBoost produced a compact 6.1 MB model offering higher predictive accuracy.

---

## 6. Recommendations & Model Selection Matrix

| Deployment Objective | Recommended Model | Justification |
| :--- | :--- | :--- |
| **Primary Production Classifier (AirSense Engine)** | **XGBoost** | Best overall accuracy (80.82%), highest macro F1 (0.7816), optimal precision on health-critical classes, compact artifact size (6.1 MB), and low inference latency. |
| **High-Throughput Ensemble Backup** | **Random Forest** | Robust out-of-the-box generalization (78.81% accuracy, 0.7944 OOB), no early-stopping tuning required, stable variance reduction. |
| **Explainable Audit & Edge Microcontroller** | **Decision Tree** | Microsecond inference, 68 KB disk footprint, human-auditable IF-THEN rules directly transparent for environmental inspectors and policymakers. |

---

## 7. Verification & Reproducibility

All models, preprocessors, and test suites are fully version-controlled and reproducible:
- **Test Suite Pass Rate:** 109 / 109 tests passing (`pytest tests/ -v`).
- **Reproducibility Guarantee:** Random seeds (`random_state=42`) fixed across train/test splits, decision tree estimators, bagging seeds, and boosting seeds.
- **Artifact Verification:** Serialized pipelines stored in `models/` with JSON parameter and evaluation manifests.
