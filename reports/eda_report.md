# AirSense — Exploratory Data Analysis (EDA) & Data Cleaning Report

**Project:** AirSense — Air Quality Prediction and Analysis Using Machine Learning  
**Primary Dataset:** `city_day.csv` (India Air Quality, 2015–2020)  
**Pipeline Script:** `scripts/run_eda.py`  
**Status:** Complete empirical analysis  

---

## 1. Executive Summary of Findings

This report documents the empirical properties, distributions, anomalies, and structural characteristics of the primary `city_day.csv` dataset. Key insights include:

1. **Clean Target Subsetting:** From the initial **29,531 raw records**, exactly **24,850 records** contain ground-truth `AQI_Bucket` and `AQI` labels. The 4,681 records missing target labels (15.85%) were strictly segregated rather than imputed, honoring regulatory data-integrity principles.
2. **Zero Duplicate or Negative Records:** Across all 24,850 observations and 12 criteria pollutants, there are **0 full duplicate records**, **0 `(City, Date)` key collisions**, and **0 negative concentration values**.
3. **Severe Class Imbalance:** `Moderate` (35.53%) and `Satisfactory` (33.09%) comprise **68.62%** of all valid days, while the critical public health warning tiers—`Good` (5.40%) and `Severe` (5.38%)—are rare minority classes. Standard accuracy will produce optimistic bias; multi-class Macro-F1 and balanced weighting are mandatory.
4. **High Right-Skew in Particulate Matter:** Criteria pollutants ($PM_{2.5}, PM_{10}, NO_2, CO$) exhibit extreme positive skewness ($\gamma_1 > 2.0$), driven by episodic northern winter smog crises and harvest residue burning where $PM_{2.5}$ surges up to $949.99\,\mu	ext{g}/	ext{m}^3$.
5. **Strong Collinearity with Particulate Matter:** Continuous AQI correlates most strongly with $PM_{2.5}$ ($r = 0.66$) and $PM_{10}$ ($r = 0.80$), confirming that particulate matter is the predominant driver of air quality degradation in Indian cities.
6. **Pronounced Seasonality:** AQI exhibits a strong bimodal seasonal cycle across all 26 cities: maximum peak severity occurs in **Winter** (median AQI: 168.0), followed by Post-Monsoon (median: 140.0), while **Monsoon** brings major wash-out cleansing (median AQI: 76.0, drop of over 54%).

---

## 2. Missing-Value Audit & Cleaning Decisions

![Missing Values Chart](figures/eda/missing_values.png)

### Empirical Missingness Profile:
| Feature | Raw Missing Count | Raw Missing % | Labeled Missing Count | Labeled Missing % | Imputation Strategy / Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `City` | 0 | 0.00% | 0 | 0.00% | Low sparsity: Impute with training City-Month median |
| `Date` | 0 | 0.00% | 0 | 0.00% | Low sparsity: Impute with training City-Month median |
| `PM2.5` | 4,598 | 15.57% | 678 | 2.73% | Low sparsity: Impute with training City-Month median |
| `PM10` | 11,140 | 37.72% | 7,086 | 28.52% | Moderate sparsity: Impute with training City median, global fallback |
| `NO` | 3,582 | 12.13% | 387 | 1.56% | Low sparsity: Impute with training City-Month median |
| `NO2` | 3,585 | 12.14% | 391 | 1.57% | Low sparsity: Impute with training City-Month median |
| `NOx` | 4,185 | 14.17% | 1,857 | 7.47% | Moderate sparsity: Impute with training City median, global fallback |
| `NH3` | 10,328 | 34.97% | 6,536 | 26.30% | Moderate sparsity: Impute with training City median, global fallback |
| `CO` | 2,059 | 6.97% | 445 | 1.79% | Low sparsity: Impute with training City-Month median |
| `SO2` | 3,854 | 13.05% | 605 | 2.43% | Low sparsity: Impute with training City-Month median |
| `O3` | 4,022 | 13.62% | 807 | 3.25% | Low sparsity: Impute with training City-Month median |
| `Benzene` | 5,623 | 19.04% | 3,535 | 14.23% | Moderate sparsity: Impute with training City median, global fallback |
| `Toluene` | 8,041 | 27.23% | 5,826 | 23.44% | Moderate sparsity: Impute with training City median, global fallback |
| `Xylene` | 18,109 | 61.32% | 15,372 | 61.86% | High sparsity: Evaluate model performance with and without feature |
| `AQI` | 4,681 | 15.85% | 0 | 0.00% | Continuous target: Excluded from input features to avoid leakage |
| `AQI_Bucket` | 4,681 | 15.85% | 0 | 0.00% | Target label: Drop missing 4,681 rows (NO imputation) |

### Critical Preprocessing Principles:
- **No Imputation of Target:** Missing target labels are never fabricated. Under CPCB rules, if fewer than 3 pollutants or neither PM2.5/PM10 are monitored, AQI is legally undefined.
- **Target Leakage Prohibition:** `AQI` is mathematically computed from pollutant sub-indices. It is strictly excluded from predictor matrix $X$ when classifying `AQI_Bucket`.
- **Fold Segregation:** Median imputation tables are computed strictly on `train_df` and mapped to `val_df` and `test_df` without look-ahead leakage.

---

## 3. Pollutant Distributions & Summary Statistics

![Pollutant Distributions](figures/eda/pollutant_distributions.png)

### Statistical Distribution of Features (Labeled Dataset, $N = 24,850$):
| Feature | Valid $N$ | Mean | Std | Min | Median | Max | Skewness | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PM2.5` | 24,172 | 67.48 | 63.08 | 0.04 | 48.78 | 914.94 | 2.96 | Extreme right-skew |
| `PM10` | 17,764 | 118.45 | 89.49 | 0.03 | 96.18 | 917.08 | 1.93 | Moderate skew |
| `NO` | 24,463 | 17.62 | 22.42 | 0.03 | 9.91 | 390.68 | 3.52 | Extreme right-skew |
| `NO2` | 24,459 | 28.98 | 24.63 | 0.01 | 22.10 | 362.21 | 2.44 | Extreme right-skew |
| `NOx` | 22,993 | 32.29 | 30.71 | 0.00 | 23.68 | 378.24 | 2.40 | Extreme right-skew |
| `NH3` | 18,314 | 23.85 | 25.88 | 0.01 | 16.31 | 352.89 | 4.11 | Extreme right-skew |
| `CO` | 24,405 | 2.35 | 7.08 | 0.00 | 0.93 | 175.81 | 8.63 | Extreme right-skew |
| `SO2` | 24,245 | 14.36 | 17.43 | 0.01 | 9.22 | 186.08 | 3.96 | Extreme right-skew |
| `O3` | 24,043 | 34.91 | 21.72 | 0.01 | 31.25 | 257.73 | 1.33 | Moderate skew |
| `Benzene` | 21,315 | 3.46 | 16.04 | 0.00 | 1.29 | 455.03 | 21.58 | Extreme right-skew |
| `Toluene` | 19,024 | 9.53 | 20.88 | 0.00 | 3.58 | 454.85 | 11.36 | Extreme right-skew |
| `Xylene` | 9,478 | 3.59 | 6.75 | 0.00 | 1.42 | 170.37 | 7.53 | Extreme right-skew |
| `AQI` | 24,850 | 166.46 | 140.70 | 13.00 | 118.00 | 2049.00 | 3.40 | Extreme right-skew |

---

## 4. Target Category Analysis & Class Imbalance

![Target Distribution](figures/eda/aqi_category_distribution.png)

### Ground-Truth Distribution of `AQI_Bucket`:
| Category | Daily Count | Percentage (Valid) | Imbalance Ratio vs Majority | CPCB Numeric AQI Range | Public Health Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Good** | 1,341 | 5.40% | 1 : 6.6 | 0 – 50 | Minimal impact |
| **Satisfactory** | 8,224 | 33.09% | 1 : 1.1 | 51 – 100 | Minor breathing discomfort to sensitive people |
| **Moderate** | 8,829 | 35.53% | 1 : 1.0 | 101 – 200 | Breathing discomfort to people with asthma/heart disease |
| **Poor** | 2,781 | 11.19% | 1 : 3.2 | 201 – 300 | Breathing discomfort to most people on prolonged exposure |
| **Very Poor** | 2,337 | 9.40% | 1 : 3.8 | 301 – 400 | Respiratory illness on prolonged exposure |
| **Severe** | 1,338 | 5.38% | 1 : 6.6 | 401 – 500+ | Affects healthy people; serious impacts on diseased |

### Modeling Implications of Class Imbalance:
- The majority class (`Moderate`, $N=8,829$) outnumbers the critical `Severe` class ($N=1,338$) by **6.6 to 1**.
- Models must employ `class_weight='balanced'` in Random Forest and SVM, and calibrated objective weights in XGBoost.
- The primary evaluation metric must be **Macro F1-Score**, which weights all 6 categories equally.

---

## 5. Correlation Analysis & Feature Interactions

![Correlation Heatmap](figures/eda/correlation_heatmap.png)

### Key Correlation Observations:
1. **Strong AQI Drivers:** Particulate matter shows the highest linear correlation with overall AQI ($PM_{10}$: $r = 0.80$, $PM_{2.5}$: $r = 0.66$).
2. **Nitrogen Oxide Collinearity:** Nitric Oxide ($NO$) and Nitrogen Dioxide ($NO_2$) are strongly correlated with total oxides of nitrogen ($NO_x$, $r = 0.79$). Tree models handle this collinearity naturally, but feature importance can be split across collinear features.
3. **Volatile Organic Compounds (VOCs):** Benzene and Toluene exhibit moderate correlation ($r = 0.54$), reflecting co-emission from vehicular exhaust and industrial solvent evaporation.
4. **Secondary Ozone Photochemistry:** Ozone ($O_3$) exhibits low or negative correlations with primary vehicular emissions ($NO$, $NO_x$, $r pprox -0.10$), consistent with the titration effect where high local $NO$ concentrations deplete localized ground-level ozone.

---

## 6. City-Wise Comparisons & Spatial Disparities

![City Comparisons](figures/eda/city_comparisons.png)

### Geographic Disparities:
- **Most Polluted Cities:** Northern and Indo-Gangetic Plain cities experience the highest median AQI and highest frequency of `Severe` days:
  - **Delhi:** Median AQI = 258.0 (Poor), with **19.8% of days classified as Severe**.
  - **Ahmedabad:** Median AQI = 277.5, with **19.6% of days classified as Severe**.
  - **Patna:** Median AQI = 217.0, with **13.4% of days classified as Severe**.
  - **Gurugram:** Median AQI = 206.0, with **10.5% of days classified as Severe**.
- **Cleanest Cities:** Coastal and southern/northeastern cities maintain significantly cleaner air:
  - **Aizawl:** Median AQI = 31.0 (Good), **0.0% Severe days**.
  - **Shillong:** Median AQI = 48.0 (Good), **0.0% Severe days**.
  - **Thiruvananthapuram:** Median AQI = 68.0 (Satisfactory), **0.0% Severe days**.
  - **Bengaluru:** Median AQI = 90.0 (Satisfactory), **0.0% Severe days**.

---

## 7. Temporal Trends & Seasonal Cycles

![Temporal Trends](figures/eda/temporal_trends.png)

![Seasonal Profiles](figures/eda/seasonal_profiles.png)

### Seasonal Dynamics:
- **Winter Peak (December–February):** National median AQI surges to **168.0**. Atmospheric temperature inversions, shallow mixing heights, and low wind speeds trap emissions near ground level.
- **Monsoon Cleansing (June–September):** National median AQI plummets to **76.0** (Satisfactory tier). Heavy wet deposition (rain wash-out) rapidly scavenges particulate matter and soluble gases.
- **Post-Monsoon Biomass Burning (October–November):** National median AQI rebounds to **140.0**, characterized by sudden episodic spikes corresponding to post-harvest agricultural burning in northwestern India.

---

## 8. Verification & Next Steps

- [x] All 7 publication-quality EDA figures generated and verified in `reports/figures/eda/`.
- [x] Statistical audit computed on actual disk files without approximations.
- [x] Zero weather variables confirmed across all distributions.
- [x] Clean preprocessing pipeline validated via unit tests.
- [x] Foundation established for Stage 5: Baseline Model Training (Decision Tree, Random Forest, SVM, XGBoost).
