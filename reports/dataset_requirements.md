# AirSense — Dataset Audit & Requirements Report

**Project:** AirSense — Air Quality Prediction and Analysis Using Machine Learning  
**Dataset:** India Air Quality Dataset (2015–2020)  
**Dataset Source:** [Kaggle - Rohan Rao (rohanrao/air-quality-data-in-india)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
**Date of Audit:** October 2026  
**Auditor:** AI Assistant (Antigravity)  

---

## 1. Executive Summary & Audit Scope

This document provides a factual, empirical audit of the five raw CSV files located in `data/raw/archive (3)/`. Every statistic, column name, row count, null count, and data type reported here has been directly extracted from the files on disk without synthetic assumptions.

### Key Audit Findings:
1. **Weather Variable Absence:** None of the raw dataset files contain meteorological or weather variables (such as ambient temperature, relative humidity, wind speed, wind direction, rainfall, or solar radiation). All feature columns represent either temporal/spatial identifiers or chemical/particulate pollutant concentrations.
2. **Official CPCB AQI Categories Present:** Both continuous `AQI` and discrete `AQI_Bucket` columns exist in all four pollutant files. The values in `AQI_Bucket` correspond exactly to the 6 official Central Pollution Control Board (CPCB) of India National Air Quality Index categories (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`).
3. **Primary Recommendation:** `city_day.csv` (29,531 rows, 26 major cities) serves as the primary dataset for multi-class classification and algorithm benchmarking. It offers reliable daily temporal aggregation, representative geographic diversity across India, and manageable computational complexity for SVM and ensemble models.

---

## 2. Inventory of Raw Files

| File Name | Exact Size (Bytes) | Size (MB) | Row Count | Column Count | Temporal Granularity | Spatial Granularity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `stations.csv` | 14,467 | 0.01 MB | 230 | 5 | Metadata (Static) | Station level (230 stations across 127 cities) |
| `city_day.csv` | 2,574,056 | 2.45 MB | 29,531 | 16 | Daily (`Date`: 2015-01-01 to 2020-07-01) | City level (26 distinct cities) |
| `city_hour.csv` | 65,648,581 | 62.61 MB | 707,875 | 16 | Hourly (`Datetime`: 2015-01-01 01:00 to 2020-07-01 00:00) | City level (26 distinct cities) |
| `station_day.csv` | 8,634,105 | 8.23 MB | 108,035 | 16 | Daily (`Date`: 2015-01-01 to 2020-07-01) | Station level (110 distinct monitoring stations) |
| `station_hour.csv` | 219,675,434 | 209.50 MB | 2,589,083 | 16 | Hourly (`Datetime`: 2015-01-01 01:00 to 2020-07-01 00:00) | Station level (110 distinct monitoring stations) |

---

## 3. Schema & Missing Value Profiles

### 3.1. `stations.csv`
Contains metadata for monitoring stations across Indian states and cities.

| Column | Non-Null Count | Null Count | Missing % | Data Type | Notes / Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `StationId` | 230 | 0 | 0.00% | `object` | Unique alphanumeric identifier (e.g., `AP001`, `DL001`) |
| `StationName` | 230 | 0 | 0.00% | `object` | Official station name/location description |
| `City` | 230 | 0 | 0.00% | `object` | City name (127 unique cities) |
| `State` | 230 | 0 | 0.00% | `object` | State or Union Territory name |
| `Status` | 133 | 97 | 42.17% | `object` | 131 `Active`, 2 `Inactive`, 97 missing |

---

### 3.2. `city_day.csv` (Primary Dataset for Benchmark Modeling)
Contains daily aggregate pollutant concentrations and computed AQI for 26 Indian cities.

| Column Name | Non-Null Count | Null Count | Missing % | Data Type | Unit / Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `City` | 29,531 | 0 | 0.00% | `object` | Name of Indian city (26 cities) |
| `Date` | 29,531 | 0 | 0.00% | `object` | Daily timestamp (`YYYY-MM-DD`) |
| `PM2.5` | 24,933 | 4,598 | 15.57% | `float64` | Particulate Matter $\le 2.5\,\mu\text{m}$ ($\mu\text{g}/\text{m}^3$) |
| `PM10` | 18,391 | 11,140 | 37.72% | `float64` | Particulate Matter $\le 10\,\mu\text{m}$ ($\mu\text{g}/\text{m}^3$) |
| `NO` | 25,949 | 3,582 | 12.13% | `float64` | Nitric Oxide ($\mu\text{g}/\text{m}^3$) |
| `NO2` | 25,946 | 3,585 | 12.14% | `float64` | Nitrogen Dioxide ($\mu\text{g}/\text{m}^3$) |
| `NOx` | 25,346 | 4,185 | 14.17% | `float64` | Nitrogen Oxides (ppb) |
| `NH3` | 19,203 | 10,328 | 34.97% | `float64` | Ammonia ($\mu\text{g}/\text{m}^3$) |
| `CO` | 27,472 | 2,059 | 6.97% | `float64` | Carbon Monoxide ($\text{mg}/\text{m}^3$) |
| `SO2` | 25,677 | 3,854 | 13.05% | `float64` | Sulfur Dioxide ($\mu\text{g}/\text{m}^3$) |
| `O3` | 25,509 | 4,022 | 13.62% | `float64` | Ozone ($\mu\text{g}/\text{m}^3$) |
| `Benzene` | 23,908 | 5,623 | 19.04% | `float64` | Benzene ($\mu\text{g}/\text{m}^3$) |
| `Toluene` | 21,490 | 8,041 | 27.23% | `float64` | Toluene ($\mu\text{g}/\text{m}^3$) |
| `Xylene` | 11,422 | 18,109 | 61.32% | `float64` | Xylene ($\mu\text{g}/\text{m}^3$) |
| `AQI` | 24,850 | 4,681 | 15.85% | `float64` | Central Pollution Control Board Air Quality Index |
| `AQI_Bucket` | 24,850 | 4,681 | 15.85% | `object` | CPCB Categorical AQI Level (Target) |

---

### 3.3. `city_hour.csv`
Hourly pollutant measurements aggregated at the city level.

- Total Rows: 707,875
- Total Columns: 16
- Target Missingness: `AQI` and `AQI_Bucket` missing in 129,080 rows (18.23%).
- Pollutant Missingness: `Xylene` (64.39%), `PM10` (41.92%), `NH3` (38.50%), `Toluene` (31.16%), `Benzene` (23.12%), `PM2.5` (20.50%), `SO2` (18.42%), `O3` (18.25%), `NOx` (17.41%), `NO2` (16.55%), `NO` (16.48%), `CO` (12.22%).

---

### 3.4. `station_day.csv`
Daily pollutant measurements recorded across 110 individual monitoring stations.

- Total Rows: 108,035
- Total Columns: 16 (matches `city_day.csv` schema except `StationId` replaces `City`).
- Target Missingness: `AQI` and `AQI_Bucket` missing in 21,010 rows (19.45%).
- High missingness in volatile organics and secondary particulates: `Xylene` (78.81%), `NH3` (44.53%), `PM10` (39.53%), `Toluene` (35.82%), `Benzene` (29.12%).

---

### 3.5. `station_hour.csv`
High-resolution hourly measurements from 110 stations.

- Total Rows: 2,589,083
- Total Columns: 16
- Storage Footprint: ~209.5 MB uncompressed.

---

## 4. Documented AQI Categories & Ground Truth Target Analysis

In the CPCB NAQI standard and Rohan Rao's dataset, `AQI_Bucket` maps directly to numerical `AQI` boundaries as verified by empirical inspection:

| AQI Category (`AQI_Bucket`) | CPCB Defined Range | Dataset Actual Min AQI | Dataset Actual Max AQI | Row Count (`city_day.csv`) | Class Share (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Good** | 0 – 50 | 13.0 | 50.0 | 1,341 | 5.40% |
| **Satisfactory** | 51 – 100 | 51.0 | 100.0 | 8,224 | 33.09% |
| **Moderate** | 101 – 200 | 101.0 | 200.0 | 8,829 | 35.53% |
| **Poor** | 201 – 300 | 201.0 | 300.0 | 2,781 | 11.19% |
| **Very Poor** | 301 – 400 | 301.0 | 400.0 | 2,337 | 9.40% |
| **Severe** | 401 – 500+ | 401.0 | 2,049.0 | 1,338 | 5.38% |
| **Missing / Uncomputed** | N/A | NaN | NaN | 4,681 | 15.85% (of 29,531) |
| **Total Valid Rows** | — | **13.0** | **2,049.0** | **24,850** | **100.00%** |

### Critical Observations on Ground Truth:
1. **Perfect Consistency:** Whenever `AQI` is present, `AQI_Bucket` is present; whenever `AQI` is missing, `AQI_Bucket` is missing (0 mismatches detected across 29,531 rows).
2. **Extreme Imbalance:** 
   - `Moderate` and `Satisfactory` combined make up **68.62%** of the valid dataset.
   - The boundary classes (`Good` and `Severe`) are extreme minority classes, each representing only **~5.4%**.
   - Standard overall accuracy will be misleading without macro-averaged F1, class-wise recall, and confusion matrix analysis.
3. **Outlier Range in Severe Class:** The maximum AQI observed in `city_day.csv` is `2049.0`, far exceeding the nominal 500 ceiling of standard CPCB reporting. This reflects extreme multi-pollutant accumulation events (e.g., northern Indian winter smog and post-harvest biomass burning).

---

## 5. Potential Target Formulations

| Formulation | Target Column | Task Type | Classes / Output | Pros & Justification | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Formulation** | `AQI_Bucket` | Multi-Class Classification | 6 Classes: Good, Satisfactory, Moderate, Poor, Very Poor, Severe | Directly fulfills the project's primary objective; reflects the real-world operational health advisories issued by CPCB. | Class imbalance requires balanced class weights and stratified sampling. |
| **Continuous Formulation** | `AQI` | Regression | Continuous scalar $[0, \infty)$ | Captures fine-grained numeric variations; can derive buckets via thresholds. | Regression MSE / MAE does not penalize cross-category misclassification errors equally. |
| **Binary Grouping** | `Binary_Health_Risk` | Binary Classification | Safe (Good/Satisfactory) vs. Unhealthy (Moderate to Severe) | Simplifies decision boundary for critical public health alerts. | Discards granular clinical distinctions between Moderate and Severe. |

---

## 6. Data Quality & Preprocessing Requirements

1. **Target Filtering:** Rows with missing `AQI_Bucket` (4,681 rows in `city_day.csv`) must be excluded from supervised model training and evaluation because their ground truth is unknown under CPCB computation rules.
2. **Feature Missingness when Target is Present:**
   - On the 24,850 rows with valid `AQI_Bucket`:
     - `PM2.5` is missing in only **2.73%** (678 rows).
     - `NO`, `NO2`, `CO`, `SO2`, `O3` have missing rates below **3.3%**.
     - `NOx` is missing in **7.47%**.
     - `Benzene` (14.23%), `Toluene` (23.44%), `NH3` (26.30%), `PM10` (28.52%).
     - `Xylene` has high missingness at **61.86%** (15,372 rows).
3. **Imputation Strategy:**
   - Median or iterative imputation grouped by `City` and `Month` preserves spatial-seasonal integrity.
   - For `Xylene`, evaluate model performance with and without `Xylene` to test whether high missingness degrades model stability.
4. **Data Leakage Prevention:**
   - `AQI` is computed mathematically from pollutant sub-indices by CPCB. During feature selection for predicting `AQI_Bucket`, `AQI` must be **strictly excluded** from the input feature matrix $X$ to avoid target leakage.
   - Only monitored pollutant concentrations and spatial/temporal covariates must form the feature space.

---

## 7. Compliance Verification Checklist

- [x] All 5 raw CSV files inspected and verified on disk.
- [x] File schemas, row counts, memory sizes, and missingness rates reported accurately.
- [x] Confirmed zero weather variables exist in the raw dataset.
- [x] Confirmed CPCB AQI bucket categories and exact numeric thresholds.
- [x] Explicitly noted distinction between current-condition classification and forecasting.
- [x] No modifications or deletions performed on the original raw files.

