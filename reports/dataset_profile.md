# AirSense — Comprehensive Dataset Profile & Empirical Audit

**Project:** AirSense — Air Quality Prediction and Analysis Using Machine Learning  
**Dataset:** India Air Quality Dataset (2015–2020)  
**Source:** Kaggle (Rohan Rao) — `rohanrao/air-quality-data-in-india`  
**Audit Execution:** Fully automated empirical audit via `scripts/inspect_dataset.py`  

---

## 1. Executive Summary & Inventory

This document provides a factual, line-by-line inspection of all five CSV files found in the dataset directory. Every metric, shape, count, and threshold has been verified by parsing the physical CSV files directly.

| Filename | Role | Resolution | Rows | Columns | File Size | Key Duplicates | Target Availability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `stations.csv` | Metadata | Static (Metadata) | 230 | 5 | 0.01 MB | 0 | N/A |
| `city_day.csv` | Observations | Daily (24-hour aggregate) | 29,531 | 16 | 2.45 MB | 0 | 4681 missing |
| `city_hour.csv` | Observations | Hourly (1-hour interval) | 707,875 | 16 | 62.61 MB | 0 | 129080 missing |
| `station_day.csv` | Observations | Daily (24-hour aggregate) | 108,035 | 16 | 8.23 MB | 0 | 21010 missing |
| `station_hour.csv` | Observations | Hourly (1-hour interval) | 2,589,083 | 16 | 209.5 MB | 0 | 570190 missing |

---

## 2. Detailed Profile for Each File

### 2.1 `stations.csv`
- **File Role:** Metadata (Station Registry)
- **Physical Size:** 14,467 bytes (0.01 MB)
- **Dimensions:** 230 rows $\times$ 5 columns
- **Duplicates:** Full duplicates: **0** | Natural key duplicates: **0**
- **Negative / Implausible Values:** None (0 negative values across all numeric columns)
- **City Coverage:** 127 cities
- **Station Coverage:** 230 monitoring stations
- **State Coverage:** 21 states/UTs
- **Pollutants Measured (0):** None
- **Weather Fields:** **NONE (0 fields found).** Confirmed zero meteorological measurements.

#### Column Schema & Missingness
| Column Name | Inferred Type | Null Count | Missing % | Description / Notes |
| :--- | :--- | :--- | :--- | :--- |
| `StationId` | `object` | 0 | 0.00% | Identifier |
| `StationName` | `object` | 0 | 0.00% | Identifier |
| `City` | `object` | 0 | 0.00% | Identifier |
| `State` | `object` | 0 | 0.00% | Identifier |
| `Status` | `object` | 97 | 42.17% | Identifier |

---

### 2.2 `city_day.csv`
- **File Role:** Observation Table (Daily (24-hour aggregate))
- **Physical Size:** 2,574,056 bytes (2.45 MB)
- **Dimensions:** 29,531 rows $\times$ 16 columns
- **Duplicates:** Full duplicates: **0** | Natural key duplicates: **0**
- **Negative / Implausible Values:** None (0 negative values across all numeric columns)
- **Temporal Scope:** 2015-01-01 to 2020-07-01
- **Timestamp Format:** `YYYY-MM-DD (ISO 8601 Date string)`
- **City Coverage:** 26 cities
- **Pollutants Measured (12):** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Weather Fields:** **NONE (0 fields found).** Confirmed zero meteorological measurements.

#### Column Schema & Missingness
| Column Name | Inferred Type | Null Count | Missing % | Description / Notes |
| :--- | :--- | :--- | :--- | :--- |
| `City` | `object` | 0 | 0.00% | Identifier |
| `Date` | `object` | 0 | 0.00% | Identifier |
| `PM2.5` | `float64` | 4,598 | 15.57% | Criteria Pollutant |
| `PM10` | `float64` | 11,140 | 37.72% | Criteria Pollutant |
| `NO` | `float64` | 3,582 | 12.13% | Criteria Pollutant |
| `NO2` | `float64` | 3,585 | 12.14% | Criteria Pollutant |
| `NOx` | `float64` | 4,185 | 14.17% | Criteria Pollutant |
| `NH3` | `float64` | 10,328 | 34.97% | Criteria Pollutant |
| `CO` | `float64` | 2,059 | 6.97% | Criteria Pollutant |
| `SO2` | `float64` | 3,854 | 13.05% | Criteria Pollutant |
| `O3` | `float64` | 4,022 | 13.62% | Criteria Pollutant |
| `Benzene` | `float64` | 5,623 | 19.04% | Criteria Pollutant |
| `Toluene` | `float64` | 8,041 | 27.23% | Criteria Pollutant |
| `Xylene` | `float64` | 18,109 | 61.32% | Criteria Pollutant |
| `AQI` | `float64` | 4,681 | 15.85% | CPCB Air Quality Index (Continuous Target) |
| `AQI_Bucket` | `object` | 4,681 | 15.85% | CPCB Health Advisory Category (Primary Classification Target) |

#### Target Distribution (`AQI_Bucket`)
| Category | Count | Percentage (Valid) | Overall % | CPCB AQI Range |
| :--- | :--- | :--- | :--- | :--- |
| **Good** | 1,341 | 5.40% | 4.54% | 0 – 50 |
| **Satisfactory** | 8,224 | 33.09% | 27.85% | 51 – 100 |
| **Moderate** | 8,829 | 35.53% | 29.90% | 101 – 200 |
| **Poor** | 2,781 | 11.19% | 9.42% | 201 – 300 |
| **Very Poor** | 2,337 | 9.40% | 7.91% | 301 – 400 |
| **Severe** | 1,338 | 5.38% | 4.53% | 401 – 500+ |
| *Unlabeled / NaN* | 4,681 | — | 15.85% | N/A |

---

### 2.3 `city_hour.csv`
- **File Role:** Observation Table (Hourly (1-hour interval))
- **Physical Size:** 65,648,581 bytes (62.61 MB)
- **Dimensions:** 707,875 rows $\times$ 16 columns
- **Duplicates:** Full duplicates: **0** | Natural key duplicates: **0**
- **Negative / Implausible Values:** None (0 negative values across all numeric columns)
- **Temporal Scope:** 2015-01-01 01:00:00 to 2020-07-01 00:00:00
- **Timestamp Format:** `YYYY-MM-DD HH:MM:SS (ISO 8601 Datetime string)`
- **City Coverage:** 26 cities
- **Pollutants Measured (12):** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Weather Fields:** **NONE (0 fields found).** Confirmed zero meteorological measurements.

#### Column Schema & Missingness
| Column Name | Inferred Type | Null Count | Missing % | Description / Notes |
| :--- | :--- | :--- | :--- | :--- |
| `City` | `object` | 0 | 0.00% | Identifier |
| `Datetime` | `object` | 0 | 0.00% | Identifier |
| `PM2.5` | `float64` | 145,088 | 20.50% | Criteria Pollutant |
| `PM10` | `float64` | 296,737 | 41.92% | Criteria Pollutant |
| `NO` | `float64` | 116,632 | 16.48% | Criteria Pollutant |
| `NO2` | `float64` | 117,122 | 16.55% | Criteria Pollutant |
| `NOx` | `float64` | 123,224 | 17.41% | Criteria Pollutant |
| `NH3` | `float64` | 272,542 | 38.50% | Criteria Pollutant |
| `CO` | `float64` | 86,517 | 12.22% | Criteria Pollutant |
| `SO2` | `float64` | 130,373 | 18.42% | Criteria Pollutant |
| `O3` | `float64` | 129,208 | 18.25% | Criteria Pollutant |
| `Benzene` | `float64` | 163,646 | 23.12% | Criteria Pollutant |
| `Toluene` | `float64` | 220,607 | 31.16% | Criteria Pollutant |
| `Xylene` | `float64` | 455,829 | 64.39% | Criteria Pollutant |
| `AQI` | `float64` | 129,080 | 18.23% | CPCB Air Quality Index (Continuous Target) |
| `AQI_Bucket` | `object` | 129,080 | 18.23% | CPCB Health Advisory Category (Primary Classification Target) |

#### Target Distribution (`AQI_Bucket`)
| Category | Count | Percentage (Valid) | Overall % | CPCB AQI Range |
| :--- | :--- | :--- | :--- | :--- |
| **Good** | 38,611 | 6.67% | 5.45% | 0 – 50 |
| **Satisfactory** | 189,434 | 32.73% | 26.76% | 51 – 100 |
| **Moderate** | 198,991 | 34.38% | 28.11% | 101 – 200 |
| **Poor** | 66,654 | 11.52% | 9.42% | 201 – 300 |
| **Very Poor** | 57,455 | 9.93% | 8.12% | 301 – 400 |
| **Severe** | 27,650 | 4.78% | 3.91% | 401 – 500+ |
| *Unlabeled / NaN* | 129,080 | — | 18.23% | N/A |

---

### 2.4 `station_day.csv`
- **File Role:** Observation Table (Daily (24-hour aggregate))
- **Physical Size:** 8,634,105 bytes (8.23 MB)
- **Dimensions:** 108,035 rows $\times$ 16 columns
- **Duplicates:** Full duplicates: **0** | Natural key duplicates: **0**
- **Negative / Implausible Values:** None (0 negative values across all numeric columns)
- **Temporal Scope:** 2015-01-01 to 2020-07-01
- **Timestamp Format:** `YYYY-MM-DD (ISO 8601 Date string)`
- **Station Coverage:** 110 monitoring stations
- **Pollutants Measured (12):** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Weather Fields:** **NONE (0 fields found).** Confirmed zero meteorological measurements.

#### Column Schema & Missingness
| Column Name | Inferred Type | Null Count | Missing % | Description / Notes |
| :--- | :--- | :--- | :--- | :--- |
| `StationId` | `object` | 0 | 0.00% | Identifier |
| `Date` | `object` | 0 | 0.00% | Identifier |
| `PM2.5` | `float64` | 21,625 | 20.02% | Criteria Pollutant |
| `PM10` | `float64` | 42,706 | 39.53% | Criteria Pollutant |
| `NO` | `float64` | 17,106 | 15.83% | Criteria Pollutant |
| `NO2` | `float64` | 16,547 | 15.32% | Criteria Pollutant |
| `NOx` | `float64` | 15,500 | 14.35% | Criteria Pollutant |
| `NH3` | `float64` | 48,105 | 44.53% | Criteria Pollutant |
| `CO` | `float64` | 12,998 | 12.03% | Criteria Pollutant |
| `SO2` | `float64` | 25,204 | 23.33% | Criteria Pollutant |
| `O3` | `float64` | 25,568 | 23.67% | Criteria Pollutant |
| `Benzene` | `float64` | 31,455 | 29.12% | Criteria Pollutant |
| `Toluene` | `float64` | 38,702 | 35.82% | Criteria Pollutant |
| `Xylene` | `float64` | 85,137 | 78.81% | Criteria Pollutant |
| `AQI` | `float64` | 21,010 | 19.45% | CPCB Air Quality Index (Continuous Target) |
| `AQI_Bucket` | `object` | 21,010 | 19.45% | CPCB Health Advisory Category (Primary Classification Target) |

#### Target Distribution (`AQI_Bucket`)
| Category | Count | Percentage (Valid) | Overall % | CPCB AQI Range |
| :--- | :--- | :--- | :--- | :--- |
| **Good** | 5,510 | 6.33% | 5.10% | 0 – 50 |
| **Satisfactory** | 23,636 | 27.16% | 21.88% | 51 – 100 |
| **Moderate** | 29,417 | 33.80% | 27.23% | 101 – 200 |
| **Poor** | 11,493 | 13.21% | 10.64% | 201 – 300 |
| **Very Poor** | 11,762 | 13.52% | 10.89% | 301 – 400 |
| **Severe** | 5,207 | 5.98% | 4.82% | 401 – 500+ |
| *Unlabeled / NaN* | 21,010 | — | 19.45% | N/A |

---

### 2.5 `station_hour.csv`
- **File Role:** Observation Table (Hourly (1-hour interval))
- **Physical Size:** 219,675,434 bytes (209.5 MB)
- **Dimensions:** 2,589,083 rows $\times$ 16 columns
- **Duplicates:** Full duplicates: **0** | Natural key duplicates: **0**
- **Negative / Implausible Values:** None (0 negative values across all numeric columns)
- **Temporal Scope:** 2015-01-01 01:00:00 to 2020-07-01 00:00:00
- **Timestamp Format:** `YYYY-MM-DD HH:MM:SS (ISO 8601 Datetime string)`
- **Station Coverage:** 110 monitoring stations
- **Pollutants Measured (12):** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Weather Fields:** **NONE (0 fields found).** Confirmed zero meteorological measurements.

#### Column Schema & Missingness
| Column Name | Inferred Type | Null Count | Missing % | Description / Notes |
| :--- | :--- | :--- | :--- | :--- |
| `StationId` | `object` | 0 | 0.00% | Identifier |
| `Datetime` | `object` | 0 | 0.00% | Identifier |
| `PM2.5` | `float64` | 647,689 | 25.02% | Criteria Pollutant |
| `PM10` | `float64` | 1,119,252 | 43.23% | Criteria Pollutant |
| `NO` | `float64` | 553,711 | 21.39% | Criteria Pollutant |
| `NO2` | `float64` | 528,973 | 20.43% | Criteria Pollutant |
| `NOx` | `float64` | 490,808 | 18.96% | Criteria Pollutant |
| `NH3` | `float64` | 1,236,618 | 47.76% | Criteria Pollutant |
| `CO` | `float64` | 499,302 | 19.28% | Criteria Pollutant |
| `SO2` | `float64` | 742,737 | 28.69% | Criteria Pollutant |
| `O3` | `float64` | 725,973 | 28.04% | Criteria Pollutant |
| `Benzene` | `float64` | 861,579 | 33.28% | Criteria Pollutant |
| `Toluene` | `float64` | 1,042,366 | 40.26% | Criteria Pollutant |
| `Xylene` | `float64` | 2,075,104 | 80.15% | Criteria Pollutant |
| `AQI` | `float64` | 570,190 | 22.02% | CPCB Air Quality Index (Continuous Target) |
| `AQI_Bucket` | `object` | 570,190 | 22.02% | CPCB Health Advisory Category (Primary Classification Target) |

#### Target Distribution (`AQI_Bucket`)
| Category | Count | Percentage (Valid) | Overall % | CPCB AQI Range |
| :--- | :--- | :--- | :--- | :--- |
| **Good** | 152,113 | 7.53% | 5.88% | 0 – 50 |
| **Satisfactory** | 530,164 | 26.26% | 20.48% | 51 – 100 |
| **Moderate** | 675,008 | 33.43% | 26.07% | 101 – 200 |
| **Poor** | 239,990 | 11.89% | 9.27% | 201 – 300 |
| **Very Poor** | 301,150 | 14.92% | 11.63% | 301 – 400 |
| **Severe** | 120,468 | 5.97% | 4.65% | 401 – 500+ |
| *Unlabeled / NaN* | 570,190 | — | 22.02% | N/A |

---

## 3. Explicit Audit of Weather & Meteorological Variables

> [!IMPORTANT]
> **Definitive Finding:** None of the 5 files contain any meteorological or weather measurements (such as ambient temperature, relative humidity, wind speed, wind direction, rainfall/precipitation, solar radiation, or barometric pressure).

A programmatic regex and keyword scan across all 80 column headers in the 5 CSV files confirmed zero matches for weather-related parameters:
- Tested keywords: `temp`, `temperature`, `humidity`, `rh`, `wind`, `ws`, `wd`, `rain`, `precip`, `solar`, `rad`, `pressure`, `baro`.
- Matched columns: **0 columns**.
- **Implication for AirSense Modeling:** Models will strictly evaluate pollutant concentrations ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3, etc.$) and temporal indicators. Weather variables will **never be assumed, synthesized, or fabricated**.

---

## 4. File Overlap, Temporal Resolutions & Table Relationships

The five CSV files form a cohesive hierarchical monitoring system rather than five unrelated datasets:

1. **Metadata vs. Observation:**
   - `stations.csv` is a static metadata catalog defining 230 stations across 127 Indian cities.
   - The remaining four files are dynamic observational time-series tables.
2. **Spatial Hierarchies (Station vs. City):**
   - `station_day.csv` and `station_hour.csv` contain readings from **110 active monitoring stations**.
   - When joined against `stations.csv`, these 110 stations reside in exactly **26 unique cities**.
   - `city_day.csv` and `city_hour.csv` represent the municipal aggregations across those same **26 cities** (Ahmedabad, Bengaluru, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Patna, etc.).
3. **Temporal Hierarchies (Hourly vs. Daily):**
   - The hourly files (`city_hour.csv` and `station_hour.csv`) capture intra-day dynamics (1-hour resolution, 2015 to 2020).
   - The daily files (`city_day.csv` and `station_day.csv`) aggregate 24-hour pollutant concentrations in accordance with CPCB NAQI reporting protocols (requiring $\ge 16$ hours of valid data for a valid sub-index).
4. **Blind Merging Warning:**
   - Merging daily and hourly files together would create duplicate observation counts and inconsistent time scales.
   - Merging city and station files without multi-level aggregation would create pseudo-replication and data leakage.

---

## 5. Recommendation for Primary Classification Experiment

### Recommended Primary File: `city_day.csv`

### Comprehensive Justification:
1. **Alignment with Problem Objective:** The primary goal is to compare Decision Tree, Random Forest, SVM, and XGBoost for air-quality classification under official CPCB `AQI_Bucket` standards. Public AQI advisories and municipal alerts in India are officially defined and issued on a **daily, city-wide basis**.
2. **Sufficient Statistical Power without Memory Bottlenecks:**
   - `city_day.csv` contains **24,850 fully labeled records** across 26 major cities spanning 5.5 years.
   - This provides robust statistical power for multi-class classification and stratified 5-fold cross-validation.
   - Concurrently, 24,850 rows ensures that kernel Support Vector Machines ($\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$ computational complexity) and SHAP TreeExplainer run in minutes on a standard student laptop, whereas running kernel SVM on 2.58 million rows (`station_hour.csv`) would freeze the system and exhaust memory.
3. **Superior Signal-to-Noise Ratio & Lower Missingness:**
   - In `city_day.csv`, valid rows have remarkably low missingness for primary criteria pollutants: $PM_{2.5}$ missingness is only **2.73%**, $NO_2$ is **1.57%**, and $CO$ is **1.79%**.
   - In contrast, hourly sensor streams suffer from frequent instantaneous dropouts, telecommunication dropouts, and maintenance calibration spikes.
4. **Zero Record Duplication:** Empirical verification confirmed 0 full duplicate rows and 0 `(City, Date)` key duplicates in `city_day.csv`.

---

## 6. Audit Verification Sign-Off

- [x] All 5 CSV files inspected directly by parsing disk files.
- [x] Zero weather variables confirmed across all datasets.
- [x] Full duplicates and key duplicates empirically counted (0 found across all files).
- [x] Negative and non-physical values checked (0 negative values found).
- [x] Relationship between metadata, hourly, and daily files documented.
- [x] Primary file recommendation (`city_day.csv`) justified on clinical, statistical, and algorithmic grounds.
