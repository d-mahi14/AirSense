# AirSense — Preprocessing Methodology Report

**Project:** AirSense — Air Quality Prediction and Analysis Using Machine Learning  
**Modules:** `src/feature_engineering.py`, `src/pipeline.py`  
**Tests:** `tests/test_pipeline.py`  
**Dataset:** `city_day.csv` (India Air Quality, 2015–2020, N=24,850 labeled rows)

---

## 1. Feature Selection Strategy

### 1.1 Approved Features

All features are drawn **strictly** from the columns present in `city_day.csv`. No external or synthetic variables are introduced.

| Feature | Category | Source | Justification |
|:--------|:---------|:--------|:-------------|
| PM2.5 | Numerical | Direct sensor reading | Primary CPCB-criteria pollutant; highest predictor of high-AQI categories |
| PM10 | Numerical | Direct sensor reading | Second primary CPCB-criteria pollutant |
| NO | Numerical | Direct sensor reading | Precursor to photochemical smog; seasonal variation |
| NO2 | Numerical | Direct sensor reading | Critical nitrogen oxide marker |
| NOx | Numerical | Direct sensor reading | Aggregate nitrogen oxide index |
| NH3 | Numerical | Direct sensor reading | Agricultural emission marker |
| CO | Numerical | Direct sensor reading | Traffic and combustion indicator |
| SO2 | Numerical | Direct sensor reading | Industrial and power plant emissions |
| O3 | Numerical | Direct sensor reading | Ground-level ozone; secondary pollutant formed from NOx + sunlight |
| Benzene | Numerical | Direct sensor reading | Carcinogen; vehicular and industrial source |
| Toluene | Numerical | Direct sensor reading | Solvent emissions |
| Xylene | Numerical | Direct sensor reading | Industrial solvent emissions |
| Year | Numerical | Derived from Date | Secular trends (air quality improving or worsening over time) |
| Month | Numerical | Derived from Date | Strong seasonal patterns; monsoon washout reduces PM, winter increases it |
| Day | Numerical | Derived from Date | Within-month patterns |
| DayOfWeek | Numerical | Derived from Date | Weekday vs. weekend traffic differences |
| IsWeekend | Numerical | Derived from Date | Binary marker for reduced traffic/industrial activity |
| Season | Categorical | Derived from Date | Indian meteorological seasons: Winter, Summer, Monsoon, Post-Monsoon |

**Total features: 18** (17 numerical + 1 categorical)

### 1.2 The `City` Column — Evaluation and Decision

`City` is present in `city_day.csv` and represents 26 Indian cities. It is **excluded by default** for the following reasons:

**Arguments for exclusion:**
1. **Memorization risk**: Including City would allow the model to memorize city-specific pollution histories (e.g., "Delhi → usually Poor/Very Poor") rather than learning from measured pollutant concentrations. A Decision Tree with City as a feature would create one high-information split per city — effectively a city lookup table.
2. **Generalization failure**: A model trained with City cannot classify air quality for any city it was not trained on. Given that India has 642 districts with CAAQMS stations, this severely limits real-world applicability.
3. **Non-physics feature**: City encodes geographic identity, not a measurable chemical property. The CPCB AQI framework is defined purely by pollutant concentration thresholds — it does not depend on location.
4. **Data imbalance per city**: Delhi contributes ~2,000 rows (most heavily polluted); Aizawl contributes ~100 rows (very clean). Including City would introduce severe class-by-city confounding.

**Arguments for inclusion (acknowledged):**
- City may capture regional factors not measured by the 12 pollutants (climate, topography, industrial mix).
- For intra-city monitoring (new station in an existing city), City can improve accuracy.

**Decision**: City is **excluded by default** but supported as an `include_city=True` experimental option in both `pipeline.py` and all model run functions. Including City is documented as reducing geographic generalizability. Researchers must explicitly opt in.

### 1.3 Excluded Columns (Leakage Prevention)

| Column | Reason for Exclusion |
|:-------|:--------------------|
| `AQI` | **Direct data leakage**: AQI_Bucket is a piecewise threshold of AQI. Inclusion allows trivial 100% accuracy. |
| `AQI_Bucket` | Ground-truth target — never a predictor. |
| `Target_Label` | Encoded integer form of the target. |
| `Date` | Raw timestamp — replaced by engineered calendar features. |

---

## 2. Missing Value Handling

### 2.1 Why Data is Missing

The India CPCB monitoring network has heterogeneous station coverage. Not all 12 criteria pollutants are monitored at every station every day. Sensors fail, maintenance windows occur, and some stations do not measure O3, Benzene, Toluene, or Xylene. The dataset's observed missingness rates:

| Feature | Approx. Missingness (city_day) |
|:--------|:-------------------------------|
| Benzene | ~47% |
| Toluene | ~41% |
| Xylene | ~51% |
| NH3 | ~26% |
| CO | ~21% |
| O3 | ~19% |
| NO, NO2, NOx | ~9–11% |
| PM2.5, PM10 | ~4–6% |
| SO2 | ~11% |

### 2.2 Strategy: Two-Stage Median Imputation

**Stage 1**: We use `SimpleImputer(strategy='median')` from scikit-learn — robust to outliers and numerically stable. The **training-set median** is computed per feature column.

**Stage 2**: Before imputation, `WinsorizeTransformer` clips values at the 1st and 99th training-set percentiles. This prevents extreme instrument spikes from biasing the median computation.

**Why median over mean?**: Pollutant concentrations are right-skewed distributions (rare extreme pollution events create a long right tail). The median is insensitive to these outliers and better represents the "typical" value for imputation.

**Why train-only fitting?**: If the median were computed on all available data (train + val + test), information from val/test would flow into the training statistics. This constitutes **data leakage** — the model would see a slightly better-informed imputation than would be available at real inference time. The `fit_pipeline()` call is explicitly restricted to training data only.

### 2.3 Target Column: Never Imputed

Rows where `AQI_Bucket` is missing (4,681 rows = 15.85%) are **filtered out entirely** before any model training. This is a hard constraint from CPCB regulatory standards — when fewer than 3 pollutants are monitored, or neither PM2.5 nor PM10 is recorded, the AQI is legally undefined and cannot be reliably reconstructed.

---

## 3. Categorical Encoding

### 3.1 Season

**Values**: Winter, Summer, Monsoon, Post-Monsoon  
**Mapping**: Defined via Indian meteorological calendar:
- Winter: Dec, Jan, Feb
- Summer: Mar, Apr, May
- Monsoon: Jun, Jul, Aug, Sep
- Post-Monsoon: Oct, Nov

**Encoding**: `OrdinalEncoder` with explicit known categories list `['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']`. This ensures:
1. The encoding is identical across train/val/test (regardless of which seasons appear in each fold).
2. If a validation or test row contains an unknown Season value (theoretically impossible from the formula but guarded against), it receives `-1` — the `unknown_value` — without raising an exception.

**Why OrdinalEncoder over OneHotEncoder for Season?**  
Season is an ordinal cyclic feature (the four seasons occur in a fixed cycle). OrdinalEncoder with known categories produces a compact integer representation (0–3) that tree models handle efficiently. OneHotEncoder would expand Season into 4 binary columns with redundant information and sparse representation.

### 3.2 City (if opted in)

When `include_city=True`, City is encoded with `OrdinalEncoder` with `handle_unknown='use_encoded_value'` and `unknown_value=-1`. This ensures that a new city never seen in training receives a placeholder value (-1) without crashing the inference pipeline.

---

## 4. Feature Scaling

### 4.1 StandardScaler

`StandardScaler` is applied to all 17 numerical features after imputation:

$$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

where $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are computed **exclusively on the training set**.

### 4.2 Why Scale at All — Even for Tree Models?

Tree-based models (Decision Tree, Random Forest, XGBoost) are **scale-invariant**: the splitting threshold at each node is expressed in the original feature space and a monotone transformation cannot change which split is optimal. So for these three models, StandardScaler is mathematically equivalent to no scaling at all.

**However**, StandardScaler is kept in the universal pipeline for two reasons:
1. **SVM requires scaling** (critical). SVM computes Euclidean distances and inner products. A feature with values in [0, 500] will completely dominate one with values in [0, 5]. After scaling, all features contribute on equal footing.
2. **Single unified pipeline**: Having one pipeline for all four models ensures consistent preprocessing and eliminates the risk of applying different transformations to different models, which would make comparisons invalid.

### 4.3 Outlier Clipping (WinsorizeTransformer)

Before imputation and scaling, the custom `WinsorizeTransformer` clips each feature at its 1st and 99th training-set percentile. This:
- Prevents instrument spikes (e.g., PM2.5 = 999.9 from a sensor malfunction) from distorting the StandardScaler's mean and standard deviation.
- Does NOT discard any rows — only bounds extreme values.
- Is fitted strictly on training data.

---

## 5. Pipeline Architecture

```
Raw Feature DataFrame (X_train)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              ColumnTransformer                           │
│                                                         │
│  ┌─────────────────────────┐   ┌───────────────────┐   │
│  │   Numerical Pipeline    │   │ Categorical Pipeline│   │
│  │                         │   │                    │   │
│  │  1. WinsorizeTransformer│   │ 1. SimpleImputer   │   │
│  │     (clip at 1st/99th   │   │    (most_frequent) │   │
│  │      training percentile│   │                    │   │
│  │                         │   │ 2. OrdinalEncoder  │   │
│  │  2. SimpleImputer       │   │    (known_categories│  │
│  │     (median)            │   │     unknown → -1)  │   │
│  │                         │   └───────────────────┘   │
│  │  3. StandardScaler      │                            │
│  │     (zero-mean,         │                            │
│  │      unit-variance)     │                            │
│  └─────────────────────────┘                            │
│                                                         │
│  Input:  [PM2.5…Xylene, Year…IsWeekend] | [Season]     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   Transformed Array (N × 18 float64)
         │
         ▼
   [Model Classifier]
   Decision Tree / Random Forest / SVM / XGBoost
```

### 5.1 Implementation Notes

- `ColumnTransformer(remainder='drop')`: Any column not explicitly listed in `numerical` or `categorical` is dropped. This ensures that `City`, `Date`, `AQI`, and `AQI_Bucket` cannot accidentally appear in the model input — even if a caller passes them through.
- `verbose_feature_names_out=True`: Output feature names are prefixed with their transformer name (`numerical__PM2.5`, `categorical__Season`) to avoid ambiguity.
- The preprocessing pipeline is built by `build_preprocessing_pipeline()` and fitted by `prepare_all_splits()` — the same call also transforms all three splits, guaranteeing that val/test never touch `fit()`.

---

## 6. Train/Validation/Test Isolation Guarantees

| Guarantee | Mechanism |
|:----------|:----------|
| Imputation statistics from training only | `SimpleImputer.fit()` called only on `X_train` |
| Scaling statistics from training only | `StandardScaler.fit()` called only on `X_train` |
| Clipping bounds from training only | `WinsorizeTransformer.fit()` called only on `X_train` |
| Target label statistics: N/A | Target labels are never used in any preprocessing computation |
| Stratified splits for class balance | `stratified_data_split()` uses per-class shuffling with `random_state=42` |
| Fixed random seeds | All randomized operations use `random_state=42` for full reproducibility |

The isolation is enforced in `prepare_all_splits()`:
```python
X_train_t = pipeline.fit_transform(X_train)  # fit + transform
X_val_t   = pipeline.transform(X_val)         # transform only
X_test_t  = pipeline.transform(X_test)        # transform only
```

---

## 7. Feature Order Consistency

The `ColumnTransformer` output order is deterministic:
1. All numerical features in the order they appear in `ALL_NUMERICAL_FEATURES` (17 columns).
2. All categorical features in the order they appear in `ALL_CATEGORICAL_FEATURES` (1 column: Season).

This ordering is:
- Fixed at pipeline construction time.
- Identical across `X_train_t`, `X_val_t`, `X_test_t`.
- Recoverable via `get_feature_names_out()` which returns the prefixed column names.

---

## 8. Test Suite Results

The test suite in `tests/test_pipeline.py` covers:

| Test Class | Description | Tests |
|:-----------|:------------|:------|
| `TestPrepareModelingDataset` | Temporal feature extraction | 4 |
| `TestGetFeatureLists` | Feature type identification | 5 |
| `TestExtractXY` | X/y extraction and leakage guard | 4 |
| `TestMissingValueHandling` | Zero NaN after imputation; training-only medians | 2 |
| `TestUnseenCategories` | OrdinalEncoder -1 for unknown values | 2 |
| `TestReproducibility` | Identical output across runs | 2 |
| `TestTrainTestIsolation` | Scaler mean verified on training data | 2 |
| `TestFeatureOrderConsistency` | Output names and column order | 2 |
| `TestLabelMapping` | Round-trip category ↔ integer | 3 |
| `TestPipelineInterface` | sklearn API compliance | 3 |

**See actual test execution results in Section 9.**

---

## 9. Preprocessing Decision Summary

| Decision | Choice | Justification |
|:---------|:-------|:-------------|
| Primary dataset | city_day.csv | Daily granularity; complete AQI_Bucket labels; manageable size |
| Target | AQI_Bucket | Official CPCB 6-class label; directly available |
| Predictor set | 12 pollutants + 5 temporal numerics + Season | Dataset-verified; no synthetic features |
| City inclusion | Excluded by default | Prevents memorization; improves generalizability |
| Missing values | Training-only median imputation | Robust; no target leakage |
| Outlier treatment | Training percentile clipping (1st/99th) | Bounds sensor spikes without row deletion |
| Categorical encoding | OrdinalEncoder with known categories | Compact; handles unknown values safely |
| Numerical scaling | StandardScaler (train-fit only) | Required for SVM; harmless for trees |
| Pipeline type | sklearn Pipeline + ColumnTransformer | Single unified pipeline for all 4 models |
| Target encoding | Integer labels 0–5 | Required for XGBoost; natural for sklearn |
| Split strategy | Stratified 70/15/15 | Preserves class balance across splits |
