# AirSense limitations and future work

**Retrospective test exposure.** Earlier project work and the audit had already inspected test results. Corrected code can prevent further selection leakage but cannot make the same data historically unseen. A fresh external evaluation is the highest-priority outstanding requirement.

**Mixed temporal and spatial change.** The test includes 919 rows from six cities absent in training. Without paired same-city and deliberately city-disjoint analyses, changes cannot be attributed uniquely to time or geography. Shared-date CV and a seven-day gap do not remove longer autocorrelation or correlated monitoring conditions.

**Target and information availability.** Same-day pollutant measurements are closely related to the supplied AQI label. This is easier and fundamentally different from forecasting. AQI-derived input columns are excluded, but no prospective feature-availability study or independent sensor/unit audit has been performed. Values beyond the official published AQI scale remain a source-quality concern.

**Missingness and preprocessing.** Many records depend on imputation; Xylene is missing in 62.91% of training rows, PM10 in 38.66% and NH3 in 33.22%. Missingness may be informative and nonrandom, yet explicit missingness features are not used. Clipping can remove meaningful extremes. Applying it to calendar numerics also clips 2020 Year values to 2019; cyclic encoding and ablations were not evaluated.

**Imbalance and uncertainty.** Severe has 38 test examples; rare-class recalls can change substantially with a few observations. No confidence intervals, repeated seeds, blocked bootstrap, significance testing or cost-sensitive ordinal evaluation is included. The observed differences must not be advertised as statistically established superiority.

**Search and fairness of budgets.** Only six candidates per model were tried. SVM's smaller CV subset has different date windows and cannot be treated as a perfectly matched full-data tuning comparison. Single validation-period selection may favor a particular seasonal mixture. The search does not establish a global optimum.

**Interpretability and deployment.** Validation is used for selection and exploratory feature analysis, so importance findings are not an independent confirmation. SHAP and permutation use small samples, and their stability is unmeasured. Weather is absent; probabilities are uncalibrated; SVM probabilities are unavailable. No prospective deployment, clinical/public-health decision validation, calibrated uncertainty, new-city guarantee or operational scaling study is provided.

## Prioritized scientific follow-up

AirSense provides a traceable retrospective comparison of four classifiers for supplied same-day AQI categories. Baseline RF is the validation-selected candidate and yields test macro F1 0.7826 with accuracy 0.8077. Baseline XGBoost's slightly higher accuracy and lower macro F1 show why a single headline number is inadequate. Per-class results and measured adjacent-category errors reveal behavior hidden by aggregate scores. The bounded tuning study does not consistently improve validation performance, and these negative results are retained.

Future work should first obtain new, untouched observations with verified units and availability timestamps, then preregister the target/horizon and split rules. Separate same-city temporal tests from leave-city-out tests; quantify uncertainty with dependence-aware resampling and repeated periods. Compare against a deterministic AQI procedure only where the required pollutant data, units and averaging windows are genuinely comparable. Test missingness indicators, calendar encodings, pollutant-only ablations, calibrated probabilities and explicit ordinal costs. Weather enrichment would need a verified joined dataset rather than inferred inputs. If forecasting is pursued, construct a genuinely future target and past-only predictors before evaluating sequence models.

This document is a complete research-style project draft, not a claim of publication readiness. Peer review, venue formatting, authorship confirmation, independent validation and the unresolved limitations above remain outstanding.

## Remaining submission checks

- Execute the supplied commands on the target Windows computer; path logic testing here is not Windows execution.
- Inspect all five pages in a real desktop/narrow-window browser; AppTest and HTTP checks do not establish cross-browser layout.
- Confirm paper authorship, affiliation, college format and claim wording before submission.
- Retain raw-data attribution and the retrospective-test caveat. No model is an official AQI measurement or a validated future forecast.
- Review the historical archive if you need the preceding work; use final_results.csv and evaluation_results.json for current results.

No additional model changes, external deployment or independent-data collection have been performed without review.
