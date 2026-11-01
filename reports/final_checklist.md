# AirSense final checklist

Audit scope: the corrected chronological experiment and its authoritative submission artifacts. Historical material is retained and labelled; old random-split scores are not final evidence.

| No. | Status | Check | Evidence / limit |
| --- | --- | --- | --- |
| 1 | PASS | All five original CSV files unchanged | SHA-256 and byte sizes match the original user-uploaded ZIP; raw_data_integrity.csv records all five. This does not certify pre-upload provenance. |
| 2 | PARTIAL / MANUAL | Windows dataset paths | Pathlib-based discovery, archive(3)/archive (3), spaces, environment overrides and Windows path joining are tested. Actual Windows installation and PowerShell launch have not been executed here. |
| 3 | PASS | Selected CSV and target documented | city_day.csv; AQI_Bucket; source period 2015-01-01 to 2020-07-01; 29,531 raw / 24,850 labeled rows. |
| 4 | PASS | No fabricated weather features | All five CSV headers checked; all eight fitted schemas contain only the 12 pollutant fields and six derived calendar fields. Weather is explicitly unavailable. |
| 5 | PASS, QUALIFIED | No evident target leakage in the corrected pipeline | Forbidden target/AQI-derived columns are absent. Fitted percentile bounds, imputation and scaling statistics were independently reconstructed from training rows only. Same-day target construction and historical test exposure remain limitations. |
| 6 | PASS | Separated partitions and CV | Train 16,645 / validation 3,784 / test 4,421. City-Date identities are pairwise disjoint; periods ordered; CV uses whole-date expanding windows and a seven-day gap. Generic split audit now actually detects identity overlap. |
| 7 | PASS | All four model families train | Four full-data baselines successfully retrained in temporary storage and reproduced saved validation confusion matrices. Eight submitted artifacts load and predict successfully. No failure results were invented. |
| 8 | PASS | Metrics derived from actual predictions | Current models reproduced saved validation/test predictions; aggregate metrics, per-class reports and confusion matrices agree with the independent calculations. |
| 9 | PASS | Same test observations for all models | All eight candidates use the same ordered 4,421 City-Date keys, targets and observations. |
| 10 | PASS, QUALIFIED | Selection without current test-set tuning | The frozen Random Forest choice matches recomputed validation macro F1 across all candidates. Corrected training/tuning never evaluate test rows. Historical test knowledge prevents a pristine holdout claim. |
| 11 | PASS | Prediction reuses fitted preprocessing | Complete saved pipelines are loaded and cached. Tests compare input values/order/dtypes with the training builder for every artifact; no fitting occurs on refresh. |
| 12 | PASS | Dashboard errors handled | AppTest exercises invalid/blank/nonfinite/out-of-range/date/extra-feature cases and missing, corrupt, stale and unsupported artifacts, including SVM probabilities and absent SHAP. |
| 13 | PASS | Correct primary paper citation | Publisher and full paper checked: Mendez, Merayo and Nunez, 2023; Artificial Intelligence Review 56, 10031-10066; DOI 10.1007/s10462-023-10424-4. Accented names appear in the bibliography/paper. |
| 14 | PASS | Tables and graphs match the experiment | Comparison CSVs agree with current predictions and original timings. Built-in importance and saved SHAP means agree; validation permutation was recomputed. Plot-generation code consumes these artifacts; the existing PDF was rendered/reviewed in prompt 14. |
| 15 | PASS | Methodology and limitations discussed | The existing 17-page paper, methodology.md, results_and_discussion.md and final limitations document cover chronology, missingness, imbalance, coverage shift, leakage risks and generalization. |
| 16 | PASS | Required dependencies listed | Pinned app/ML dependencies plus ReportLab and pypdf in requirements.txt; pytest via requirements-dev.txt. pip check passed. |
| 17 | PASS | Clear Windows setup instructions | README includes C:\dev\AirSense, virtual environment creation/activation, activation-free fallback, dependency install, path resolution, tests, audit, backed-up training and Streamlit launch. |
| 18 | PASS | Loading through prediction tested | 164 tests passed in the final full run. Coverage includes data loaders, transforms, model fitting, fold-local CV, frozen selection, metrics, real artifact prediction, dashboard failure modes and paper validation. |

**Review status:** software/artifact audit complete; actual Windows/browser checks and scientific/authorship review remain manual. PASS, QUALIFIED does not erase the stated scientific limitation. Nothing is published and no Git mutation was performed. See final_audit_commands.md for exact executed commands and outcomes.


## Windows follow-up correction (1 November 2026)

User Windows run: 161 passed, 3 failed. Matching pinned versions ruled out a
simple dependency mismatch. Added native XGBoost JSON bundles and a shared
checksum-verified loader, without retraining or altering model scores. See
reports/windows_portability_fix.md (or windows_portability_fix.md within reports).
Earlier 164-test counts are historical Linux results; Windows confirmation of
this correction remains pending.
