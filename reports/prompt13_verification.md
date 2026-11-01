# Prompt 13 — Application verification

Story: choose a model and observed measurements → validate and order inputs → apply the saved fitted pipeline → render the category and supported probabilities. The same dashboard exposes actual evaluation/feature-analysis artifacts and historical data.

| Boundary | Status | Evidence |
| --- | --- | --- |
| Local Streamlit server | Passed | Health HTTP 200 (`ok`); root HTTP 200 and app HTML received; logs in prompt13_verification/ |
| Five dashboard sections | Passed | Streamlit AppTest exercises all pages with no application exceptions/errors |
| Form → model → displayed result | Passed | Every one of the eight baseline/tuned pipelines returns the same result as direct saved-pipeline inference |
| Feature order, values and types | Passed | Prediction input DataFrames equal the training feature builder outputs, including dtypes |
| Actual comparison results | Passed | Baseline and tuned tables checked against evaluation_results.json; charts consume those same tables |
| Cache / no retraining | Passed | Repeated loads return the same cached object; UI is tested with Pipeline.fit forbidden |
| Missing/invalid inputs | Passed | Blank, all-blank, nonnumeric, negative, nonfinite, excessive, extra-weather and invalid-date cases |
| Optional/unavailable artifacts | Passed | Missing/corrupt/changed models, stale evaluation, missing raw data and absent SHAP are handled without invented substitutes |
| Probability handling | Passed | DT/RF/XGBoost probabilities match saved pipelines; classes are aligned; SVM reports unsupported probabilities |
| Dependencies | Passed | pip check reports no broken requirements; requirements.txt pins the verified runtime versions |
| Scope preservation | Passed | Prior ZIP files outside the four edited existing prompt-13 files remain byte-identical, including all prompt 11/12 trained models, experimental code and results |

**151 tests passed**, including 29 new prompt-13 tests. See prompt13_verification/pytest.log, summary.json and local_server_check.json. All verification ran locally on Linux with Python 3.12; Windows PowerShell launch commands are supplied in README.md but were not executed on Windows.

Browser layout screenshot QA was not performed because the headless browser binary download was unavailable in this environment. Streamlit AppTest covers page rendering, chart construction and user form interactions; live HTTP checks verify server startup and app delivery. These checks do not measure visual layout in every browser or viewport.

The application introduces no model retraining, tuning changes or replacement performance numbers. It distinguishes same-day classification from future forecasting and retains the historical test-exposure limitation from the experiment.
