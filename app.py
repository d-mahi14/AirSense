"""AirSense prompt 13: read-only Streamlit dashboard over saved ML artifacts.
Run from the project root: python -m streamlit run app.py
"""
from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.prediction_service import (
    ROOT, MODEL_DIR, REPORTS_DIR, MODEL_NAMES, POLLUTANT_COLS, AQI_CATEGORIES,
    ArtifactError, InputValidationError, signature, read_json, load_evaluation,
    load_dataset, training_rows, input_ranges, model_schema, load_model,
    candidate_key, candidate_label, build_model_input, predict,
)
from src.ui_components import (
    apply_style, header, metric_cards, render_comparison,
    render_prediction_result, importance_chart,
)

st.set_page_config(page_title="AirSense | Air Quality ML", page_icon="◌", layout="wide", initial_sidebar_state="expanded")
apply_style()
PAGES = ["Home", "Air Quality Prediction", "Model Comparison", "Pollution Analysis", "Research and Methodology"]
EVALUATION, EVALUATION_ERROR = load_evaluation()


def dataset_or_message():
    try:
        return load_dataset()
    except ArtifactError as exc:
        st.warning(str(exc))
        return None


def home():
    st.markdown('''<div class="hero"><div class="eyebrow">Air quality / Machine learning / Research</div>
    <h1>AirSense</h1><p>Air Quality Prediction and Analysis<br>Using Machine Learning</p>
    <span class="pill">Six AQI categories</span><span class="pill">Four model families</span>
    <span class="pill">Saved experimental evidence</span></div>''', unsafe_allow_html=True)
    selected = EVALUATION.get("selected_candidate")
    record = EVALUATION.get("candidates", {}).get(selected, {})
    metrics = record.get("test_metrics", {})
    metric_cards([
        ("Validation-selected model", MODEL_NAMES.get((selected or "").removeprefix("tuned_"), "Unavailable"), "Tuned" if selected and selected.startswith("tuned_") else "Baseline" if selected else "No verified selection"),
        ("Selected model · test macro F1", f"{metrics['macro_f1']:.4f}" if metrics else "Unavailable", "Equal weight to each category"),
        ("Selected model · test accuracy", f"{metrics['accuracy']:.2%}" if metrics else "Unavailable", "Retrospective chronological test"),
        ("Test observations", f"{EVALUATION['test_rows']:,}" if EVALUATION else "Unavailable", "Same city-days for every candidate"),
    ])
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Understand the category behind the readings")
        st.write("AirSense explores how Decision Tree, Random Forest, SVM and XGBoost classify air-quality categories from measured pollutants and calendar context. It connects a sample prediction to the experiment that produced it.")
        st.subheader("The problem")
        st.write("Pollution records can be incomplete, class frequencies are uneven, and overall accuracy can hide weak performance on less frequent categories. This project compares classifiers with per-class evidence and reports where the experiment is limited.")
    with right:
        st.subheader("Project objectives")
        st.markdown("- Classify available measurements into six CPCB AQI categories.\n- Compare four model families using the same test observations.\n- Prioritize validation macro F1 when choosing a model.\n- Explain predictive associations through importance and SHAP.\n- Make limitations and experimental provenance visible.")
    st.divider()
    st.subheader("How the application works")
    for col, title, body in zip(st.columns(3), ["01 · Supply measurements", "02 · Apply the saved pipeline", "03 · Inspect the evidence"],
        ["Choose a saved model, enter its pollutant readings and the measurement date. Calendar fields are derived automatically.",
         "The original imputation, clipping, encoding and scaling steps run inside the fitted pipeline. Refreshing the app does not train a model.",
         "Read the estimated category and supported class probabilities, then explore real test results, errors and feature associations."]):
        with col:
            st.markdown(f"**{title}**")
            st.write(body)
    st.info("Classification, not forecasting: the inputs and target describe the same day. Entering a date does not make the model predict future pollution.")
    st.caption("Research prototype. Estimates are not official government AQI readings. Historical test exposure is disclosed in Research and Methodology.")


def prediction_page():
    header("Air Quality Prediction", "Classify an observed sample using the complete saved preprocessing and model pipeline.")
    variant = st.radio("Model variant", ["Baseline", "Tuned"], horizontal=True, key="prediction_variant").lower()
    selected = EVALUATION.get("selected_candidate", "random_forest").removeprefix("tuned_")
    names = list(MODEL_NAMES.values())
    default = list(MODEL_NAMES).index(selected) if selected in MODEL_NAMES else 0
    name = st.selectbox("Model", names, index=default, key="prediction_model")
    candidate = candidate_key(name, variant)
    try:
        with st.spinner("Loading saved model…"):
            pipeline = load_model(candidate, EVALUATION)
        schema = model_schema(pipeline)
    except ArtifactError as exc:
        st.error(str(exc))
        st.info("Restore the matching model file from the supplied ZIP. The application does not train replacement models.")
        return
    data = dataset_or_message()
    if data is None:
        st.info("Restore city_day.csv to derive the documented training-based input ranges. Prediction input is disabled until ranges can be verified.")
        return
    training = training_rows(data, EVALUATION.get("experiment", {}).get("train_end", "2019-06-30"))
    try:
        ranges = input_ranges(training)
    except ArtifactError as exc:
        st.error(str(exc))
        return
    required = [f for f in schema if f in POLLUTANT_COLS]
    record = EVALUATION.get("candidates", {}).get(candidate, {})
    metrics = record.get("test_metrics", {})
    if metrics:
        st.caption(f"{candidate_label(candidate)} · test accuracy {metrics['accuracy']:.2%} · test macro F1 {metrics['macro_f1']:.4f}. These are dataset-level results, not a guarantee for this sample.")
    else:
        st.info("Saved pipeline is available, but verified evaluation metrics are unavailable.")
    st.markdown("#### Pollutant readings")
    st.caption("Use the same source-dataset units and daily aggregation as the training data; no unit conversion is performed. Blank readings use the saved imputer. Do not substitute hourly readings or values with different units.")
    if st.button("Load a training example", key="load_example"):
        complete = training.dropna(subset=required)
        if complete.empty:
            st.warning("No fully observed training example is available.")
        else:
            example = complete.iloc[len(complete) // 2]
            for feature in required:
                st.session_state[f"input_{candidate}_{feature}"] = float(example[feature])
            st.session_state[f"date_{candidate}"] = pd.Timestamp(example.Date).date()
            st.session_state[f"example_{candidate}"] = f"Training example: {example.City}, {pd.Timestamp(example.Date).date()}. These are historical inputs, not live measurements."
            st.session_state.pop("last_prediction", None)
    if st.session_state.get(f"example_{candidate}"):
        st.caption(st.session_state[f"example_{candidate}"] + " Editing fields creates a custom sample.")
    with st.form(f"prediction_form_{candidate}"):
        observed_date = st.date_input("Measurement date", value=date(2019, 6, 30), min_value=date(2015, 1, 1), max_value=date.today(), key=f"date_{candidate}",
                                      help="Date of the observed measurements. Training ends June 2019; later dates may be outside the learned distribution.")
        columns = st.columns(3)
        values = {}
        for i, feature in enumerate(required):
            limit = ranges[feature]
            with columns[i % 3]:
                values[feature] = st.number_input(feature, min_value=limit["min"], max_value=limit["max"], value=None,
                    step=0.1, format="%.2f", key=f"input_{candidate}_{feature}", placeholder="Blank if unavailable",
                    help=f"Permitted: 0–{limit['max']:g} source-dataset units. Training median {limit['median']:.2f}; observed maximum {limit['training_max']:.2f}.")
        submitted = st.form_submit_button("Classify sample", type="primary", width="stretch")
    with st.expander("Input contract and permitted ranges"):
        st.write("Numeric readings are nonnegative and finite. Permitted maxima are 120% of each observed training maximum, rounded up; these are application bounds, not regulatory thresholds or sensor-validity limits. The saved pipeline may clip extremes using its fitted training percentiles.")
        st.dataframe(pd.DataFrame([{"Feature": f, **ranges[f]} for f in required]), hide_index=True, width="stretch")
        st.code(", ".join(schema), language=None)
        st.caption("This is the fitted input order. Pollutants use float64; calendar fields and Season are generated with the training feature extractor. No city or weather fields are added.")
    if submitted:
        st.session_state.pop("last_prediction", None)
        try:
            frame, notes = build_model_input(pipeline, values, observed_date, ranges)
            result = predict(pipeline, frame)
            st.session_state["last_prediction"] = {"candidate": candidate, "frame": frame, "notes": notes, "result": result,
                                                   "measurement_date": str(observed_date)}
        except (InputValidationError, ArtifactError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("last_prediction")
    if saved and saved["candidate"] == candidate:
        st.markdown(f"#### Last submitted sample · {saved['measurement_date']}")
        notes = saved["notes"]
        if notes["imputed_pollutants"]:
            st.warning("Unavailable readings imputed by the saved pipeline: " + ", ".join(notes["imputed_pollutants"]) + ". Sparse inputs may make the estimate unreliable.")
        if notes["above_training_max"]:
            st.warning("Above the observed training maximum: " + ", ".join(notes["above_training_max"]) + ". The estimate may extrapolate or use clipped values.")
        if pd.Timestamp(saved["measurement_date"]) > training.Date.max():
            st.warning("The sample date is later than the training period. Temporal drift and calendar extrapolation can reduce reliability.")
        render_prediction_result(saved["result"])
        with st.expander("Exact row passed to the saved pipeline"):
            st.dataframe(saved["frame"], hide_index=True, width="stretch")
    else:
        st.info("Enter measurements or load a historical training example, then choose Classify sample.")


def comparison_page():
    header("Model Comparison", "Actual predictions on one shared chronological test set, with equal attention to every AQI category.")
    if not EVALUATION:
        st.warning(EVALUATION_ERROR or "Verified evaluation results are unavailable.")
        return
    render_comparison(EVALUATION)
    with st.expander("Read the generated evaluation report"):
        path = REPORTS_DIR / "model_comparison.md"
        if path.exists():
            st.download_button("Download model comparison report", path.read_bytes(), file_name=path.name, mime="text/markdown")
        st.caption("Macro F1 weighs all six classes equally. High overall accuracy can coexist with weak minority-class recall. Performance differences are descriptive; no statistical-significance claim is made.")


def feature_artifact(path):
    try:
        frame = pd.read_csv(path)
        if frame.empty or "feature" not in frame:
            raise ValueError("No feature records")
        return frame
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.info(f"Feature artifact unavailable: {Path(path).name}. {exc}")
        return None


def pollution_page():
    header("Pollution Analysis", "Explore measured distributions and correlations, then inspect the saved models' feature associations.")
    data = dataset_or_message()
    if data is None:
        return
    period = st.radio("Data shown", ["Training period", "Full historical dataset"], horizontal=True, key="analysis_period")
    if period == "Training period":
        data = training_rows(data, EVALUATION.get("experiment", {}).get("train_end", "2019-06-30"))
    st.caption(f"{len(data):,} city-days · {data.Date.min().date()} to {data.Date.max().date()} · {data.City.nunique()} cities. Missing values are omitted pairwise for correlation and per feature for distributions; no imputation is applied to this view.")
    st.subheader("Pollutant distributions")
    feature = st.selectbox("Distribution feature", POLLUTANT_COLS, key="distribution_feature")
    observed = pd.to_numeric(data[feature], errors="coerce").dropna()
    observed = observed[np.isfinite(observed)]
    if observed.empty:
        st.info("No finite measurements are available for this feature.")
    else:
        counts, edges = np.histogram(observed, bins=45)
        histogram = pd.DataFrame({"Lower": edges[:-1], "Upper": edges[1:], "Count": counts})
        chart = alt.Chart(histogram).mark_bar(color="#16847d").encode(
            x=alt.X("Lower:Q", bin="binned", title=f"{feature} · source-dataset units"), x2="Upper:Q",
            y=alt.Y("Count:Q", title="City-days"), tooltip=["Lower", "Upper", "Count"],
        ).properties(height=290)
        st.altair_chart(chart, width="stretch")
    summary = data[POLLUTANT_COLS].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    summary["missing_percent"] = data[POLLUTANT_COLS].isna().mean() * 100
    with st.expander("All pollutant distribution statistics"):
        st.dataframe(summary.style.format(precision=2), width="stretch")
    st.subheader("Correlation analysis")
    fields = st.multiselect("Pollutants in correlation matrix", POLLUTANT_COLS, default=["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"], key="correlation_fields")
    method = st.selectbox("Correlation method", ["Spearman", "Pearson"], key="correlation_method").lower()
    if len(fields) < 2:
        st.info("Select at least two pollutants to compare correlations.")
    else:
        corr = data[fields].corr(method=method).rename_axis("Feature").reset_index().melt("Feature", var_name="Other feature", value_name="Correlation")
        chart = alt.Chart(corr).mark_rect().encode(
            x=alt.X("Feature:N", sort=fields, axis=alt.Axis(labelAngle=-20)), y=alt.Y("Other feature:N", sort=fields),
            color=alt.Color("Correlation:Q", scale=alt.Scale(domain=[-1,1], scheme="redblue")),
            tooltip=["Feature", "Other feature", alt.Tooltip("Correlation:Q", format=".3f")],
        ).properties(height=350)
        st.altair_chart(chart, width="stretch")
        st.caption("Correlation describes co-variation in these records. Missingness, shared sources and seasonality can affect it; it does not establish causation.")
    st.divider()
    st.subheader("Saved feature-importance analysis")
    st.caption("These are the baseline models' saved training/validation analyses from prompt 12. They do not recompute with the dataset-view selector and do not describe a single prediction.")
    manifest, error = read_json(REPORTS_DIR / "feature_analysis" / "sampling_manifest.json")
    if error:
        st.info(error)
        return
    tabs = st.tabs(["Random Forest", "XGBoost", "SHAP"])
    for tab, key, kind in [(tabs[0], "random_forest", "mdi"), (tabs[1], "xgboost", "gain")]:
        with tab:
            record = manifest.get("models", {}).get(key, {})
            evaluated = EVALUATION.get("candidates", {}).get(key, {})
            if not evaluated or record.get("model_sha256") != evaluated.get("model_sha256"):
                st.warning("Feature-importance artifacts do not match a verified evaluation model; they are not displayed.")
                continue
            for suffix, column, title in [(kind, "importance", f"{MODEL_NAMES[key]} · {kind.upper()} importance"),
                                           ("permutation", "importance_mean", f"{MODEL_NAMES[key]} · validation permutation")]:
                frame = feature_artifact(REPORTS_DIR / "feature_analysis" / f"{key}_{suffix}.csv")
                if frame is not None:
                    importance_chart(frame, column, title, std="importance_std" if suffix == "permutation" else None)
            st.caption(f"Permutation: {manifest.get('permutation_rows', 'unavailable')} validation rows, {manifest.get('permutation_repeats', 'unavailable')} shuffles. Error bars show variation across shuffles, not confidence intervals.")
    with tabs[2]:
        shap_record = manifest.get("models", {}).get("xgboost", {})
        evaluated = EVALUATION.get("candidates", {}).get("xgboost", {})
        if shap_record.get("shap_status") == "completed" and evaluated and shap_record.get("model_sha256") == evaluated.get("model_sha256"):
            frame = feature_artifact(REPORTS_DIR / "feature_analysis" / "xgboost_shap.csv")
            if frame is not None:
                importance_chart(frame, "importance", "XGBoost · mean absolute TreeSHAP")
                st.caption(f"{manifest.get('shap_training_background_rows')} training background rows; {manifest.get('shap_validation_rows')} validation explanation rows. Mean absolute SHAP is averaged across rows and classes, in raw model-margin units. It shows influence magnitude, not whether a feature increases pollution.")
        else:
            st.info(shap_record.get("shap_error", "Matching SHAP results are unavailable. No SHAP calculation runs on refresh."))
        st.caption("Random Forest SHAP was not attempted in the bounded experiment; RF MDI and permutation are available.")
    st.info("Importance is association, not causation. Correlated predictors may share or substitute information; MDI/gain, permutation and SHAP have different assumptions and biases. Weather inputs are absent and location is excluded from the model.")


def research_page():
    header("Research and Methodology", "Trace the data, model assumptions and experiment behind every displayed result.")
    tabs = st.tabs(["Dataset", "Algorithms", "Methodology", "Results & limitations", "References"])
    with tabs[0]:
        st.subheader("CPCB India · city-day observations")
        st.markdown("Source: [Rohan Rao, Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india), curated from CPCB monitoring records. The application uses `city_day.csv`, not the hourly or station-level tables.")
        data = dataset_or_message()
        if data is not None:
            metric_cards([("Raw city-days", f"{len(data):,}", "Including records without labels"),
                          ("Labeled city-days", f"{data.AQI_Bucket.notna().sum():,}", "Available AQI_Bucket"),
                          ("Cities", str(data.City.nunique()), "Historical monitoring coverage")])
            counts = data.AQI_Bucket.value_counts().reindex(AQI_CATEGORIES, fill_value=0).rename_axis("Class").reset_index(name="Labeled records")
            st.dataframe(counts, hide_index=True, width="stretch")
        st.write("Target: AQI_Bucket, an ordinal category from Good through Severe. The model does not output an official continuous AQI value. The trained inputs are 12 measured pollutant columns plus Year, Month, Day, DayOfWeek, IsWeekend and Season. City identity and AQI-derived target columns are excluded.")
        st.write("No temperature, humidity, wind or other weather measurements are present. Calendar features are not substitutes for measured meteorology. Units are preserved as supplied; the app does not certify sensor units or perform conversions.")
    with tabs[1]:
        st.dataframe(pd.DataFrame([
            {"Algorithm":"Decision Tree", "Foundation":"Recursive impurity-reducing splits; leaf class distributions", "Key limitation":"Variance and overfitting; depth/leaf/pruning controls matter"},
            {"Algorithm":"Random Forest", "Foundation":"Bootstrap-aggregated trees with random feature subsets", "Key limitation":"Less transparent than one tree; impurity importance can be biased"},
            {"Algorithm":"SVM", "Foundation":"Maximum-margin classifier with an RBF kernel and regularization", "Key limitation":"Kernel cost and scaling sensitivity; this artifact has no probabilities"},
            {"Algorithm":"XGBoost", "Foundation":"Sequential gradient-boosted trees optimizing a regularized multiclass objective", "Key limitation":"Tuning complexity; probabilities are uncalibrated"},
        ]), hide_index=True, width="stretch")
        st.caption("Balanced class weights are used by DT, RF and SVM. XGBoost does not use the same class_weight parameter. See each saved configuration and the generated tuning report for actual settings.")
    with tabs[2]:
        st.subheader("Preprocessing stays inside the pipeline")
        st.markdown("1. Derive calendar fields from the measurement date using the same training feature extractor.\n2. Apply fitted 1st–99th percentile winsorisation, median imputation and numerical standardization.\n3. Apply the fitted ordinal Season encoder.\n4. Pass the transformed row to the saved classifier, preserving fitted column order.")
        st.subheader("Chronological evaluation")
        experiment = EVALUATION.get("experiment", {})
        if experiment:
            st.write(f"Train through {experiment['train_end']}; validation after that through {experiment['validation_end']}; test after the validation end. The current test has {EVALUATION['test_rows']:,} labeled city-days.")
        st.write("Training-only hyperparameter search uses three expanding date windows with a seven-day gap. Same-date observations stay together, and preprocessing is fitted within each fold. All finalists are refitted on full training data; SVM search alone uses a bounded subsample. The highest validation macro F1 freezes the selected candidate before final test evaluation.")
        st.write("Reported metrics include accuracy, macro precision/recall/F1, weighted F1, per-class metrics, confusion matrices and full-pipeline prediction timings. Training/refit cost is separate from search cost. Test scores do not change selection. No model is trained, tuned or explained computationally during app refresh.")
        if experiment:
            with st.expander("Dataset and environment provenance"):
                st.json(experiment)
    with tabs[3]:
        if EVALUATION:
            st.write(f"Validation-selected candidate: **{candidate_label(EVALUATION['selected_candidate'])}**.")
            from src.ui_components import results_table
            st.dataframe(results_table(EVALUATION, "baseline").style.format(precision=4, na_rep="Unavailable"), hide_index=True, width="stretch")
        st.markdown("- This is same-day classification, not future forecasting.\n- The original project already inspected test outcomes. The corrected run is retrospective; independent external data is required for confirmation.\n- One dataset, one seed and fixed periods do not establish broad generalization or statistical significance.\n- The 2020 test period includes pandemic and seasonal shifts; rare classes have limited support.\n- New-city transfer, calibrated probabilities and prospective deployment are untested.\n- Missingness, monitoring coverage, winsorisation and label availability can bias estimates.\n- Feature importance does not establish causation; weather measurements are absent.\n- Results are not directly comparable with published numbers using different datasets, targets, splits or metrics.")
        for filename in ["model_comparison.md", "hyperparameter_tuning.md", "feature_analysis.md"]:
            path = REPORTS_DIR / filename
            if path.exists():
                st.download_button(f"Download {filename}", path.read_bytes(), file_name=filename, mime="text/markdown", key=f"research_{filename}")
    with tabs[4]:
        st.subheader("Primary research paper")
        st.markdown("Méndez, M., Merayo, M. G., & Núñez, M. (2023). *Machine learning algorithms to forecast air quality: a survey*. Artificial Intelligence Review, 56, 10031–10066. [Publisher / DOI](https://doi.org/10.1007/s10462-023-10424-4).")
        st.write("The survey's Section 3.3 discusses predictor groups and Section 3.4 discusses evaluation metrics. Its forecasting literature spans different targets and datasets. AirSense uses that taxonomy to document inputs and evaluation choices; it does not claim to reproduce the survey or match published scores. Macro-F1 selection and the date-based CV protocol are this project's choices.")
        st.markdown("- Dataset: [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).\n- Original monitoring authority: [Central Pollution Control Board](https://cpcb.nic.in/).\n- Framework: [Streamlit documentation](https://docs.streamlit.io/).")


with st.sidebar:
    st.markdown("## AirSense")
    st.caption("AIR QUALITY · MACHINE LEARNING")
    st.divider()
    page = st.radio("Explore", PAGES, key="navigation", label_visibility="collapsed")
    st.divider()
    st.caption("Historical data · Saved pipelines\n\nSame-day classification, not forecasting")
    st.caption("Research prototype. Not an official AQI service.")

if EVALUATION_ERROR:
    st.warning("Verified performance metrics are unavailable. " + EVALUATION_ERROR)

try:
    {"Home": home, "Air Quality Prediction": prediction_page, "Model Comparison": comparison_page,
     "Pollution Analysis": pollution_page, "Research and Methodology": research_page}[page]()
except (ArtifactError, InputValidationError, OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"This section could not load its saved data: {exc}")
    st.info("Restore the matching data/report/model files from the supplied ZIP. No replacement metrics or models are generated by the app.")
