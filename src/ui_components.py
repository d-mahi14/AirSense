"""Presentation helpers. All charts and tables receive real saved/data values."""
from __future__ import annotations

from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.prediction_service import AQI_CATEGORIES, AQI_COLORS, MODEL_NAMES, candidate_label

CHART_COLORS = ["#16847d", "#5869b1", "#c28b3d", "#86629a"]


def apply_style():
    st.markdown('''<style>
    .stApp {background:#f6f8fa;color:#172f3e;}
    .block-container {padding-top:2.3rem;padding-bottom:3rem;max-width:1320px;}
    h1,h2,h3 {letter-spacing:-.035em;color:#163342;}
    h1 {font-weight:750;}
    [data-testid="stSidebar"] {background:#102f3c;}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color:#e1eeee;}
    [data-testid="stMetric"] {background:#fff;border:1px solid #dfe7ec;border-radius:14px;padding:17px 20px;}
    [data-testid="stMetricLabel"] {color:#617582;font-size:.82rem;}
    [data-testid="stMetricValue"] {font-size:1.8rem;letter-spacing:-.04em;}
    .eyebrow {font-size:.72rem;letter-spacing:.17em;text-transform:uppercase;font-weight:700;color:#18867b;margin-bottom:10px;}
    .hero {background:linear-gradient(115deg,#102f3c,#214c59);border:1px solid #315665;border-radius:20px;padding:38px 40px;margin:0 0 24px;}
    .hero h1 {color:#f4fbfc!important;font-size:3.5rem;margin:0;line-height:1.05;}
    .hero p {color:#c8dde0;font-size:1.15rem;max-width:690px;line-height:1.55;margin:15px 0;}
    .hero .eyebrow {color:#84d4be;}
    .pill {display:inline-block;padding:5px 11px;border:1px solid #64878b;border-radius:20px;font-size:.73rem;color:#d2e8e5;margin-right:8px;margin-top:10px;}
    .section-note {color:#627683;max-width:820px;margin-bottom:22px;line-height:1.6;}
    .result {background:#fff;border:1px solid #dce7e6;border-left:6px solid;border-radius:14px;padding:24px 28px;margin:15px 0;}
    div[data-testid="stForm"] {border-color:#dbe5e9;border-radius:16px;background:#fff;}
    [data-testid="stCaptionContainer"] {line-height:1.5;}
    </style>''', unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown('<div class="eyebrow">AirSense / Research dashboard</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="section-note">{escape(subtitle)}</div>', unsafe_allow_html=True)


def metric_cards(items):
    for col, (label, value, detail) in zip(st.columns(len(items)), items):
        with col:
            st.metric(label, value)
            if detail:
                st.caption(detail)


def show_image(path, caption=""):
    if Path(path).is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Visualization unavailable: {Path(path).name}")


def results_table(evaluation, variant):
    rows = []
    prefix = "tuned_" if variant == "tuned" else ""
    for key, label in MODEL_NAMES.items():
        record = evaluation.get("candidates", {}).get(prefix + key, {})
        metrics = record.get("test_metrics", {})
        row = {"Model": label, "Status": "Available" if metrics else record.get("error", "Not evaluated")}
        for display, field in [("Accuracy", "accuracy"), ("Macro precision", "macro_precision"),
                               ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")]:
            row[display] = metrics.get(field)
        row["Train / refit (s)"] = record.get("train_time_sec")
        row["Inference (s)"] = metrics.get("inference_time_sec")
        rows.append(row)
    return pd.DataFrame(rows)


def render_comparison(evaluation):
    variant = st.radio("Candidate family", ["Baseline", "Tuned"], horizontal=True, key="comparison_variant").lower()
    table = results_table(evaluation, variant)
    selected = evaluation.get("selected_candidate")
    if selected:
        st.info(f"Validation-selected candidate: {candidate_label(selected)}. Test results below describe performance; they do not select the winner.")
    st.dataframe(table.style.format(precision=4, na_rep="Unavailable"), hide_index=True, width="stretch")
    st.caption(f"Same chronological test set: {evaluation.get('test_rows', 'unavailable'):,} city-days. Inference includes preprocessing; tuned training time is the full-data refit, with search costs documented separately.")
    available = table.dropna(subset=["Accuracy", "Macro F1"])
    if not available.empty:
        a, b = st.columns(2)
        for col, metric in [(a, "Accuracy"), (b, "Macro F1")]:
            chart = alt.Chart(available).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X("Model:N", sort=list(MODEL_NAMES.values()), axis=alt.Axis(labelAngle=-15)),
                y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
                color=alt.Color("Model:N", scale=alt.Scale(range=CHART_COLORS), legend=None),
                tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
            ).properties(height=290, title=metric)
            col.altair_chart(chart, width="stretch")
    st.subheader("Confusion matrices")
    normalized = st.toggle("Normalize each true-class row", value=True, key="normalize_confusion")
    tabs = st.tabs(list(MODEL_NAMES.values()))
    prefix = "tuned_" if variant == "tuned" else ""
    per_class = []
    for (key, label), tab in zip(MODEL_NAMES.items(), tabs):
        record = evaluation.get("candidates", {}).get(prefix + key, {})
        metrics = record.get("test_metrics")
        with tab:
            if not metrics:
                st.warning(record.get("error", "Evaluation unavailable for this candidate."))
                continue
            cm = np.asarray(metrics["confusion_matrix"])
            rows = []
            for i, true in enumerate(AQI_CATEGORIES):
                for j, predicted in enumerate(AQI_CATEGORIES):
                    rows.append({"True class": true, "Predicted class": predicted, "Count": int(cm[i,j]),
                                 "Row share": float(cm[i,j] / cm[i].sum()) if cm[i].sum() else 0.0})
            data = pd.DataFrame(rows)
            field = "Row share" if normalized else "Count"
            base = alt.Chart(data).encode(
                x=alt.X("Predicted class:N", sort=AQI_CATEGORIES, axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("True class:N", sort=AQI_CATEGORIES),
                tooltip=["True class", "Predicted class", "Count", alt.Tooltip("Row share:Q", format=".1%")],
            )
            heat = base.mark_rect().encode(color=alt.Color(f"{field}:Q", scale=alt.Scale(scheme="tealblues")))
            text = base.mark_text(fontSize=12).encode(text=alt.Text(f"{field}:Q", format=".0%" if normalized else "d"),
                                                    color=alt.condition(alt.datum[field] > (0.5 if normalized else float(cm.max()) * 0.5), alt.value("white"), alt.value("#142e3a")))
            st.altair_chart((heat + text).properties(height=350, title=f"{label} · test set"), width="stretch")
            st.caption("Rows = true class; columns = predicted class. Hover to see counts and row percentages. A row with no examples is shown as zero; recall for that class is unestimated.")
            for category in AQI_CATEGORIES:
                values = metrics["per_class_report"][category]
                per_class.append({"Model": label, "Class": category, "Precision": values["precision"],
                                  "Recall": values["recall"], "F1": values["f1-score"], "Support": int(values["support"])})
    st.subheader("Per-class performance")
    if per_class:
        data = pd.DataFrame(per_class)
        metric = st.selectbox("Per-class metric", ["F1", "Recall", "Precision"], key="per_class_metric")
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X("Class:N", sort=AQI_CATEGORIES, axis=alt.Axis(labelAngle=-15)),
            xOffset=alt.XOffset("Model:N"), y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0,1])),
            color=alt.Color("Model:N", scale=alt.Scale(range=CHART_COLORS)),
            tooltip=["Model", "Class", alt.Tooltip(f"{metric}:Q", format=".4f"), "Support"],
        ).properties(height=330)
        st.altair_chart(chart, width="stretch")
        with st.expander("Precision, recall, F1 and support for every class"):
            st.dataframe(data.style.format({"Precision":"{:.4f}", "Recall":"{:.4f}", "F1":"{:.4f}"}), hide_index=True, width="stretch")
    st.download_button("Download current comparison CSV", table.to_csv(index=False), file_name=f"airsense_{variant}_comparison.csv", mime="text/csv")


def render_prediction_result(result):
    category = result["category"]
    st.markdown(f'<div class="result" style="border-left-color:{AQI_COLORS[category]}"><div class="eyebrow">Estimated AQI category</div><h2 style="color:{AQI_COLORS[category]};margin:0">{escape(category)}</h2></div>', unsafe_allow_html=True)
    probabilities = result["probabilities"]
    if probabilities is not None:
        table = pd.DataFrame({"Category": AQI_CATEGORIES, "Probability": probabilities})
        chart = alt.Chart(table).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Category:N", sort=AQI_CATEGORIES),
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format=".0%")),
            color=alt.Color("Category:N", scale=alt.Scale(domain=AQI_CATEGORIES, range=[AQI_COLORS[c] for c in AQI_CATEGORIES]), legend=None),
            tooltip=["Category", alt.Tooltip("Probability:Q", format=".2%")],
        ).properties(height=240, title="Model class probabilities")
        st.altair_chart(chart, width="stretch")
        st.caption("These are uncalibrated model probabilities, not verified confidence levels or health-risk estimates.")
    if result.get("probability_note"):
        st.info(result["probability_note"])
    st.warning("This is a machine-learning estimate of the same-day AQI category, not an official government measurement. It is not a forecast or a substitute for monitoring-station readings.")


def importance_chart(frame, value, title, std=None):
    plot = frame.head(12).sort_values(value, ascending=False)
    chart = alt.Chart(plot).mark_bar(color="#16847d", cornerRadiusEnd=3).encode(
        y=alt.Y("feature:N", sort=alt.SortField(field=value, order="descending"), title=None),
        x=alt.X(f"{value}:Q", title="Drop in macro F1" if value == "importance_mean" else "Importance"),
        tooltip=["feature", alt.Tooltip(f"{value}:Q", format=".5f")],
    ).properties(height=340, title=title)
    if std and std in frame:
        plot = plot.assign(lower=plot[value] - plot[std], upper=plot[value] + plot[std])
        errors = alt.Chart(plot).mark_rule(color="#163342").encode(
            y=alt.Y("feature:N", sort=alt.SortField(field=value, order="descending")),
            x="lower:Q", x2="upper:Q")
        chart = chart + errors
    st.altair_chart(chart, width="stretch")
